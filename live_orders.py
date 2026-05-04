"""
Order placement and fill management for the live trading bot.

Wraps `binance.um_futures.UMFutures` to:
  - place maker-only LIMIT entries with timeInForce='GTX' (post-only)
  - poll for fill within a timeout, cancel if unfilled
  - place exchange-side conditional exits (TAKE_PROFIT_MARKET, STOP_MARKET)
  - fetch exchange filters (tickSize, stepSize, minNotional) once and cache

Maker fee (entry) + taker fee (exit) on Binance USDT-M Futures default tier
≈ 0.02% + 0.04% = 0.06% round-trip, vs 0.08% all-taker.  At ±2% barriers this
moves breakeven WR from 54% to 51.5%.

Paper / dry-run mode is signalled by `client=None`.  All calls become no-ops
returning synthetic order ids prefixed `paper-`.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass
from typing import Optional

# binance-futures-connector
try:
    from binance.um_futures import UMFutures
    from binance.error import ClientError
except ImportError:
    UMFutures = None  # type: ignore
    ClientError = Exception  # type: ignore


# ─────────────────────────── Filters cache ─────────────────────────────────


@dataclass
class SymbolFilters:
    symbol: str
    tick_size: float  # price increment
    step_size: float  # qty increment
    min_qty: float
    min_notional: float


def fetch_filters(client: "UMFutures", symbol: str) -> SymbolFilters:
    info = client.exchange_info()
    for s in info["symbols"]:
        if s["symbol"] != symbol:
            continue
        tick = step = min_qty = min_notional = None
        for f in s["filters"]:
            t = f["filterType"]
            if t == "PRICE_FILTER":
                tick = float(f["tickSize"])
            elif t == "LOT_SIZE":
                step = float(f["stepSize"])
                min_qty = float(f["minQty"])
            elif t == "MIN_NOTIONAL":
                min_notional = float(f.get("notional", f.get("minNotional", "0")))
        if tick is None or step is None:
            raise RuntimeError(f"Missing filters for {symbol}")
        return SymbolFilters(symbol, tick, step, min_qty or 0.0, min_notional or 5.0)
    raise RuntimeError(f"Symbol {symbol} not found in exchange_info")


def round_down(x: float, step: float) -> float:
    if step <= 0:
        return x
    n = math.floor(x / step)
    # Avoid float drift: format to step's decimals
    decs = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    return round(n * step, decs)


def round_up(x: float, step: float) -> float:
    if step <= 0:
        return x
    n = math.ceil(x / step)
    decs = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    return round(n * step, decs)


# ─────────────────────────── Maker entry ───────────────────────────────────


@dataclass
class FillResult:
    filled: bool
    order_id: Optional[str]
    client_order_id: Optional[str]
    avg_price: Optional[float]
    executed_qty: float
    note: str = ""


def place_maker_buy(
    client: Optional["UMFutures"],
    filters: SymbolFilters,
    best_bid: float,
    quantity: float,
    timeout_sec: float = 30.0,
    poll_sec: float = 2.0,
    paper: bool = True,
) -> FillResult:
    """Submit a LIMIT BUY at best_bid with timeInForce=GTX (post-only).

    Binance rejects GTX orders that would immediately match -- so we set the
    price to (best_bid) which sits at or below the inside; if the spread has
    flipped by the time the request lands, the exchange will reject and we
    retry one tick lower.

    Polls until filled or timeout; cancels unfilled remainder.
    Returns FillResult(filled=True/False, ...).
    """
    coid = f"livebot-{uuid.uuid4().hex[:16]}"
    qty = round_down(quantity, filters.step_size)
    if qty < filters.min_qty or qty * best_bid < filters.min_notional:
        return FillResult(False, None, coid, None, 0.0, note="below min qty/notional")

    price = round_down(best_bid, filters.tick_size)

    if paper or client is None:
        # In paper mode we assume the order fills instantly at the bid.
        return FillResult(
            filled=True,
            order_id=f"paper-{coid}",
            client_order_id=coid,
            avg_price=price,
            executed_qty=qty,
            note="paper fill",
        )

    # Real submission ----------------------------------------------------
    try:
        resp = client.new_order(
            symbol=filters.symbol,
            side="BUY",
            type="LIMIT",
            quantity=qty,
            price=price,
            timeInForce="GTX",  # post-only
            newClientOrderId=coid,
        )
    except ClientError as e:
        return FillResult(False, None, coid, None, 0.0, note=f"submit error: {e}")

    order_id = str(resp["orderId"])
    deadline = time.monotonic() + timeout_sec

    while time.monotonic() < deadline:
        time.sleep(poll_sec)
        try:
            o = client.query_order(symbol=filters.symbol, orderId=order_id)
        except ClientError as e:
            return FillResult(False, order_id, coid, None, 0.0, note=f"poll error: {e}")
        status = o.get("status")
        executed = float(o.get("executedQty", 0))
        if status == "FILLED":
            avg = float(o.get("avgPrice", price)) or price
            return FillResult(True, order_id, coid, avg, executed)
        if status in ("EXPIRED", "CANCELED", "REJECTED"):
            return FillResult(
                False, order_id, coid, None, executed, note=f"status={status}"
            )

    # Timeout → cancel remaining
    try:
        client.cancel_order(symbol=filters.symbol, orderId=order_id)
    except ClientError as e:
        # already filled / canceled race; check status one more time
        try:
            o = client.query_order(symbol=filters.symbol, orderId=order_id)
            if o.get("status") == "FILLED":
                avg = float(o.get("avgPrice", price)) or price
                return FillResult(
                    True, order_id, coid, avg, float(o.get("executedQty", qty))
                )
        except Exception:
            pass
        return FillResult(False, order_id, coid, None, 0.0, note=f"cancel error: {e}")
    return FillResult(False, order_id, coid, None, 0.0, note="timed out, canceled")


# ─────────────────────────── Conditional exits ─────────────────────────────


def place_tp_sl(
    client: Optional["UMFutures"],
    filters: SymbolFilters,
    side: str,  # the entry side: "LONG" only for now
    quantity: float,
    tp_price: float,
    sl_price: float,
    paper: bool = True,
) -> tuple[Optional[str], Optional[str]]:
    """Place exchange-side TAKE_PROFIT_MARKET and STOP_MARKET orders.

    Both are reduce-only and trigger off contract last/mark price.
    Returns (tp_order_id, sl_order_id).
    """
    if side != "LONG":
        raise ValueError("Only LONG entries supported in MVP")

    qty = round_down(quantity, filters.step_size)
    tp_round = round_up(
        tp_price, filters.tick_size
    )  # be slightly above TP for safer trigger
    sl_round = round_down(sl_price, filters.tick_size)

    if paper or client is None:
        return f"paper-tp-{uuid.uuid4().hex[:8]}", f"paper-sl-{uuid.uuid4().hex[:8]}"

    tp_id = sl_id = None
    try:
        tp = client.new_order(
            symbol=filters.symbol,
            side="SELL",
            type="TAKE_PROFIT_MARKET",
            quantity=qty,
            stopPrice=tp_round,
            workingType="MARK_PRICE",
            reduceOnly="true",
            newClientOrderId=f"tp-{uuid.uuid4().hex[:12]}",
        )
        tp_id = str(tp["orderId"])
    except ClientError as e:
        tp_id = None
        # not fatal but caller should know
        print(f"[WARN] TP placement failed: {e}")
    try:
        sl = client.new_order(
            symbol=filters.symbol,
            side="SELL",
            type="STOP_MARKET",
            quantity=qty,
            stopPrice=sl_round,
            workingType="MARK_PRICE",
            reduceOnly="true",
            newClientOrderId=f"sl-{uuid.uuid4().hex[:12]}",
        )
        sl_id = str(sl["orderId"])
    except ClientError as e:
        sl_id = None
        print(f"[WARN] SL placement failed: {e}")
    return tp_id, sl_id


def cancel_order_safe(
    client: Optional["UMFutures"],
    symbol: str,
    order_id: Optional[str],
    paper: bool = True,
) -> None:
    if order_id is None or paper or client is None:
        return
    try:
        client.cancel_order(symbol=symbol, orderId=order_id)
    except ClientError as e:
        # most common reason: already filled or already canceled
        msg = str(e)
        if "-2011" in msg or "Unknown order" in msg:
            return
        print(f"[WARN] cancel {order_id} failed: {e}")


def market_close_long(
    client: Optional["UMFutures"],
    filters: SymbolFilters,
    quantity: float,
    paper: bool = True,
) -> Optional[str]:
    qty = round_down(quantity, filters.step_size)
    if paper or client is None:
        return f"paper-close-{uuid.uuid4().hex[:8]}"
    try:
        resp = client.new_order(
            symbol=filters.symbol,
            side="SELL",
            type="MARKET",
            quantity=qty,
            reduceOnly="true",
            newClientOrderId=f"close-{uuid.uuid4().hex[:12]}",
        )
        return str(resp["orderId"])
    except ClientError as e:
        print(f"[ERROR] market close failed for {filters.symbol}: {e}")
        return None


# ─────────────────────────── Book ticker ───────────────────────────────────


def get_book_ticker(client: Optional["UMFutures"], symbol: str) -> tuple[float, float]:
    """Returns (best_bid, best_ask).  In paper mode and no client, returns (0,0)."""
    if client is None:
        return 0.0, 0.0
    bt = client.book_ticker(symbol=symbol)
    return float(bt["bidPrice"]), float(bt["askPrice"])


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("live_orders module OK. UMFutures available:", UMFutures is not None)
