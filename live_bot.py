"""
Live trading bot — MVP.

Phase A scope (per .planning roadmap, May 2026):
  - Binance USDT-M Futures (matches training venue)
  - Maker-only entries (timeInForce=GTX), exchange-side TP/SL conditional exits
  - REST-polling main loop (1h cadence; no WS yet)
  - Paper mode by default; --live required for real orders
  - Testnet by default; --mainnet required for prod URL
  - Single-process, persistent state in SQLite (live_bot_state.db)

Decision pipeline per closed bar:
    backfill ~30 days of 1h klines + 30 days of funding
    -> live_features.compute_features()
    -> XGBoost predict_proba()
    -> sigmoid calibrator (Platt)
    -> threshold check (raw or calibrated; configurable)
    -> if pass: place_maker_buy at best_bid, place TP/SL on fill

Default barriers: TP=+2%, SL=-2%, max-hold 48h.  These match the dataset
the model was trained on (global_alpha_dataset_1h_2pct.csv).

Usage:
    python live_bot.py --symbols BTCUSDT --paper             # default: testnet, paper
    python live_bot.py --symbols BTCUSDT --paper --once      # one tick then exit (smoke test)
    python live_bot.py --symbols BTCUSDT,ETHUSDT --live      # testnet, real orders
    python live_bot.py --symbols BTCUSDT --live --mainnet    # PROD: real money
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

# Pickle resolution for calibrator class
try:
    from train_tbm_model_v2 import IsotonicCalibratorWrapper  # noqa: F401
except ImportError:
    pass

import live_state as state
import live_features
import live_orders as lo

try:
    from binance.um_futures import UMFutures
    from binance.error import ClientError
except ImportError:
    UMFutures = None
    ClientError = Exception

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ─────────────────────────── Constants & paths ────────────────────────────

REPO = Path(__file__).parent
MODEL_PATH = REPO / "tbm_xgboost_model_v2.json"
CALIBRATOR_PATH = REPO / "tbm_xgboost_model_v2_calibrated.pkl"
THRESHOLD_PATH = REPO / "tbm_xgboost_model_v2_threshold.json"
PER_SYMBOL_THRESHOLDS_PATH = REPO / "tbm_xgboost_model_v2_per_symbol_thresholds.json"

TESTNET_REST = "https://testnet.binancefuture.com"
MAINNET_REST = "https://fapi.binance.com"

# Public REST endpoints (no auth) for klines + funding -- always use mainnet
# data even in testnet mode, because testnet historical data is sparse.
PUBLIC_DATA_BASE = "https://fapi.binance.com"


# ─────────────────────────── Config ────────────────────────────────────────


@dataclass
class BotConfig:
    symbols: list[str]
    paper: bool = True
    testnet: bool = True
    interval: str = "1h"
    backfill_days: int = 30  # how much history to load for features
    tp_pct: float = 0.02
    sl_pct: float = 0.02
    max_hold_hours: int = 48
    risk_per_trade_usdt: float = 50.0  # capital allocated per trade (paper default)
    max_open_positions: int = 5
    threshold_override: Optional[float] = None  # overrides loaded threshold
    use_raw_threshold: bool = False  # if True, compare raw_proba vs threshold
    funding_min: float = 0.0001  # min funding_rate (1bp/8h) to enter; gates regime edge
    fill_timeout_sec: float = 30.0
    poll_interval_sec: float = 60.0  # main loop cadence


def parse_args() -> BotConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="BTCUSDT", help="Comma-separated symbols")
    p.add_argument("--paper", action="store_true", help="Paper mode (no real orders)")
    p.add_argument("--live", action="store_true", help="Live mode (real orders)")
    p.add_argument(
        "--mainnet", action="store_true", help="Use mainnet (default: testnet)"
    )
    p.add_argument(
        "--once", action="store_true", help="Run a single iteration and exit"
    )
    p.add_argument(
        "--threshold", type=float, default=None, help="Override decision threshold"
    )
    p.add_argument(
        "--use-raw",
        action="store_true",
        help="Compare raw probability against threshold (vs calibrated)",
    )
    p.add_argument("--risk-usdt", type=float, default=50.0, help="USDT per trade")
    p.add_argument("--max-open", type=int, default=5)
    p.add_argument("--tp", type=float, default=0.02)
    p.add_argument("--sl", type=float, default=0.02)
    p.add_argument("--max-hold-hours", type=int, default=48)
    p.add_argument("--poll-sec", type=float, default=60.0)
    p.add_argument(
        "--funding-min",
        type=float,
        default=0.0001,
        help=(
            "Minimum funding_rate to allow a long entry (default: 0.0001 = 1bp/8h). "
            "Backtest-validated regime filter: at thr=0.50 calibrated, OOF WR is "
            "53.6%% (377 trades over 3y) when funding_rate > 0.0001, vs 46.5%% "
            "across all funding regimes. Set to a very negative number to disable."
        ),
    )
    args = p.parse_args()

    paper = True
    if args.live and not args.paper:
        paper = False
    if args.live and args.paper:
        print("[FATAL] Pass either --paper or --live, not both.")
        sys.exit(2)

    return BotConfig(
        symbols=[s.strip().upper() for s in args.symbols.split(",") if s.strip()],
        paper=paper,
        testnet=not args.mainnet,
        threshold_override=args.threshold,
        use_raw_threshold=args.use_raw,
        risk_per_trade_usdt=args.risk_usdt,
        max_open_positions=args.max_open,
        tp_pct=args.tp,
        sl_pct=args.sl,
        max_hold_hours=args.max_hold_hours,
        poll_interval_sec=args.poll_sec,
        funding_min=args.funding_min,
    ), args.once


# ─────────────────────────── Model loading ────────────────────────────────


@dataclass
class ModelBundle:
    model: xgb.XGBClassifier
    calibrator: object
    threshold: float
    feature_order: list[str]
    per_symbol: dict  # {sym: {threshold, use_funding_filter}, "_default": {...}}


def load_model_bundle(cfg: BotConfig) -> ModelBundle:
    print(f"Loading model from {MODEL_PATH} ...")
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    print(f"Loading calibrator from {CALIBRATOR_PATH} ...")
    with open(CALIBRATOR_PATH, "rb") as f:
        calibrator = pickle.load(f)

    print(f"Loading threshold from {THRESHOLD_PATH} ...")
    with open(THRESHOLD_PATH) as f:
        thr_data = json.load(f)
    threshold = float(thr_data["threshold"])
    if cfg.threshold_override is not None:
        print(
            f"  [override] using threshold={cfg.threshold_override} (was {threshold})"
        )
        threshold = cfg.threshold_override
    print(
        f"  effective global threshold = {threshold:.4f}  "
        f"({'raw' if cfg.use_raw_threshold else 'calibrated'} scale)"
    )

    # Per-symbol thresholds (optional but strongly preferred — see JSON _meta).
    per_symbol: dict = {}
    if PER_SYMBOL_THRESHOLDS_PATH.exists():
        print(f"Loading per-symbol thresholds from {PER_SYMBOL_THRESHOLDS_PATH} ...")
        with open(PER_SYMBOL_THRESHOLDS_PATH) as f:
            per_symbol = json.load(f)
        listed = list(per_symbol.get("symbols", {}).keys())
        default = per_symbol.get("_default", {})
        print(
            f"  per-symbol entries: {listed}  default thr={default.get('threshold', '?')} "
            f"(unlisted symbols are skipped if default thr>1)"
        )
    else:
        print(
            f"[WARN] {PER_SYMBOL_THRESHOLDS_PATH.name} missing; falling back to "
            "global threshold for every symbol."
        )

    return ModelBundle(
        model=model,
        calibrator=calibrator,
        threshold=threshold,
        feature_order=live_features.FEATURE_COLUMNS,
        per_symbol=per_symbol,
    )


def get_symbol_policy(
    bundle: ModelBundle, symbol: str, cfg: BotConfig
) -> tuple[float, bool]:
    """Returns (threshold, use_funding_filter) for `symbol`.

    Resolution order:
      1) cfg.threshold_override always wins (CLI debug).
      2) bundle.per_symbol['symbols'][symbol] if listed.
      3) bundle.per_symbol['_default'] if present.
      4) bundle.threshold (global) with funding filter on.
    """
    if cfg.threshold_override is not None:
        return cfg.threshold_override, (cfg.funding_min > -1e9)
    syms = bundle.per_symbol.get("symbols", {}) if bundle.per_symbol else {}
    if symbol in syms:
        e = syms[symbol]
        return float(e["threshold"]), bool(e.get("use_funding_filter", True))
    default = bundle.per_symbol.get("_default") if bundle.per_symbol else None
    if default is not None:
        return float(default["threshold"]), bool(
            default.get("use_funding_filter", True)
        )
    return bundle.threshold, True


def predict_one(bundle: ModelBundle, features_row: pd.DataFrame) -> tuple[float, float]:
    """Returns (raw_proba, cal_proba)."""
    X = features_row[bundle.feature_order]
    raw = float(bundle.model.predict_proba(X)[0][1])
    if bundle.calibrator is not None:
        cal = float(bundle.calibrator.predict_proba(X)[0][1])
    else:
        cal = raw
    return raw, cal


# ─────────────────────────── Data fetch ────────────────────────────────────


import requests

_session = requests.Session()


def fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch up to `days` days of klines from public mainnet endpoint.

    Always uses mainnet REST for data (testnet has sparse history).  No auth.
    """
    url = f"{PUBLIC_DATA_BASE}/fapi/v1/klines"
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1500,
        }
        r = _session.get(url, params=params, timeout=15)
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        rows.extend(chunk)
        last = chunk[-1][6]  # close time
        if last <= cursor:
            break
        cursor = last + 1
        if len(chunk) < 1500:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "qav",
            "trades",
            "tbbav",
            "tbqav",
            "ignore",
        ],
    )
    # Match training: dataset is indexed by bar OPEN time (matches build_global_dataset.py).
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df = df.set_index("open_time").sort_index()
    # Drop the still-forming bar at the tail (its OHLCV is partial).
    bar_minutes = {"1h": 60, "2h": 120, "4h": 240, "8h": 480}.get(interval, 60)
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(minutes=bar_minutes)
    df = df[df.index <= cutoff]
    return df[["open", "high", "low", "close", "volume"]]


def fetch_funding(symbol: str, days: int) -> pd.DataFrame:
    url = f"{PUBLIC_DATA_BASE}/fapi/v1/fundingRate"
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        r = _session.get(url, params=params, timeout=15)
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        rows.extend(chunk)
        last = chunk[-1]["fundingTime"]
        if last <= cursor:
            break
        cursor = last + 1
        if len(chunk) < 1000:
            break
    if not rows:
        return pd.DataFrame(columns=["funding_rate"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df.set_index("timestamp")[["funding_rate"]].sort_index()


# ─────────────────────────── Bar tracking ──────────────────────────────────


def latest_closed_bar_time(now: datetime, interval: str = "1h") -> datetime:
    """Returns the close time of the most recently *closed* 1h bar.
    For 1h bars, that's the most recent top-of-hour ≤ `now` minus an epsilon.
    """
    if interval != "1h":
        raise NotImplementedError("MVP only supports 1h bars")
    floored = now.replace(
        minute=0, second=0, microsecond=0, tzinfo=now.tzinfo or timezone.utc
    )
    # If we're within the first 30s of a new bar, the bar that JUST closed
    # may not be reflected in REST yet -- conservatively wait till :00:30.
    if (now - floored).total_seconds() < 30:
        floored -= timedelta(hours=1)
    return floored


# ─────────────────────────── Decision + execution ──────────────────────────


@dataclass
class SymbolCtx:
    symbol: str
    filters: lo.SymbolFilters
    last_evaluated_bar: Optional[datetime] = None


def evaluate_symbol(
    ctx: SymbolCtx,
    bundle: ModelBundle,
    cfg: BotConfig,
    client: Optional["UMFutures"],
) -> None:
    sym = ctx.symbol
    bar_close = latest_closed_bar_time(datetime.now(timezone.utc), cfg.interval)

    if ctx.last_evaluated_bar == bar_close:
        return  # already handled this bar
    print(f"\n[{sym}] evaluating bar close {bar_close.isoformat()}")

    # 1. Fetch & compute features
    klines = fetch_klines(sym, cfg.interval, cfg.backfill_days)
    if klines.empty:
        print(f"  [{sym}] no klines returned, skipping")
        return
    funding = fetch_funding(sym, cfg.backfill_days)
    feats = live_features.compute_features(klines, funding, cfg.interval)
    if feats.empty or feats.iloc[-1].isnull().any():
        print(f"  [{sym}] features have NaNs in latest row, skipping")
        return

    latest = feats.iloc[-1:]
    latest_bar_time = latest.index[-1]

    # Per-symbol decision policy: threshold + whether to apply funding filter.
    sym_thr, sym_use_funding = get_symbol_policy(bundle, sym, cfg)

    # 2. Predict
    raw, cal = predict_one(bundle, latest)
    decision_proba = raw if cfg.use_raw_threshold else cal
    pass_thr = decision_proba >= sym_thr

    # Funding-regime filter — backtested on 3-year OOF predictions.
    # Per-symbol JSON specifies whether a symbol benefits from the filter.
    # SOL/DOGE/XRP/AVAX edges are STRONGER without it (their per-symbol
    # OOF analysis selected use_funding_filter=false).  Default for any
    # unlisted symbol is to require it (and a threshold>1 so they skip).
    funding_now = float(latest.iloc[0]["funding_rate"])
    if sym_use_funding:
        pass_funding = funding_now > cfg.funding_min
    else:
        pass_funding = True  # symbol's edge does not require the filter

    print(
        f"  [{sym}] raw={raw:.4f}  cal={cal:.4f}  thr={sym_thr:.4f}  "
        f"pass_model={pass_thr}  funding={funding_now:+.5f}  "
        f"use_funding_filter={sym_use_funding}  pass_funding={pass_funding}"
    )

    # Record intent regardless of outcome
    intent_id = state.insert_intent(
        symbol=sym,
        bar_close_time=latest_bar_time.isoformat(),
        proba_raw=raw,
        proba_cal=cal,
        threshold=sym_thr,
        action=int(pass_thr and pass_funding),
        features={k: float(v) for k, v in latest.iloc[0].items()},
    )

    if not pass_thr:
        state.update_intent_outcome(intent_id, "skipped", note="below threshold")
        ctx.last_evaluated_bar = bar_close
        return

    if not pass_funding:
        state.update_intent_outcome(
            intent_id,
            "skipped",
            note=f"funding {funding_now:+.5f} <= min {cfg.funding_min:+.5f}",
        )
        ctx.last_evaluated_bar = bar_close
        return

    # 3. Position-limit checks
    if state.has_open_position(sym):
        print(f"  [{sym}] already have open position, skipping")
        state.update_intent_outcome(intent_id, "skipped", note="symbol already open")
        ctx.last_evaluated_bar = bar_close
        return
    if state.count_open_positions() >= cfg.max_open_positions:
        print(f"  [{sym}] max_open_positions reached, skipping")
        state.update_intent_outcome(intent_id, "skipped", note="max open reached")
        ctx.last_evaluated_bar = bar_close
        return

    # 4. Get inside quote
    # In paper mode, always pull bid/ask from mainnet (public endpoint, no auth)
    # so simulated fills reflect real-world spreads, not testnet's thin book.
    if cfg.paper:
        try:
            r = _session.get(
                f"{PUBLIC_DATA_BASE}/fapi/v1/ticker/bookTicker",
                params={"symbol": sym},
                timeout=10,
            )
            r.raise_for_status()
            bt = r.json()
            best_bid = float(bt["bidPrice"])
            best_ask = float(bt["askPrice"])
        except Exception as e:
            print(f"  [{sym}] mainnet book_ticker failed: {e}; using last close")
            last_close = float(klines["close"].iloc[-1])
            best_bid, best_ask = last_close, last_close
    elif client is not None:
        try:
            best_bid, best_ask = lo.get_book_ticker(client, sym)
        except Exception as e:
            print(f"  [{sym}] book_ticker failed: {e}")
            state.update_intent_outcome(intent_id, "error", note=f"book_ticker: {e}")
            ctx.last_evaluated_bar = bar_close
            return
    else:
        last_close = float(klines["close"].iloc[-1])
        best_bid, best_ask = last_close, last_close
    print(f"  [{sym}] bid={best_bid}  ask={best_ask}")

    # 5. Quantity sizing
    qty = cfg.risk_per_trade_usdt / max(best_ask, 1e-12)

    # 6. Place maker buy
    fill = lo.place_maker_buy(
        client=client,
        filters=ctx.filters,
        best_bid=best_bid,
        quantity=qty,
        timeout_sec=cfg.fill_timeout_sec,
        paper=cfg.paper,
    )
    if not fill.filled:
        print(f"  [{sym}] entry not filled: {fill.note}")
        state.update_intent_outcome(
            intent_id,
            "unfilled",
            client_order_id=fill.client_order_id,
            exchange_order_id=fill.order_id,
            note=fill.note,
        )
        ctx.last_evaluated_bar = bar_close
        return

    entry = fill.avg_price or best_bid
    tp_price = entry * (1.0 + cfg.tp_pct)
    sl_price = entry * (1.0 - cfg.sl_pct)
    max_hold = (
        datetime.now(timezone.utc) + timedelta(hours=cfg.max_hold_hours)
    ).isoformat()

    pos_id = state.insert_position(
        intent_id=intent_id,
        symbol=sym,
        side="LONG",
        entry_order_id=fill.order_id or "",
        entry_price=entry,
        quantity=fill.executed_qty,
        tp_price=tp_price,
        sl_price=sl_price,
        max_hold_until=max_hold,
    )

    # 7. Place exchange-side TP / SL
    tp_id, sl_id = lo.place_tp_sl(
        client=client,
        filters=ctx.filters,
        side="LONG",
        quantity=fill.executed_qty,
        tp_price=tp_price,
        sl_price=sl_price,
        paper=cfg.paper,
    )
    state.attach_exit_orders(pos_id, tp_id, sl_id)
    state.update_intent_outcome(
        intent_id,
        "filled",
        client_order_id=fill.client_order_id,
        exchange_order_id=fill.order_id,
    )
    state.log_event(
        "position_opened",
        symbol=sym,
        payload={
            "pos_id": pos_id,
            "entry": entry,
            "qty": fill.executed_qty,
            "tp": tp_price,
            "sl": sl_price,
            "max_hold": max_hold,
        },
    )
    print(
        f"  [{sym}] OPENED LONG  entry={entry}  qty={fill.executed_qty}  "
        f"TP={tp_price:.6f}  SL={sl_price:.6f}  pos_id={pos_id}"
    )

    ctx.last_evaluated_bar = bar_close


# ─────────────────────────── Position monitor ──────────────────────────────


def monitor_positions(
    bundle: ModelBundle,
    cfg: BotConfig,
    client: Optional["UMFutures"],
    filters_by_symbol: dict[str, lo.SymbolFilters],
) -> None:
    """Check open positions for TP/SL fills (real mode) or barrier hits (paper)."""
    open_pos = state.list_open_positions()
    if not open_pos:
        return
    now = datetime.now(timezone.utc)

    for p in open_pos:
        sym = p["symbol"]
        max_hold_dt = datetime.fromisoformat(p["max_hold_until"])
        # In paper mode we have no exchange; use latest kline close to simulate.
        if cfg.paper:
            try:
                kl = fetch_klines(sym, cfg.interval, days=2)
                if kl.empty:
                    continue
                # Simulate barriers against the bar high/low since open
                opened_at = datetime.fromisoformat(p["opened_at"])
                window = kl[kl.index >= opened_at.replace(tzinfo=None)]
                if window.empty:
                    continue
                hit_tp = (window["high"] >= p["tp_price"]).any()
                hit_sl = (window["low"] <= p["sl_price"]).any()
                # Pick the EARLIER of TP/SL by walking the bars
                close_reason = None
                close_price = None
                for ts, row in window.iterrows():
                    if row["low"] <= p["sl_price"]:
                        close_reason = "SL"
                        close_price = p["sl_price"]
                        break
                    if row["high"] >= p["tp_price"]:
                        close_reason = "TP"
                        close_price = p["tp_price"]
                        break
                if close_reason is None and now >= max_hold_dt:
                    close_reason = "MAX_HOLD"
                    close_price = float(window["close"].iloc[-1])
                if close_reason:
                    pnl = (close_price - p["entry_price"]) / p["entry_price"]
                    state.close_position(p["id"], close_price, close_reason, pnl)
                    state.log_event(
                        "position_closed",
                        symbol=sym,
                        payload={
                            "pos_id": p["id"],
                            "reason": close_reason,
                            "close_price": close_price,
                            "pnl_pct": pnl,
                        },
                    )
                    print(
                        f"  [{sym}] CLOSED ({close_reason}) close={close_price:.6f} "
                        f"pnl={pnl * 100:+.3f}%  pos_id={p['id']}"
                    )
            except Exception as e:
                print(f"  [{sym}] paper-monitor error: {e}")
            continue

        # Real mode: query TP/SL order status
        try:
            tp_status = (
                client.query_order(symbol=sym, orderId=int(p["tp_order_id"]))
                if p["tp_order_id"]
                else None
            )
            sl_status = (
                client.query_order(symbol=sym, orderId=int(p["sl_order_id"]))
                if p["sl_order_id"]
                else None
            )
        except Exception as e:
            print(f"  [{sym}] order query error: {e}")
            continue

        close_reason = close_price = sibling_to_cancel = None
        if tp_status and tp_status.get("status") == "FILLED":
            close_reason = "TP"
            close_price = float(tp_status.get("avgPrice") or p["tp_price"])
            sibling_to_cancel = p["sl_order_id"]
        elif sl_status and sl_status.get("status") == "FILLED":
            close_reason = "SL"
            close_price = float(sl_status.get("avgPrice") or p["sl_price"])
            sibling_to_cancel = p["tp_order_id"]
        elif now >= max_hold_dt:
            # Cancel both, market close.
            lo.cancel_order_safe(client, sym, p["tp_order_id"], paper=False)
            lo.cancel_order_safe(client, sym, p["sl_order_id"], paper=False)
            close_oid = lo.market_close_long(
                client, filters_by_symbol[sym], p["quantity"], paper=False
            )
            close_reason = "MAX_HOLD"
            # Best estimate of close price = last book ticker mid
            try:
                bid, ask = lo.get_book_ticker(client, sym)
                close_price = (bid + ask) / 2 if bid and ask else p["entry_price"]
            except Exception:
                close_price = p["entry_price"]

        if close_reason:
            if sibling_to_cancel:
                lo.cancel_order_safe(client, sym, sibling_to_cancel, paper=False)
            pnl = (close_price - p["entry_price"]) / p["entry_price"]
            state.close_position(p["id"], close_price, close_reason, pnl)
            state.log_event(
                "position_closed",
                symbol=sym,
                payload={
                    "pos_id": p["id"],
                    "reason": close_reason,
                    "close_price": close_price,
                    "pnl_pct": pnl,
                },
            )
            print(
                f"  [{sym}] CLOSED ({close_reason}) close={close_price:.6f} "
                f"pnl={pnl * 100:+.3f}%  pos_id={p['id']}"
            )


# ─────────────────────────── Main loop ─────────────────────────────────────


_stop = False


def _handle_signal(signum, frame):
    global _stop
    print(f"\n[signal {signum}] stopping after current iteration ...")
    _stop = True


def main() -> int:
    cfg, run_once = parse_args()
    print("─" * 70)
    print(
        f"live_bot.py  paper={cfg.paper}  testnet={cfg.testnet}  symbols={cfg.symbols}"
    )
    print("─" * 70)

    state.init_db()
    state.log_event(
        "bot_start",
        payload={
            "paper": cfg.paper,
            "testnet": cfg.testnet,
            "symbols": cfg.symbols,
            "tp_pct": cfg.tp_pct,
            "sl_pct": cfg.sl_pct,
            "max_hold_hours": cfg.max_hold_hours,
        },
    )

    bundle = load_model_bundle(cfg)

    # Init exchange client only in live mode (paper mode uses public endpoints).
    client: Optional["UMFutures"] = None
    if not cfg.paper:
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")
        base_url = TESTNET_REST if cfg.testnet else MAINNET_REST
        if not api_key or not api_secret:
            print(
                "[FATAL] --live requires BINANCE_API_KEY and BINANCE_API_SECRET env vars."
            )
            return 2
        try:
            client = UMFutures(key=api_key, secret=api_secret, base_url=base_url)
            client.ping()
            print(
                f"[OK] {('TESTNET' if cfg.testnet else 'MAINNET')} reachable: {base_url}"
            )
        except Exception as e:
            print(f"[FATAL] exchange client init failed: {e}")
            return 2
    else:
        print(
            "[paper mode] no exchange client; using public mainnet endpoints for data"
        )

    # Cache filters per symbol — use mainnet exchangeInfo in paper mode
    # (testnet symbol list and tick sizes can drift from mainnet).
    filters_by_symbol: dict[str, lo.SymbolFilters] = {}
    contexts: list[SymbolCtx] = []

    if cfg.paper:
        try:
            r = _session.get(f"{PUBLIC_DATA_BASE}/fapi/v1/exchangeInfo", timeout=15)
            r.raise_for_status()
            mainnet_info = r.json()
        except Exception as e:
            print(f"[FATAL] could not fetch mainnet exchangeInfo: {e}")
            return 2
        info_by_sym = {s["symbol"]: s for s in mainnet_info["symbols"]}
        for sym in cfg.symbols:
            s = info_by_sym.get(sym)
            if s is None:
                print(f"[{sym}] NOT FOUND on mainnet, skipping")
                continue
            tick = step = min_qty = min_notional = None
            for fl in s["filters"]:
                t = fl["filterType"]
                if t == "PRICE_FILTER":
                    tick = float(fl["tickSize"])
                elif t == "LOT_SIZE":
                    step = float(fl["stepSize"])
                    min_qty = float(fl["minQty"])
                elif t == "MIN_NOTIONAL":
                    min_notional = float(fl.get("notional", fl.get("minNotional", "0")))
            f = lo.SymbolFilters(
                sym,
                tick or 0.1,
                step or 0.001,
                min_qty or 0.001,
                min_notional or 5.0,
            )
            filters_by_symbol[sym] = f
            contexts.append(SymbolCtx(symbol=sym, filters=f))
            print(
                f"[{sym}] filters: tick={f.tick_size}  step={f.step_size}  "
                f"min_qty={f.min_qty}  min_notional={f.min_notional}"
            )
    else:
        for sym in cfg.symbols:
            if client is not None:
                try:
                    f = lo.fetch_filters(client, sym)
                except Exception as e:
                    print(f"[{sym}] fetch_filters failed: {e}; using defaults")
                    f = lo.SymbolFilters(sym, 0.1, 0.001, 0.001, 5.0)
            else:
                f = lo.SymbolFilters(sym, 0.1, 0.001, 0.001, 5.0)
            filters_by_symbol[sym] = f
            contexts.append(SymbolCtx(symbol=sym, filters=f))
            print(
                f"[{sym}] filters: tick={f.tick_size}  step={f.step_size}  "
                f"min_qty={f.min_qty}  min_notional={f.min_notional}"
            )

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    iteration = 0
    while not _stop:
        iteration += 1
        try:
            for ctx in contexts:
                evaluate_symbol(ctx, bundle, cfg, client)
            monitor_positions(bundle, cfg, client, filters_by_symbol)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[loop error] {type(e).__name__}: {e}")
            state.log_event("loop_error", level="ERROR", payload={"err": str(e)})
        if run_once:
            break
        # Sleep, but check stop flag frequently
        slept = 0.0
        while slept < cfg.poll_interval_sec and not _stop:
            time.sleep(min(2.0, cfg.poll_interval_sec - slept))
            slept += 2.0

    state.log_event("bot_stop", payload={"iterations": iteration})
    print(f"\nbot_stop  total_iterations={iteration}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
