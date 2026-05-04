"""
paper_summary.py — quick health-check and WR estimate for the running paper bot.

Reads live_bot_state.db and prints:
  - Bot uptime (oldest event)
  - Intents per symbol (decided / entered / skipped)
  - Closed-position win rate, EPnL, and per-reason breakdown
  - Currently-open positions
  - Last 5 events (for crash diagnostics)

Usage:  python paper_summary.py
"""

from __future__ import annotations

import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB = Path(__file__).parent / "live_bot_state.db"


def fmt_pct(x: float | None) -> str:
    return f"{x * 100:+.3f}%" if x is not None else "n/a"


def main() -> int:
    if not DB.exists():
        print(f"[no db yet] {DB} — bot hasn't started? Launch with start_paper_bot.ps1")
        return 0

    cx = sqlite3.connect(DB)
    cx.row_factory = sqlite3.Row

    # Uptime
    e_first = cx.execute("SELECT MIN(created_at) AS t FROM events").fetchone()["t"]
    e_last = cx.execute("SELECT MAX(created_at) AS t FROM events").fetchone()["t"]
    if e_first:
        first = datetime.fromisoformat(e_first)
        last = datetime.fromisoformat(e_last)
        now = datetime.now(timezone.utc)
        uptime = now - first
        last_seen = now - last
        print(f"\nBot uptime: {uptime}  |  last event: {last_seen} ago")
    else:
        print("\nNo events recorded yet.")

    # Intents per symbol
    print("\n" + "=" * 72)
    print("INTENTS by symbol (model decisions)")
    print("=" * 72)
    rows = cx.execute(
        """
        SELECT symbol,
               COUNT(*) AS n_decisions,
               SUM(CASE WHEN action=1 THEN 1 ELSE 0 END) AS n_passed,
               SUM(CASE WHEN outcome='filled' THEN 1 ELSE 0 END) AS n_filled,
               SUM(CASE WHEN outcome='unfilled' THEN 1 ELSE 0 END) AS n_unfilled,
               SUM(CASE WHEN outcome='skipped' THEN 1 ELSE 0 END) AS n_skipped,
               AVG(proba_cal) AS avg_proba
        FROM intents
        GROUP BY symbol
        ORDER BY symbol
        """
    ).fetchall()
    if not rows:
        print("  (no intents yet — bot hasn't seen a closed bar yet, or just started)")
    else:
        print(
            f"  {'symbol':<10} {'bars':>5} {'passed':>7} {'filled':>7} {'unfilled':>9} {'skipped':>8} {'avg_p':>7}"
        )
        for r in rows:
            print(
                f"  {r['symbol']:<10} {r['n_decisions']:>5} {r['n_passed'] or 0:>7} "
                f"{r['n_filled'] or 0:>7} {r['n_unfilled'] or 0:>9} {r['n_skipped'] or 0:>8} "
                f"{(r['avg_proba'] or 0):>7.4f}"
            )

    # Closed positions: WR, EPnL
    print("\n" + "=" * 72)
    print("CLOSED POSITIONS")
    print("=" * 72)
    closed = cx.execute(
        """
        SELECT symbol, close_reason, gross_pnl_pct, opened_at, closed_at, entry_price, close_price
        FROM positions
        WHERE closed_at IS NOT NULL
        ORDER BY closed_at DESC
        """
    ).fetchall()
    if not closed:
        print(
            "  (no closed trades yet — be patient; bot needs ~hours to days for first signal)"
        )
    else:
        wins = sum(1 for r in closed if (r["gross_pnl_pct"] or 0) > 0)
        n = len(closed)
        wr = wins / n if n else 0
        epnl = sum((r["gross_pnl_pct"] or 0) for r in closed) / n if n else 0
        reason_breakdown = Counter(r["close_reason"] for r in closed)
        print(
            f"  total trades: {n}   wins: {wins}   WR: {wr * 100:.2f}%   avg PnL/trade: {fmt_pct(epnl)}"
        )
        print(f"  close reasons: {dict(reason_breakdown)}")
        print(
            f"\n  {'symbol':<10} {'reason':<10} {'pnl':>10} {'opened':>20} {'closed':>20}"
        )
        for r in closed[:20]:
            print(
                f"  {r['symbol']:<10} {r['close_reason'] or '?':<10} "
                f"{fmt_pct(r['gross_pnl_pct']):>10} "
                f"{r['opened_at'][:19]:>20} {r['closed_at'][:19]:>20}"
            )
        if len(closed) > 20:
            print(f"  ... and {len(closed) - 20} more")

        # Per-symbol WR
        print(f"\n  Per-symbol WR:")
        per_sym = {}
        for r in closed:
            per_sym.setdefault(r["symbol"], []).append((r["gross_pnl_pct"] or 0) > 0)
        for sym, wins_list in per_sym.items():
            n = len(wins_list)
            wr = sum(wins_list) / n
            print(f"    {sym:<10} n={n:>3}  WR={wr * 100:.2f}%")

    # Open positions
    print("\n" + "=" * 72)
    print("OPEN POSITIONS")
    print("=" * 72)
    opened = cx.execute(
        "SELECT symbol, entry_price, tp_price, sl_price, opened_at, max_hold_until "
        "FROM positions WHERE closed_at IS NULL"
    ).fetchall()
    if not opened:
        print("  (none)")
    else:
        for r in opened:
            print(
                f"  {r['symbol']:<10}  entry={r['entry_price']}  "
                f"TP={r['tp_price']:.6f}  SL={r['sl_price']:.6f}  "
                f"opened={r['opened_at'][:19]}  max_hold={r['max_hold_until'][:19]}"
            )

    # Recent events (for crash diagnostics)
    print("\n" + "=" * 72)
    print("LAST 5 EVENTS")
    print("=" * 72)
    evts = cx.execute(
        "SELECT created_at, level, kind, symbol, payload FROM events ORDER BY id DESC LIMIT 5"
    ).fetchall()
    for e in evts:
        sym = f" [{e['symbol']}]" if e["symbol"] else ""
        pay = (e["payload"] or "")[:80]
        print(f"  {e['created_at']} {e['level']:<5} {e['kind']:<20}{sym}  {pay}")

    print("\nRun again any time:  python paper_summary.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
