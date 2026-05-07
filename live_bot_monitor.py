"""
Real-time monitor of live_bot_state.db.

Usage:  python live_bot_monitor.py             # one-shot summary
        python live_bot_monitor.py --watch     # refresh every 30s
        python live_bot_monitor.py --intents   # last 30 intents only
        python live_bot_monitor.py --positions # all positions
        python live_bot_monitor.py --events 50 # last 50 events
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB = Path(__file__).parent / "live_bot_state.db"


def conn():
    cx = sqlite3.connect(DB)
    cx.row_factory = sqlite3.Row
    return cx


def summary() -> None:
    cx = conn()
    print(f"\n=== live_bot_state.db @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    # Intents
    n_intents = cx.execute("SELECT COUNT(*) FROM intents").fetchone()[0]
    n_pass = cx.execute("SELECT COUNT(*) FROM intents WHERE action=1").fetchone()[0]
    print(
        f"\nIntents: total={n_intents}  passed-threshold={n_pass}  "
        f"trade-rate={n_pass / max(n_intents, 1) * 100:.2f}%"
    )

    rows = cx.execute(
        "SELECT outcome, COUNT(*) AS n FROM intents WHERE action=1 GROUP BY outcome"
    ).fetchall()
    if rows:
        print("  Outcomes (action=1):")
        for r in rows:
            print(f"    {r['outcome']:>10}: {r['n']}")

    # Positions
    n_open = cx.execute(
        "SELECT COUNT(*) FROM positions WHERE closed_at IS NULL"
    ).fetchone()[0]
    n_closed = cx.execute(
        "SELECT COUNT(*) FROM positions WHERE closed_at IS NOT NULL"
    ).fetchone()[0]
    print(f"\nPositions: open={n_open}  closed={n_closed}")

    if n_closed > 0:
        # Win rate, mean PnL
        agg = cx.execute(
            "SELECT close_reason, COUNT(*) AS n, AVG(gross_pnl_pct) AS avg_pnl "
            "FROM positions WHERE closed_at IS NOT NULL GROUP BY close_reason"
        ).fetchall()
        print("  Close reasons:")
        for r in agg:
            print(
                f"    {r['close_reason']:>10}: n={r['n']:>3}  avg_pnl={r['avg_pnl'] * 100:+.3f}%"
            )

        wr = cx.execute(
            "SELECT AVG(CASE WHEN gross_pnl_pct > 0 THEN 1.0 ELSE 0.0 END) AS wr "
            "FROM positions WHERE closed_at IS NOT NULL"
        ).fetchone()
        avg = cx.execute(
            "SELECT AVG(gross_pnl_pct) AS avg, "
            "SUM(gross_pnl_pct) AS total FROM positions WHERE closed_at IS NOT NULL"
        ).fetchone()
        print(
            f"  WR={wr['wr'] * 100:.1f}%  avg_pnl={avg['avg'] * 100:+.3f}%  "
            f"cumulative_pnl={avg['total'] * 100:+.2f}%"
        )

    # Open positions detail
    if n_open > 0:
        print("\n  Open positions:")
        for r in cx.execute(
            "SELECT id, symbol, opened_at, entry_price, tp_price, sl_price, max_hold_until "
            "FROM positions WHERE closed_at IS NULL"
        ):
            print(
                f"    #{r['id']} {r['symbol']:>10}  opened={r['opened_at'][:16]}  "
                f"entry={r['entry_price']:>10.4f}  TP={r['tp_price']:>10.4f}  "
                f"SL={r['sl_price']:>10.4f}  hold-until={r['max_hold_until'][:16]}"
            )


def show_intents(limit: int = 30) -> None:
    cx = conn()
    rows = cx.execute(
        "SELECT created_at, symbol, bar_close_time, proba_raw, proba_cal, "
        "threshold, action, outcome, note FROM intents ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    print(f"\nLast {len(rows)} intents (newest first):")
    print(
        f"  {'time':<19} {'sym':>9} {'bar':<19} {'raw':>6} {'cal':>6} "
        f"{'thr':>6} {'pass':>4} {'outcome':<10} note"
    )
    for r in rows:
        print(
            f"  {r['created_at'][:19]:<19} {r['symbol']:>9} "
            f"{r['bar_close_time'][:19]:<19} {r['proba_raw']:.4f} "
            f"{(r['proba_cal'] or 0):.4f} {r['threshold']:.4f} "
            f"{r['action']:>4} {(r['outcome'] or '-'):<10} {r['note'] or ''}"
        )


def show_positions() -> None:
    cx = conn()
    rows = cx.execute("SELECT * FROM positions ORDER BY id").fetchall()
    print(f"\nAll positions ({len(rows)}):")
    for r in rows:
        status = "OPEN" if r["closed_at"] is None else r["close_reason"]
        pnl = (
            f"{r['gross_pnl_pct'] * 100:+.3f}%"
            if r["gross_pnl_pct"] is not None
            else "-"
        )
        print(
            f"  #{r['id']:>3} {r['symbol']:>9}  {status:<10}  "
            f"opened={r['opened_at'][:16]}  entry={r['entry_price']:.4f}  pnl={pnl}"
        )


def show_events(limit: int = 50) -> None:
    cx = conn()
    rows = cx.execute(
        "SELECT created_at, level, kind, symbol, payload FROM events "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    print(f"\nLast {len(rows)} events:")
    for r in rows:
        sym = r["symbol"] or ""
        pl = (r["payload"] or "")[:120]
        print(
            f"  {r['created_at'][:19]} [{r['level']:<5}] {r['kind']:<22} {sym:<10} {pl}"
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--watch", action="store_true", help="Refresh every 30s")
    p.add_argument("--intents", action="store_true")
    p.add_argument("--positions", action="store_true")
    p.add_argument("--events", type=int, default=0)
    args = p.parse_args()

    if not DB.exists():
        print(f"[FATAL] {DB} does not exist. Run live_bot.py first.")
        return 1

    while True:
        if args.intents:
            show_intents()
        elif args.positions:
            show_positions()
        elif args.events:
            show_events(args.events)
        else:
            summary()
        if not args.watch:
            return 0
        time.sleep(30)
        print("\n" + "=" * 70)


if __name__ == "__main__":
    sys.exit(main())
