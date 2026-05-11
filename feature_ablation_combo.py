"""
Combo ablation: drop multiple features simultaneously to see if effects
compound.  Based on single-drop ablation, best individual hurters were:
  bar_range_pct       (+3.6pp edge sum, +2.7pp global)
  realized_vol_24h    (+2.2pp edge sum, mixed by-symbol)
  funding_zscore_7d   (+2.0pp edge sum, hurts global)

Test 6 combos: just bar_range, all-3 hurters, and pairwise.
Reuses train_oof + metrics from feature_ablation.
"""

from __future__ import annotations

import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402

from feature_ablation import (  # noqa: E402
    DATA,
    DROP_COLS,
    TARGET_COL,
    FEATURES,
    train_oof,
    metrics,
)


COMBOS = [
    ("baseline", []),
    ("drop_BR", ["bar_range_pct"]),
    ("drop_BR+RV", ["bar_range_pct", "realized_vol_24h"]),
    ("drop_BR+FZ7", ["bar_range_pct", "funding_zscore_7d"]),
    ("drop_BR+RV+FZ7", ["bar_range_pct", "realized_vol_24h", "funding_zscore_7d"]),
    ("drop_RV+FZ7", ["realized_vol_24h", "funding_zscore_7d"]),
]


def main():
    print(f"Loading {DATA}...")
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    bht = df["barrier_hit_time"].astype(int).values
    drop_cols_present = [c for c in DROP_COLS if c in df.columns]
    base_X = df.drop(columns=drop_cols_present + [TARGET_COL])
    y = df[TARGET_COL].astype(int)

    rows_summary = []
    for label, dropfeats in COMBOS:
        print("\n" + "=" * 78)
        print(f"COMBO: {label}  drops={dropfeats}")
        print("=" * 78)
        X = base_X.drop(columns=dropfeats) if dropfeats else base_X
        t0 = time.time()
        raw, cal, iters = train_oof(X, y, bht)
        m = metrics(df, cal)
        g = m["global_thr050_filter"]
        sol = m["per_symbol"].get("SOLUSDT") or {}
        dog = m["per_symbol"].get("DOGEUSDT") or {}
        xrp = m["per_symbol"].get("XRPUSDT") or {}
        avx = m["per_symbol"].get("AVAXUSDT") or {}
        edge_sum = sum(d.get("wr", 0) for d in (sol, dog, xrp, avx))
        print(
            f"  global thr=0.50+filter:  n={g['n']}  WR={g['wr']:.4f}  EPnL={g['epnl']:+.5f}"
        )
        print(
            f"  SOL={sol.get('wr', 0):.4f}({sol.get('n', '-')}) "
            f"DOGE={dog.get('wr', 0):.4f}({dog.get('n', '-')}) "
            f"XRP={xrp.get('wr', 0):.4f}({xrp.get('n', '-')}) "
            f"AVAX={avx.get('wr', 0):.4f}({avx.get('n', '-')})"
        )
        print(
            f"  edge_sum_WR={edge_sum:.4f}  iters={iters}  brier={m['brier']:.4f}  time={time.time() - t0:.0f}s"
        )
        rows_summary.append((label, g, sol, dog, xrp, avx, edge_sum, m["brier"]))

    print("\n" + "=" * 110)
    print("COMBO SUMMARY")
    print("=" * 110)
    print(
        f"{'config':<22} {'gN':>4} {'gWR':>7} {'gEPnL':>9} "
        f"{'SOL':>14} {'DOGE':>14} {'XRP':>14} {'AVAX':>14} {'sumWR':>7} {'brier':>7}"
    )
    for label, g, sol, dog, xrp, avx, edge, brier in rows_summary:

        def fmt(d):
            if not d:
                return "--"
            return f"{d.get('wr', 0):.4f}/{d.get('n', '-'):>4}"

        print(
            f"{label:<22} {g['n']:>4} {g['wr']:>7.4f} {g['epnl']:>+9.5f} "
            f"{fmt(sol):>14} {fmt(dog):>14} {fmt(xrp):>14} {fmt(avx):>14} {edge:>7.4f} {brier:>7.4f}"
        )


if __name__ == "__main__":
    main()
