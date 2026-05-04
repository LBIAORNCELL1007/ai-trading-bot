"""
Compare baseline vs enriched OOF predictions: WR/EPnL/trade-count sweeps.
Both use raw probabilities for fair monotonic comparison.
"""

import pandas as pd
import numpy as np
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TP = 0.02
SL = 0.02
FEE = 0.0008  # round-trip 0.08%


def sweep(oof_path, name):
    df = pd.read_csv(oof_path)
    p = df["oof_proba_raw"].values
    y = df["y"].values
    print(f"\n=== {name} ===")
    print(
        f"  rows={len(df)}  unique probas={len(np.unique(p))}  proba range=[{p.min():.4f}, {p.max():.4f}]"
    )
    # per-trade gross PnL: y=1 -> +TP, y=0 -> -SL.  Net = gross - fee.
    print(
        f"  {'thr':>6} {'trades':>7} {'wins':>6} {'WR':>7} {'gross/tr':>9} {'net/tr':>9} {'sum_net%':>10}"
    )
    for thr in [0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65]:
        mask = p >= thr
        n = int(mask.sum())
        if n == 0:
            print(
                f"  {thr:>6.3f} {0:>7d} {'-':>6} {'-':>7} {'-':>9} {'-':>9} {'-':>10}"
            )
            continue
        wins = int(y[mask].sum())
        wr = wins / n
        gross = wins * TP + (n - wins) * (-SL)  # total
        net = gross - n * FEE
        gross_per = gross / n * 100
        net_per = net / n * 100
        print(
            f"  {thr:>6.3f} {n:>7d} {wins:>6d} {wr:>7.3f} {gross_per:>8.3f}% {net_per:>8.3f}% {net * 100:>9.2f}%"
        )


sweep(r"D:\ai-trading-bot\joint_baseline_oof.csv", "BASELINE (kline+funding)")
sweep(r"D:\ai-trading-bot\joint_enriched_oof.csv", "ENRICHED (+ book features)")

# matched trade-count comparison: at the threshold yielding ~N trades for each, compare WR
print(
    "\n=== matched-N comparison (raw thresholds chosen to yield similar trade counts) ==="
)
for target_n in [50, 100, 200]:
    for path, name in [
        (r"D:\ai-trading-bot\joint_baseline_oof.csv", "baseline"),
        (r"D:\ai-trading-bot\joint_enriched_oof.csv", "enriched"),
    ]:
        df = pd.read_csv(path)
        p = df["oof_proba_raw"].values
        y = df["y"].values
        sorted_p = np.sort(p)[::-1]
        if target_n > len(sorted_p):
            continue
        thr = sorted_p[target_n - 1]
        mask = p >= thr
        n = int(mask.sum())
        wins = int(y[mask].sum())
        wr = wins / n
        net = (wins * TP + (n - wins) * (-SL) - n * FEE) / n * 100
        print(
            f"  target_n={target_n:>4}  {name:>9}  thr={thr:.4f}  actual_n={n:>4}  wins={wins:>3}  WR={wr:.3f}  net/tr={net:+.3f}%"
        )
