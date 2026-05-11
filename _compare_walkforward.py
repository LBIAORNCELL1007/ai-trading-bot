"""
Compare walk-forward OOF (Experiment B) against the static-model OOF on
identical rows.  Reports global threshold sweep, per-symbol best, and
chronological stability for BOTH models on the same row set so we can
isolate the effect of retraining schedule.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = "global_alpha_dataset_1h_2pct.csv"
WF_OOF = "walkforward_oof_1y.csv"
STATIC_OOF = "tbm_xgboost_model_v2_oof.csv"
TP = 0.02
SL = 0.02


def epnl(wr: float) -> float:
    return wr * TP - (1.0 - wr) * SL


def report_block(df_label: str, df: pd.DataFrame, proba_col: str):
    print("\n" + "=" * 78)
    print(f"[{df_label}]  rows={len(df):,}  base_rate={df['tbm_label'].mean():.4f}")
    print("=" * 78)

    # 1. Global sweep, no funding filter
    print("\n  -- Global sweep (no filter) --")
    print(f"  {'thr':>5} {'trades':>7} {'WR':>6} {'EPnL':>8}")
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60]:
        sel = df[df[proba_col] >= thr]
        if len(sel) == 0:
            continue
        wr = sel["tbm_label"].mean()
        print(f"  {thr:>5.2f} {len(sel):>7d} {wr:>6.3f} {epnl(wr):>+8.4f}")

    # 2. Funding filter
    df_f = df[df["funding_rate"] > 0.0001]
    print(f"\n  -- Funding>0.0001 sweep (n={len(df_f):,}) --")
    print(f"  {'thr':>5} {'trades':>7} {'WR':>6} {'EPnL':>8}")
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60]:
        sel = df_f[df_f[proba_col] >= thr]
        if len(sel) == 0:
            continue
        wr = sel["tbm_label"].mean()
        print(f"  {thr:>5.2f} {len(sel):>7d} {wr:>6.3f} {epnl(wr):>+8.4f}")

    # 3. Per-symbol best (no filter, min 50 trades)
    print("\n  -- Per-symbol best thr (no filter, min 50 trades) --")
    print(f"  {'symbol':<10} {'thr':>5} {'trades':>7} {'WR':>6} {'EPnL':>8}")
    for sym, g in df.groupby("symbol"):
        best = None
        for thr in np.arange(0.40, 0.75, 0.01):
            sel = g[g[proba_col] >= thr]
            if len(sel) < 50:
                break
            wr = sel["tbm_label"].mean()
            ep = epnl(wr)
            cand = (ep, thr, len(sel), wr)
            if best is None or cand[0] > best[0]:
                best = cand
        if best is None:
            print(f"  {sym:<10} {'--':>5} {'--':>7} {'--':>6} {'--':>8}")
            continue
        ep, thr, n, wr = best
        print(f"  {sym:<10} {thr:>5.2f} {n:>7d} {wr:>6.3f} {ep:>+8.4f}")

    # 4. Chronological stability (3 thirds within this row set)
    print("\n  -- Chronological stability (thr=0.50 + funding filter, 3 thirds) --")
    df_f_sorted = df_f.sort_values("timestamp").reset_index(drop=True)
    n3 = len(df_f_sorted) // 3
    for i, (lo, hi) in enumerate([(0, n3), (n3, 2 * n3), (2 * n3, len(df_f_sorted))]):
        chunk = df_f_sorted.iloc[lo:hi]
        sel = chunk[chunk[proba_col] >= 0.50]
        if len(sel) == 0:
            print(f"    P{i + 1}: 0 trades")
            continue
        wr = sel["tbm_label"].mean()
        print(
            f"    P{i + 1}: rows={len(chunk):>6,}  trades={len(sel):>4}  WR={wr:.4f}  EPnL={epnl(wr):+.4f}"
        )


def main() -> int:
    print(f"Loading {DATA} ...")
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    print(f"Loading walk-forward OOF: {WF_OOF}")
    wf = pd.read_csv(WF_OOF)
    if len(wf) != len(df):
        print(f"[FATAL] WF OOF rows ({len(wf)}) != dataset rows ({len(df)})")
        return 1

    print(f"Loading static OOF: {STATIC_OOF}")
    static = pd.read_csv(STATIC_OOF)
    if len(static) != len(df):
        print(f"[FATAL] static OOF rows ({len(static)}) != dataset rows ({len(df)})")
        return 1

    df["proba_wf"] = wf["oof_proba"].values
    df["proba_static"] = static["oof_proba"].values

    # Compare on identical row set: rows where WF has a prediction
    covered = df["proba_wf"].notna()
    df_cov = df[covered].copy().reset_index(drop=True)
    print(f"\nCovered rows (last 12 months walk-forward window): {len(df_cov):,}")
    print(f"Range: {df_cov['timestamp'].min()} -> {df_cov['timestamp'].max()}")

    report_block(
        "STATIC model (full-3y purged-kfold), restricted to last-12mo rows",
        df_cov,
        "proba_static",
    )
    report_block(
        "WALK-FORWARD model (2y trailing, monthly retrain)", df_cov, "proba_wf"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
