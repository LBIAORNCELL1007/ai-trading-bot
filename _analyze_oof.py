"""
Analyze OOF predictions from train_tbm_model_v2.py for the 23-feature model.

Compares against the 15-feature + funding-filter baseline:
  thr=0.50, funding_rate > 0.0001  -->  53.6% WR / 377 trades / +0.103% EPnL.

Reports:
  1. Global threshold sweep (no funding filter)
  2. Global threshold sweep (with funding_rate > 0.0001)
  3. Per-symbol best threshold (where WR >= 0.52 and trades >= 100)
  4. Per-symbol best threshold WITH funding filter
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = "global_alpha_dataset_1h_2pct.csv"
OOF = "tbm_xgboost_model_v2_oof.csv"
TP = 0.02
SL = 0.02


# Realistic edge (maker-only entry, taker-out via TP/SL).  Same convention
# used during the prior 15-feature analysis so comparisons are apples-apples.
def epnl(wr: float) -> float:
    return wr * TP - (1.0 - wr) * SL


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data", default=DATA, help="Dataset CSV (must align row-for-row with --oof)."
    )
    ap.add_argument(
        "--oof", default=OOF, help="OOF predictions CSV from train_tbm_model_v2.py."
    )
    args = ap.parse_args()
    data_path = args.data
    oof_path = args.oof
    print(f"Loading {data_path} ...")
    df = pd.read_csv(data_path)
    print(f"Loading {oof_path} ...")
    oof = pd.read_csv(oof_path)
    if len(df) != len(oof):
        print(f"[FATAL] dataset len {len(df)} != oof len {len(oof)}")
        return 1
    df["proba"] = oof["oof_proba"].values
    df["proba_raw"] = oof["oof_proba_raw"].values

    print(f"  rows: {len(df):,}  symbols: {df['symbol'].nunique()}")
    print(f"  base rate (P(y=1)): {df['tbm_label'].mean():.4f}")

    print("\n" + "=" * 78)
    print("[1] GLOBAL THRESHOLD SWEEP — no funding filter")
    print("=" * 78)
    print(
        f"{'thr':>6}  {'trades':>7}  {'wins':>6}  {'WR':>6}  {'EPnL/trade':>10}  {'tot_PnL':>10}"
    )
    for thr in [0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.63, 0.65, 0.70]:
        sel = df[df["proba"] >= thr]
        n = len(sel)
        if n == 0:
            continue
        wr = sel["tbm_label"].mean()
        ep = epnl(wr)
        print(
            f"{thr:>6.2f}  {n:>7d}  {sel['tbm_label'].sum():>6d}  {wr:>6.3f}  {ep:>+10.4f}  {ep * n:>+10.2f}"
        )

    print("\n" + "=" * 78)
    print("[2] GLOBAL THRESHOLD SWEEP — funding_rate > 0.0001 (regime filter)")
    print("=" * 78)
    df_f = df[df["funding_rate"] > 0.0001]
    print(
        f"  rows after filter: {len(df_f):,}  base rate: {df_f['tbm_label'].mean():.4f}"
    )
    print(
        f"{'thr':>6}  {'trades':>7}  {'wins':>6}  {'WR':>6}  {'EPnL/trade':>10}  {'tot_PnL':>10}"
    )
    for thr in [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.63]:
        sel = df_f[df_f["proba"] >= thr]
        n = len(sel)
        if n == 0:
            continue
        wr = sel["tbm_label"].mean()
        ep = epnl(wr)
        print(
            f"{thr:>6.2f}  {n:>7d}  {sel['tbm_label'].sum():>6d}  {wr:>6.3f}  {ep:>+10.4f}  {ep * n:>+10.2f}"
        )

    print("\n" + "=" * 78)
    print("[3] PER-SYMBOL — best thr (no funding filter, min 100 trades)")
    print("=" * 78)
    print(f"{'symbol':<10} {'best_thr':>8} {'trades':>7} {'WR':>6} {'EPnL':>9}")
    per_sym_thr: dict = {}
    for sym, g in df.groupby("symbol"):
        best = None
        for thr in np.arange(0.40, 0.75, 0.01):
            sel = g[g["proba"] >= thr]
            n = len(sel)
            if n < 100:
                break
            wr = sel["tbm_label"].mean()
            ep = epnl(wr)
            cand = (ep, thr, n, wr)
            if best is None or cand[0] > best[0]:
                best = cand
        if best is None:
            print(f"{sym:<10} {'--':>8} {'--':>7} {'--':>6} {'--':>8}")
            continue
        ep, thr, n, wr = best
        per_sym_thr[sym] = float(round(thr, 2))
        print(f"{sym:<10} {thr:>8.2f} {n:>7d} {wr:>6.3f} {ep:>+8.4f}")

    print("\n" + "=" * 78)
    print("[4] PER-SYMBOL — best thr WITH funding_rate > 0.0001")
    print("=" * 78)
    print(f"{'symbol':<10} {'best_thr':>8} {'trades':>7} {'WR':>6} {'EPnL':>9}")
    for sym, g in df_f.groupby("symbol"):
        best = None
        for thr in np.arange(0.40, 0.75, 0.01):
            sel = g[g["proba"] >= thr]
            n = len(sel)
            if n < 50:
                break
            wr = sel["tbm_label"].mean()
            ep = epnl(wr)
            cand = (ep, thr, n, wr)
            if best is None or cand[0] > best[0]:
                best = cand
        if best is None:
            print(f"{sym:<10} {'--':>8} {'--':>7} {'--':>6} {'--':>8}")
            continue
        ep, thr, n, wr = best
        print(f"{sym:<10} {thr:>8.2f} {n:>7d} {wr:>6.3f} {ep:>+8.4f}")

    # Stability: split [3] into 3 chronological periods at the global level.
    print("\n" + "=" * 78)
    print("[5] CHRONOLOGICAL STABILITY — global thr=0.50 + funding filter, 3 periods")
    print("=" * 78)
    df_f_sorted = df_f.sort_values("timestamp").reset_index(drop=True)
    n3 = len(df_f_sorted) // 3
    for i, (lo, hi) in enumerate([(0, n3), (n3, 2 * n3), (2 * n3, len(df_f_sorted))]):
        chunk = df_f_sorted.iloc[lo:hi]
        sel = chunk[chunk["proba"] >= 0.50]
        if len(sel) == 0:
            print(f"  P{i + 1}: no trades")
            continue
        wr = sel["tbm_label"].mean()
        print(
            f"  P{i + 1}  rows={len(chunk):,}  trades={len(sel)}  WR={wr:.4f}  EPnL={epnl(wr):+.4f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
