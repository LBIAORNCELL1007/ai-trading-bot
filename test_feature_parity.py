"""
Feature-parity test: live_features.compute_features() must produce values
that match the rows in global_alpha_dataset_1h_2pct.csv for the same
(symbol, timestamp).  Any drift here means live predictions disagree with
backtested predictions and the model's edge becomes unverifiable.

Tests all 14 features (was 15: bar_range_pct dropped after leave-one-out
ablation showed it actively hurt predictions; Tier 2 features were tried +
reverted as confirmed harmful in an earlier session).

Run:  python test_feature_parity.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import live_features
import live_bot

DATASET = r"D:\ai-trading-bot\global_alpha_dataset_1h_2pct.csv"
SYMBOL = "BTCUSDT"
TOL = 1e-6


def main() -> int:
    print(f"Loading {DATASET} ...")
    df = pd.read_csv(DATASET)
    df = df[df["symbol"] == SYMBOL].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    print(f"  {SYMBOL}: {len(df)} rows, range {df.index.min()} .. {df.index.max()}")

    df = df[df["tbm_label"].notna()]
    target_ts = df.index[-200]
    print(f"  initial target timestamp: {target_ts}")

    print("Fetching klines / funding from mainnet REST ...")
    klines = live_bot.fetch_klines(SYMBOL, "1h", days=60)
    funding = live_bot.fetch_funding(SYMBOL, days=60)
    print(f"  klines: {len(klines)} rows  funding: {len(funding)} rows")

    if klines.empty:
        print("[FATAL] no klines returned")
        return 1

    if target_ts not in klines.index:
        print(
            f"[INFO] target_ts not in 60d window ({klines.index.min()}..{klines.index.max()}). "
            "Picking a recent row from the dataset that's covered."
        )
        candidates = df.index.intersection(klines.index)
        if candidates.empty:
            print("[FATAL] no overlap between dataset and live klines window")
            return 1
        target_ts = candidates[-50]
        print(f"  new target timestamp: {target_ts}")

    expected = df.loc[target_ts]
    feats = live_features.compute_features(klines, funding, "1h")

    if target_ts not in feats.index:
        print(f"[FATAL] target_ts {target_ts} not in computed features index")
        return 1
    actual = feats.loc[target_ts]

    print("\n" + "─" * 72)
    print(f"{'feature':<24} {'expected':>14} {'actual':>14} {'diff':>14} STATUS")
    print("─" * 72)

    fail = False
    for col in live_features.FEATURE_COLUMNS:
        exp = float(expected[col]) if col in expected else float("nan")
        act = float(actual[col])
        diff = abs(exp - act) if not (np.isnan(exp) and np.isnan(act)) else 0.0
        ok = (np.isnan(exp) and np.isnan(act)) or diff < TOL
        status = "OK" if ok else "FAIL"
        if not ok:
            fail = True
        print(f"{col:<24} {exp:>14.6f} {act:>14.6f} {diff:>14.2e}  {status}")
    print("─" * 72)
    if fail:
        print("RESULT: PARITY FAILED")
        return 1
    print("RESULT: PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
