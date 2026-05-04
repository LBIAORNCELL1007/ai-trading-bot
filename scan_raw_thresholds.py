"""
Raw-proba threshold scan.

Diagnostic: the isotonic calibrator has been collapsing 46k unique raw probas
to ~21 plateaus, destroying the model's discrimination.  This script bypasses
calibration and simulates the same chronological no-overlap walk-forward logic
directly on the *raw* (pre-calibration) OOF probabilities, sweeping a grid of
thresholds to find the operating point with real edge.

Inputs:  global_alpha_dataset_1h_2pct.csv, tbm_xgboost_model_v2_oof.csv
Outputs: pipeline_logs/raw_threshold_scan.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "global_alpha_dataset_1h_2pct.csv"
OOF_FILE = ROOT / "tbm_xgboost_model_v2_oof.csv"
THR_FILE = ROOT / "tbm_xgboost_model_v2_threshold.json"
OUT_FILE = ROOT / "pipeline_logs" / "raw_threshold_scan.csv"
OUT_FILE.parent.mkdir(exist_ok=True)


def load_merged():
    cfg = json.loads(THR_FILE.read_text())
    # NOTE: trainer hardcodes tp/sl=0.01 in its JSON regardless of dataset.
    # Override from the dataset's actual barrier hit semantics: read from
    # the top-of-file env var or pass via CLI.  For now, infer from the
    # dataset filename ("_2pct" => 0.02) or trust the JSON.
    tp = float(cfg.get("tp_pct", 0.02))
    sl = abs(float(cfg.get("sl_pct", 0.02)))
    if "_2pct" in DATA_FILE.name and tp < 0.015:
        # Trainer wrote ±1% defaults but dataset is ±2%.  Override.
        tp = 0.02
        sl = 0.02
        print(
            f"[override] dataset filename indicates ±2% barriers; using tp=sl={tp:.4f}"
        )
    fee = float(cfg.get("fee_pct", 0.0008))

    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    oof = pd.read_csv(OOF_FILE)
    if len(oof) != len(df):
        raise RuntimeError(f"row mismatch: data={len(df)} oof={len(oof)}")
    df["raw_proba"] = oof["oof_proba_raw"].values
    df = df.dropna(subset=["raw_proba", "tbm_label", "barrier_hit_time"]).copy()
    return df, tp, sl, fee


def simulate(df: pd.DataFrame, threshold: float, tp: float, sl: float, fee: float):
    """No-overlap chronological walk-forward on raw probas."""
    blocked_until: dict[str, pd.Timestamp] = {}
    nets: list[float] = []
    n_wins = 0
    n_skipped_overlap = 0
    first_ts = None
    last_exit = None
    rt_fee = 2.0 * fee

    p = df["raw_proba"].values
    ts_arr = df["timestamp"].values
    sym_arr = df["symbol"].values
    label_arr = df["tbm_label"].values
    bht_arr = df["barrier_hit_time"].values

    for i in range(len(df)):
        if p[i] < threshold:
            continue
        sym = sym_arr[i]
        ts = ts_arr[i]
        block = blocked_until.get(sym)
        if block is not None and ts < block:
            n_skipped_overlap += 1
            continue
        label = int(label_arr[i])
        bht = int(bht_arr[i])
        gross = tp if label == 1 else -sl
        net = gross - rt_fee
        nets.append(net)
        if label == 1:
            n_wins += 1
        exit_ts = ts + np.timedelta64(bht, "h")
        blocked_until[sym] = exit_ts
        if first_ts is None:
            first_ts = ts
        last_exit = exit_ts

    n = len(nets)
    if n == 0:
        return None
    arr = np.array(nets)
    mean_net = float(arr.mean())
    std_net = float(arr.std(ddof=1)) if n > 1 else 0.0
    win_rate = n_wins / n
    sharpe_pt = mean_net / std_net if std_net > 0 else 0.0
    span_hours = (last_exit - first_ts).astype("timedelta64[s]").astype(float) / 3600.0
    tpy = (n / span_hours) * (24 * 365) if span_hours > 0 else 0.0
    ann_sharpe = sharpe_pt * np.sqrt(tpy)
    equity = float(np.prod(1.0 + arr))
    return {
        "threshold": threshold,
        "n_trades": n,
        "n_skipped_overlap": n_skipped_overlap,
        "win_rate": win_rate,
        "mean_net_per_trade_pct": mean_net * 100,
        "std_net_per_trade_pct": std_net * 100,
        "sharpe_per_trade": sharpe_pt,
        "annualised_sharpe": ann_sharpe,
        "trades_per_year": tpy,
        "total_return_pct": (equity - 1.0) * 100,
    }


def main():
    df, tp, sl, fee = load_merged()
    print(f"Loaded {len(df):,} rows  TP={tp:+.4f}  SL=-{sl:.4f}  fee={fee:.4f}")
    print(f"Round-trip fee = {2 * fee:.4f}  =>  break-even WR = {0.5 + fee / sl:.4f}")
    print()

    grid = [round(x, 3) for x in np.arange(0.50, 0.66, 0.01)]
    rows = []
    for thr in grid:
        r = simulate(df, thr, tp, sl, fee)
        if r is None:
            print(f"  thr={thr:.3f}: 0 trades")
            continue
        print(
            f"  thr={thr:.3f}  trades={r['n_trades']:5d}  "
            f"WR={r['win_rate']:.4f}  net/trd={r['mean_net_per_trade_pct']:+.4f}%  "
            f"annSharpe={r['annualised_sharpe']:+.3f}  total={r['total_return_pct']:+.2f}%"
        )
        rows.append(r)

    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(OUT_FILE, index=False)
        print(f"\nSaved {OUT_FILE}")

        positive = out[out["mean_net_per_trade_pct"] > 0]
        if len(positive):
            best = positive.sort_values("annualised_sharpe", ascending=False).iloc[0]
            print(
                f"\nBest profitable threshold: {best['threshold']:.3f}  "
                f"trades={int(best['n_trades'])}  WR={best['win_rate']:.4f}  "
                f"net/trd={best['mean_net_per_trade_pct']:+.4f}%  "
                f"annSharpe={best['annualised_sharpe']:+.3f}"
            )
        else:
            print("\nNo positive-net threshold in grid.")


if __name__ == "__main__":
    main()
