"""
Walk-forward paper-trade simulator.

Purpose
-------
The training pipeline produces a per-trade *expected PnL* number (the
``oof_expected_pnl_per_trade`` field in ``tbm_xgboost_model_v2_threshold.json``).
That number is computed analytically as ``win_rate * tp - (1-win_rate) * sl - fee``
on the OOF act-set.

This script *validates* that diagnostic by simulating the strategy
chronologically -- one bar at a time, across the full dataset -- using
the same OOF probabilities that produced the diagnostic.  Three things
become visible here that the analytic formula hides:

  1. Trade overlap.  When a barrier-hit-time of 24 bars is followed by
     a fresh signal at bar +1, the bot cannot take both.  The analytic
     formula treats every signal as a fresh trade.
  2. Fee compounding on the equity curve.  Per-trade EPnL averages out
     symmetric outcomes; the equity curve magnifies the drawdown.
  3. Per-symbol vs portfolio behaviour.  We hold at most one open
     position per symbol but can run multiple symbols in parallel --
     exactly what the live bot would do.

If the realised per-trade mean PnL from this simulator is within ~10%
of the analytic diagnostic, the diagnostic is trustworthy and we can
use it as the optimisation target for future feature work.  If the
simulator produces a materially worse (or better) number, there is a
hidden bug in the diagnostic and feature work would be wasted on a
broken scoreboard.

Inputs
------
  global_alpha_dataset.csv             (chronological, multi-symbol)
  tbm_xgboost_model_v2_oof.csv         (per-row OOF probabilities)
  tbm_xgboost_model_v2_threshold.json  (threshold + tp/sl/fee config)

Output
------
  pipeline_logs/walk_forward_report.json    (machine-readable stats)
  pipeline_logs/walk_forward_equity.csv     (equity curve, per-trade)

Run:
  python walk_forward_simulator.py
  python walk_forward_simulator.py --threshold 0.55 --fee-pct 0.0008
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data" / "processed" / "global_alpha_dataset.parquet"
OOF_FILE = ROOT / "tbm_xgboost_model_v2_oof.csv"
THRESHOLD_FILE = ROOT / "tbm_xgboost_model_v2_threshold.json"
LOGS_DIR = ROOT / "pipeline_logs"
LOGS_DIR.mkdir(exist_ok=True)


def realised_return(
    label: int, 
    tp_pct: float, 
    sl_pct: float,
    spread_pct: float = 0.02, # 2bp typical for majors
    slippage_pct: float = 0.01 # 1bp typical
) -> float:
    """Realistic mapping from TBM label to realised return.

    Adds bid/ask spread impact and execution slippage penalties.
    label = 1 (TP) -> +tp_pct - spread - slippage
    label = 0 (SL) -> -sl_pct - spread - slippage
    """
    penalty = (spread_pct + slippage_pct) / 100.0
    if label == 1:
        return float(tp_pct) - penalty
    return -abs(float(sl_pct)) - penalty


def simulate(
    df: pd.DataFrame,
    threshold: float,
    tp_pct: float,
    sl_pct: float,
    maker_fee_pct: float,
    taker_fee_pct: float,
    spread_pct: float,
    slippage_pct: float,
    max_concurrent_per_symbol: int = 1,
) -> dict:
    """Walk-forward chronological simulation.

    Parameters
    ----------
    df : DataFrame with columns
        timestamp, symbol, oof_proba, tbm_label, barrier_hit_time
        (must already be sorted by timestamp).
    threshold : probability cutoff for "act"
    tp_pct, sl_pct : barrier sizes (sl_pct > 0)
    maker_fee_pct : entry fee (assuming smart limit chase)
    taker_fee_pct : exit fee (assuming TP/SL market trigger)
    spread_pct : bid/ask spread in %
    slippage_pct : execution slippage in %
    max_concurrent_per_symbol : 1 enforces "no overlap on a symbol";
    """
    trades: list[dict] = []
    skipped_overlap = 0
    skipped_below_thr = 0

    blocked_until: dict[str, pd.Timestamp] = {}

    for _, row in df.iterrows():
        proba = float(row["oof_proba"])
        if proba < threshold:
            skipped_below_thr += 1
            continue

        sym = str(row["symbol"])
        ts = row["timestamp"]
        block = blocked_until.get(sym)
        if block is not None and ts < block:
            skipped_overlap += 1
            continue

        bht = int(row["barrier_hit_time"])
        label = int(row["tbm_label"])

        # Entry = Maker, Exit = Taker
        gross = realised_return(label, tp_pct, sl_pct, spread_pct, slippage_pct)
        total_fees = maker_fee_pct + taker_fee_pct
        net = gross - total_fees

        exit_ts = ts + pd.Timedelta(hours=bht)
        blocked_until[sym] = exit_ts

        trades.append(
            {
                "entry_ts": ts,
                "exit_ts": exit_ts,
                "symbol": sym,
                "proba": proba,
                "bars_held": bht,
                "label": label,
                "gross_return": gross,
                "fee_paid": total_fees,
                "net_return": net,
            }
        )

    if not trades:
        return {
            "n_trades": 0,
            "skipped_below_threshold": skipped_below_thr,
            "skipped_overlap": skipped_overlap,
            "warning": "no trades executed at this threshold",
        }

    tdf = pd.DataFrame(trades).sort_values("entry_ts").reset_index(drop=True)

    # Equity curve: compound *additive* simple returns.  We use simple
    # returns (1 + r) compounded; the multiplicative interpretation is
    # standard for a fixed-fraction-of-equity strategy.
    tdf["equity"] = (1.0 + tdf["net_return"]).cumprod()
    tdf["drawdown"] = tdf["equity"] / tdf["equity"].cummax() - 1.0

    n = len(tdf)
    n_wins = int((tdf["label"] == 1).sum())
    win_rate = n_wins / n

    mean_net = float(tdf["net_return"].mean())
    std_net = float(tdf["net_return"].std(ddof=1)) if n > 1 else 0.0
    # Sharpe per trade (NOT annualised) -- annualisation needs trades/yr.
    sharpe_per_trade = mean_net / std_net if std_net > 0 else 0.0

    # Approximate annualised Sharpe assuming trades happen at the rate
    # implied by the simulation period.
    span_hours = (tdf["exit_ts"].max() - tdf["entry_ts"].min()).total_seconds() / 3600.0
    trades_per_year = (n / span_hours) * (24 * 365) if span_hours > 0 else 0.0
    annualised_sharpe = sharpe_per_trade * np.sqrt(trades_per_year)

    final_equity = float(tdf["equity"].iloc[-1])
    total_return = final_equity - 1.0
    max_drawdown = float(tdf["drawdown"].min())

    # Save equity curve
    eq_path = LOGS_DIR / "walk_forward_equity.csv"
    tdf.to_csv(eq_path, index=False)

    return {
        "n_trades": n,
        "n_wins": n_wins,
        "win_rate": win_rate,
        "mean_net_return_per_trade": mean_net,
        "std_net_return_per_trade": std_net,
        "median_net_return_per_trade": float(tdf["net_return"].median()),
        "sharpe_per_trade": sharpe_per_trade,
        "annualised_sharpe": annualised_sharpe,
        "trades_per_year": trades_per_year,
        "final_equity": final_equity,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "skipped_below_threshold": skipped_below_thr,
        "skipped_overlap": skipped_overlap,
        "span_days": span_hours / 24.0,
        "equity_curve_path": str(eq_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward paper-trade simulator.")
    parser.add_argument(
        "--data", type=str, default=str(DATA_FILE), help="Input dataset CSV."
    )
    parser.add_argument(
        "--oof", type=str, default=str(OOF_FILE), help="OOF predictions CSV."
    )
    parser.add_argument(
        "--threshold-file",
        type=str,
        default=str(THRESHOLD_FILE),
        help="Threshold sidecar JSON.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override threshold (default: read from sidecar).",
    )
    parser.add_argument("--tp-pct", type=float, default=None)
    parser.add_argument("--sl-pct", type=float, default=None)
    parser.add_argument("--maker-fee-pct", type=float, default=0.0002, help="Default 0.02%")
    parser.add_argument("--taker-fee-pct", type=float, default=0.0004, help="Default 0.04%")
    parser.add_argument("--spread-pct", type=float, default=0.02, help="Default 2bp")
    parser.add_argument("--slippage-pct", type=float, default=0.01, help="Default 1bp")
    args = parser.parse_args()

    # ── Load sidecar config ──────────────────────────────────────────────
    with open(args.threshold_file) as f:
        cfg = json.load(f)

    threshold = (
        args.threshold if args.threshold is not None else float(cfg["threshold"])
    )
    tp_pct = args.tp_pct if args.tp_pct is not None else float(cfg.get("tp_pct", 0.010))
    sl_pct = (
        args.sl_pct if args.sl_pct is not None else float(cfg.get("sl_pct", -0.010))
    )
    diag_epnl = cfg.get("oof_expected_pnl_per_trade")

    print(f"Threshold:       {threshold:.4f}")
    print(f"TP / SL:         +{tp_pct:.4f} / {sl_pct:.4f}")
    print(f"Fees (M/T):      {args.maker_fee_pct*100:.4f}% / {args.taker_fee_pct*100:.4f}%")
    print(f"Spread/Slip:     {args.spread_pct:.4f}% / {args.slippage_pct:.4f}%")
    
    if diag_epnl is not None:
        print(f"Diagnostic EPnL: {diag_epnl * 100:.4f}% per trade  (analytic)")

    # ── Load and join data ───────────────────────────────────────────────
    print(f"\nLoading dataset from {args.data}...")
    if args.data.endswith(".parquet"):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data, parse_dates=["timestamp"])
    print(f"  rows: {len(df):,}  symbols: {df['symbol'].nunique()}")

    print(f"Loading OOF predictions from {args.oof}...")
    oof = pd.read_csv(args.oof)
    if len(oof) != len(df):
        raise SystemExit(
            f"OOF row count ({len(oof)}) != dataset row count ({len(df)}). "
            "Re-run train_tbm_model_v2.py --purged-kfold on this dataset."
        )

    df["oof_proba"] = oof["oof_proba"].values
    # Drop rows with no OOF prediction (the first/last fold edges occasionally
    # have NaN in pathological cases).
    df = df.dropna(subset=["oof_proba", "tbm_label", "barrier_hit_time"]).copy()
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    print(f"  usable rows: {len(df):,}")

    # ── Run simulation ───────────────────────────────────────────────────
    print("\nRunning walk-forward simulation...")
    stats = simulate(
        df=df,
        threshold=threshold,
        tp_pct=tp_pct,
        sl_pct=abs(sl_pct),
        maker_fee_pct=args.maker_fee_pct,
        taker_fee_pct=args.taker_fee_pct,
        spread_pct=args.spread_pct,
        slippage_pct=args.slippage_pct,
    )

    # ── Report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  WALK-FORWARD SIMULATION RESULTS")
    print("=" * 70)
    if stats["n_trades"] == 0:
        print("  No trades executed.  Threshold may be too tight.")
        print(f"  Skipped below threshold: {stats['skipped_below_threshold']:,}")
        out_path = LOGS_DIR / "walk_forward_report.json"
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        return

    print(f"  Span (days):                       {stats['span_days']:>10.1f}")
    print(f"  Trades executed:                   {stats['n_trades']:>10,}")
    print(
        f"  Skipped (below threshold):         {stats['skipped_below_threshold']:>10,}"
    )
    print(f"  Skipped (already in position):     {stats['skipped_overlap']:>10,}")
    print(f"  Trades per year (extrapolated):    {stats['trades_per_year']:>10,.0f}")
    print()
    print(f"  Win rate:                          {stats['win_rate']:>10.4f}")
    print(
        f"  Mean NET return per trade:         {stats['mean_net_return_per_trade'] * 100:>9.4f}%"
    )
    print(
        f"  Median NET return per trade:       {stats['median_net_return_per_trade'] * 100:>9.4f}%"
    )
    print(
        f"  Std NET return per trade:          {stats['std_net_return_per_trade'] * 100:>9.4f}%"
    )
    print()
    print(f"  Sharpe per trade:                  {stats['sharpe_per_trade']:>10.4f}")
    print(f"  Annualised Sharpe (approx):        {stats['annualised_sharpe']:>10.4f}")
    print()
    print(f"  Final equity (start=1.0):          {stats['final_equity']:>10.4f}")
    print(f"  Total return:                      {stats['total_return'] * 100:>9.2f}%")
    print(f"  Max drawdown:                      {stats['max_drawdown'] * 100:>9.2f}%")

    if diag_epnl is not None:
        delta = stats["mean_net_return_per_trade"] - diag_epnl
        rel = abs(delta) / max(abs(diag_epnl), 1e-9)
        print()
        print("  -- Diagnostic calibration check --")
        print(f"  Diagnostic EPnL/trade:             {diag_epnl * 100:>9.4f}%")
        print(
            f"  Simulated mean NET return:         {stats['mean_net_return_per_trade'] * 100:>9.4f}%"
        )
        print(f"  Absolute delta:                    {delta * 100:>9.4f}%")
        print(f"  Relative delta:                    {rel * 100:>9.2f}%")
        if rel < 0.10:
            verdict = "TRUSTWORTHY  (within 10%)"
        elif rel < 0.30:
            verdict = "ROUGHLY OK   (within 30%, watch for edge cases)"
        else:
            verdict = "MISCALIBRATED (over 30% off -- investigate)"
        print(f"  Verdict: {verdict}")

    print("=" * 70)
    print(f"\n  Equity curve saved to {stats['equity_curve_path']}")

    out_path = LOGS_DIR / "walk_forward_report.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Report saved to {out_path}\n")


if __name__ == "__main__":
    main()
