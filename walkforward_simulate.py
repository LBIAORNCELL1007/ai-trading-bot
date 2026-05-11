"""
Walk-forward retraining simulator (Experiment B).

Goal: directly attack P3 instability (38.8% WR most recent year on the static
3y purged-kfold model).  If the model degrades because feature->outcome
relationships drift over time, then retraining on a trailing 2y window
each month should adapt to drift and recover the per-symbol edge.

Method (Lopez de Prado section 7.4 walk-forward, plus calibration on
held-out train tail):

  1. Sort dataset by timestamp.
  2. Define test windows: last 12 months, monthly stride, no overlap.
  3. For each test window starting at t0:
        train_end   = t0 - embargo (168 bars = 1 week, > max barrier_hit_time)
        train_start = train_end - 2y
        Within train: last 5% reserved for Platt calibration + early stopping.
        Train XGBoost with sample-uniqueness weights on train ex-ES slice.
        Calibrate Platt on ES slice raw probas.
        Predict on test window with model + calibrator.
  4. Concatenate test-window predictions into a walk-forward OOF.
  5. Compare to the static-model P3 on identical rows.

We use the SAME features and the SAME XGBoost params as train_tbm_model_v2.py
so any difference in OOF metrics is purely due to the retraining schedule.
"""

from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from purged_kfold import compute_average_uniqueness  # noqa: E402

# Mirrors train_tbm_model_v2.py
DROP_COLS = ["timestamp", "target_return_4h", "barrier_hit_time", "close_fd", "symbol"]
TARGET_COL = "tbm_label"

# Walk-forward knobs (defaults match the proposal in this experiment)
TRAIN_DAYS = 730  # 2y trailing
TEST_DAYS = 30  # 1 month per fold
N_TEST_WINDOWS = 12  # last 12 months
EMBARGO_BARS = 168  # 1 week, > max(barrier_hit_time)=24
ES_FRACTION = 0.05  # last 5% of train used for ES + Platt calibration


def make_xgb(scale_pos_weight: float, n_estimators: int = 1000, **overrides):
    params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    params.update(overrides)
    return xgb.XGBClassifier(**params)


def prepare_xy(df: pd.DataFrame):
    bht = df["barrier_hit_time"].astype(int).values
    cols = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=cols + [TARGET_COL])
    y = df[TARGET_COL].astype(int).values
    return X, y, bht


def train_one_fold(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    fold_id: int,
):
    """Train on df_train (chronological), predict on df_test. Returns
    (raw_proba_test, cal_proba_test, best_iter, n_train, n_test).
    """
    X_tr_full, y_tr_full, bht_tr_full = prepare_xy(df_train)
    X_te, y_te, _ = prepare_xy(df_test)

    n_tr_full = len(X_tr_full)
    es_n = max(500, int(n_tr_full * ES_FRACTION))
    fit_n = n_tr_full - es_n

    X_fit = X_tr_full.iloc[:fit_n]
    y_fit = y_tr_full[:fit_n]
    bht_fit = bht_tr_full[:fit_n]

    X_es = X_tr_full.iloc[fit_n:]
    y_es = y_tr_full[fit_n:]

    weights = compute_average_uniqueness(bht_fit)

    spw = (y_fit == 0).sum() / max(1, (y_fit == 1).sum())
    model = make_xgb(scale_pos_weight=float(spw))
    model.fit(
        X_fit,
        y_fit,
        sample_weight=weights,
        eval_set=[(X_es, y_es)],
        verbose=False,
    )
    best_iter = int(model.best_iteration or model.n_estimators)

    # Platt calibrator on ES slice raw probas (mimics production: calibrate
    # on the most recent labelled tail of training).
    raw_es = model.predict_proba(X_es)[:, 1]
    iso = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
    iso.fit(raw_es.reshape(-1, 1), y_es)

    raw_te = model.predict_proba(X_te)[:, 1]
    cal_te = np.clip(iso.predict_proba(raw_te.reshape(-1, 1))[:, 1], 0.0, 1.0)

    print(
        f"  fold {fold_id:2d}: train={n_tr_full:>6}  test={len(X_te):>5}  "
        f"best_iter={best_iter:>3}  base_rate_train={y_fit.mean():.3f}  "
        f"base_rate_test={y_te.mean():.3f}"
    )
    return raw_te, cal_te, best_iter, n_tr_full, len(X_te)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default="global_alpha_dataset_1h_2pct.csv",
        help="Input dataset CSV (must contain timestamp, tbm_label, barrier_hit_time, symbol).",
    )
    ap.add_argument(
        "--output",
        default="walkforward_oof.csv",
        help="Walk-forward OOF predictions output (one row per test bar).",
    )
    ap.add_argument("--train-days", type=int, default=TRAIN_DAYS)
    ap.add_argument("--test-days", type=int, default=TEST_DAYS)
    ap.add_argument("--n-folds", type=int, default=N_TEST_WINDOWS)
    ap.add_argument("--embargo-bars", type=int, default=EMBARGO_BARS)
    args = ap.parse_args()

    print(f"Loading {args.data}...")
    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    print(
        f"  rows={len(df):,}  timestamp range: {df['timestamp'].min()}  ->  {df['timestamp'].max()}"
    )

    test_end = df["timestamp"].max()
    test_start_first = test_end - pd.Timedelta(days=args.n_folds * args.test_days)
    print(
        f"\nWalk-forward plan: {args.n_folds} folds x {args.test_days}d test windows "
        f"({test_start_first} -> {test_end}). "
        f"Each fold trains on {args.train_days}d trailing, embargo={args.embargo_bars} bars."
    )

    # Pre-build a row -> timestamp map for fast slicing.
    ts = df["timestamp"].values

    all_idx_test = []
    all_raw = []
    all_cal = []
    fold_iters = []

    for k in range(args.n_folds):
        fold_test_start = test_start_first + pd.Timedelta(days=k * args.test_days)
        fold_test_end = fold_test_start + pd.Timedelta(days=args.test_days)

        test_mask = (df["timestamp"] >= fold_test_start) & (
            df["timestamp"] < fold_test_end
        )
        test_rows = np.where(test_mask)[0]
        if len(test_rows) == 0:
            print(
                f"  fold {k + 1:2d}: no test rows in [{fold_test_start}, {fold_test_end}); skipping"
            )
            continue

        # Train slice ends `embargo_bars` rows before fold_test_start in time.
        # Easier: take all rows with timestamp < fold_test_start, then drop
        # the last `embargo_bars` rows (they may have label horizons that
        # touch the test window).
        pretrain_mask = df["timestamp"] < fold_test_start
        pretrain_rows = np.where(pretrain_mask)[0]
        if len(pretrain_rows) <= args.embargo_bars + 1000:
            print(
                f"  fold {k + 1:2d}: insufficient pretrain rows ({len(pretrain_rows)}); skipping"
            )
            continue
        train_end_row = pretrain_rows[-1] - args.embargo_bars
        train_end_ts = df.iloc[train_end_row]["timestamp"]
        train_start_ts = train_end_ts - pd.Timedelta(days=args.train_days)
        train_mask = (df["timestamp"] >= train_start_ts) & (
            df["timestamp"] <= train_end_ts
        )
        train_rows = np.where(train_mask)[0]

        df_train = df.iloc[train_rows].copy()
        df_test = df.iloc[test_rows].copy()

        raw_te, cal_te, best_iter, n_tr, n_te = train_one_fold(
            df_train, df_test, fold_id=k + 1
        )
        all_idx_test.append(test_rows)
        all_raw.append(raw_te)
        all_cal.append(cal_te)
        fold_iters.append(best_iter)

    if not all_idx_test:
        print("No folds completed.  Aborting.")
        return 1

    idx = np.concatenate(all_idx_test)
    raw = np.concatenate(all_raw)
    cal = np.concatenate(all_cal)

    bi = np.array(fold_iters, dtype=float)
    print(
        f"\nFold best_iter stats: n={len(bi)}  mean={bi.mean():.1f}  "
        f"std={bi.std():.1f}  min={bi.min():.0f}  max={bi.max():.0f}  "
        f"cv={(bi.std() / bi.mean() if bi.mean() > 0 else 0):.2f}"
    )

    # Build OOF dataframe aligned to dataset rows.  Rows NOT covered by any
    # walk-forward fold get NaN (train-only or pre-walk-forward history).
    oof = pd.DataFrame(
        {
            "y": np.full(len(df), np.nan, dtype=float),
            "oof_proba": np.full(len(df), np.nan, dtype=float),
            "oof_proba_raw": np.full(len(df), np.nan, dtype=float),
        }
    )
    oof.loc[idx, "y"] = df.iloc[idx][TARGET_COL].values
    oof.loc[idx, "oof_proba"] = cal
    oof.loc[idx, "oof_proba_raw"] = raw

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(args.output, index=False)
    print(f"Saved walk-forward OOF -> {args.output}  (covered rows: {len(idx):,})")

    # Quick comparison: WR at thr=0.50 + funding filter on covered rows
    df_cov = df.iloc[idx].copy().reset_index(drop=True)
    df_cov["proba"] = cal
    fund = df_cov[df_cov["funding_rate"] > 0.0001]
    sel = fund[fund["proba"] >= 0.50]
    if len(sel) > 0:
        wr = sel[TARGET_COL].mean()
        print(
            f"\nQuick check (thr=0.50 + funding>0.0001): n={len(sel)}  WR={wr:.4f}  "
            f"EPnL@TP=SL=2%: {(2 * wr - 1) * 0.02:+.4f}"
        )
    else:
        print("\nQuick check: no trades pass thr=0.50 + funding filter.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
