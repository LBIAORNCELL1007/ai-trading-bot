"""
Feature ablation (Experiment I).

For each of the 15 features in the production model, drop it, retrain with
the SAME purged-kfold setup (seed=42, k=5, embargo=1%), and measure the
resulting per-symbol WR + global EPnL.  Compare to the baseline (all 15).

Interpretation:
  - drop -> metrics IMPROVE  : feature is actively HURTING (drop in production)
  - drop -> metrics flat     : feature is DEAD WEIGHT (drop saves complexity)
  - drop -> metrics WORSEN   : feature is USEFUL (keep)

Cheap because we reuse the same CSV; only the column-drop differs per run.
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from purged_kfold import PurgedKFold, compute_average_uniqueness  # noqa: E402

DATA = "global_alpha_dataset_1h_2pct.csv"
TARGET_COL = "tbm_label"
DROP_COLS = ["timestamp", "target_return_4h", "barrier_hit_time", "close_fd", "symbol"]
N_SPLITS = 5
EMBARGO_PCT = 0.01
TP = 0.02
SL = 0.02

# The 15 features used by the production model (matches live_features.FEATURE_COLUMNS).
FEATURES = [
    "volume",
    "funding_rate",
    "volume_change_1h",
    "buying_rejection",
    "selling_rejection",
    "realized_vol_24h",
    "rsi_14",
    "atr_14_pct",
    "bar_range_pct",
    "volume_zscore_24h",
    "close_to_vwap_24h",
    "funding_change_8h",
    "funding_zscore_7d",
    "funding_sign_streak",
    "close_fd_04",
]


def epnl(wr: float) -> float:
    return wr * TP - (1.0 - wr) * SL


def make_xgb(scale_pos_weight, n_estimators=500, **overrides):
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


def train_oof(X: pd.DataFrame, y: pd.Series, bht: np.ndarray):
    """Run purged-kfold and return calibrated OOF + raw OOF + fold best_iters."""
    weights = compute_average_uniqueness(bht)
    cv = PurgedKFold(n_splits=N_SPLITS, barrier_hit_time=bht, embargo_pct=EMBARGO_PCT)
    n = len(X)
    oof_raw = np.full(n, np.nan)
    iters = []
    for fold, (tr_idx, te_idx) in enumerate(cv.split(X), 1):
        cut = int(len(tr_idx) * 0.9)
        es_tr = tr_idx[:cut]
        es_va = tr_idx[cut:]
        y_tr = y.iloc[es_tr]
        spw = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
        m = make_xgb(scale_pos_weight=float(spw))
        m.fit(
            X.iloc[es_tr],
            y_tr,
            sample_weight=weights[es_tr],
            eval_set=[(X.iloc[es_va], y.iloc[es_va])],
            verbose=False,
        )
        oof_raw[te_idx] = m.predict_proba(X.iloc[te_idx])[:, 1]
        iters.append(int(m.best_iteration or m.n_estimators))
    valid = ~np.isnan(oof_raw)
    iso = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
    iso.fit(oof_raw[valid].reshape(-1, 1), y.values[valid])
    oof_cal = np.full(n, np.nan)
    oof_cal[valid] = np.clip(
        iso.predict_proba(oof_raw[valid].reshape(-1, 1))[:, 1], 0.0, 1.0
    )
    return oof_raw, oof_cal, iters


def metrics(df: pd.DataFrame, proba: np.ndarray):
    """Compute headline metrics matching _analyze_oof.py conventions."""
    valid = ~np.isnan(proba)
    df_v = df[valid].copy()
    df_v["proba"] = proba[valid]

    # Global thr=0.50 + funding>0.0001
    df_f = df_v[df_v["funding_rate"] > 0.0001]
    sel = df_f[df_f["proba"] >= 0.50]
    g_n = len(sel)
    g_wr = float(sel[TARGET_COL].mean()) if g_n else float("nan")
    g_ep = epnl(g_wr) if g_n else float("nan")

    # Per-symbol best thr (no filter, min 100 trades)
    per_sym = {}
    for sym, g in df_v.groupby("symbol"):
        best = None
        for thr in np.arange(0.40, 0.75, 0.01):
            s = g[g["proba"] >= thr]
            if len(s) < 100:
                break
            wr = s[TARGET_COL].mean()
            cand = (epnl(wr), float(thr), len(s), float(wr))
            if best is None or cand[0] > best[0]:
                best = cand
        if best is None:
            per_sym[sym] = None
        else:
            ep, thr, n, wr = best
            per_sym[sym] = {
                "thr": round(thr, 2),
                "n": n,
                "wr": round(wr, 4),
                "epnl": round(ep, 5),
            }

    # Brier on covered rows
    brier = (
        brier_score_loss(df_v[TARGET_COL].values, df_v["proba"].values)
        if len(df_v)
        else float("nan")
    )
    return {
        "global_thr050_filter": {"n": g_n, "wr": g_wr, "epnl": g_ep},
        "per_symbol": per_sym,
        "brier": brier,
    }


def main():
    print(f"Loading {DATA}...")
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    print(f"  rows={len(df):,}")

    bht = df["barrier_hit_time"].astype(int).values
    drop_cols_present = [c for c in DROP_COLS if c in df.columns]
    base_X = df.drop(columns=drop_cols_present + [TARGET_COL])
    y = df[TARGET_COL].astype(int)

    missing = [f for f in FEATURES if f not in base_X.columns]
    if missing:
        print(f"[FATAL] missing features in dataset: {missing}")
        return 1
    print(f"  features (15): {FEATURES}")

    runs = []  # list of (label, metrics_dict, fold_iters)

    # Baseline: all 15 features
    print("\n" + "=" * 78)
    print("RUN 0: BASELINE (all 15 features)")
    print("=" * 78)
    t0 = time.time()
    raw, cal, iters = train_oof(base_X, y, bht)
    m0 = metrics(df, cal)
    runs.append(("baseline", m0, iters, time.time() - t0))
    print(
        f"  global thr=0.50+filter:  n={m0['global_thr050_filter']['n']}  "
        f"WR={m0['global_thr050_filter']['wr']:.4f}  EPnL={m0['global_thr050_filter']['epnl']:+.5f}"
    )
    for sym in ("SOLUSDT", "DOGEUSDT", "XRPUSDT", "AVAXUSDT"):
        info = m0["per_symbol"].get(sym)
        if info:
            print(
                f"  {sym}: thr={info['thr']} n={info['n']} WR={info['wr']:.4f} EPnL={info['epnl']:+.5f}"
            )
    print(
        f"  fold best_iters: {iters}  brier={m0['brier']:.4f}  time={time.time() - t0:.0f}s"
    )

    # Drop-one ablations
    for i, feat in enumerate(FEATURES, 1):
        print("\n" + "=" * 78)
        print(f"RUN {i}/15: DROP {feat}")
        print("=" * 78)
        X = base_X.drop(columns=[feat])
        t0 = time.time()
        raw, cal, iters = train_oof(X, y, bht)
        m = metrics(df, cal)
        runs.append((f"drop_{feat}", m, iters, time.time() - t0))
        print(
            f"  global thr=0.50+filter:  n={m['global_thr050_filter']['n']}  "
            f"WR={m['global_thr050_filter']['wr']:.4f}  EPnL={m['global_thr050_filter']['epnl']:+.5f}"
        )
        for sym in ("SOLUSDT", "DOGEUSDT", "XRPUSDT", "AVAXUSDT"):
            info = m["per_symbol"].get(sym)
            if info:
                print(
                    f"  {sym}: thr={info['thr']} n={info['n']} WR={info['wr']:.4f} EPnL={info['epnl']:+.5f}"
                )
        print(
            f"  fold best_iters: {iters}  brier={m['brier']:.4f}  time={time.time() - t0:.0f}s"
        )

    # Final summary table
    print("\n" + "=" * 100)
    print("ABLATION SUMMARY (deltas vs baseline; positive = improves on dropping)")
    print("=" * 100)
    base_g = m0["global_thr050_filter"]
    print(
        f"{'config':<28} {'glob_n':>6} {'glob_WR':>8} {'glob_EPnL':>10} "
        f"{'SOL_WR':>7} {'DOGE_WR':>8} {'XRP_WR':>7} {'AVAX_WR':>8} {'brier':>7}"
    )
    for label, m, iters, dur in runs:
        g = m["global_thr050_filter"]
        sol = m["per_symbol"].get("SOLUSDT") or {}
        dog = m["per_symbol"].get("DOGEUSDT") or {}
        xrp = m["per_symbol"].get("XRPUSDT") or {}
        avx = m["per_symbol"].get("AVAXUSDT") or {}
        print(
            f"{label:<28} {g['n']:>6} {g['wr']:>8.4f} {g['epnl']:>+10.5f} "
            f"{sol.get('wr', 0):>7.4f} {dog.get('wr', 0):>8.4f} {xrp.get('wr', 0):>7.4f} {avx.get('wr', 0):>8.4f} {m['brier']:>7.4f}"
        )

    Path("ablation_results.json").write_text(
        json.dumps(
            [
                {"label": l, "metrics": m, "fold_iters": it, "time_s": d}
                for l, m, it, d in runs
            ],
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print("\nFull results -> ablation_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
