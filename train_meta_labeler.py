"""
Meta-Labeling Layer (López de Prado §3.6).

The primary TBM classifier (``tbm_xgboost_model_v2``) decides the *side* of the
trade — i.e. whether the conditions for a "win" label have been met.  Its main
weakness is that it is forced to commit to a label on every bar; many of those
labels are low-conviction and contribute most of the false positives.

A meta-labeler is a *second* classifier whose only job is to answer
**"given that the primary said BUY, will it be right this time?"**.  Its label
is the conjunction:

    meta_label = 1   if (primary signaled act) AND (TBM label == 1)
    meta_label = 0   otherwise (within the rows where primary said act)

By construction the meta-label depends on the primary's decision, so the meta
model is trained ONLY on the rows where the primary said "act".  At inference
time the final position-sizing rule is:

    final_proba = p_meta(act-and-win | features, p_primary)
    position    = kelly(final_proba)         (or 0 if final_proba < threshold)

This is strictly safer than acting on the primary alone: meta-labeling never
*adds* trades, it only *removes* the lowest-quality ones.

This script consumes the OOF predictions written by
``train_tbm_model_v2.py --purged-kfold`` so that meta training does not see
in-sample primary probabilities.
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
)

from purged_kfold import PurgedKFold, compute_average_uniqueness

warnings.filterwarnings("ignore")


class IsotonicCalibratorWrapper:
    """Module-level so pickle can find the class on load."""

    def __init__(self, base, iso, feature_order):
        self.base = base
        self.iso = iso
        self.feature_order = feature_order

    def predict_proba(self, X):
        X_ord = X[self.feature_order] if isinstance(X, pd.DataFrame) else X
        raw = self.base.predict_proba(X_ord)[:, 1]
        cal = np.clip(self.iso.transform(raw), 0.0, 1.0)
        return np.column_stack([1.0 - cal, cal])


# ── Configuration ────────────────────────────────────────────────────────────
DATA_FILE = "fracdiff_alpha_dataset.csv"
OOF_FILE = "tbm_xgboost_model_v2_oof.csv"
PRIMARY_THRESHOLD_FILE = "tbm_xgboost_model_v2_threshold.json"

META_MODEL_PATH = "meta_xgboost_model.json"
META_CALIB_PATH = "meta_xgboost_model_calibrated.pkl"
META_THRESHOLD_PATH = "meta_xgboost_model_threshold.json"

DROP_COLS = ["timestamp", "target_return_4h", "barrier_hit_time", "close_fd", "symbol"]
TARGET_COL = "tbm_label"
EMBARGO_FRACTION = 0.01


def find_best_threshold(y_true, probas, grid=None):
    # Wider grid (was 0.30-0.80) so a boundary-pinned optimum is visible.
    if grid is None:
        grid = np.arange(0.20, 0.851, 0.01)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (probas >= t).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        f = f1_score(y_true, pred)
        if f > best_f1:
            best_f1, best_t = f, t
    return float(best_t), float(best_f1)


def load_inputs(data_file: str = DATA_FILE):
    print(f"Loading dataset from {data_file}...")
    df = pd.read_csv(data_file)

    bht = (
        df["barrier_hit_time"].astype(int).values
        if "barrier_hit_time" in df.columns
        else None
    )

    print(f"Loading primary OOF predictions from {OOF_FILE}...")
    oof = pd.read_csv(OOF_FILE)
    if len(oof) != len(df):
        raise ValueError(
            f"OOF length ({len(oof)}) does not match dataset length ({len(df)}). "
            "Re-run train_tbm_model_v2.py --purged-kfold to regenerate."
        )

    print(f"Loading primary threshold from {PRIMARY_THRESHOLD_FILE}...")
    with open(PRIMARY_THRESHOLD_FILE) as f:
        primary_threshold = float(json.load(f)["threshold"])
    print(f"  primary threshold = {primary_threshold:.3f}")

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df_features = df.drop(columns=cols_to_drop + [TARGET_COL])
    if "close" in df_features.columns:
        df_features = df_features.drop(columns=["close"])

    y = df[TARGET_COL].astype(int).values
    p_primary = oof["oof_proba"].values

    return df_features, y, p_primary, primary_threshold, bht


def main():
    parser = argparse.ArgumentParser(description="Train the meta-labeling classifier.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--data",
        type=str,
        default=DATA_FILE,
        help=f"Input CSV (default: {DATA_FILE}).",
    )
    args = parser.parse_args()

    X_full, y_full, p_primary, primary_threshold, bht_full = load_inputs(args.data)

    # 1. Restrict to rows where the primary said "act" — meta-labeling only
    #    cares about those.  Rows where the primary already abstained are
    #    out of scope (the meta-model never has to override them).
    act_mask = p_primary >= primary_threshold
    n_act = int(act_mask.sum())
    n_total = len(y_full)
    print(
        f"\nPrimary signalled act on {n_act}/{n_total} rows "
        f"({100.0 * n_act / n_total:.2f}%)."
    )
    if n_act < 200:
        raise RuntimeError(
            "Too few primary 'act' rows to train a meta-labeler. "
            "Check the primary threshold or train on more data."
        )

    # 2. Build the meta-target: was the primary right?
    meta_y = (y_full == 1).astype(int)

    # 3. Build the META feature space.
    #
    # Concern #4 fix: previously meta used the *entire primary feature set*
    # plus `primary_proba`.  That gave meta the same view of the world as
    # the primary, so meta learned the same decision boundary and abstained
    # on ~1% of rows -- effectively identity.  For meta-labeling to add
    # value (López §3.6) it must condition on *orthogonal* information:
    # how confident the primary is, what *regime* the market is in, and
    # signal-density features that the primary does not explicitly model.
    #
    # Orthogonal meta features:
    #   - primary_proba                    (only primary-derived signal)
    #   - realized_vol_24h, atr_14_pct     (volatility regime)
    #   - rsi_14                           (mean-reversion / momentum regime)
    #   - hour_of_day, day_of_week         (calendar effects)
    #   - bars_since_last_signal           (signal-density / fatigue)
    #
    # We deliberately exclude direction-leaning features (close_fd_04,
    # buying/selling_rejection, volume_change_1h) so meta cannot just
    # re-derive the primary's signal.
    REGIME_COLS = ["realized_vol_24h", "atr_14_pct", "rsi_14"]
    available_regime = [c for c in REGIME_COLS if c in X_full.columns]

    meta_full = pd.DataFrame(index=X_full.index)
    meta_full["primary_proba"] = p_primary
    for c in available_regime:
        meta_full[c] = X_full[c].values

    # Calendar features from the timestamp column on the source df.
    # We re-load the timestamp here because X_full had it dropped.
    src_df = pd.read_csv(args.data)
    if "timestamp" in src_df.columns:
        ts = pd.to_datetime(src_df["timestamp"])
        meta_full["hour_of_day"] = ts.dt.hour.values
        meta_full["day_of_week"] = ts.dt.dayofweek.values

    # Bars since the previous primary "act" -- captures signal density.
    # When acts cluster (chop), meta tends to learn "abstain"; when acts
    # are sparse (clean trend), meta tends to learn "trust primary".
    act_indicator = (p_primary >= primary_threshold).astype(int)
    bars_since = np.zeros(len(act_indicator), dtype=float)
    counter = 0
    for i, a in enumerate(act_indicator):
        bars_since[i] = counter
        counter = 0 if a == 1 else counter + 1
    meta_full["bars_since_last_signal"] = bars_since

    print(f"Meta feature space (orthogonal to primary): {list(meta_full.columns)}")

    # 4. Subset to act rows.
    X_meta = meta_full.loc[act_mask].copy()
    y_meta = meta_y[act_mask]
    bht_meta = bht_full[act_mask] if bht_full is not None else None

    print(f"Meta dataset: X={X_meta.shape} positives={int(y_meta.sum())}")

    # 4. Purged k-fold OOF predictions for honest calibration + threshold.
    if bht_meta is not None:
        cv = PurgedKFold(
            n_splits=args.n_splits,
            barrier_hit_time=bht_meta,
            embargo_pct=EMBARGO_FRACTION,
        )
        weights = compute_average_uniqueness(bht_meta)
    else:
        # Fallback: simple sequential KFold without purging (label horizons unknown).
        from sklearn.model_selection import KFold

        cv = KFold(n_splits=args.n_splits, shuffle=False)
        weights = np.ones(len(y_meta))

    print(
        f"\n=== Meta purged k-fold (k={args.n_splits}, embargo={EMBARGO_FRACTION:.1%}) ==="
    )

    n = len(y_meta)
    oof_meta = np.full(n, np.nan)
    best_iters = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_meta), start=1):
        cut = int(len(train_idx) * 0.9)
        es_train = train_idx[:cut]
        es_val = train_idx[cut:]

        spw = (y_meta[es_train] == 0).sum() / max(1, (y_meta[es_train] == 1).sum())
        m = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=float(spw),
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,  # smaller: meta model is intentionally low-capacity
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
            early_stopping_rounds=50,
        )
        m.fit(
            X_meta.iloc[es_train],
            y_meta[es_train],
            sample_weight=weights[es_train],
            eval_set=[(X_meta.iloc[es_val], y_meta[es_val])],
            verbose=False,
        )
        oof_meta[test_idx] = m.predict_proba(X_meta.iloc[test_idx])[:, 1]
        best_iters.append(int(m.best_iteration or m.n_estimators))
        print(
            f"  fold {fold}: train={len(train_idx)} test={len(test_idx)} "
            f"best_iter={best_iters[-1]}"
        )

    valid = ~np.isnan(oof_meta)
    y_oof = y_meta[valid]
    p_oof = oof_meta[valid]

    # 5. Calibrate + pick threshold on OOF.
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_oof, y_oof)
    p_oof_cal = iso.transform(p_oof)
    best_threshold, best_f1 = find_best_threshold(y_oof, p_oof_cal)
    brier = brier_score_loss(y_oof, p_oof_cal)
    print(
        f"\nMeta OOF threshold (max F1): {best_threshold:.3f}  "
        f"(F1={best_f1:.4f}, Brier={brier:.4f})"
    )
    pred = (p_oof_cal >= best_threshold).astype(int)
    print("\n--- Meta OOF classification report (act vs abstain) ---")
    print(classification_report(y_oof, pred, zero_division=0))
    print("--- Meta OOF confusion matrix ---")
    print(confusion_matrix(y_oof, pred))

    # 6. Refit final meta model on ALL act rows.
    final_n = max(50, int(np.median(best_iters)))
    spw_full = (y_meta == 0).sum() / max(1, (y_meta == 1).sum())
    print(
        f"\nFitting final meta booster (n_estimators={final_n}, "
        f"scale_pos_weight={spw_full:.3f})..."
    )
    final_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=float(spw_full),
        n_estimators=final_n,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    final_model.fit(X_meta, y_meta, sample_weight=weights, verbose=False)

    # Wrap in calibrator (isotonic on OOF, applied to final-model raw probs).
    # IsotonicCalibratorWrapper is defined at module scope above so pickle
    # can resolve the class when api.py loads the artifact.
    calibrator = IsotonicCalibratorWrapper(final_model, iso, list(X_meta.columns))

    # 7. Persist artifacts.
    final_model.save_model(META_MODEL_PATH)
    print(f"Saved raw meta model to {META_MODEL_PATH}")

    with open(META_CALIB_PATH, "wb") as f:
        pickle.dump(calibrator, f)
    print(f"Saved calibrated meta wrapper to {META_CALIB_PATH}")

    with open(META_THRESHOLD_PATH, "w") as f:
        json.dump(
            {
                "threshold": best_threshold,
                "oof_f1": best_f1,
                "oof_brier": brier,
                "primary_threshold": primary_threshold,
                "n_splits": args.n_splits,
                "embargo_pct": EMBARGO_FRACTION,
                "method": "purged_kfold_meta_labeler",
                "feature_order": list(X_meta.columns),
                "note": (
                    "Meta-labeler operates on rows where primary >= primary_threshold. "
                    "Final act-decision: primary signals AND meta_proba >= threshold."
                ),
            },
            f,
            indent=2,
        )
    print(f"Saved meta threshold sidecar to {META_THRESHOLD_PATH}")


if __name__ == "__main__":
    main()
