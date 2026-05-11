import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

try:
    # sklearn >= 1.6 uses FrozenEstimator; cv='prefit' removed in 1.8.
    from sklearn.frozen import FrozenEstimator

    _HAS_FROZEN = True
except ImportError:
    _HAS_FROZEN = False
import json
import warnings
import argparse

from purged_kfold import PurgedKFold, compute_average_uniqueness

warnings.filterwarnings("ignore")


class IsotonicCalibratorWrapper:
    """Mimics CalibratedClassifierCV(prefit).predict_proba shape.

    Defined at module scope (not inside a function) so pickle can find
    the class on load.  api.py and load_calibrator helpers import this
    name to deserialize the artifact.

    Despite the historical name, this wrapper now holds a *sigmoid* (Platt)
    calibrator by default -- isotonic was found to collapse 46k unique
    raw probas to ~21 plateaus when the underlying classifier signal is
    weak, destroying threshold-tuning granularity.  See git log
    around "calibration fix" for context.  The wrapper accepts any
    object with either an ``.transform`` method (isotonic) or a
    ``.predict_proba`` method (LogisticRegression / Platt) so the
    artifact format is forward-compatible.
    """

    def __init__(self, base_estimator, calibrator):
        self.base_estimator = base_estimator
        # Keep the legacy attribute name for back-compat with old pickles.
        self.iso = calibrator

    def _apply_calibrator(self, raw):
        cal = self.iso
        if hasattr(cal, "transform"):
            # IsotonicRegression
            out = cal.transform(raw)
        elif hasattr(cal, "predict_proba"):
            # LogisticRegression (Platt) — expects 2-D input.
            out = cal.predict_proba(np.asarray(raw).reshape(-1, 1))[:, 1]
        else:
            raise TypeError(f"Unsupported calibrator type: {type(cal).__name__}")
        return np.clip(out, 0.0, 1.0)

    def predict_proba(self, X):
        raw = self.base_estimator.predict_proba(X)[:, 1]
        cal = self._apply_calibrator(raw)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# ── Configuration ────────────────────────────────────────────────────────────
DATA_FILE = "fracdiff_alpha_dataset.csv"
TARGET_COL = "tbm_label"
# DROP_COLS — leakage-prevention list.  See README of train_model.py for the
# full justification of each entry; the critical ones for accuracy are:
#   target_return_4h  (regression label = close.shift(-4)−close — pure leak)
#   barrier_hit_time  (TBM resolution time, knows the future)
#   close_fd          (legacy name; collides with current close_fd_04)
#   symbol            (text column in multi-symbol global datasets — text is
#                      not a valid XGBoost feature; the per-symbol scale
#                      effects are already captured by the z-scored
#                      volume/OI/funding columns, so dropping it lets one
#                      model learn the cross-asset structure cleanly)
#   bar_range_pct     (REMOVED 2026-05 after leave-one-out ablation: dropping
#                      this feature improved global WR by +2.7pp and SOL WR
#                      by +3.0pp.  Highly correlated with realized_vol_24h /
#                      atr_14_pct so it added noise without information.
#                      Still computed in build_global_dataset.py for dataset
#                      stability, but excluded at training time.)
DROP_COLS = [
    "timestamp",
    "target_return_4h",
    "barrier_hit_time",
    "close_fd",
    "symbol",
    "bar_range_pct",
]
MODEL_SAVE_PATH = "tbm_xgboost_model_v2.json"
CALIBRATOR_SAVE_PATH = "tbm_xgboost_model_v2_calibrated.pkl"
THRESHOLD_SAVE_PATH = "tbm_xgboost_model_v2_threshold.json"
EMBARGO_FRACTION = 0.01  # 1% embargo gap to prevent label-leakage between train/val

def load_and_prepare_data(data_file: str = DATA_FILE):
    """
    Loads the fractionally differenced dataset and prepares it for time-series training.
    Supports both CSV and Parquet formats.
    """
    print(f"Loading data from {data_file}...")
    if data_file.endswith(".parquet"):
        df = pd.read_parquet(data_file)
    else:
        df = pd.read_csv(data_file)

    # Capture barrier_hit_time BEFORE dropping it — it is needed downstream
    # for sample uniqueness weights and purged k-fold splits.
    bht = (
        df["barrier_hit_time"].astype(int).values
        if "barrier_hit_time" in df.columns
        else None
    )

    # If timestamp is a column (not index), it will be dropped
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    print(f"Dropping non-feature columns: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

    # Separate features (X) and target (y)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    # Explicitly drop 'close' if it leaked in
    if "close" in X.columns:
        print("Dropping 'close' from features.")
        X = X.drop(columns=["close"])

    print(f"Features used for training ({len(X.columns)}): {list(X.columns)}")
    return X, y, bht


def time_series_split(X, y, train_size=0.70, val_size=0.15, embargo=EMBARGO_FRACTION):
    """
    Sequential split with an EMBARGO gap between train/val and val/test
    (López de Prado §7.4).  Without an embargo, labels generated by TBM near
    the train/val boundary can leak future information into the validation
    set because each label looks `timeHorizon` bars forward.
    """
    print(
        f"\nSplitting data sequentially ({int(train_size * 100)}/{int(val_size * 100)}/{int((1 - train_size - val_size) * 100)}) with {embargo * 100:.1f}% embargo..."
    )

    n = len(X)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    embargo_n = int(n * embargo)

    X_train = X.iloc[: train_end - embargo_n]
    y_train = y.iloc[: train_end - embargo_n]

    X_val = X.iloc[train_end : val_end - embargo_n]
    y_val = y.iloc[train_end : val_end - embargo_n]

    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    print(f"Train samples: {len(X_train)}  (embargo={embargo_n} bars dropped at tail)")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def find_best_threshold(
    y_true,
    probas,
    grid=None,
    *,
    tp_pct: float = 0.010,
    sl_pct: float = 0.010,
    fee_pct: float = 0.0008,
    objective: str = "expected_pnl",
):
    """
    Pick a probability threshold on calibrated probabilities.

    Two objectives are supported:

    * ``"expected_pnl"`` (default): maximise the dollar-weighted edge per
      trade.  For every threshold we compute the empirical win-rate on
      *acted* rows (where ``proba >= t``); expected pnl per trade is
      ``win_rate * tp_pct - (1 - win_rate) * sl_pct - fee_pct``.  This
      directly aligns the threshold with what the strategy actually earns.

    * ``"f1"`` (legacy): maximise F1.  Kept so the unit tests and old
      sidecars stay reproducible.

    A wider grid is searched (``[0.20, 0.85]``) so we can tell when the
    optimum is degenerate (pinned at the boundary).  The caller should
    inspect the returned ``boundary_warning`` flag.

    Returns
    -------
    best_t : float
    best_score : float          (objective value at best_t)
    info : dict                 diagnostics: trade_rate, win_rate at best_t,
                                 grid sweep, boundary_warning
    """
    if grid is None:
        # Wider than the old 0.30-0.80 so we can detect degenerate optima.
        grid = np.arange(0.20, 0.851, 0.01)

    sweep = []
    for t in grid:
        pred = (probas >= t).astype(int)
        n_act = int(pred.sum())
        if n_act == 0 or n_act == len(pred):
            sweep.append((float(t), 0.0, 0.0, 0.0, 0.0))
            continue
        tp = int(((pred == 1) & (y_true == 1)).sum())
        win_rate = tp / n_act
        trade_rate = n_act / len(pred)
        f1 = f1_score(y_true, pred, zero_division=0)
        # Expected PnL per *trade* (not per bar) -- what a sniper strategy cares about.
        # Round-trip fee = 2 * fee_pct (entry + exit, both at notional).  The
        # walk-forward simulator surfaced this: a single-side fee under-counts
        # cost by exactly half the bid/ask spread per trade.
        epnl = win_rate * tp_pct - (1.0 - win_rate) * abs(sl_pct) - 2.0 * fee_pct
        sweep.append(
            (float(t), float(epnl), float(f1), float(win_rate), float(trade_rate))
        )

    if objective == "expected_pnl":
        score_idx = 1
    elif objective == "f1":
        score_idx = 2
    else:
        raise ValueError(f"unknown objective: {objective}")

    # Pick best threshold among non-degenerate rows (trade_rate in (0, 1)).
    valid = [row for row in sweep if 0.0 < row[4] < 1.0]
    if not valid:
        return (
            0.5,
            0.0,
            {
                "boundary_warning": True,
                "reason": "no non-degenerate threshold found",
                "sweep": sweep,
            },
        )
    best_row = max(valid, key=lambda r: r[score_idx])
    best_t, _, best_f1, best_wr, best_tr = best_row
    best_score = best_row[score_idx]

    # Boundary guardrail: warn if the optimum is at the lowest or highest 2%
    # of the grid (within one step of either end).  A boundary optimum means
    # F1/PnL is monotone in the threshold and the model has not learned
    # discriminative structure -- saving such a threshold gives a false
    # sense of confidence.
    grid_min, grid_max = float(grid[0]), float(grid[-1])
    step = float(grid[1] - grid[0])
    boundary_warning = (best_t <= grid_min + step) or (best_t >= grid_max - step)

    info = {
        "objective": objective,
        "best_threshold": best_t,
        "expected_pnl_at_best": best_row[1],
        "f1_at_best": best_f1,
        "win_rate_at_best": best_wr,
        "trade_rate_at_best": best_tr,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "fee_pct": fee_pct,
        "grid_min": grid_min,
        "grid_max": grid_max,
        "boundary_warning": bool(boundary_warning),
        "sweep": sweep,
    }
    return float(best_t), float(best_score), info


def _make_xgb(scale_pos_weight: float, **overrides) -> xgb.XGBClassifier:
    """Factory so the purged-CV path and the simple-split path use identical hyper-params.

    Concern #2 fix: previously capped ``n_estimators`` at 100 to dampen
    fold-to-fold variance.  After the ±2%/48h relabel that variance is
    already tame (cv_ratio≈0.12), and 4/5 folds were hitting the 100 ceiling
    -- the model wants more capacity.  Raised to 500 with ``learning_rate=0.05``;
    early stopping in :func:`train_with_purged_kfold` keeps any single fold
    from over-training, and the ``hit_ceiling`` instability check still fires
    if folds saturate the new limit.
    """
    params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    params.update(overrides)
    return xgb.XGBClassifier(**params)


def train_with_purged_kfold(
    X: pd.DataFrame,
    y: pd.Series,
    barrier_hit_time: np.ndarray,
    n_splits: int = 5,
    embargo_pct: float = EMBARGO_FRACTION,
):
    """
    Purged k-fold training (López de Prado §4.5 + §7.4).

    1. Computes per-sample average uniqueness weights from ``barrier_hit_time``.
    2. Generates out-of-fold (OOF) predictions using ``PurgedKFold`` so that
       calibration and threshold selection do not see leaked labels.
    3. Fits an isotonic calibrator on the OOF probabilities.
    4. Refits a final booster on ALL data (with the same uniqueness weights)
       and wraps it with the OOF-fit calibrator.

    Returns
    -------
    final_model : xgb.XGBClassifier
        The booster fit on all data (no early stopping — uses the median of
        the per-fold ``best_iteration`` values).
    calibrator : object exposing predict_proba (the isotonic regressor wrapped
        in a tiny adapter for parity with CalibratedClassifierCV).
    best_threshold, best_f1, brier : floats
        Diagnostics computed on the OOF predictions.
    oof_proba : np.ndarray
        Per-row OOF probability of class 1.  Useful for downstream meta-labeling.
    """
    print(f"\n=== Purged K-Fold training (k={n_splits}, embargo={embargo_pct:.1%}) ===")
    n = len(X)

    print("Computing sample uniqueness weights from barrier_hit_time...")
    weights = compute_average_uniqueness(barrier_hit_time)
    print(
        f"  weights: min={weights.min():.4f} mean={weights.mean():.4f} "
        f"max={weights.max():.4f}"
    )

    cv = PurgedKFold(
        n_splits=n_splits,
        barrier_hit_time=barrier_hit_time,
        embargo_pct=embargo_pct,
    )

    oof_proba = np.full(n, np.nan)
    best_iters = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        # Use a small slice of train as a holdout for early stopping (last 10%).
        cut = int(len(train_idx) * 0.9)
        es_train = train_idx[:cut]
        es_val = train_idx[cut:]

        y_tr = y.iloc[es_train]
        spw = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
        m = _make_xgb(scale_pos_weight=float(spw), early_stopping_rounds=50)
        m.fit(
            X.iloc[es_train],
            y_tr,
            sample_weight=weights[es_train],
            eval_set=[(X.iloc[es_val], y.iloc[es_val])],
            verbose=False,
        )
        oof_proba[test_idx] = m.predict_proba(X.iloc[test_idx])[:, 1]
        best_iters.append(int(m.best_iteration or m.n_estimators))
        print(
            f"  fold {fold}: train={len(train_idx)}  test={len(test_idx)}  "
            f"best_iter={best_iters[-1]}"
        )

    valid = ~np.isnan(oof_proba)
    if not valid.any():
        raise RuntimeError("Purged k-fold produced no OOF predictions.")

    # Concern #2 fix: fold-stability check on best_iter.  Two failure modes
    # we surface:
    #   (a) std/mean ratio high  -> different folds fit very different signals
    #   (b) any fold hit the n_estimators ceiling  -> early stopping never
    #       fired, that fold is overfit relative to the others
    bi = np.array(best_iters, dtype=float)
    bi_mean = float(bi.mean()) if bi.size else 0.0
    bi_std = float(bi.std()) if bi.size else 0.0
    n_estimators_ceiling = int(_make_xgb(scale_pos_weight=1.0).n_estimators)
    hit_ceiling = bi.size > 0 and bool((bi >= n_estimators_ceiling).any())
    cv_ratio = (bi_std / bi_mean) if bi_mean > 0 else 0.0
    fold_unstable = bi.size >= 3 and (cv_ratio > 1.0 or hit_ceiling)
    print(
        f"\nFold best_iter stats: mean={bi_mean:.1f} std={bi_std:.1f} "
        f"min={bi.min():.0f} max={bi.max():.0f} cv={cv_ratio:.2f} "
        f"hit_ceiling={hit_ceiling}"
    )
    if fold_unstable:
        reasons = []
        if cv_ratio > 1.0:
            reasons.append(f"std/mean={cv_ratio:.2f} > 1.0")
        if hit_ceiling:
            reasons.append(f"some fold hit n_estimators={n_estimators_ceiling}")
        print(
            f"  [WARN] Folds are unstable ({'; '.join(reasons)}). "
            "Different time periods are fitting different signals -- treat "
            "OOF metrics as optimistic. Likely root causes: regime shift, "
            "weak features, or insufficient embargo."
        )

    y_oof = y.values[valid]
    p_oof = oof_proba[valid]

    # ── Calibration ────────────────────────────────────────────────────
    # We use SIGMOID (Platt) calibration here, not isotonic.  Isotonic
    # regression on a weak signal collapses the proba space to ~20
    # plateaus, which destroys the threshold tuner's ability to find
    # operating points in the high-confidence tail.  Platt is a 2-
    # parameter logistic that preserves all granularity while still
    # remapping raw scores to better-calibrated probabilities.
    print("\nFitting sigmoid (Platt) calibrator on OOF predictions...")
    iso = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
    iso.fit(p_oof.reshape(-1, 1), y_oof)

    p_oof_cal = iso.predict_proba(p_oof.reshape(-1, 1))[:, 1]
    best_threshold, best_score, thr_info = find_best_threshold(
        y_oof, p_oof_cal, objective="expected_pnl"
    )
    # Stash fold-stability stats inside thr_info so the threshold sidecar
    # can record them alongside the threshold.
    thr_info["fold_best_iter_mean"] = bi_mean
    thr_info["fold_best_iter_std"] = bi_std
    thr_info["fold_best_iters"] = [int(x) for x in best_iters]
    thr_info["fold_unstable"] = bool(fold_unstable)
    best_f1 = thr_info["f1_at_best"]
    brier = brier_score_loss(y_oof, p_oof_cal)
    print(
        f"OOF threshold (max expected PnL): {best_threshold:.3f}  "
        f"(EPnL/trade={best_score * 100:.3f}%, win={thr_info['win_rate_at_best']:.3f}, "
        f"trade_rate={thr_info['trade_rate_at_best']:.3f}, F1={best_f1:.4f}, "
        f"Brier={brier:.4f})"
    )
    if thr_info["boundary_warning"]:
        print(
            "  [WARN] Optimum threshold is pinned at the search-grid boundary. "
            "The model has not learned discriminative structure -- treat the "
            "saved threshold as unreliable until features/data improve."
        )

    # Refit a final booster on ALL data, no early stopping. Use the median
    # best-iteration count from the folds as n_estimators.
    final_n = max(50, int(np.median(best_iters)))
    spw_full = (y == 0).sum() / max(1, (y == 1).sum())
    print(
        f"\nFitting final booster on full dataset (n_estimators={final_n}, "
        f"scale_pos_weight={spw_full:.3f}, weighted by sample uniqueness)..."
    )
    final_model = _make_xgb(scale_pos_weight=float(spw_full), n_estimators=final_n)
    # Remove early_stopping_rounds for the final fit
    final_model.set_params(early_stopping_rounds=None)
    final_model.fit(X, y, sample_weight=weights, verbose=False)

    # Wrap the calibrator in a module-level helper so the artifact
    # pickles cleanly (see IsotonicCalibratorWrapper above -- legacy name,
    # now holds a Platt calibrator).
    calibrator = IsotonicCalibratorWrapper(final_model, iso)
    # Build a calibrated OOF series aligned to the *full* row index.  Rows
    # that had no OOF prediction (NaN) stay NaN; rows that had a raw OOF
    # proba get sigmoid-transformed.  This is the probability scale on
    # which the saved threshold lives, so it is what every downstream
    # consumer (meta-labeler, walk-forward simulator, regime tuner)
    # MUST filter on.
    oof_proba_cal = np.full_like(oof_proba, np.nan, dtype=float)
    valid_mask = ~np.isnan(oof_proba)
    oof_proba_cal[valid_mask] = np.clip(
        iso.predict_proba(oof_proba[valid_mask].reshape(-1, 1))[:, 1], 0.0, 1.0
    )

    return (
        final_model,
        calibrator,
        best_threshold,
        best_f1,
        brier,
        oof_proba,
        oof_proba_cal,
        thr_info,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train the v2 TBM XGBoost classifier with optional purged k-fold CV."
    )
    parser.add_argument(
        "--purged-kfold",
        action="store_true",
        help=(
            "Use purged k-fold CV with sample-uniqueness weights "
            "(López de Prado §4.5 + §7.4) instead of the simple 70/15/15 split. "
            "Recommended whenever barrier_hit_time is available in the dataset."
        ),
    )
    parser.add_argument(
        "--n-splits", type=int, default=5, help="Number of purged folds (default: 5)."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DATA_FILE,
        help=(
            f"Input CSV (default: {DATA_FILE}). For the multi-symbol path, "
            "pass --data global_alpha_dataset.csv."
        ),
    )
    args = parser.parse_args()

    try:
        # 1. Load Data
        X, y, bht = load_and_prepare_data(args.data)

        if args.purged_kfold:
            if bht is None:
                raise ValueError(
                    "--purged-kfold requires the dataset to contain a "
                    "'barrier_hit_time' column. Re-run tbm_labeler.py."
                )
            (
                model,
                calibrator,
                best_threshold,
                best_f1,
                brier,
                oof_proba,
                oof_proba_cal,
                thr_info,
            ) = train_with_purged_kfold(
                X, y, bht, n_splits=args.n_splits, embargo_pct=EMBARGO_FRACTION
            )

            # Persist OOF probabilities -- BOTH raw and calibrated.
            # Downstream consumers (walk-forward simulator, meta-labeler,
            # regime tuner) MUST filter on `oof_proba_cal` because the
            # saved `threshold` field is on the calibrated scale.  The raw
            # column is kept only for diagnostic plots / introspection.
            oof_path = "tbm_xgboost_model_v2_oof.csv"
            pd.DataFrame(
                {
                    "y": y.values,
                    "oof_proba": oof_proba_cal,  # << calibrated, threshold-comparable
                    "oof_proba_raw": oof_proba,  # << pre-calibration, diagnostic only
                }
            ).to_csv(oof_path, index=False)
            print(f"Saved OOF predictions (calibrated) to {oof_path}")

            # Save artifacts
            print(f"\nSaving raw model to {MODEL_SAVE_PATH}...")
            model.save_model(MODEL_SAVE_PATH)

            print(f"Saving calibrator to {CALIBRATOR_SAVE_PATH}...")
            import pickle

            with open(CALIBRATOR_SAVE_PATH, "wb") as f:
                pickle.dump(calibrator, f)

            print(f"Saving best threshold to {THRESHOLD_SAVE_PATH}...")
            with open(THRESHOLD_SAVE_PATH, "w") as f:
                json.dump(
                    {
                        "threshold": best_threshold,
                        "oof_f1": best_f1,
                        "oof_brier": brier,
                        "oof_expected_pnl_per_trade": thr_info["expected_pnl_at_best"],
                        "oof_win_rate": thr_info["win_rate_at_best"],
                        "oof_trade_rate": thr_info["trade_rate_at_best"],
                        "objective": thr_info["objective"],
                        "tp_pct": thr_info["tp_pct"],
                        "sl_pct": thr_info["sl_pct"],
                        "fee_pct": thr_info["fee_pct"],
                        "grid_min": thr_info["grid_min"],
                        "grid_max": thr_info["grid_max"],
                        "boundary_warning": thr_info["boundary_warning"],
                        "fold_best_iter_mean": thr_info.get("fold_best_iter_mean"),
                        "fold_best_iter_std": thr_info.get("fold_best_iter_std"),
                        "fold_best_iters": thr_info.get("fold_best_iters"),
                        "fold_unstable": thr_info.get("fold_unstable"),
                        "n_splits": args.n_splits,
                        "embargo_pct": EMBARGO_FRACTION,
                        "method": "purged_kfold_with_uniqueness_weights",
                        "note": (
                            "Threshold maximises expected PnL per trade on "
                            "calibrated OOF probabilities. boundary_warning=True "
                            "means the optimum is at the search-grid edge -- "
                            "do NOT trust this threshold for live trading."
                        ),
                    },
                    f,
                    indent=2,
                )
            print("Done! Purged k-fold artifacts saved.")
            return

        # ── Legacy 70/15/15 split path (kept for back-compat) ───────────────
        # 2. Time-Series Split
        X_train, y_train, X_val, y_val, X_test, y_test = time_series_split(X, y)

        # 3. Handle Class Imbalance
        print("\nCalculating class imbalance ratio for scale_pos_weight...")
        count_0 = sum(y_train == 0)
        count_1 = sum(y_train == 1)

        if count_1 == 0:
            raise ValueError(
                "Training set has no positive (1) samples. Cannot train model. Try adjusting the threshold or obtaining more data."
            )

        scale_pos_weight = count_0 / count_1
        print(f"Count of class 0 (Loss/Time Expiry): {count_0}")
        print(f"Count of class 1 (Win): {count_1}")
        print(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")

        # 4. Model Initialization
        print("\nInitializing XGBoost Classifier...")
        model = _make_xgb(scale_pos_weight=scale_pos_weight, early_stopping_rounds=50)

        # 5. Model Training with Early Stopping
        print("Training model with early stopping (patience=50)...")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        best_iteration = model.best_iteration
        print(f"Training completed. Best iteration: {best_iteration}")

        # ── 6a. PROBABILITY CALIBRATION ───────────────────────────────────
        print(
            "\nFitting isotonic calibrator on validation set (sklearn CalibratedClassifierCV)..."
        )
        if _HAS_FROZEN:
            # sklearn >= 1.6: cv='prefit' deprecated; wrap in FrozenEstimator instead.
            calibrator = CalibratedClassifierCV(
                FrozenEstimator(model), method="isotonic", cv=2
            )
        else:
            calibrator = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        calibrator.fit(X_val, y_val)

        # ── 6b. THRESHOLD OPTIMISATION ────────────────────────────────────
        val_proba = calibrator.predict_proba(X_val)[:, 1]
        best_threshold, best_score, thr_info = find_best_threshold(
            y_val, val_proba, objective="expected_pnl"
        )
        best_val_f1 = thr_info["f1_at_best"]
        brier = brier_score_loss(y_val, val_proba)
        print(
            f"Best threshold (max expected PnL): {best_threshold:.3f}  "
            f"(EPnL/trade={best_score * 100:.3f}%, win={thr_info['win_rate_at_best']:.3f}, "
            f"trade_rate={thr_info['trade_rate_at_best']:.3f}, F1={best_val_f1:.4f}, "
            f"Brier={brier:.4f})"
        )
        if thr_info["boundary_warning"]:
            print(
                "  [WARN] Optimum threshold pinned at search-grid boundary -- "
                "model has not learned discriminative structure."
            )

        # 7. Evaluation on Test Set with calibrated model + tuned threshold
        print("\nEvaluating CALIBRATED model on the unseen Test Set...")
        test_proba = calibrator.predict_proba(X_test)[:, 1]
        y_test_pred = (test_proba >= best_threshold).astype(int)

        print("\n--- Classification Report (calibrated, tuned-threshold) ---")
        print(classification_report(y_test, y_test_pred))

        print("--- Confusion Matrix ---")
        print(confusion_matrix(y_test, y_test_pred))

        # 8. Save Model + calibrator + threshold
        print(f"\nSaving raw model to {MODEL_SAVE_PATH}...")
        model.save_model(MODEL_SAVE_PATH)

        print(f"Saving calibrator to {CALIBRATOR_SAVE_PATH}...")
        import pickle

        with open(CALIBRATOR_SAVE_PATH, "wb") as f:
            pickle.dump(calibrator, f)

        print(f"Saving best threshold to {THRESHOLD_SAVE_PATH}...")
        with open(THRESHOLD_SAVE_PATH, "w") as f:
            json.dump(
                {
                    "threshold": best_threshold,
                    "val_f1": best_val_f1,
                    "val_brier": brier,
                    "val_expected_pnl_per_trade": thr_info["expected_pnl_at_best"],
                    "val_win_rate": thr_info["win_rate_at_best"],
                    "val_trade_rate": thr_info["trade_rate_at_best"],
                    "objective": thr_info["objective"],
                    "boundary_warning": thr_info["boundary_warning"],
                    "method": "single_split",
                    "note": (
                        "Threshold maximises expected PnL per trade. "
                        "boundary_warning=True means do NOT trust this threshold."
                    ),
                },
                f,
                indent=2,
            )
        print("Done! All artifacts saved.")

    except FileNotFoundError:
        print(
            f"Error: Could not find '{DATA_FILE}'. Please ensure you run apply_fracdiff.py first."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
