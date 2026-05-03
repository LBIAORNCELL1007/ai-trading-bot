import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.calibration import CalibratedClassifierCV
import joblib
import json

"""
Institutional Model Trainer: Universal Blindness XGBoost (v2)

Upgrades over the original:
  - Drops leakage columns (barrier_hit_time, target_return_*, raw close_fd alias).
  - Inserts a 1% embargo gap between train/val and val/test to prevent leakage
    from overlapping triple-barrier label horizons (López de Prado §7.4).
  - Wraps the trained booster in CalibratedClassifierCV(method='isotonic',
    cv='prefit') so downstream probabilities are well-calibrated.
  - Searches the validation set for the F1-optimal decision threshold and
    persists it to a sidecar JSON so api.py can use it instead of a
    hardcoded 0.65 cutoff.
"""

# Columns that MUST be dropped from features to prevent target/future leakage.
# - tbm_label             : the target itself
# - barrier_hit_time      : timestamp at which the barrier was hit (future info)
# - target_return_*       : forward returns used elsewhere as labels
# - close_fd              : legacy duplicate of close_fd_04 (kept under canonical name)
# - barrier_hit_label     : alternative label encoding
LEAKAGE_COLS = {
    "tbm_label",
    "barrier_hit_time",
    "barrier_hit_label",
    "close_fd",
}


def find_best_threshold(y_true, proba, grid=None):
    """Scan thresholds and return the one with maximum F1 on the held-out set."""
    if grid is None:
        grid = np.linspace(0.30, 0.80, 51)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def train_institutional_model():
    # 1. Load Dataset
    print("Loading institutional_alpha_dataset.csv...")
    try:
        df = pd.read_csv("institutional_alpha_dataset.csv")
    except FileNotFoundError:
        print(
            "Error: institutional_alpha_dataset.csv not found. Run the data pipeline first."
        )
        return

    # 2. Data Preparation
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.drop(columns=["timestamp"], inplace=True)

    # Normalize legacy column name
    if "close_fracdiff" in df.columns and "close_fd_04" not in df.columns:
        df.rename(columns={"close_fracdiff": "close_fd_04"}, inplace=True)

    # Drop forward-return target columns (any 'target_return_*' family)
    fwd_return_cols = [c for c in df.columns if c.startswith("target_return")]
    drop_now = [
        c
        for c in (list(LEAKAGE_COLS) + fwd_return_cols)
        if c in df.columns and c != "tbm_label"
    ]
    if drop_now:
        print(f"Dropping leakage columns from features: {drop_now}")

    y = df["tbm_label"].astype(int)
    X = df.drop(columns=["tbm_label"] + drop_now, errors="ignore")

    print(f"Dataset Shape: {X.shape}")
    print(f"Features Detected: {list(X.columns)}")

    # 3. Sequential Split with embargo (70/15/15 with 1% gap on each side)
    n = len(df)
    embargo = max(1, int(n * 0.01))
    train_end = int(n * 0.70)
    val_start = train_end + embargo
    val_end = int(n * 0.85)
    test_start = val_end + embargo

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[val_start:val_end], y.iloc[val_start:val_end]
    X_test, y_test = X.iloc[test_start:], y.iloc[test_start:]

    print(
        f"Train: {len(X_train)} | embargo={embargo} | Val: {len(X_val)} | "
        f"embargo={embargo} | Test: {len(X_test)}"
    )

    # 4. Class Imbalance
    pos_count = int((y_train == 1).sum())
    neg_count = int((y_train == 0).sum())
    spw = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"Calculated scale_pos_weight: {spw:.4f}")

    # 5. Model
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        objective="binary:logistic",
        tree_method="hist",
        scale_pos_weight=spw,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric=["logloss", "auc"],
    )

    # 6. Train with early stopping on the embargoed validation slice
    print("Training XGBoost Model...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    # 7. Probability calibration on the validation set (prefit booster)
    print("\nCalibrating probabilities on validation set (isotonic, prefit)...")
    calibrator = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrator.fit(X_val, y_val)

    # 8. Threshold search on validation
    val_proba = calibrator.predict_proba(X_val)[:, 1]
    best_t, best_f1 = find_best_threshold(y_val.values, val_proba)
    print(f"Validation-optimal threshold: {best_t:.3f} (F1={best_f1:.4f})")

    # 9. Evaluation on Test (using calibrated probs + tuned threshold)
    print("\n" + "=" * 40)
    print("EVALUATION ON TEST SET (15% UNSEEN, calibrated)")
    test_proba = calibrator.predict_proba(X_test)[:, 1]
    y_pred = (test_proba >= best_t).astype(int)

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    acc = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, zero_division=0)
    print(f"Final Test Accuracy: {acc:.4f} | F1: {f1_test:.4f}")
    print("=" * 40)

    # 10. Feature Importance (from underlying booster)
    print("\n--- Feature Importance ---")
    importances = model.feature_importances_
    for name, imp in sorted(zip(X.columns, importances), key=lambda kv: -kv[1]):
        print(f"{name}: {imp:.4f}")

    # 11. Persist artifacts
    raw_model_name = "institutional_xgboost_model.json"
    model.save_model(raw_model_name)
    print(f"\nRaw booster saved as: {raw_model_name}")

    calibrated_path = "institutional_xgboost_model_calibrated.pkl"
    joblib.dump(calibrator, calibrated_path)
    print(f"Calibrated wrapper saved as: {calibrated_path}")

    threshold_path = "institutional_xgboost_model_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump(
            {
                "threshold": best_t,
                "val_f1": best_f1,
                "test_accuracy": acc,
                "test_f1": f1_test,
                "embargo_rows": embargo,
                "features": list(X.columns),
            },
            f,
            indent=2,
        )
    print(f"Threshold sidecar saved as: {threshold_path}")


if __name__ == "__main__":
    train_institutional_model()
