import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, precision_recall_curve, roc_curve, f1_score
)
import json
import os

"""
═══════════════════════════════════════════════════════════════════
  Model Evaluation Report — Institutional XGBoost Trading Model
═══════════════════════════════════════════════════════════════════
Evaluates the trained model on the UNSEEN test set and prints
a comprehensive performance report including:
  • Accuracy, Precision, Recall, F1-Score
  • Confusion Matrix
  • ROC-AUC Score
  • Win Rate at Sniper Threshold (0.65)
  • Feature Importance Ranking
  • Confidence Distribution
"""

def load_model(model_path):
    """Load a saved XGBoost model."""
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print(f"✅ Loaded model: {model_path}")
    return model


def evaluate_institutional_model():
    """Evaluate the institutional model on the dataset it was trained on."""
    
    dataset_path = 'institutional_alpha_dataset.csv'
    model_path = 'institutional_xgboost_model.json'
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        print("   Run: python build_institutional_dataset.py")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Run: python train_institutional_model.py")
        return
    
    # Load model
    model = load_model(model_path)
    
    # Load dataset
    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Clean up
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df.drop(columns=['timestamp'], inplace=True)
    
    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Rename if needed
    if 'close_fracdiff' in df.columns:
        df.rename(columns={'close_fracdiff': 'close_fd_04'}, inplace=True)
    
    # Separate
    y = df['tbm_label'].astype(int)
    X = df.drop(columns=['tbm_label'])
    
    # Sequential split (same as training: 70/15/15)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    
    print(f"\nDataset: {len(df)} total rows")
    print(f"Test Set: {len(X_test)} rows (last 15% — purely unseen)")
    print(f"Features: {list(X.columns)}")
    
    # ──────────────────────────────────────────────
    # PREDICTIONS
    # ──────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # probability of class 1 (buy)
    
    # ══════════════════════════════════════════════
    # REPORT
    # ══════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  📊 MODEL PERFORMANCE REPORT")
    print("═" * 60)
    
    # 1. Overall Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Overall Accuracy:  {acc * 100:.2f}%")
    
    # 2. ROC-AUC
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"  ROC-AUC Score:     {roc_auc:.4f}")
    except:
        roc_auc = None
        print(f"  ROC-AUC Score:     N/A (single class in test set)")
    
    # 3. F1 Score
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"  Weighted F1:       {f1:.4f}")
    
    # 4. Classification Report
    print("\n" + "─" * 60)
    print("  CLASSIFICATION REPORT")
    print("─" * 60)
    print(classification_report(y_test, y_pred, target_names=["Hold (0)", "Buy (1)"]))
    
    # 5. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("─" * 60)
    print("  CONFUSION MATRIX")
    print("─" * 60)
    print(f"                 Predicted Hold  Predicted Buy")
    print(f"  Actual Hold:   {cm[0][0]:>10}     {cm[0][1]:>10}")
    print(f"  Actual Buy:    {cm[1][0]:>10}     {cm[1][1]:>10}")
    
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Positives (correct buys):    {tp}")
    print(f"  True Negatives (correct holds):   {tn}")
    print(f"  False Positives (bad buys):       {fp}")
    print(f"  False Negatives (missed buys):    {fn}")
    
    # 6. Win Rate at Sniper Threshold
    print("\n" + "─" * 60)
    print("  SNIPER THRESHOLD ANALYSIS")
    print("─" * 60)
    
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        sniper_pred = (y_pred_proba >= threshold).astype(int)
        trades_taken = sniper_pred.sum()
        if trades_taken > 0:
            wins = ((sniper_pred == 1) & (y_test.values == 1)).sum()
            win_rate = wins / trades_taken * 100
            coverage = trades_taken / len(y_test) * 100
            print(f"  Threshold {threshold:.2f}:  Win Rate = {win_rate:.1f}%  |  Trades = {trades_taken}/{len(y_test)} ({coverage:.1f}%)")
        else:
            print(f"  Threshold {threshold:.2f}:  No trades taken at this threshold")
    
    print(f"\n  ► Your API uses threshold = 0.65 (Sniper Mode)")
    
    # 7. Confidence Distribution
    print("\n" + "─" * 60)
    print("  CONFIDENCE DISTRIBUTION")
    print("─" * 60)
    
    print(f"  Min confidence:    {y_pred_proba.min():.4f}")
    print(f"  Max confidence:    {y_pred_proba.max():.4f}")
    print(f"  Mean confidence:   {y_pred_proba.mean():.4f}")
    print(f"  Median confidence: {np.median(y_pred_proba):.4f}")
    print(f"  Std deviation:     {y_pred_proba.std():.4f}")
    
    # Histogram buckets
    print(f"\n  Probability Buckets:")
    buckets = [(0, 0.2), (0.2, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
    for low, high in buckets:
        count = ((y_pred_proba >= low) & (y_pred_proba < high)).sum()
        bar = "█" * max(1, int(count / len(y_pred_proba) * 40))
        print(f"    [{low:.1f}-{high:.1f}): {count:>5}  {bar}")
    
    # 8. Feature Importance
    print("\n" + "─" * 60)
    print("  FEATURE IMPORTANCE RANKING")
    print("─" * 60)
    
    importances = model.feature_importances_
    feature_imp = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
    
    max_imp = max(importances) if max(importances) > 0 else 1
    for name, imp in feature_imp:
        bar = "█" * int(imp / max_imp * 30)
        print(f"  {name:<25} {imp:.4f}  {bar}")
    
    # 9. Class Distribution
    print("\n" + "─" * 60)
    print("  CLASS BALANCE")
    print("─" * 60)
    
    total_labels = y.value_counts()
    test_labels = y_test.value_counts()
    
    print(f"  Full Dataset: Class 0 (Hold) = {total_labels.get(0, 0)}, Class 1 (Buy) = {total_labels.get(1, 0)}")
    print(f"  Test Set:     Class 0 (Hold) = {test_labels.get(0, 0)}, Class 1 (Buy) = {test_labels.get(1, 0)}")
    
    ratio = total_labels.get(0, 1) / max(total_labels.get(1, 1), 1)
    print(f"  Imbalance Ratio: {ratio:.2f}:1 (Hold : Buy)")
    
    # 10. Summary Verdict
    print("\n" + "═" * 60)
    print("  📋 SUMMARY")
    print("═" * 60)
    
    grade = "?"
    if acc >= 0.70:
        grade = "A — Strong"
    elif acc >= 0.60:
        grade = "B — Good"
    elif acc >= 0.55:
        grade = "C — Fair"
    elif acc >= 0.50:
        grade = "D — Marginal"
    else:
        grade = "F — Below Random"
    
    print(f"  Accuracy Grade:  {grade}")
    print(f"  Accuracy:        {acc * 100:.2f}%")
    if roc_auc:
        auc_note = "Excellent" if roc_auc >= 0.8 else "Good" if roc_auc >= 0.7 else "Fair" if roc_auc >= 0.6 else "Poor"
        print(f"  AUC:             {roc_auc:.4f} ({auc_note})")
    print(f"  F1 Score:        {f1:.4f}")
    print(f"  Test Set Size:   {len(y_test)} samples")
    print("═" * 60)


if __name__ == "__main__":
    evaluate_institutional_model()
