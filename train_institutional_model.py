import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json

"""
Institutional Model Trainer: Universal Blindness XGBoost
This script trains a classifier on the Institutional Alpha Dataset.
It respects the pre-calculated Rolling Z-Scores and enforces 
strict sequential splitting to prevent look-ahead bias.
"""

def train_institutional_model():
    # 1. Load Dataset
    print("Loading institutional_alpha_dataset.csv...")
    try:
        df = pd.read_csv('institutional_alpha_dataset.csv')
    except FileNotFoundError:
        print("Error: institutional_alpha_dataset.csv not found. Run the data pipeline first.")
        return

    # 2. Data Preparation
    # Ensure chronological order if timestamp exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df.drop(columns=['timestamp'], inplace=True)

    # Separate Features and Label
    # Expected Features: ['volume', 'open_interest', 'funding_rate', 'oi_change_1h', 'volume_change_1h', 'buying_rejection', 'selling_rejection', 'close_fracdiff']
    # Note: The pipeline named it 'close_fracdiff', but we will check for both names to be safe.
    if 'close_fracdiff' in df.columns:
        df.rename(columns={'close_fracdiff': 'close_fd_04'}, inplace=True)

    y = df['tbm_label'].astype(int)
    X = df.drop(columns=['tbm_label'])

    print(f"Dataset Shape: {X.shape}")
    print(f"Features Detected: {list(X.columns)}")

    # 3. Sequential Split (No Shuffling)
    # 70% Train, 15% Val, 15% Test
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    print(f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")

    # 4. Handle Class Imbalance
    # scale_pos_weight = count(negative) / count(positive)
    pos_count = sum(y_train == 1)
    neg_count = sum(y_train == 0)
    spw = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"Calculated scale_pos_weight: {spw:.4f}")

    # 5. Model Initialization
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        objective='binary:logistic',
        tree_method='hist',  # Faster training
        scale_pos_weight=spw,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric=['logloss', 'auc']
    )

    # 6. Training with Early Stopping
    print("Training XGBoost Model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    # 7. Evaluation on Unseen Test Set
    print("\n" + "="*40)
    print("EVALUATION ON TEST SET (15% UNSEEN)")
    y_pred = model.predict(X_test)
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Final Test Accuracy: {acc:.4f}")
    print("="*40)

    # 8. Feature Importance
    print("\n--- Feature Importance ---")
    importances = model.feature_importances_
    for name, imp in zip(X.columns, importances):
        print(f"{name}: {imp:.4f}")

    # 9. Save Model
    model_name = "institutional_xgboost_model.json"
    model.save_model(model_name)
    print(f"\nModel saved as: {model_name}")

if __name__ == "__main__":
    train_institutional_model()
