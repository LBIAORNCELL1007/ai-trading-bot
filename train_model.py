import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import optuna
import json
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATA_FILE = 'btc_training_dataset.csv'
TARGET_COL = 'tbm_label'
DROP_COLS = ['timestamp', 'close_fd', 'barrier_hit_time']
MODEL_SAVE_PATH = 'xgb_model_v1.json'

def load_and_preprocess_data():
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    
    # Filter out the "Hold" class (0)
    print("Filtering out rows where target is 0...")
    df = df[df[TARGET_COL] != 0].copy()
    
    # Drop non-predictive columns
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    print(f"Dropping columns: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)
    
    # Map target (-1, 1) -> (0, 1)
    # Mapping logic: -1 -> 0, 1 -> 1
    label_map = {-1: 0, 1: 1}
    print("Mapping labels from [-1, 1] to [0, 1]")
    df[TARGET_COL] = df[TARGET_COL].map(label_map)
    
    # Ensure there are no NaNs
    df = df.dropna()
    
    # Separate features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    
    # Sequential Split (70% Train, 15% Val, 15% Test)
    print("Splitting data sequentially (70/15/15)...")
    train_idx = int(len(X) * 0.70)
    val_idx = int(len(X) * 0.85)
    
    X_train = X.iloc[:train_idx]
    y_train = y.iloc[:train_idx]
    
    X_val = X.iloc[train_idx:val_idx]
    y_val = y.iloc[train_idx:val_idx]
    
    X_test = X.iloc[val_idx:]
    y_test = y.iloc[val_idx:]
    
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'tree_method': 'hist', # Faster training for tabular
        'n_jobs': -1
    }
    
    model = xgb.XGBClassifier(**param)
    
    # Fit the model
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predict on validation set
    y_val_pred = model.predict(X_val)
    
    # Calculate Binary F1
    binary_f1 = f1_score(y_val, y_val_pred)
    return binary_f1

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()
    
    # Calculate scale_pos_weight for highly imbalanced data
    # scale_pos_weight = count(negative examples) / count(positive examples)
    # negative class = 0 (was -1), positive class = 1
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"\nCalculated scale_pos_weight: {scale_pos_weight:.4f} (Neg: {neg_count}, Pos: {pos_count})")
    
    # Optuna Hyperparameter Tuning
    print("\nStarting Optuna hyperparameter tuning (30 trials)...")
    study = optuna.create_study(direction='maximize', study_name='XGBoost_Binary_Classification')
    
    # To limit search time just in case, limiting trials to 30. The user can adjust if needed.
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight), n_trials=30)
    
    print("\nBest parameters found:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Best Binary F1 on Validation: {study.best_value:.4f}")
    
    # Train the final model with best parameters
    print("\nTraining final model with best parameters...")
    best_params = study.best_params
    best_params.update({
        'objective': 'binary:logistic',
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'tree_method': 'hist',
        'n_jobs': -1
    })
    
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False # Set to True to see final eval_set progress
    )
    
    # Evaluate on Test Set
    print("\nEvaluating on Test Set...")
    y_test_pred = final_model.predict(X_test)
    
    # Map predictions and true labels back to original [-1, 1]
    reverse_label_map = {0: -1, 1: 1}
    y_test_original = y_test.map(reverse_label_map)
    y_test_pred_original = pd.Series(y_test_pred).map(reverse_label_map)
    
    print("\n--- Classification Report (Test Set) ---")
    print(classification_report(y_test_original, y_test_pred_original))
    
    print("\n--- Confusion Matrix (Test Set) ---")
    print(confusion_matrix(y_test_original, y_test_pred_original))
    
    # Save Model
    print(f"\nSaving final model to {MODEL_SAVE_PATH}...")
    final_model.save_model(MODEL_SAVE_PATH)
    print("Done!")

if __name__ == '__main__':
    main()