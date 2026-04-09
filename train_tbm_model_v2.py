import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATA_FILE = 'fracdiff_alpha_dataset.csv'
TARGET_COL = 'tbm_label'
DROP_COLS = ['timestamp', 'target_return_4h']  # tbm_label is handled separately
MODEL_SAVE_PATH = 'tbm_xgboost_model_v2.json'

def load_and_prepare_data():
    """
    Loads the fractionally differenced dataset and prepares it for time-series training.
    """
    print(f"Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    
    # If timestamp is a column (not index), it will be dropped
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    print(f"Dropping non-feature columns: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)
    
    # Separate features (X) and target (y)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    
    print(f"Features used for training: {list(X.columns)}")
    
    # Verify 'close' is gone and 'close_fd_04' is present
    if 'close' in X.columns:
        print("WARNING: 'close' column found in features. Dropping it.")
        X = X.drop(columns=['close'])
    if 'close_fd_04' not in X.columns:
        print("WARNING: 'close_fd_04' column NOT found in features.")
        
    return X, y

def time_series_split(X, y, train_size=0.70, val_size=0.15):
    """
    Splits the data sequentially to prevent look-ahead bias.
    Defaults to 70% Train, 15% Val, 15% Test.
    """
    print(f"\nSplitting data sequentially ({int(train_size*100)}/{int(val_size*100)}/{int((1-train_size-val_size)*100)})...")
    
    n_samples = len(X)
    train_end = int(n_samples * train_size)
    val_end = int(n_samples * (train_size + val_size))
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    try:
        # 1. Load Data
        X, y = load_and_prepare_data()
        
        # 2. Time-Series Split
        X_train, y_train, X_val, y_val, X_test, y_test = time_series_split(X, y)
        
        # 3. Handle Class Imbalance
        print("\nCalculating class imbalance ratio for scale_pos_weight...")
        count_0 = sum(y_train == 0)
        count_1 = sum(y_train == 1)
        
        if count_1 == 0:
            raise ValueError("Training set has no positive (1) samples. Cannot train model. Try adjusting the threshold or obtaining more data.")
            
        scale_pos_weight = count_0 / count_1
        print(f"Count of class 0 (Loss/Time Expiry): {count_0}")
        print(f"Count of class 1 (Win): {count_1}")
        print(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")
        
        # 4. Model Initialization
        print("\nInitializing XGBoost Classifier...")
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist', # Faster for tabular data
            n_jobs=-1,
            early_stopping_rounds=50
        )
        
        # 5. Model Training with Early Stopping
        print("Training model with early stopping (patience=50)...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        best_iteration = model.best_iteration
        print(f"Training completed. Best iteration: {best_iteration}")
        
        # 6. Evaluation on Test Set
        print("\nEvaluating model on the unseen Test Set...")
        y_test_pred = model.predict(X_test)
        
        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_test_pred))
        
        print("--- Confusion Matrix ---")
        print(confusion_matrix(y_test, y_test_pred))
        
        # 7. Save Model
        print(f"\nSaving trained model to {MODEL_SAVE_PATH}...")
        model.save_model(MODEL_SAVE_PATH)
        print("Done! Model saved successfully.")
        
    except FileNotFoundError:
        print(f"Error: Could not find '{DATA_FILE}'. Please ensure you run apply_fracdiff.py first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()