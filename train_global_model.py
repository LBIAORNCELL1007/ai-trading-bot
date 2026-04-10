import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

# Suppress XGBoost warnings for cleaner output
warnings.filterwarnings("ignore")

def load_and_prepare_data(file_path='global_alpha_dataset.csv'):
    """
    Loads the global multi-asset dataset, sorts chronologically, and separates features from targets.
    """
    print(f"Loading dataset from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please run build_global_dataset.py first.")
        return None, None
    
    # 1. Sort chronologically by timestamp
    # This is critical for time-series data to prevent future data from leaking into the past
    print("Sorting data chronologically across all assets...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 2. Define Features (X) and Target (y)
    print("Isolating features and target...")
    
    # We drop 'timestamp', 'tbm_label', and 'symbol'. 
    # Dropping 'symbol' forces the model to learn universal market structures 
    # rather than memorizing specific behaviors of individual coins.
    columns_to_drop = ['timestamp', 'tbm_label', 'symbol']
    # Also drop any unnamed index columns that might have been saved in the CSV
    columns_to_drop.extend([col for col in df.columns if 'Unnamed' in col])
    
    X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    y = df['tbm_label'].astype(int)
    
    print(f"Features used for training ({len(X.columns)}): {list(X.columns)}")
    return X, y

def time_series_split(X, y, train_ratio=0.70, val_ratio=0.15):
    """
    Splits the data sequentially into Train, Validation, and Test sets.
    """
    print("\nSplitting data chronologically (No random shuffling)...")
    n_samples = len(X)
    
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    X_train, y_train = X.iloc[:train_end].copy(), y.iloc[:train_end].copy()
    X_val, y_val = X.iloc[train_end:val_end].copy(), y.iloc[train_end:val_end].copy()
    X_test, y_test = X.iloc[val_end:].copy(), y.iloc[val_end:].copy()
    
    print(f"Train Set: {len(X_train)} rows")
    print(f"Validation Set: {len(X_val)} rows")
    print(f"Test Set: {len(X_test)} rows")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def normalize_features(X_train, X_val, X_test):
    """
    Normalizes specific columns that have vastly different scales across assets.
    The scaler is fitted ONLY on the training set to prevent data leakage.
    """
    print("\nNormalizing absolute scale features to prevent dominance by large-cap assets...")
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Define columns to scale
    cols_to_scale = ['volume', 'open_interest', 'funding_rate']
    
    # Verify columns exist before scaling
    missing_cols = [col for col in cols_to_scale if col not in X_train.columns]
    if missing_cols:
        print(f"Warning: The following columns to scale are missing: {missing_cols}")
        cols_to_scale = [col for col in cols_to_scale if col in X_train.columns]
    
    if cols_to_scale:
        print(f"Scaling columns: {cols_to_scale}")
        
        # Fit ONLY on the training set
        scaler.fit(X_train[cols_to_scale])
        
        # Transform Train, Validation, and Test sets
        X_train.loc[:, cols_to_scale] = scaler.transform(X_train[cols_to_scale])
        X_val.loc[:, cols_to_scale] = scaler.transform(X_val[cols_to_scale])
        X_test.loc[:, cols_to_scale] = scaler.transform(X_test[cols_to_scale])
        
        # Save the scaler for deployment
        scaler_file = 'global_scaler.pkl'
        joblib.dump(scaler, scaler_file)
        print(f"Saved fitted StandardScaler to {scaler_file}")
    else:
        print("No absolute columns found to scale.")
        
    return X_train, X_val, X_test

def calculate_imbalance_ratio(y_train):
    """
    Calculates the ratio of negative class (0) to positive class (1).
    Used for scale_pos_weight in XGBoost to handle imbalanced datasets.
    """
    counts = y_train.value_counts()
    count_0 = counts.get(0, 0)
    count_1 = counts.get(1, 0)
    
    if count_1 == 0:
        print("Warning: No positive class (1) found in training data!")
        return 1.0
        
    ratio = count_0 / count_1
    print(f"\nClass Imbalance Ratio (Class 0 / Class 1): {ratio:.4f}")
    return ratio

def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, scale_pos_weight):
    """
    Initializes, trains, and evaluates the XGBoost Classifier.
    """
    print("\nInitializing Universal XGBoost Classifier...")
    
    # Initialize the model with parameters tuned for financial time series
    model = xgb.XGBClassifier(
        n_estimators=1000,          # Maximum number of trees
        max_depth=4,                # Shallow trees prevent overfitting to noise
        learning_rate=0.01,         # Slow learning rate for better generalization
        subsample=0.8,              # Use 80% of rows per tree
        colsample_bytree=0.8,       # Use 80% of features per tree
        scale_pos_weight=scale_pos_weight, # Handle TBM class imbalance
        random_state=42,
        eval_metric='logloss',      # Evaluation metric
        early_stopping_rounds=50    # Stop if validation doesn't improve for 50 rounds
    )
    
    print("Training model (with early stopping)...")
    
    # Train the model, validating against the validation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100  # Print progress every 100 rounds
    )
    
    print(f"\nTraining completed. Best iteration: {model.best_iteration}")
    
    # ---------------------------------------------------------
    # Evaluation on Unseen Test Data
    # ---------------------------------------------------------
    print("\n--- Evaluating Model on Unseen Test Data ---")
    
    # Generate predictions on the purely unseen test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Calculate custom metrics (Optional but useful for trading)
    win_rate = (y_pred[y_test == 1] == 1).sum() / max(1, sum(y_pred))
    print(f"Precision on Positive Signals (Win Rate of trades taken): {win_rate * 100:.2f}%")
    
    return model

def main():
    # 1. Prepare Data
    X, y = load_and_prepare_data('global_alpha_dataset.csv')
    if X is None:
        return
        
    # 2. Chronological Split
    X_train, y_train, X_val, y_val, X_test, y_test = time_series_split(X, y)
    
    # 3. Normalize specific features to prevent large-cap dominance
    X_train, X_val, X_test = normalize_features(X_train, X_val, X_test)
    
    # 4. Handle Imbalance
    imbalance_ratio = calculate_imbalance_ratio(y_train)
    
    # 5. Train & Evaluate
    model = train_and_evaluate_model(
        X_train, y_train, 
        X_val, y_val, 
        X_test, y_test, 
        scale_pos_weight=imbalance_ratio
    )
    
    # 6. Save the Universal Model
    output_model_file = 'global_tbm_xgboost_model.json'
    print(f"\nSaving the trained Universal Multi-Asset Model to {output_model_file}...")
    model.save_model(output_model_file)
    print("Success! You can now deploy this model and global_scaler.pkl to your FastAPI microservice.")

if __name__ == "__main__":
    main()
