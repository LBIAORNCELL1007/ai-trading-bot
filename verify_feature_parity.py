import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import json
from pathlib import Path
import live_features
from build_global_dataset import BinanceDataLoader, FeatureEngineer

def verify_feature_parity():
    print("🔍 Starting Feature Parity Check...")
    
    # 1. Fetch some real data
    symbol = "BTCUSDT"
    interval = "1h"
    days = 30
    loader = BinanceDataLoader()
    df_ohlcv = loader.fetch_ohlcv(symbol, interval, days)
    df_funding = loader.fetch_funding(symbol, days)
    
    # Join (Training Style)
    df_train_style = df_ohlcv.copy()
    if not df_funding.empty:
        df_train_style["funding_rate"] = df_funding["funding_rate"].reindex(df_train_style.index, method="ffill").fillna(0.0)
    else:
        df_train_style["funding_rate"] = 0.0
        
    # 2. Compute features using training pipeline
    df_train_feats = FeatureEngineer.engineer_features(df_train_style, interval)
    
    # 3. Compute features using live pipeline
    # live_features.py expects klines and funding DFs
    df_live_feats = live_features.compute_features(df_ohlcv, df_funding, interval)
    
    # 4. Compare latest row
    common_cols = [c for c in df_train_feats.columns if c in df_live_feats.columns]
    # Filter to exactly the 15 features the model expects (or the 12 we have now)
    model_features = [
        'volume', 'funding_rate', 'volume_change_1h', 'buying_rejection', 
        'selling_rejection', 'realized_vol_24h', 'rsi_14', 'volume_zscore_24h', 
        'funding_change_8h', 'funding_zscore_7d', 'funding_sign_streak', 'close_fd_04'
    ]
    
    # Use the second to last row to avoid partial bar issues if any
    row_idx = -2
    train_row = df_train_feats[model_features].iloc[row_idx]
    live_row = df_live_feats[model_features].iloc[row_idx]
    
    print("\n--- Feature Comparison (Latest Full Bar) ---")
    diffs = []
    for col in model_features:
        v_train = train_row[col]
        v_live = live_row[col]
        diff = abs(v_train - v_live)
        status = "✅" if diff < 1e-6 else "❌"
        print(f"{status} {col:20}: Train={v_train:10.6f} | Live={v_live:10.6f} | Diff={diff:.8f}")
        if status == "❌":
            diffs.append(col)
            
    if not diffs:
        print("\n✨ FEATURE PARITY VERIFIED: Live and Training features match perfectly.")
        return True
    else:
        print(f"\n⚠️ FEATURE PARITY FAILED: {len(diffs)} columns mismatch.")
        return False

if __name__ == "__main__":
    verify_feature_parity()
