from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import numpy as np
import requests
import time

"""
Institutional FastAPI Backend: Universal Blindness Model
This microservice fetches live Binance Futures data, performs dynamic 
Rolling Z-Score normalization, and executes a sniper-threshold prediction.
"""

app = FastAPI(title="AI Trading Bot Institutional API")

# Configure CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for debugging connection issues
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Setup & Model Loading
model = xgb.XGBClassifier()
MODEL_PATH = "institutional_xgboost_model.json"

try:
    model.load_model(MODEL_PATH)
    print(f"[SUCCESS] Loaded institutional model: {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    # Fallback to older model if institutional is missing
    try:
        model.load_model("global_tbm_xgboost_model.json")
        print("[FALLBACK] Loaded global_tbm_xgboost_model.json")
    except:
        pass

class PredictRequest(BaseModel):
    symbol: str

def get_weights_ffd(d, thres=1e-4):
    w, k = [1.], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres: break
        w.append(w_)
        k += 1
    return np.array(w[::-1])

def apply_frac_diff_live(series, d, thres=1e-4):
    """
    Applies Fractional Differencing to a live series.
    Uses convolution for efficiency and returns the last (current) value.
    """
    w = get_weights_ffd(d, thres)
    width = len(w)
    if len(series) < width:
        # If not enough data, use truncated weights for a rough estimate
        # or return NaN. To ensure production stability, we fetch more rows in fetcher.
        return np.nan
    
    vals = series.values
    output = np.convolve(vals, w, mode='valid')
    return float(output[-1])

# 2. The Live Fetch
def fetch_binance_data(symbol):
    """
    Fetches 500 hours of data with timestamp-based joining to ensure
    all features (OHLCV, OI, Funding) are perfectly aligned and non-NaN.
    """
    try:
        limit = 500
        # 1. OHLCV
        ohlcv_url = "https://fapi.binance.com/fapi/v1/klines"
        res_ohlcv = requests.get(ohlcv_url, params={"symbol": symbol, "interval": "1h", "limit": limit})
        res_ohlcv.raise_for_status()
        df = pd.DataFrame(res_ohlcv.json(), columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'ct', 'qv', 'nt', 'tb', 'tq', 'i'])
        df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        # 2. Open Interest
        oi_url = "https://fapi.binance.com/futures/data/openInterestHist"
        res_oi = requests.get(oi_url, params={"symbol": symbol, "period": "1h", "limit": limit})
        res_oi.raise_for_status()
        df_oi = pd.DataFrame(res_oi.json())
        if not df_oi.empty:
            df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'], unit='ms')
            df_oi.set_index('timestamp', inplace=True)
            df_oi['open_interest'] = df_oi['sumOpenInterest'].astype(float)
            df = df.join(df_oi[['open_interest']], how='left')

        # 3. Funding Rate
        fr_url = "https://fapi.binance.com/fapi/v1/fundingRate"
        res_fr = requests.get(fr_url, params={"symbol": symbol, "limit": limit})
        res_fr.raise_for_status()
        df_fr = pd.DataFrame(res_fr.json())
        if not df_fr.empty:
            df_fr['timestamp'] = pd.to_datetime(df_fr['fundingTime'], unit='ms')
            df_fr.set_index('timestamp', inplace=True)
            df_fr['funding_rate'] = df_fr['fundingRate'].astype(float)
            df = df.join(df_fr[['funding_rate']], how='left')

        # Final Clean: Forward fill gaps (like 8h funding) and zero-fill remaining
        df['open_interest'] = df['open_interest'].ffill().fillna(0)
        df['funding_rate'] = df['funding_rate'].ffill().fillna(0)

        return df
    except Exception as e:
        print(f"Data Fetch Error: {e}")
        return None

@app.post("/api/predict")
async def predict(request: PredictRequest):
    symbol = request.symbol.upper()
    print(f"\n--- Live Prediction Request: {symbol} ---")

    # Fetch data
    df = fetch_binance_data(symbol)
    if df is None or df.empty:
        raise HTTPException(status_code=500, detail="Failed to fetch data from Binance.")

    try:
        # 3. Dynamic Rolling Normalization & Feature Engineering
        # Calculate changes/rejections BEFORE normalization
        df['oi_change_1h'] = df['open_interest'].pct_change()
        df['volume_change_1h'] = df['volume'].pct_change()
        df['buying_rejection'] = df['high'] - df['close']
        df['selling_rejection'] = df['close'] - df['low']
        
        # Stationary Close (FracDiff d=0.4)
        w = get_weights_ffd(d=0.4, thres=1e-4)
        width = len(w)
        vals = df['close'].values
        
        # Safety Check: Ensure data is longer than the weights window
        if len(vals) >= width:
            fd_values = np.convolve(vals, w, mode='valid')
            full_fd_series = np.full(len(df), np.nan)
            full_fd_series[width-1:] = fd_values
            df['close_fd_04'] = full_fd_series
        else:
            print(f"Warning: Data length ({len(vals)}) is less than FracDiff window ({width})")
            df['close_fd_04'] = 0.0 # Fallback

        # 3. Dynamic Rolling Normalization (CRITICAL)
        # Use only the last 30 hours for Z-Score statistics as requested
        window_30_stats = df.iloc[-30:].copy()
        
        for col in ['volume', 'open_interest', 'funding_rate']:
            mean = window_30_stats[col].mean()
            std = window_30_stats[col].std()
            # Normalize the current (last) row using the 30-period rolling stats
            df.at[df.index[-1], col] = (df.iloc[-1][col] - mean) / (std if std != 0 else 1)

        # 4. Model Execution
        # Isolate the final formatted row
        feature_cols = [
            'volume', 'open_interest', 'funding_rate', 'oi_change_1h', 
            'volume_change_1h', 'buying_rejection', 'selling_rejection', 'close_fd_04'
        ]
        
        live_row = df[feature_cols].iloc[-1:].copy()
        
        # Verify no NaNs in the prediction row
        if live_row.isnull().values.any():
            print(f"Warning: NaNs detected in live row: \n{live_row}")
            live_row.fillna(0, inplace=True)

        print(f"Live Normalized Row:\n{live_row.to_dict('records')[0]}")

        # Predict Probabilities
        probs = model.predict_proba(live_row)
        raw_confidence = float(probs[0][1])

        # 5. The Sniper Threshold (0.65)
        final_action = 1 if raw_confidence >= 0.65 else 0

        print(f"Raw Confidence (Win): {raw_confidence:.4f}")
        print(f"Final Action Decision: {final_action}")

        # 6. Return payload
        return {
            "symbol": symbol,
            "action": final_action,
            "confidence": raw_confidence
        }

    except Exception as e:
        print(f"Prediction Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
