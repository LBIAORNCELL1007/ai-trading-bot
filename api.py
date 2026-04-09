from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import numpy as np
import requests

app = FastAPI(title="AI Trading Bot Model API")

# Configure CORS to allow requests from the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the XGBClassifier model globally when the application starts
model = xgb.XGBClassifier()
try:
    model.load_model("tbm_xgboost_model_v2.json")
    print("Successfully loaded tbm_xgboost_model_v2.json")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'tbm_xgboost_model_v2.json' is in the same directory as this script.")

class PredictRequest(BaseModel):
    symbol: str

def get_weights_ffd(d, thres=1e-4):
    """
    Computes the weights for the Fixed-Width Window Fractional Differencing (FFD) method.
    """
    w, k = [1.], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series, d, thres=1e-4):
    """
    Applies Fixed-Width Window Fractional Differencing.
    """
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    
    df = {}
    for name in series.columns:
        series_f = series[[name]].ffill().dropna()
        df_ = pd.Series(dtype=float, index=series_f.index)
        
        for iloc1 in range(width, series_f.shape[0]):
            loc0 = series_f.index[iloc1 - width]
            loc1 = series_f.index[iloc1]
            
            if not np.isfinite(series.loc[loc1, name]):
                continue
                
            df_[loc1] = np.dot(w.T, series_f.loc[loc0:loc1, name])[0]
            
        df[name] = df_.copy(deep=True)
        
    return pd.concat(df, axis=1)

def fetch_binance_futures_data(symbol, interval="1h", limit=300):
    """
    Fetches the last N hours of OHLCV, Open Interest, and Funding Rate data from Binance Futures.
    Note: limit is set to 300 (instead of 100) to ensure the fracdiff function has enough lookback 
    history (~282 periods) to generate a valid current value without returning NaN.
    """
    # 1. Fetch OHLCV (Klines)
    ohlcv_url = "https://fapi.binance.com/fapi/v1/klines"
    res = requests.get(ohlcv_url, params={"symbol": symbol, "interval": interval, "limit": limit})
    res.raise_for_status()
    
    df_ohlcv = pd.DataFrame(res.json(), columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df_ohlcv['timestamp'] = pd.to_datetime(df_ohlcv['timestamp'], unit='ms')
    df_ohlcv.set_index('timestamp', inplace=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df_ohlcv[col] = df_ohlcv[col].astype(float)
        
    # 2. Fetch Open Interest (OI)
    oi_url = "https://fapi.binance.com/futures/data/openInterestHist"
    res = requests.get(oi_url, params={"symbol": symbol, "period": interval, "limit": limit})
    res.raise_for_status()
    
    df_oi = pd.DataFrame(res.json())
    df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'], unit='ms')
    df_oi.set_index('timestamp', inplace=True)
    df_oi['open_interest'] = df_oi['sumOpenInterest'].astype(float)
    
    # 3. Fetch Funding Rate History
    fr_url = "https://fapi.binance.com/fapi/v1/fundingRate"
    res = requests.get(fr_url, params={"symbol": symbol, "limit": limit})
    res.raise_for_status()
    
    df_fr = pd.DataFrame(res.json())
    df_fr['timestamp'] = pd.to_datetime(df_fr['fundingTime'], unit='ms')
    df_fr.set_index('timestamp', inplace=True)
    df_fr['funding_rate'] = df_fr['fundingRate'].astype(float)
    
    # 4. Merge DataFrames on Timestamp
    df = df_ohlcv[['open', 'high', 'low', 'close', 'volume']].join(df_oi[['open_interest']], how='left')
    df = df.join(df_fr[['funding_rate']], how='left')
    
    # Forward-fill and back-fill missing funding rates and open interest
    df['funding_rate'] = df['funding_rate'].ffill().bfill().infer_objects(copy=False)
    df['open_interest'] = df['open_interest'].ffill().bfill().infer_objects(copy=False)
    
    return df

def engineer_live_features(df):
    """
    Engineers the exact features required by the XGBoost model.
    """
    # Percentage changes
    df['oi_change_1h'] = df['open_interest'].pct_change(fill_method=None)
    df['volume_change_1h'] = df['volume'].pct_change(fill_method=None)
    
    # Order Flow Toxicity Proxies
    df['buying_rejection'] = df['high'] - df['close']
    df['selling_rejection'] = df['close'] - df['low']
    
    # Fractional Differencing (d=0.4)
    fd_df = frac_diff_ffd(df[['close']], d=0.4, thres=1e-4)
    df['close_fd_04'] = fd_df['close']
    
    return df

@app.post("/api/predict")
async def predict(request: PredictRequest):
    try:
        # 1. Fetch live data
        print(f"\n--- [LIVE FETCH] Initiating request to Binance for {request.symbol} ---")
        # We fetch 300 hours to ensure fracdiff has enough lookback data to calculate the final row.
        df = fetch_binance_futures_data(symbol=request.symbol, interval="1h", limit=300)
        
        # 2. Engineer features in real-time
        df = engineer_live_features(df)
        
        # 3. Drop rows with NaNs (specifically caused by fracdiff and pct_change lookback)
        df = df.dropna()
        
        if df.empty:
            raise HTTPException(status_code=500, detail="Not enough data after feature engineering to make a prediction.")
            
        print(f"[LIVE DATA] Current Market State:\n{df[['close', 'volume', 'open_interest', 'funding_rate']].tail(1)}\n")

        # 4. Isolate the very last row (the current, live hour)
        live_row = df.iloc[[-1]]
        
        # Ensure exact column match with training data
        feature_columns = [
            'volume', 'open_interest', 'funding_rate', 'oi_change_1h', 
            'volume_change_1h', 'buying_rejection', 'selling_rejection', 'close_fd_04'
        ]
        
        live_features = live_row[feature_columns]
        
        # 5. Generate prediction
        prediction = model.predict(live_features)
        probability = model.predict_proba(live_features)
        
        # Cast to native Python types
        final_action = int(prediction[0])
        final_confidence = float(probability[0][1])

        print(f"[AI PREDICTION] Action: {final_action}, Confidence: {final_confidence}\n----------------------------------------------------")
        
        return {
            "symbol": request.symbol,
            "action": final_action,
            "confidence": final_confidence
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Binance API Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch live data from Binance Futures API.")
    except Exception as e:
        print(f"Internal Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
