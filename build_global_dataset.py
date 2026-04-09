import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from datetime import datetime, timedelta

def get_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=0.3,
        status_forcelist=(500, 502, 504),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

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

def apply_triple_barrier(df, tp_pct=0.015, sl_pct=-0.01, time_limit=24):
    """
    Applies the Triple Barrier Method to label data.
    """
    print(f"  Applying Triple Barrier Method (TP: {tp_pct*100}%, SL: {sl_pct*100}%, Time Limit: {time_limit} periods)...")
    
    labels = np.full(len(df), np.nan)
    closes = df['close'].values
    
    n_rows = len(df)
    
    for i in range(n_rows - time_limit):
        current_close = closes[i]
        
        # Look ahead `time_limit` periods
        future_closes = closes[i + 1 : i + 1 + time_limit]
        
        # Calculate cumulative returns from the current close
        returns = (future_closes - current_close) / current_close
        
        hit_label = 0 # Default to 0 (Time Expiry Loss)
        
        for ret in returns:
            if ret >= tp_pct:
                hit_label = 1 # Win
                break
            elif ret <= sl_pct:
                hit_label = 0 # Loss
                break
                
        labels[i] = hit_label
        
    df['tbm_label'] = labels
    return df

def fetch_binance_futures_data(symbol, interval="1h", days=30):
    """
    Fetches OHLCV, Open Interest, and Funding Rate data from Binance Futures API.
    """
    session = get_session()
    
    end_time = int(time.time() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    print(f"  Fetching data for {symbol} from {pd.to_datetime(start_time, unit='ms')} to {pd.to_datetime(end_time, unit='ms')}")

    # 1. Fetch OHLCV
    ohlcv_url = "https://fapi.binance.com/fapi/v1/klines"
    all_klines = []
    current_start = start_time
    
    while current_start < end_time:
        params = {
            "symbol": symbol, 
            "interval": interval, 
            "startTime": current_start, 
            "endTime": end_time, 
            "limit": 1500
        }
        try:
            res = session.get(ohlcv_url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
            if not data:
                break
            
            last_timestamp = data[-1][0]
            if last_timestamp + 1 <= current_start:
                break
                
            all_klines.extend(data)
            current_start = last_timestamp + 1
            time.sleep(1) # Sleep to respect rate limits
        except requests.exceptions.RequestException as e:
            print(f"    Error fetching OHLCV: {e}")
            break

    df_ohlcv = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    if not df_ohlcv.empty:
        df_ohlcv['timestamp'] = pd.to_datetime(df_ohlcv['timestamp'], unit='ms')
        df_ohlcv.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_ohlcv[col] = df_ohlcv[col].astype(float)
        df_ohlcv = df_ohlcv[['open', 'high', 'low', 'close', 'volume']]
    else:
        return pd.DataFrame()

    # 2. Fetch Open Interest
    oi_url = "https://fapi.binance.com/futures/data/openInterestHist"
    all_oi = []
    current_start = start_time
    
    while current_start < end_time:
        params = {
            "symbol": symbol, 
            "period": interval, 
            "startTime": current_start, 
            "endTime": end_time, 
            "limit": 500
        }
        try:
            res = session.get(oi_url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
            if not data:
                break
            
            last_timestamp = data[-1]['timestamp']
            if last_timestamp + 1 <= current_start:
                break
                
            all_oi.extend(data)
            current_start = last_timestamp + 1
            time.sleep(1)
        except requests.exceptions.RequestException as e:
            print(f"    Error fetching Open Interest: {e}")
            break
            
    df_oi = pd.DataFrame(all_oi)
    if not df_oi.empty:
        df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'], unit='ms')
        df_oi.set_index('timestamp', inplace=True)
        df_oi['open_interest'] = df_oi['sumOpenInterest'].astype(float)
        df_oi = df_oi[['open_interest']]
    else:
        df_oi = pd.DataFrame(columns=['open_interest'])

    # 3. Fetch Funding Rate
    fr_url = "https://fapi.binance.com/fapi/v1/fundingRate"
    all_fr = []
    current_start = start_time
    
    while current_start < end_time:
        params = {
            "symbol": symbol, 
            "startTime": current_start, 
            "endTime": end_time, 
            "limit": 1000
        }
        try:
            res = session.get(fr_url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
            if not data:
                break
            
            last_timestamp = data[-1]['fundingTime']
            if last_timestamp + 1 <= current_start:
                break
                
            all_fr.extend(data)
            current_start = last_timestamp + 1
            time.sleep(1)
        except requests.exceptions.RequestException as e:
            print(f"    Error fetching Funding Rates: {e}")
            break

    df_fr = pd.DataFrame(all_fr)
    if not df_fr.empty:
        df_fr['timestamp'] = pd.to_datetime(df_fr['fundingTime'], unit='ms')
        df_fr.set_index('timestamp', inplace=True)
        df_fr['funding_rate'] = df_fr['fundingRate'].astype(float)
        df_fr = df_fr[['funding_rate']]
    else:
        df_fr = pd.DataFrame(columns=['funding_rate'])

    # 4. Merge DataFrames
    df = df_ohlcv.join(df_oi, how='left')
    df = df.join(df_fr, how='left')
    
    df['funding_rate'] = df['funding_rate'].ffill().bfill().infer_objects(copy=False)
    df['open_interest'] = df['open_interest'].ffill().bfill().infer_objects(copy=False)

    return df

def process_symbol(symbol):
    print(f"\n--- Processing {symbol} ---")
    
    # 1. Fetch Market Data
    df = fetch_binance_futures_data(symbol, interval="1h", days=30)
    if df.empty:
        print(f"  No data returned for {symbol}.")
        return None
        
    # 2. Engineer Features
    print("  Calculating engineered features...")
    df['oi_change_1h'] = df['open_interest'].pct_change(fill_method=None)
    df['volume_change_1h'] = df['volume'].pct_change(fill_method=None)
    df['buying_rejection'] = df['high'] - df['close']
    df['selling_rejection'] = df['close'] - df['low']
    
    # 3. Label using the Triple Barrier Method
    # (Must be done before we drop the raw 'close' column)
    df = apply_triple_barrier(df, tp_pct=0.015, sl_pct=-0.01, time_limit=24)
    
    # 4. Apply Fractional Differencing to create a stationary close
    print("  Applying fractional differencing to 'close'...")
    fd_df = frac_diff_ffd(df[['close']], d=0.4, thres=1e-4)
    df['close_fd_04'] = fd_df['close']
    
    # Drop raw OHLC columns as they are non-stationary
    cols_to_drop = ['open', 'high', 'low', 'close']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # 5. Track the Universe Asset
    df['symbol'] = symbol
    
    # 6. Drop NaNs caused by the fracdiff lookback, pct_change, and TBM vertical barriers
    df = df.dropna(subset=['tbm_label', 'close_fd_04', 'oi_change_1h'])
    df['tbm_label'] = df['tbm_label'].astype(int)
    
    return df

def main():
    universe = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
    all_dfs = []
    
    for sym in universe:
        df_sym = process_symbol(sym)
        if df_sym is not None and not df_sym.empty:
            all_dfs.append(df_sym)
        
        # Give the API a breather before moving to the next symbol
        time.sleep(1.5)
        
    if not all_dfs:
        print("No data processed. Script aborted.")
        return
        
    # 7. Aggregate and Output
    print("\n--- Aggregating Global Dataset ---")
    global_df = pd.concat(all_dfs)
    
    output_file = 'global_alpha_dataset.csv'
    global_df.to_csv(output_file)
    print(f"Global dataset successfully built and saved to {output_file}!")
    
    print("\n--- Global Dataset Stats ---")
    print(f"Total Rows: {len(global_df)}")
    print("\n--- TBM Label Class Distribution ---")
    print("1 = Win (Hit TP: 1.5%)")
    print("0 = Loss (Hit SL: -1.0% or Time Expiry: 24h)")
    print(global_df['tbm_label'].value_counts())
    
if __name__ == "__main__":
    main()