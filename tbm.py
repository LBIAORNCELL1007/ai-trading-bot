import numpy as np
import pandas as pd
from numba import njit

@njit
def _compute_tbm_labels(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atrs: np.ndarray,
    timestamps: np.ndarray,
    upper_mult: float,
    lower_mult: float,
    vertical_barrier: int
):
    n = len(closes)
    labels = np.full(n, np.nan)
    hit_times = np.empty(n, dtype=timestamps.dtype)
    
    for i in range(n):
        if np.isnan(atrs[i]):
            continue
            
        entry_price = closes[i]
        upper_barrier = entry_price + (upper_mult * atrs[i])
        lower_barrier = entry_price - (lower_mult * atrs[i])
        
        # Look forward up to vertical_barrier candles
        end_idx = min(i + vertical_barrier, n)
        
        hit = False
        for j in range(i + 1, end_idx):
            curr_high = highs[j]
            curr_low = lows[j]
            
            # Check if lower barrier is hit first (conservative approach for intra-candle)
            if curr_low <= lower_barrier:
                labels[i] = -1.0
                hit_times[i] = timestamps[j]
                hit = True
                break
                
            # Check if upper barrier is hit
            if curr_high >= upper_barrier:
                labels[i] = 1.0
                hit_times[i] = timestamps[j]
                hit = True
                break
                
        # If no price barrier hit before vertical barrier
        if not hit:
            labels[i] = 0.0
            if i + vertical_barrier < n:
                hit_times[i] = timestamps[i + vertical_barrier]
            else:
                hit_times[i] = timestamps[n - 1]
                
    return labels, hit_times

def triple_barrier_labels(df: pd.DataFrame, 
                          upper_mult: float = 2.0, 
                          lower_mult: float = 1.5, 
                          vertical_barrier: int = 24) -> pd.DataFrame:
    """
    Applies the Triple Barrier Method (TBM) to a DataFrame.
    """
    df = df.copy()
    
    # 1. Calculate ATR (14-period) if not already in dataframe
    # The build_dataset.py script already adds 'atr_14', so we can use it or fallback to recalculating
    if 'atr_14' in df.columns:
        atr_col = df['atr_14']
    else:
        prev_close = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_col = tr.rolling(window=14).mean()
        df['atr_14'] = atr_col
    
    # Convert to numpy arrays for Numba
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    atrs = atr_col.values
    timestamps = df.index.values.astype('datetime64[ns]').astype(np.int64)
    
    # Calculate labels and hit times using Numba
    labels, hit_times = _compute_tbm_labels(
        closes, highs, lows, atrs, timestamps, upper_mult, lower_mult, vertical_barrier
    )
    
    df['tbm_label'] = labels
    # Convert int64 timestamps back to datetime, handling NaT where necessary
    df['barrier_hit_time'] = pd.to_datetime(hit_times)
    
    # Optional: clean up rows where ATR wasn't available
    df.loc[atr_col.isna(), 'barrier_hit_time'] = pd.NaT
    
    return df
