import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def get_weights_ffd(d: float, thres: float = 1e-5) -> np.ndarray:
    """
    Generate the weights for fractional differencing using the fixed-width window method
    described by Marcos Lopez de Prado.
    """
    w: list[float] = [1.0]
    k: int = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])

def apply_frac_diff(series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
    """
    Applies fractional differencing to a pandas Series using a fixed-width window.
    
    Args:
        series (pd.Series): The input time series (e.g., raw close prices).
        d (float): The differencing factor.
        thres (float): The threshold for discarding insignificant weights.
        
    Returns:
        pd.Series: The fractionally differenced series.
    """
    w = get_weights_ffd(d, thres)
    width = len(w)
    
    # Initialize the output series with NaNs to preserve the original index alignment
    diff_series = pd.Series(np.nan, index=series.index)
    series_values = series.values
    
    # Ensure there is enough data for the window
    if len(series) < width:
        return diff_series.dropna()
        
    # Apply the weights to rolling windows
    for i in range(width - 1, len(series_values)):
        window = series_values[i - width + 1 : i + 1]
        
        # Handle potential NaNs in the data properly
        if not np.isnan(window).any():
            diff_series.iloc[i] = np.dot(w, window)
            
    return diff_series.dropna()

def find_optimal_d(series: pd.Series, p_val_threshold: float = 0.05, step: float = 0.05) -> float:
    """
    Finds the lowest fractional differencing value 'd' that makes the series stationary,
    preserving maximum market memory.
    
    Args:
        series (pd.Series): The input time series.
        p_val_threshold (float): The p-value threshold for the ADF test.
        step (float): The increment to test 'd' values.
        
    Returns:
        float: The optimal 'd' value between 0.05 and 1.0.
    """
    optimal_d = 1.0
    
    # Iteratively test values from 0.05 to 1.0 inclusive
    for d in np.arange(step, 1.0 + step, step):
        diff_series = apply_frac_diff(series, d)
        
        # Ensure we have enough data to run the ADF test reliably
        if len(diff_series) < 10:
            continue
            
        # adfuller returns a tuple; index 1 is the p-value
        adf_result = adfuller(diff_series.dropna())
        p_value = float(adf_result[1])
        
        if p_value < p_val_threshold:
            optimal_d = round(d, 2)
            break
            
    return optimal_d

if __name__ == "__main__":
    # ---------------------------------------------------------
    # TEST BLOCK
    # Demonstrates how to pass a pandas DataFrame into the functions
    # ---------------------------------------------------------
    
    print("Generating simulated OHLCV financial data...")
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate a random walk for prices (non-stationary by nature)
    prices = np.cumsum(np.random.randn(n_samples)) + 50000
    
    # Create a DataFrame typical of Cryptocurrency OHLCV data
    dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="D")
    df = pd.DataFrame({
        "open": prices + np.random.randn(n_samples) * 10,
        "high": prices + np.random.randn(n_samples) * 20 + 20,
        "low": prices - np.random.randn(n_samples) * 20 - 20,
        "close": prices,
        "volume": np.random.randint(1000, 10000, n_samples)
    }, index=dates)
    
    print("DataFrame Head:")
    print(df.head())
    
    # Extract the 'close' series for differencing
    close_series: pd.Series = df["close"]
    
    print("\n1. Finding Optimal 'd' (this may take a moment)...")
    optimal_d = find_optimal_d(close_series, step=0.05)
    print(f"-> The optimal 'd' to achieve stationarity (p < 0.05) is: {optimal_d}")
    
    print(f"\n2. Applying Fractional Differencing with d={optimal_d}...")
    fractional_close = apply_frac_diff(close_series, optimal_d)
    
    print("\nResults:")
    print(f"Original Series Length: {len(close_series)}")
    print(f"Differenced Series Length: {len(fractional_close)}")
    print(f"Data loss due to fixed-width window: {len(close_series) - len(fractional_close)} periods")
    
    # Verify the results with an ADF test
    p_val = adfuller(fractional_close)[1]
    print(f"ADF p-value of the transformed series: {p_val:.5f}")
    
    # Store the result back into the DataFrame (aligns properly by index padding with NaNs)
    df["close_frac_diff"] = fractional_close
    print("\nUpdated DataFrame Tail:")
    print(df.tail())