import pandas as pd
import numpy as np

def get_weights_ffd(d, thres=1e-4):
    """
    Computes the weights for the Fixed-Width Window Fractional Differencing (FFD) method.
    From Marcos Lopez de Prado's "Advances in Financial Machine Learning".
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
        # Use fillna with ffill explicitly to avoid deprecation warnings
        series_f = series[[name]].ffill().dropna()
        df_ = pd.Series(dtype=float, index=series_f.index)
        
        for iloc1 in range(width, series_f.shape[0]):
            loc0 = series_f.index[iloc1 - width]
            loc1 = series_f.index[iloc1]
            
            if not np.isfinite(series.loc[loc1, name]):
                continue
                
            # Dot product of weights and the price window
            df_[loc1] = np.dot(w.T, series_f.loc[loc0:loc1, name])[0]
            
        df[name] = df_.copy(deep=True)
        
    return pd.concat(df, axis=1)

def apply_fractional_differencing(input_file, output_file, d=0.4):
    """
    Applies fractional differencing to the 'close' price column to make it stationary
    while preserving memory (memory from the IEEE research methodology).
    
    Parameters:
    - input_file: Path to the labeled alpha dataset.
    - output_file: Path to save the new fractionally differenced dataset.
    - d: The fractional differencing order (typically between 0 and 1).
    """
    print(f"Loading dataset from {input_file}...")
    try:
        df = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}. Please ensure you have generated it.")
        return
        
    print(f"Applying fractional differencing (d={d}) to the 'close' column natively...")
    
    # 1. Apply Fractional Differencing using custom FFD methodology
    # Extract the close prices into a DataFrame for the FFD function
    close_df = df[['close']]
    
    # Thres=1e-4 is standard. For a small dataset, it generates a lookback window of ~282 periods.
    fd_df = frac_diff_ffd(close_df, d=d, thres=1e-4)
    
    # Assign the differenced data to a new column
    df['close_fd_04'] = fd_df['close']
    
    # 2. Cleanup: Drop original non-stationary columns
    cols_to_drop = ['open', 'high', 'low', 'close']
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    print(f"Dropping non-stationary original price columns: {existing_cols_to_drop}")
    df = df.drop(columns=existing_cols_to_drop)
    
    # 3. Cleanup: Drop rows with NaNs (specifically at the top due to the FFD window lookback)
    initial_len = len(df)
    df = df.dropna()
    final_len = len(df)
    
    if initial_len != final_len:
        print(f"Dropped {initial_len - final_len} rows containing NaNs at the beginning of the dataset.")
    
    # 4. Output: Save the new dataset
    print(f"Saving fractionally differenced dataset to {output_file}...")
    df.to_csv(output_file)
    
    print(f"Done! Created '{output_file}' with {final_len} rows and {len(df.columns)} columns.")

if __name__ == "__main__":
    INPUT_CSV = 'labeled_alpha_dataset.csv'
    OUTPUT_CSV = 'fracdiff_alpha_dataset.csv'
    DIFFERENCING_ORDER = 0.4
    
    apply_fractional_differencing(INPUT_CSV, OUTPUT_CSV, d=DIFFERENCING_ORDER)