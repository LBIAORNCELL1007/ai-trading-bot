import pandas as pd
import numpy as np

def apply_triple_barrier(df, tp_pct=0.015, sl_pct=-0.01, time_limit=24):
    """
    Applies the Triple Barrier Method to label data.
    
    Parameters:
    - df: DataFrame containing a 'close' column.
    - tp_pct: Take Profit percentage (Upper Barrier).
    - sl_pct: Stop Loss percentage (Lower Barrier).
    - time_limit: Maximum periods to look ahead (Vertical Barrier).
    
    Returns:
    - Labeled DataFrame.
    """
    print(f"Applying Triple Barrier Method (TP: {tp_pct*100}%, SL: {sl_pct*100}%, Time Limit: {time_limit} periods)...")
    
    # Initialize the label column with NaNs
    labels = np.full(len(df), np.nan)
    closes = df['close'].values
    
    n_rows = len(df)
    
    # Iterate through the dataset up to the point where a full lookahead is possible
    # We drop the last `time_limit` rows to prevent look-ahead bias
    for i in range(n_rows - time_limit):
        current_close = closes[i]
        
        # Look ahead `time_limit` periods
        future_closes = closes[i + 1 : i + 1 + time_limit]
        
        # Calculate cumulative returns from the current close
        returns = (future_closes - current_close) / current_close
        
        hit_label = 0 # Default to 0 (Time Expiry Loss)
        
        # Check step-by-step to see which barrier is hit first
        for ret in returns:
            if ret >= tp_pct:
                hit_label = 1 # Win
                break
            elif ret <= sl_pct:
                hit_label = 0 # Loss
                break
                
        labels[i] = hit_label
        
    df['tbm_label'] = labels
    
    # Drop the rows at the end where full lookahead was impossible
    initial_len = len(df)
    df = df.dropna(subset=['tbm_label']).copy()
    final_len = len(df)
    
    print(f"Dropped {initial_len - final_len} rows at the end to prevent look-ahead bias.")
    
    # Convert label to integer
    df['tbm_label'] = df['tbm_label'].astype(int)
    
    return df

def main():
    input_file = 'alpha_dataset.csv'
    output_file = 'labeled_alpha_dataset.csv'
    
    try:
        # 1. Load the alpha dataset
        print(f"Loading dataset from {input_file}...")
        df = pd.read_csv(input_file, index_col='timestamp', parse_dates=True)
        
        # 2. Apply TBM logic
        df_labeled = apply_triple_barrier(df, tp_pct=0.015, sl_pct=-0.01, time_limit=24)
        
        # 3. Export labeled dataset
        print(f"Saving labeled dataset to {output_file}...")
        df_labeled.to_csv(output_file)
        
        # 4. Print value counts
        print("\n--- TBM Label Class Distribution ---")
        print("1 = Win (Hit TP)")
        print("0 = Loss (Hit SL or Time Expiry)")
        print(df_labeled['tbm_label'].value_counts())
        print(f"Total labeled rows: {len(df_labeled)}")
        
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}. Please ensure you have run build_dataset.py first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()