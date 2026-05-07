import pandas as pd
import numpy as np


def apply_triple_barrier(df, tp_pct=0.015, sl_pct=-0.015, time_limit=24):
    """
    Triple Barrier Method (López de Prado §3) — labels each bar by which of
    three barriers is hit first within `time_limit` bars:
        +tp_pct   → label = 1   (win)
        −sl_pct   → label = 0   (loss)
        time-out  → label = 0   (time-expiry treated as no-go)

    Parameters
    ----------
    df : DataFrame
        Must contain 'close', 'high', 'low' columns (intra-bar resolution).
    tp_pct : float
        Take-profit barrier as a fraction (e.g. 0.015 = 1.5% above entry).
    sl_pct : float
        Stop-loss barrier as a NEGATIVE fraction (e.g. -0.015 = 1.5% below).
    time_limit : int
        Maximum number of forward bars to evaluate.

    Returns
    -------
    DataFrame with two new columns:
        tbm_label         ∈ {0, 1}
        barrier_hit_time  ∈ [1 .. time_limit]   bars-to-resolution

    Notes (changes vs. previous implementation)
    -------------------------------------------
    1) **Symmetric thresholds.**  Previous default tp=1.5% / sl=1.0% biased
       labels toward losses (the SL was statistically much closer than TP)
       which inflated false-label rates and gave the model a non-stationary
       win-rate target.  Default is now symmetric ±1.5%.
    2) **Intra-bar resolution.**  Previously only `close` was inspected: a
       bar that hit TP at its high but closed below was mis-labelled as
       "no event".  Now uses bar `high`/`low` so a TP wick is registered as
       a win at the bar where it occurred.
    3) **Conservative ordering.**  When BOTH high≥tp AND low≤sl on the same
       bar (rare: violent days), we treat it as SL-first — the conservative
       assumption used by every realistic backtest engine.
    4) **`barrier_hit_time` exposed.**  Needed by uniqueness-weighting code
       (López de Prado §4.5) so co-occurring labels can be down-weighted.
    """
    print(
        f"Applying Triple Barrier Method (TP: {tp_pct * 100:+.2f}%, "
        f"SL: {sl_pct * 100:+.2f}%, Time Limit: {time_limit} periods)..."
    )

    n = len(df)
    closes = df["close"].values
    highs = df["high"].values if "high" in df.columns else closes
    lows = df["low"].values if "low" in df.columns else closes

    labels = np.full(n, np.nan)
    hit_times = np.full(n, np.nan)

    for i in range(n - time_limit):
        entry = closes[i]
        upper = entry * (1 + tp_pct)
        lower = entry * (1 + sl_pct)  # sl_pct is negative

        hit_label = 0  # default: time-expiry → loss
        hit_time = time_limit
        for k in range(1, time_limit + 1):
            j = i + k
            hi = highs[j]
            lo = lows[j]
            sl_hit = lo <= lower
            tp_hit = hi >= upper
            if sl_hit and tp_hit:
                # Conservative: assume SL first when both touched same bar
                hit_label = 0
                hit_time = k
                break
            elif sl_hit:
                hit_label = 0
                hit_time = k
                break
            elif tp_hit:
                hit_label = 1
                hit_time = k
                break

        labels[i] = hit_label
        hit_times[i] = hit_time

    df = df.copy()
    df["tbm_label"] = labels
    df["barrier_hit_time"] = hit_times

    initial_len = len(df)
    df = df.dropna(subset=["tbm_label"]).copy()
    print(f"Dropped {initial_len - len(df)} tail rows (no full lookahead).")

    df["tbm_label"] = df["tbm_label"].astype(int)
    df["barrier_hit_time"] = df["barrier_hit_time"].astype(int)
    return df


def main():
    input_file = "alpha_dataset.csv"
    output_file = "labeled_alpha_dataset.csv"

    try:
        print(f"Loading dataset from {input_file}...")
        df = pd.read_csv(input_file, index_col="timestamp", parse_dates=True)

        # Default: SYMMETRIC ±1.5% thresholds (was 1.5%/-1.0%, asymmetric).
        df_labeled = apply_triple_barrier(
            df, tp_pct=0.015, sl_pct=-0.015, time_limit=24
        )

        print(f"Saving labeled dataset to {output_file}...")
        df_labeled.to_csv(output_file)

        print("\n--- TBM Label Class Distribution ---")
        print("1 = Win  (hit +1.5% TP first)")
        print("0 = Loss (hit -1.5% SL first OR time-expired)")
        print(df_labeled["tbm_label"].value_counts())
        print(
            f"Mean barrier_hit_time: {df_labeled['barrier_hit_time'].mean():.2f} bars"
        )
        print(f"Total labeled rows: {len(df_labeled)}")

    except FileNotFoundError:
        print(
            f"Error: Could not find {input_file}. Please ensure you have run build_dataset.py first."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
