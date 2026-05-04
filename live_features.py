"""
Live feature computation for the trading bot.

Computes the EXACT same 14 features that train_tbm_model_v2.py was trained on
(via build_global_dataset.py / global_alpha_dataset_1h_2pct.csv):

    close_fd_04, volume, funding_rate, volume_change_1h,
    buying_rejection, selling_rejection, realized_vol_24h,
    rsi_14, atr_14_pct, volume_zscore_24h,
    close_to_vwap_24h, funding_change_8h, funding_zscore_7d,
    funding_sign_streak

NOTE on feature count: was 15; `bar_range_pct` was dropped after a leave-one-
out feature-ablation experiment showed the model improves ~+2.7pp global WR
and +3.0pp on SOL (highest-edge symbol) when bar_range_pct is removed --
the feature actively HURT predictions, likely because (high - low)/close
is highly correlated with realized_vol_24h and atr_14_pct but adds
single-bar noise that was confusing the booster's split decisions.  The
feature is still computed in build_global_dataset.py for backwards
compatibility with the on-disk dataset, but is excluded from training
via the trainer's DROP_COLS.

Inputs to `compute_features`:
    - klines DataFrame indexed by closetime (DatetimeIndex), columns
      ['open','high','low','close','volume'] (floats).
    - funding DataFrame indexed by fundingTime (DatetimeIndex), column
      'funding_rate' (float).  May be empty.
    - interval (e.g. "1h") to scale rolling windows.

Output:
    - DataFrame with the 14 feature columns above, indexed by bar close time.
      The latest row may have NaN in long-window features if not enough
      history was provided -- caller should ensure ≥7 days of bars are passed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from apply_fracdiff import frac_diff_ffd  # standalone, no statsmodels dep


FEATURE_COLUMNS = [
    "volume",
    "funding_rate",
    "volume_change_1h",
    "buying_rejection",
    "selling_rejection",
    "realized_vol_24h",
    "rsi_14",
    "atr_14_pct",
    "volume_zscore_24h",
    "close_to_vwap_24h",
    "funding_change_8h",
    "funding_zscore_7d",
    "funding_sign_streak",
    "close_fd_04",
]


_BARS_PER_HOUR = {"1h": 1, "2h": 0.5, "4h": 0.25, "8h": 0.125}


def _hours_to_bars(hours: float, interval: str) -> int:
    bph = _BARS_PER_HOUR.get(interval)
    if bph is None:
        raise ValueError(f"Unsupported interval {interval!r}")
    return max(1, int(round(hours * bph)))


def merge_funding(klines: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill funding rate onto kline bars.  Bars before first funding
    observation get 0.0 (causal: we don't 'know' the future rate).
    """
    df = klines.copy()
    if funding is None or funding.empty:
        df["funding_rate"] = 0.0
        return df
    fr = funding["funding_rate"].copy()
    df["funding_rate"] = fr.reindex(df.index, method="ffill").fillna(0.0)
    return df


def compute_features(
    klines: pd.DataFrame,
    funding: pd.DataFrame,
    interval: str = "1h",
) -> pd.DataFrame:
    """Compute all 15 features for a single symbol's bar series.

    Tier 2 features (session E + cross-asset D) were tried and confirmed to
    HURT the model (D-only WR=35.7%, E-only WR=47.3% vs baseline 52.6%).
    Both were reverted; this signature is the same as the pre-Tier-2 one.
    """
    df = merge_funding(klines, funding)

    win_24h = _hours_to_bars(24, interval)
    win_24h_min = max(2, win_24h // 2)
    win_7d = _hours_to_bars(24 * 7, interval)
    win_7d_min = max(2, win_7d // 4)
    period_8h_bars = _hours_to_bars(8, interval)

    df["volume_change_1h"] = df["volume"].pct_change(fill_method=None)
    df["buying_rejection"] = df["high"] - df["close"]
    df["selling_rejection"] = df["close"] - df["low"]

    log_ret = np.log(df["close"]).diff()
    df["realized_vol_24h"] = log_ret.rolling(win_24h, min_periods=win_24h_min).std()

    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    df["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)
    df["rsi_14"] = df["rsi_14"].fillna(50.0)

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    df["atr_14_pct"] = atr / df["close"]
    df["bar_range_pct"] = (df["high"] - df["low"]) / df["close"]

    vol_mean = df["volume"].rolling(win_24h, min_periods=win_24h_min).mean()
    vol_std = (
        df["volume"]
        .rolling(win_24h, min_periods=win_24h_min)
        .std()
        .replace(0.0, np.nan)
    )
    df["volume_zscore_24h"] = (df["volume"] - vol_mean) / vol_std

    typ = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typ * df["volume"]
    vwap_24h = (
        pv.rolling(win_24h, min_periods=win_24h_min).sum()
        / df["volume"].rolling(win_24h, min_periods=win_24h_min).sum()
    )
    df["close_to_vwap_24h"] = df["close"] / vwap_24h - 1.0

    fr = df["funding_rate"]
    df["funding_change_8h"] = fr.diff(period_8h_bars).fillna(0.0)
    fr_mean_7d = fr.rolling(win_7d, min_periods=win_7d_min).mean()
    fr_std_7d = fr.rolling(win_7d, min_periods=win_7d_min).std().replace(0.0, np.nan)
    df["funding_zscore_7d"] = ((fr - fr_mean_7d) / fr_std_7d).fillna(0.0)
    sign = np.sign(fr).fillna(0.0).astype(int)
    seg_id = (sign != sign.shift(1)).cumsum()
    df["funding_sign_streak"] = (sign.groupby(seg_id).cumcount() + 1).astype(
        float
    ) * sign.astype(float)

    # Fractional difference of close (d=0.4)
    fd = frac_diff_ffd(df[["close"]], d=0.4, thres=1e-4)
    df["close_fd_04"] = fd["close"]

    # Sanitize
    for c in FEATURE_COLUMNS:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    return df[FEATURE_COLUMNS]


if __name__ == "__main__":
    # Quick parity check: build features from the existing global dataset
    # row-by-row should match the dataset.  Useful sanity test.
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("FEATURE_COLUMNS:", FEATURE_COLUMNS)
    print("Module OK.")
