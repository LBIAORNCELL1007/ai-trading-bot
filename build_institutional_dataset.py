import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from datetime import datetime, timedelta

"""
Institutional Data Pipeline: Optimized Per-Asset Rolling Z-Score Normalization
"""

# 1. Setup
symbols = [
    "BTCUSDT",
    "ETHUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "SHIBUSDT",
    "LINKUSDT",
    "MATICUSDT",
    "UNIUSDT",
]
processed_dfs = []


def get_session():
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=0.3, status_forcelist=(500, 502, 504))
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


def fetch_binance_futures_data(symbol, interval="1h", days=30):
    session = get_session()
    end_time = int(time.time() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    # 1. Fetch OHLCV
    print(f"    - Fetching OHLCV data...")
    ohlcv_url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": 1500,
    }
    res = session.get(ohlcv_url, params=params, timeout=10)
    res.raise_for_status()
    data = res.json()

    df_ohlcv = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    if df_ohlcv.empty:
        return pd.DataFrame()
    df_ohlcv["timestamp"] = pd.to_datetime(df_ohlcv["timestamp"], unit="ms")
    df_ohlcv.set_index("timestamp", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df_ohlcv[col] = df_ohlcv[col].astype(float)
    df_ohlcv = df_ohlcv[["open", "high", "low", "close", "volume"]]

    # 2. Fetch Open Interest
    print(f"    - Fetching Open Interest...")
    oi_url = "https://fapi.binance.com/futures/data/openInterestHist"
    params = {
        "symbol": symbol,
        "period": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": 500,
    }
    res = session.get(oi_url, params=params, timeout=10)
    res.raise_for_status()
    df_oi = pd.DataFrame(res.json())
    if not df_oi.empty:
        df_oi["timestamp"] = pd.to_datetime(df_oi["timestamp"], unit="ms")
        df_oi.set_index("timestamp", inplace=True)
        df_oi["open_interest"] = df_oi["sumOpenInterest"].astype(float)
        df_oi = df_oi[["open_interest"]]

    # 3. Fetch Funding Rate
    print(f"    - Fetching Funding Rates...")
    fr_url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {
        "symbol": symbol,
        "startTime": start_time,
        "endTime": end_time,
        "limit": 1000,
    }
    res = session.get(fr_url, params=params, timeout=10)
    res.raise_for_status()
    df_fr = pd.DataFrame(res.json())
    if not df_fr.empty:
        df_fr["timestamp"] = pd.to_datetime(df_fr["fundingTime"], unit="ms")
        df_fr.set_index("timestamp", inplace=True)
        df_fr["funding_rate"] = df_fr["fundingRate"].astype(float)
        df_fr = df_fr[["funding_rate"]]

    # Merge and Clean
    df = df_ohlcv.join(df_oi, how="left").join(df_fr, how="left")
    # ffill ONLY — bfill leaks future values backwards (López de Prado §3).
    df["funding_rate"] = df["funding_rate"].ffill().fillna(0.0)
    df["open_interest"] = df["open_interest"].ffill().fillna(0.0)
    return df


def get_weights_ffd(d, thres=1e-4):
    w, k = [1.0], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1])


def apply_frac_diff_optimized(series, d, thres=1e-4):
    w = get_weights_ffd(d, thres)
    width = len(w)
    if len(series) < width:
        return pd.Series(np.nan, index=series.index)

    # Using numpy convolution for massive speedup
    vals = series.values
    output = np.convolve(vals, w, mode="valid")

    # Re-align with pandas index
    res = pd.Series(np.nan, index=series.index)
    res.iloc[width - 1 :] = output
    return res


# Canonical TBM implementation lives in tbm_labeler.py:
#   - symmetric ±1.5% defaults (was 1.5%/-1.0% here, which biased toward losses)
#   - intra-bar resolution via high/low (this version was close-only)
#   - exposes `barrier_hit_time` for sample-uniqueness weighting
from tbm_labeler import apply_triple_barrier  # noqa: E402,F401


# Main Execution Loop
print("Starting Institutional Data Pipeline...")
for symbol in symbols:
    try:
        print(f"\n[+] Processing {symbol}...")
        df = fetch_binance_futures_data(symbol, days=30)
        if df.empty:
            print(f"    ! Warning: No data for {symbol}")
            continue

        print(f"    - Engineering features...")
        df["oi_change_1h"] = df["open_interest"].pct_change()
        df["volume_change_1h"] = df["volume"].pct_change()
        df["buying_rejection"] = df["high"] - df["close"]
        df["selling_rejection"] = df["close"] - df["low"]

        print(f"    - Applying Rolling Z-Score (Window: 30)...")
        window = 30
        for col in ["volume", "open_interest", "funding_rate"]:
            df[col] = (df[col] - df[col].rolling(window).mean()) / df[col].rolling(
                window
            ).std()

        df = df.iloc[window:].copy()

        print(f"    - Calculating Stationary Close (FracDiff d=0.4)...")
        df["close_fracdiff"] = apply_frac_diff_optimized(df["close"], d=0.4)

        print(f"    - Labeling with Triple Barrier Method...")
        df = apply_triple_barrier(df, tp_pct=0.015, sl_pct=-0.015, time_limit=24)

        df.drop(columns=["open", "high", "low", "close"], inplace=True, errors="ignore")
        df = df.iloc[:-24].copy()
        df.dropna(inplace=True)

        processed_dfs.append(df)
        print(f"    - Done. Processed {len(df)} rows.")

        print(f"    - Waiting 3s for rate limits...")
        time.sleep(3)

    except Exception as e:
        print(f"    ! ERROR processing {symbol}: {e}")
        continue

if processed_dfs:
    final_df = pd.concat(processed_dfs, axis=0)
    final_df.to_csv("institutional_alpha_dataset.csv")
    print("\n" + "=" * 40)
    print("SUCCESS: Institutional Dataset Created")
    print(f"Total Rows: {len(final_df)}")
    print("Distribution:\n", final_df["tbm_label"].value_counts())
    print("=" * 40)
else:
    print("\n[!] Pipeline failed: No data processed.")
