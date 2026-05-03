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
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_binance_futures_data(symbol="BTCUSDT", interval="1h", days=30):
    """
    Fetches OHLCV, Open Interest, and Funding Rate data from Binance Futures API.
    Handles pagination and rate limits automatically.
    """
    session = get_session()

    # Define start and end times in milliseconds
    end_time = int(time.time() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    print(
        f"Fetching data for {symbol} from {pd.to_datetime(start_time, unit='ms')} to {pd.to_datetime(end_time, unit='ms')}"
    )

    # 1. Fetch OHLCV (Klines)
    print("Fetching OHLCV data...")
    ohlcv_url = "https://fapi.binance.com/fapi/v1/klines"
    all_klines = []
    current_start = start_time

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1500,
        }
        try:
            res = session.get(ohlcv_url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
            if not data:
                break

            # Check if time advanced to prevent infinite loop
            last_timestamp = data[-1][0]
            if last_timestamp + 1 <= current_start:
                break

            all_klines.extend(data)
            current_start = (
                last_timestamp + 1
            )  # Next start time is last candle's open time + 1ms
            time.sleep(0.1)  # Respect rate limits
        except requests.exceptions.RequestException as e:
            print(f"Error fetching OHLCV: {e}")
            break

    df_ohlcv = pd.DataFrame(
        all_klines,
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

    if not df_ohlcv.empty:
        df_ohlcv["timestamp"] = pd.to_datetime(df_ohlcv["timestamp"], unit="ms")
        df_ohlcv.set_index("timestamp", inplace=True)

        for col in ["open", "high", "low", "close", "volume"]:
            df_ohlcv[col] = df_ohlcv[col].astype(float)

        df_ohlcv = df_ohlcv[["open", "high", "low", "close", "volume"]]
    else:
        df_ohlcv = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # 2. Fetch Open Interest (OI) History
    print("Fetching Open Interest data...")
    oi_url = "https://fapi.binance.com/futures/data/openInterestHist"
    all_oi = []
    current_start = start_time

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "period": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 500,
        }
        try:
            res = session.get(oi_url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
            if not data:
                break

            # Check if time advanced to prevent infinite loop
            last_timestamp = data[-1]["timestamp"]
            if last_timestamp + 1 <= current_start:
                break

            all_oi.extend(data)
            current_start = last_timestamp + 1
            time.sleep(0.1)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Open Interest: {e}")
            break

    df_oi = pd.DataFrame(all_oi)
    if not df_oi.empty:
        df_oi["timestamp"] = pd.to_datetime(df_oi["timestamp"], unit="ms")
        df_oi.set_index("timestamp", inplace=True)
        df_oi["open_interest"] = df_oi["sumOpenInterest"].astype(float)
        df_oi = df_oi[["open_interest"]]
    else:
        df_oi = pd.DataFrame(columns=["open_interest"])

    # 3. Fetch Funding Rate History
    print("Fetching Funding Rate data...")
    fr_url = "https://fapi.binance.com/fapi/v1/fundingRate"
    all_fr = []
    current_start = start_time

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000,
        }
        try:
            res = session.get(fr_url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
            if not data:
                break

            # Check if time advanced to prevent infinite loop
            last_timestamp = data[-1]["fundingTime"]
            if last_timestamp + 1 <= current_start:
                break

            all_fr.extend(data)
            current_start = last_timestamp + 1
            time.sleep(0.1)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Funding Rates: {e}")
            break

    df_fr = pd.DataFrame(all_fr)
    if not df_fr.empty:
        df_fr["timestamp"] = pd.to_datetime(df_fr["fundingTime"], unit="ms")
        df_fr.set_index("timestamp", inplace=True)
        df_fr["funding_rate"] = df_fr["fundingRate"].astype(float)
        df_fr = df_fr[["funding_rate"]]
    else:
        df_fr = pd.DataFrame(columns=["funding_rate"])

    # 4. Merge DataFrames
    print("Merging datasets on timestamps...")
    # Left join Open Interest to the primary OHLCV data
    df = df_ohlcv.join(df_oi, how="left")

    # Left join funding rates and handle missing values
    df = df.join(df_fr, how="left")

    # ffill ONLY — bfill leaks future values backwards (López de Prado §3).
    # Pre-first-observation NaNs become 0.0 (neutral funding/OI prior).
    df["funding_rate"] = (
        df["funding_rate"].ffill().fillna(0.0).infer_objects(copy=False)
    )
    df["open_interest"] = (
        df["open_interest"].ffill().fillna(0.0).infer_objects(copy=False)
    )

    return df


def engineer_features(df):
    """
    Engineers custom market microstructure and leading indicator features.
    """
    print("Calculating engineered features...")

    # Percentage change of Open Interest over the last 1h period
    # Using fill_method=None to avoid pandas deprecation warnings
    df["oi_change_1h"] = df["open_interest"].pct_change(fill_method=None)

    # Percentage change of Volume over the last 1h period
    df["volume_change_1h"] = df["volume"].pct_change(fill_method=None)

    # Order Flow Toxicity Proxies
    # High - Close (Buying Rejection / Upper Wick)
    df["buying_rejection"] = df["high"] - df["close"]

    # Close - Low (Selling Rejection / Lower Wick)
    df["selling_rejection"] = df["close"] - df["low"]

    # Regression Target Variable: Expected Value 4 hours into the future
    # (Close_{t+4} - Close_t) / Close_t
    df["target_return_4h"] = (df["close"].shift(-4) - df["close"]) / df["close"]

    return df


def main():
    try:
        # Step 1: Acquire raw market data (last 30 days)
        df = fetch_binance_futures_data(symbol="BTCUSDT", interval="1h", days=30)

        # Step 2: Feature Engineering
        df = engineer_features(df)

        # Step 3: Data Cleaning
        print("Cleaning data (removing rows with NaNs to prevent look-ahead bias)...")
        initial_len = len(df)

        # Print NaN counts per column to debug any missing data issues
        print("\nNaN count per column BEFORE dropna():")
        print(df.isna().sum())
        print()

        # Drop rows where target or engineered features are NaN
        df = df.dropna()
        final_len = len(df)

        print(f"Dropped {initial_len - final_len} rows containing NaNs.")

        # Step 4: Export to CSV
        output_file = "alpha_dataset.csv"
        df.to_csv(output_file)
        print(
            f"Dataset successfully built and saved to {output_file} (Total rows: {final_len})."
        )

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
