import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from datetime import datetime, timedelta
import universe as universe_lib


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


def get_weights_ffd(d, thres=1e-4):
    """
    Computes the weights for the Fixed-Width Window Fractional Differencing (FFD) method.
    """
    w, k = [1.0], 1
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


# Canonical TBM implementation lives in tbm_labeler.py:
#   - symmetric ±1.5% defaults (was 1.5%/-1.0% here, which biased toward losses)
#   - intra-bar resolution via high/low (this version was close-only)
#   - exposes `barrier_hit_time` for sample-uniqueness weighting
# We re-export it under the local name so existing call sites keep working.
from tbm_labeler import apply_triple_barrier  # noqa: E402,F401


def fetch_binance_futures_data(symbol, interval="1h", days=30):
    """
    Fetches OHLCV, Open Interest, and Funding Rate data from Binance Futures API.
    """
    session = get_session()

    end_time = int(time.time() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    print(
        f"  Fetching data for {symbol} from {pd.to_datetime(start_time, unit='ms')} to {pd.to_datetime(end_time, unit='ms')}"
    )

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
            "limit": 1500,
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
            time.sleep(1)  # Sleep to respect rate limits
        except requests.exceptions.RequestException as e:
            print(f"    Error fetching OHLCV: {e}")
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
            "limit": 500,
        }
        try:
            res = session.get(oi_url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
            if not data:
                break

            last_timestamp = data[-1]["timestamp"]
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
        df_oi["timestamp"] = pd.to_datetime(df_oi["timestamp"], unit="ms")
        df_oi.set_index("timestamp", inplace=True)
        df_oi["open_interest"] = df_oi["sumOpenInterest"].astype(float)
        df_oi = df_oi[["open_interest"]]
    else:
        df_oi = pd.DataFrame(columns=["open_interest"])

    # 3. Fetch Funding Rate
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

            last_timestamp = data[-1]["fundingTime"]
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
        df_fr["timestamp"] = pd.to_datetime(df_fr["fundingTime"], unit="ms")
        df_fr = df_fr.sort_values("timestamp")
        df_fr.set_index("timestamp", inplace=True)
        df_fr["funding_rate"] = df_fr["fundingRate"].astype(float)
        df_fr = df_fr[["funding_rate"]]
    else:
        df_fr = pd.DataFrame(columns=["funding_rate"])

    # 4. Merge DataFrames.
    #
    # BUGFIX: Binance funding timestamps drift by tens of milliseconds from
    # the exact 00/08/16 UTC mark (e.g. 00:00:00.021).  A naive
    # `df_ohlcv.join(df_fr)` joins on exact DatetimeIndex equality, so most
    # funding rows do NOT line up with any kline (which are at HH:00:00.000).
    # The result is that 2 of every 3 funding settlements drop out and the
    # subsequent ffill drags a stale value forward for ~24h instead of the
    # true 8h.  Fix: reindex with method="ffill", which finds the most
    # recent funding ts <= each kline ts.  This matches the live bot
    # (live_features.merge_funding) exactly.
    df = df_ohlcv.join(df_oi, how="left")

    if not df_fr.empty:
        df["funding_rate"] = (
            df_fr["funding_rate"].reindex(df.index, method="ffill").fillna(0.0)
        )
    else:
        df["funding_rate"] = 0.0

    df["open_interest"] = (
        df["open_interest"].ffill().fillna(0.0).infer_objects(copy=False)
    )

    return df


# Map Binance kline interval string -> bars per hour.  Used to rescale all
# rolling-window features so that "24h volatility" stays 24h regardless of
# whether we're on 1h or 4h bars.  Also used to scale TBM time_limit and
# funding_change diff period.
_BARS_PER_HOUR = {"1h": 1, "2h": 0.5, "4h": 0.25, "8h": 0.125}


def _hours_to_bars(hours, interval):
    """Number of bars in `hours` hours, rounded to the nearest integer >= 1."""
    bph = _BARS_PER_HOUR.get(interval)
    if bph is None:
        raise ValueError(
            f"Unsupported interval {interval!r} (supported: {list(_BARS_PER_HOUR)})"
        )
    return max(1, int(round(hours * bph)))


def process_symbol(
    symbol,
    days=30,
    tp_pct=0.010,
    sl_pct=-0.010,
    time_limit=24,
    interval="1h",
):
    print(f"\n--- Processing {symbol} ({days}d, interval={interval}) ---")

    # 1. Fetch Market Data
    df = fetch_binance_futures_data(symbol, interval=interval, days=days)
    if df.empty:
        print(f"  No data returned for {symbol}.")
        return None

    # 2. Engineer Features
    #
    # Concern #3 fix: open_interest is dead (Binance only serves ~30d of
    # openInterestHist).  Replaced with 6 features that compute from klines
    # alone -- no extra API, no truncation risk, all causal.
    print("  Calculating engineered features (kline-only, causal)...")

    # Window sizes (in bars) derived from the interval, so feature semantics
    # ("24h vol", "7d funding z-score", ...) stay constant across bar sizes.
    win_24h = _hours_to_bars(24, interval)
    win_24h_min = max(2, win_24h // 2)
    win_7d = _hours_to_bars(24 * 7, interval)
    win_7d_min = max(2, win_7d // 4)
    period_8h_bars = _hours_to_bars(8, interval)
    print(
        f"  Feature windows: 24h={win_24h}b  7d={win_7d}b  "
        f"funding_diff={period_8h_bars}b"
    )

    # --- Carry-overs that proved useful and remain causal ---
    df["volume_change_1h"] = df["volume"].pct_change(fill_method=None)
    df["buying_rejection"] = df["high"] - df["close"]
    df["selling_rejection"] = df["close"] - df["low"]

    # --- (a) Realized volatility (24h rolling std of log-returns) ---
    log_ret = np.log(df["close"]).diff()
    df["realized_vol_24h"] = log_ret.rolling(win_24h, min_periods=win_24h_min).std()

    # --- (b) RSI-14 (Wilder smoothing) ---
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    df["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)
    df["rsi_14"] = df["rsi_14"].fillna(50.0)  # neutral when undefined

    # --- (c) ATR-14 normalised by close (volatility-of-range) ---
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

    # --- (d) Intra-bar range as % of close ---
    df["bar_range_pct"] = (df["high"] - df["low"]) / df["close"]

    # --- (e) Volume z-score over 24h ---
    vol_mean = df["volume"].rolling(win_24h, min_periods=win_24h_min).mean()
    vol_std = (
        df["volume"]
        .rolling(win_24h, min_periods=win_24h_min)
        .std()
        .replace(0.0, np.nan)
    )
    df["volume_zscore_24h"] = (df["volume"] - vol_mean) / vol_std

    # --- (f) Distance from 24h VWAP (mean-reversion signal) ---
    typ = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typ * df["volume"]
    vwap_24h = (
        pv.rolling(win_24h, min_periods=win_24h_min).sum()
        / df["volume"].rolling(win_24h, min_periods=win_24h_min).sum()
    )
    df["close_to_vwap_24h"] = df["close"] / vwap_24h - 1.0

    # --- (g) Funding-rate momentum (NEW: option-a step 1) ---
    #
    # Funding rate is settled every 8h on Binance perps and is forward-filled
    # to every 1h bar by fetch_binance_futures_data.  The *level* of funding
    # already lives in df["funding_rate"]; here we add three momentum
    # derivatives that are causally computed from past funding only:
    #
    #   1. funding_change_8h    First difference at the natural 8-bar period.
    #                           Captures whether funding is accelerating in
    #                           one direction (squeeze precursor).
    #   2. funding_zscore_7d    Standardised level over a 7-day window.  Lets
    #                           the model compare BTC funding (small in abs
    #                           terms) to DOGE funding (often 10x larger)
    #                           on a common scale.
    #   3. funding_sign_streak  Consecutive bars with same-sign funding.
    #                           A long streak of positive funding marks
    #                           crowded longs -- often a reversal setup.
    if "funding_rate" in df.columns:
        fr = df["funding_rate"]

        df["funding_change_8h"] = fr.diff(period_8h_bars).fillna(0.0)

        fr_mean_7d = fr.rolling(win_7d, min_periods=win_7d_min).mean()
        fr_std_7d = (
            fr.rolling(win_7d, min_periods=win_7d_min).std().replace(0.0, np.nan)
        )
        df["funding_zscore_7d"] = ((fr - fr_mean_7d) / fr_std_7d).fillna(0.0)

        # Sign-streak: groupby contiguous-sign segments and count.
        sign = np.sign(fr).fillna(0.0).astype(int)
        # New segment whenever sign changes; cumsum of (sign != prev_sign)
        # gives a unique id per segment, then cumcount within id is the
        # streak length.
        seg_id = (sign != sign.shift(1)).cumsum()
        df["funding_sign_streak"] = (sign.groupby(seg_id).cumcount() + 1).astype(
            float
        ) * sign.astype(float)
    else:
        df["funding_change_8h"] = 0.0
        df["funding_zscore_7d"] = 0.0
        df["funding_sign_streak"] = 0.0

    # 3. Label using the Triple Barrier Method.
    #    Concern #5 fix: tightened to symmetric +/-1% (was +/-1.5%) so the
    #    horizon is shorter -> higher fire-rate -> more *unique* samples per
    #    unit time.  Effective N rises noticeably.
    df = apply_triple_barrier(df, tp_pct=tp_pct, sl_pct=sl_pct, time_limit=time_limit)

    # 4. Apply Fractional Differencing to create a stationary close
    print("  Applying fractional differencing to 'close'...")
    fd_df = frac_diff_ffd(df[["close"]], d=0.4, thres=1e-4)
    df["close_fd_04"] = fd_df["close"]

    # Drop raw OHLC columns as they are non-stationary
    cols_to_drop = ["open", "high", "low", "close"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Drop OI columns -- kept the fetch for backwards compat in the dataframe
    # join, but they are not features any more (Binance only serves ~30d of
    # OI history).  funding_rate IS retained -- the API serves the full
    # history and we now use it + three momentum derivatives.
    for c in ("open_interest", "oi_change_1h"):
        if c in df.columns:
            df = df.drop(columns=[c])

    # 5. Track the Universe Asset
    df["symbol"] = symbol

    # 6. Sanitize all engineered columns: replace inf, then dropna on the
    #    columns the trainer actually consumes.
    feature_cols = [
        "volume",
        "volume_change_1h",
        "buying_rejection",
        "selling_rejection",
        "realized_vol_24h",
        "rsi_14",
        "atr_14_pct",
        "bar_range_pct",
        "volume_zscore_24h",
        "close_to_vwap_24h",
        "close_fd_04",
        "funding_rate",
        "funding_change_8h",
        "funding_zscore_7d",
        "funding_sign_streak",
    ]
    for c in feature_cols:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=["tbm_label", "close_fd_04"] + feature_cols)
    df["tbm_label"] = df["tbm_label"].astype(int)

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the multi-symbol global alpha dataset."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1095,
        help="Days of 1h history to fetch per symbol (default: 1095 = 3y).",
    )
    
    universe_lib.add_universe_args(parser)

    parser.add_argument(
        "--output",
        type=str,
        default="global_alpha_dataset.csv",
        help="Output CSV path (default: global_alpha_dataset.csv).",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        choices=list(_BARS_PER_HOUR.keys()),
        help="Binance kline interval (default: 1h). Rolling-window features auto-rescale.",
    )
    parser.add_argument(
        "--tp-pct",
        type=float,
        default=None,
        help="Triple-barrier take-profit %% (default: 0.010 for 1h, 0.005 for 4h).",
    )
    parser.add_argument(
        "--sl-pct",
        type=float,
        default=None,
        help="Triple-barrier stop-loss %% (default: -0.010 for 1h, -0.005 for 4h).",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help="TBM time limit in bars (default: 24 for 1h, 6 for 4h = 24h hold).",
    )
    args = parser.parse_args()

    # Sensible per-interval defaults for TBM if user didn't override.
    if args.tp_pct is None:
        args.tp_pct = 0.005 if args.interval == "4h" else 0.010
    if args.sl_pct is None:
        args.sl_pct = -0.005 if args.interval == "4h" else -0.010
    if args.time_limit is None:
        args.time_limit = _hours_to_bars(24, args.interval)

    selected_universe = universe_lib.resolve_universe(args)
    print(
        f"Universe: {selected_universe}  |  days={args.days}  |  interval={args.interval}  |  "
        f"TBM=+{args.tp_pct:.4f}/{args.sl_pct:.4f}/{args.time_limit}bars  |  "
        f"output={args.output}"
    )
    all_dfs = []

    for sym in selected_universe:
        df_sym = process_symbol(
            sym,
            days=args.days,
            tp_pct=args.tp_pct,
            sl_pct=args.sl_pct,
            time_limit=args.time_limit,
            interval=args.interval,
        )
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

    # Sort globally by timestamp so purged-k-fold splits are chronologically
    # contiguous across the full multi-symbol dataset (López §7.4 assumes
    # time-ordered rows). Within a single timestamp, ties are broken by
    # symbol order, which is fine because rows from different symbols at the
    # same bar do not share label horizons.
    global_df = global_df.sort_index(kind="mergesort")

    output_file = args.output
    global_df.to_csv(output_file)
    print(f"Global dataset successfully built and saved to {output_file}!")

    print("\n--- Global Dataset Stats ---")
    print(f"Total Rows: {len(global_df)}")
    print("\n--- TBM Label Class Distribution ---")
    print("1 = Win (Hit TP: 1.5%)")
    print("0 = Loss (Hit SL: -1.0% or Time Expiry: 24h)")
    print(global_df["tbm_label"].value_counts())


if __name__ == "__main__":
    main()
