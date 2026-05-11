"""
Institutional-grade Data Ingestion and Engineering Pipeline.

Features:
- Dynamic Universe: Integrates with universe.py (auto top-N by volume).
- Dual-Storage: Raw data (Parquet) separated from Engineered features.
- Quality Assurance: Automated checks for gaps, duplicates, staleness, and outliers.
- Causal Integrity: Forward-fill only, no look-ahead bias in joins.
- Parquet Support: High-performance storage with metadata preservation.
"""

import os
import sys
import time
import json
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

# Local imports
import universe as universe_lib
from tbm_labeler import apply_triple_barrier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Constants
ROOT_DIR = Path(__file__).parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Ensure directories exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

class DataQualityAuditor:
    """Performs rigorous quality checks on market data."""
    
    @staticmethod
    def audit(symbol: str, df: pd.DataFrame, expected_interval: str = "1h") -> Dict[str, Any]:
        report = {
            "symbol": symbol,
            "rows": len(df),
            "start": df.index.min().isoformat() if not df.empty else None,
            "end": df.index.max().isoformat() if not df.empty else None,
            "issues": []
        }
        
        if df.empty:
            report["issues"].append("EMPTY_DATASET")
            return report

        # 1. Duplicate Timestamps
        dupes = df.index.duplicated().sum()
        if dupes > 0:
            report["issues"].append(f"DUPLICATE_TIMESTAMPS: {dupes}")

        # 2. Missing Candles (Gaps)
        if expected_interval == "1h":
            expected_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1h')
            missing = len(expected_range) - len(df)
            if missing > 0:
                report["issues"].append(f"MISSING_CANDLES: {missing} ({missing/len(expected_range):.2%})")

        # 3. Extreme Outliers (Price jumps > 50% in a single bar)
        returns = df['close'].pct_change().abs()
        outliers = (returns > 0.5).sum()
        if outliers > 0:
            report["issues"].append(f"EXTREME_OUTLIERS: {outliers} bars with >50% move")

        # 4. Zero Volume
        zero_vol = (df['volume'] == 0).sum()
        if zero_vol > 0:
            report["issues"].append(f"ZERO_VOLUME_BARS: {zero_vol}")

        # 5. Stale Funding (if present)
        if 'funding_rate' in df.columns:
            # Check if funding hasn't changed in a long time (stale API)
            # 24h of identical funding is common, but 7 days is suspicious
            stale_funding = (df['funding_rate'].rolling(24*7).std() == 0).sum()
            if stale_funding > 0:
                report["issues"].append(f"SUSPECT_STALE_FUNDING: {stale_funding} bars")

        return report

class BinanceDataLoader:
    """Fetches and manages raw data from Binance Futures."""
    
    def __init__(self, session=None):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        if session:
            self.session = session
        else:
            self.session = requests.Session()
            retry = Retry(total=5, backoff_factor=0.3, status_forcelist=(500, 502, 504))
            self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def fetch_ohlcv(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        url = "https://fapi.binance.com/fapi/v1/klines"
        end_time = int(time.time() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        all_klines = []
        current_start = start_time
        
        while current_start < end_time:
            params = {"symbol": symbol, "interval": interval, "startTime": current_start, "endTime": end_time, "limit": 1500}
            try:
                res = self.session.get(url, params=params, timeout=10)
                res.raise_for_status()
                data = res.json()
                if not data: break
                all_klines.extend(data)
                current_start = data[-1][0] + 1
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error fetching OHLCV for {symbol}: {e}")
                break
                
        if not all_klines: return pd.DataFrame()
        
        df = pd.DataFrame(all_klines, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "qav", "trades", "tbb", "tbq", "ignore"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df[["open", "high", "low", "close", "volume"]].sort_index()

    def fetch_funding(self, symbol: str, days: int) -> pd.DataFrame:
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        end_time = int(time.time() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        all_fr = []
        current_start = start_time
        while current_start < end_time:
            params = {"symbol": symbol, "startTime": current_start, "endTime": end_time, "limit": 1000}
            try:
                res = self.session.get(url, params=params, timeout=10)
                res.raise_for_status()
                data = res.json()
                if not data: break
                all_fr.extend(data)
                current_start = data[-1]["fundingTime"] + 1
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error fetching Funding for {symbol}: {e}")
                break
                
        if not all_fr: return pd.DataFrame(columns=["funding_rate"])
        
        df = pd.DataFrame(all_fr)
        df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
        df["funding_rate"] = df["fundingRate"].astype(float)
        return df.set_index("timestamp")[["funding_rate"]].sort_index()

class FeatureEngineer:
    """Computes research-quality features with causal integrity."""
    
    @staticmethod
    def get_weights_ffd(d, thres=1e-4):
        w, k = [1.0], 1
        while True:
            w_ = -w[-1] / k * (d - k + 1)
            if abs(w_) < thres: break
            w.append(w_)
            k += 1
        return np.array(w[::-1]).reshape(-1, 1)

    @classmethod
    def frac_diff_ffd(cls, series, d, thres=1e-4):
        w = cls.get_weights_ffd(d, thres)
        width = len(w) - 1
        df = {}
        for name in series.columns:
            series_f = series[[name]].ffill().dropna()
            df_ = pd.Series(dtype=float, index=series_f.index)
            for iloc1 in range(width, series_f.shape[0]):
                loc0 = series_f.index[iloc1 - width]
                loc1 = series_f.index[iloc1]
                if not np.isfinite(series.loc[loc1, name]): continue
                df_[loc1] = np.dot(w.T, series_f.loc[loc0:loc1, name])[0]
            df[name] = df_.copy(deep=True)
        return pd.concat(df, axis=1)

    @staticmethod
    def hours_to_bars(hours, interval):
        mapping = {"1h": 1, "2h": 0.5, "4h": 0.25, "8h": 0.125}
        bph = mapping.get(interval, 1)
        return max(1, int(round(hours * bph)))

    @classmethod
    def engineer_features(cls, df: pd.DataFrame, interval: str = "1h") -> pd.DataFrame:
        logger.info(f"Engineering features for {len(df)} rows...")
        
        # Windows
        win_24h = cls.hours_to_bars(24, interval)
        win_24h_min = max(2, win_24h // 2)
        win_7d = cls.hours_to_bars(24 * 7, interval)
        win_7d_min = max(2, win_7d // 4)
        period_8h_bars = cls.hours_to_bars(8, interval)

        # Basic momentum/rejection
        df["volume_change_1h"] = df["volume"].pct_change()
        df["buying_rejection"] = df["high"] - df["close"]
        df["selling_rejection"] = df["close"] - df["low"]

        # Volatility & RSI
        log_ret = np.log(df["close"]).diff()
        df["realized_vol_24h"] = log_ret.rolling(win_24h, min_periods=win_24h_min).std()
        
        delta = df["close"].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        df["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)
        df["rsi_14"] = df["rsi_14"].fillna(50.0)

        # Normalised ranges
        df["bar_range_pct"] = (df["high"] - df["low"]) / df["close"]
        df["volume_zscore_24h"] = (df["volume"] - df["volume"].rolling(win_24h).mean()) / df["volume"].rolling(win_24h).std()

        # Causal Funding Features
        if "funding_rate" in df.columns:
            fr = df["funding_rate"]
            df["funding_change_8h"] = fr.diff(period_8h_bars).fillna(0.0)
            df["funding_zscore_7d"] = ((fr - fr.rolling(win_7d).mean()) / fr.rolling(win_7d).std()).fillna(0.0)
            
            # Sign streak
            sign = np.sign(fr).fillna(0.0).astype(int)
            seg_id = (sign != sign.shift(1)).cumsum()
            df["funding_sign_streak"] = (sign.groupby(seg_id).cumcount() + 1).astype(float) * sign.astype(float)

        # Stationary Close (FFD)
        fd_df = cls.frac_diff_ffd(df[["close"]], d=0.4)
        df["close_fd_04"] = fd_df["close"]

        return df

def main():
    parser = argparse.ArgumentParser(description="Build institutional alpha dataset.")
    parser.add_argument("--days", type=int, default=1095, help="History days (default 3y)")
    parser.add_argument("--interval", type=str, default="1h", choices=["1h", "2h", "4h", "8h"])
    parser.add_argument("--output", type=str, default="global_alpha_dataset.parquet")
    parser.add_argument("--tp-pct", type=float, default=0.01)
    parser.add_argument("--sl-pct", type=float, default=-0.01)
    parser.add_argument("--time-limit", type=int, default=24)
    universe_lib.add_universe_args(parser)
    args = parser.parse_args()

    # 1. Resolve Universe
    selected_universe = universe_lib.resolve_universe(args)
    logger.info(f"Target Universe: {selected_universe}")

    loader = BinanceDataLoader()
    auditor = DataQualityAuditor()
    all_reports = []
    all_processed_dfs = []

    for symbol in selected_universe:
        logger.info(f"Processing {symbol}...")
        
        # 2. Fetch Raw Data
        df_ohlcv = loader.fetch_ohlcv(symbol, args.interval, args.days)
        if df_ohlcv.empty:
            logger.warning(f"No OHLCV for {symbol}, skipping.")
            continue
            
        df_funding = loader.fetch_funding(symbol, args.days)
        
        # 3. Join Causally (Forward-fill funding only)
        # We reindex funding to OHLCV timestamps using ffill to avoid look-ahead bias
        df = df_ohlcv.copy()
        if not df_funding.empty:
            df["funding_rate"] = df_funding["funding_rate"].reindex(df.index, method="ffill").fillna(0.0)
        else:
            df["funding_rate"] = 0.0
            
        # 4. Save Raw Data
        raw_path = DATA_RAW_DIR / f"{symbol}_{args.interval}.parquet"
        df.to_parquet(raw_path)
        logger.info(f"Saved raw data to {raw_path}")

        # 5. Audit Quality
        report = auditor.audit(symbol, df, args.interval)
        all_reports.append(report)
        if report["issues"]:
            logger.warning(f"QA Report for {symbol}: {report['issues']}")
        else:
            logger.info(f"QA Report for {symbol}: PASSED")

        # 6. Engineer Features
        df_feat = FeatureEngineer.engineer_features(df, args.interval)
        
        # 7. Label (TBM)
        df_labeled = apply_triple_barrier(df_feat, tp_pct=args.tp_pct, sl_pct=args.sl_pct, time_limit=args.time_limit)
        
        # Cleanup
        df_labeled["symbol"] = symbol
        cols_to_drop = ["open", "high", "low", "close"]
        df_final = df_labeled.drop(columns=[c for c in cols_to_drop if c in df_labeled.columns])
        
        # Sanitize
        df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna()
        all_processed_dfs.append(df_final)
        
        time.sleep(1) # Breathe

    # 8. Aggregate and Save Final Dataset
    if not all_processed_dfs:
        logger.error("No data processed. Aborting.")
        return

    global_df = pd.concat(all_processed_dfs).sort_index(kind="mergesort")
    
    # Save Report
    report_path = ROOT_DIR / "ingestion_report.json"
    with open(report_path, "w") as f:
        json.dump(all_reports, f, indent=4)
    logger.info(f"Ingestion report saved to {report_path}")

    # Save Final Dataset
    output_path = DATA_PROCESSED_DIR / args.output
    if args.output.endswith(".parquet"):
        global_df.to_parquet(output_path)
    else:
        global_df.to_csv(output_path)
    logger.info(f"Global dataset saved to {output_path} ({len(global_df)} rows)")

if __name__ == "__main__":
    main()
