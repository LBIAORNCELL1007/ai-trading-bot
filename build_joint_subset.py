"""
Build joint subset of global_alpha_dataset_1h_2pct.csv x book_features_*.parquet
for (BTC, ETH, SOL) over 2024-03-08..2024-03-21.

Outputs:
  joint_subset_baseline.csv   -- kline+funding features only (same rows)
  joint_subset_enriched.csv   -- kline+funding + book features

Both CSVs have identical row sets so train/val splits are comparable.
"""

import polars as pl
import sys, os

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GLOBAL = r"D:\ai-trading-bot\global_alpha_dataset_1h_2pct.csv"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
START = "2024-03-08 00:00:00"
END = "2024-03-21 23:00:00"

# Load global, filter
print("Loading global dataset (lazy filter)...")
g = (
    pl.scan_csv(GLOBAL)
    .filter(pl.col("symbol").is_in(SYMBOLS))
    .with_columns(pl.col("timestamp").str.to_datetime())
    .filter(
        (pl.col("timestamp") >= pl.lit(START).str.to_datetime())
        & (pl.col("timestamp") <= pl.lit(END).str.to_datetime())
    )
    .collect()
)
print("global rows after filter:", g.height)
print("per-symbol counts:")
print(g.group_by("symbol").len().sort("symbol"))

# Load and concat book features per symbol
book_frames = []
for sym in SYMBOLS:
    bf = pl.read_parquet(rf"D:\ai-trading-bot\book_features_{sym}.parquet")
    bf = bf.with_columns(pl.lit(sym).alias("symbol"))
    book_frames.append(bf)
book = pl.concat(book_frames)
print("book rows total:", book.height)

# Join on (symbol, hour==timestamp)
joined = g.join(
    book, left_on=["symbol", "timestamp"], right_on=["symbol", "hour"], how="inner"
)
print("joined rows:", joined.height)
print("per-symbol after join:")
print(joined.group_by("symbol").len().sort("symbol"))

# Drop rows with no label (barrier didn't resolve - tail of window)
joined = joined.filter(pl.col("tbm_label").is_not_null())
print("rows after dropping null labels:", joined.height)
print("label distribution:")
print(joined.group_by("tbm_label").len().sort("tbm_label"))

# Define feature sets
KLINE_FUND_FEATURES = [
    "volume",
    "funding_rate",
    "volume_change_1h",
    "buying_rejection",
    "selling_rejection",
    "realized_vol_24h",
    "rsi_14",
    "atr_14_pct",
    "bar_range_pct",
    "volume_zscore_24h",
    "close_to_vwap_24h",
    "funding_change_8h",
    "funding_zscore_7d",
    "funding_sign_streak",
]
BOOK_FEATURES = [
    "spread_bps_mean",
    "spread_bps_std",
    "imb_mean",
    "imb_std",
    "mp_tilt_bps_mean",
    "mp_tilt_bps_std",
    "n_updates",
]
META = ["timestamp", "symbol", "tbm_label", "barrier_hit_time", "close_fd_04"]

baseline = joined.select(META + KLINE_FUND_FEATURES).sort(["timestamp", "symbol"])
enriched = joined.select(META + KLINE_FUND_FEATURES + BOOK_FEATURES).sort(
    ["timestamp", "symbol"]
)

baseline.write_csv(r"D:\ai-trading-bot\joint_subset_baseline.csv")
enriched.write_csv(r"D:\ai-trading-bot\joint_subset_enriched.csv")
print(
    "Wrote joint_subset_baseline.csv  rows=", baseline.height, " cols=", baseline.width
)
print(
    "Wrote joint_subset_enriched.csv  rows=", enriched.height, " cols=", enriched.width
)

# null check on book features
null_counts = enriched.select(
    [pl.col(c).is_null().sum().alias(c) for c in BOOK_FEATURES]
)
print("book feature nulls:")
print(null_counts)
