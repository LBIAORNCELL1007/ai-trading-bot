"""
Build per-symbol hourly L2 (top-of-book) features from Binance Vision
bookTicker dumps.

Pipeline:
  1. For each (symbol, date) pair in the requested window:
       a. If the aggregated parquet row already exists, skip.
       b. If the zip is not in book_ticker_cache/, curl it down.
       c. Open the zip, read the CSV with polars (explicit schema, fast path).
       d. Compute per-snapshot:
              spread_bps    = (ask - bid) / mid * 10000
              imb           = (bid_qty - ask_qty) / (bid_qty + ask_qty)
              mp_tilt_bps   = (microprice - mid) / mid * 10000
       e. Group by 1h bucket, compute mean/std + update count.
       f. Append to per-symbol parquet at book_features_{symbol}.parquet.
       g. Optionally delete the zip.

Run:
  python build_book_features.py --symbols BTCUSDT,ETHUSDT,SOLUSDT \
      --start 2024-01-01 --end 2024-03-31 --keep-zips

Output: book_features_BTCUSDT.parquet, etc.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

ROOT = Path(__file__).parent
CACHE = ROOT / "book_ticker_cache"
CACHE.mkdir(exist_ok=True)

URL = (
    "https://data.binance.vision/data/futures/um/daily/bookTicker/"
    "{sym}/{sym}-bookTicker-{d}.zip"
)

# Explicit schema speeds up polars read_csv ~3x by skipping dtype inference.
SCHEMA = {
    "update_id": pl.Int64,
    "best_bid_price": pl.Float64,
    "best_bid_qty": pl.Float64,
    "best_ask_price": pl.Float64,
    "best_ask_qty": pl.Float64,
    "transaction_time": pl.Int64,
    "event_time": pl.Int64,
}


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def download(url: str, out_path: Path, retries: int = 3) -> bool:
    """Download via curl.exe (much faster than urllib for large files on Win)."""
    if out_path.exists() and out_path.stat().st_size > 1024:
        return True
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            subprocess.run(
                ["curl.exe", "-fsSL", "--max-time", "600", "-o", str(tmp), url],
                check=True,
            )
            tmp.rename(out_path)
            return True
        except subprocess.CalledProcessError:
            if attempt == retries:
                return False
            time.sleep(2 * attempt)
    return False


def aggregate_zip(zip_path: Path) -> pl.DataFrame | None:
    """Read a single bookTicker zip and return the hourly aggregate.

    Streams the CSV via an extracted temp file rather than reading the
    entire 2-3 GB decompressed payload into RAM (which OOMs on 16 GB
    machines once two files are in flight).
    """
    tmp_csv = zip_path.with_suffix(".csv")
    try:
        try:
            with zipfile.ZipFile(zip_path) as z:
                name = z.namelist()[0]
                with z.open(name) as src, open(tmp_csv, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        except (zipfile.BadZipFile, OSError) as e:
            print(f"  [SKIP] bad zip {zip_path.name}: {e}", file=sys.stderr)
            return None

        # scan_csv streams through chunks; the group_by collects only the
        # aggregated 24-row result, so peak memory stays ~150 MB.
        lf = pl.scan_csv(str(tmp_csv), schema=SCHEMA)
        mid_expr = (pl.col("best_ask_price") + pl.col("best_bid_price")) / 2.0
        spread_expr = pl.col("best_ask_price") - pl.col("best_bid_price")
        micro_expr = (
            pl.col("best_ask_price") * pl.col("best_bid_qty")
            + pl.col("best_bid_price") * pl.col("best_ask_qty")
        ) / (pl.col("best_bid_qty") + pl.col("best_ask_qty"))

        agg = (
            lf.with_columns(
                [
                    (spread_expr / mid_expr * 10000).alias("spread_bps"),
                    (
                        (pl.col("best_bid_qty") - pl.col("best_ask_qty"))
                        / (pl.col("best_bid_qty") + pl.col("best_ask_qty"))
                    ).alias("imb"),
                    ((micro_expr - mid_expr) / mid_expr * 10000).alias("mp_tilt_bps"),
                    pl.from_epoch("transaction_time", time_unit="ms")
                    .dt.truncate("1h")
                    .alias("hour"),
                ]
            )
            .group_by("hour")
            .agg(
                [
                    pl.col("spread_bps").mean().alias("spread_bps_mean"),
                    pl.col("spread_bps").std().alias("spread_bps_std"),
                    pl.col("imb").mean().alias("imb_mean"),
                    pl.col("imb").std().alias("imb_std"),
                    pl.col("mp_tilt_bps").mean().alias("mp_tilt_bps_mean"),
                    pl.col("mp_tilt_bps").std().alias("mp_tilt_bps_std"),
                    pl.len().alias("n_updates"),
                ]
            )
            .sort("hour")
            .collect()
        )
        if agg.height == 0:
            return None
        return agg
    finally:
        tmp_csv.unlink(missing_ok=True)


def process_symbol(sym: str, start: date, end: date, keep_zips: bool) -> Path:
    """Download + aggregate every day in [start, end] for one symbol."""
    out_path = ROOT / f"book_features_{sym}.parquet"

    # Resume support: if parquet exists, find the last hour we have so we
    # can skip already-processed days.
    have_dates: set[date] = set()
    if out_path.exists():
        existing = pl.read_parquet(out_path)
        if existing.height > 0:
            have_dates = set(existing["hour"].dt.date().unique().to_list())
            print(f"[{sym}] resume: {existing.height} hourly rows already on disk")

    new_pieces: list[pl.DataFrame] = []
    for d in daterange(start, end):
        if d in have_dates:
            continue
        d_str = d.strftime("%Y-%m-%d")
        url = URL.format(sym=sym, d=d_str)
        zip_path = CACHE / f"{sym}-bookTicker-{d_str}.zip"

        t0 = time.time()
        if not download(url, zip_path):
            print(f"  [{sym} {d_str}] download FAILED, skipping")
            continue
        dl_s = time.time() - t0

        t0 = time.time()
        agg = aggregate_zip(zip_path)
        agg_s = time.time() - t0
        if agg is None:
            continue
        new_pieces.append(agg)
        print(
            f"  [{sym} {d_str}] dl={dl_s:5.1f}s  agg={agg_s:5.1f}s  "
            f"hours={agg.height:>3}  size={zip_path.stat().st_size / 1e6:.0f}MB"
        )
        if not keep_zips:
            zip_path.unlink()

        # Periodically flush to parquet so a crash mid-run doesn't lose work.
        if len(new_pieces) >= 5:
            _flush(out_path, new_pieces)
            new_pieces.clear()

    if new_pieces:
        _flush(out_path, new_pieces)
    return out_path


def _flush(out_path: Path, pieces: list[pl.DataFrame]):
    new = pl.concat(pieces).sort("hour")
    if out_path.exists():
        existing = pl.read_parquet(out_path)
        combined = pl.concat([existing, new]).unique(subset=["hour"]).sort("hour")
    else:
        combined = new
    combined.write_parquet(out_path)
    print(f"  [flush] wrote {out_path.name}  total rows={combined.height}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-03-31")
    ap.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep raw zips after processing (default: delete to save disk).",
    )
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    n_days = (end - start).days + 1
    print(
        f"Building book features for {syms} from {start} to {end}  "
        f"({n_days} days x {len(syms)} symbols = {n_days * len(syms)} files)"
    )
    print(f"Cache dir: {CACHE}  (free space check below)")
    free = shutil.disk_usage(CACHE).free / (1 << 30)
    print(f"  Free disk: {free:.1f} GB")

    for sym in syms:
        out = process_symbol(sym, start, end, args.keep_zips)
        if out.exists():
            df = pl.read_parquet(out)
            print(
                f"[{sym}] DONE  total hourly rows={df.height}  "
                f"window={df['hour'].min()} .. {df['hour'].max()}"
            )


if __name__ == "__main__":
    main()
