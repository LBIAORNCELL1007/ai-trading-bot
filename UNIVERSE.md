# Binance Futures Dynamic Universe

The `universe.py` module provides a production-style dynamic universe selection for Binance USDT-M Futures. It replaces hardcoded symbol lists with a robust discovery system based on real-time market data.

## Features

- **Dynamic Discovery:** Automatically fetches and filters symbols from the Binance Futures `exchangeInfo` and `24h ticker` endpoints.
- **Filtering Criteria:**
  - **Status:** Only includes `TRADING` contracts.
  - **Quote Asset:** Only includes `USDT`-quoted pairs.
  - **Contract Type:** Only includes `PERPETUAL` contracts.
  - **Volume:** Filters by 24h quote volume to ensure liquidity.
  - **Spread:** Filters by bid/ask spread percentage to minimize slippage.
  - **Age:** Filters by `onboardDate` to avoid extremely new and volatile assets.
  - **Blacklist:** Excludes specific symbols if necessary.
- **Modes:**
  - `manual`: Uses a comma-separated list of symbols (backward compatible).
  - `auto`: Dynamically selects the top N symbols by volume that pass all filters.

## Integration

The module is integrated into the following scripts:

- `build_global_dataset.py`
- `run_full_pipeline.py`
- `live_bot.py`

## CLI Options

All integrated scripts now support the following universe options:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--universe` | `str` | `manual` | Universe selection mode (`auto` or `manual`). |
| `--top-n` | `int` | `50` | Number of top symbols by volume in `auto` mode. |
| `--symbols` | `str` | *Majors* | Comma-separated list of symbols for `manual` mode. |
| `--min-volume` | `float` | `0.0` | Minimum 24h quote volume for `auto` mode. |
| `--max-spread` | `float` | `100.0` | Maximum bid/ask spread percentage for `auto` mode. |
| `--min-age` | `int` | `0` | Minimum age in days for `auto` mode. |

## Usage Examples

### Build Global Dataset with Top 20 Pairs

```bash
python build_global_dataset.py --universe auto --top-n 20 --days 365
```

### Run Full Pipeline with Volume and Spread Filters

```bash
python run_full_pipeline.py --universe auto --top-n 30 --min-volume 100000000 --max-spread 0.05
```

### Run Live Bot with a Manual List (Default)

```bash
python live_bot.py --symbols BTCUSDT,ETHUSDT --paper
```

### Run Live Bot with Dynamic Top 10 Pairs

```bash
python live_bot.py --universe auto --top-n 10 --paper
```
