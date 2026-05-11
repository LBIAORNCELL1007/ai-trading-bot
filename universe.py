"""
Binance Futures Dynamic Universe Module.

Fetches exchangeInfo and 24h ticker data from Binance USDT-M Futures to 
dynamically determine the trading universe based on volume, liquidity, 
and contract status.
"""

import argparse
import logging
import sys
import time
import requests
from typing import List, Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://fapi.binance.com"
EXCHANGE_INFO_URL = f"{BASE_URL}/fapi/v1/exchangeInfo"
TICKER_24H_URL = f"{BASE_URL}/fapi/v1/ticker/24hr"
BOOK_TICKER_URL = f"{BASE_URL}/fapi/v1/ticker/bookTicker"

# Default blacklist for known problematic or low-liquidity assets
DEFAULT_BLACKLIST = []

def fetch_exchange_info() -> List[Dict[str, Any]]:
    """Fetches exchange information from Binance Futures."""
    try:
        response = requests.get(EXCHANGE_INFO_URL, timeout=10)
        response.raise_for_status()
        return response.json().get('symbols', [])
    except Exception as e:
        logger.error(f"Failed to fetch exchangeInfo: {e}")
        return []

def fetch_24h_tickers() -> List[Dict[str, Any]]:
    """Fetches 24h ticker data for all symbols."""
    try:
        response = requests.get(TICKER_24H_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch 24h tickers: {e}")
        return []

def fetch_book_tickers() -> List[Dict[str, Any]]:
    """Fetches book ticker data (bid/ask) for all symbols."""
    try:
        response = requests.get(BOOK_TICKER_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch book tickers: {e}")
        return []

def get_universe(
    mode: str = "auto",
    manual_symbols: Optional[List[str]] = None,
    top_n: int = 50,
    quote_asset: str = "USDT",
    contract_type: str = "PERPETUAL",
    min_quote_volume: float = 0.0,
    max_spread_pct: float = 100.0, # Default: effectively no filter
    blacklist: Optional[List[str]] = None,
    min_age_days: int = 0
) -> List[str]:
    """
    Returns a list of symbols based on the specified criteria.
    
    Args:
        mode: "auto" for dynamic discovery, "manual" for manual list.
        manual_symbols: List of symbols to use if mode is "manual".
        top_n: Number of top symbols by quote volume to return in auto mode.
        quote_asset: Only include symbols with this quote asset.
        contract_type: Only include symbols with this contract type.
        min_quote_volume: Minimum 24h quote volume to include.
        max_spread_pct: Maximum bid/ask spread percentage (e.g., 0.1 for 0.1%).
        blacklist: List of symbols to exclude.
        min_age_days: Minimum days since onboardDate.
    """
    if mode == "manual" and manual_symbols:
        logger.info(f"Using manual universe: {manual_symbols}")
        return [s.upper() for s in manual_symbols]

    logger.info(f"Fetching dynamic universe (mode={mode}, top_n={top_n})...")
    
    symbols_info = fetch_exchange_info()
    if not symbols_info:
        return manual_symbols if manual_symbols else ["BTCUSDT"]

    tickers = fetch_24h_tickers()
    volume_map = {t['symbol']: float(t['quoteVolume']) for t in tickers}
    
    book_tickers = fetch_book_tickers()
    spread_map = {}
    for bt in book_tickers:
        bid = float(bt['bidPrice'])
        ask = float(bt['askPrice'])
        if bid > 0:
            spread_map[bt['symbol']] = (ask - bid) / bid * 100
        else:
            spread_map[bt['symbol']] = 999.0
    
    blacklist = blacklist or DEFAULT_BLACKLIST
    now_ms = int(time.time() * 1000)
    min_age_ms = min_age_days * 24 * 60 * 60 * 1000

    qualified = []
    
    for s in symbols_info:
        symbol = s['symbol']
        
        # 1. Basic Filters
        if s['status'] != 'TRADING':
            logger.info(f"Excluded {symbol}: Status is {s['status']}")
            continue
        if s['quoteAsset'] != quote_asset:
            continue
        if s['contractType'] != contract_type:
            continue
            
        # 2. Blacklist
        if symbol in blacklist:
            logger.info(f"Excluded {symbol}: Blacklisted")
            continue
            
        # 3. Min Age
        onboard_date = s.get('onboardDate', 0)
        if min_age_ms > 0 and (now_ms - onboard_date) < min_age_ms:
            logger.info(f"Excluded {symbol}: Too young (onboarded {onboard_date})")
            continue
            
        # 4. Volume
        volume = volume_map.get(symbol, 0)
        if volume < min_quote_volume:
            logger.info(f"Excluded {symbol}: Low volume ({volume:.2f} < {min_quote_volume})")
            continue
            
        # 5. Spread
        spread = spread_map.get(symbol, 999.0)
        if spread > max_spread_pct:
            logger.info(f"Excluded {symbol}: Wide spread ({spread:.4f}% > {max_spread_pct}%)")
            continue
            
        qualified.append({
            'symbol': symbol,
            'volume': volume
        })

    # Sort by volume descending
    qualified.sort(key=lambda x: x['volume'], reverse=True)
    
    # Take top N
    final_symbols = [q['symbol'] for q in qualified[:top_n]]
    
    logger.info(f"Final Universe ({len(final_symbols)} symbols): {final_symbols}")
    return final_symbols

def add_universe_args(parser: argparse.ArgumentParser):
    """Adds universe-related arguments to an existing parser."""
    group = parser.add_argument_group("Universe Options")
    group.add_argument(
        "--universe",
        type=str,
        default="manual",
        choices=["auto", "manual"],
        help="Universe selection mode (default: manual)."
    )
    group.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top symbols by volume to include in auto mode (default: 50)."
    )
    group.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,DOGEUSDT,MATICUSDT",
        help="Comma-separated list of symbols for manual mode."
    )
    group.add_argument(
        "--min-volume",
        type=float,
        default=0.0,
        help="Minimum 24h quote volume for auto mode."
    )
    group.add_argument(
        "--max-spread",
        type=float,
        default=100.0,
        help="Maximum bid/ask spread percentage for auto mode (default: 100.0)."
    )
    group.add_argument(
        "--min-age",
        type=int,
        default=0,
        help="Minimum age in days for auto mode."
    )

def resolve_universe(args: argparse.Namespace) -> List[str]:
    """Resolves the universe based on parsed arguments."""
    manual_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    return get_universe(
        mode=args.universe,
        manual_symbols=manual_list,
        top_n=args.top_n,
        min_quote_volume=args.min_volume,
        max_spread_pct=args.max_spread,
        min_age_days=args.min_age
    )

if __name__ == "__main__":
    # Standalone usage
    parser = argparse.ArgumentParser(description="Test Binance Universe Module")
    add_universe_args(parser)
    args = parser.parse_args()
    
    symbols = resolve_universe(args)
    print("\n".join(symbols))
