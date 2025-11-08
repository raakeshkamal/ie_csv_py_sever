"""
Price fetching and currency conversion module for InvestEngine CSV Server.
Handles yfinance API calls, currency detection, and FX rate conversions.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import concurrent.futures
import logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime

from .database import get_cached_prices, cache_price, get_cached_fx_rates, get_connection

logger = logging.getLogger(__name__)

# FX rate configuration: (ticker, multiply_flag)
# multiply_flag True means multiply by rate, False means divide
FX_CONFIG = {
    "USD": ("GBPUSD=X", False),  # divide by rate (USD prices ÷ USDGBP = GBP)
    "EUR": ("EURGBP=X", True),   # multiply by rate (EUR prices × EURGBP = GBP)
}


def get_currency(ticker: str) -> str:
    """
    Get the reported currency for a ticker from yfinance.
    Returns currency code (GBP, USD, EUR, GBp, etc.) or 'Unknown'.
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        reported = yf_ticker.info.get("currency", "Unknown")
        return reported
    except Exception as e:
        logger.error(f"Error getting currency for {ticker}: {e}")
        return "GBP"  # Default fallback


def get_currencies_parallel(tickers: List[str]) -> Dict[str, str]:
    """
    Fetch currencies for multiple tickers in parallel.
    Returns dict: {ticker: currency}
    """
    reported_currencies = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(get_currency, ticker): ticker for ticker in tickers}
        for future in concurrent.futures.as_completed(futures):
            ticker = futures[future]
            try:
                currency = future.result()
                reported_currencies[ticker] = currency
                logger.info(f"Ticker {ticker}: Reported={currency}")
            except Exception as e:
                logger.error(f"Error getting currency for {ticker}: {e}")
                reported_currencies[ticker] = "GBP"

    return reported_currencies


def get_min_hist_date(ticker: str) -> Optional[datetime.date]:
    """Get the earliest available historical date for a ticker."""
    try:
        yf_ticker = yf.Ticker(ticker)
        hist = yf_ticker.history(period="max")
        if not hist.empty:
            hist_min = hist.index.min()
            # Handle timezone
            try:
                if hist_min.tz is not None:
                    hist_min = hist_min.tz_localize(None)
            except (AttributeError, TypeError):
                pass
            return hist_min.date() if hasattr(hist_min, "date") else hist_min
        return None
    except Exception as e:
        logger.error(f"Error getting min date for {ticker}: {e}")
        return None


def get_common_start_date(tickers: List[str]) -> Optional[datetime.date]:
    """
    Find the latest (most recent) of all tickers' start dates.
    This ensures all tickers have data from this date forward.
    """
    common_start_dates = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(get_min_hist_date, ticker): ticker for ticker in tickers}
        for future in concurrent.futures.as_completed(futures):
            ticker = futures[future]
            try:
                min_hist = future.result()
                if min_hist:
                    common_start_dates[ticker] = min_hist
                    logger.info(f"Min hist date for {ticker}: {min_hist}")
                else:
                    logger.warning(f"No historical data for {ticker}")
            except Exception as e:
                logger.error(f"Error getting min date for {ticker}: {e}")

    if not common_start_dates:
        return None

    common_start = max(common_start_dates.values())
    logger.info(f"Common start date: {common_start}")
    return common_start


def fetch_and_convert_history(
    ticker: str,
    start_date: datetime.date,
    end_date: datetime.date,
    reported_currency: str,
    dates: pd.DatetimeIndex
) -> Tuple[str, np.ndarray]:
    """
    Fetch historical prices for a ticker and convert to GBP.
    Returns: (ticker, price_array)
    """
    try:
        # Check cache first
        cached = get_cached_prices(ticker, str(start_date), str(end_date))
        date_list = [d.date() for d in dates]

        if cached is not None:
            cached_prices = cached.reindex(date_list).ffill()["close"].fillna(0.0).values
            non_zero_count = np.count_nonzero(cached_prices > 0)
            logger.info(f"Cached prices for {ticker}: {non_zero_count} non-zero out of {len(cached_prices)}")
            if non_zero_count == len(cached_prices):
                return ticker, cached_prices
        else:
            cached_prices = np.zeros(len(dates))

        # Fetch from yfinance if not fully cached
        logger.info(f"Fetching history for {ticker} from {start_date} to {end_date}")
        yf_ticker = yf.Ticker(ticker)
        hist = yf_ticker.history(start=start_date, end=end_date + pd.Timedelta(days=1))

        if hist.empty:
            logger.warning(f"No historical data for {ticker}")
            return ticker, cached_prices

        # Handle timezone
        try:
            if hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
        except (AttributeError, TypeError):
            pass

        # Get prices and forward-fill missing days
        prices = hist["Close"].reindex(dates).ffill().fillna(0.0)

        # Convert currency if needed
        converted_prices = convert_currency(prices, reported_currency, ticker, dates)
        converted_prices = converted_prices.fillna(0.0)

        non_zero = (converted_prices > 0).sum()
        logger.info(f"Non-zero converted prices for {ticker}: {non_zero} out of {len(converted_prices)}")

        # Return converted prices
        return ticker, converted_prices.values

    except Exception as e:
        logger.error(f"Exception in fetch_and_convert_history for {ticker}: {e}")
        return ticker, np.zeros(len(dates))


def convert_currency(
    prices: pd.Series,
    reported_currency: str,
    ticker: str,
    dates: pd.DatetimeIndex
) -> pd.Series:
    """Convert prices from reported currency to GBP."""
    converted_prices = prices.copy()
    is_lse = ticker.endswith('.L')

    # Handle GBp (LSE pence)
    if is_lse and reported_currency == "GBp":
        converted_prices = prices / 100.0
        logger.info(f"Converted {ticker} (LSE, GBp) to GBP (divided by 100)")
        return converted_prices

    # Handle currencies that need FX conversion
    if reported_currency in FX_CONFIG:
        fx_ticker_str, multiply = FX_CONFIG[reported_currency]
        rates = get_fx_rates(fx_ticker_str, dates, reported_currency)

        if rates is not None:
            if multiply:
                converted_prices = prices * rates.reindex(dates).ffill().fillna(1.0)
            else:
                converted_prices = prices / rates.reindex(dates).ffill().fillna(1.0)
            logger.info(f"Converted {ticker} ({reported_currency}) to GBP using FX rates")

    return converted_prices


def get_fx_rates(currency_or_fx_ticker: str, dates: pd.DatetimeIndex, currency: Optional[str] = None, db_path: Optional[str] = None) -> Optional[pd.Series]:
    """
    Get FX rates for a currency, using cache or fetching from yfinance.

    Can be called in two ways:
    1. With FX ticker: get_fx_rates(fx_ticker, dates, currency, db_path)
    2. With currency code: get_fx_rates(currency_code, dates, db_path=db_path)

    Returns pandas Series of rates indexed by dates.
    """
    try:
        # Handle flexible calling patterns
        if currency is None:
            # Called with (currency, dates) - need to look up FX ticker
            currency_param = currency_or_fx_ticker
            if currency_param in FX_CONFIG:
                fx_ticker_str, _ = FX_CONFIG[currency_param]
            elif currency_param in ["GBP", "GBp"]:
                # No FX conversion needed
                return pd.Series(np.ones(len(dates)), index=dates)
            else:
                logger.warning(f"Unknown currency {currency_param}, assuming 1:1")
                return pd.Series(np.ones(len(dates)), index=dates)
        else:
            # Called with (fx_ticker, dates, currency)
            fx_ticker_str = currency_or_fx_ticker
            currency_param = currency

        # Check cache first
        start_date = dates.min().date()
        end_date = dates.max().date()
        cached = get_cached_fx_rates(fx_ticker_str, str(start_date), str(end_date), db_path)

        if cached is not None:
            date_list = [d.date() for d in dates]
            cached_rates = cached.reindex(date_list).ffill()["close"].fillna(1.0).values
            logger.info(f"Cached FX rates for {fx_ticker_str}")
            return pd.Series(cached_rates, index=dates)

        # Fetch from yfinance
        logger.info(f"Fetching FX rates for {currency_param}: {fx_ticker_str}")
        fx_ticker_obj = yf.Ticker(fx_ticker_str)
        fx_hist = fx_ticker_obj.history(start=start_date, end=end_date + pd.Timedelta(days=1))

        if fx_hist.empty:
            logger.warning(f"No FX data for {currency_param}, assuming 1:1")
            return pd.Series(np.ones(len(dates)), index=dates)

        # Handle timezone
        try:
            if fx_hist.index.tz is not None:
                fx_hist.index = fx_hist.index.tz_localize(None)
        except Exception:
            pass

        # Forward-fill and return
        fx_rates = fx_hist["Close"].reindex(dates).ffill().fillna(1.0)

        # Cache the fetched rates
        conn = get_connection(db_path)
        try:
            for i in range(len(fx_rates)):
                fx_val = fx_rates.iloc[i]
                if pd.notna(fx_val) and fx_val > 0:
                    date_str = dates[i].strftime("%Y-%m-%d")
                    conn.execute(
                        "INSERT OR REPLACE INTO prices (ticker, date, close) VALUES (?, ?, ?)",
                        (fx_ticker_str, date_str, float(fx_val))
                    )
            conn.commit()
            logger.info(f"FX rates for {currency_param} fetched and cached: {len(fx_rates)} days")
        finally:
            conn.close()

        return fx_rates

    except Exception as e:
        logger.error(f"Error fetching FX for {currency_param}: {e}, assuming 1:1")
        return pd.Series(np.ones(len(dates)), index=dates)


def fetch_prices_parallel(
    tickers: List[str],
    start_date,
    end_date,
    currencies: Dict[str, str],
    dates: pd.DatetimeIndex
) -> Dict[str, np.ndarray]:
    """
    Fetch prices for multiple tickers in parallel.
    Returns dict: {ticker: price_array}
    """
    price_data = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(
                fetch_and_convert_history,
                ticker,
                start_date,
                end_date,
                currencies.get(ticker, "GBP"),
                dates
            ): ticker
            for ticker in tickers
        }

        for future in concurrent.futures.as_completed(futures):
            ticker = futures[future]
            try:
                ticker_result, prices_arr = future.result()
                price_data[ticker_result] = prices_arr
            except Exception as e:
                logger.error(f"Error fetching prices for {ticker}: {e}")
                price_data[ticker] = np.zeros(len(dates))

    return price_data


def get_needed_currencies(currencies: Dict[str, str]) -> set:
    """Get set of currencies that need FX conversion (non-GBP/GBp)."""
    needed = set()
    for currency in currencies.values():
        if currency not in ["GBP", "GBp", "Unknown"] and currency in FX_CONFIG:
            needed.add(currency)
    return needed
