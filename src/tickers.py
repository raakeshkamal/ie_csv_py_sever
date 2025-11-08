"""
Ticker extraction module for InvestEngine CSV Server.
Handles ISIN parsing and Yahoo Finance ticker search.
"""

import re
import logging
from typing import Optional, Dict, Tuple
import requests

logger = logging.getLogger(__name__)


def extract_security_and_isin(text: str) -> Tuple[str, Optional[str]]:
    """
    Extract security name and ISIN from text like "Vanguard ... ETF / ISIN GB00B3XXRP09"
    Returns (security_name, isin) where isin may be None
    """
    match = re.search(r"(.*?)\s*/\s*ISIN\s+([A-Z]{2}[A-Z0-9]{9}[0-9])", str(text))
    if match:
        return match.group(1).strip(), match.group(2)
    return str(text).strip(), None


def search_ticker_for_isin(security_name: str, isin: str) -> Optional[str]:
    """
    Use Yahoo Finance search API to find ticker by searching ETF name first,
    then match/filter preferring LSE (.L) tickers.
    Fallback to ISIN search if needed.
    """
    try:
        # Primary: Search with ETF name
        search_query = security_name
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={search_query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if data.get("quotes"):
            # Prefer LSE exchange or .L suffix, and ETF/Equity types
            lse_candidates = [
                q
                for q in data["quotes"]
                if (q.get("exchange") == "LSE" or q.get("symbol", "").endswith(".L"))
                and q.get("quoteType") in ["ETF", "EQUITY"]
            ]
            if lse_candidates:
                # Sort by relevance (e.g., name similarity), take first
                quote = lse_candidates[0]
                symbol = quote.get("symbol")
                if symbol and symbol != isin:
                    # Optional: Verify if possible (yfinance doesn't easily provide ISIN, so assume match)
                    return symbol
            # If no LSE candidates, continue to ISIN search fallback

        # Fallback: Search with ISIN
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={isin}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if data.get("quotes"):
            lse_candidates = [
                q
                for q in data["quotes"]
                if q.get("exchange") == "LSE" or q.get("symbol", "").endswith(".L")
            ]
            if lse_candidates:
                quote = lse_candidates[0]
            else:
                valid_quotes = [
                    q
                    for q in data["quotes"]
                    if q.get("quoteType") in ["ETF", "EQUITY", "MUTUALFUND", "CURRENCY"]
                ]
                if valid_quotes:
                    quote = valid_quotes[0]
                else:
                    quote = data["quotes"][0]

            symbol = quote.get("symbol")
            if symbol and not symbol.startswith(isin) and symbol != isin:
                return symbol

        return None
    except Exception as e:
        logger.error(f"Error searching for {security_name} ({isin}): {e}")
        return None


def extract_tickers_for_df(df) -> Dict[str, Optional[str]]:
    """
    Extract tickers for all unique securities in the dataframe.
    Accepts either a DataFrame or a dictionary that can be converted to DataFrame.
    Returns dict mapping security_name_or_isin_string -> ticker (or None/Not found)
    """
    import pandas as pd

    # Convert dict to DataFrame if needed
    if isinstance(df, dict):
        df = pd.DataFrame(df)

    # Get unique security texts
    unique_securities = df["Security / ISIN"].drop_duplicates().tolist()

    # Build mapping from security_text -> ticker
    security_to_ticker: Dict[str, Optional[str]] = {}
    for security_text in unique_securities:
        name, isin = extract_security_and_isin(security_text)

        if isin is not None:
            # Has ISIN - search for ticker
            ticker = search_ticker_for_isin(name, isin)
            security_to_ticker[security_text] = ticker
        else:
            # No ISIN - mark as None (not found)
            security_to_ticker[security_text] = None

    return security_to_ticker


def add_tickers_to_df(df) -> object:  # Returns pd.DataFrame but avoid import overhead
    """Add Ticker column to dataframe by extracting tickers."""
    security_to_ticker = extract_tickers_for_df(df)

    def get_ticker(security_text: str) -> str:
        # security_to_ticker now uses security_text as key
        return security_to_ticker.get(security_text, "Not found") or "Not found"

    df["Ticker"] = df["Security / ISIN"].apply(get_ticker)
    return df
