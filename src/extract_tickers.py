import re
import sqlite3
from typing import Dict, Optional

import pandas as pd
import requests


def extract_isin(security_text: str) -> Optional[str]:
    """
    Extract ISIN from 'Security / ISIN' column using regex.
    ISIN format: 2 letters + 9 alphanumeric + 1 digit.
    """
    match = re.search(r"ISIN ([A-Z]{2}[A-Z0-9]{9}[0-9])", str(security_text))
    return match.group(1) if match else None


def search_ticker_for_isin(security_name: str, isin: str) -> Optional[str]:
    """
    Use Yahoo Finance search API to find ticker by searching ETF name first, then match/filter preferring LSE (.L) tickers.
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

        print(
            f"Debug for {security_name} ({isin}): Name search quotes = {len(data.get('quotes', []))} items"
        )

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
            else:
                # Fallback to first valid ETF/Equity
                valid_quotes = [
                    q for q in data["quotes"] if q.get("quoteType") in ["ETF", "EQUITY"]
                ]
                if valid_quotes:
                    quote = valid_quotes[0]
                else:
                    quote = data["quotes"][0]

            symbol = quote.get("symbol")
            if symbol and symbol != isin:
                # Optional: Verify if possible (yfinance doesn't easily provide ISIN, so assume match)
                return symbol

        # Fallback: Search with ISIN
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={isin}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        print(
            f"Debug fallback for {isin}: ISIN search quotes = {len(data.get('quotes', []))} items"
        )

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
        print(f"Error searching for {security_name} ({isin}): {e}")
        return None


def main():
    # Read the merged SQLite database
    db_path = "db/merged_trading.db"
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM trades", conn)
    conn.close()

    # Extract unique securities and ISINs
    def extract_security_and_isin(text: str) -> tuple:
        match = re.search(r"(.*?) / ISIN ([A-Z]{2}[A-Z0-9]{9}[0-9])", str(text))
        if match:
            return match.group(1).strip(), match.group(2)
        return str(text).strip(), None

    unique_securities = (
        df["Security / ISIN"].apply(extract_security_and_isin).drop_duplicates()
    )
    unique_securities = [
        s for s in unique_securities if s[1] is not None
    ]  # Only those with ISIN

    print("Unique Securities and ISINs extracted:")
    for name, isin in unique_securities:
        print(f"- {name} / {isin}")

    # Map to tickers
    security_to_ticker: Dict[tuple[str, str], str] = {}
    for name, isin in unique_securities:
        ticker = search_ticker_for_isin(name, isin)
        security_to_ticker[(name, isin)] = ticker or "Not found"
        print(f"{name} / {isin} -> {ticker or 'Not found'}")

    # Apply tickers to all rows
    def get_ticker(security_text: str) -> str:
        name, isin = extract_security_and_isin(security_text)
        if isin:
            return security_to_ticker.get((name, isin), "Not found")
        return "Not found"

    df["Ticker"] = df["Security / ISIN"].apply(get_ticker)

    # Update the database with the new Ticker column
    conn = sqlite3.connect(db_path)
    df.to_sql("trades", conn, if_exists="replace", index=False)
    conn.close()

    print(f"\nTickers added to database for {len(unique_securities)} unique securities")


if __name__ == "__main__":
    main()
