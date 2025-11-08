import sqlite3
import pandas as pd
import yfinance as yf
import re
import requests
from datetime import datetime
from typing import Dict, Optional, List, Tuple

# Reuse the search_ticker_for_isin function from main.py
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

def extract_security_and_isin(text: str) -> Tuple[str, Optional[str]]:
    match = re.search(r"(.*?) / ISIN ([A-Z]{2}[A-Z0-9]{9}[0-9])", str(text))
    if match:
        return match.group(1).strip(), match.group(2)
    return str(text).strip(), None

def load_trades_from_db(db_path: str = "db/merged_trading.db") -> pd.DataFrame:
    """Load trades from DB, including Ticker if available."""
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM trades", conn)
        conn.close()
        print(f"Loaded {len(df)} trades from DB")
        return df
    except Exception as e:
        print(f"Error loading from DB: {e}. Falling back to CSV load.")
        return load_trades_from_csv()

def load_trades_from_csv(csv_path: str = "trading_statements/GIA_Trading_statement_6_Jun_2022_to_7_Nov_2025.csv") -> pd.DataFrame:
    """Load and process the CSV similar to merge_csv.py."""
    import io
    def clean_currency(value: str) -> float:
        if pd.isna(value):
            return 0.0
        value_str = str(value).replace("£", "").replace(",", "")
        try:
            return float(value_str)
        except ValueError:
            return 0.0

    try:
        df = pd.read_csv(csv_path, skiprows=1)
        df.columns = df.columns.str.strip()
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
        df["Share Price"] = df["Share Price"].apply(clean_currency)
        df["Total Trade Value"] = df["Total Trade Value"].apply(clean_currency)
        # Vectorized date parsing
        df["Trade Date/Time"] = pd.to_datetime(df["Trade Date/Time"], format="%d/%m/%y %H:%M:%S", errors="coerce")
        mask_no_time = df["Trade Date/Time"].isna()
        df.loc[mask_no_time, "Trade Date/Time"] = pd.to_datetime(df.loc[mask_no_time, "Trade Date/Time"], format="%d/%m/%y", errors="coerce")
        df = df.dropna(subset=["Trade Date/Time"])
        df["Account_Type"] = "GIA"  # Hardcode for single file
        print(f"Loaded {len(df)} trades from CSV: {csv_path}")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()

def add_tickers_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extract unique securities and add tickers."""
    unique_securities = df["Security / ISIN"].apply(extract_security_and_isin).drop_duplicates()
    unique_securities = [s for s in unique_securities if s[1] is not None]

    security_to_ticker = {}
    for name, isin in unique_securities:
        ticker = search_ticker_for_isin(name, isin)
        security_to_ticker[(name, isin)] = ticker or "Not found"

    def get_ticker(security_text: str) -> str:
        name, isin = extract_security_and_isin(security_text)
        if isin:
            return security_to_ticker.get((name, isin), "Not found")
        return "Not found"

    df["Ticker"] = df["Security / ISIN"].apply(get_ticker)
    print(f"Added tickers for {len(unique_securities)} unique securities")
    return df

def fetch_yf_price(ticker: str, trade_date: pd.Timestamp) -> Optional[float]:
    """Fetch yfinance Close price for the given date (or nearest previous trading day), converted to GBP."""
    from datetime import timedelta
    try:
        yf_ticker = yf.Ticker(ticker)
        currency = yf_ticker.info.get("currency", "GBP")
        print(f"Ticker {ticker} currency: {currency}")
        start_date = (trade_date - timedelta(days=5)).date()
        end_date = (trade_date + timedelta(days=1)).date()
        hist = yf_ticker.history(start=start_date, end=end_date)
        if not hist.empty:
            # Handle timezone
            if hasattr(hist.index, 'tz') and hist.index.tz is not None:
                hist = hist.tz_localize(None)
            # Get dates
            hist_dates = [idx.date() for idx in hist.index]
            # Find valid indices where date <= trade_date.date()
            valid_indices = [i for i, d in enumerate(hist_dates) if d <= trade_date.date()]
            if valid_indices:
                closest_i = max(valid_indices)
                yf_price = hist.iloc[closest_i]['Close']
                if yf_price > 0:
                    converted_price = yf_price
                    if currency == "USD":
                        # Fetch FX for GBPUSD (USD per GBP, so USD / rate = GBP)
                        fx_ticker = yf.Ticker("GBPUSD=X")
                        fx_hist = fx_ticker.history(start=start_date, end=end_date)
                        if not fx_hist.empty:
                            if hasattr(fx_hist.index, 'tz') and fx_hist.index.tz is not None:
                                fx_hist = fx_hist.tz_localize(None)
                            fx_dates = [idx.date() for idx in fx_hist.index]
                            fx_valid = [i for i, d in enumerate(fx_dates) if d <= trade_date.date()]
                            if fx_valid:
                                fx_closest_i = max(fx_valid)
                                fx_rate = fx_hist.iloc[fx_closest_i]['Close']
                                if fx_rate > 0:
                                    converted_price = yf_price / fx_rate
                                    print(f"Converted USD {yf_price:.2f} to GBP {converted_price:.2f} using rate {fx_rate:.4f}")
                                else:
                                    print(f"No valid FX rate for {trade_date.date()}, using raw USD price")
                            else:
                                print(f"No FX data for {trade_date.date()}, using raw USD price")
                        else:
                            print(f"No FX history for {trade_date.date()}, using raw USD price")
                    elif currency == "GBp":
                        converted_price = yf_price / 100.0
                        print(f"Converted GBp {yf_price:.2f} to GBP {converted_price:.2f}")
                    return converted_price
        return None
    except Exception as e:
        print(f"Error fetching price for {ticker} on {trade_date.date()}: {e}")
        return None

def test_price_factors(df: pd.DataFrame, min_trades_per_ticker: int = 2) -> Dict[str, Dict]:
    """Test for constant multiplication factor per ticker.
    Computes ratio = converted_yf_price / csv_price for each trade, checks if constant across trades per ticker.
    CSV prices are in GBP; yf prices converted to GBP, focusing on unit mismatch like pence vs pounds.
    """
    results = {}
    df["Trade Date/Time"] = pd.to_datetime(df["Trade Date/Time"], errors="coerce")
    df = df.dropna(subset=["Trade Date/Time"])
    buy_trades = df[df["Transaction Type"] == "Buy"].copy()  # Use buys for price comparison
    buy_trades["Trade_Date"] = buy_trades["Trade Date/Time"].dt.date

    unique_tickers = buy_trades["Ticker"].unique()
    for ticker in unique_tickers:
        if ticker == "Not found" or pd.isna(ticker):
            continue
        ticker_trades = buy_trades[buy_trades["Ticker"] == ticker]
        if len(ticker_trades) < min_trades_per_ticker:
            print(f"Skipping {ticker}: only {len(ticker_trades)} buy trades")
            continue

        ratios = []
        for _, trade in ticker_trades.iterrows():
            csv_price = trade["Share Price"]
            if csv_price <= 0:
                continue
            trade_date = pd.Timestamp(trade["Trade Date/Time"])
            yf_price = fetch_yf_price(ticker, trade_date)
            if yf_price is None or yf_price <= 0:
                continue
            ratio = yf_price / csv_price  # Now both in GBP; ratio should be ~1 or unit factor
            ratios.append({
                "date": trade["Trade_Date"],
                "csv_price": csv_price,
                "yf_price": yf_price,
                "ratio": ratio
            })

        if len(ratios) >= min_trades_per_ticker:
            avg_ratio = sum(r["ratio"] for r in ratios) / len(ratios)
            std_ratio = (sum((r["ratio"] - avg_ratio)**2 for r in ratios) / len(ratios))**0.5
            is_constant = std_ratio < 0.01  # Threshold for "constant" (allow small variance)
            results[ticker] = {
                "trades": len(ratios),
                "avg_ratio": avg_ratio,
                "std_ratio": std_ratio,
                "is_constant": is_constant,
                "samples": ratios[:3]  # First 3 samples
            }
            print(f"\n{ticker}: {len(ratios)} trades, avg ratio {avg_ratio:.2f} (std {std_ratio:.4f}), constant: {is_constant}")
            for sample in ratios[:3]:
                print(f"  {sample['date']}: CSV £{sample['csv_price']:.2f} vs YF {sample['yf_price']:.2f} (ratio {sample['ratio']:.2f})")

    return results

if __name__ == "__main__":
    db_path = "db/merged_trading.db"
    csv_path = "trading_statements/GIA_Trading_statement_6_Jun_2022_to_7_Nov_2025.csv"

    # Load data
    df = load_trades_from_db(db_path)
    if df.empty:
        df = load_trades_from_csv(csv_path)

    if df.empty:
        print("No data loaded. Ensure CSV or DB exists.")
    else:
        # Ensure datetime
        df["Trade Date/Time"] = pd.to_datetime(df["Trade Date/Time"], errors="coerce")
        df = df.dropna(subset=["Trade Date/Time"])
        # Add tickers if not present
        if "Ticker" not in df.columns or df["Ticker"].isna().all():
            df = add_tickers_to_df(df)

        # Test factors
        factors = test_price_factors(df)
        print(f"\nSummary: Tested {len(factors)} tickers")
        constants = {k: v for k, v in factors.items() if v["is_constant"]}
        print(f"Constant factors found: {len(constants)} tickers")
        if constants:
            for ticker, info in constants.items():
                print(f"  {ticker}: Factor ~{info['avg_ratio']:.2f}")