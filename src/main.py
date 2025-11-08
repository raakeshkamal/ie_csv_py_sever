import re
import sqlite3
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import concurrent.futures
from fastapi import BackgroundTasks, FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.merge_csv import merge_csv_files  # Import the merging function

import logging

# Configure logging with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

db_path = "db/merged_trading.db"


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


app = FastAPI(title="InvestEngine CSV Server")

# Mount templates directory (though Jinja handles it)
templates = Jinja2Templates(directory="src/templates")

# If we add static files later, mount static/
# app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    """
    Render the upload form page.
    """
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/upload/")
async def upload_files(
    background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)
):
    """
    Handle batch CSV upload, merge them, and save to merged_trading.csv (overwriting old).
    Returns JSON data for client-side rendering.
    """
    logger.info("Endpoint /upload/ called") 
    if not files:
        return JSONResponse(
            content={"success": False, "error": "No files uploaded"},
            status_code=400,
        )

    file_data = []
    for file in files:
        if file.filename and not file.filename.endswith(".csv"):
            continue  # Skip non-CSV
        content = await file.read()
        content_str = content.decode("utf-8")
        file_data.append((file.filename, content_str))

    if not file_data:
        return JSONResponse(
            content={"success": False, "error": "No valid CSV files"},
            status_code=400,
        )

    try:
        # Call the merge function
        merged_df = merge_csv_files(file_data)
        # Save to SQLite database (overwrites if exists)
        conn = sqlite3.connect(db_path)
        merged_df.to_sql("trades", conn, if_exists="replace", index=False)
        conn.close()

        # Extract tickers synchronously to ensure they are available for charts
        extract_tickers_to_db()

        logger.info(
            f"DEBUG: Merged DF shape: {merged_df.shape}, Date range: {merged_df['Trade Date/Time'].min()} to {merged_df['Trade Date/Time'].max()}"
        )
        logger.info(
            f"DEBUG: Sample Transaction Types: {merged_df['Transaction Type'].unique()}"
        )

        # Success JSON
        min_date = merged_df["Trade Date/Time"].min()
        max_date = merged_df["Trade Date/Time"].max()
        return JSONResponse(
            content={
                "success": True,
                "total_transactions": len(merged_df),
                "min_date": str(min_date),
                "max_date": str(max_date),
            }
        )
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500,
        )


@app.get("/portfolio-values/")
async def get_portfolio_values():
    """
    Compute and return monthly net contributions and daily actual portfolio values using yfinance.
    """
    logger.info("Endpoint /portfolio-values/ called")
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM trades", conn)
        logger.info(f"Loaded {len(df)} trades from database")
        logger.info(f"Columns: {list(df.columns)}")
        df["Trade Date/Time"] = pd.to_datetime(
            df["Trade Date/Time"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
        df = df.dropna(subset=["Trade Date/Time"])
        conn.close()

        # Log sample CSV prices and currencies
        logger.info("=== DEBUG: Sample CSV Trade Data ===")
        sample_columns = ["Ticker", "Trade Date/Time", "Share Price", "Total Trade Value", "Transaction Type"]
        sample_trades = df[sample_columns].head(5)
        logger.info(sample_trades.to_string(index=False))
        logger.info("CSV prices are in GBP (from '£' removal in merge_csv.py)")

        if df.empty:
            return JSONResponse(
                content={"success": False, "error": "No trades data in database"},
                status_code=404,
            )

        # Compute monthly net contributions
        df = df.copy()
        df["Month"] = df["Trade Date/Time"].dt.to_period("M")
        df["Net_Value"] = df.apply(
            lambda row: row["Total Trade Value"]
            if row["Transaction Type"] == "Buy"
            else -row["Total Trade Value"],
            axis=1,
        )
        monthly_net = df.groupby("Month")["Net_Value"].sum().reset_index()
        monthly_net["Month"] = monthly_net["Month"].astype(str)
        monthly_net_data = monthly_net.to_dict("records")

        # For daily portfolio values
        # Date range computed later based on common start

        # Unique valid tickers
        unique_tickers = (
            df.loc[df["Ticker"] != "Not found", "Ticker"].drop_duplicates().tolist()
        )
        logger.info(f"Unique valid tickers: {unique_tickers}")
        if not unique_tickers:
            return JSONResponse(
                content={
                    "success": False,
                    "error": "No valid tickers found in trades",
                },
                status_code=404,
            )

        # Fetch currencies for all tickers in parallel
        def get_currency(ticker):
            try:
                yf_ticker = yf.Ticker(ticker)
                reported = yf_ticker.info.get("currency", "Unknown")
                return ticker, reported
            except Exception as e:
                logger.error(f"Error getting currency for {ticker}: {e}")
                return ticker, "GBP"

        reported_currencies = {}
        needed_fx = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(get_currency, ticker) for ticker in unique_tickers]
            for future in concurrent.futures.as_completed(futures):
                ticker, reported = future.result()
                reported_currencies[ticker] = reported
                if reported != "Unknown" and reported != "GBP" and reported != "GBp":
                    needed_fx.add(reported)
                logger.info(f"Ticker {ticker}: Reported={reported}")

        # Log yfinance currencies for tickers
        logger.info("=== DEBUG: yfinance Currencies ===")
        for ticker in unique_tickers[:3]:  # Sample first 3
            currency = reported_currencies.get(ticker, "Unknown")
            logger.info(f"Ticker {ticker}: Currency = {currency}")

        # Compute common start date in parallel
        def get_min_hist_date(ticker):
            try:
                yf_ticker = yf.Ticker(ticker)
                hist = yf_ticker.history(period="max")
                if not hist.empty:
                    # Handle timezone
                    hist_min = hist.index.min()
                    try:
                        if hist_min.tz is not None:
                            hist_min = hist_min.tz_localize(None)
                    except (AttributeError, TypeError):
                        pass  # Not tz-aware or error, proceed
                    min_hist = (
                        hist_min.date() if hasattr(hist_min, "date") else hist_min
                    )
                    return ticker, min_hist
                else:
                    return ticker, None
            except Exception as e:
                logger.error(f"Error getting min date for {ticker}: {e}")
                return ticker, None

        common_start_dates = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(get_min_hist_date, ticker) for ticker in unique_tickers]
            for future in concurrent.futures.as_completed(futures):
                ticker, min_hist = future.result()
                if min_hist:
                    common_start_dates[ticker] = min_hist
                    logger.info(f"Min hist date for {ticker}: {min_hist}")
                else:
                    logger.warning(f"No historical data for {ticker}")

        if not common_start_dates:
            return JSONResponse(
                content={
                    "success": False,
                    "error": "No historical data for any ticker",
                },
                status_code=404,
            )

        common_start = max(common_start_dates.values())
        logger.info(f"Common start date: {common_start}")

        max_date = df["Trade Date/Time"].max().date()
        if common_start > max_date:
            return JSONResponse(
                content={"success": False, "error": "No valid trade date range"},
                status_code=404,
            )

        dates = pd.date_range(start=common_start, end=max_date, freq="D")
        daily_dates = [d.strftime("%Y-%m-%d") for d in dates]

        # Compute initial holdings
        initial_holdings = {}
        for ticker in unique_tickers:
            pre_trades_mask = (df["Ticker"] == ticker) & (
                df["Trade Date/Time"].dt.date < common_start
            )
            pre_trades = df[pre_trades_mask].copy()
            if not pre_trades.empty:
                pre_trades["Quantity_Adj"] = np.where(
                    pre_trades["Transaction Type"] == "Buy",
                    pre_trades["Quantity"],
                    -pre_trades["Quantity"],
                )
                initial_qty = pre_trades["Quantity_Adj"].sum()
            else:
                initial_qty = 0.0
            initial_holdings[ticker] = initial_qty
            logger.info(f"Initial holding for {ticker}: {initial_qty}")

        # Create prices table if not exists (now for converted GBP prices)
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                ticker TEXT,
                date DATE,
                close REAL,
                PRIMARY KEY (ticker, date)
            )
        """)
        conn.commit()

        # Fetch/cache FX rates for non-GBP currencies
        fx_config = {
            "USD": ("GBPUSD=X", False),  # divide by rate (USD/GBP)
            "EUR": ("EURGBP=X", True),   # multiply by rate (GBP/EUR)
            # Add more currencies as needed
        }
        fx_data = {}
        for curr in needed_fx:
            if curr in fx_config:
                fx_ticker_str, multiply = fx_config[curr]
                # Check cache for FX
                cached_fx_query = f"""
                    SELECT date, close FROM prices
                    WHERE ticker = '{fx_ticker_str}' AND date BETWEEN '{common_start}' AND '{max_date}'
                    ORDER BY date
                """
                cached_fx = pd.read_sql_query(cached_fx_query, conn)
                if not cached_fx.empty:
                    cached_fx["date"] = pd.to_datetime(cached_fx["date"]).dt.date
                    cached_fx.set_index("date", inplace=True)
                    date_list_fx = [d.date() for d in dates]
                    cached_fx_prices = cached_fx.reindex(date_list_fx).ffill()["close"].fillna(1.0).values
                    logger.info(f"Cached FX rates for {fx_ticker_str}")
                    fx_data[curr] = {"rates": cached_fx_prices, "multiply": multiply}
                else:
                    try:
                        logger.info(f"Fetching FX for {curr}: {fx_ticker_str}")
                        fx_ticker = yf.Ticker(fx_ticker_str)
                        fx_hist = fx_ticker.history(
                            start=common_start, end=max_date + pd.Timedelta(days=1)
                        )
                        if not fx_hist.empty:
                            try:
                                if fx_hist.index.tz is not None:
                                    fx_hist.index = fx_hist.index.tz_localize(None)
                            except (AttributeError, TypeError):
                                pass
                            fx_close = fx_hist["Close"].reindex(dates).ffill().fillna(1.0)
                            # Cache FX rates
                            for i in range(len(fx_close)):
                                fx_val = fx_close.iloc[i]
                                if pd.notna(fx_val) and fx_val > 0:
                                    date_str = dates[i].strftime("%Y-%m-%d")
                                    conn.execute(
                                        "INSERT OR REPLACE INTO prices (ticker, date, close) VALUES (?, ?, ?)",
                                        (fx_ticker_str, date_str, float(fx_val)),
                                    )
                            conn.commit()
                            fx_data[curr] = {"rates": fx_close.values, "multiply": multiply}
                            logger.info(
                                f"FX rates for {curr} fetched and cached: {len(fx_close)} days, sample: {fx_close.values[:3]}"
                            )
                        else:
                            logger.warning(f"No FX data for {curr}, assuming 1:1")
                            fx_data[curr] = {"rates": np.ones(len(dates)), "multiply": multiply}
                    except Exception as e:
                        logger.error(f"Error fetching FX for {curr}: {e}, assuming 1:1")
                        fx_data[curr] = {"rates": np.ones(len(dates)), "multiply": multiply}
            else:
                logger.warning(f"Unsupported currency {curr}, assuming prices in GBP")
        if not needed_fx:
            logger.info("All currencies are GBP or GBp, no FX needed")

        # Simulate holdings for each ticker (reduced debug output)
        holdings_data = {}
        date_list = [d.date() for d in dates]
        for ticker in unique_tickers:
            ticker_trades = (
                df[
                    (df["Ticker"] == ticker)
                    & (df["Trade Date/Time"].dt.date >= common_start)
                ]
                .sort_values("Trade Date/Time")
                .copy()
            )
            initial_qty = initial_holdings.get(ticker, 0.0)
            logger.info(f"DEBUG HOLDINGS [{ticker}]: initial_qty = {initial_qty}")
            if ticker_trades.empty:
                logger.info(
                    f"DEBUG HOLDINGS [{ticker}]: No trades >= {common_start}, setting all to initial_qty = {initial_qty}"
                )
                holdings_data[ticker] = np.full(len(dates), initial_qty)
                continue

            # Quantity adjustment: Buy positive, Sell negative
            ticker_trades["Quantity_Adj"] = np.where(
                ticker_trades["Transaction Type"] == "Buy",
                ticker_trades["Quantity"],
                -ticker_trades["Quantity"],
            )

            # Aggregate adjustments by date (handle multiple trades per day)
            daily_adj = ticker_trades.groupby(ticker_trades["Trade Date/Time"].dt.date)[
                "Quantity_Adj"
            ].sum()
            logger.info(
                f"DEBUG HOLDINGS [{ticker}]: Number of unique trade dates: {len(daily_adj)}, total adj: {daily_adj.sum()}"
            )
            if len(daily_adj) > 0:
                logger.info(
                    f"DEBUG HOLDINGS [{ticker}]: Sample daily adj: {dict(list(daily_adj.items())[:3])}"
                )

            # Build daily holdings by iterating through dates and accumulating (carries forward previous holding)
            # Reduced logging: only summary, no per-adjustment details
            daily_holdings = pd.Series(index=dates, dtype=float)
            cum_qty = initial_qty
            for i, date in enumerate(dates):
                adj_today = daily_adj.get(date.date(), 0.0)
                cum_qty += adj_today
                daily_holdings.iloc[i] = cum_qty

            # No NaNs or ffill needed; all days explicitly set with carry-forward
            logger.info(
                f"DEBUG HOLDINGS [{ticker}]: Final: NaNs={daily_holdings.isna().sum()}, min={daily_holdings.min():.6f}, max={daily_holdings.max():.6f}"
            )
            non_zero_count = (
                abs(daily_holdings) > 1e-6
            ).sum()  # Ignore floating-point zeros
            logger.info(
                f"DEBUG HOLDINGS [{ticker}]: non-zero days: {non_zero_count} / {len(daily_holdings)}, max holding: {daily_holdings.max() if non_zero_count > 0 else 0}"
            )
            holdings_data[ticker] = daily_holdings.values

# Fetch/cache prices in parallel
        def fetch_and_convert_history(ticker):
            thread_conn = None
            try:
                thread_conn = sqlite3.connect(db_path)
                cached_query = f"""
                    SELECT date, close FROM prices
                    WHERE ticker = '{ticker}' AND date BETWEEN '{common_start}' AND '{max_date}'
                    ORDER BY date
                """
                cached = pd.read_sql_query(cached_query, thread_conn)
                if not cached.empty:
                    cached["date"] = pd.to_datetime(cached["date"]).dt.date
                    cached.set_index("date", inplace=True)
                    date_list = [d.date() for d in dates]
                    cached_prices = cached.reindex(date_list).ffill()["close"].fillna(0.0).values
                    logger.info(f"Cached prices for {ticker}: non-zero count = {np.count_nonzero(cached_prices > 0)} out of {len(cached_prices)}")
                    if not np.any(cached_prices == 0.0):
                        return ticker, cached_prices, True
                else:
                    cached_prices = np.zeros(len(dates))
                    logger.info(f"No cached prices for {ticker}")

                if np.any(cached_prices == 0.0):
                    logger.info(f"Fetching history for {ticker} from {common_start} to {max_date}")
                    yf_ticker = yf.Ticker(ticker)
                    hist = yf_ticker.history(start=common_start, end=max_date + pd.Timedelta(days=1))
                    logger.info(f"Hist length for {ticker}: {len(hist)}")
                    if not hist.empty:
                        logger.info(f"Hist date range for {ticker}: {hist.index.min()} to {hist.index.max()}")
                        try:
                            if hist.index.tz is not None:
                                hist.index = hist.index.tz_localize(None)
                        except (AttributeError, TypeError):
                            pass
                        prices = hist["Close"].reindex(dates).ffill().fillna(0.0)

                        reported = reported_currencies.get(ticker, "GBP")
                        converted_prices = prices.copy()
                        is_lse = ticker.endswith('.L')
                        if is_lse and reported == "GBp":
                            converted_prices = prices / 100.0
                            logger.info(f"Converted {ticker} (LSE, GBp) to GBP (divided by 100)")
                        elif reported == "USD":
                            fx_info = fx_data.get("USD")
                            if fx_info:
                                rates_series = pd.Series(fx_info["rates"], index=dates)
                                converted_prices = prices / rates_series.reindex(dates).ffill().fillna(1.0)
                                logger.info(f"Converted {ticker} (reported USD) to GBP using FX rates")
                        elif reported != "GBP":
                            fx_info = fx_data.get(reported)
                            if fx_info:
                                rates_series = pd.Series(fx_info["rates"], index=dates)
                                if fx_info["multiply"]:
                                    converted_prices = prices * rates_series.reindex(dates).ffill().fillna(1.0)
                                else:
                                    converted_prices = prices / rates_series.reindex(dates).ffill().fillna(1.0)
                                logger.info(f"Converted {ticker} {reported} to GBP using FX rates")

                        converted_prices = converted_prices.fillna(0.0)
                        non_zero = (converted_prices > 0).sum()
                        logger.info(f"Non-zero converted prices for {ticker}: {non_zero} out of {len(converted_prices)}")
                        return ticker, converted_prices.values, False
                    return ticker, np.zeros(len(dates)), False
            except Exception as e:
                logger.error(f"Exception in fetch_and_convert_history for {ticker}: {e}")
                return ticker, np.zeros(len(dates)), True
            finally:
                if thread_conn:
                    thread_conn.close()

        price_data = {}
        to_cache = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(fetch_and_convert_history, ticker) for ticker in unique_tickers]
            for future in concurrent.futures.as_completed(futures):
                ticker, prices_arr, fully_cached = future.result()
                price_data[ticker] = prices_arr
                if not fully_cached:
                    to_cache.append((ticker, prices_arr))

        # Sample comparisons in main thread
        for ticker in unique_tickers:
            if ticker in price_data:
                converted_prices = pd.Series(price_data[ticker], index=dates)
                ticker_trades_sample = df[(df["Ticker"] == ticker) & (df["Transaction Type"] == "Buy")].head(1)
                if not ticker_trades_sample.empty:
                    sample_trade_date = ticker_trades_sample["Trade Date/Time"].iloc[0].date()
                    sample_csv_price = ticker_trades_sample["Share Price"].iloc[0]
                    valid_dates = [d for d in dates if d.date() >= sample_trade_date]
                    if valid_dates:
                        closest_date = min(valid_dates, key=lambda x: abs((x.date() - sample_trade_date).days))
                        price_idx = list(dates).index(closest_date)
                        if 0 <= price_idx < len(converted_prices):
                            yf_converted = converted_prices.iloc[price_idx]
                            logger.info(f"Sample comparison for {ticker} on ~{sample_trade_date}: CSV £{sample_csv_price:.2f} vs yf Converted GBP £{yf_converted:.2f}")

        # Cache non-fully-cached prices using main conn
        for ticker, prices in to_cache:
            converted_prices = pd.Series(prices, index=dates)
            for i in range(len(converted_prices)):
                price_val = converted_prices.iloc[i]
                if pd.notna(price_val) and price_val > 0:
                    date_str = dates[i].strftime("%Y-%m-%d")
                    conn.execute("INSERT OR REPLACE INTO prices (ticker, date, close) VALUES (?, ?, ?)", (ticker, date_str, float(price_val)))
            logger.info(f"Cached prices for {ticker}")
        conn.commit()
        def fetch_and_convert_history(ticker):
            thread_conn = None
            try:
                thread_conn = sqlite3.connect(db_path)
                cached_query = f"""
                    SELECT date, close FROM prices
                    WHERE ticker = '{ticker}' AND date BETWEEN '{common_start}' AND '{max_date}'
                    ORDER BY date
                """
                cached = pd.read_sql_query(cached_query, thread_conn)
                if not cached.empty:
                    cached["date"] = pd.to_datetime(cached["date"]).dt.date
                    cached.set_index("date", inplace=True)
                    date_list = [d.date() for d in dates]
                    cached_prices = cached.reindex(date_list).ffill()["close"].fillna(0.0).values
                    logger.info(f"Cached prices for {ticker}: non-zero count = {np.count_nonzero(cached_prices > 0)} out of {len(cached_prices)}")
                    if not np.any(cached_prices == 0.0):
                        return ticker, cached_prices, True
                else:
                    cached_prices = np.zeros(len(dates))
                    logger.info(f"No cached prices for {ticker}")

                if np.any(cached_prices == 0.0):
                    logger.info(f"Fetching history for {ticker} from {common_start} to {max_date}")
                    yf_ticker = yf.Ticker(ticker)
                    hist = yf_ticker.history(start=common_start, end=max_date + pd.Timedelta(days=1))
                    logger.info(f"Hist length for {ticker}: {len(hist)}")
                    if not hist.empty:
                        logger.info(f"Hist date range for {ticker}: {hist.index.min()} to {hist.index.max()}")
                        try:
                            if hist.index.tz is not None:
                                hist.index = hist.index.tz_localize(None)
                        except (AttributeError, TypeError):
                            pass
                        prices = hist["Close"].reindex(dates).ffill().fillna(0.0)

                        reported = reported_currencies.get(ticker, "GBP")
                        converted_prices = prices.copy()
                        is_lse = ticker.endswith('.L')
                        if is_lse and reported == "GBp":
                            converted_prices = prices / 100.0
                            logger.info(f"Converted {ticker} (LSE, GBp) to GBP (divided by 100)")
                        elif reported == "USD":
                            fx_info = fx_data.get("USD")
                            if fx_info:
                                rates_series = pd.Series(fx_info["rates"], index=dates)
                                converted_prices = prices / rates_series.reindex(dates).ffill().fillna(1.0)
                                logger.info(f"Converted {ticker} (reported USD) to GBP using FX rates")
                        elif reported != "GBP":
                            fx_info = fx_data.get(reported)
                            if fx_info:
                                rates_series = pd.Series(fx_info["rates"], index=dates)
                                if fx_info["multiply"]:
                                    converted_prices = prices * rates_series.reindex(dates).ffill().fillna(1.0)
                                else:
                                    converted_prices = prices / rates_series.reindex(dates).ffill().fillna(1.0)
                                logger.info(f"Converted {ticker} {reported} to GBP using FX rates")

                        converted_prices = converted_prices.fillna(0.0)
                        non_zero = (converted_prices > 0).sum()
                        logger.info(f"Non-zero converted prices for {ticker}: {non_zero} out of {len(converted_prices)}")
                        return ticker, converted_prices.values, False
                    return ticker, np.zeros(len(dates)), False
            except Exception as e:
                logger.error(f"Exception in fetch_and_convert_history for {ticker}: {e}")
                return ticker, np.zeros(len(dates)), True
            finally:
                if thread_conn:
                    thread_conn.close()

        price_data = {}
        to_cache = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(fetch_and_convert_history, ticker) for ticker in unique_tickers]
            for future in concurrent.futures.as_completed(futures):
                ticker, prices_arr, fully_cached = future.result()
                price_data[ticker] = prices_arr
                if not fully_cached:
                    to_cache.append((ticker, prices_arr))

        # Sample comparisons in main thread
        for ticker in unique_tickers:
            if ticker in price_data:
                converted_prices = pd.Series(price_data[ticker], index=dates)
                ticker_trades_sample = df[(df["Ticker"] == ticker) & (df["Transaction Type"] == "Buy")].head(1)
                if not ticker_trades_sample.empty:
                    sample_trade_date = ticker_trades_sample["Trade Date/Time"].iloc[0].date()
                    sample_csv_price = ticker_trades_sample["Share Price"].iloc[0]
                    valid_dates = [d for d in dates if d.date() >= sample_trade_date]
                    if valid_dates:
                        closest_date = min(valid_dates, key=lambda x: abs((x.date() - sample_trade_date).days))
                        price_idx = list(dates).index(closest_date)
                        if 0 <= price_idx < len(converted_prices):
                            yf_converted = converted_prices.iloc[price_idx]
                            logger.info(f"Sample comparison for {ticker} on ~{sample_trade_date}: CSV £{sample_csv_price:.2f} vs yf Converted GBP £{yf_converted:.2f}")

        # Cache non-fully-cached prices using main conn
        for ticker, prices in to_cache:
            converted_prices = pd.Series(prices, index=dates)
            for i in range(len(converted_prices)):
                price_val = converted_prices.iloc[i]
                if pd.notna(price_val) and price_val > 0:
                    date_str = dates[i].strftime("%Y-%m-%d")
                    conn.execute("INSERT OR REPLACE INTO prices (ticker, date, close) VALUES (?, ?, ?)", (ticker, date_str, float(price_val)))
            logger.info(f"Cached prices for {ticker}")
        conn.commit()

        # Move sample comparisons to main thread (now that prices are fetched)
        for ticker in unique_tickers:
            if ticker in price_data:
                converted_prices = pd.Series(price_data[ticker], index=dates)
                ticker_trades_sample = df[
                    (df["Ticker"] == ticker) & (df["Transaction Type"] == "Buy")
                ].head(1)
                if not ticker_trades_sample.empty:
                    sample_trade_date = ticker_trades_sample["Trade Date/Time"].iloc[0].date()
                    sample_csv_price = ticker_trades_sample["Share Price"].iloc[0]
                    valid_dates = [d for d in dates if d.date() >= sample_trade_date]
                    if valid_dates:
                        closest_date = min(valid_dates, key=lambda x: abs((x.date() - sample_trade_date).days))
                        price_idx = list(dates).index(closest_date)
                        if 0 <= price_idx < len(converted_prices):
                            yf_converted = converted_prices.iloc[price_idx]
                            logger.info(f"Sample comparison for {ticker} on ~{sample_trade_date}: CSV £{sample_csv_price:.2f} vs yf Converted GBP £{yf_converted:.2f}")

        # Cache non-fully-cached prices
        for ticker, prices in to_cache:
            converted_prices = pd.Series(prices, index=dates)
            for i in range(len(converted_prices)):
                price_val = converted_prices.iloc[i]
                if pd.notna(price_val) and price_val > 0:
                    date_str = dates[i].strftime("%Y-%m-%d")
                    conn.execute(
                        "INSERT OR REPLACE INTO prices (ticker, date, close) VALUES (?, ?, ?)",
                        (ticker, date_str, float(price_val)),
                    )
            conn.commit()
            logger.info(f"Cached prices for {ticker}")

        conn.close()

        # Ensure all price_data have no NaN and correct length
        for ticker in unique_tickers:
            arr = np.array(price_data[ticker])
            if len(arr) != len(dates):
                logger.warning(
                    f"Warning: price_data length mismatch for {ticker}: {len(arr)} vs {len(dates)}, adjusting"
                )
                if len(arr) < len(dates):
                    arr = np.pad(
                        arr, (0, len(dates) - len(arr)), "constant", constant_values=0.0
                    )
                else:
                    arr = arr[: len(dates)]
            price_data[ticker] = np.nan_to_num(arr, nan=0.0)

        # Compute daily portfolio values
        daily_values = []
        for i in range(len(dates)):
            day_value = 0.0
            for ticker in unique_tickers:
                holding = holdings_data[ticker][i]
                price = price_data[ticker][i]
                day_value += holding * price
            daily_values.append(float(np.nan_to_num(day_value, nan=0.0)))

        return JSONResponse(
            content={
                "success": True,
                "monthly_net": monthly_net_data,
                "daily_dates": daily_dates,
                "daily_values": daily_values,
            }
        )

    except Exception as e:
        logger.error(f"Error in get_portfolio_values: {e}")
        import traceback

        traceback.print_exc()
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500,
        )


def extract_tickers_to_db():
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM trades", conn)
        conn.close()

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

        security_to_ticker: Dict[tuple[str, str], str] = {}
        for name, isin in unique_securities:
            ticker = search_ticker_for_isin(name, isin)
            security_to_ticker[(name, isin)] = ticker or "Not found"

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

        logger.info(
            f"Tickers extracted and added to database for {len(unique_securities)} unique securities"
        )

    except Exception as ticker_err:
        logger.error(f"Ticker extraction error: {ticker_err}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
