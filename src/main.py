import re
import sqlite3
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fastapi import BackgroundTasks, FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.merge_csv import merge_csv_files  # Import the merging function

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
        print(f"Error searching for {security_name} ({isin}): {e}")
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

        print(
            f"DEBUG: Merged DF shape: {merged_df.shape}, Date range: {merged_df['Trade Date/Time'].min()} to {merged_df['Trade Date/Time'].max()}"
        )
        print(
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
    print("Endpoint /portfolio-values/ called")
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM trades", conn)
        print(f"Loaded {len(df)} trades from database")
        print(f"Columns: {list(df.columns)}")
        df["Trade Date/Time"] = pd.to_datetime(
            df["Trade Date/Time"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
        df = df.dropna(subset=["Trade Date/Time"])
        conn.close()

        # Log sample CSV prices and currencies
        print("=== DEBUG: Sample CSV Trade Data ===")
        sample_trades = df.head(5)[
            [
                "Ticker",
                "Trade Date/Time",
                "Share Price",
                "Total Trade Value",
                "Transaction Type",
            ]
        ]
        print(sample_trades.to_string(index=False))
        print("CSV prices are in GBP (from '£' removal in merge_csv.py)")

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
        print(f"Unique valid tickers: {unique_tickers}")
        if not unique_tickers:
            return JSONResponse(
                content={
                    "success": False,
                    "error": "No valid tickers found in trades",
                },
                status_code=404,
            )

        # Fetch currencies for all tickers
        reported_currencies = {}
        needed_fx = set()
        for ticker in unique_tickers:
            try:
                yf_ticker = yf.Ticker(ticker)
                reported = yf_ticker.info.get("currency", "Unknown")
                reported_currencies[ticker] = reported
                if reported != "Unknown" and reported != "GBP" and reported != "GBp":
                    needed_fx.add(reported)
                print(f"Ticker {ticker}: Reported={reported}")
            except Exception as e:
                print(f"Error getting currency for {ticker}: {e}")
                reported_currencies[ticker] = "GBP"

        # Log yfinance currencies for tickers
        print("=== DEBUG: yfinance Currencies ===")
        for ticker in unique_tickers[:3]:  # Sample first 3
            try:
                yf_ticker = yf.Ticker(ticker)
                currency = yf_ticker.info.get("currency", "Unknown")
                print(f"Ticker {ticker}: Currency = {currency}")
            except Exception as e:
                print(f"Error getting currency for {ticker}: {e}")

        # Compute common start date
        common_start_dates = {}
        for ticker in unique_tickers:
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
                    common_start_dates[ticker] = min_hist
                    print(f"Min hist date for {ticker}: {min_hist}")
            except Exception as e:
                print(f"Error getting min date for {ticker}: {e}")

        if not common_start_dates:
            return JSONResponse(
                content={
                    "success": False,
                    "error": "No historical data for any ticker",
                },
                status_code=404,
            )

        common_start = max(common_start_dates.values())
        print(f"Common start date: {common_start}")

        max_date = df["Trade Date/Time"].max().date()
        if common_start > max_date:
            return JSONResponse(
                content={"success": False, "error": "No overlapping historical data"},
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
            print(f"Initial holding for {ticker}: {initial_qty}")

        # Create prices table if not exists (now for converted GBP prices)
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE IF EXISTS prices")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                ticker TEXT,
                date DATE,
                close REAL,
                PRIMARY KEY (ticker, date)
            )
        """)
        conn.commit()

        # Fetch FX rates for non-GBP currencies
        fx_config = {
            "USD": ("GBPUSD=X", False),  # divide by rate (USD/GBP)
            "EUR": ("EURGBP=X", True),   # multiply by rate (GBP/EUR)
            # Add more currencies as needed
        }
        fx_data = {}
        for curr in needed_fx:
            if curr in fx_config:
                fx_ticker_str, multiply = fx_config[curr]
                try:
                    print(f"Fetching FX for {curr}: {fx_ticker_str}")
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
                        fx_close = (
                            fx_hist["Close"].reindex(dates).ffill().fillna(1.0)
                        )
                        fx_data[curr] = {"rates": fx_close.values, "multiply": multiply}
                        print(
                            f"FX rates for {curr} fetched: {len(fx_close)} days, sample: {fx_close.values[:3]}"
                        )
                    else:
                        print(f"No FX data for {curr}, assuming 1:1")
                        fx_data[curr] = {"rates": np.ones(len(dates)), "multiply": multiply}
                except Exception as e:
                    print(f"Error fetching FX for {curr}: {e}, assuming 1:1")
                    fx_data[curr] = {"rates": np.ones(len(dates)), "multiply": multiply}
            else:
                print(f"Unsupported currency {curr}, assuming prices in GBP")
        if not needed_fx:
            print("All currencies are GBP or GBp, no FX needed")

        print(
            f"DEBUG HOLDINGS: common_start = {common_start}, max_date = {max_date}, num_days = {len(dates)}"
        )
        print(
            f"DEBUG HOLDINGS: date range min {min([d.date() for d in dates])}, max {max([d.date() for d in dates])}"
        )

        # Simulate holdings for each ticker
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
            print(f"DEBUG HOLDINGS [{ticker}]: initial_qty = {initial_qty}")
            if ticker_trades.empty:
                print(
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
            print(
                f"DEBUG HOLDINGS [{ticker}]: Number of unique trade dates: {len(daily_adj)}, total adj: {daily_adj.sum()}"
            )
            if len(daily_adj) > 0:
                print(
                    f"DEBUG HOLDINGS [{ticker}]: Sample daily adj: {dict(list(daily_adj.items())[:3])}"
                )

            # Build daily holdings by iterating through dates and accumulating (carries forward previous holding)
            daily_holdings = pd.Series(index=dates, dtype=float)
            cum_qty = initial_qty
            for i, date in enumerate(dates):
                adj_today = daily_adj.get(date.date(), 0.0)
                if abs(adj_today) > 1e-6:  # Log only significant adjustments
                    print(
                        f"DEBUG HOLDINGS [{ticker}]: On {date.date()} (idx {i}), adj={adj_today:.6f}, cum before={cum_qty:.6f}"
                    )
                cum_qty += adj_today
                daily_holdings.iloc[i] = cum_qty
                if abs(adj_today) > 1e-6:
                    print(f"DEBUG HOLDINGS [{ticker}]: Updated to cum={cum_qty:.6f}")

            # No NaNs or ffill needed; all days explicitly set with carry-forward
            print(
                f"DEBUG HOLDINGS [{ticker}]: Final: NaNs={daily_holdings.isna().sum()}, min={daily_holdings.min():.6f}, max={daily_holdings.max():.6f}"
            )
            non_zero_count = (
                abs(daily_holdings) > 1e-6
            ).sum()  # Ignore floating-point zeros
            print(
                f"DEBUG HOLDINGS [{ticker}]: non-zero days: {non_zero_count} / {len(daily_holdings)}, max holding: {daily_holdings.max() if non_zero_count > 0 else 0}"
            )
            holdings_data[ticker] = daily_holdings.values

        # Fetch/cache prices
        price_data = {}
        for ticker in unique_tickers:
            # Check cached prices for the range
            cached_query = f"""
                SELECT date, close FROM prices
                WHERE ticker = '{ticker}' AND date BETWEEN '{common_start}' AND '{max_date}'
                ORDER BY date
            """
            cached = pd.read_sql_query(cached_query, conn)
            if not cached.empty:
                cached["date"] = pd.to_datetime(cached["date"]).dt.date
                cached.set_index("date", inplace=True)
                date_list = [d.date() for d in dates]
                cached_prices = (
                    cached.reindex(date_list).ffill()["close"].fillna(0.0).values
                )
                print(
                    f"Cached prices for {ticker}: non-zero count = {np.count_nonzero(np.array(cached_prices) > 0)} out of {len(cached_prices)}"
                )
            else:
                cached_prices = np.zeros(len(dates))
                print(f"No cached prices for {ticker}")

            # If any missing (0.0), fetch
            if np.any(np.array(cached_prices) == 0.0):
                print(
                    f"Fetching history for {ticker} from {common_start} to {max_date}"
                )
                try:
                    yf_ticker = yf.Ticker(ticker)
                    hist = yf_ticker.history(
                        start=common_start, end=max_date + pd.Timedelta(days=1)
                    )
                    print(f"Hist length for {ticker}: {len(hist)}")
                    print(
                        f"Hist date range for {ticker}: {hist.index.min()} to {hist.index.max()}"
                    )
                    if len(hist) > 0:
                        print(
                            f"Sample hist dates for {ticker}: {hist.index[:3].tolist() if len(hist) >= 3 else hist.index.tolist()}"
                        )
                        # Remove timezone from hist DataFrame to match naive dates
                        try:
                            if hist.index.tz is not None:
                                hist.index = hist.index.tz_localize(None)
                        except (AttributeError, TypeError):
                            pass

                        # Always use Close price
                        prices = hist["Close"].reindex(dates).ffill().fillna(0.0)

                        # Convert to GBP: for .L tickers, handle reported currency specially
                        reported = reported_currencies.get(ticker, "GBP")
                        converted_prices = prices.copy()
                        is_lse = ticker.endswith('.L')
                        if is_lse:
                            if reported == "GBp":
                                converted_prices = prices / 100.0
                                print(f"Converted {ticker} (LSE, GBp) to GBP (divided by 100)")
                            elif reported == "USD":
                                # Convert USD to GBP even for LSE
                                fx_info = fx_data.get("USD")
                                if fx_info:
                                    rates_series = pd.Series(fx_info["rates"], index=dates)
                                    converted_prices = prices / rates_series.reindex(dates).ffill().fillna(1.0)
                                    print(f"Converted {ticker} (LSE, reported USD) to GBP using FX rates")
                                else:
                                    print(f"No FX for USD on {ticker}, assuming already in GBP")
                            else:
                                print(f"{ticker} (LSE, {reported}) assumed in GBP, no change")
                        else:
                            # Non-LSE: use reported
                            if reported == "GBp":
                                converted_prices = prices / 100.0
                                print(f"Converted {ticker} GBp to GBP (divided by 100)")
                            elif reported == "USD":
                                fx_info = fx_data.get("USD")
                                if fx_info:
                                    rates_series = pd.Series(fx_info["rates"], index=dates)
                                    converted_prices = prices / rates_series.reindex(dates).ffill().fillna(1.0)
                                    print(f"Converted {ticker} USD to GBP using FX rates")
                            elif reported != "GBP":
                                # Handle other currencies if needed
                                fx_info = fx_data.get(reported)
                                if fx_info:
                                    rates_series = pd.Series(fx_info["rates"], index=dates)
                                    if fx_info["multiply"]:
                                        converted_prices = prices * rates_series.reindex(dates).ffill().fillna(1.0)
                                    else:
                                        converted_prices = prices / rates_series.reindex(dates).ffill().fillna(1.0)
                                    print(f"Converted {ticker} {reported} to GBP using FX rates")
                                else:
                                    print(f"No FX info for {reported} on {ticker}, assuming already in GBP")
                            else:
                                print(f"{ticker} ({reported}) assumed in GBP, no change")

                        # Replace any remaining NaN with 0
                        converted_prices = converted_prices.fillna(0.0)

                        non_zero = (converted_prices > 0).sum()
                        print(
                            f"Non-zero converted prices for {ticker}: {non_zero} out of {len(converted_prices)}"
                        )

                        # Sample comparison with converted prices
                        ticker_trades_sample = df[
                            (df["Ticker"] == ticker) & (df["Transaction Type"] == "Buy")
                        ].head(1)
                        if (
                            not ticker_trades_sample.empty
                            and len(ticker_trades_sample) > 0
                        ):
                            sample_trade_date = (
                                ticker_trades_sample["Trade Date/Time"].iloc[0].date()
                            )
                            sample_csv_price = ticker_trades_sample["Share Price"].iloc[
                                0
                            ]
                            valid_dates = [
                                d for d in dates if d.date() >= sample_trade_date
                            ]
                            if valid_dates:
                                closest_date = min(
                                    valid_dates,
                                    key=lambda x: abs(
                                        (x.date() - sample_trade_date).days
                                    ),
                                )
                                price_idx = list(dates).index(closest_date)
                                if isinstance(price_idx, int) and price_idx < len(
                                    converted_prices
                                ):
                                    yf_converted = converted_prices.iloc[price_idx]
                                    print(
                                        f"Sample comparison for {ticker} on ~{sample_trade_date}: CSV £{sample_csv_price:.2f} vs yf Converted GBP £{yf_converted:.2f}"
                                    )
                                else:
                                    print(
                                        f"Invalid index for sample comparison for {ticker}"
                                    )

                        # Cache converted GBP prices (only non-NaN >0)
                        for i in range(len(converted_prices)):
                            price_val = converted_prices.iloc[i]
                            if pd.notna(price_val) and price_val > 0:
                                date_str = dates[i].strftime("%Y-%m-%d")
                                conn.execute(
                                    "INSERT OR REPLACE INTO prices (ticker, date, close) VALUES (?, ?, ?)",
                                    (ticker, date_str, float(price_val)),
                                )
                        conn.commit()
                        price_data[ticker] = converted_prices.values
                    else:
                        print(f"No historical data returned for {ticker}")
                        price_data[ticker] = cached_prices
                except Exception as e:
                    print(f"Exception fetching {ticker}: {e}")
                    import traceback

                    traceback.print_exc()
                    price_data[ticker] = cached_prices
            else:
                print(f"All prices cached for {ticker}, using cache")
                # Assume cached are already converted GBP
                price_data[ticker] = cached_prices

        conn.close()

        # Ensure all price_data have no NaN and correct length
        for ticker in unique_tickers:
            arr = np.array(price_data[ticker])
            if len(arr) != len(dates):
                print(
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
        print(f"Error in get_portfolio_values: {e}")
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

        print(
            f"Tickers extracted and added to database for {len(unique_securities)} unique securities"
        )

    except Exception as ticker_err:
        print(f"Ticker extraction error: {ticker_err}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
