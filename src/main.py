from fastapi import FastAPI, UploadFile, File, Request, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import io
from typing import List, Dict, Optional
import pandas as pd
import re
import requests
import plotly.express as px
import sqlite3

from .merge_csv import merge_csv_files  # Import the merging function

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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if data.get('quotes'):
            # Prefer LSE exchange or .L suffix, and ETF/Equity types
            lse_candidates = [
                q for q in data['quotes']
                if (q.get('exchange') == 'LSE' or q.get('symbol', '').endswith('.L'))
                and q.get('quoteType') in ['ETF', 'EQUITY']
            ]
            if lse_candidates:
                # Sort by relevance (e.g., name similarity), take first
                quote = lse_candidates[0]
            else:
                # Fallback to first valid ETF/Equity
                valid_quotes = [q for q in data['quotes'] if q.get('quoteType') in ['ETF', 'EQUITY']]
                if valid_quotes:
                    quote = valid_quotes[0]
                else:
                    quote = data['quotes'][0]
            
            symbol = quote.get('symbol')
            if symbol and symbol != isin:
                # Optional: Verify if possible (yfinance doesn't easily provide ISIN, so assume match)
                return symbol
        
        # Fallback: Search with ISIN
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={isin}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if data.get('quotes'):
            lse_candidates = [q for q in data['quotes'] if q.get('exchange') == 'LSE' or q.get('symbol', '').endswith('.L')]
            if lse_candidates:
                quote = lse_candidates[0]
            else:
                valid_quotes = [q for q in data['quotes'] if q.get('quoteType') in ['ETF', 'EQUITY', 'MUTUALFUND', 'CURRENCY']]
                if valid_quotes:
                    quote = valid_quotes[0]
                else:
                    quote = data['quotes'][0]
            
            symbol = quote.get('symbol')
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
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
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

        # Add background task for ticker extraction and update database
        def extract_tickers_to_db():
            try:
                conn = sqlite3.connect(db_path)
                df = pd.read_sql_query("SELECT * FROM trades", conn)
                conn.close()
                
                def extract_security_and_isin(text: str) -> tuple:
                    match = re.search(r'(.*?) / ISIN ([A-Z]{2}[A-Z0-9]{9}[0-9])', str(text))
                    if match:
                        return match.group(1).strip(), match.group(2)
                    return str(text).strip(), None
                
                unique_securities = df['Security / ISIN'].apply(extract_security_and_isin).drop_duplicates()
                unique_securities = [s for s in unique_securities if s[1] is not None]  # Only those with ISIN
                
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
                
                df['Ticker'] = df['Security / ISIN'].apply(get_ticker)
                
                # Update the database with the new Ticker column
                conn = sqlite3.connect(db_path)
                df.to_sql("trades", conn, if_exists="replace", index=False)
                conn.close()
                
                print(f"Background: Tickers extracted and added to database for {len(unique_securities)} unique securities")
                
            except Exception as ticker_err:
                print(f"Background ticker extraction error: {ticker_err}")
                import traceback
                traceback.print_exc()
        
        background_tasks.add_task(extract_tickers_to_db)

        print(f"DEBUG: Merged DF shape: {merged_df.shape}, Date range: {merged_df['Trade Date/Time'].min()} to {merged_df['Trade Date/Time'].max()}")
        print(f"DEBUG: Sample Transaction Types: {merged_df['Transaction Type'].unique()}")

        try:
            # Compute monthly net contributions and cumulative value
            merged_df["Month"] = merged_df["Trade Date/Time"].dt.to_period("M")
            merged_df["Net_Value"] = merged_df.apply(
                lambda row: row["Total Trade Value"]
                if row["Transaction Type"] == "Buy"
                else -row["Total Trade Value"],
                axis=1,
            )
            monthly_net = merged_df.groupby("Month")["Net_Value"].sum().reset_index()
            monthly_net["Month"] = monthly_net["Month"].astype(str)
            cumulative = monthly_net["Net_Value"].cumsum()

            print(f"DEBUG: Monthly net shape: {monthly_net.shape}")
            print(f"DEBUG: Monthly net sample: {monthly_net.head().to_dict()}")
            print(f"DEBUG: Cumulative sample: {cumulative.head().to_dict()}")

            # Prepare data for client-side charts
            monthly_net_data = monthly_net.to_dict("records")
            cumulative_data = cumulative.tolist()

            print(f"DEBUG: monthly_net_data length: {len(monthly_net_data)}, cumulative_data length: {len(cumulative_data)}")

        except Exception as compute_err:
            print(f"DEBUG: Error in computation/graph gen: {compute_err}")
            import traceback
            traceback.print_exc()
            monthly_net_data = []
            cumulative_data = []

        # Success JSON
        min_date = merged_df["Trade Date/Time"].min()
        max_date = merged_df["Trade Date/Time"].max()
        return JSONResponse(
            content={
                "success": True,
                "total_transactions": len(merged_df),
                "min_date": str(min_date),
                "max_date": str(max_date),
                "monthly_net": monthly_net_data,
                "cumulative": cumulative_data,
            }
        )
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500,
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
