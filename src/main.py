import logging
from typing import List, Dict, Any

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Request, UploadFile, HTTPException, Depends, Security
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from .merge_csv import merge_csv_files
from .database import (
    has_trades_data,
    reset_database,
    save_trades,
    export_trades_as_list,
    create_prices_table,
    load_trades,
)
from .tickers import add_tickers_to_df
from .portfolio import calculate_portfolio_values
from .background_processor import precompute_portfolio_data, get_precomputed_portfolio_data, export_precomputed_data, create_precomputed_tables

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="InvestEngine CSV Server")
templates = Jinja2Templates(directory="src/templates")

# HTTP Basic Auth configuration
security = HTTPBasic()

# Default credentials - override via environment variables
# Example: export AUTH_USERNAME="admin" && export AUTH_PASSWORD="secret123"
import os
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "password")


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Verify HTTP Basic Auth credentials.
    """
    correct_username = AUTH_USERNAME == credentials.username
    correct_password = AUTH_PASSWORD == credentials.password

    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials


@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request, auth: HTTPBasicCredentials = Depends(verify_credentials)):
    """Render the upload form page."""
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/reset/")
async def reset_database_endpoint(auth: HTTPBasicCredentials = Depends(verify_credentials)):
    """
    Reset the database by clearing all trades and prices data.
    This must be called before uploading new files.
    """
    try:
        reset_database()
        create_prices_table()
        create_precomputed_tables()
        logger.info("Database reset successfully")
        return JSONResponse(
            content={
                "success": True,
                "message": "Database reset successfully. You can now upload new files."
            }
        )
    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        return JSONResponse(
            content={"success": False, "error": f"Failed to reset database: {str(e)}"},
            status_code=500
        )


@app.get("/export/trades/")
async def export_trades(auth: HTTPBasicCredentials = Depends(verify_credentials)):
    """
    Export the entire trades table as JSON.
    Returns all trade data in the database.
    """
    try:
        if not has_trades_data():
            return JSONResponse(
                content={"success": False, "error": "No trades data in database"},
                status_code=404
            )

        trades = export_trades_as_list()
        return JSONResponse(content={"success": True, "trades": trades})
    except Exception as e:
        logger.error(f"Error exporting trades: {e}")
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )


@app.post("/upload/")
async def upload_files(
    background_tasks: BackgroundTasks,
    auth: HTTPBasicCredentials = Depends(verify_credentials),
    files: List[UploadFile] = File(...)
):
    """
    Handle batch CSV upload, merge them, and save to database (overwrites old data).
    Requires calling /reset/ first to ensure data consistency.
    """
    logger.info("Endpoint /upload/ called")

    # Check if database has existing data and hasn't been reset
    if has_trades_data():
        return JSONResponse(
            content={
                "success": False,
                "error": (
                    "Database contains existing data. "
                    "Please call /reset/ first before uploading new files to ensure data consistency. "
                    "This prevents accidental mixing of data from different batches."
                )
            },
            status_code=400,
        )

    if not files:
        return JSONResponse(
            content={"success": False, "error": "No files uploaded"},
            status_code=400,
        )

    file_data = []
    for file in files:
        if file.filename and not file.filename.endswith(".csv"):
            continue
        content = await file.read()
        content_str = content.decode("utf-8")
        file_data.append((file.filename, content_str))

    if not file_data:
        return JSONResponse(
            content={"success": False, "error": "No valid CSV files"},
            status_code=400,
        )

    try:
        # Merge CSV files
        merged_df = merge_csv_files(file_data)

        # Save to database
        save_trades(merged_df)

        # Extract tickers
        logger.info("Extracting tickers...")
        df_with_tickers = add_tickers_to_df(merged_df)
        save_trades(df_with_tickers)

        # Trigger background processing of yfinance data
        logger.info("Starting background yfinance data collection...")
        background_tasks.add_task(precompute_portfolio_data, df_with_tickers)

        logger.info(
            f"DEBUG: Merged DF shape: {merged_df.shape}, "
            f"Date range: {merged_df['Trade Date/Time'].min()} to {merged_df['Trade Date/Time'].max()}"
        )
        logger.info(f"DEBUG: Sample Transaction Types: {merged_df['Transaction Type'].unique()}")

        # Success response
        min_date = merged_df["Trade Date/Time"].min()
        max_date = merged_df["Trade Date/Time"].max()

        return JSONResponse(
            content={
                "success": True,
                "total_transactions": len(merged_df),
                "min_date": str(min_date),
                "max_date": str(max_date),
                "message": (
                    f"Successfully uploaded {len(merged_df)} transactions. "
                    f"Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
                )
            }
        )
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"success": False, "error": f"Failed to process upload: {str(e)}"},
            status_code=500
        )


@app.get("/portfolio-values/")
async def get_portfolio_values(auth: HTTPBasicCredentials = Depends(verify_credentials)):
    """
    Compute and return monthly net contributions and daily actual portfolio values using yfinance.
    """
    logger.info("Endpoint /portfolio-values/ called")

    try:
        # Try to get precomputed data first (faster)
        result = get_precomputed_portfolio_data()

        if result is not None:
            # Return precomputed data if available
            return JSONResponse(
                content={
                    "success": True,
                    **result
                }
            )

        # If no precomputed data, check for trades
        df = load_trades()
        logger.info(f"Loaded {len(df)} trades from database")

        if df.empty:
            return JSONResponse(
                content={
                    "success": False,
                    "error": "No trades data in database. Please upload files first."
                },
                status_code=404,
            )

        # Ensure prices table exists
        create_prices_table()

        # Fallback to live calculation if precomputed data not available
        logger.warning("Precomputed data not available, falling back to live calculation...")
        result = calculate_portfolio_values(df)

        if result is None:
            return JSONResponse(
                content={
                    "success": False,
                    "error": "Failed to calculate portfolio values"
                },
                status_code=500,
            )

        return JSONResponse(
            content={
                "success": True,
                **result
            }
        )

    except Exception as e:
        logger.error(f"Error in get_portfolio_values: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )


@app.get("/export/prices/")
async def export_ticker_prices(auth: HTTPBasicCredentials = Depends(verify_credentials)):
    """
    Export daily ticker prices for each ticker in both yfinance currency and target currency (GBP).
    Returns ticker prices data in JSON format.
    """
    try:
        data = export_precomputed_data()

        if "error" in data:
            return JSONResponse(
                content={"success": False, "error": data["error"]},
                status_code=500
            )

        # Return only ticker prices data
        return JSONResponse(content={
            "success": True,
            "ticker_prices": data.get("ticker_prices", []),
            "count": data.get("count", {}).get("ticker_prices", 0),
            "status": data.get("status", {})
        })

    except Exception as e:
        logger.error(f"Error exporting ticker prices: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
