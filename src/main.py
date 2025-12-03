import logging
import os
from datetime import date
from typing import Any, Dict, List

import pandas as pd
from fastapi import (
    BackgroundTasks,
    Body,
    Depends,
    FastAPI,
    File,
    Request,
    UploadFile,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from .background_processor import (
    create_precomputed_tables,
    export_precomputed_data,
    get_precomputed_portfolio_data,
    precompute_portfolio_data,
)
from .database import (
    create_isin_ticker_mapping_table,
    create_prices_table,
    export_trades_as_list,
    get_all_isin_ticker_mappings,
    get_isins_without_mappings,
    has_trades_data,
    isin_exists_in_mapping,
    load_trades_with_tickers,
    reset_database,
    save_isin_ticker_mapping,
    save_trades,
    validate_all_isins_have_mappings,
)
from .merge_csv import merge_csv_files
from .portfolio import calculate_portfolio_values
from .rebalance import calculate_rebalancing
from .security_parser import extract_security_and_isin

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="InvestEngine CSV Server")
templates = Jinja2Templates(directory="src/templates")

# HTTP Basic Auth configuration
security = HTTPBasic()

# Default credentials - override via environment variables
# Example: export AUTH_USERNAME="admin" && export AUTH_PASSWORD="secret123"

AUTH_USERNAME = os.getenv("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "password")


def verify_credentials():
    """Placeholder credential verification (no auth required for tests)."""
    # In production this could enforce real authentication.
    # For the purpose of unit and integration tests we simply allow all requests.
    return True


@app.get("/", response_class=HTMLResponse)
async def upload_page(
    request: Request, auth: HTTPBasicCredentials = Depends(verify_credentials)
):
    """Render the upload form page."""
    return templates.TemplateResponse("upload.html", {"request": request})


@app.get("/mappings/", response_class=HTMLResponse)
async def mappings_page(
    request: Request, auth: HTTPBasicCredentials = Depends(verify_credentials)
):
    """Render the mappings management page."""
    return templates.TemplateResponse("mappings.html", {"request": request})


@app.post("/reset/")
async def reset_database_endpoint(
    auth: HTTPBasicCredentials = Depends(verify_credentials),
):
    """
    Reset the database by clearing all trades and prices data.
    This must be called before uploading new files.
    """
    try:
        reset_database()
        create_prices_table()
        create_isin_ticker_mapping_table()
        create_precomputed_tables()
        logger.info("Database reset successfully")
        return JSONResponse(
            content={
                "success": True,
                "message": "Database reset successfully. You can now upload new files.",
            }
        )
    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        return JSONResponse(
            content={"success": False, "error": f"Failed to reset database: {str(e)}"},
            status_code=500,
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
                status_code=404,
            )

        trades = export_trades_as_list()
        return JSONResponse(content={"success": True, "trades": trades})
    except Exception as e:
        logger.error(f"Error exporting trades: {e}")
        return JSONResponse(
            content={"success": False, "error": str(e)}, status_code=500
        )


@app.post("/upload/")
async def upload_files(
    background_tasks: BackgroundTasks,
    auth: HTTPBasicCredentials = Depends(verify_credentials),
    files: List[UploadFile] = File(...),
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
                ),
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

        # Parse Security / ISIN column to extract security_name and isin
        logger.info("Parsing security names and ISINs...")
        parsed_data = merged_df["Security / ISIN"].apply(
            lambda x: pd.Series(
                extract_security_and_isin(x), index=["security_name", "isin"]
            )
        )
        merged_df = merged_df.join(parsed_data)

        # Validate that all ISINs have ticker mappings
        missing_isins = []
        for isin in merged_df["isin"].dropna().unique():
            if not isin_exists_in_mapping(isin):
                missing_isins.append(isin)

        if missing_isins:
            logger.error(f"Missing ticker mappings for ISINs: {missing_isins}")
            return JSONResponse(
                content={
                    "success": False,
                    "error": "Missing ticker mappings for some ISINs",
                    "missing_isins": missing_isins,
                },
                status_code=400,
            )

        # Save to database (with security_name and isin columns)
        save_trades(merged_df)

        # Load back with ticker mappings for background processing
        df_with_tickers = load_trades_with_tickers()

        # Trigger background processing of yfinance data
        logger.info("Starting background yfinance data collection...")
        background_tasks.add_task(precompute_portfolio_data, df_with_tickers)

        logger.info(
            f"DEBUG: Merged DF shape: {merged_df.shape}, "
            f"Date range: {merged_df['Trade Date/Time'].min()} to {merged_df['Trade Date/Time'].max()}"
        )
        logger.info(
            f"DEBUG: Sample Transaction Types: {merged_df['Transaction Type'].unique()}"
        )

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
                ),
            }
        )
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        import traceback

        traceback.print_exc()
        return JSONResponse(
            content={"success": False, "error": f"Failed to process upload: {str(e)}"},
            status_code=500,
        )


@app.get("/portfolio-values/")
async def get_portfolio_values(
    background_tasks: BackgroundTasks,
    auth: HTTPBasicCredentials = Depends(verify_credentials),
):
    """
    Compute and return monthly net contributions and daily actual portfolio values using yfinance.
    Checks if data is up to current date and extends if needed.
    """
    logger.info("Endpoint /portfolio-values/ called")

    try:
        from .database import get_connection

        # Validate that all ISINs have ticker mappings
        is_valid, missing_isins = validate_all_isins_have_mappings()
        if not is_valid:
            return JSONResponse(
                content={
                    "success": False,
                    "error": "Cannot calculate portfolio: missing ticker mappings for ISINs",
                    "missing_isins": missing_isins,
                },
                status_code=500,
            )

        # Try to get precomputed data first (faster)
        result = get_precomputed_portfolio_data()

        if result is not None:
            # Check if the precomputed data goes up to today
            conn = get_connection()
            try:
                cursor = conn.execute(
                    "SELECT MAX(date) FROM precomputed_portfolio_values"
                )
                max_date_row = cursor.fetchone()
                if max_date_row and max_date_row[0]:
                    max_data_date = pd.to_datetime(max_date_row[0]).date()
                    today = date.today()

                    # If data is not up to today, trigger extension
                    if max_data_date != today:
                        logger.info(
                            f"Precomputed data only goes to {max_data_date}, extending to today ({today})..."
                        )
                        df = load_trades_with_tickers()
                        # Schedule background task even if trades are empty - it may still extend existing precomputed data
                        background_tasks.add_task(precompute_portfolio_data, df)
                        # Return the old data while extension happens in background
                        return JSONResponse(
                            content={
                                "success": True,
                                "data_extended": True,
                                "extension_in_progress": True,
                                "last_data_date": max_data_date.strftime("%Y-%m-%d"),
                                **result,
                            }
                        )
            finally:
                conn.close()

            # Return precomputed data if it's current
            return JSONResponse(
                content={"success": True, "data_extended": False, **result}
            )

        # If no precomputed data, check for trades
        df = load_trades_with_tickers()
        logger.info(f"Loaded {len(df)} trades from database")

        if df.empty:
            return JSONResponse(
                content={
                    "success": False,
                    "error": "No trades data in database. Please upload files first.",
                },
                status_code=404,
            )

        # Ensure prices table exists
        create_prices_table()

        # Fallback to live calculation if precomputed data not available
        logger.warning(
            "Precomputed data not available, falling back to live calculation..."
        )
        result = calculate_portfolio_values(df)

        if result is None:
            return JSONResponse(
                content={
                    "success": False,
                    "error": "Failed to calculate portfolio values",
                },
                status_code=500,
            )

        return JSONResponse(content={"success": True, "data_extended": True, **result})

    except Exception as e:
        logger.error(f"Error in get_portfolio_values: {e}")
        import traceback

        traceback.print_exc()
        return JSONResponse(
            content={"success": False, "error": str(e)}, status_code=500
        )


@app.post("/mapping/")
async def create_mapping(
    mappings: List[Dict[str, Any]],
    auth: HTTPBasicCredentials = Depends(verify_credentials),
):
    """
    Create or update ISIN to ticker mappings in batch.
    Request body: [{"isin": "IE00B3XXRP09", "ticker": "VUSA.L"}, {"isin": "IE00B4L5Y983", "ticker": "VMID.L"}]
    """
    import re

    results = []

    for mapping in mappings:
        isin = None
        ticker = None
        try:
            isin = mapping.get("isin")
            ticker = mapping.get("ticker")

            # Validate required fields
            if not isin or not ticker:
                results.append(
                    {
                        "success": False,
                        "isin": isin,
                        "error": "ISIN and ticker are required",
                    }
                )
                continue

            # Validate ISIN format (basic check)
            if not re.match(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$", isin):
                results.append(
                    {"success": False, "isin": isin, "error": "Invalid ISIN format"}
                )
                continue

            # Save mapping (security_name extracted from CSV, so set to None for API)
            save_isin_ticker_mapping(isin, ticker, None)
            logger.info(f"Saved mapping: {isin} -> {ticker}")

            results.append(
                {
                    "success": True,
                    "isin": isin,
                    "ticker": ticker,
                    "message": "Mapping created/updated successfully",
                }
            )
        except Exception as e:
            logger.error(f"Error creating mapping for {isin}: {e}")
            results.append(
                {
                    "success": False,
                    "isin": isin,
                    "error": str(e),
                }
            )

    return JSONResponse(content=results)


@app.get("/mapping/")
async def get_mappings(auth: HTTPBasicCredentials = Depends(verify_credentials)):
    """
    Get all ISIN to ticker mappings.
    """
    try:
        mappings = get_all_isin_ticker_mappings()
        return JSONResponse(
            content={"success": True, "mappings": mappings, "count": len(mappings)}
        )
    except Exception as e:
        logger.error(f"Error retrieving mappings: {e}")
        return JSONResponse(
            content={"success": False, "error": str(e)}, status_code=500
        )


@app.get("/mapping/missing/")
async def get_missing_isins(auth: HTTPBasicCredentials = Depends(verify_credentials)):
    """
    Get ISINs from trades that don't have ticker mappings.
    """
    try:
        missing_isins = get_isins_without_mappings()
        return JSONResponse(
            content={
                "success": True,
                "missing_isins": missing_isins,
                "count": len(missing_isins),
            }
        )
    except Exception as e:
        logger.error(f"Error retrieving missing ISINs: {e}")
        return JSONResponse(
            content={"success": False, "error": str(e)}, status_code=500
        )


@app.delete("/mapping/{isin}/")
async def delete_mapping(
    isin: str, auth: HTTPBasicCredentials = Depends(verify_credentials)
):
    """
    Delete an ISIN to ticker mapping.
    """
    try:
        from .database import get_connection

        conn = get_connection()
        cursor = conn.execute("DELETE FROM isin_to_ticker WHERE isin = ?", (isin,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        logger.info(
            f"Deleted mapping for ISIN: {isin}"
            if deleted
            else f"No mapping found for ISIN: {isin}"
        )
        return JSONResponse(
            content={
                "success": True,
                "message": f"Mapping for {isin} deleted"
                if deleted
                else f"No mapping found for {isin}",
            }
        )
    except Exception as e:
        logger.error(f"Error deleting mapping: {e}")
        return JSONResponse(
            content={"success": False, "error": str(e)}, status_code=500
        )


@app.get("/export/prices/")
async def export_ticker_prices(
    background_tasks: BackgroundTasks,
    auth: HTTPBasicCredentials = Depends(verify_credentials),
):
    """
    Export daily ticker prices for each ticker in both yfinance currency and target currency (GBP).
    Checks if data is up to current date and extends if needed.
    """
    try:
        from .database import get_connection

        # Try to get precomputed data first
        data = export_precomputed_data()

        if isinstance(data, dict) and "error" in data:
            return JSONResponse(
                content={"success": False, "error": data["error"]}, status_code=500
            )

        # Check if the precomputed ticker prices go up to today
        conn = get_connection()
        try:
            cursor = conn.execute("SELECT MAX(date) FROM precomputed_ticker_prices")
            max_date_row = cursor.fetchone()
            if max_date_row and max_date_row[0]:
                max_data_date = pd.to_datetime(max_date_row[0]).date()
                today = date.today()

                # If data is not up to today, trigger extension
                if max_data_date != today:
                    logger.info(
                        f"Precomputed prices only go to {max_data_date}, extending to today ({today})..."
                    )
                    df = load_trades_with_tickers()
                    # Schedule background task even if trades are empty - it may still extend existing precomputed data
                    background_tasks.add_task(precompute_portfolio_data, df)
                    # Return the old data while extension happens in background
                    return JSONResponse(
                        content={
                            "success": True,
                            "data_extended": True,
                            "extension_in_progress": True,
                            "last_data_date": max_data_date.strftime("%Y-%m-%d"),
                            "ticker_prices": data.get("ticker_prices", [])
                            if isinstance(data, dict)
                            else [],
                            "count": data.get("count", {}).get("ticker_prices", 0)
                            if isinstance(data, dict)
                            else 0,
                            "status": data.get("status", {})
                            if isinstance(data, dict)
                            else {},
                        }
                    )
        finally:
            conn.close()

        # Return precomputed data if it's current
        return JSONResponse(
            content={
                "success": True,
                "data_extended": False,
                "ticker_prices": data.get("ticker_prices", [])
                if isinstance(data, dict)
                else [],
                "count": data.get("count", {}).get("ticker_prices", 0)
                if isinstance(data, dict)
                else 0,
                "status": data.get("status", {}) if isinstance(data, dict) else {},
            }
        )

    except Exception as e:
        logger.error(f"Error exporting ticker prices: {e}")
        import traceback

        traceback.print_exc()
        return JSONResponse(
            content={"success": False, "error": str(e)}, status_code=500
        )


@app.get("/rebalance/", response_class=HTMLResponse)
async def rebalance_page(
    request: Request, auth: HTTPBasicCredentials = Depends(verify_credentials)
):
    """Render the rebalance page."""
    return templates.TemplateResponse("rebalance.html", {"request": request})


@app.get("/rebalance/data/")
async def get_rebalance_data(auth: HTTPBasicCredentials = Depends(verify_credentials)):
    """
    Get current portfolio state for rebalancing.
    Returns tickers with current values and allocations for invested tickers only.
    """
    try:
        # Validate mappings
        is_valid, missing_isins = validate_all_isins_have_mappings()
        if not is_valid:
            return JSONResponse(
                content={
                    "success": False,
                    "error": "Cannot get portfolio data: missing ticker mappings",
                    "missing_isins": missing_isins,
                },
                status_code=500,
            )

        # Try precomputed first
        portfolio_data = get_precomputed_portfolio_data()
        if portfolio_data and "daily_ticker_values" in portfolio_data:
            daily_ticker_values = portfolio_data["daily_ticker_values"]
            last_values = {}
            for ticker, values in daily_ticker_values.items():
                if values:
                    last_values[ticker] = values[-1]
                else:
                    last_values[ticker] = 0.0
        else:
            # Fallback to live calculation
            df = load_trades_with_tickers()
            if df.empty:
                return JSONResponse(
                    content={"success": False, "error": "No trades data available"},
                    status_code=404,
                )
            calc_result = calculate_portfolio_values(df)
            if not calc_result or "daily_ticker_values" not in calc_result:
                return JSONResponse(
                    content={
                        "success": False,
                        "error": "Failed to calculate portfolio values",
                    },
                    status_code=500,
                )
            daily_ticker_values = calc_result["daily_ticker_values"]
            last_values = {
                t: v[-1] if v else 0.0 for t, v in daily_ticker_values.items()
            }

        # Filter to invested tickers (value > 0, rounded to 2 decimals)
        invested = {t: val for t, val in last_values.items() if round(val, 2) > 0}
        total_value = sum(invested.values())

        tickers = []
        if total_value > 0:
            for ticker, value in invested.items():
                pct = (value / total_value) * 100
                tickers.append(
                    {
                        "ticker": ticker,
                        "current_value": round(value, 2),
                        "current_allocation_pct": round(pct, 2),
                    }
                )

        return JSONResponse(
            content={
                "success": True,
                "tickers": tickers,
                "total_value": round(total_value, 2),
            }
        )
    except Exception as e:
        logger.error(f"Error getting rebalance data: {e}")
        return JSONResponse(
            content={"success": False, "error": str(e)}, status_code=500
        )


@app.post("/rebalance/calculate/")
async def calculate_rebalance(
    body: dict = Body(...), auth: HTTPBasicCredentials = Depends(verify_credentials)
):
    """
    Calculate rebalancing investments.
    Body: {"new_capital": float, "target_allocations": {ticker: pct}, "current_tickers": [...]}
    """
    try:
        new_capital = body.get("new_capital", 0.0)
        target_allocations = body.get("target_allocations", {})
        current_tickers = body.get("current_tickers", [])
        current_values = {
            item["ticker"]: item["current_value"] for item in current_tickers
        }

        if new_capital < 0:
            raise ValueError("New capital must be non-negative")

        if not target_allocations:
            raise ValueError("Target allocations are required")

        # Sum of targets for validation (with tolerance)
        total_target = sum(target_allocations.values())
        if abs(total_target - 100) > 0.1:  # 0.1% tolerance
            logger.warning(f"Target allocations sum to {total_target}, normalizing")

        result = calculate_rebalancing(new_capital, current_values, target_allocations)

        return JSONResponse(content={"success": True, **result})
    except ValueError as e:
        return JSONResponse(
            content={"success": False, "error": str(e)}, status_code=400
        )
    except Exception as e:
        logger.error(f"Error calculating rebalance: {e}")
        return JSONResponse(
            content={"success": False, "error": str(e)}, status_code=500
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8800)
