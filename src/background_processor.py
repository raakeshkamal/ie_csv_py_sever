"""
Background processor module for precomputing yfinance data.
Handles asynchronous collection of ticker prices, FX rates, and portfolio values.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd

from .database import get_connection
from .prices import fetch_prices_parallel, get_currencies_parallel, get_common_start_date, get_needed_currencies
from .portfolio import compute_monthly_net_contributions, simulate_holdings, compute_daily_portfolio_values, get_valid_unique_tickers
from datetime import datetime

logger = logging.getLogger(__name__)


def create_precomputed_tables(conn=None, db_path: Optional[str] = None):
    """Create tables for precomputed portfolio data and metadata."""
    from .database import get_connection
    close_conn = False
    if conn is None:
        conn = get_connection(db_path)
        close_conn = True
    try:
        # Precomputed portfolio values table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS precomputed_portfolio_values (
                date DATE PRIMARY KEY,
                daily_value REAL,
                last_updated TIMESTAMP
            )
        """)

        # Precomputed monthly contributions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS precomputed_monthly_contributions (
                month TEXT PRIMARY KEY,
                net_value REAL,
                last_updated TIMESTAMP
            )
        """)

        # Metadata table to track completion status
        conn.execute("""
            CREATE TABLE IF NOT EXISTS precompute_status (
                id INTEGER PRIMARY KEY,
                status TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                total_tickers INTEGER,
                last_error TEXT
            )
        """)

        conn.commit()
        logger.info("Precomputed tables created successfully")
    finally:
        if close_conn:
            conn.close()


def precompute_portfolio_data(df, conn=None, db_path: Optional[str] = None) -> bool:
    """
    Precompute portfolio values and monthly contributions in the background.
    Returns True if successful, False otherwise.
    """
    from .database import get_connection
    close_conn = False
    if conn is None:
        conn = get_connection(db_path)
        close_conn = True

    # Insert status record
    status_id = None
    try:
        cursor = conn.execute(
            "INSERT INTO precompute_status (status, started_at) VALUES (?, ?)",
            ("in_progress", datetime.now())
        )
        status_id = cursor.lastrowid
        conn.commit()

        # Ensure tables exist
        create_precomputed_tables(conn, db_path)

        # Prepare data
        df = df.copy()
        df["Trade Date/Time"] = pd.to_datetime(
            df["Trade Date/Time"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
        df = df.dropna(subset=["Trade Date/Time"])

        # Get unique valid tickers
        unique_tickers = get_valid_unique_tickers(df)

        if not unique_tickers:
            logger.error("No valid tickers found in trades")
            conn.execute(
                "UPDATE precompute_status SET status = ?, last_error = ? WHERE id = ?",
                ("failed", "No valid tickers found", status_id)
            )
            conn.commit()
            return False

        logger.info(f"Precomputing data for {len(unique_tickers)} tickers...")

        # Update status with ticker count
        conn.execute(
            "UPDATE precompute_status SET total_tickers = ? WHERE id = ?",
            (len(unique_tickers), status_id)
        )
        conn.commit()

        # Get date range
        common_start = get_common_start_date(unique_tickers)
        max_date = df["Trade Date/Time"].max().date()

        if not common_start or common_start > max_date:
            logger.error("No valid date range")
            conn.execute(
                "UPDATE precompute_status SET status = ?, last_error = ? WHERE id = ?",
                ("failed", "Invalid date range", status_id)
            )
            conn.commit()
            return False

        dates = pd.date_range(start=common_start, end=max_date, freq="D")

        # Get currencies
        currencies = get_currencies_parallel(unique_tickers)

        # Fetch all price data in one go
        price_data = fetch_prices_parallel(
            unique_tickers, common_start, max_date, currencies, dates
        )

        # Simulate holdings
        holdings_data, initial_holdings = simulate_holdings(df, unique_tickers, common_start, max_date)

        # Compute daily portfolio values
        daily_values = compute_daily_portfolio_values(holdings_data, price_data, dates)

        # Save precomputed daily values
        conn.execute("DELETE FROM precomputed_portfolio_values")
        for i in range(len(dates)):
            conn.execute(
                "INSERT OR REPLACE INTO precomputed_portfolio_values (date, daily_value, last_updated) VALUES (?, ?, ?)",
                (dates[i].strftime("%Y-%m-%d"), float(daily_values[i]), datetime.now())
            )

        # Compute and save monthly net contributions
        monthly_net = compute_monthly_net_contributions(df)
        conn.execute("DELETE FROM precomputed_monthly_contributions")
        for item in monthly_net:
            conn.execute(
                "INSERT OR REPLACE INTO precomputed_monthly_contributions (month, net_value, last_updated) VALUES (?, ?, ?)",
                (item["Month"], float(item["Net_Value"]), datetime.now())
            )

        # Update status to completed
        conn.execute(
            "UPDATE precompute_status SET status = ?, completed_at = ? WHERE id = ?",
            ("completed", datetime.now(), status_id)
        )
        conn.commit()

        logger.info(f"Successfully precomputed {len(daily_values)} daily values and {len(monthly_net)} monthly contributions")
        return True

    except Exception as e:
        logger.error(f"Error in precompute_portfolio_data: {e}", exc_info=True)
        if status_id and conn:
            conn.execute(
                "UPDATE precompute_status SET status = ?, last_error = ? WHERE id = ?",
                ("failed", str(e), status_id)
            )
            conn.commit()
        return False

    finally:
        if close_conn and conn:
            conn.close()


def get_precomputed_portfolio_data(db_path: Optional[str] = None) -> Optional[Dict]:
    """
    Retrieve precomputed portfolio data from database.
    Returns dict with monthly_net and daily values, or None if not available.
    """
    from .database import get_connection
    conn = get_connection(db_path)
    try:
        # Check if data exists and is fresh (within last 24 hours)
        cursor = conn.execute("""
            SELECT COUNT(*) FROM precomputed_portfolio_values
            WHERE last_updated > datetime('now', '-1 day')
        """)
        has_recent_data = cursor.fetchone()[0] > 0

        if not has_recent_data:
            logger.warning("No recent precomputed data found")
            return None

        # Get daily values
        daily_df = pd.read_sql_query(
            "SELECT date, daily_value FROM precomputed_portfolio_values ORDER BY date",
            conn
        )
        if daily_df.empty:
            return None

        daily_df["date"] = pd.to_datetime(daily_df["date"])
        daily_dates = [d.strftime("%Y-%m-%d") for d in daily_df["date"]]
        daily_values = daily_df["daily_value"].tolist()

        # Get monthly contributions
        monthly_df = pd.read_sql_query(
            "SELECT month, net_value FROM precomputed_monthly_contributions ORDER BY month",
            conn
        )
        monthly_net = []
        if not monthly_df.empty:
            monthly_net = [{"Month": row["month"], "Net_Value": float(row["net_value"])}
                          for _, row in monthly_df.iterrows()]

        return {
            "monthly_net": monthly_net,
            "daily_dates": daily_dates,
            "daily_values": daily_values,
        }

    except Exception as e:
        logger.error(f"Error retrieving precomputed data: {e}")
        return None

    finally:
        conn.close()


def get_precompute_status(db_path: Optional[str] = None) -> Dict:
    """
    Get the current precomputation status.
    Returns dict with status information.
    """
    from .database import get_connection
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("""
            SELECT status, started_at, completed_at, total_tickers, last_error
            FROM precompute_status
            ORDER BY started_at DESC
            LIMIT 1
        """)
        row = cursor.fetchone()

        if row:
            return {
                "status": row[0],
                "started_at": row[1],
                "completed_at": row[2],
                "total_tickers": row[3],
                "last_error": row[4],
                "has_data": True
            }
        else:
            return {"status": "not_started", "has_data": False}

    except Exception as e:
        logger.error(f"Error getting precompute status: {e}")
        return {"status": "error", "error": str(e), "has_data": False}

    finally:
        conn.close()


def export_precomputed_data(db_path: Optional[str] = None):
    """
    Export all precomputed data as JSON.
    Returns dict with portfolio values, monthly contributions, and status.
    """
    from .database import get_connection
    import json

    conn = get_connection(db_path)
    try:
        # Get portfolio values
        portfolio_df = pd.read_sql_query(
            "SELECT date, daily_value, last_updated FROM precomputed_portfolio_values ORDER BY date",
            conn
        )

        # Get monthly contributions
        monthly_df = pd.read_sql_query(
            "SELECT month, net_value, last_updated FROM precomputed_monthly_contributions ORDER BY month",
            conn
        )

        # Get status
        status = get_precompute_status(db_path)

        return {
            "portfolio_values": portfolio_df.to_dict("records"),
            "monthly_contributions": monthly_df.to_dict("records"),
            "status": status,
            "count": {
                "portfolio_values": len(portfolio_df),
                "monthly_contributions": len(monthly_df)
            }
        }

    except Exception as e:
        logger.error(f"Error exporting precomputed data: {e}")
        return {"error": str(e)}

    finally:
        conn.close()
