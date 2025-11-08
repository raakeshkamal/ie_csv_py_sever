"""
Database operations module for InvestEngine CSV Server.
Provides database connection management and CRUD operations.
"""

import sqlite3
from typing import Optional, List, Dict, Any
from pathlib import Path
import pandas as pd

DB_PATH = "db/merged_trading.db"


def get_db_path() -> str:
    """Return the database path."""
    return DB_PATH


def ensure_db_directory():
    """Ensure the database directory exists."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Get a database connection."""
    return sqlite3.connect(DB_PATH)


def reset_database():
    """Reset the database by dropping trades and prices tables."""
    ensure_db_directory()
    conn = get_connection()
    try:
        conn.execute("DROP TABLE IF EXISTS trades")
        conn.execute("DROP TABLE IF EXISTS prices")
        conn.commit()
    finally:
        conn.close()


def table_exists(table_name: str) -> bool:
    """Check if a table exists in the database."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return cursor.fetchone() is not None
    finally:
        conn.close()


def has_trades_data() -> bool:
    """Check if the trades table exists and has data."""
    if not table_exists("trades"):
        return False
    conn = get_connection()
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM trades")
        count = cursor.fetchone()[0]
        return count > 0
    finally:
        conn.close()


def create_prices_table():
    """Create the prices table if it doesn't exist."""
    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                ticker TEXT,
                date DATE,
                close REAL,
                PRIMARY KEY (ticker, date)
            )
        """)
        conn.commit()
    finally:
        conn.close()


def save_trades(df, conn: Optional[sqlite3.Connection] = None):
    """Save trades DataFrame to database."""
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        df.to_sql("trades", conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        if close_conn:
            conn.close()


def save_ticker_column(df):
    """Update existing trades table with Ticker column."""
    conn = get_connection()
    try:
        df.to_sql("trades", conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()


def load_trades(conn: Optional[sqlite3.Connection] = None):
    """Load trades from database as DataFrame."""
    import pandas as pd
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        df = pd.read_sql_query("SELECT * FROM trades", conn)
        return df
    finally:
        if close_conn:
            conn.close()


def export_trades_as_list() -> List[Dict[str, Any]]:
    """Export trades table as a list of dictionaries."""
    df = load_trades()
    # Convert datetime columns to strings for JSON serialization
    for col in df.columns:
        if df[col].dtype.kind == 'M':  # datetime
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df.to_dict('records')


def cache_price(ticker: str, date: str, price: float, conn: Optional[sqlite3.Connection] = None):
    """Cache a single price point."""
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True
    try:
        conn.execute(
            "INSERT OR REPLACE INTO prices (ticker, date, close) VALUES (?, ?, ?)",
            (ticker, date, float(price))
        )
    finally:
        if close_conn:
            conn.commit()
            conn.close()


def get_cached_prices(ticker: str, start_date: str, end_date: str) -> Optional[dict]:
    """Get cached prices for a ticker between dates."""
    import pandas as pd
    conn = get_connection()
    try:
        cached_query = f"""
            SELECT date, close FROM prices
            WHERE ticker = '{ticker}' AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date
        """
        cached = pd.read_sql_query(cached_query, conn)
        if not cached.empty:
            cached["date"] = pd.to_datetime(cached["date"]).dt.date
            cached.set_index("date", inplace=True)
            return cached
        return None
    finally:
        conn.close()


def get_cached_fx_rates(fx_ticker: str, start_date: str, end_date: str) -> Optional[dict]:
    """Get cached FX rates."""
    return get_cached_prices(fx_ticker, start_date, end_date)
