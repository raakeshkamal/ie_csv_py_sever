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


def ensure_db_directory(db_path: Optional[str] = None):
    """Ensure the database directory exists."""
    path = db_path or DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Get a database connection."""
    path = db_path or DB_PATH
    ensure_db_directory(path)
    return sqlite3.connect(path)


def reset_database(db_path: Optional[str] = None):
    """Reset the database by dropping all tables."""
    ensure_db_directory(db_path)
    conn = get_connection(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS trades")
        conn.execute("DROP TABLE IF EXISTS prices")
        conn.execute("DROP TABLE IF EXISTS precomputed_portfolio_values")
        conn.execute("DROP TABLE IF EXISTS precomputed_monthly_contributions")
        conn.execute("DROP TABLE IF EXISTS precomputed_ticker_prices")
        conn.execute("DROP TABLE IF EXISTS precompute_status")
        conn.commit()
    finally:
        conn.close()


def table_exists(table_name: str, db_path: Optional[str] = None) -> bool:
    """Check if a table exists in the database."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return cursor.fetchone() is not None
    finally:
        conn.close()


def has_trades_data(db_path: Optional[str] = None) -> bool:
    """Check if the trades table exists and has data."""
    if not table_exists("trades", db_path):
        return False
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM trades")
        count = cursor.fetchone()[0]
        return count > 0
    finally:
        conn.close()


def create_prices_table(db_path: Optional[str] = None):
    """Create the prices table if it doesn't exist."""
    conn = get_connection(db_path)
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


def save_trades(df, conn: Optional[sqlite3.Connection] = None, db_path: Optional[str] = None):
    """Save trades DataFrame to database."""
    close_conn = False
    if conn is None:
        conn = get_connection(db_path)
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


def load_trades(conn: Optional[sqlite3.Connection] = None, db_path: Optional[str] = None):
    """Load trades from database as DataFrame."""
    import pandas as pd
    import sqlite3
    from pandas.errors import DatabaseError
    close_conn = False
    if conn is None:
        conn = get_connection(db_path)
        close_conn = True
    try:
        try:
            df = pd.read_sql_query("SELECT * FROM trades", conn)
            return df
        except (sqlite3.OperationalError, DatabaseError) as e:
            if "no such table: trades" in str(e):
                # Return empty DataFrame with expected columns
                return pd.DataFrame()
            raise
    finally:
        if close_conn:
            conn.close()


def export_trades_as_list(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Export trades table as a list of dictionaries."""
    df = load_trades(db_path=db_path)
    # Convert datetime columns to strings for JSON serialization
    for col in df.columns:
        if df[col].dtype.kind == 'M':  # datetime
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df.to_dict('records')


def cache_price(ticker: str, dates, prices, conn: Optional[sqlite3.Connection] = None, db_path: Optional[str] = None):
    """Cache price points for a ticker. Accepts either single values or lists/arrays."""
    import pandas as pd

    close_conn = False
    if conn is None:
        conn = get_connection(db_path)
        close_conn = True
    try:
        # Handle single values
        if isinstance(dates, (str, pd.Timestamp)) and isinstance(prices, (int, float)):
            date_str = str(dates)[:10] if not isinstance(dates, str) else dates
            conn.execute(
                "INSERT OR REPLACE INTO prices (ticker, date, close) VALUES (?, ?, ?)",
                (ticker, date_str, float(prices))
            )
        # Handle lists/arrays
        else:
            # Convert to pandas Series for consistent iteration
            price_series = pd.Series(prices, index=dates)
            for i in range(len(price_series)):
                price_val = price_series.iloc[i]
                if pd.notna(price_val) and price_val > 0:
                    date_str = dates[i].strftime("%Y-%m-%d") if hasattr(dates[i], 'strftime') else str(dates[i])[:10]
                    conn.execute(
                        "INSERT OR REPLACE INTO prices (ticker, date, close) VALUES (?, ?, ?)",
                        (ticker, date_str, float(price_val))
                    )
    finally:
        if close_conn:
            conn.commit()
            conn.close()


def get_cached_prices(ticker: str, start_date: str, end_date: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Get cached prices for a ticker between dates."""
    import pandas as pd
    conn = get_connection(db_path)
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
