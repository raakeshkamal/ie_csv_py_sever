import pytest
import tempfile
import sqlite3
import datetime
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.main import app

@pytest.fixture
def temp_db_path():
    """Create a temporary SQLite DB and yield its path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        path = tmp.name
    yield path
    # Cleanup is handled by the OS when the file is closed

def setup_isin_mapping_table(conn):
    """Create ISIN to ticker mapping table and add test mapping."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS isin_ticker_mapping (
            isin TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            security_name TEXT NOT NULL
        )
    """)
    # Add a test mapping for VUSA.L ticker used in tests
    conn.execute(
        "INSERT OR REPLACE INTO isin_ticker_mapping (isin, ticker, security_name) VALUES (?, ?, ?)",
        ("TEST_ISIN", "VUSA.L", "Test Security"),
    )
    conn.commit()

def setup_precomputed_portfolio(conn, max_date_str):
    """Create tables and insert a single precomputed portfolio value with given max date."""
    # Create ISIN mapping table first
    setup_isin_mapping_table(conn)

    # Create required tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS precomputed_portfolio_values (
            date DATE PRIMARY KEY,
            daily_value REAL,
            last_updated TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS precomputed_monthly_contributions (
            month TEXT PRIMARY KEY,
            net_value REAL,
            last_updated TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS precomputed_ticker_daily_values (
            date DATE,
            ticker TEXT,
            daily_value REAL,
            last_updated TIMESTAMP,
            PRIMARY KEY (date, ticker)
        )
    """)
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
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Insert a single row with the provided max date
    conn.execute(
        "INSERT INTO precomputed_portfolio_values (date, daily_value, last_updated) VALUES (?, ?, ?)",
        (max_date_str, 100.0, now),
    )
    # Insert a dummy monthly contribution row
    conn.execute(
        "INSERT INTO precomputed_monthly_contributions (month, net_value, last_updated) VALUES (?, ?, ?)",
        ("2024-01", 1000.0, now),
    )
    # Insert status row marked completed
    conn.execute(
        "INSERT INTO precompute_status (status, started_at, completed_at, total_tickers) VALUES (?, ?, ?, ?)",
        ("completed", now, now, 1),
    )
    conn.commit()

def test_portfolio_values_stale_data_triggers_extension(temp_db_path):
    """When precomputed data is older than today, the endpoint should trigger background extension."""
    # Set up DB with a date of yesterday
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    conn = sqlite3.connect(temp_db_path)
    setup_precomputed_portfolio(conn, yesterday)
    conn.close()

    # Patch DB_PATH in src.database to use our temporary DB
    with patch.object(__import__("src.database", fromlist=["DB_PATH"]), "DB_PATH", temp_db_path):
        # Mock precompute_portfolio_data to avoid real background work
        with patch("src.main.precompute_portfolio_data") as mock_precompute:
            client = TestClient(app)
            response = client.get("/portfolio-values/")
            assert response.status_code == 200
            data = response.json()
            # Should indicate extension in progress and data_extended true
            assert data.get("extension_in_progress") is True
            assert data.get("data_extended") is True
            # Ensure the old data is still present
            assert "daily_dates" in data and "daily_values" in data
            # Verify that the mock was called (background task scheduled)
            mock_precompute.assert_called()

def test_portfolio_values_current_data_no_extension(temp_db_path):
    """When precomputed data is up to today, no extension should be triggered."""
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    conn = sqlite3.connect(temp_db_path)
    setup_precomputed_portfolio(conn, today_str)
    conn.close()

    with patch.object(__import__("src.database", fromlist=["DB_PATH"]), "DB_PATH", temp_db_path):
        with patch("src.main.precompute_portfolio_data") as mock_precompute:
            client = TestClient(app)
            response = client.get("/portfolio-values/")
            assert response.status_code == 200
            data = response.json()
            assert data.get("extension_in_progress") is None
            assert data.get("data_extended") is False
            mock_precompute.assert_not_called()

def test_export_prices_stale_data_triggers_extension(temp_db_path):
    """Stale precomputed ticker prices should cause extension flag in export endpoint."""
    # Insert stale ticker price data (yesterday)
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    conn = sqlite3.connect(temp_db_path)
    # Create tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS precomputed_ticker_prices (
            ticker TEXT,
            date DATE,
            original_currency TEXT,
            original_price REAL,
            converted_price_gbp REAL,
            last_updated TIMESTAMP,
            PRIMARY KEY (ticker, date)
        )
    """)
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
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO precomputed_ticker_prices (ticker, date, original_currency, original_price, converted_price_gbp, last_updated) VALUES (?, ?, ?, ?, ?, ?)",
        ("VUSA.L", yesterday, "GBP", 31.0, 31.0, now),
    )
    # Insert status row
    conn.execute(
        "INSERT INTO precompute_status (status, started_at, completed_at, total_tickers) VALUES (?, ?, ?, ?)",
        ("completed", now, now, 1),
    )
    conn.commit()
    conn.close()

    with patch.object(__import__("src.database", fromlist=["DB_PATH"]), "DB_PATH", temp_db_path):
        with patch("src.main.precompute_portfolio_data") as mock_precompute:
            client = TestClient(app)
            response = client.get("/export/prices/")
            assert response.status_code == 200
            data = response.json()
            assert data.get("extension_in_progress") is True
            assert data.get("data_extended") is True
            mock_precompute.assert_called()

def test_export_prices_current_data_no_extension(temp_db_path):
    """When price data is up to today, export endpoint should not trigger extension."""
    today = datetime.date.today().strftime("%Y-%m-%d")
    conn = sqlite3.connect(temp_db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS precomputed_ticker_prices (
            ticker TEXT,
            date DATE,
            original_currency TEXT,
            original_price REAL,
            converted_price_gbp REAL,
            last_updated TIMESTAMP,
            PRIMARY KEY (ticker, date)
        )
    """)
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
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO precomputed_ticker_prices (ticker, date, original_currency, original_price, converted_price_gbp, last_updated) VALUES (?, ?, ?, ?, ?, ?)",
        ("VUSA.L", today, "GBP", 31.0, 31.0, now),
    )
    conn.execute(
        "INSERT INTO precompute_status (status, started_at, completed_at, total_tickers) VALUES (?, ?, ?, ?)",
        ("completed", now, now, 1),
    )
    conn.commit()
    conn.close()

    with patch.object(__import__("src.database", fromlist=["DB_PATH"]), "DB_PATH", temp_db_path):
        with patch("src.main.precompute_portfolio_data") as mock_precompute:
            client = TestClient(app)
            response = client.get("/export/prices/")
            assert response.status_code == 200
            data = response.json()
            assert data.get("extension_in_progress") is None
            assert data.get("data_extended") is False
            mock_precompute.assert_not_called()
