"""
Unit tests for database module.
Tests database operations, CRUD, and caching.
"""

import pytest
import sqlite3
import pandas as pd
import tempfile
import sys
from pathlib import Path
from datetime import datetime

src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

from src.database import (
    get_connection,
    reset_database,
    has_trades_data,
    create_prices_table,
    save_trades,
    load_trades,
    export_trades_as_list,
    cache_price,
    get_cached_prices
)


@pytest.mark.unit
class TestConnectionManagement:
    def test_get_connection_creates_file(self):
        """Test that get_connection creates the database file."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name

        # Override db_path to use temp file
        conn = get_connection(db_path)

        assert conn is not None
        assert isinstance(conn, sqlite3.Connection)

        conn.close()

    def test_get_connection_checks_directory_exists(self):
        """Test that get_connection handles directory creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = f"{tmp_dir}/subdir/db/merged_trading.db"

            conn = get_connection(db_path)

            assert conn is not None
            conn.close()


@pytest.mark.unit
class TestHasTradesData:
    def test_has_trades_data_empty_db(self):
        """Test checking for trades data in empty database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name

        result = has_trades_data(db_path)

        assert result is False

    def test_has_trades_data_with_data(self, temp_db_path):
        """Test checking for trades data with existing trades."""
        # Create trades table with data
        conn = sqlite3.connect(temp_db_path)
        conn.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO trades (id) VALUES (1)")
        conn.commit()
        conn.close()

        result = has_trades_data(temp_db_path)

        assert result is True

    def test_has_trades_data_only_prices_table(self, temp_db_path):
        """Test checking for trades data when only prices table exists."""
        conn = sqlite3.connect(temp_db_path)
        conn.execute("CREATE TABLE prices (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        result = has_trades_data(temp_db_path)

        assert result is False


@pytest.mark.unit
class TestResetDatabase:
    def test_reset_database_clears_all_tables(self, temp_db_path):
        """Test that reset_database clears all tables."""
        # Create some tables
        conn = sqlite3.connect(temp_db_path)
        conn.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE prices (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO trades (id) VALUES (1)")
        conn.execute("INSERT INTO prices (id) VALUES (1)")
        conn.commit()
        conn.close()

        # Reset database
        reset_database(temp_db_path)

        # Verify tables are gone
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert 'trades' not in tables
        assert 'prices' not in tables

    def test_reset_database_handles_missing_tables(self, temp_db_path):
        """Test reset_database when tables don't exist."""
        # Should not raise error
        reset_database(temp_db_path)


@pytest.mark.unit
class TestTradesCRUD:
    def test_save_trades_creates_table(self, temp_db_path):
        """Test that save_trades creates the trades table."""
        df = pd.DataFrame({
            'Security / ISIN': ['Vanguard ETF / ISIN IE00BYYHSQ67'],
            'Ticker': ['VUSA.L']
        })

        save_trades(df, db_path=temp_db_path)

        # Verify table exists
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
        tables = cursor.fetchall()
        conn.close()

        assert len(tables) == 1

    def test_save_trades_replaces_existing_data(self, temp_db_path):
        """Test that save_trades replaces existing data (not appends)."""
        df1 = pd.DataFrame({
            'Security / ISIN': ['Vanguard ETF / ISIN IE00BYYHSQ67'],
            'Ticker': ['VUSA.L']
        })

        save_trades(df1, db_path=temp_db_path)

        df2 = pd.DataFrame({
            'Security / ISIN': ['iShares ETF / ISIN IE00B4L5Y983'],
            'Ticker': ['IUIT.L']
        })

        save_trades(df2, db_path=temp_db_path)

        # Load and verify only df2 data exists
        df_loaded = load_trades(db_path=temp_db_path)

        assert len(df_loaded) == 1
        assert df_loaded.iloc[0]['Ticker'] == 'IUIT.L'

    def test_load_trades_empty_database(self, temp_db_path):
        """Test loading trades from empty database."""
        # Should create empty dataframe with expected columns
        df = load_trades(db_path=temp_db_path)

        assert isinstance(df, pd.DataFrame)

    def test_load_trades_preserves_data_types(self, temp_db_path):
        """Test that load_trades preserves data types."""
        df = pd.DataFrame({
            'Security / ISIN': ['Vanguard ETF / ISIN IE00BYYHSQ67'],
            'Ticker': ['VUSA.L'],
            'Trade Date/Time': [pd.Timestamp('2024-01-01')],
            'Quantity': [10]
        })

        save_trades(df, db_path=temp_db_path)
        df_loaded = load_trades(db_path=temp_db_path)

        assert df_loaded.iloc[0]['Ticker'] == 'VUSA.L'
        assert df_loaded.iloc[0]['Quantity'] == 10

    def test_export_trades_as_list(self, temp_db_path):
        """Test exporting trades as a list of dictionaries."""
        df = pd.DataFrame({
            'Security / ISIN': ['Vanguard ETF / ISIN IE00BYYHSQ67'],
            'Ticker': ['VUSA.L'],
            'Trade Date/Time': [pd.Timestamp('2024-01-01 12:00:00')],
            'Quantity': [10],
            'Share Price': [100.0]
        })

        save_trades(df, db_path=temp_db_path)
        trades_list = export_trades_as_list(temp_db_path)

        assert isinstance(trades_list, list)
        assert len(trades_list) == 1
        assert trades_list[0]['Ticker'] == 'VUSA.L'
        assert trades_list[0]['Quantity'] == 10
        # Verify datetime is serialized
        assert isinstance(trades_list[0]['Trade Date/Time'], str)


@pytest.mark.unit
class TestPricesTable:
    def test_create_prices_table(self, temp_db_path):
        """Test creating the prices table."""
        create_prices_table(temp_db_path)

        # Verify table exists
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='prices'
        """)
        tables = cursor.fetchall()
        conn.close()

        assert len(tables) == 1

    def test_create_prices_table_creates_index(self, temp_db_path):
        """Test that prices table has proper index."""
        create_prices_table(temp_db_path)

        # Verify index exists
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND tbl_name='prices'
        """)
        indexes = cursor.fetchall()
        conn.close()

        # Should have at least one index
        assert len(indexes) >= 1


@pytest.mark.unit
class TestPriceCaching:
    def test_cache_price_stores_data(self, temp_db_path):
        """Test caching price data in the database."""
        create_prices_table(temp_db_path)

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        dates = pd.date_range(start_date, end_date, freq='D')
        prices = [100.0 + i for i in range(len(dates))]

        cache_price('VUSA.L', dates, prices, db_path=temp_db_path)

        # Verify data was cached
        cached = get_cached_prices('VUSA.L', '2024-01-01', '2024-01-10', db_path=temp_db_path)

        assert cached is not None
        assert len(cached) == len(prices)

    def test_get_cached_prices_returns_none_when_missing(self, temp_db_path):
        """Test getting cached prices when data doesn't exist."""
        create_prices_table(temp_db_path)

        cached = get_cached_prices('VUSA.L', '2024-01-01', '2024-01-10', db_path=temp_db_path)

        assert cached is None

    def test_get_cached_prices_partial_range(self, temp_db_path):
        """Test getting cached prices for a partial date range."""
        create_prices_table(temp_db_path)

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 10)
        dates = pd.date_range(start_date, end_date, freq='D')
        prices = [100.0 + i for i in range(len(dates))]

        cache_price('VUSA.L', dates, prices, db_path=temp_db_path)

        # Request subset of dates
        cached = get_cached_prices('VUSA.L', '2024-01-05', '2024-01-08', db_path=temp_db_path)

        assert cached is not None
        assert len(cached) == 4
        assert list(cached['close']) == [104.0, 105.0, 106.0, 107.0]  # Jan 5-8 prices

    def test_cache_price_overwrites_existing(self, temp_db_path):
        """Test that caching overwrites existing data for same ticker."""
        create_prices_table(temp_db_path)

        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
        prices1 = [100.0, 101.0, 102.0, 103.0, 104.0]

        cache_price('VUSA.L', dates, prices1, db_path=temp_db_path)

        prices2 = [200.0, 201.0, 202.0, 203.0, 204.0]
        cache_price('VUSA.L', dates, prices2, db_path=temp_db_path)

        cached = get_cached_prices('VUSA.L', '2024-01-01', '2024-01-05', db_path=temp_db_path)

        assert list(cached['close']) == prices2

    def test_get_cached_prices_date_format_handling(self, temp_db_path):
        """Test that date formats are handled correctly."""
        create_prices_table(temp_db_path)

        dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
        prices = [100.0 + i for i in range(len(dates))]

        cache_price('VUSA.L', dates, prices, db_path=temp_db_path)

        # Test with string dates in different formats
        cached = get_cached_prices('VUSA.L', '2024-01-05', '2024-01-08', db_path=temp_db_path)

        assert cached is not None
        assert len(cached) == 4

    def test_cache_multiple_tickers(self, temp_db_path):
        """Test caching prices for multiple tickers."""
        create_prices_table(temp_db_path)

        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
        prices1 = [100.0, 101.0, 102.0, 103.0, 104.0]
        prices2 = [50.0, 51.0, 52.0, 53.0, 54.0]

        cache_price('VUSA.L', dates, prices1, db_path=temp_db_path)
        cache_price('VWRL.L', dates, prices2, db_path=temp_db_path)

        cached1 = get_cached_prices('VUSA.L', '2024-01-01', '2024-01-05', db_path=temp_db_path)
        cached2 = get_cached_prices('VWRL.L', '2024-01-01', '2024-01-05', db_path=temp_db_path)

        assert len(cached1) == len(prices1)
        assert len(cached2) == len(prices2)
        # Extract close values and compare
        assert list(cached1['close']) == prices1
        assert list(cached2['close']) == prices2

    def test_cache_price_empty_dates(self, temp_db_path):
        """Test caching with empty date range."""
        create_prices_table(temp_db_path)

        dates = pd.date_range('2024-01-01', '2024-01-01', freq='D')
        prices = [100.0]

        cache_price('VUSA.L', dates, prices, db_path=temp_db_path)

        cached = get_cached_prices('VUSA.L', '2024-01-01', '2024-01-01', db_path=temp_db_path)

        assert len(cached) == 1
        assert cached.iloc[0]['close'] == 100.0
