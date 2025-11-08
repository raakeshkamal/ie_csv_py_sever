"""
Tests for background_processor module.
Tests precomputed data functionality, caching, and export features.
"""

import sqlite3
import tempfile
import datetime
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

from src.background_processor import (
    create_precomputed_tables,
    precompute_portfolio_data,
    get_precomputed_portfolio_data,
    get_precompute_status,
    export_precomputed_data,
)
from src.database import get_connection


@pytest.fixture
def temp_db_path():
    """Create a temporary database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        path = f.name
    yield path


@pytest.fixture
def sample_trades_df():
    """Create sample trades DataFrame."""
    data = {
        'Security / ISIN': ['Vanguard FTSE 250 / ISIN IE00BYYHSQ67', 'Vanguard FTSE 250 / ISIN IE00BYYHSQ67'],
        'Transaction Type': ['Buy', 'Buy'],
        'Quantity': [5, 10],
        'Share Price': [30.41, 31.00],
        'Total Trade Value': [152.05, 310.00],
        'Trade Date/Time': pd.to_datetime(['2024-01-02', '2024-01-15']),
        'Ticker': ['VUSA.L', 'VUSA.L']
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_yfinance_success():
    """Mock yfinance with successful responses."""
    with patch('yfinance.Ticker') as mock_ticker_class:
        # Mock for currency info
        mock_ticker_info = MagicMock()
        mock_ticker_info.info = {'currency': 'GBP'}

        # Mock for price history
        mock_ticker_hist = MagicMock()
        dates = pd.date_range('2024-01-01', '2024-02-01', freq='D')
        prices = [100.0 + i * 0.5 for i in range(len(dates))]
        mock_hist_df = pd.DataFrame({'Close': prices}, index=dates)
        mock_ticker_hist.history.return_value = mock_hist_df

        def create_ticker(ticker):
            if ticker == 'currency':
                return mock_ticker_info
            return mock_ticker_hist

        mock_ticker_class.side_effect = create_ticker
        yield mock_ticker_class


class TestCreatePrecomputedTables:
    """Tests for create_precomputed_tables function."""

    def test_create_tables(self, temp_db_path):
        """Test that all precomputed tables are created."""
        create_precomputed_tables(db_path=temp_db_path)

        conn = get_connection(temp_db_path)
        cursor = conn.cursor()

        # Check all tables exist
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name IN (
                'precomputed_portfolio_values',
                'precomputed_monthly_contributions',
                'precompute_status'
            )
        """)
        tables = [row[0] for row in cursor.fetchall()]

        assert 'precomputed_portfolio_values' in tables
        assert 'precomputed_monthly_contributions' in tables
        assert 'precompute_status' in tables
        conn.close()

    def test_tables_are_idempotent(self, temp_db_path):
        """Test that calling create tables multiple times is safe."""
        create_precomputed_tables(db_path=temp_db_path)
        create_precomputed_tables(db_path=temp_db_path)
        create_precomputed_tables(db_path=temp_db_path)

        conn = get_connection(temp_db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        # Should have exactly these 3 tables
        expected = {'precomputed_portfolio_values', 'precomputed_monthly_contributions', 'precompute_status'}
        actual = set(tables)
        assert expected.issubset(actual)
        conn.close()


class TestPrecomputePortfolioData:
    """Tests for precompute_portfolio_data function."""

    def test_precompute_success(self, temp_db_path, sample_trades_df, mock_yfinance_success):
        """Test successful precomputation of portfolio data."""
        create_precomputed_tables(db_path=temp_db_path)

        success = precompute_portfolio_data(sample_trades_df, db_path=temp_db_path)

        assert success is True

        # Verify data was stored
        conn = get_connection(temp_db_path)

        # Check portfolio values table
        cursor = conn.execute("SELECT COUNT(*) FROM precomputed_portfolio_values")
        pv_count = cursor.fetchone()[0]
        assert pv_count > 0

        # Check monthly contributions table
        cursor = conn.execute("SELECT COUNT(*) FROM precomputed_monthly_contributions")
        mc_count = cursor.fetchone()[0]
        assert mc_count > 0

        # Check status table
        cursor = conn.execute("SELECT status FROM precompute_status")
        status = cursor.fetchone()[0]
        assert status == "completed"

        conn.close()

    def test_precompute_with_no_tickers(self, temp_db_path):
        """Test precomputation with trades that have no tickers."""
        create_precomputed_tables(db_path=temp_db_path)

        data = {
            'Security / ISIN': ['Invalid / ISIN ABC123'],
            'Transaction Type': ['Buy'],
            'Quantity': [5],
            'Share Price': [30.41],
            'Total Trade Value': [152.05],
            'Trade Date/Time': pd.to_datetime(['2024-01-01']),
            'Ticker': [None]
        }
        df = pd.DataFrame(data)

        success = precompute_portfolio_data(df, db_path=temp_db_path)

        assert success is False

        # Check status shows failure
        conn = get_connection(temp_db_path)
        cursor = conn.execute("SELECT status, last_error FROM precompute_status")
        status, error = cursor.fetchone()
        assert status == "failed"
        assert "No valid tickers" in error
        conn.close()

    def test_precompute_status_tracking(self, temp_db_path, sample_trades_df, mock_yfinance_success):
        """Test that status is properly tracked throughout precomputation."""
        create_precomputed_tables(db_path=temp_db_path)

        precompute_portfolio_data(sample_trades_df, db_path=temp_db_path)

        conn = get_connection(temp_db_path)
        cursor = conn.execute("""
            SELECT status, started_at, completed_at, total_tickers
            FROM precompute_status
            ORDER BY started_at DESC
            LIMIT 1
        """)
        row = cursor.fetchone()

        assert row[0] == "completed"
        assert row[1] is not None  # started_at
        assert row[2] is not None  # completed_at
        assert row[3] == 1  # total_tickers
        conn.close()


class TestGetPrecomputedPortfolioData:
    """Tests for get_precomputed_portfolio_data function."""

    def test_get_data_when_exists(self, temp_db_path):
        """Test retrieving precomputed data when it exists and is fresh."""
        create_precomputed_tables(db_path=temp_db_path)
        conn = get_connection(temp_db_path)

        # Insert sample data
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("""
            INSERT INTO precomputed_portfolio_values (date, daily_value, last_updated)
            VALUES (?, ?, ?)
        """, ("2024-01-01", 1000.0, now))
        conn.execute("""
            INSERT INTO precomputed_monthly_contributions (month, net_value, last_updated)
            VALUES (?, ?, ?)
        """, ("2024-01", 1000.0, now))
        conn.commit()
        conn.close()

        result = get_precomputed_portfolio_data(db_path=temp_db_path)

        assert result is not None
        assert "daily_dates" in result
        assert "daily_values" in result
        assert "monthly_net" in result
        assert len(result["daily_dates"]) == 1
        assert len(result["daily_values"]) == 1
        assert result["daily_values"][0] == 1000.0

    def test_get_data_empty(self, temp_db_path):
        """Test retrieving data when database is empty."""
        create_precomputed_tables(db_path=temp_db_path)

        result = get_precomputed_portfolio_data(db_path=temp_db_path)

        assert result is None

    def test_get_data_old_data(self, temp_db_path):
        """Test retrieving data when data is stale (older than 24 hours)."""
        create_precomputed_tables(db_path=temp_db_path)
        conn = get_connection(temp_db_path)

        # Insert old data
        old_date = (datetime.datetime.now() - datetime.timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("""
            INSERT INTO precomputed_portfolio_values (date, daily_value, last_updated)
            VALUES (?, ?, ?)
        """, ("2024-01-01", 1000.0, old_date))
        conn.commit()
        conn.close()

        result = get_precomputed_portfolio_data(db_path=temp_db_path)

        assert result is None


class TestGetPrecomputeStatus:
    """Tests for get_precompute_status function."""

    def test_get_status_no_data(self, temp_db_path):
        """Test getting status when no precompute jobs exist."""
        create_precomputed_tables(db_path=temp_db_path)

        status = get_precompute_status(db_path=temp_db_path)

        assert status["status"] == "not_started"
        assert status["has_data"] is False

    def test_get_status_with_job(self, temp_db_path, sample_trades_df, mock_yfinance_success):
        """Test getting status when a job has run."""
        create_precomputed_tables(db_path=temp_db_path)
        precompute_portfolio_data(sample_trades_df, db_path=temp_db_path)

        status = get_precompute_status(db_path=temp_db_path)

        assert status["status"] == "completed"
        assert status["has_data"] is True
        assert status["total_tickers"] == 1
        assert status["started_at"] is not None
        assert status["completed_at"] is not None
        assert status["last_error"] is None

    def test_get_status_failed_job(self, temp_db_path):
        """Test getting status for a failed job."""
        create_precomputed_tables(db_path=temp_db_path)
        conn = get_connection(temp_db_path)

        # Insert failed job record
        conn.execute("""
            INSERT INTO precompute_status (status, started_at, completed_at, total_tickers, last_error)
            VALUES (?, ?, ?, ?, ?)
        """, ("failed", datetime.datetime.now(), datetime.datetime.now(), 0, "Test error"))
        conn.commit()
        conn.close()

        status = get_precompute_status(db_path=temp_db_path)

        assert status["status"] == "failed"
        assert status["has_data"] is True
        assert status["last_error"] == "Test error"


class TestExportPrecomputedData:
    """Tests for export_precomputed_data function."""

    def test_export_empty_data(self, temp_db_path):
        """Test exporting when no data exists."""
        create_precomputed_tables(db_path=temp_db_path)

        result = export_precomputed_data(db_path=temp_db_path)

        assert "error" not in result
        assert result["portfolio_values"] == []
        assert result["monthly_contributions"] == []
        assert result["count"]["portfolio_values"] == 0
        assert result["count"]["monthly_contributions"] == 0

    def test_export_with_data(self, temp_db_path, sample_trades_df, mock_yfinance_success):
        """Test exporting when data exists."""
        create_precomputed_tables(db_path=temp_db_path)
        precompute_portfolio_data(sample_trades_df, db_path=temp_db_path)

        result = export_precomputed_data(db_path=temp_db_path)

        assert "error" not in result
        assert len(result["portfolio_values"]) > 0
        assert len(result["monthly_contributions"]) == 1
        assert result["portfolio_values"][0]["daily_value"] is not None
        assert result["portfolio_values"][0]["date"] is not None
        assert "status" in result
        assert result["status"]["status"] == "completed"

    def test_export_structure(self, temp_db_path, sample_trades_df, mock_yfinance_success):
        """Test that export has correct structure."""
        create_precomputed_tables(db_path=temp_db_path)
        precompute_portfolio_data(sample_trades_df, db_path=temp_db_path)

        result = export_precomputed_data(db_path=temp_db_path)

        # Check top-level keys
        assert "portfolio_values" in result
        assert "monthly_contributions" in result
        assert "status" in result
        assert "count" in result

        # Check portfolio_values structure
        pv = result["portfolio_values"][0]
        assert "date" in pv
        assert "daily_value" in pv
        assert "last_updated" in pv

        # Check monthly_contributions structure
        mc = result["monthly_contributions"][0]
        assert "month" in mc
        assert "net_value" in mc
        assert "last_updated" in mc

        # Check status structure
        status = result["status"]
        assert "status" in status
        assert "has_data" in status
        assert status["status"] == "completed"

        # Check count structure
        count = result["count"]
        assert "portfolio_values" in count
        assert "monthly_contributions" in count
