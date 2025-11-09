"""
Pytest fixtures for the InvestEngine CSV Server test suite.
"""

import sqlite3
import tempfile
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import Mock, MagicMock, patch

# Add src to path for imports
import sys
from pathlib import Path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

# Import using package-qualified imports to support relative imports
from src.main import app
from src.merge_csv import merge_csv_files
from src.database import (
    get_connection,
    reset_database,
    has_trades_data,
    create_prices_table,
    save_trades,
    load_trades,
    export_trades_as_list,
    cache_price,
    get_cached_prices,
    save_isin_ticker_mapping
)
from src.security_parser import extract_security_and_isin
from src.prices import (
    get_currency,
    get_currencies_parallel,
    get_min_hist_date,
    get_common_start_date,
    fetch_and_convert_history,
    convert_currency,
    get_fx_rates,
    fetch_prices_parallel,
    get_needed_currencies
)
from src.portfolio import (
    compute_monthly_net_contributions,
    simulate_holdings,
    compute_daily_portfolio_values,
    get_valid_unique_tickers,
    calculate_portfolio_values
)
from src.background_processor import (
    create_precomputed_tables,
    precompute_portfolio_data,
    get_precomputed_portfolio_data,
    export_precomputed_data
)


@pytest.fixture
def sample_gia_csv():
    """Sample GIA CSV content for testing."""
    return """Trading Statement,
Security / ISIN,Transaction Type,Quantity,Share Price,Total Trade Value,Trade Date/Time,Settlement Date,Broker
Vanguard FTSE 250 ETF Dist / ISIN IE00BYYHSQ67,Buy,5,£30.41,£152.05,01/11/24 12:45:32,05/11/24,InvestEngine
"""


@pytest.fixture
def sample_isa_csv():
    """Sample ISA CSV content for testing."""
    return """Trading Statement,
Security / ISIN,Transaction Type,Quantity,Share Price,Total Trade Value,Trade Date/Time,Settlement Date,Broker
Vanguard FTSE All-World ETF Acc / ISIN IE00BK5BQT80,Buy,10,£80.50,£805.00,02/11/24 10:30:15,06/11/24,InvestEngine
"""


@pytest.fixture
def mock_merged_df(sample_gia_csv, sample_isa_csv):
    """Create a mock merged dataframe for testing."""
    file_data = [
        ("GIA_Trading_test.csv", sample_gia_csv),
        ("ISA_Trading_test.csv", sample_isa_csv)
    ]
    return merge_csv_files(file_data)


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        yield f.name


@pytest.fixture
def mock_db_with_data(temp_db_path, mock_merged_df):
    """Create a temporary database with test data."""
    conn = sqlite3.connect(temp_db_path)
    mock_merged_df.to_sql("trades", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    return temp_db_path


@pytest.fixture
def mock_db_empty(temp_db_path):
    """Create an empty temporary database."""
    return temp_db_path


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_yfinance_ticker():
    """Mock yfinance Ticker object."""
    with patch('yfinance.Ticker') as mock_ticker_class:
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker

        # Mock info
        mock_ticker.info = {"currency": "GBP"}

        # Mock history
        mock_hist = pd.DataFrame({
            'Close': [100.0, 101.0, 102.0]
        }, index=pd.date_range('2024-01-01', '2024-01-03', freq='D'))
        mock_ticker.history.return_value = mock_hist

        yield mock_ticker_class


@pytest.fixture
def mock_yfinance_fx():
    """Mock yfinance for FX rates."""
    with patch('yfinance.Ticker') as mock_ticker_class:
        def create_fx_ticker(ticker_str):
            mock_ticker = MagicMock()
            if ticker_str == "GBPUSD=X":
                mock_hist = pd.DataFrame({
                    'Close': [1.25, 1.26, 1.27]
                }, index=pd.date_range('2024-01-01', '2024-01-03', freq='D'))
            elif ticker_str == "EURGBP=X":
                mock_hist = pd.DataFrame({
                    'Close': [0.85, 0.86, 0.87]
                }, index=pd.date_range('2024-01-01', '2024-01-03', freq='D'))
            else:
                mock_hist = pd.DataFrame()
            mock_ticker.history.return_value = mock_hist
            return mock_ticker

        mock_ticker_class.side_effect = create_fx_ticker
        yield mock_ticker_class


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for ticker search."""
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "quotes": [
                {
                    "symbol": "VUSA.L",
                    "exchange": "LSE",
                    "quoteType": "ETF"
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        yield mock_get


@pytest.fixture
def sample_trades_df():
    """Sample trades dataframe for testing."""
    data = {
        'Security / ISIN': ['Vanguard FTSE 250 ETF Dist / ISIN IE00BYYHSQ67'] * 3,
        'Transaction Type': ['Buy', 'Buy', 'Sell'],
        'Quantity': [5, 10, 3],
        'Share Price': [30.41, 31.00, 32.00],
        'Total Trade Value': [152.05, 310.00, 96.00],
        'Trade Date/Time': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
        'isin': ['IE00BYYHSQ67', 'IE00BYYHSQ67', 'IE00BYYHSQ67'],
        'security_name': ['Vanguard FTSE 250 ETF Dist'] * 3
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_isin_mappings():
    """Sample ISIN to ticker mappings for testing."""
    return [
        ("IE00BYYHSQ67", "VMID.L", "Vanguard FTSE 250 ETF Dist"),
        ("IE00BK5BQT80", "VWRL.L", "Vanguard FTSE All-World ETF Acc"),
    ]

