"""
Unit tests for portfolio calculation logic.
Tests the portfolio module functions directly.
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

from src.portfolio import (
    compute_monthly_net_contributions,
    simulate_holdings,
    compute_daily_portfolio_values,
    get_valid_unique_tickers,
    calculate_portfolio_values
)


@pytest.mark.unit
class TestMonthlyNetCalculations:
    def test_monthly_net_buy_transactions(self):
        """Test monthly net calculation with only buy transactions."""
        data = {
            'Trade Date/Time': pd.to_datetime(['2024-01-15', '2024-01-20', '2024-02-15']),
            'Transaction Type': ['Buy', 'Buy', 'Buy'],
            'Total Trade Value': [100.0, 200.0, 150.0]
        }
        df = pd.DataFrame(data)

        result = compute_monthly_net_contributions(df)

        assert len(result) == 2
        assert result[0]['Month'] == '2024-01'
        assert result[0]['Net_Value'] == 300.0  # 100 + 200
        assert result[1]['Month'] == '2024-02'
        assert result[1]['Net_Value'] == 150.0

    def test_monthly_net_with_sell_transactions(self):
        """Test monthly net calculation with buy and sell transactions."""
        data = {
            'Trade Date/Time': pd.to_datetime(['2024-01-15', '2024-01-20', '2024-02-15']),
            'Transaction Type': ['Buy', 'Sell', 'Buy'],
            'Total Trade Value': [100.0, 50.0, 150.0]
        }
        df = pd.DataFrame(data)

        result = compute_monthly_net_contributions(df)

        assert len(result) == 2
        assert result[0]['Month'] == '2024-01'
        assert result[0]['Net_Value'] == 50.0  # 100 - 50
        assert result[1]['Month'] == '2024-02'
        assert result[1]['Net_Value'] == 150.0

    def test_monthly_net_mixed_transactions_per_month(self):
        """Test monthly net with mixed transactions in same month."""
        data = {
            'Trade Date/Time': pd.to_datetime(['2024-01-01', '2024-01-15', '2024-01-20', '2024-02-01']),
            'Transaction Type': ['Buy', 'Buy', 'Sell', 'Buy'],
            'Total Trade Value': [500.0, 300.0, 200.0, 400.0]
        }
        df = pd.DataFrame(data)

        result = compute_monthly_net_contributions(df)

        assert len(result) == 2
        assert result[0]['Month'] == '2024-01'
        assert result[0]['Net_Value'] == 600.0  # 500 + 300 - 200
        assert result[1]['Month'] == '2024-02'
        assert result[1]['Net_Value'] == 400.0


@pytest.mark.unit
class TestHoldingsSimulation:
    def test_simulate_holdings_no_pre_start_trades(self):
        """Test holdings simulation with no pre-start trades."""
        data = {
            'Ticker': ['VUSA.L', 'VUSA.L'],
            'Trade Date/Time': pd.to_datetime(['2024-02-15', '2024-03-15']),
            'Transaction Type': ['Buy', 'Buy'],
            'Quantity': [5, 3]
        }
        df = pd.DataFrame(data)
        unique_tickers = ['VUSA.L']
        common_start = datetime(2024, 2, 1).date()
        max_date = datetime(2024, 3, 31).date()

        holdings_data, dates = simulate_holdings(df, unique_tickers, common_start, max_date)

        assert 'VUSA.L' in holdings_data
        assert len(dates) > 0
        # Initial holdings should be 0
        assert holdings_data['VUSA.L'][0] == 0

    def test_simulate_holdings_with_pre_trades(self):
        """Test holdings simulation with pre-start trades."""
        data = {
            'Ticker': ['VUSA.L'] * 4,
            'Trade Date/Time': pd.to_datetime([
                '2024-01-10',  # Pre-start
                '2024-01-15',  # Pre-start
                '2024-02-15',  # Post-start
                '2024-03-15'   # Post-start
            ]),
            'Transaction Type': ['Buy', 'Buy', 'Sell', 'Buy'],
            'Quantity': [5, 3, 2, 4]
        }
        df = pd.DataFrame(data)
        unique_tickers = ['VUSA.L']
        common_start = datetime(2024, 2, 1).date()
        max_date = datetime(2024, 3, 31).date()

        holdings_data, dates = simulate_holdings(df, unique_tickers, common_start, max_date)

        assert 'VUSA.L' in holdings_data
        # Initial holdings should be 8 (5 + 3)
        assert holdings_data['VUSA.L'][0] == 8

    def test_simulate_holdings_multiple_tickers(self):
        """Test holdings simulation with multiple tickers."""
        data = {
            'Ticker': ['VUSA.L', 'VWRL.L', 'VUSA.L', 'VWRL.L'],
            'Trade Date/Time': pd.to_datetime(['2024-01-15', '2024-01-15', '2024-02-15', '2024-02-15']),
            'Transaction Type': ['Buy', 'Buy', 'Buy', 'Buy'],
            'Quantity': [5, 10, 3, 8]
        }
        df = pd.DataFrame(data)
        unique_tickers = ['VUSA.L', 'VWRL.L']
        common_start = datetime(2024, 2, 1).date()
        max_date = datetime(2024, 2, 28).date()

        holdings_data, dates = simulate_holdings(df, unique_tickers, common_start, max_date)

        assert 'VUSA.L' in holdings_data
        assert 'VWRL.L' in holdings_data
        assert holdings_data['VUSA.L'][0] == 5
        assert holdings_data['VWRL.L'][0] == 10

    def test_simulate_holdings_daily_accumulation(self):
        """Test daily holdings accumulation over time."""
        data = {
            'Ticker': ['VUSA.L'] * 3,
            'Trade Date/Time': pd.to_datetime(['2024-01-03', '2024-01-10', '2024-01-15']),
            'Transaction Type': ['Buy', 'Buy', 'Buy'],
            'Quantity': [5, 3, 2]
        }
        df = pd.DataFrame(data)
        unique_tickers = ['VUSA.L']
        common_start = datetime(2024, 1, 1).date()
        max_date = datetime(2024, 1, 20).date()

        holdings_data, dates = simulate_holdings(df, unique_tickers, common_start, max_date)

        vusa_holdings = holdings_data['VUSA.L']
        # Jan 1-2: 0, Jan 3+: 5, Jan 10+: 8, Jan 15+: 10
        assert vusa_holdings[0] == 0  # Jan 1
        assert vusa_holdings[1] == 0  # Jan 2
        assert vusa_holdings[2] == 5  # Jan 3
        assert vusa_holdings[8] == 5  # Jan 9 (last day before 2nd trade)
        assert vusa_holdings[9] == 8  # Jan 10 (after 2nd trade)
        assert vusa_holdings[10] == 8  # Jan 11
        assert vusa_holdings[14] == 10  # Jan 15


@pytest.mark.unit
class TestDailyPortfolioValueCalculations:
    def test_compute_daily_values_single_ticker(self):
        """Test portfolio value calculation for single ticker."""
        holdings = {
            'VUSA.L': np.array([0, 0, 5, 5, 5])
        }
        price_data = {
            'VUSA.L': np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        }
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')

        daily_values = compute_daily_portfolio_values(holdings, price_data, dates)

        # Jan 1-2: 0, Jan 3: 5 * 102 = 510, Jan 4: 5 * 103 = 515, Jan 5: 5 * 104 = 520
        assert len(daily_values) == 5
        assert daily_values[0] == 0
        assert daily_values[1] == 0
        assert daily_values[2] == 510
        assert daily_values[3] == 515
        assert daily_values[4] == 520

    def test_compute_daily_values_multiple_tickers(self):
        """Test portfolio value calculation with multiple tickers."""
        holdings = {
            'VUSA.L': np.array([0, 5, 5, 5, 5]),
            'VWRL.L': np.array([0, 0, 10, 10, 8])
        }
        price_data = {
            'VUSA.L': np.array([100.0, 101.0, 102.0, 103.0, 104.0]),
            'VWRL.L': np.array([50.0, 51.0, 52.0, 53.0, 54.0])
        }
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')

        daily_values = compute_daily_portfolio_values(holdings, price_data, dates)

        # Jan 1: 0, Jan 2: 5 * 101 = 505, Jan 3: (5 * 102) + (10 * 52) = 1030
        assert daily_values[0] == 0
        assert daily_values[1] == 505
        assert daily_values[2] == 1030

    def test_compute_daily_values_with_zero_holdings(self):
        """Test portfolio value calculation with zero holdings."""
        holdings = {
            'VUSA.L': np.array([0, 0, 0, 0, 0])
        }
        price_data = {
            'VUSA.L': np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        }
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')

        daily_values = compute_daily_portfolio_values(holdings, price_data, dates)

        assert all(value == 0 for value in daily_values)


@pytest.mark.unit
class TestTickerValidation:
    def test_get_valid_unique_tickers_removes_not_found(self):
        """Test that tickers with 'Not found' are filtered out."""
        data = {
            'Ticker': ['VUSA.L', 'Not found', 'VWRL.L', 'Not found', 'IUIT.L']
        }
        df = pd.DataFrame(data)

        result = get_valid_unique_tickers(df)

        assert 'Not found' not in result
        assert 'VUSA.L' in result
        assert 'VWRL.L' in result
        assert 'IUIT.L' in result

    def test_get_valid_unique_tickers_handles_empty(self):
        """Test that empty dataframe returns empty list."""
        df = pd.DataFrame({'Ticker': []})

        result = get_valid_unique_tickers(df)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_valid_unique_tickers_all_not_found(self):
        """Test when all tickers are 'Not found'."""
        data = {
            'Ticker': ['Not found', 'Not found', 'Not found']
        }
        df = pd.DataFrame(data)

        result = get_valid_unique_tickers(df)

        assert isinstance(result, list)
        assert len(result) == 0
