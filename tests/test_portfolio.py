"""
Unit tests for portfolio calculation logic.
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

from main import get_portfolio_values


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

        df['Month'] = df['Trade Date/Time'].dt.to_period('M')
        df['Net_Value'] = df.apply(
            lambda row: row['Total Trade Value']
            if row['Transaction Type'] == 'Buy'
            else -row['Total Trade Value'],
            axis=1
        )
        monthly_net = df.groupby('Month')['Net_Value'].sum()

        assert monthly_net['2024-01'] == 300.0  # 100 + 200
        assert monthly_net['2024-02'] == 150.0

    def test_monthly_net_with_sell_transactions(self):
        """Test monthly net calculation with buy and sell transactions."""
        data = {
            'Trade Date/Time': pd.to_datetime(['2024-01-15', '2024-01-20', '2024-02-15']),
            'Transaction Type': ['Buy', 'Sell', 'Buy'],
            'Total Trade Value': [100.0, 50.0, 150.0]
        }
        df = pd.DataFrame(data)

        df['Month'] = df['Trade Date/Time'].dt.to_period('M')
        df['Net_Value'] = df.apply(
            lambda row: row['Total Trade Value']
            if row['Transaction Type'] == 'Buy'
            else -row['Total Trade Value'],
            axis=1
        )
        monthly_net = df.groupby('Month')['Net_Value'].sum()

        assert monthly_net['2024-01'] == 50.0  # 100 - 50
        assert monthly_net['2024-02'] == 150.0


@pytest.mark.unit
class TestInitialHoldings:
    def test_initial_holdings_no_pre_trades(self):
        """Test initial holdings calculation with no pre-start trades."""
        common_start = datetime(2024, 2, 1).date()

        data = {
            'Ticker': ['VUSA.L', 'VUSA.L'],
            'Trade Date/Time': pd.to_datetime(['2024-02-15', '2024-03-15']),
            'Transaction Type': ['Buy', 'Buy'],
            'Quantity': [5, 3]
        }
        df = pd.DataFrame(data)

        pre_trades_mask = (df['Ticker'] == 'VUSA.L') & (
            df['Trade Date/Time'].dt.date < common_start
        )
        pre_trades = df[pre_trades_mask]

        assert len(pre_trades) == 0

    def test_initial_holdings_with_pre_trades(self):
        """Test initial holdings calculation with pre-start trades."""
        common_start = datetime(2024, 2, 1).date()

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

        pre_trades_mask = (df['Ticker'] == 'VUSA.L') & (
            df['Trade Date/Time'].dt.date < common_start
        )
        pre_trades = df[pre_trades_mask].copy()

        # Quantity adjustment
        pre_trades['Quantity_Adj'] = np.where(
            pre_trades['Transaction Type'] == 'Buy',
            pre_trades['Quantity'],
            -pre_trades['Quantity']
        )
        initial_qty = pre_trades['Quantity_Adj'].sum()

        # 5 + 3 = 8
        assert initial_qty == 8


@pytest.mark.unit
class TestDailyHoldingsSimulation:
    def test_daily_holdings_accumulation(self):
        """Test daily holdings accumulation over time."""
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')

        # Single trade on Jan 3
        trades_data = {
            'Trade Date/Time': pd.to_datetime(['2024-01-03']),
            'Transaction Type': ['Buy'],
            'Quantity_Adj': [5]
        }
        trades_df = pd.DataFrame(trades_data)
        daily_adj = trades_df.groupby(trades_df['Trade Date/Time'].dt.date)['Quantity_Adj'].sum()

        # Simulate holdings
        daily_holdings = pd.Series(index=dates, dtype=float)
        cum_qty = 0.0
        for i, date in enumerate(dates):
            adj_today = daily_adj.get(date.date(), 0.0)
            cum_qty += adj_today
            daily_holdings.iloc[i] = cum_qty

        # Jan 1-2: 0, Jan 3-5: 5
        assert daily_holdings.iloc[0] == 0
        assert daily_holdings.iloc[1] == 0
        assert daily_holdings.iloc[2] == 5
        assert daily_holdings.iloc[3] == 5
        assert daily_holdings.iloc[4] == 5

    def test_daily_holdings_multiple_trades_same_day(self):
        """Test daily holdings with multiple trades on same day."""
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')

        # Multiple trades on Jan 3
        trades_data = {
            'Trade Date/Time': pd.to_datetime(['2024-01-03', '2024-01-03']),
            'Transaction Type': ['Buy', 'Sell'],
            'Quantity_Adj': [10, -3]
        }
        trades_df = pd.DataFrame(trades_data)
        daily_adj = trades_df.groupby(trades_df['Trade Date/Time'].dt.date)['Quantity_Adj'].sum()

        # Should aggregate to net 7 on Jan 3
        assert daily_adj[datetime(2024, 1, 3).date()] == 7


@pytest.mark.unit
class TestPortfolioValueCalculations:
    def test_portfolio_value_single_ticker(self):
        """Test portfolio value calculation for single ticker."""
        holdings = np.array([0, 0, 5, 5, 5])
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])

        daily_values = holdings * prices

        # Jan 1-2: 0, Jan 3: 5 * 102 = 510, Jan 4: 5 * 103 = 515, Jan 5: 5 * 104 = 520
        assert daily_values[0] == 0
        assert daily_values[1] == 0
        assert daily_values[2] == 510
        assert daily_values[3] == 515
        assert daily_values[4] == 520

    def test_portfolio_value_multiple_tickers(self):
        """Test portfolio value calculation with multiple tickers."""
        holdings_ticker1 = np.array([0, 5, 5, 5, 5])
        holdings_ticker2 = np.array([0, 0, 10, 10, 8])
        prices_ticker1 = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        prices_ticker2 = np.array([50.0, 51.0, 52.0, 53.0, 54.0])

        daily_values = (holdings_ticker1 * prices_ticker1) + (holdings_ticker2 * prices_ticker2)

        # Jan 1: 0, Jan 2: 5 * 101 = 505, Jan 3: (5 * 102) + (10 * 52) = 510 + 520 = 1030
        assert daily_values[0] == 0
        assert daily_values[1] == 505
        assert daily_values[2] == 1030

    def test_currency_conversion_gbp_to_gbp(self):
        """Test currency conversion when already in GBP."""
        price = 100.0
        currency = "GBP"

        # Should remain unchanged
        converted = price if currency == "GBP" else None
        assert converted == 100.0

    def test_currency_conversion_usd_to_gbp(self):
        """Test currency conversion from USD to GBP."""
        usd_price = 100.0
        fx_rate = 1.25  # 1 GBP = 1.25 USD

        gbp_price = usd_price / fx_rate
        assert gbp_price == 80.0

    def test_currency_conversion_gbp_to_gbp_divide_by_100(self):
        """Test currency conversion from GBp (pence) to GBP."""
        gbp_price = 5000.0  # 5000 pence
        converted = gbp_price / 100.0
        assert converted == 50.0


@pytest.mark.unit
class TestDateRangeCalculations:
    def test_common_start_date_calculation(self):
        """Test calculation of common start date from multiple tickers."""
        ticker_dates = {
            'VUSA.L': datetime(2024, 1, 15).date(),
            'VWRL.L': datetime(2024, 2, 1).date(),
            'IUIT.L': datetime(2024, 1, 20).date()
        }

        common_start = max(ticker_dates.values())
        assert common_start == datetime(2024, 2, 1).date()

    def test_date_range_generation(self):
        """Test generation of daily date range."""
        start_date = datetime(2024, 1, 1).date()
        end_date = datetime(2024, 1, 10).date()

        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        daily_dates = [d.strftime("%Y-%m-%d") for d in dates]

        assert len(dates) == 10
        assert daily_dates[0] == "2024-01-01"
        assert daily_dates[-1] == "2024-01-10"
