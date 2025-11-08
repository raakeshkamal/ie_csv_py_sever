"""
Unit tests for prices module.
Tests price fetching, conversion, and caching logic.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta, date
from unittest.mock import MagicMock, patch

src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

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


@pytest.mark.unit
class TestGetCurrency:
    def test_get_currency_gbp(self, mock_yfinance_ticker):
        """Test getting currency for GBP ticker."""
        # Mock yfinance Ticker to return GBP
        mock_ticker = MagicMock()
        mock_ticker.info = {'currency': 'GBP'}
        mock_yfinance_ticker.return_value = mock_ticker

        currency = get_currency('VUSA.L')

        assert currency == 'GBP'

    def test_get_currency_usd(self, mock_yfinance_ticker):
        """Test getting currency for USD ticker."""
        mock_ticker = MagicMock()
        mock_ticker.info = {'currency': 'USD'}
        mock_yfinance_ticker.return_value = mock_ticker

        currency = get_currency('SPY')

        assert currency == 'USD'

    def test_get_currency_eur(self, mock_yfinance_ticker):
        """Test getting currency for EUR ticker."""
        mock_ticker = MagicMock()
        mock_ticker.info = {'currency': 'EUR'}
        mock_yfinance_ticker.return_value = mock_ticker

        currency = get_currency('EUNL.DE')

        assert currency == 'EUR'

    def test_get_currency_gbp_pence(self, mock_yfinance_ticker):
        """Test getting currency for GBp (pence) ticker."""
        mock_ticker = MagicMock()
        mock_ticker.info = {'currency': 'GBp'}
        mock_yfinance_ticker.return_value = mock_ticker

        currency = get_currency('VUSA')

        assert currency == 'GBp'

    def test_get_currency_error(self, mock_yfinance_ticker):
        """Test getting currency when yfinance fails."""
        mock_yfinance_ticker.side_effect = Exception("Network error")

        currency = get_currency('INVALID.TICKER')

        # Should return GBP fallback on error
        assert currency == "GBP"


@pytest.mark.unit
class TestGetCurrenciesParallel:
    def test_get_currencies_parallel_single_ticker(self, mock_yfinance_ticker):
        """Test getting currencies for single ticker in parallel."""
        mock_ticker = MagicMock()
        mock_ticker.info = {'currency': 'GBP'}
        mock_yfinance_ticker.return_value = mock_ticker

        tickers = ['VUSA.L']
        currencies = get_currencies_parallel(tickers)

        assert len(currencies) == 1
        assert currencies['VUSA.L'] == 'GBP'

    def test_get_currencies_parallel_multiple_tickers(self, mock_yfinance_ticker):
        """Test getting currencies for multiple tickers in parallel."""
        def create_ticker(ticker_str):
            mock_ticker = MagicMock()
            if ticker_str == 'VUSA.L':
                mock_ticker.info = {'currency': 'GBP'}
            elif ticker_str == 'VWRL.L':
                mock_ticker.info = {'currency': 'GBP'}
            elif ticker_str == 'EUNL.DE':
                mock_ticker.info = {'currency': 'EUR'}
            return mock_ticker

        mock_yfinance_ticker.side_effect = create_ticker

        tickers = ['VUSA.L', 'VWRL.L', 'EUNL.DE']
        currencies = get_currencies_parallel(tickers)

        assert len(currencies) == 3
        assert currencies['VUSA.L'] == 'GBP'
        assert currencies['VWRL.L'] == 'GBP'
        assert currencies['EUNL.DE'] == 'EUR'

    def test_get_currencies_parallel_with_invalid_ticker(self, mock_yfinance_ticker):
        """Test getting currencies when one ticker fails."""
        def create_ticker(ticker_str):
            mock_ticker = MagicMock()
            if ticker_str == 'VUSA.L':
                mock_ticker.info = {'currency': 'GBP'}
            else:
                mock_ticker.info = {}
                # Force KeyError for missing currency
                del mock_ticker.info['currency']
            return mock_ticker

        mock_yfinance_ticker.side_effect = create_ticker

        tickers = ['VUSA.L', 'INVALID.TICKER']
        currencies = get_currencies_parallel(tickers)

        assert len(currencies) == 2
        assert currencies['VUSA.L'] == 'GBP'
        assert currencies['INVALID.TICKER'] == "GBP"


@pytest.mark.unit
class TestGetMinHistDate:
    def test_get_min_hist_date_valid_ticker(self, mock_yfinance_ticker):
        """Test getting minimum historical date for a ticker."""
        mock_ticker = MagicMock()
        # Mock history with 30 days of data
        dates = pd.date_range('2024-01-01', '2024-01-30', freq='D')
        mock_hist = pd.DataFrame({'Close': range(30)}, index=dates)
        mock_ticker.history.return_value = mock_hist
        mock_yfinance_ticker.return_value = mock_ticker

        min_date = get_min_hist_date('VUSA.L')

        assert min_date == datetime(2024, 1, 1).date()

    def test_get_min_hist_date_empty_history(self, mock_yfinance_ticker):
        """Test getting minimum historical date when history is empty."""
        mock_ticker = MagicMock()
        mock_hist = pd.DataFrame({'Close': []})
        mock_ticker.history.return_value = mock_hist
        mock_yfinance_ticker.return_value = mock_ticker

        min_date = get_min_hist_date('VUSA.L')

        # Should return None or a default date
        assert min_date is None or isinstance(min_date, datetime)

    def test_get_min_hist_date_ticker_not_found(self, mock_yfinance_ticker):
        """Test getting minimum historical date when ticker not found."""
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = Exception("Ticker not found")
        mock_yfinance_ticker.return_value = mock_ticker

        min_date = get_min_hist_date('INVALID.TICKER')

        # Should return None or a default
        assert min_date is None or isinstance(min_date, datetime)


@pytest.mark.unit
class TestGetCommonStartDate:
    def test_get_common_start_date_single_ticker(self, mock_yfinance_ticker):
        """Test getting common start date for single ticker."""
        mock_ticker = MagicMock()
        mock_hist = pd.DataFrame({'Close': [100.0]}, index=pd.date_range('2024-01-15', periods=1))
        mock_ticker.history.return_value = mock_hist
        mock_yfinance_ticker.return_value = mock_ticker

        tickers = ['VUSA.L']
        common_start = get_common_start_date(tickers)

        # Mock should return the date from our mocked history
        assert common_start is not None
        assert isinstance(common_start, (date, pd.Timestamp))

    def test_get_common_start_date_multiple_tickers(self, mock_yfinance_ticker):
        """Test getting common start date for multiple tickers."""
        def create_ticker(ticker_str):
            mock_ticker = MagicMock()
            if ticker_str == 'VUSA.L':
                dates = pd.date_range('2024-01-15', periods=1)
            elif ticker_str == 'VWRL.L':
                dates = pd.date_range('2024-02-01', periods=1)
            else:  # IUIT.L
                dates = pd.date_range('2024-01-20', periods=1)
            mock_hist = pd.DataFrame({'Close': [100.0]}, index=dates)
            mock_ticker.history.return_value = mock_hist
            return mock_ticker

        mock_yfinance_ticker.side_effect = create_ticker

        tickers = ['VUSA.L', 'VWRL.L', 'IUIT.L']
        common_start = get_common_start_date(tickers)

        # Should be the maximum (latest date) of the mocked data
        assert common_start is not None
        assert isinstance(common_start, (date, pd.Timestamp))

    def test_get_common_start_date_with_none_values(self, mock_yfinance_ticker):
        """Test getting common start date when some tickers have None."""
        def create_ticker(ticker_str):
            mock_ticker = MagicMock()
            if ticker_str == 'INVALID.TICKER':
                # Return empty history
                mock_hist = pd.DataFrame({'Close': []})
            else:
                dates = pd.date_range('2024-01-20', periods=1)
                mock_hist = pd.DataFrame({'Close': [100.0]}, index=dates)
            mock_ticker.history.return_value = mock_hist
            return mock_ticker

        mock_yfinance_ticker.side_effect = create_ticker

        tickers = ['VUSA.L', 'INVALID.TICKER', 'VWRL.L']
        common_start = get_common_start_date(tickers)

        # Should ignore None values and return valid date
        assert common_start is not None
        assert isinstance(common_start, (date, pd.Timestamp))


@pytest.mark.unit
class TestConvertCurrency:
    def test_convert_currency_gbp_to_gbp(self):
        """Test converting from GBP to GBP (no change)."""
        prices = pd.Series([100.0, 101.0, 102.0])
        reported_currency = 'GBP'
        ticker = 'VUSA.L'
        dates = pd.date_range('2024-01-01', periods=3, freq='D')

        converted = convert_currency(prices, reported_currency, ticker, dates)

        np.testing.assert_array_equal(converted.values, prices.values)

    def test_convert_currency_gbp_pence_to_gbp(self):
        """Test converting from GBp (pence) to GBP (divide by 100)."""
        prices = pd.Series([5000.0, 5100.0, 5200.0])
        reported_currency = 'GBp'
        ticker = 'VUSA.L'  # Need .L for LSE ticker
        dates = pd.date_range('2024-01-01', periods=3, freq='D')

        converted = convert_currency(prices, reported_currency, ticker, dates)

        expected = np.array([50.0, 51.0, 52.0])
        np.testing.assert_array_almost_equal(converted.values, expected)

    @patch('src.prices.get_fx_rates')
    def test_convert_currency_usd_to_gbp(self, mock_get_fx_rates):
        """Test converting from USD to GBP using FX rates."""
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        mock_rates = pd.Series([1.25, 1.25, 1.25], index=dates)
        mock_get_fx_rates.return_value = mock_rates

        prices = pd.Series([100.0, 101.0, 102.0], index=dates)
        reported_currency = 'USD'
        ticker = 'SPY'

        converted = convert_currency(prices, reported_currency, ticker, dates=dates)

        expected = np.array([80.0, 80.8, 81.6])
        np.testing.assert_array_almost_equal(converted.values, expected)

    @patch('src.prices.get_fx_rates')
    def test_convert_currency_eur_to_gbp(self, mock_get_fx_rates):
        """Test converting from EUR to GBP using FX rates."""
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        mock_rates = pd.Series([1.15, 1.15, 1.15], index=dates)
        mock_get_fx_rates.return_value = mock_rates

        prices = pd.Series([100.0, 101.0, 102.0], index=dates)
        reported_currency = 'EUR'
        ticker = 'EUNL.DE'

        converted = convert_currency(prices, reported_currency, ticker, dates=dates)

        expected = np.array([115.0, 116.15, 117.3])
        np.testing.assert_array_almost_equal(converted.values, expected)


@pytest.mark.unit
class TestGetFXRates:
    def test_get_fx_rates_gbp_usd(self, mock_yfinance_fx):
        """Test getting GBP/USD FX rates."""
        rates = get_fx_rates('USD', pd.date_range('2024-01-01', '2024-01-05', freq='D'))

        assert rates is not None
        assert len(rates) == 5

    def test_get_fx_rates_gbp_eur(self, mock_yfinance_fx):
        """Test getting EUR/GBP FX rates."""
        rates = get_fx_rates('EUR', pd.date_range('2024-01-01', '2024-01-05', freq='D'))

        assert rates is not None
        assert len(rates) == 5

    def test_get_fx_rates_gbp_to_gbp(self):
        """Test getting FX rates for GBP to GBP (should return 1.0)."""
        rates = get_fx_rates('GBP', pd.date_range('2024-01-01', '2024-01-05', freq='D'))

        assert rates is not None
        assert all(rate == 1.0 for rate in rates)

    def test_get_fx_rates_gbp_pence_to_gbp(self):
        """Test getting FX rates for GBp to GBP (should return 1.0)."""
        rates = get_fx_rates('GBp', pd.date_range('2024-01-01', '2024-01-05', freq='D'))

        assert rates is not None
        assert all(rate == 1.0 for rate in rates)

    def test_get_fx_rates_uses_cache(self, mock_yfinance_fx):
        """Test that FX rates are cached in database."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name

        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
        rates1 = get_fx_rates('USD', dates, db_path)

        # Second call should use cache
        rates2 = get_fx_rates('USD', dates, db_path)

        # Rates should be identical
        np.testing.assert_array_equal(rates1, rates2)


@pytest.mark.unit
class TestFetchAndConvertHistory:
    def test_fetch_and_convert_history_gbp(self, mock_yfinance_ticker):
        """Test fetching and converting history for GBP ticker."""
        mock_ticker = MagicMock()
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
        mock_hist = pd.DataFrame({
            'Close': [100.0, 101.0, 102.0, 103.0, 104.0]
        }, index=dates)
        mock_ticker.history.return_value = mock_hist
        mock_yfinance_ticker.return_value = mock_ticker

        start_date = datetime(2024, 1, 1).date()
        end_date = datetime(2024, 1, 5).date()

        ticker, prices = fetch_and_convert_history(
            'VUSA.L', start_date, end_date, 'GBP', dates
        )

        assert ticker == 'VUSA.L'
        assert len(prices) == 5
        np.testing.assert_array_equal(prices, [100.0, 101.0, 102.0, 103.0, 104.0])

    def test_fetch_and_convert_history_gbp_pence(self, mock_yfinance_ticker):
        """Test fetching and converting history for GBp ticker."""
        mock_ticker = MagicMock()
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
        # GBp prices (in pence)
        mock_hist = pd.DataFrame({
            'Close': [10000.0, 10100.0, 10200.0, 10300.0, 10400.0]
        }, index=dates)
        mock_ticker.history.return_value = mock_hist
        mock_yfinance_ticker.return_value = mock_ticker

        start_date = datetime(2024, 1, 1).date()
        end_date = datetime(2024, 1, 5).date()

        ticker, prices = fetch_and_convert_history(
            'VUSA.L', start_date, end_date, 'GBp', dates  # Need .L for LSE ticker
        )

        # Prices should be divided by 100
        expected = [100.0, 101.0, 102.0, 103.0, 104.0]
        np.testing.assert_array_almost_equal(prices, expected)

    @patch('src.prices.get_cached_prices')
    def test_fetch_and_convert_history_uses_cache(self, mock_get_cached, mock_yfinance_ticker):
        """Test that price fetching uses cache when available."""
        # Mock should return a DataFrame with a 'close' column
        import pandas as pd
        mock_df = pd.DataFrame({
            'close': [100.0, 101.0, 102.0, 103.0, 104.0]
        }, index=pd.date_range('2024-01-01', '2024-01-05', freq='D'))
        mock_get_cached.return_value = mock_df

        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
        start_date = datetime(2024, 1, 1).date()
        end_date = datetime(2024, 1, 5).date()

        ticker, prices = fetch_and_convert_history(
            'VUSA.L', start_date, end_date, 'GBP', dates
        )

        # Should return cached prices, not call yfinance
        assert ticker == 'VUSA.L'
        np.testing.assert_array_equal(prices, [100.0, 101.0, 102.0, 103.0, 104.0])
        mock_yfinance_ticker.assert_not_called()

    def test_fetch_and_convert_history_forward_fill(self, mock_yfinance_ticker):
        """Test that missing prices are forward-filled."""
        mock_ticker = MagicMock()
        # Provide data for only first 3 days
        dates_with_data = pd.date_range('2024-01-01', '2024-01-03', freq='D')
        mock_hist = pd.DataFrame({
            'Close': [100.0, 101.0, 102.0]
        }, index=dates_with_data)
        mock_ticker.history.return_value = mock_hist
        mock_yfinance_ticker.return_value = mock_ticker

        # Request 7-day range
        dates = pd.date_range('2024-01-01', '2024-01-07', freq='D')
        start_date = datetime(2024, 1, 1).date()
        end_date = datetime(2024, 1, 7).date()

        ticker, prices = fetch_and_convert_history(
            'VUSA.L', start_date, end_date, 'GBP', dates
        )

        # Should have 7 prices (forward filled)
        assert len(prices) == 7
        # First 3 from data, last 4 forward filled
        assert prices[0] == 100.0
        assert prices[2] == 102.0
        assert prices[3] == 102.0  # Forward filled
        assert prices[6] == 102.0  # Forward filled


@pytest.mark.unit
class TestFetchPricesParallel:
    def test_fetch_prices_parallel_single_ticker(self, mock_yfinance_ticker):
        """Test fetching prices for single ticker in parallel."""
        mock_ticker = MagicMock()
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
        mock_hist = pd.DataFrame({
            'Close': [100.0, 101.0, 102.0, 103.0, 104.0]
        }, index=dates)
        mock_ticker.history.return_value = mock_hist
        mock_yfinance_ticker.return_value = mock_ticker

        start_date = datetime(2024, 1, 1).date()
        end_date = datetime(2024, 1, 5).date()

        price_data = fetch_prices_parallel(
            ['VUSA.L'], start_date, end_date, {'VUSA.L': 'GBP'}, dates
        )

        assert 'VUSA.L' in price_data
        assert len(price_data['VUSA.L']) == 5

    def test_fetch_prices_parallel_multiple_tickers(self, mock_yfinance_ticker):
        """Test fetching prices for multiple tickers in parallel."""
        def create_ticker(ticker_str):
            mock_ticker = MagicMock()
            dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
            if ticker_str == 'VUSA.L':
                prices = [100.0, 101.0, 102.0, 103.0, 104.0]
            else:
                prices = [50.0, 51.0, 52.0, 53.0, 54.0]
            mock_hist = pd.DataFrame({'Close': prices}, index=dates)
            mock_ticker.history.return_value = mock_hist
            mock_ticker.info = {'currency': 'GBP'}
            return mock_ticker

        mock_yfinance_ticker.side_effect = create_ticker

        start_date = datetime(2024, 1, 1).date()
        end_date = datetime(2024, 1, 5).date()
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')

        price_data = fetch_prices_parallel(
            ['VUSA.L', 'VWRL.L'], start_date, end_date,
            {'VUSA.L': 'GBP', 'VWRL.L': 'GBP'}, dates
        )

        assert 'VUSA.L' in price_data
        assert 'VWRL.L' in price_data
        assert price_data['VUSA.L'][0] == 100.0
        assert price_data['VWRL.L'][0] == 50.0

    def test_fetch_prices_parallel_with_invalid_ticker(self, mock_yfinance_ticker):
        """Test fetching prices when one ticker is invalid."""
        def create_ticker(ticker_str):
            mock_ticker = MagicMock()
            if ticker_str == 'VUSA.L':
                dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
                mock_hist = pd.DataFrame({
                    'Close': [100.0, 101.0, 102.0, 103.0, 104.0]
                }, index=dates)
                mock_ticker.history.return_value = mock_hist
            else:
                mock_ticker.history.side_effect = Exception("Ticker not found")
            mock_ticker.info = {'currency': 'GBP'}
            return mock_ticker

        mock_yfinance_ticker.side_effect = create_ticker

        start_date = datetime(2024, 1, 1).date()
        end_date = datetime(2024, 1, 5).date()
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')

        price_data = fetch_prices_parallel(
            ['VUSA.L', 'INVALID.TICKER'], start_date, end_date,
            {'VUSA.L': 'GBP', 'INVALID.TICKER': 'GBP'}, dates
        )

        # Valid ticker should have data, invalid should have zeros or empty
        assert 'VUSA.L' in price_data
        assert 'INVALID.TICKER' in price_data


@pytest.mark.unit
class TestGetNeededCurrencies:
    def test_get_needed_currencies_single_non_gbp(self):
        """Test getting needed currencies for single non-GBP currency."""
        currencies = {'VUSA.L': 'USD'}

        needed = get_needed_currencies(currencies)

        assert 'USD' in needed
        assert len(needed) == 1

    def test_get_needed_currencies_multiple_non_gbp(self):
        """Test getting needed currencies for multiple non-GBP currencies."""
        currencies = {
            'VUSA.L': 'USD',
            'EUNL.DE': 'EUR',
            'VWRL.L': 'GBP'
        }

        needed = get_needed_currencies(currencies)

        assert 'USD' in needed
        assert 'EUR' in needed
        assert 'GBP' not in needed  # GBP should be filtered out
        assert len(needed) == 2

    def test_get_needed_currencies_all_gbp(self):
        """Test getting needed currencies when all are GBP."""
        currencies = {
            'VUSA.L': 'GBP',
            'VWRL.L': 'GBP',
            'IUIT.L': 'GBP'
        }

        needed = get_needed_currencies(currencies)

        assert len(needed) == 0

    def test_get_needed_currencies_with_none(self):
        """Test getting needed currencies when some currencies are None."""
        currencies = {
            'VUSA.L': 'USD',
            'INVALID.TICKER': None,
            'VWRL.L': 'GBP'
        }

        needed = get_needed_currencies(currencies)

        assert 'USD' in needed
        assert None not in needed
        assert len(needed) == 1
