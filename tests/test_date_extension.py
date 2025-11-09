import pytest
import pandas as pd
from datetime import datetime, date, timedelta
from unittest.mock import patch

# Import the function under test
from src.portfolio import calculate_portfolio_values

@pytest.fixture
def sample_trades_df():
    """Create a trades DataFrame with a last trade date in the past.
    The trades are for a single ticker "VUSA.L".
    """
    # Use a trade date two days ago
    two_days_ago = (date.today() - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "Security / ISIN": ["Vanguard FTSE 250 / ISIN IE00BYYHSQ67"],
        "Transaction Type": ["Buy"],
        "Quantity": [5],
        "Share Price": [30.41],
        "Total Trade Value": [152.05],
        "Trade Date/Time": [pd.to_datetime(two_days_ago)],
        "isin": ["IE00BYYHSQ67"],
        "security_name": ["Vanguard FTSE 250"],
        "ticker": ["VUSA.L"]
    }
    return pd.DataFrame(data)

def dummy_get_common_start_date(tickers):
    # Return the earliest possible start date (today minus 4 days)
    return date.today() - timedelta(days=4)

def dummy_get_currencies_parallel(tickers):
    # All tickers are in GBP for simplicity
    return {ticker: "GBP" for ticker in tickers}

def dummy_get_needed_currencies(currencies):
    return []  # No FX needed for GBP

def dummy_fetch_prices_parallel(tickers, start_date, end_date, currencies, dates):
    # Return a constant price array matching the length of dates
    price = 100.0
    price_array = [price for _ in dates]
    return {ticker: price_array for ticker in tickers}

def test_calculate_portfolio_values_extends_to_today(sample_trades_df):
    """Ensure calculate_portfolio_values returns data up to the current date.
    The function should internally extend max_date to today.
    """
    # Patch the helper functions used inside calculate_portfolio_values
    with (
        patch("src.portfolio.get_common_start_date", side_effect=dummy_get_common_start_date),
        patch("src.portfolio.get_currencies_parallel", side_effect=dummy_get_currencies_parallel),
        patch("src.portfolio.get_needed_currencies", side_effect=dummy_get_needed_currencies),
        patch("src.portfolio.fetch_prices_parallel", side_effect=dummy_fetch_prices_parallel)
    ):
        result = calculate_portfolio_values(sample_trades_df)

    # Verify the result structure
    assert result is not None
    assert "daily_dates" in result and "daily_values" in result
    # The number of days should be from common_start to today inclusive
    expected_start = dummy_get_common_start_date(["VUSA.L"]).strftime("%Y-%m-%d")
    expected_end = date.today().strftime("%Y-%m-%d")
    assert result["daily_dates"][0] == expected_start
    assert result["daily_dates"][-1] == expected_end
    # Verify that daily values length matches dates length
    assert len(result["daily_dates"]) == len(result["daily_values"])
    # All values should be non‑negative (price is constant 100, quantity 5)
    assert all(v >= 0 for v in result["daily_values"])
