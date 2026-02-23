"""
Tests for portfolio statistics (IRR, TWR, etc.)
"""

from datetime import date, datetime

import pytest

from src.portfolio_stats import calculate_portfolio_stats, calculate_twr, xirr


def test_xirr_simple():
    # Simple case: Invest 100, get 110 after 1 year -> 10%
    dates = [date(2023, 1, 1), date(2024, 1, 1)]
    amounts = [-100, 110]
    rate = xirr(dates, amounts)
    assert abs(rate - 0.10) < 0.0001


def test_xirr_complex():
    # Invest 1000 at start
    # Invest 500 after 6 months
    # Value 1600 after 1 year
    dates = [date(2023, 1, 1), date(2023, 7, 1), date(2024, 1, 1)]
    amounts = [-1000, -500, 1600]

    # Rough check: Total invested 1500, profit 100.
    # Time weighted average capital ~ 1000*1 + 500*0.5 = 1250
    # Return ~ 100/1250 = 8%
    rate = xirr(dates, amounts)
    assert 0.05 < rate < 0.15


def test_calculate_portfolio_stats_empty():
    stats = calculate_portfolio_stats([], 1000, date(2023, 1, 1))
    assert stats["irr"] == 0.0
    assert stats["total_invested"] == 0.0
    assert stats["profit_loss"] == 0.0


def test_calculate_portfolio_stats_basic():
    # Setup cash flows (DB format: +ve net_amount is Deposit)
    # Deposit 1000 on 2023-01-01
    cash_flows = [{"date": "2023-01-01", "net_amount": 1000.0}]
    current_value = 1100.0
    current_date = date(2024, 1, 1)  # 1 year later

    stats = calculate_portfolio_stats(cash_flows, current_value, current_date)

    # Check IRR (approx 10%)
    assert abs(stats["irr"] - 0.10) < 0.0001
    assert stats["total_invested"] == 1000.0
    assert stats["current_value"] == 1100.0
    assert stats["profit_loss"] == 100.0
    assert stats["return_percentage"] == 0.10


def test_calculate_portfolio_stats_with_withdrawal():
    # Deposit 1000 on 2023-01-01
    # Withdraw 500 on 2023-07-02 (approx mid year)
    # Value 600 on 2024-01-01

    # DB format: Withdrawal is Negative net_amount
    cash_flows = [
        {"date": "2023-01-01", "net_amount": 1000.0},
        {"date": "2023-07-02", "net_amount": -500.0},
    ]
    current_value = 600.0
    current_date = date(2024, 1, 1)

    stats = calculate_portfolio_stats(cash_flows, current_value, current_date)

    # Invested: 1000
    # Withdrawn: 500
    # Current: 600
    # P&L: 600 + 500 - 1000 = 100
    assert stats["profit_loss"] == 100.0
    assert stats["total_invested"] == 1000.0
    assert stats["total_withdrawn"] == 500.0

    # IRR should be positive (around 13-14% perhaps?)
    # 1000 out, 500 back in 0.5yr, 600 back in 1yr.
    # NPV(r) = -1000 + 500/(1+r)^0.5 + 600/(1+r)^1 = 0
    # Let's just check it's positive and consistent
    assert stats["irr"] > 0.10


def test_calculate_twr_basic():
    daily_dates = ["2023-01-01", "2023-07-01", "2024-01-01"]
    daily_values = [1000.0, 1050.0, 1100.0]
    cash_flows = [{"date": "2023-01-01", "net_amount": 1000.0}]
    current_date = date(2024, 1, 1)

    twr = calculate_twr(daily_dates, daily_values, cash_flows, current_date)

    assert 0.08 < twr < 0.12


def test_calculate_twr_with_cash_flow():
    daily_dates = ["2023-01-01", "2023-07-01", "2024-01-01"]
    daily_values = [1000.0, 1550.0, 1650.0]
    cash_flows = [
        {"date": "2023-01-01", "net_amount": 1000.0},
        {"date": "2023-07-01", "net_amount": 500.0},
    ]
    current_date = date(2024, 1, 1)

    twr = calculate_twr(daily_dates, daily_values, cash_flows, current_date)

    assert twr > 0.05


def test_calculate_portfolio_stats_with_twr():
    daily_dates = ["2023-01-01", "2024-01-01"]
    daily_values = [1000.0, 1100.0]
    cash_flows = [{"date": "2023-01-01", "net_amount": 1000.0}]
    current_value = 1100.0
    current_date = date(2024, 1, 1)

    stats = calculate_portfolio_stats(
        cash_flows,
        current_value,
        current_date,
        daily_portfolio_values={
            "daily_dates": daily_dates,
            "daily_values": daily_values,
        },
    )

    assert "twr" in stats
    assert 0.08 < stats["twr"] < 0.12


def test_calculate_twr_empty():
    assert calculate_twr([], [], [], date(2023, 1, 1)) == 0.0
    assert calculate_twr(["2023-01-01"], [1000.0], [], date(2023, 1, 1)) == 0.0
