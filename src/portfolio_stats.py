"""
Portfolio statistics module for InvestEngine CSV Server.
Handles Time-Weighted Return (TWR) calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


def calculate_time_weighted_return(
    daily_dates: List[str], 
    daily_values: List[float], 
    monthly_contributions: List[Dict],
    cash_flow_events: Optional[List[Dict]] = None
) -> Optional[Dict]:
    """
    Calculate Time-Weighted Return (TWR) for the portfolio.

    Args:
        daily_dates: List of dates in 'YYYY-MM-DD' format
        daily_values: List of daily portfolio values
        monthly_contributions: List of monthly net contributions (for fallback)
        cash_flow_events: List of dicts with 'date': str, 'net_amount': float for exact cash flows

    Returns:
        Dict containing TWR metrics
    """
    if not daily_dates or not daily_values or len(daily_dates) != len(daily_values):
        return None

    # Convert to pandas Series for easier manipulation
    df = pd.DataFrame({"date": pd.to_datetime(daily_dates), "value": daily_values})
    df = df.set_index("date").sort_index()
    df.index = pd.DatetimeIndex(df.index)

    # Get cashflow dates from cash_flow_events or fallback to monthly
    cashflow_dates = _get_cashflow_dates(monthly_contributions, df.index, cash_flow_events)

    # Calculate sub-period returns
    sub_periods = _calculate_sub_period_returns(df, cashflow_dates)

    # Calculate overall TWR
    twr = _chain_link_returns(sub_periods)

    # Calculate additional metrics
    total_return = _calculate_total_return(df)
    start_date = df.index[0].date()
    end_date = df.index[-1].date()
    annualized_return = _annualize_return(twr, start_date, end_date)

    total_days = (end_date - start_date).days + 1 if start_date != end_date else 1

    return {
        "time_weighted_return": twr,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sub_periods": sub_periods,
        "period_start": df.index[0].strftime("%Y-%m-%d"),
        "period_end": df.index[-1].strftime("%Y-%m-%d"),
        "total_days": total_days,
    }


def _get_cashflow_dates(
    monthly_contributions: List[Dict], 
    date_index: pd.Index,
    cash_flow_events: Optional[List[Dict]] = None
) -> List[pd.Timestamp]:
    """
    Identify dates when cashflows occurred using exact events if provided, else monthly.
    Returns list of timestamps for sub-period ends (just before each cashflow).
    """
    if cash_flow_events:
        cashflow_dates = []
        for cf in cash_flow_events:
            if abs(cf.get('net_amount', 0)) > 1e-6:
                try:
                    ts = pd.Timestamp(cf['date'])
                    if date_index[0] < ts <= date_index[-1]:  # within range
                        cashflow_dates.append(ts)
                except ValueError:
                    logger.warning(f"Invalid date in cash flow: {cf['date']}")
        return sorted(set(cashflow_dates))  # unique, sorted
    else:
        # Fallback to monthly
        if not monthly_contributions:
            return []
        cashflow_dates = []
        for contribution in monthly_contributions:
            month_str = contribution["Month"]  # 'YYYY-MM' format
            net_value = contribution["Net_Value"]
            if abs(net_value) > 1e-6:
                month_start = pd.Timestamp(month_str + "-01")
                if month_start > date_index[0]:
                    cashflow_date = month_start - pd.Timedelta(days=1)
                    cashflow_dates.append(cashflow_date)
        return sorted(cashflow_dates)


def _calculate_sub_period_returns(
    df: pd.DataFrame, 
    cashflow_dates: List[pd.Timestamp]
) -> List[Dict]:
    """
    Calculate returns for each sub-period between cashflows.
    Uses the last available date strictly before the cashflow date as the sub-period end.
    """
    sub_periods = []

    if not cashflow_dates:
        # No cashflows, single period
        start_value = df["value"].iloc[0]
        end_value = df["value"].iloc[-1]

        if start_value > 0:
            period_return = (end_value / start_value) - 1
        else:
            period_return = 0.0

        sub_periods.append(
            {
                "start_date": df.index[0].strftime("%Y-%m-%d"),
                "end_date": df.index[-1].strftime("%Y-%m-%d"),
                "start_value": float(start_value),
                "end_value": float(end_value),
                "return": float(period_return),
            }
        )
        return sub_periods

    # Multiple periods with cashflows
    period_start_idx = 0

    for cashflow_date in cashflow_dates:
        # Find the largest index where df.index < cashflow_date
        pre_mask = df.index < cashflow_date
        if not pre_mask.any():
            continue
        period_end_idx = np.where(pre_mask)[0][-1]

        if period_end_idx <= period_start_idx:
            continue

        # Calculate return for this sub-period
        start_value = df["value"].iloc[period_start_idx]
        end_value = df["value"].iloc[period_end_idx]

        if start_value > 0:
            period_return = (end_value / start_value) - 1
        else:
            period_return = 0.0

        sub_periods.append(
            {
                "start_date": df.index[period_start_idx].strftime("%Y-%m-%d"),
                "end_date": df.index[period_end_idx].strftime("%Y-%m-%d"),
                "start_value": float(start_value),
                "end_value": float(end_value),
                "return": float(period_return),
            }
        )

        # Next period starts on or after cashflow_date
        post_mask = df.index >= cashflow_date
        if post_mask.any():
            next_start_idx = np.where(post_mask)[0][0]
        else:
            next_start_idx = len(df)
        period_start_idx = max(next_start_idx, period_end_idx + 1)

    # Last period (from last cashflow to end)
    if period_start_idx < len(df):
        start_value = df["value"].iloc[period_start_idx]
        end_value = df["value"].iloc[-1]

        if start_value > 0:
            period_return = (end_value / start_value) - 1
        else:
            period_return = 0.0

        sub_periods.append(
            {
                "start_date": df.index[period_start_idx].strftime("%Y-%m-%d"),
                "end_date": df.index[-1].strftime("%Y-%m-%d"),
                "start_value": float(start_value),
                "end_value": float(end_value),
                "return": float(period_return),
            }
        )

    return sub_periods


def _chain_link_returns(sub_periods: List[Dict]) -> float:
    """
    Chain-link sub-period returns to get overall TWR.
    TWR = (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
    """
    if not sub_periods:
        return 0.0

    chained = 1.0
    for period in sub_periods:
        chained *= 1 + period["return"]

    return float(chained - 1)


def _calculate_total_return(df: pd.DataFrame) -> float:
    """
    Calculate total return over the entire period.
    """
    start_value = df["value"].iloc[0]
    end_value = df["value"].iloc[-1]

    if start_value > 0:
        return float((end_value / start_value) - 1)
    return 0.0


def _annualize_return(return_rate: float, start_date: date, end_date: date) -> float:
    """
    Annualize the return rate using (1 + r)^(365 / days) - 1.
    """
    days = (end_date - start_date).days + 1
    if days <= 0:
        return 0.0
    exponent = 365.25 / days
    return float((1 + return_rate) ** exponent - 1)
