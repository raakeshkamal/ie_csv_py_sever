"""
Portfolio statistics module for InvestEngine CSV Server.
Handles Internal Rate of Return (IRR) calculations using exact cash flows.
"""

import logging
from datetime import date, datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_twr(
    daily_dates: List[str],
    daily_values: List[float],
    cash_flow_events: List[Dict],
    current_date: Union[str, date, datetime],
) -> float:
    """
    Calculate Time-Weighted Return (TWR), annualized.

    TWR breaks the return into sub-periods at each cash flow, eliminating
    the impact of cash flows on the return calculation.

    Args:
        daily_dates: List of date strings (YYYY-MM-DD)
        daily_values: List of portfolio values corresponding to daily_dates
        cash_flow_events: List of dicts with 'date' and 'net_amount'
        current_date: The final valuation date

    Returns:
        Annualized TWR as a decimal (e.g., 0.10 = 10%)
    """
    if not daily_dates or not daily_values or len(daily_dates) != len(daily_values):
        return 0.0

    date_to_value = {str(d)[:10]: v for d, v in zip(daily_dates, daily_values)}
    sorted_dates = sorted(date_to_value.keys())

    if len(sorted_dates) < 2:
        return 0.0

    start_date = sorted_dates[0]
    end_date = (
        str(current_date)[:10]
        if hasattr(current_date, "strftime")
        else str(current_date)[:10]
    )

    cash_flows_by_date = {}
    for event in cash_flow_events:
        try:
            d = str(pd.to_datetime(event["date"]).date())
            net_amount = float(event["net_amount"])
            cash_flows_by_date[d] = cash_flows_by_date.get(d, 0.0) + net_amount
        except (ValueError, TypeError):
            continue

    period_dates = set(cash_flows_by_date.keys())
    period_dates.add(start_date)
    period_dates.add(end_date)
    period_dates = sorted([d for d in period_dates if d in date_to_value])

    if len(period_dates) < 2:
        return 0.0

    twr = 1.0
    for i in range(len(period_dates) - 1):
        period_start = period_dates[i]
        period_end = period_dates[i + 1]

        start_val = date_to_value.get(period_start, 0.0)
        end_val = date_to_value.get(period_end, 0.0)

        if start_val <= 0:
            continue

        cash_flow_at_end = cash_flows_by_date.get(period_end, 0.0)

        end_val_before_cf = end_val - cash_flow_at_end
        period_return = end_val_before_cf / start_val - 1.0
        twr *= 1.0 + period_return

    twr -= 1.0

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end_dt - start_dt).days

    if days <= 0:
        return 0.0

    if (1.0 + twr) <= 0:
        return 0.0

    annualized_twr = (1.0 + twr) ** (365.25 / days) - 1.0
    return float(annualized_twr)


def calculate_portfolio_stats(
    cash_flow_events: List[Dict],
    current_value: float,
    current_date: Union[str, date, datetime],
    daily_portfolio_values: Optional[Dict] = None,
) -> Dict:
    """
    Calculate portfolio statistics including IRR, TWR, and Profit/Loss.

    Args:
        cash_flow_events: List of dicts with 'date' and 'net_amount' (from DB).
                          Note: DB 'net_amount' is (Credit - Debit), so:
                          Positive = Deposit (Inflow to account)
                          Negative = Withdrawal (Outflow from account)
        current_value: Total portfolio value at current_date
        current_date: Date of the current valuation
        daily_portfolio_values: Optional dict with 'daily_dates' and 'daily_values'
                                for TWR calculation

    Returns:
        Dict containing IRR, TWR, total invested, and P&L metrics.
    """
    if not cash_flow_events:
        if isinstance(current_date, str):
            curr_date_str = current_date[:10]
        elif isinstance(current_date, datetime):
            curr_date_str = current_date.strftime("%Y-%m-%d")
        else:
            curr_date_str = current_date.strftime("%Y-%m-%d")
        return {
            "irr": 0.0,
            "twr": 0.0,
            "total_invested": 0.0,
            "current_value": current_value,
            "profit_loss": 0.0,
            "return_percentage": 0.0,
            "calc_date": curr_date_str,
        }

    # valid cash flows
    # DB Net Amount: +ve = Deposit, -ve = Withdrawal
    # IRR Perspective: Deposit = Negative (Investment), Withdrawal = Positive (Return)
    # So we negate the DB net_amount for IRR calculation.

    dates = []
    amounts = []
    total_invested = 0.0
    total_withdrawn = 0.0

    # 1. Process historical cash flows
    for event in cash_flow_events:
        try:
            d = pd.to_datetime(event["date"]).date()
            net_amount = float(event["net_amount"])

            # For stats:
            if net_amount > 0:
                total_invested += net_amount
            else:
                total_withdrawn += abs(net_amount)

            # For IRR: Negate (Deposit becomes negative cashflow)
            dates.append(d)
            amounts.append(-net_amount)

        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping invalid cash flow event: {event} - {e}")

    # 2. Add current value as final positive cash flow (liquidation value)
    curr_date_obj = pd.to_datetime(current_date).date()
    dates.append(curr_date_obj)
    amounts.append(current_value)

    # 3. Calculate metrics
    irr = xirr(dates, amounts)

    # 4. Calculate TWR if daily portfolio values provided
    twr = 0.0
    if daily_portfolio_values:
        twr = calculate_twr(
            daily_portfolio_values.get("daily_dates", []),
            daily_portfolio_values.get("daily_values", []),
            cash_flow_events,
            current_date,
        )

    # Simple P&L = Current Value + Withdrawals - Deposits
    profit_loss = current_value + total_withdrawn - total_invested

    # Simple Return % (Money Weighted roughly)
    # If total_invested is 0, handle gracefully
    if total_invested > 0:
        return_percentage = profit_loss / total_invested
    else:
        return_percentage = 0.0

    return {
        "irr": irr,
        "twr": twr,
        "total_invested": total_invested,
        "total_withdrawn": total_withdrawn,
        "current_value": current_value,
        "profit_loss": profit_loss,
        "return_percentage": return_percentage,
        "calc_date": curr_date_obj.strftime("%Y-%m-%d"),
    }


def xirr(dates, amounts, guess=0.1) -> float:
    """
    Calculate the Internal Rate of Return (XIRR) for irregular intervals.
    Uses Newton-Raphson method.
    """
    if len(dates) != len(amounts) or len(dates) < 2:
        return 0.0

    # Sort by date
    data = sorted(zip(dates, amounts), key=lambda x: x[0])
    dates, amounts = zip(*data)

    start_date = dates[0]
    # Calculate fraction of years for each cash flow
    years = np.array([(d - start_date).days / 365.0 for d in dates])
    amounts = np.array(amounts)

    # Check validity: must have at least one positive and one negative value
    if np.all(amounts >= 0) or np.all(amounts <= 0):
        return 0.0

    # Newton-Raphson
    rate = guess
    max_iter = 100
    tol = 1e-6

    for _ in range(max_iter):
        # f(r) = sum( C_i / (1+r)^t_i )
        # To avoid complex numbers with negative base, we iterate on (1+r) or ensure base > 0
        # Usually XIRR is > -100%, so 1+r > 0.

        if rate <= -1.0:
            rate = -0.99  # clamp to avoid division by zero or complex

        # Calculate NPV and derivative
        # factor = (1+rate)^(-t)
        base = 1.0 + rate
        try:
            factors = np.power(base, -years)
            f_val = np.sum(amounts * factors)

            # f'(r) = sum( C_i * -t_i * (1+r)^(-t_i - 1) )
            #       = sum( C_i * -t_i * factor / (1+r) )
            df_val = np.sum(amounts * -years * factors) / base

        except (OverflowError, FloatingPointError, ZeroDivisionError):
            return 0.0

        if abs(f_val) < tol:
            return rate

        if abs(df_val) < 1e-9:  # Derivative too small
            break

        new_rate = rate - f_val / df_val

        if abs(new_rate - rate) < tol:
            return new_rate

        rate = new_rate

    return rate if not pd.isna(rate) else 0.0
