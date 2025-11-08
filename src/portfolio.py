"""
Portfolio calculations module for InvestEngine CSV Server.
Handles monthly net contributions and daily portfolio value calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

from .database import get_connection
from .prices import fetch_prices_parallel, get_currencies_parallel, get_common_start_date, get_needed_currencies
from .tickers import extract_security_and_isin

logger = logging.getLogger(__name__)


def compute_monthly_net_contributions(df) -> List[Dict]:
    """
    Compute monthly net contributions (buys positive, sells negative).
    Returns list of dicts: [{'Month': '2024-01', 'Net_Value': 1000.0}, ...]
    """
    df = df.copy()
    df["Month"] = df["Trade Date/Time"].dt.to_period("M")
    df["Net_Value"] = df.apply(
        lambda row: row["Total Trade Value"]
        if row["Transaction Type"] == "Buy"
        else -row["Total Trade Value"],
        axis=1,
    )
    monthly_net = df.groupby("Month")["Net_Value"].sum().reset_index()
    monthly_net["Month"] = monthly_net["Month"].astype(str)
    return monthly_net.to_dict("records")


def simulate_holdings(
    df, unique_tickers: List[str], start_date, end_date
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Simulate holdings for each ticker over time.
    Returns: (holdings_data, initial_holdings)
    holdings_data: {ticker: daily_holdings_array}
    initial_holdings: {ticker: initial_quantity}
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Calculate initial holdings (before start_date)
    initial_holdings = {}
    for ticker in unique_tickers:
        pre_trades_mask = (df["Ticker"] == ticker) & (df["Trade Date/Time"].dt.date < start_date)
        pre_trades = df[pre_trades_mask].copy()

        if not pre_trades.empty:
            pre_trades["Quantity_Adj"] = np.where(
                pre_trades["Transaction Type"] == "Buy",
                pre_trades["Quantity"],
                -pre_trades["Quantity"],
            )
            initial_qty = pre_trades["Quantity_Adj"].sum()
        else:
            initial_qty = 0.0

        initial_holdings[ticker] = initial_qty
        logger.info(f"Initial holding for {ticker}: {initial_qty}")

    # Simulate daily holdings for each ticker
    holdings_data = {}
    date_list = [d.date() for d in dates]

    for ticker in unique_tickers:
        # Get trades for this ticker from start_date onward
        ticker_trades = df[
            (df["Ticker"] == ticker) & (df["Trade Date/Time"].dt.date >= start_date)
        ].sort_values("Trade Date/Time").copy()

        initial_qty = initial_holdings.get(ticker, 0.0)

        if ticker_trades.empty:
            logger.info(f"No trades for {ticker} from {start_date}, using initial holdings")
            holdings_data[ticker] = np.full(len(dates), initial_qty)
            continue

        # Calculate daily adjustments
        ticker_trades["Quantity_Adj"] = np.where(
            ticker_trades["Transaction Type"] == "Buy",
            ticker_trades["Quantity"],
            -ticker_trades["Quantity"],
        )

        # Aggregate by trade date
        daily_adj = ticker_trades.groupby(ticker_trades["Trade Date/Time"].dt.date)["Quantity_Adj"].sum()

        logger.info(f"{ticker}: {len(daily_adj)} trade dates, total adj: {daily_adj.sum()}")

        # Build daily holdings with forward-fill
        daily_holdings = pd.Series(index=dates, dtype=float)
        cum_qty = initial_qty

        for i, date in enumerate(dates):
            adj_today = daily_adj.get(date.date(), 0.0)
            cum_qty += adj_today
            daily_holdings.iloc[i] = cum_qty

        holdings_data[ticker] = daily_holdings.values

        non_zero_count = (abs(daily_holdings) > 1e-6).sum()
        logger.info(f"{ticker}: {non_zero_count} non-zero days out of {len(daily_holdings)}")

    return holdings_data, initial_holdings


def compute_daily_portfolio_values(
    holdings_data: Dict[str, np.ndarray],
    price_data: Dict[str, np.ndarray],
    dates: pd.DatetimeIndex
) -> List[float]:
    """
    Compute daily portfolio values by multiplying holdings by prices.
    Returns list of daily values in chronological order.
    """
    daily_values = []

    for i in range(len(dates)):
        day_value = 0.0
        for ticker in holdings_data:
            holding = holdings_data[ticker][i]
            if ticker in price_data:
                price = price_data[ticker][i]
                day_value += holding * price
        daily_values.append(float(np.nan_to_num(day_value, nan=0.0)))

    return daily_values


def get_valid_unique_tickers(df) -> List[str]:
    """Get list of unique valid tickers from the dataframe."""
    return df.loc[df["Ticker"] != "Not found", "Ticker"].drop_duplicates().tolist()


def calculate_portfolio_values(df) -> Optional[Dict]:
    """
    Main function to calculate portfolio values.
    Returns dict with monthly_net, daily_dates, and daily_values, or None if error.
    """
    try:
        # Validate data
        if df.empty:
            logger.error("No trades data in database")
            return None

        df = df.copy()
        df["Trade Date/Time"] = pd.to_datetime(
            df["Trade Date/Time"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
        df = df.dropna(subset=["Trade Date/Time"])

        # Get unique valid tickers
        unique_tickers = get_valid_unique_tickers(df)

        if not unique_tickers:
            logger.error("No valid tickers found in trades")
            return None

        # Compute monthly net contributions
        monthly_net = compute_monthly_net_contributions(df)

        # Get date range
        common_start = get_common_start_date(unique_tickers)
        max_date = df["Trade Date/Time"].max().date()

        if not common_start or common_start > max_date:
            logger.error("No valid date range")
            return None

        dates = pd.date_range(start=common_start, end=max_date, freq="D")
        daily_dates = [d.strftime("%Y-%m-%d") for d in dates]

        # Get currencies and needed FX rates
        currencies = get_currencies_parallel(unique_tickers)
        needed_fx = get_needed_currencies(currencies)

        if needed_fx:
            logger.info(f"Needed FX conversions: {needed_fx}")

        # Simulate holdings
        holdings_data, initial_holdings = simulate_holdings(df, unique_tickers, common_start, max_date)

        # Fetch prices
        price_data = fetch_prices_parallel(
            unique_tickers, common_start, max_date, currencies, dates
        )

        # Compute portfolio values
        daily_values = compute_daily_portfolio_values(holdings_data, price_data, dates)

        # Cache prices that weren't fully cached
        cache_new_prices(unique_tickers, price_data, dates)

        return {
            "monthly_net": monthly_net,
            "daily_dates": daily_dates,
            "daily_values": daily_values,
        }

    except Exception as e:
        logger.error(f"Error calculating portfolio values: {e}")
        import traceback
        traceback.print_exc()
        return None


def cache_new_prices(tickers: List[str], price_data: Dict[str, np.ndarray], dates: pd.DatetimeIndex):
    """Cache newly fetched prices to database."""
    conn = get_connection()
    try:
        for ticker in tickers:
            if ticker in price_data:
                prices = pd.Series(price_data[ticker], index=dates)
                for i in range(len(prices)):
                    price_val = prices.iloc[i]
                    if pd.notna(price_val) and price_val > 0:
                        date_str = dates[i].strftime("%Y-%m-%d")
                        conn.execute(
                            "INSERT OR REPLACE INTO prices (ticker, date, close) VALUES (?, ?, ?)",
                            (ticker, date_str, float(price_val))
                        )
        conn.commit()
        logger.info(f"Cached prices for {len(tickers)} tickers")
    finally:
        conn.close()
