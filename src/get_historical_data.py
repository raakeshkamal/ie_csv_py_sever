import yfinance as yf
import pandas as pd

def get_historical_data(ticker='CSH2.L'):
    """
    Fetch all available historical data for a given stock ticker using yfinance.

    Parameters:
    -----------
    ticker : str, default 'AAPL'
        The stock ticker symbol (e.g., 'AAPL' for Apple Inc.).

    Returns:
    --------
    pandas.DataFrame
        Historical data with columns: Open, High, Low, Close, Adj Close, Volume.
        Indexed by Date.

    Raises:
    -------
    ValueError
        If the ticker is invalid or no data is found.
    """
    try:
        data = yf.download(ticker, period='max')
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        return data
    except Exception as e:
        raise ValueError(f"Error fetching data for {ticker}: {str(e)}")


if __name__ == "__main__":
    try:
        df = get_historical_data()
        print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"Data shape: {df.shape}")
    except ValueError as e:
        print(f"Error: {e}")
