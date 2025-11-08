"""
Unit tests for ticker extraction and search functionality.
Tests the tickers module functions directly.
"""

import pytest
import re
import sys
from pathlib import Path

src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

from src.tickers import (
    extract_security_and_isin,
    search_ticker_for_isin,
    extract_tickers_for_df,
    add_tickers_to_df
)


@pytest.mark.unit
class TestExtractSecurityAndISIN:
    def test_extract_security_and_isin_valid(self):
        """Test extraction of security name and ISIN from valid text."""
        security_text = "Vanguard FTSE 250 ETF Dist / ISIN IE00BYYHSQ67"
        name, isin = extract_security_and_isin(security_text)

        assert name == "Vanguard FTSE 250 ETF Dist"
        assert isin == "IE00BYYHSQ67"

    def test_extract_security_and_isin_no_isin(self):
        """Test extraction when ISIN is not present."""
        security_text = "Vanguard ETF without ISIN"
        name, isin = extract_security_and_isin(security_text)

        assert name == "Vanguard ETF without ISIN"
        assert isin is None

    def test_extract_security_and_isin_with_whitespace(self):
        """Test extraction with extra whitespace."""
        security_text = "  Vanguard FTSE 250 ETF Dist  / ISIN  IE00BYYHSQ67  "
        name, isin = extract_security_and_isin(security_text)

        assert name == "Vanguard FTSE 250 ETF Dist"
        assert isin == "IE00BYYHSQ67"

    def test_extract_security_and_isin_multiple_spaces(self):
        """Test extraction with multiple spaces in security name."""
        security_text = "iShares Core MSCI World UCITS ETF / ISIN IE00B4L5Y983"
        name, isin = extract_security_and_isin(security_text)

        assert name == "iShares Core MSCI World UCITS ETF"
        assert isin == "IE00B4L5Y983"

    def test_extract_security_and_isin_numeric_security(self):
        """Test extraction with numeric security names."""
        security_text = "SPDR S&P 500 ETF Trust / ISIN US78462F1030"
        name, isin = extract_security_and_isin(security_text)

        assert name == "SPDR S&P 500 ETF Trust"
        assert isin == "US78462F1030"


@pytest.mark.unit
class TestSearchTickerForISIN:
    def test_search_ticker_with_valid_response(self, mock_requests_get):
        """Test searching ticker with valid API response."""
        ticker = search_ticker_for_isin("Vanguard FTSE 250 ETF", "IE00BYYHSQ67")

        # Should return a ticker symbol
        assert ticker is not None
        assert isinstance(ticker, str)
        assert ticker == "VUSA.L"
        mock_requests_get.assert_called()

    def test_search_ticker_api_error(self, mock_requests_get):
        """Test searching ticker with API error."""
        mock_requests_get.side_effect = Exception("Network error")

        ticker = search_ticker_for_isin("Invalid ETF", "INVALID")
        assert ticker is None

    def test_search_ticker_no_quotes(self, mock_requests_get):
        """Test searching ticker with no quotes in response."""
        mock_response = mock_requests_get.return_value
        mock_response.json.return_value = {"quotes": []}

        ticker = search_ticker_for_isin("Unknown ETF", "UNKNOWN")
        assert ticker is None

    def test_search_ticker_lse_preference(self, mock_requests_get):
        """Test that LSE tickers are preferred over other exchanges."""
        mock_response = mock_requests_get.return_value
        mock_response.json.return_value = {
            "quotes": [
                {
                    "symbol": "VUSA",
                    "exchange": "NYSE",
                    "quoteType": "ETF"
                },
                {
                    "symbol": "VUSA.L",
                    "exchange": "LSE",
                    "quoteType": "ETF"
                }
            ]
        }

        ticker = search_ticker_for_isin("Vanguard S&P 500", "IE00B3XXRP09")
        assert ticker == "VUSA.L"

    def test_search_ticker_fallback_to_isin(self, mock_requests_get):
        """Test fallback to ISIN search when name search fails."""
        # First call (name search) returns no quotes
        # Second call (ISIN search) returns valid data
        mock_response = mock_requests_get.return_value
        mock_response.json.side_effect = [
            {"quotes": []},  # First call - name search
            {  # Second call - ISIN search
                "quotes": [
                    {
                        "symbol": "VWRL.L",
                        "exchange": "LSE",
                        "quoteType": "ETF"
                    }
                ]
            }
        ]

        ticker = search_ticker_for_isin("Vanguard FTSE All-World", "IE00BK5BQT80")
        # Should have made 2 calls
        assert mock_requests_get.call_count == 2
        assert ticker == "VWRL.L"

    def test_search_ticker_multiple_etfs_on_lse(self, mock_requests_get):
        """Test selection when multiple ETFs available on LSE."""
        mock_response = mock_requests_get.return_value
        mock_response.json.return_value = {
            "quotes": [
                {
                    "symbol": "VWRL.L",
                    "exchange": "LSE",
                    "quoteType": "ETF"
                },
                {
                    "symbol": "VWRP.L",
                    "exchange": "LSE",
                    "quoteType": "ETF"
                },
                {
                    "symbol": "VHVG.L",
                    "exchange": "LSE",
                    "quoteType": "EQUITY"
                }
            ]
        }

        ticker = search_ticker_for_isin("Vanguard FTSE All-World", "IE00BK5BQT80")
        # Should return the first ETF on LSE
        assert ticker in ["VWRL.L", "VWRP.L"]


@pytest.mark.unit
class TestExtractTickersForDF:
    def test_extract_tickers_for_df_unique_securities(self, mock_requests_get):
        """Test extracting tickers for a dataframe with unique securities."""
        data = {
            'Security / ISIN': [
                'Vanguard FTSE 250 ETF Dist / ISIN IE00BYYHSQ67',
                'Vanguard FTSE All-World ETF / ISIN IE00BK5BQT80',
                'iShares MSCI World / ISIN IE00B4L5Y983'
            ]
        }
        # Mock returns different tickers for different calls
        mock_response = mock_requests_get.return_value
        mock_response.json.side_effect = [
            {"quotes": [{"symbol": "VUSA.L", "exchange": "LSE", "quoteType": "ETF"}]},
            {"quotes": [{"symbol": "VWRL.L", "exchange": "LSE", "quoteType": "ETF"}]},
            {"quotes": [{"symbol": "IUIT.L", "exchange": "LSE", "quoteType": "ETF"}]}
        ]

        securities_dict = extract_tickers_for_df(data)

        assert len(securities_dict) == 3
        assert securities_dict['Vanguard FTSE 250 ETF Dist / ISIN IE00BYYHSQ67'] == 'VUSA.L'
        assert securities_dict['Vanguard FTSE All-World ETF / ISIN IE00BK5BQT80'] == 'VWRL.L'
        assert securities_dict['iShares MSCI World / ISIN IE00B4L5Y983'] == 'IUIT.L'

    def test_extract_tickers_for_df_duplicate_securities(self, mock_requests_get):
        """Test extracting tickers when securities appear multiple times."""
        data = {
            'Security / ISIN': [
                'Vanguard FTSE 250 ETF Dist / ISIN IE00BYYHSQ67',
                'Vanguard FTSE 250 ETF Dist / ISIN IE00BYYHSQ67',
                'iShares MSCI World / ISIN IE00B4L5Y983',
                'Vanguard FTSE 250 ETF Dist / ISIN IE00BYYHSQ67'
            ]
        }
        # Should only make 2 API calls (for unique securities)
        mock_response = mock_requests_get.return_value
        mock_response.json.side_effect = [
            {"quotes": [{"symbol": "VUSA.L", "exchange": "LSE", "quoteType": "ETF"}]},
            {"quotes": [{"symbol": "IUIT.L", "exchange": "LSE", "quoteType": "ETF"}]}
        ]

        securities_dict = extract_tickers_for_df(data)

        assert len(securities_dict) == 2
        assert securities_dict['Vanguard FTSE 250 ETF Dist / ISIN IE00BYYHSQ67'] == 'VUSA.L'
        assert securities_dict['iShares MSCI World / ISIN IE00B4L5Y983'] == 'IUIT.L'
        # Should only have 2 API calls despite 4 rows
        assert mock_requests_get.call_count == 2

    def test_extract_tickers_for_df_no_isin(self, mock_requests_get):
        """Test extracting tickers when some securities don't have ISIN."""
        data = {
            'Security / ISIN': [
                'Vanguard FTSE 250 ETF Dist / ISIN IE00BYYHSQ67',
                'Unknown ETF without ISIN',
                'iShares MSCI World / ISIN IE00B4L5Y983'
            ]
        }
        mock_response = mock_requests_get.return_value
        mock_response.json.side_effect = [
            {"quotes": [{"symbol": "VUSA.L", "exchange": "LSE", "quoteType": "ETF"}]},
            {"quotes": []},  # Second security has no ticker
            {"quotes": [{"symbol": "IUIT.L", "exchange": "LSE", "quoteType": "ETF"}]}
        ]

        securities_dict = extract_tickers_for_df(data)

        assert len(securities_dict) == 3
        assert securities_dict['Vanguard FTSE 250 ETF Dist / ISIN IE00BYYHSQ67'] == 'VUSA.L'
        assert securities_dict['Unknown ETF without ISIN'] is None
        assert securities_dict['iShares MSCI World / ISIN IE00B4L5Y983'] == 'IUIT.L'


@pytest.mark.unit
class TestAddTickersToDF:
    def test_add_tickers_to_df(self, mock_requests_get):
        """Test adding ticker column to dataframe."""
        data = {
            'Security / ISIN': [
                'Vanguard FTSE 250 ETF Dist / ISIN IE00BYYHSQ67',
                'Vanguard FTSE All-World ETF / ISIN IE00BK5BQT80'
            ]
        }
        import pandas as pd
        df = pd.DataFrame(data)

        # Mock API responses
        mock_response = mock_requests_get.return_value
        mock_response.json.side_effect = [
            {"quotes": [{"symbol": "VUSA.L", "exchange": "LSE", "quoteType": "ETF"}]},
            {"quotes": [{"symbol": "VWRL.L", "exchange": "LSE", "quoteType": "ETF"}]}
        ]

        result_df = add_tickers_to_df(df)

        assert 'Ticker' in result_df.columns
        assert result_df.iloc[0]['Ticker'] == 'VUSA.L'
        assert result_df.iloc[1]['Ticker'] == 'VWRL.L'

    def test_add_tickers_to_df_not_found(self, mock_requests_get):
        """Test adding tickers when some are not found."""
        data = {
            'Security / ISIN': [
                'Vanguard FTSE 250 ETF Dist / ISIN IE00BYYHSQ67',
                'Unknown ETF without ISIN'
            ]
        }
        import pandas as pd
        df = pd.DataFrame(data)

        mock_response = mock_requests_get.return_value
        mock_response.json.side_effect = [
            {"quotes": [{"symbol": "VUSA.L", "exchange": "LSE", "quoteType": "ETF"}]},
            {"quotes": []}  # Not found
        ]

        result_df = add_tickers_to_df(df)

        assert 'Ticker' in result_df.columns
        assert result_df.iloc[0]['Ticker'] == 'VUSA.L'
        assert result_df.iloc[1]['Ticker'] == 'Not found'

    def test_add_tickers_to_df_uses_cache(self, mock_requests_get):
        """Test that duplicate securities use cached tickers."""
        data = {
            'Security / ISIN': [
                'Vanguard FTSE 250 ETF Dist / ISIN IE00BYYHSQ67',
                'Vanguard FTSE 250 ETF Dist / ISIN IE00BYYHSQ67',
                'iShares MSCI World / ISIN IE00B4L5Y983'
            ]
        }
        import pandas as pd
        df = pd.DataFrame(data)

        mock_response = mock_requests_get.return_value
        mock_response.json.side_effect = [
            {"quotes": [{"symbol": "VUSA.L", "exchange": "LSE", "quoteType": "ETF"}]},
            {"quotes": [{"symbol": "IUIT.L", "exchange": "LSE", "quoteType": "ETF"}]}
        ]

        result_df = add_tickers_to_df(df)

        assert 'Ticker' in result_df.columns
        assert result_df.iloc[0]['Ticker'] == 'VUSA.L'
        assert result_df.iloc[1]['Ticker'] == 'VUSA.L'  # Should use cache
        assert result_df.iloc[2]['Ticker'] == 'IUIT.L'
        # Should only make 2 API calls (for unique securities)
        assert mock_requests_get.call_count == 2
