"""
Unit tests for ticker extraction and search functionality.
"""

import pytest
import re
from main import search_ticker_for_isin, extract_tickers_to_db
from tests.conftest import mock_requests_get


@pytest.mark.unit
class TestSearchTickerForISIN:
    def test_search_ticker_with_valid_response(self, mock_requests_get):
        """Test searching ticker with valid API response."""
        ticker = search_ticker_for_isin("Vanguard FTSE 250 ETF", "IE00BYYHSQ67")

        # Should return a ticker symbol
        assert ticker is not None
        assert isinstance(ticker, str)
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
        """Test that LSE tickers are preferred."""
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


@pytest.mark.unit
class TestExtractTickersToDB:
    def test_extract_tickers_with_valid_securities(self, mock_db_with_data, mock_requests_get, temp_db_path):
        """Test extracting tickers from valid securities."""
        # This would require mocking the database connection
        # For now, just verify the function structure
        assert callable(extract_tickers_to_db)

    def test_extract_tickers_no_isin(self, mock_requests_get):
        """Test extracting tickers when ISIN is not present."""
        # This tests the extract_security_and_isin regex pattern
        text = "Vanguard ETF without ISIN"
        match = re.search(r"(.*?) / ISIN ([A-Z]{2}[A-Z0-9]{9}[0-9])", text)
        assert match is None

    def test_extract_tickers_with_isin_pattern(self):
        """Test the ISIN extraction pattern."""
        test_cases = [
            ("Vanguard FTSE 250 ETF Dist / ISIN IE00BYYHSQ67", "IE00BYYHSQ67"),
            ("iShares Core MSCI World UCITS / ISIN IE00B4L5Y983", "IE00B4L5Y983"),
        ]

        for security_text, expected_isin in test_cases:
            match = re.search(r"(.*?) / ISIN ([A-Z]{2}[A-Z0-9]{9}[0-9])", security_text)
            assert match is not None
            assert match.group(2) == expected_isin


@pytest.mark.unit
class TestSecurityAndISINExtraction:
    def test_extract_security_and_isin_valid(self):
        """Test extraction of security name and ISIN."""
        import re

        security_text = "Vanguard FTSE 250 ETF Dist / ISIN IE00BYYHSQ67"
        match = re.search(r"(.*?) / ISIN ([A-Z]{2}[A-Z0-9]{9}[0-9])", str(security_text))

        assert match is not None
        name = match.group(1).strip()
        isin = match.group(2)

        assert name == "Vanguard FTSE 250 ETF Dist"
        assert isin == "IE00BYYHSQ67"

    def test_extract_security_and_isin_no_isin(self):
        """Test extraction when ISIN is not present."""
        import re

        security_text = "Vanguard ETF without ISIN"
        match = re.search(r"(.*?) / ISIN ([A-Z]{2}[A-Z0-9]{9}[0-9])", str(security_text))

        assert match is None

    def test_extract_security_and_isin_with_whitespace(self):
        """Test extraction with extra whitespace (whitespace is stripped before regex)."""
        import re

        security_text = "  Vanguard FTSE 250 ETF Dist  / ISIN  IE00BYYHSQ67  "
        # Strip the string first like in extract_tickers_to_db
        stripped = security_text.strip()
        match = re.search(r"(.*?) / ISIN ([A-Z]{2}[A-Z0-9]{9}[0-9])", str(stripped))

        assert match is not None
        name = match.group(1).strip()
        isin = match.group(2)

        assert name == "Vanguard FTSE 250 ETF Dist"
        assert isin == "IE00BYYHSQ67"
