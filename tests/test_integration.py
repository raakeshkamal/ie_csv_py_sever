"""
Integration tests for FastAPI endpoints.
Tests the complete flow with the refactored codebase.
"""

import os
import tempfile
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from io import BytesIO

import sys
from pathlib import Path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

import src.database
from src.main import app


def setup_isin_mapping(client, isin, ticker):
    """Helper function to set up ISIN to ticker mapping."""
    response = client.post(
        "/mapping/",
        json=[{
            "isin": isin,
            "ticker": ticker
        }]
    )
    return response


@pytest.mark.integration
class TestRootEndpoint:
    def test_root_endpoint_renders_upload_page(self):
        """Test the root endpoint renders the upload page."""
        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Upload" in response.text or "InvestEngine" in response.text


@pytest.mark.integration
class TestResetEndpoint:
    def test_reset_endpoint_clears_database(self):
        """Test that /reset/ clears the database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)

            # Reset first to create tables
            client.post("/reset/")
            # First upload some data (need to setup ISIN→ticker mapping first)
            setup_isin_mapping(client, "IE00BYYHSQ67", "VUSA.L")

            csv_content = """Trading Statement,
Security / ISIN,Transaction Type,Quantity,Share Price,Total Trade Value,Trade Date/Time,Settlement Date,Broker
Vanguard FTSE 250 / ISIN IE00BYYHSQ67,Buy,5,£30.41,£152.05,01/11/24 12:45:32,05/11/24,InvestEngine
"""
            file_content = BytesIO(csv_content.encode('utf-8'))

            with patch('yfinance.Ticker') as mock_ticker_class:
                mock_ticker = MagicMock()
                mock_ticker.info = {'currency': 'GBP'}
                mock_ticker_class.return_value = mock_ticker

                client.post(
                    "/upload/",
                    files={"files": ("test.csv", file_content, "text/csv")}
                )

            # Verify data exists
            from database import has_trades_data
            assert has_trades_data(tmp_db_path) is True

            # Reset database
            response = client.post("/reset/")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "reset" in data["message"].lower()

            # Verify data is cleared
            assert has_trades_data(tmp_db_path) is False

    def test_reset_endpoint_idempotent(self):
        """Test that /reset/ can be called multiple times safely."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)

            # Call reset multiple times
            response1 = client.post("/reset/")
            response2 = client.post("/reset/")
            response3 = client.post("/reset/")

            assert response1.status_code == 200
            assert response2.status_code == 200
            assert response3.status_code == 200


@pytest.mark.integration
class TestExportTradesEndpoint:
    def test_export_trades_empty_database(self):
        """Test exporting trades when database is empty."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)

            response = client.get("/export/trades/")

            assert response.status_code == 404
            data = response.json()
            assert data["success"] is False
            assert "No trades data" in data["error"]

    def test_export_trades_with_data(self):
        """Test exporting trades when database has data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)

            # Reset first to create tables
            client.post("/reset/")
            # Setup ISIN→ticker mappings
            setup_isin_mapping(client, "IE00BYYHSQ67", "VUSA.L")
            setup_isin_mapping(client, "IE00B4L5Y983", "VMID.L")

            # Upload some data first
            csv_content = """Trading Statement,
Security / ISIN,Transaction Type,Quantity,Share Price,Total Trade Value,Trade Date/Time,Settlement Date,Broker
Vanguard FTSE 250 / ISIN IE00BYYHSQ67,Buy,5,£30.41,£152.05,01/11/24 12:45:32,05/11/24,InvestEngine
iShares Core MSCI / ISIN IE00B4L5Y983,Buy,10,£50.00,£500.00,02/11/24 10:30:00,06/11/24,InvestEngine
"""
            file_content = BytesIO(csv_content.encode('utf-8'))

            with patch('yfinance.Ticker') as mock_ticker_class:
                mock_ticker = MagicMock()
                mock_ticker.info = {'currency': 'GBP'}
                mock_ticker_class.return_value = mock_ticker

                with patch('requests.get') as mock_requests:
                    mock_response = MagicMock()
                    mock_response.json.return_value = {
                        "quotes": [{"symbol": "VUSA.L", "exchange": "LSE", "quoteType": "ETF"}]
                    }
                    mock_requests.return_value = mock_response

                    client.post(
                        "/upload/",
                        files={"files": ("test.csv", file_content, "text/csv")}
                    )

            # Export trades
            response = client.get("/export/trades/")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "trades" in data
            assert len(data["trades"]) == 2

            # Verify trade structure
            trade = data["trades"][0]
            assert "Security / ISIN" in trade
            assert "ticker" in trade
            assert "Quantity" in trade
            assert isinstance(trade["Trade Date/Time"], str)

    def test_export_trades_format(self):
        """Test that exported trades have correct format."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)

            # Reset first to create tables
            client.post("/reset/")
            # Setup ISIN→ticker mapping
            setup_isin_mapping(client, "IE00BYYHSQ67", "VUSA.L")

            # Upload one trade
            csv_content = """Trading Statement,
Security / ISIN,Transaction Type,Quantity,Share Price,Total Trade Value,Trade Date/Time,Settlement Date,Broker
Vanguard FTSE 250 / ISIN IE00BYYHSQ67,Buy,5,£30.41,£152.05,01/11/24 12:45:32,05/11/24,InvestEngine
"""
            file_content = BytesIO(csv_content.encode('utf-8'))

            with patch('yfinance.Ticker') as mock_ticker_class:
                mock_ticker = MagicMock()
                mock_ticker.info = {'currency': 'GBP'}
                mock_ticker_class.return_value = mock_ticker

                with patch('requests.get') as mock_requests:
                    mock_response = MagicMock()
                    mock_response.json.return_value = {
                        "quotes": [{"symbol": "VUSA.L", "exchange": "LSE", "quoteType": "ETF"}]
                    }
                    mock_requests.return_value = mock_response

                    client.post(
                        "/upload/",
                        files={"files": ("test.csv", file_content, "text/csv")}
                    )

            # Export and verify format
            response = client.get("/export/trades/")

            trade = response.json()["trades"][0]

            # Check numeric values
            assert trade["Quantity"] == 5
            assert trade["Share Price"] == 30.41
            assert trade["Total Trade Value"] == 152.05

            # Check string values
            assert "IE00BYYHSQ67" in trade["Security / ISIN"]
            assert trade["ticker"] == "VUSA.L"
            assert trade["Transaction Type"] == "Buy"


@pytest.mark.integration
class TestUploadEndpoint:
    def test_upload_empty_files(self):
        """Test uploading with no files."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)
            response = client.post("/upload/", files=[])

            # FastAPI returns 422 for validation errors
            assert response.status_code == 422

    def test_upload_single_csv_file(self, sample_gia_csv):
        """Test uploading a single CSV file."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)
            # Reset first to create tables
            client.post("/reset/")
            # Setup ISIN→ticker mapping
            setup_isin_mapping(client, "IE00BYYHSQ67", "VUSA.L")
            file_content = BytesIO(sample_gia_csv.encode('utf-8'))

            with patch('requests.get') as mock_requests:
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "quotes": [{"symbol": "VUSA.L", "exchange": "LSE", "quoteType": "ETF"}]
                }
                mock_requests.return_value = mock_response

                response = client.post(
                    "/upload/",
                    files={"files": ("test_gia.csv", file_content, "text/csv")}
                )

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["total_transactions"] == 1
                assert "message" in data

    def test_upload_multiple_csv_files(self, sample_gia_csv, sample_isa_csv):
        """Test uploading multiple CSV files."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)
            # Reset first to create tables
            client.post("/reset/")
            # Setup ISIN→ticker mappings for both files
            setup_isin_mapping(client, "IE00BYYHSQ67", "VUSA.L")
            setup_isin_mapping(client, "IE00BK5BQT80", "VWRL.L")
            gia_file = BytesIO(sample_gia_csv.encode('utf-8'))
            isa_file = BytesIO(sample_isa_csv.encode('utf-8'))

            with patch('requests.get') as mock_requests:
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "quotes": [{"symbol": "VUSA.L", "exchange": "LSE", "quoteType": "ETF"}]
                }
                mock_requests.return_value = mock_response

                files = [
                    ("files", ("gia.csv", gia_file, "text/csv")),
                    ("files", ("isa.csv", isa_file, "text/csv"))
                ]

                response = client.post("/upload/", files=files)

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["total_transactions"] == 2

    def test_upload_without_resetting_first(self):
        """Test that upload fails without calling reset first."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)

            # Reset first to create tables
            client.post("/reset/")

            # Setup ISIN→ticker mapping for first upload
            setup_isin_mapping(client, "IE00BYYHSQ67", "VUSA.L")

            # Upload first file
            csv_content_first = """Trading Statement,
Security / ISIN,Transaction Type,Quantity,Share Price,Total Trade Value,Trade Date/Time,Settlement Date,Broker
Vanguard FTSE 250 / ISIN IE00BYYHSQ67,Buy,5,£30.41,£152.05,01/11/24 12:45:32,05/11/24,InvestEngine
"""
            file_content = BytesIO(csv_content_first.encode('utf-8'))

            with patch('requests.get') as mock_requests:
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "quotes": [{"symbol": "VUSA.L", "exchange": "LSE", "quoteType": "ETF"}]
                }
                mock_requests.return_value = mock_response

                response1 = client.post(
                    "/upload/",
                    files={"files": ("test.csv", file_content, "text/csv")}
                )

                assert response1.status_code == 200

            # Try to upload second file without resetting (no mapping needed as it will fail before validation)
            csv_content_second = """Trading Statement,
Security / ISIN,Transaction Type,Quantity,Share Price,Total Trade Value,Trade Date/Time,Settlement Date,Broker
iShares Core MSCI / ISIN IE00B4L5Y983,Buy,10,£50.00,£500.00,02/11/24 10:30:00,06/11/24,InvestEngine
"""
            file_content = BytesIO(csv_content_second.encode('utf-8'))

            response2 = client.post(
                "/upload/",
                files={"files": ("test2.csv", file_content, "text/csv")}
            )

            # Should return 400 error
            assert response2.status_code == 400
            data = response2.json()
            assert data["success"] is False
            assert "reset" in data["error"].lower()

    def test_upload_after_reset(self):
        """Test that upload works after calling reset."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)

            # Reset first to create tables
            client.post("/reset/")
            # Setup ISIN→ticker mapping
            setup_isin_mapping(client, "IE00BYYHSQ67", "VUSA.L")

            # Upload first file
            csv_content_first = """Trading Statement,
Security / ISIN,Transaction Type,Quantity,Share Price,Total Trade Value,Trade Date/Time,Settlement Date,Broker
Vanguard FTSE 250 / ISIN IE00BYYHSQ67,Buy,5,£30.41,£152.05,01/11/24 12:45:32,05/11/24,InvestEngine
"""
            file_content = BytesIO(csv_content_first.encode('utf-8'))

            with patch('requests.get') as mock_requests:
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "quotes": [{"symbol": "VUSA.L", "exchange": "LSE", "quoteType": "ETF"}]
                }
                mock_requests.return_value = mock_response

                response1 = client.post(
                    "/upload/",
                    files={"files": ("test.csv", file_content, "text/csv")}
                )

                assert response1.status_code == 200

            # Reset
            reset_response = client.post("/reset/")
            assert reset_response.status_code == 200

            # Setup ISIN→ticker mapping for second security
            setup_isin_mapping(client, "IE00B4L5Y983", "ISF.L")

            # Upload different file after reset
            csv_content_second = """Trading Statement,
Security / ISIN,Transaction Type,Quantity,Share Price,Total Trade Value,Trade Date/Time,Settlement Date,Broker
iShares Core MSCI / ISIN IE00B4L5Y983,Buy,10,£50.00,£500.00,02/11/24 10:30:00,06/11/24,InvestEngine
"""
            file_content = BytesIO(csv_content_second.encode('utf-8'))

            response2 = client.post(
                "/upload/",
                files={"files": ("test2.csv", file_content, "text/csv")}
            )

            # Should succeed now
            assert response2.status_code == 200
            data = response2.json()
            assert data["success"] is True
            assert data["total_transactions"] == 1

    def test_upload_non_csv_file(self, sample_gia_csv):
        """Test uploading a non-CSV file (should return 400)."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)
            file_content = BytesIO(b"This is not a CSV file")

            response = client.post(
                "/upload/",
                files={"files": ("test.txt", file_content, "text/plain")}
            )

            # API returns 400 for invalid files
            assert response.status_code == 400
            data = response.json()
            assert data["success"] is False
            assert "CSV" in data["error"]

    def test_upload_invalid_csv_format(self):
        """Test uploading CSV with invalid format."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)
            invalid_csv = """Wrong,Column,Names
1,2,3
"""
            file_content = BytesIO(invalid_csv.encode('utf-8'))

            response = client.post(
                "/upload/",
                files={"files": ("invalid.csv", file_content, "text/csv")}
            )

            assert response.status_code == 500
            data = response.json()
            assert data["success"] is False
            assert "Invalid columns" in data["error"]


@pytest.mark.integration
class TestPortfolioValuesEndpoint:
    def test_portfolio_values_no_data(self):
        """Test portfolio values endpoint with no data in database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)
            response = client.get("/portfolio-values/")

            assert response.status_code == 404
            data = response.json()
            assert data["success"] is False
            assert "No trades data" in data["error"]

    def test_portfolio_values_with_valid_data(self):
        """Test portfolio values endpoint with valid data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)

            # Reset and setup ISIN mappings
            client.post("/reset/")
            setup_isin_mapping(client, "IE00BYYHSQ67", "VUSA.L")
            setup_isin_mapping(client, "IE00B4L5Y983", "ISF.L")

            # Upload data first
            csv_content_trades = """Trading Statement,
Security / ISIN,Transaction Type,Quantity,Share Price,Total Trade Value,Trade Date/Time,Settlement Date,Broker
Vanguard FTSE 250 / ISIN IE00BYYHSQ67,Buy,5,£30.41,£152.05,01/11/24 12:45:32,05/11/24,InvestEngine
iShares Core MSCI / ISIN IE00B4L5Y983,Buy,10,£50.00,£500.00,02/11/24 10:30:00,06/11/24,InvestEngine
"""
            file_content = BytesIO(csv_content_trades.encode('utf-8'))

            with patch('requests.get') as mock_requests:
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "quotes": [{"symbol": "VUSA.L", "exchange": "LSE", "quoteType": "ETF"}]
                }
                mock_requests.return_value = mock_response

                upload_response = client.post(
                    "/upload/",
                    files={"files": ("trades.csv", file_content, "text/csv")}
                )

                assert upload_response.status_code == 200

            # Mock yfinance for portfolio calculation
            with patch('yfinance.Ticker') as mock_ticker_class:
                # Mock ticker for currency info
                mock_ticker_info = MagicMock()
                mock_ticker_info.info = {'currency': 'GBP'}
                mock_ticker_class.return_value = mock_ticker_info

                # Mock ticker for history
                mock_ticker_hist = MagicMock()
                hist_data = []
                for i in range(30):
                    hist_data.append(100.0 + i)
                dates = pd.date_range('2024-11-01', '2024-11-30', freq='D')
                mock_hist_df = pd.DataFrame({'Close': hist_data}, index=dates)
                mock_ticker_hist.history.return_value = mock_hist_df

                # Handle side_effect for different ticker calls
                def create_ticker(ticker_str):
                    if 'currency' in str(ticker_str):
                        return mock_ticker_info
                    return mock_ticker_hist

                mock_ticker_class.side_effect = create_ticker

                response = client.get("/portfolio-values/")

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert "monthly_net" in data
                assert "daily_dates" in data
                assert "daily_values" in data

                # Should have daily values
                assert len(data["daily_dates"]) > 0
                assert len(data["daily_dates"]) == len(data["daily_values"])


@pytest.mark.integration
@pytest.mark.slow
class TestFullIntegrationFlow:
    def test_full_flow_reset_upload_export_portfolio(self):
        """Test the complete flow: reset, upload, export, portfolio."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)

            # Step 1: Reset
            reset_response = client.post("/reset/")
            assert reset_response.status_code == 200
            assert reset_response.json()["success"] is True

            # Setup ISIN mappings
            setup_isin_mapping(client, "IE00BYYHSQ67", "VUSA.L")
            setup_isin_mapping(client, "IE00B4L5Y983", "ISF.L")

            # Step 2: Upload trades
            csv_content_trades = """Trading Statement,
Security / ISIN,Transaction Type,Quantity,Share Price,Total Trade Value,Trade Date/Time,Settlement Date,Broker
Vanguard FTSE 250 / ISIN IE00BYYHSQ67,Buy,5,£30.41,£152.05,01/11/24 12:45:32,05/11/24,InvestEngine
iShares Core MSCI / ISIN IE00B4L5Y983,Buy,10,£50.00,£500.00,02/11/24 10:30:00,06/11/24,InvestEngine
"""
            file_content = BytesIO(csv_content_trades.encode('utf-8'))

            with patch('requests.get') as mock_requests:
                with patch('yfinance.Ticker') as mock_ticker_class:
                    mock_ticker = MagicMock()
                    mock_ticker.info = {'currency': 'GBP'}
                    mock_ticker_class.return_value = mock_ticker

                    mock_response = MagicMock()
                    mock_response.json.return_value = {
                        "quotes": [{"symbol": "VUSA.L", "exchange": "LSE", "quoteType": "ETF"}]
                    }
                    mock_requests.return_value = mock_response

                    upload_response = client.post(
                        "/upload/",
                        files={"files": ("trades.csv", file_content, "text/csv")}
                    )

                    assert upload_response.status_code == 200
                    assert upload_response.json()["success"] is True

            # Step 3: Export trades
            export_response = client.get("/export/trades/")
            assert export_response.status_code == 200
            assert export_response.json()["success"] is True
            assert len(export_response.json()["trades"]) == 2

            # Step 4: Get portfolio values
            with patch('yfinance.Ticker') as mock_ticker_class:
                mock_ticker_info = MagicMock()
                mock_ticker_info.info = {'currency': 'GBP'}
                mock_ticker_hist = MagicMock()
                hist_data = [100.0 + i for i in range(30)]
                dates = pd.date_range('2024-11-01', '2024-11-30', freq='D')
                mock_hist_df = pd.DataFrame({'Close': hist_data}, index=dates)
                mock_ticker_hist.history.return_value = mock_hist_df

                def create_ticker(ticker_str):
                    if 'currency' in str(ticker_str):
                        return mock_ticker_info
                    return mock_ticker_hist

                mock_ticker_class.side_effect = create_ticker

                portfolio_response = client.get("/portfolio-values/")

                assert portfolio_response.status_code == 200
                portfolio_data = portfolio_response.json()
                assert portfolio_data["success"] is True
                assert "monthly_net" in portfolio_data
                assert "daily_dates" in portfolio_data
                assert "daily_values" in portfolio_data

            # Step 5: Try to upload again without reset (should fail)
            file_content = BytesIO(csv_content_trades.encode('utf-8'))

            upload_again_response = client.post(
                "/upload/",
                files={"files": ("trades.csv", file_content, "text/csv")}
            )

            assert upload_again_response.status_code == 400
            assert "reset" in upload_again_response.json()["error"].lower()


@pytest.mark.integration
class TestConcurrentOperations:
    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)

            # Make multiple concurrent requests
            import threading
            results = []

            def make_request():
                response = client.get("/")
                results.append(response.status_code)

            threads = [threading.Thread(target=make_request) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All requests should succeed
            assert all(code == 200 for code in results)
            assert len(results) == 10

    def test_concurrent_upload_and_reset(self):
        """Test concurrent upload and reset operations."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)

            csv_content = """Trading Statement,
Security / ISIN,Transaction Type,Quantity,Share Price,Total Trade Value,Trade Date/Time,Settlement Date,Broker
Vanguard FTSE 250 / ISIN IE00BYYHSQ67,Buy,5,£30.41,£152.05,01/11/24 12:45:32,05/11/24,InvestEngine
"""

            import threading

            def upload_request():
                file_content = BytesIO(csv_content.encode('utf-8'))
                with patch('requests.get') as mock_requests:
                    mock_response = MagicMock()
                    mock_response.json.return_value = {
                        "quotes": [{"symbol": "VUSA.L", "exchange": "LSE", "quoteType": "ETF"}]
                    }
                    mock_requests.return_value = mock_response

                    client.post(
                        "/upload/",
                        files={"files": ("test.csv", file_content, "text/csv")}
                    )

            def reset_request():
                client.post("/reset/")

            # Run concurrently
            threads = []
            for _ in range(5):
                threads.append(threading.Thread(target=upload_request))
                threads.append(threading.Thread(target=reset_request))

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Should not crash, database should be in valid state
            from database import get_connection
            conn = get_connection(tmp_db_path)
            conn.close()

            # Should be able to upload after concurrent operations
            client.post("/reset/")


@pytest.mark.integration
class TestExportPricesEndpoint:
    """Tests for /export/prices/ endpoint."""

    def test_export_prices_empty_database(self):
        """Test exporting prices when database is empty."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)

            response = client.get("/export/prices/")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "ticker_prices" in data
            assert "status" in data
            assert "count" in data
            assert data["count"] == 0

    def test_export_prices_with_data(self):
        """Test exporting prices when precomputed data exists."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)

            # Reset database and set up ISIN mappings
            client.post("/reset/")
            setup_isin_mapping(client, "IE00BYYHSQ67", "VUSA.L")
            setup_isin_mapping(client, "IE00B4L5Y983", "ISF.L")

            # Upload trades first
            csv_content = """Trading Statement,
Security / ISIN,Transaction Type,Quantity,Share Price,Total Trade Value,Trade Date/Time,Settlement Date,Broker
Vanguard FTSE 250 / ISIN IE00BYYHSQ67,Buy,5,£30.41,£152.05,01/11/24 12:45:32,05/11/24,InvestEngine
iShares Core MSCI / ISIN IE00B4L5Y983,Buy,10,£50.00,£500.00,02/11/24 10:30:00,06/11/24,InvestEngine
"""
            file_content = BytesIO(csv_content.encode('utf-8'))

            with patch('yfinance.Ticker') as mock_ticker_class:
                # Mock ticker for info
                mock_ticker_info = MagicMock()
                mock_ticker_info.info = {'currency': 'GBP'}

                # Mock ticker for history
                mock_ticker_hist = MagicMock()
                dates = pd.date_range('2024-11-01', '2024-12-01', freq='D')
                prices = [31.0 + i * 0.1 for i in range(len(dates))]
                mock_hist_df = pd.DataFrame({'Close': prices}, index=dates)
                mock_ticker_hist.history.return_value = mock_hist_df

                def create_ticker(ticker_str):
                    if 'currency' in str(ticker_str):
                        return mock_ticker_info
                    return mock_ticker_hist

                mock_ticker_class.side_effect = create_ticker

                with patch('requests.get') as mock_requests:
                    mock_response = MagicMock()
                    mock_response.json.return_value = {
                        "quotes": [{"symbol": "VUSA.L", "exchange": "LSE", "quoteType": "ETF"}]
                    }
                    mock_requests.return_value = mock_response

                    upload_response = client.post(
                        "/upload/",
                        files={"files": ("trades.csv", file_content, "text/csv")}
                    )

                    assert upload_response.status_code == 200

            # Get export response
            response = client.get("/export/prices/")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

            # Verify structure for ticker prices
            assert "ticker_prices" in data
            assert "status" in data
            assert "count" in data

            # Verify ticker prices have correct structure
            if len(data["ticker_prices"]) > 0:
                tp = data["ticker_prices"][0]
                assert "ticker" in tp
                assert "date" in tp
                assert "original_currency" in tp
                assert "original_price" in tp
                assert "converted_price_gbp" in tp
                assert "last_updated" in tp

    def test_export_prices_format(self):
        """Test that export prices format is correct."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)

            # Insert precomputed data directly
            import sqlite3
            import datetime
            conn = sqlite3.connect(tmp_db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS precomputed_ticker_prices (
                    ticker TEXT,
                    date DATE,
                    original_currency TEXT,
                    original_price REAL,
                    converted_price_gbp REAL,
                    last_updated TIMESTAMP,
                    PRIMARY KEY (ticker, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS precompute_status (
                    id INTEGER PRIMARY KEY,
                    status TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    total_tickers INTEGER,
                    last_error TEXT
                )
            """)

            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute("""
                INSERT INTO precomputed_ticker_prices
                (ticker, date, original_currency, original_price, converted_price_gbp, last_updated)
                VALUES (?, ?, ?, ?, ?, ?), (?, ?, ?, ?, ?, ?), (?, ?, ?, ?, ?, ?)
            """, (
                "VUSA.L", "2024-12-01", "GBP", 31.5, 31.5, now,
                "VUSA.L", "2024-12-02", "GBP", 31.6, 31.6, now,
                "VWRL.L", "2024-12-01", "USD", 82.5, 65.0, now
            ))
            conn.execute("""
                INSERT INTO precompute_status (status, started_at, completed_at, total_tickers)
                VALUES (?, ?, ?, ?)
            """, ("completed", datetime.datetime.now(), datetime.datetime.now(), 2))
            conn.commit()
            conn.close()

            response = client.get("/export/prices/")

            assert response.status_code == 200
            data = response.json()

            # Verify all data is present
            assert len(data["ticker_prices"]) == 3

            # Verify structure of ticker prices
            for tp in data["ticker_prices"]:
                assert "ticker" in tp
                assert "date" in tp
                assert "original_currency" in tp
                assert "original_price" in tp
                assert "converted_price_gbp" in tp
                assert "last_updated" in tp

            # Verify correct ordering (should be sorted by ticker, then date)
            assert data["ticker_prices"][0]["ticker"] == "VUSA.L"
            assert data["ticker_prices"][0]["date"] == "2024-12-01"
            assert data["ticker_prices"][0]["original_price"] == 31.5
            assert data["ticker_prices"][2]["ticker"] == "VWRL.L"
            assert data["ticker_prices"][2]["original_currency"] == "USD"
            assert data["ticker_prices"][2]["converted_price_gbp"] == 65.0


@pytest.mark.integration
class TestBackgroundProcessingWorkflow:
    """Integration tests for background processing workflow."""

    def test_upload_triggers_background_processing(self):
        """Test that upload triggers background processing and precomputed data is stored."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)

            # Reset first
            client.post("/reset/")

            # Set up ISIN mappings
            setup_isin_mapping(client, "IE00BYYHSQ67", "VUSA.L")
            setup_isin_mapping(client, "IE00B4L5Y983", "ISF.L")

            # Upload trades
            csv_content = """Trading Statement,
Security / ISIN,Transaction Type,Quantity,Share Price,Total Trade Value,Trade Date/Time,Settlement Date,Broker
Vanguard FTSE 250 / ISIN IE00BYYHSQ67,Buy,5,£30.41,£152.05,01/11/24 12:45:32,05/11/24,InvestEngine
iShares Core MSCI / ISIN IE00B4L5Y983,Buy,10,£50.00,£500.00,02/11/24 10:30:00,06/11/24,InvestEngine
"""
            file_content = BytesIO(csv_content.encode('utf-8'))

            with patch('yfinance.Ticker') as mock_ticker_class:
                with patch('requests.get') as mock_requests:
                    # Mock ticker info
                    mock_ticker_info = MagicMock()
                    mock_ticker_info.info = {'currency': 'GBP'}

                    # Mock ticker history
                    mock_ticker_hist = MagicMock()
                    dates = pd.date_range('2024-11-01', '2024-12-01', freq='D')
                    prices = [31.0 + i * 0.1 for i in range(len(dates))]
                    mock_hist_df = pd.DataFrame({'Close': prices}, index=dates)
                    mock_ticker_hist.history.return_value = mock_hist_df

                    def create_ticker(ticker_str):
                        if 'currency' in str(ticker_str):
                            return mock_ticker_info
                        return mock_ticker_hist

                    mock_ticker_class.side_effect = create_ticker

                    mock_response = MagicMock()
                    mock_response.json.return_value = {
                        "quotes": [{"symbol": "VUSA.L", "exchange": "LSE", "quoteType": "ETF"}]
                    }
                    mock_requests.return_value = mock_response

                    # Upload - this triggers background processing
                    upload_response = client.post(
                        "/upload/",
                        files={"files": ("trades.csv", file_content, "text/csv")}
                    )

                    assert upload_response.status_code == 200

                    # Simulate waiting for background processing by directly calling the function
                    # Note: In real scenario, this would run in background
                    from src.background_processor import precompute_portfolio_data
                    from src.database import load_trades
                    df = load_trades(db_path=tmp_db_path)
                    if not df.empty:
                        precompute_portfolio_data(df, db_path=tmp_db_path)

            # Verify precomputed tables have data
            import sqlite3
            conn = sqlite3.connect(tmp_db_path)
            cursor = conn.execute("SELECT COUNT(*) FROM precomputed_portfolio_values")
            pv_count = cursor.fetchone()[0]
            cursor = conn.execute("SELECT COUNT(*) FROM precomputed_monthly_contributions")
            mc_count = cursor.fetchone()[0]
            cursor = conn.execute("SELECT status FROM precompute_status")
            status = cursor.fetchone()[0]
            conn.close()

            assert pv_count > 0, "Precomputed portfolio values should be stored"
            assert mc_count > 0, "Precomputed monthly contributions should be stored"
            assert status == "completed", "Status should be completed"

    def test_portfolio_values_uses_precomputed_data_when_available(self):
        """Test that portfolio-values endpoint returns precomputed data when available and fresh."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)

            # Reset and create tables
            client.post("/reset/")

            # Insert precomputed data directly
            import sqlite3
            import datetime
            conn = sqlite3.connect(tmp_db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS precomputed_portfolio_values (
                    date DATE PRIMARY KEY,
                    daily_value REAL,
                    last_updated TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS precomputed_monthly_contributions (
                    month TEXT PRIMARY KEY,
                    net_value REAL,
                    last_updated TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS precompute_status (
                    id INTEGER PRIMARY KEY,
                    status TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    total_tickers INTEGER,
                    last_error TEXT
                )
            """)

            # Insert fresh data (within last hour)
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute("""
                INSERT INTO precomputed_portfolio_values (date, daily_value, last_updated)
                VALUES (?, ?, ?), (?, ?, ?)
            """, ("2024-12-01", 152.05, now, "2024-12-02", 162.05, now))
            conn.execute("""
                INSERT INTO precomputed_monthly_contributions (month, net_value, last_updated)
                VALUES (?, ?, ?)
            """, ("2024-12", 1000.0, now))
            conn.commit()
            conn.close()

            # Mock trades table
            with patch('src.database.load_trades') as mock_load:
                mock_load.return_value = pd.DataFrame({
                    'Security / ISIN': [],
                    'Transaction Type': [],
                    'Quantity': [],
                    'Share Price': [],
                    'Total Trade Value': [],
                    'Trade Date/Time': [],
                    'Ticker': []
                })

                # Get portfolio values
                response = client.get("/portfolio-values/")

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True

                # Should have the precomputed data
                assert len(data["daily_dates"]) == 2
                assert data["daily_values"] == [152.05, 162.05]
                assert len(data["monthly_net"]) == 1
                assert data["monthly_net"][0]["Net_Value"] == 1000.0

    def test_full_workflow_upload_portfolio_export(self):
        """Test the complete workflow: upload → get portfolio → export prices."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)

            # Reset
            reset_response = client.post("/reset/")
            assert reset_response.status_code == 200

            # Set up ISIN mappings
            setup_isin_mapping(client, "IE00BYYHSQ67", "VUSA.L")
            setup_isin_mapping(client, "IE00B4L5Y983", "ISF.L")

            # Upload trades
            csv_content = """Trading Statement,
Security / ISIN,Transaction Type,Quantity,Share Price,Total Trade Value,Trade Date/Time,Settlement Date,Broker
Vanguard FTSE 250 / ISIN IE00BYYHSQ67,Buy,5,£30.41,£152.05,01/11/24 12:45:32,05/11/24,InvestEngine
iShares Core MSCI / ISIN IE00B4L5Y983,Buy,10,£50.00,£500.00,02/11/24 10:30:00,06/11/24,InvestEngine
"""
            file_content = BytesIO(csv_content.encode('utf-8'))

            with patch('yfinance.Ticker') as mock_ticker_class:
                with patch('requests.get') as mock_requests:
                    mock_ticker_info = MagicMock()
                    mock_ticker_info.info = {'currency': 'GBP'}

                    mock_ticker_hist = MagicMock()
                    dates = pd.date_range('2024-11-01', '2024-12-01', freq='D')
                    prices = [31.0 + i * 0.1 for i in range(len(dates))]
                    mock_hist_df = pd.DataFrame({'Close': prices}, index=dates)
                    mock_ticker_hist.history.return_value = mock_hist_df

                    def create_ticker(ticker_str):
                        if 'currency' in str(ticker_str):
                            return mock_ticker_info
                        return mock_ticker_hist

                    mock_ticker_class.side_effect = create_ticker

                    mock_response = MagicMock()
                    mock_response.json.return_value = {
                        "quotes": [{"symbol": "VUSA.L", "exchange": "LSE", "quoteType": "ETF"}]
                    }
                    mock_requests.return_value = mock_response

                    upload_response = client.post(
                        "/upload/",
                        files={"files": ("trades.csv", file_content, "text/csv")}
                    )

                    assert upload_response.status_code == 200

                    # Simulate background processing
                    from src.background_processor import precompute_portfolio_data
                    from src.database import load_trades
                    df = load_trades(db_path=tmp_db_path)
                    if not df.empty:
                        precompute_portfolio_data(df, db_path=tmp_db_path)

            # Get portfolio values
            portfolio_response = client.get("/portfolio-values/")
            assert portfolio_response.status_code == 200
            portfolio_data = portfolio_response.json()
            assert portfolio_data["success"] is True
            assert "daily_dates" in portfolio_data
            assert "daily_values" in portfolio_data
            assert "monthly_net" in portfolio_data

            # Export prices
            export_response = client.get("/export/prices/")
            assert export_response.status_code == 200
            export_data = export_response.json()
            assert export_data["success"] is True
            assert "ticker_prices" in export_data
            assert "status" in export_data
            assert "count" in export_data

            # Verify ticker prices contain expected fields
            assert len(export_data["ticker_prices"]) > 0
            assert "original_currency" in export_data["ticker_prices"][0]
            assert "original_price" in export_data["ticker_prices"][0]
            assert "converted_price_gbp" in export_data["ticker_prices"][0]
            assert "date" in export_data["ticker_prices"][0]

    def test_portfolio_values_endpoint_no_precomputed_data(self):
        """Test portfolio-values when no precomputed data exists (falls back to live calculation)."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)

            # Reset (creates tables but they're empty)
            client.post("/reset/")

            # Set up ISIN mapping
            setup_isin_mapping(client, "IE00BYYHSQ67", "VUSA.L")

            # Upload trades
            csv_content = """Trading Statement,
Security / ISIN,Transaction Type,Quantity,Share Price,Total Trade Value,Trade Date/Time,Settlement Date,Broker
Vanguard FTSE 250 / ISIN IE00BYYHSQ67,Buy,5,£30.41,£152.05,01/11/24 12:45:32,05/11/24,InvestEngine
"""
            file_content = BytesIO(csv_content.encode('utf-8'))

            with patch('yfinance.Ticker') as mock_ticker_class:
                with patch('requests.get') as mock_requests:
                    mock_ticker_info = MagicMock()
                    mock_ticker_info.info = {'currency': 'GBP'}

                    mock_ticker_hist = MagicMock()
                    dates = pd.date_range('2024-11-01', '2024-12-01', freq='D')
                    prices = [100.0 + i * 0.5 for i in range(len(dates))]
                    mock_hist_df = pd.DataFrame({'Close': prices}, index=dates)
                    mock_ticker_hist.history.return_value = mock_hist_df

                    def create_ticker(ticker_str):
                        if 'currency' in str(ticker_str):
                            return mock_ticker_info
                        return mock_ticker_hist

                    mock_ticker_class.side_effect = create_ticker

                    mock_response = MagicMock()
                    mock_response.json.return_value = {
                        "quotes": [{"symbol": "VUSA.L", "exchange": "LSE", "quoteType": "ETF"}]
                    }
                    mock_requests.return_value = mock_response

                    upload_response = client.post(
                        "/upload/",
                        files={"files": ("trades.csv", file_content, "text/csv")}
                    )
                    assert upload_response.status_code == 200

            # Get portfolio values without precomputing
            with patch('yfinance.Ticker') as mock_ticker_class:
                mock_ticker_info = MagicMock()
                mock_ticker_info.info = {'currency': 'GBP'}

                mock_ticker_hist = MagicMock()
                dates = pd.date_range('2024-11-01', '2024-12-01', freq='D')
                prices = [100.0 + i * 0.5 for i in range(len(dates))]
                mock_hist_df = pd.DataFrame({'Close': prices}, index=dates)
                mock_ticker_hist.history.return_value = mock_hist_df

                def create_ticker(ticker_str):
                    if 'currency' in str(ticker_str):
                        return mock_ticker_info
                    return mock_ticker_hist

                mock_ticker_class.side_effect = create_ticker

                response = client.get("/portfolio-values/")
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                # Should have daily values from live calculation
                assert len(data["daily_dates"]) > 0

