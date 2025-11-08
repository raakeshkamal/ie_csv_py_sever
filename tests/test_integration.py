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

            # First upload some data
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
            assert "Ticker" in trade
            assert "Quantity" in trade
            assert isinstance(trade["Trade Date/Time"], str)

    def test_export_trades_format(self):
        """Test that exported trades have correct format."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch.object(src.database, 'DB_PATH', tmp_db_path):
            client = TestClient(app)

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
            assert trade["Ticker"] == "VUSA.L"
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

            # Try to upload second file without resetting
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
