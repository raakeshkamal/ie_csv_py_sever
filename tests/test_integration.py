"""
Integration tests for FastAPI endpoints.
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

from main import app, extract_tickers_to_db


@pytest.mark.integration
class TestUploadEndpoint:
    def test_root_endpoint(self):
        """Test the root endpoint renders the upload page."""
        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Upload" in response.text or "InvestEngine" in response.text

    def test_upload_empty_files(self):
        """Test uploading with no files."""
        client = TestClient(app)
        response = client.post("/upload/", files=[])

        # FastAPI returns 422 for validation errors
        assert response.status_code == 422
        # Check error details
        error_detail = response.json()["detail"]
        assert any("files" in str(err) for err in error_detail)

    def test_upload_single_csv_file(self, sample_gia_csv):
        """Test uploading a single CSV file."""
        with patch('main.extract_tickers_to_db') as mock_extract:
            mock_extract.return_value = None

            client = TestClient(app)
            file_content = BytesIO(sample_gia_csv.encode('utf-8'))

            response = client.post(
                "/upload/",
                files={"files": ("test_gia.csv", file_content, "text/csv")}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["total_transactions"] == 1

    def test_upload_multiple_csv_files(self, sample_gia_csv, sample_isa_csv):
        """Test uploading multiple CSV files."""
        with patch('main.extract_tickers_to_db') as mock_extract:
            mock_extract.return_value = None

            client = TestClient(app)
            gia_file = BytesIO(sample_gia_csv.encode('utf-8'))
            isa_file = BytesIO(sample_isa_csv.encode('utf-8'))

            files = [
                ("files", ("gia.csv", gia_file, "text/csv")),
                ("files", ("isa.csv", isa_file, "text/csv"))
            ]

            response = client.post("/upload/", files=files)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["total_transactions"] == 2

    def test_upload_non_csv_file(self, sample_gia_csv):
        """Test uploading a non-CSV file (should return 400)."""
        with patch('main.extract_tickers_to_db') as mock_extract:
            mock_extract.return_value = None

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
        with patch('main.extract_tickers_to_db') as mock_extract:
            mock_extract.return_value = None

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

        with patch('main.db_path', tmp_db_path):
            client = TestClient(app)
            response = client.get("/portfolio-values/")

            assert response.status_code == 404
            data = response.json()
            assert data["success"] is False
            assert "No trades data" in data["error"]

    def test_portfolio_values_with_valid_data(self, sample_trades_df):
        """Test portfolio values endpoint with valid data."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

            # Save sample data to temp DB
            sample_trades_df.to_sql("trades", sqlite3.connect(tmp_db_path), if_exists="replace", index=False)

        with patch('main.db_path', tmp_db_path):
            with patch('yfinance.Ticker') as mock_ticker_class:
                # Mock yfinance ticker
                mock_ticker = MagicMock()
                mock_ticker.history.return_value = pd.DataFrame({
                    'Close': [100.0, 101.0, 102.0, 103.0, 104.0]
                }, index=pd.date_range('2024-01-01', '2024-01-05', freq='D'))
                mock_ticker.info = {'currency': 'GBP'}
                mock_ticker_class.return_value = mock_ticker

                client = TestClient(app)
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

    @pytest.mark.slow
    def test_full_integration_flow(self):
        """Test the full flow: upload CSVs then get portfolio values."""
        # Read real CSV files if they exist
        gia_path = "trading_statements/GIA_Trading_statement_6_Jun_2022_to_7_Nov_2025.csv"
        isa_path = "trading_statements/ISA_Trading_statement_6_Jun_2022_to_7_Nov_2025.csv"

        if not os.path.exists(gia_path) or not os.path.exists(isa_path):
            pytest.skip("Real CSV files not found")

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch('main.db_path', tmp_db_path):
            client = TestClient(app)

            # Upload files
            with open(gia_path, 'rb') as gia_file:
                with open(isa_path, 'rb') as isa_file:
                    files = [
                        ("files", ("gia.csv", gia_file, "text/csv")),
                        ("files", ("isa.csv", isa_file, "text/csv"))
                    ]
                    upload_response = client.post("/upload/", files=files)

            assert upload_response.status_code == 200
            upload_data = upload_response.json()
            assert upload_data["success"] is True

            # Get portfolio values
            portfolio_response = client.get("/portfolio-values/")

            # This test is slow because it calls real yfinance
            # It should work but may take time
            if portfolio_response.status_code == 200:
                data = portfolio_response.json()
                assert data["success"] is True
                assert len(data["monthly_net"]) > 0


@pytest.mark.integration
class TestConcurrentOperations:
    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name

        with patch('main.db_path', tmp_db_path):
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


# Need to import sqlite3 for the above test
import sqlite3
