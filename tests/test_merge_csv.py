"""
Unit tests for CSV merging functionality.
"""

import pandas as pd
import pytest
from merge_csv import (
    merge_csv_files,
    extract_account_type,
    clean_currency,
    parse_date
)


@pytest.mark.unit
class TestExtractAccountType:
    def test_extract_gia_from_filename(self):
        """Test extracting GIA from filename."""
        assert extract_account_type("GIA_Trading_statement.csv") == "GIA"
        assert extract_account_type("my_GIA_file.csv") == "GIA"

    def test_extract_isa_from_filename(self):
        """Test extracting ISA from filename."""
        assert extract_account_type("ISA_Trading_statement.csv") == "ISA"
        assert extract_account_type("my_ISA_file.csv") == "ISA"

    def test_extract_unknown_account_type(self):
        """Test extracting unknown account type."""
        assert extract_account_type("unknown_file.csv") == "Unknown"


@pytest.mark.unit
class TestCleanCurrency:
    def test_clean_pound_currency(self):
        """Test cleaning pound currency values."""
        assert clean_currency("£152.05") == 152.05
        assert clean_currency("£1,234.56") == 1234.56

    def test_clean_value_without_pound(self):
        """Test cleaning values without pound sign."""
        assert clean_currency("152.05") == 152.05
        assert clean_currency("1234.56") == 1234.56

    def test_clean_na_value(self):
        """Test cleaning NA/missing values."""
        assert clean_currency(pd.NA) == 0.0
        assert clean_currency("") == 0.0

    def test_clean_invalid_value(self):
        """Test cleaning invalid currency values."""
        assert clean_currency("invalid") == 0.0


@pytest.mark.unit
class TestParseDate:
    def test_parse_full_datetime(self):
        """Test parsing full datetime string."""
        result = parse_date("01/11/24 12:45:32")
        assert result.day == 1
        assert result.month == 11
        assert result.year == 2024

    def test_parse_date_only(self):
        """Test parsing date-only string."""
        result = parse_date("22/09/23")
        # Check that result is not NaT before accessing attributes
        assert not pd.isna(result), "Date parsing returned NaT"
        assert result.day == 22
        assert result.month == 9

    def test_parse_invalid_date(self):
        """Test parsing invalid date string."""
        result = parse_date("invalid")
        assert pd.isna(result)


@pytest.mark.unit
class TestMergeCSVFiles:
    def test_merge_single_csv_file(self):
        """Test merging a single CSV file."""
        csv_content = """Trading Statement,
Security / ISIN,Transaction Type,Quantity,Share Price,Total Trade Value,Trade Date/Time,Settlement Date,Broker
Vanguard FTSE 250 ETF Dist / ISIN IE00BYYHSQ67,Buy,5,£30.41,£152.05,01/11/24 12:45:32,05/11/24,InvestEngine
"""
        file_data = [("GIA_Trading_test.csv", csv_content)]
        result = merge_csv_files(file_data)

        assert len(result) == 1
        assert result.iloc[0]["Account_Type"] == "GIA"
        assert result.iloc[0]["Quantity"] == 5
        assert result.iloc[0]["Share Price"] == 30.41
        assert result.iloc[0]["Total Trade Value"] == 152.05

    def test_merge_multiple_csv_files(self, sample_gia_csv, sample_isa_csv):
        """Test merging multiple CSV files."""
        file_data = [
            ("GIA_Trading_test.csv", sample_gia_csv),
            ("ISA_Trading_test.csv", sample_isa_csv)
        ]
        result = merge_csv_files(file_data)

        assert len(result) == 2
        assert result.iloc[0]["Account_Type"] == "GIA"
        assert result.iloc[1]["Account_Type"] == "ISA"

    def test_merge_preserves_transaction_types(self, sample_gia_csv):
        """Test merging preserves different transaction types."""
        csv_content = """Trading Statement,
Security / ISIN,Transaction Type,Quantity,Share Price,Total Trade Value,Trade Date/Time,Settlement Date,Broker
Vanguard FTSE 250 ETF Dist / ISIN IE00BYYHSQ67,Buy,5,£30.41,£152.05,01/11/24 12:45:32,05/11/24,InvestEngine
Vanguard FTSE 250 ETF Dist / ISIN IE00BYYHSQ67,Sell,2,£32.00,£64.00,02/11/24 12:45:32,06/11/24,InvestEngine
"""
        file_data = [("GIA_Trading_test.csv", csv_content)]
        result = merge_csv_files(file_data)

        assert len(result) == 2
        buys = result[result["Transaction Type"] == "Buy"]
        sells = result[result["Transaction Type"] == "Sell"]

        assert len(buys) == 1
        assert len(sells) == 1
        assert buys.iloc[0]["Quantity"] == 5
        assert sells.iloc[0]["Quantity"] == 2

    def test_merge_sorts_by_trade_date(self, sample_gia_csv, sample_isa_csv):
        """Test merging sorts by trade date/time."""
        file_data = [
            ("ISA_Trading_test.csv", sample_isa_csv),  # Earlier date
            ("GIA_Trading_test.csv", sample_gia_csv)   # Later date
        ]
        result = merge_csv_files(file_data)

        # Should be sorted by date
        dates = pd.to_datetime(result["Trade Date/Time"])
        assert dates.is_monotonic_increasing

    def test_merge_invalid_columns(self):
        """Test merging with invalid column names."""
        csv_content = """Trading Statement,
Wrong,Column,Names,Here,Missing,Columns
Some,Data,Here,For,Testing,Purposes
"""
        file_data = [("GIA_Trading_test.csv", csv_content)]

        with pytest.raises(ValueError, match="Invalid columns"):
            merge_csv_files(file_data)

    def test_merge_empty_file_list(self):
        """Test merging with empty file list."""
        with pytest.raises(ValueError, match="No valid CSV data"):
            merge_csv_files([])

    def test_merge_skips_bom(self):
        """Test merging handles BOM in CSV content."""
        csv_content = "\ufeff" + """Trading Statement,
Security / ISIN,Transaction Type,Quantity,Share Price,Total Trade Value,Trade Date/Time,Settlement Date,Broker
Vanguard FTSE 250 ETF Dist / ISIN IE00BYYHSQ67,Buy,5,£30.41,£152.05,01/11/24 12:45:32,05/11/24,InvestEngine
"""
        file_data = [("GIA_Trading_test.csv", csv_content)]
        result = merge_csv_files(file_data)

        assert len(result) == 1
        assert "Security / ISIN" in result.columns

    def test_merge_real_csv_files(self):
        """Test merging with real CSV files from trading_statements."""
        import os

        gia_path = "trading_statements/GIA_Trading_statement_6_Jun_2022_to_7_Nov_2025.csv"
        isa_path = "trading_statements/ISA_Trading_statement_6_Jun_2022_to_7_Nov_2025.csv"

        if not os.path.exists(gia_path) or not os.path.exists(isa_path):
            pytest.skip("Real CSV files not found")

        with open(gia_path, 'r', encoding='utf-8') as f:
            gia_content = f.read()

        with open(isa_path, 'r', encoding='utf-8') as f:
            isa_content = f.read()

        file_data = [
            ("GIA_Trading_statement.csv", gia_content),
            ("ISA_Trading_statement.csv", isa_content)
        ]
        result = merge_csv_files(file_data)

        assert len(result) > 0
        assert "Account_Type" in result.columns
        assert "GIA" in result["Account_Type"].values
        assert "ISA" in result["Account_Type"].values
