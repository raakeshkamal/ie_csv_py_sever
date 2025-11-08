# Test Suite Documentation

## Overview
This test suite provides comprehensive testing for the InvestEngine CSV Server application, covering unit tests, integration tests, and mocking strategies.

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and test configuration
├── test_merge_csv.py        # Unit tests for CSV merging functionality
├── test_tickers.py          # Unit tests for ticker extraction and search
├── test_portfolio.py        # Unit tests for portfolio calculation logic
└── test_integration.py      # Integration tests for FastAPI endpoints
```

## Running Tests

### Install Dependencies
```bash
uv sync --extra test
```

### Run All Tests
```bash
pytest
```

### Run Specific Test Types

```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

### Coverage Report
```bash
pytest --cov=src --cov-report=term-missing
```

### Verbose Output
```bash
pytest -v
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
- **CSV Merging**: Tests for `merge_csv_files`, currency cleaning, date parsing, account type extraction
- **Ticker Extraction**: Tests for `search_ticker_for_isin`, ISIN pattern matching, Yahoo Finance API integration
- **Portfolio Calculations**: Tests for monthly net calculations, holdings simulation, currency conversion

### Integration Tests (`@pytest.mark.integration`)
- **FastAPI Endpoints**: Tests for `/` (root), `/upload/` (file upload), `/portfolio-values/` (portfolio data)
- **Full Workflow**: Tests end-to-end flow from CSV upload to portfolio calculation
- **Concurrent Operations**: Test concurrent request handling

### Slow Tests (`@pytest.mark.slow`)
- Tests that call real external APIs (yfinance, network requests)
- Tests with full CSV file processing
- Run with `pytest -m slow` or `pytest --runslow`

## Fixtures

### Data Fixtures
- `sample_gia_csv`: Sample GIA CSV content
- `sample_isa_csv`: Sample ISA CSV content
- `sample_trades_df`: Sample trades dataframe
- `mock_merged_df`: Mock merged dataframe

### Mock Fixtures
- `mock_yfinance_ticker`: Mocked yfinance Ticker object
- `mock_yfinance_fx`: Mocked yfinance for FX rates
- `mock_requests_get`: Mocked requests.get for ticker search
- `client`: FastAPI TestClient instance

### Database Fixtures
- `temp_db_path`: Temporary database path
- `mock_db_with_data`: Database with test data
- `mock_db_empty`: Empty database

## External Service Mocking

The tests extensively mock external services:

### Yahoo Finance (`yfinance`)
```python
with patch('yfinance.Ticker') as mock_ticker:
    mock_ticker.return_value.history.return_value = mock_data
```

### HTTP Requests (`requests`)
```python
with patch('requests.get') as mock_get:
    mock_response.json.return_value = {"quotes": [...]}
```

### Database (`sqlite3`)
```python
with tempfile.NamedTemporaryFile() as tmp_db:
    with patch('main.db_path', tmp_db.name):
        # Test with temp database
```

## Test Data

### Sample CSV Files
- Real CSV files located in `trading_statements/`
- Used for integration testing real data flows

### Mock CSV Data
- Located in `conftest.py` fixtures
- Representative of actual InvestEngine CSV format

## CI/CD Integration

Add to your CI pipeline:

```yaml
- name: Run tests
  run: |
    uv sync --extra test
    uv run pytest --cov=src --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Troubleshooting

### Tests Fail with "No module named"
Ensure `src` is in your Python path:
```bash
export PYTHONPATH=src:$PYTHONPATH
pytest
```

### yfinance Rate Limiting
Tests may fail due to yfinance API rate limits. Use mocks:
```bash
pytest -m "not slow"
```

### Database Locks
Integration tests use temporary databases to avoid lock conflicts.

## Future Enhancements

- Add performance benchmarks for large CSV files
- Add mutation testing with `mutmut`
- Add property-based testing with `hypothesis`
- Add API contract testing
- Add load testing for concurrent users
