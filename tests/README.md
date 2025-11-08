# Test Suite Documentation

## Overview
This comprehensive test suite provides testing for the InvestEngine CSV Server application, covering unit tests, integration tests, database operations, and mocking strategies.

## Test Structure

```
tests/
├── conftest.py                  # Pytest fixtures and test configuration
├── test_database.py             # Unit tests for database operations
├── test_merge_csv.py            # Unit tests for CSV merging functionality
├── test_tickers.py              # Unit tests for ticker extraction and search
├── test_prices.py               # Unit tests for price fetching and currency conversion
├── test_portfolio.py            # Unit tests for portfolio calculation logic
├── test_background_processor.py # Unit tests for background processing
└── test_integration.py          # Integration tests for FastAPI endpoints
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

### Run Specific Test Modules

```bash
# Database tests
pytest tests/test_database.py

# CSV merging tests
pytest tests/test_merge_csv.py

# Ticker extraction tests
pytest tests/test_tickers.py

# Price fetching tests
pytest tests/test_prices.py

# Portfolio calculation tests
pytest tests/test_portfolio.py

# Background processing tests
pytest tests/test_background_processor.py

# Integration tests (requires full app setup)
pytest tests/test_integration.py
```

### Run Tests by Markers

```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Skip slow tests (network-dependent)
pytest -m "not slow"

# Skip tests requiring network access
pytest -m "not requires_network"
```

### Coverage Report
```bash
pytest --cov=src --cov-report=term-missing
```

### Verbose Output
```bash
pytest -v
```

### Stop on First Failure
```bash
pytest -x
```

### Run Specific Test Function
```bash
pytest tests/test_portfolio.py::test_compute_monthly_net_contributions -v
```

## Test Categories

### Database Tests (`@pytest.mark.unit`)
- **Connection Management**: Database connection creation and cleanup
- **Table Operations**: CREATE, DROP, IF EXISTS checks for all tables
- **CRUD Operations**: Insert, update, delete, query operations
- **Caching**: Price and FX rate caching functionality
- **Precomputed Tables**: Creation and validation of precomputed data tables

### CSV Merging Tests (`@pytest.mark.unit`)
- **File Parsing**: CSV reading, BOM handling, header extraction
- **Currency Cleaning**: £ symbol and comma removal, float conversion
- **Date Parsing**: Multiple date format handling (`dd/mm/yy HH:MM:SS`, `dd/mm/yy`)
- **Account Type Detection**: Filename-based extraction (GIA_, ISA_)
- **Data Validation**: Column validation, missing data handling
- **Merge Operations**: Multi-file concatenation, sorting, column ordering

### Ticker Extraction Tests (`@pytest.mark.unit`, `@pytest.mark.requires_network`)
- **ISIN Parsing**: Regex pattern matching for security name and ISIN extraction
- **Yahoo Finance Search**: API integration, response parsing, error handling
- **Ticker Matching**: Preference for LSE exchange (.L suffix), ETF/Equity types
- **Fallback Logic**: Security name search, ISIN-based search
- **Edge Cases**: Invalid ISIN, missing securities, API errors

### Price Fetching Tests (`@pytest.mark.unit`, `@pytest.mark.slow`)
- **YFinance Integration**: Ticker.history() calls, data retrieval
- **Currency Detection**: Ticker.info['currency'] extraction
- **FX Rate Fetching**: GBPUSD=X, EURGBP=X rate retrieval and conversion
- **Currency Conversion**: GBp to GBP conversion (divide by 100)
- **Multi-Currency Handling**: USD, EUR conversion with proper multiply/divide logic
- **Caching**: Price and FX rate caching to SQLite
- **Parallel Processing**: ThreadPoolExecutor for concurrent price fetching

### Portfolio Calculation Tests (`@pytest.mark.unit`)
- **Monthly Net Contributions**: Buy/sell aggregation by month
- **Holdings Simulation**: Cumulative quantity calculation, forward-fill logic
- **Daily Valuations**: Holdings × price calculations across all tickers
- **Date Range Logic**: Common start date detection across tickers
- **Price Integration**: Combination of holdings data with price arrays
- **Invalid Data Handling**: Missing tickers, zero prices, NaN values

### Background Processing Tests (`@pytest.mark.unit`, `@pytest.mark.slow`)
- **Precomputation Triggering**: BackgroundTasks integration
- **Data Collection**: Parallel fetching of prices, currencies, FX rates
- **Table Population**: Writing to precomputed_portfolio_values, precomputed_monthly_contributions
- **Status Tracking**: precompute_status table updates
- **Data Retrieval**: Querying precomputed data with freshness validation
- **Export Functionality**: Full precomputed data export as JSON

### Integration Tests (`@pytest.mark.integration`)
- **FastAPI Endpoints**: All endpoints (/, /reset/, /upload/, /portfolio-values/, /export/)
- **Full Workflow**: End-to-end CSV upload to chart generation
- **File Upload**: Multi-part form data, CSV parsing, error handling
- **Chart Data Generation**: Portfolio value calculation, monthly contributions
- **Database Operations**: Full DB lifecycle in integration context
- **Error Handling**: Invalid files, missing data, API failures
- **Background Processing**: Async task triggering and completion

### Slow Tests (`@pytest.mark.slow`)
- Tests that call real external APIs (yfinance, network requests)
- Tests with full CSV file processing
- Run with `pytest -m slow` or `pytest --runslow`

## Fixtures

### Data Fixtures (conftest.py)
- `sample_gia_csv`: Sample GIA CSV content with realistic transaction data
- `sample_isa_csv`: Sample ISA CSV content
- `sample_trades_df`: Sample trades dataframe with proper dtypes
- `mock_merged_df`: Mock merged dataframe with multiple securities
- `sample_precomputed_portfolio_data`: Precomputed portfolio values for testing retrieval

### Mock Fixtures (conftest.py)
- `mock_yfinance_ticker`: Mocked yfinance Ticker object with history() and info
- `mock_yfinance_fx`: Mocked yfinance for FX rates (GBPUSD=X, EURGBP=X)
- `mock_requests_get`: Mocked requests.get for Yahoo Finance search API
- `mock_currency_detection`: Mocked currency detection for tickers

### Database Fixtures (conftest.py)
- `temp_db_path`: Temporary database path using NamedTemporaryFile
- `mock_db_with_data`: Database populated with sample trades data
- `mock_db_empty`: Fresh database with tables created but no data
- `mock_prices_table`: Database with prices table and test data
- `mock_with_precomputed_data`: Database with all precomputed tables populated

### Configuration Fixtures (conftest.py)
- `client`: FastAPI TestClient instance for endpoint testing
- `fx_config`: FX rate configuration dictionary
- `parallel_worker_count`: ThreadPoolExecutor worker count configuration

## External Service Mocking

The tests extensively mock external services to ensure fast, reliable test execution:

### Yahoo Finance (`yfinance`)
```python
with patch('yfinance.Ticker') as mock_ticker:
    mock_ticker.return_value.history.return_value = mock_price_data
    mock_ticker.return_value.info = {'currency': 'GBp'}
```
**Used in**: test_prices.py, test_portfolio.py, test_background_processor.py, test_integration.py

### HTTP Requests (`requests`)
```python
with patch('requests.get') as mock_get:
    mock_response = Mock()
    mock_response.json.return_value = {"quotes": [{"symbol": "VUSA.L", "exchange": "LSE"}]}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response
```
**Used in**: test_tickers.py, test_integration.py

### Database (`sqlite3`)
```python
with tempfile.NamedTemporaryFile(suffix='.db') as tmp_db:
    with patch('src.database.DB_PATH', tmp_db.name):
        # Test with isolated temporary database
```
**Used in**: All test modules for isolation

### Background Processing (`BackgroundTasks`)
```python
with patch('fastapi.BackgroundTasks.add_task') as mock_add_task:
    # Test that background tasks are triggered correctly
```
**Used in**: test_integration.py for upload endpoint testing

## Test Data

### Sample CSV Files
- Real CSV files located in `trading_statements/` (not committed to repo)
- Used for manual integration testing with real data flows
- Format: InvestEngine trading statements with standard columns

### Mock CSV Data
- Located in `conftest.py` fixtures
- Representative of actual InvestEngine CSV format
- Include both GIA and ISA account examples
- Contains realistic security names, ISINs, quantities, and prices

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install UV
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Install dependencies
        run: uv sync --extra test

      - name: Run tests with coverage
        run: uv run pytest --cov=src --cov-report=xml --cov-report=html

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          fail_ci_if_error: true

      - name: Upload coverage report artifact
        uses: actions/upload-artifact@v3
        with:
          name: coverage-report
          path: htmlcov/
```

### Pre-commit Hook Example

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: Run pytest
        entry: uv run pytest -x
        language: system
        pass_filenames: false
        stages: [commit]
```

## Troubleshooting

### Tests Fail with "No module named 'src'"
Ensure proper Python path configuration:
```bash
export PYTHONPATH=src:$PYTHONPATH
pytest
```
Or use UV's automatic path handling:
```bash
uv run pytest
```

### Module Import Errors
The test suite uses package-qualified imports (`from src.module import function`). Ensure:
1. Project structure is intact
2. `__init__.py` exists in `src/` directory
3. Running from project root directory

### yfinance Rate Limiting
Tests may fail due to yfinance API rate limits. Solutions:
```bash
# Skip slow/network tests
pytest -m "not slow" -m "not requires_network"

# Run with mocking only (unit tests)
pytest -m unit
```

### Database Locks
Integration tests use temporary databases to avoid lock conflicts. If you encounter locks:
1. Ensure no server is running during tests
2. Check for unclosed database connections
3. Use `temp_db_path` fixture for isolation

### Async Test Warnings
For warnings about unawaited coroutines:
1. Ensure proper `pytest-asyncio` configuration
2. Use `async def` for async test functions
3. Add `@pytest.mark.asyncio` decorator where needed

### Coverage Gaps
To identify uncovered code:
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```
Focus on:
- Error handling branches
- External API call paths
- Edge cases in calculations

## Writing New Tests

### Test Template

```python
import pytest
from unittest.mock import Mock, patch
from src.module import function_to_test

@pytest.mark.unit
def test_function_basic():
    """Test basic functionality."""
    # Arrange
    input_data = {...}
    expected = {...}

    # Act
    result = function_to_test(input_data)

    # Assert
    assert result == expected

@pytest.mark.unit
def test_function_edge_cases():
    """Test edge cases and error conditions."""
    # Test with empty data, None values, invalid inputs
    pass

@pytest.mark.integration
def test_function_integration(temp_db_path):
    """Integration test with real database."""
    # Use fixtures for setup
    pass
```

### Best Practices

1. **Use descriptive test names**: `test_calculate_portfolio_value_with_multiple_tickers`
2. **One assertion per test**: Focus on single behavior
3. **Use fixtures**: Leverage existing fixtures in conftest.py
4. **Mock external services**: Never hit real APIs in unit tests
5. **Test error cases**: Invalid inputs, edge cases, exceptions
6. **Use markers**: Properly mark tests as unit/integration/slow
7. **Document fixtures**: Add docstrings to explain fixture purpose
8. **Clean up**: Use pytest's tmp_path for temporary files

## Future Enhancements

### Testing Infrastructure
- [ ] **Performance benchmarks** for large CSV files (1000+ transactions)
- [ ] **Mutation testing** with `mutmut` to identify weak tests
- [ ] **Property-based testing** with `hypothesis` for property verification
- [ ] **API contract testing** with schemathesis or similar
- [ ] **Load testing** for concurrent user simulation
- [ ] **End-to-end testing** with Playwright/Selenium

### Test Coverage Goals
- Target: 90%+ code coverage
- Focus areas:
  - Error handling paths (currently 60%)
  - Background processing edge cases
  - Concurrent operation race conditions
  - Database connection failures

### CI/CD Improvements
- [ ] Parallel test execution with pytest-xdist
- [ ] Test result caching with pytest-cache
- [ ] Automated test result reporting to PRs
- [ ] Performance regression detection
- [ ] Coverage gate enforcement (fail if <85%)
