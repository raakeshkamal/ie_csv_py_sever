# Test Suite Documentation

## Overview
This test suite validates the InvestEngine CSV Server application, covering unit, integration, and background‑processor tests.

## Structure
```
tests/
├── conftest.py                # Pytest fixtures
├── test_database.py           # Database CRUD & caching
├── test_merge_csv.py          # CSV parsing & merging
├── test_tickers.py            # ISIN → ticker extraction
├── test_prices.py             # Price fetching & FX conversion
├── test_portfolio.py          # Portfolio calculations
├── test_background_processor.py # Pre‑computation logic
└── test_integration.py        # End‑to‑end FastAPI endpoints
```

## Running Tests
```bash
# Install test dependencies
uv sync --extra test
# Run all tests with coverage
pytest --cov=src
```
Use markers to run subsets, e.g. `pytest -m unit` or `pytest -m integration`. Skip slow/network tests with `pytest -m "not slow"`.

## CI Integration
The repository includes a GitHub Actions workflow that runs the full test suite on each push and pull request, generating coverage reports.
