# Testing Background Processor

This directory contains utilities for testing the background yfinance data collection system.

## Background Processor Overview

The background processor (`src/background_processor.py`) precomputes portfolio values and monthly contributions after CSV uploads, making subsequent API calls much faster.

### New Tables Created:
- `precomputed_portfolio_values` - Daily portfolio values
- `precomputed_monthly_contributions` - Monthly net contributions
- `precompute_status` - Background job status tracking

## Quick Test

The easiest way to test is using the running server with curl commands:

### Step 1: Start the server
```bash
uv run uvicorn src.main:app --reload
```

### Step 2: Reset the database
```bash
curl -X POST http://localhost:8000/reset/
```

### Step 3: Upload CSV files
Use the web interface at http://localhost:8000 or use curl:
```bash
curl -X POST -F "files=@trading_statements/GIA_Trading_statement_example.csv" \
     -F "files=@trading_statements/ISA_Trading_statement_example.csv" \
     http://localhost:8000/upload/
```

### Step 4: Check if background processing completed
Background processing starts automatically after upload. Check status:
```bash
curl http://localhost:8000/portfolio-values/
```

### Step 5: Export precomputed data
```bash
curl http://localhost:8000/export/prices/
```

## Expected Behavior

1. **After Upload**: The `/upload/` endpoint returns immediately, but shows a message that background processing has started
2. **Background Processing**: The system fetches yfinance data for all tickers, converts currencies, and calculates portfolio values
3. **Fast Responses**: Subsequent calls to `/portfolio-values/` return in milliseconds using cached data
4. **Export**: The `/export/prices/` endpoint returns all precomputed data including daily values, monthly contributions, and status

## What Changed

### API Endpoints:
- **POST /upload/** - Now triggers background yfinance data collection
- **GET /portfolio-values/** - Checks precomputed data first (fast!), falls back to live calculation
- **GET /export/prices/** - Exports all precomputed yfinance data and calculations

### Performance:
- First `/portfolio-values/` call after upload: ~5-30 seconds (background processing)
- Subsequent calls: ~10-100ms (from cache)
- No change to response format - fully backward compatible

## Troubleshooting

### If background processing fails:
1. Check server logs for errors
2. Verify tickers were extracted correctly: `curl http://localhost:8000/export/trades/`
3. Try direct calculation: `curl http://localhost:8000/portfolio-values/` (forces recalculation)

### If precomputed data is not available:
- The system falls back to live calculation automatically
- Check `precompute_status` table in database for error messages
- Ensure yfinance API is accessible

## Testing Script

The `test_background_processor.py` script in this directory provides:
- Manual test commands using curl
- Expected responses
- Debugging tips

Run it while the server is running:
```bash
uv run python src/temp-tests/test_background_processor.py
```
