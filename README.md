# InvestEngine CSV Server

A Python web server for uploading batches of trading statement CSV files (e.g., GIA and ISA accounts), merging them into a unified SQLite database with an added `Account_Type` column and `Ticker` enrichment. The upload endpoint handles merging and saving to DB, while a separate endpoint computes and returns data for interactive graphs: a histogram of monthly net contributions (buys positive, sells negative) and a line graph of actual portfolio value over time (including asset price changes via historical prices from yfinance). Each new upload overwrites the previous merged data to keep only the latest batch. Graphs are embedded directly in the page after upload and chart generation (no download needed). Tickers are extracted in the background using Yahoo Finance API searches.

## Features
- Web interface for batch file uploads using FastAPI and HTMX.
- Automatic parsing and merging using Pandas (skips header rows, cleans data like dates and currencies).
- Stores merged data in SQLite database (`db/merged_trading.db`), overwriting on new uploads.
- Background extraction of stock/ETF tickers for each unique security using Yahoo Finance API (prefers LSE tickers).
- Separate endpoints: `/upload/` for data processing, `/portfolio-values/` for computing monthly net and daily actual portfolio values using yfinance historical prices.
- Actual portfolio value calculation: Simulates daily holdings from trades (buys increase shares, sells decrease), fetches historical closing prices, computes sum(holdings * prices) for each day, with caching in DB for performance.
- Generates and embeds Plotly graphs: monthly net contributions histogram and cumulative portfolio value line graph (combines GIA/ISA data).
- Uses UV for dependency management, Jinja2 for templating, and Plotly for interactive visualizations.

## Project Structure
```
.
├── .gitignore              # Git ignore rules
├── pyproject.toml          # Dependencies managed by UV
├── uv.lock                 # UV lockfile (auto-generated)
├── README.md               # This file
├── .vscode/                # VSCode settings
│   └── settings.json
├── src/
│   ├── __init__.py         # Package init (empty)
│   ├── main.py             # FastAPI application with upload endpoint and background tasks
│   ├── merge_csv.py        # CSV parsing and merging logic
│   ├── extract_tickers.py  # Standalone script for ticker extraction (also called in background)
│   ├── test_merge.py       # Test script for merging sample CSVs
│   └── templates/
│       └── upload.html     # Upload form with HTMX and Plotly integration
├── db/                     # Database directory (auto-created)
│   └── merged_trading.db   # SQLite database (generated after upload)
└── trading_statements/     # Sample CSV files (not committed, add your own)
    ├── GIA_Trading_statement_*.csv
    └── ISA_Trading_statement_*.csv
```

## Setup Instructions

1. **Install UV** (if not already installed):
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Or via Homebrew: `brew install uv`.

2. **Sync Dependencies**:
   Run this in the project root to install dependencies from `pyproject.toml`:
   ```
   uv sync
   ```

3. **Run the Server**:
   ```
   uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   ```
   - `--reload` enables auto-reload for development.
   - Access the app at `http://localhost:8000`.

## Usage
1. Open `http://localhost:8000` in your browser to see the upload form.
2. Select multiple CSV files (e.g., `GIA_Trading_statement_*.csv`, `ISA_Trading_statement_*.csv`) using the "Browse Files" button.
3. The form submits automatically on file selection. The server will:
   - Parse each file (skip first row title, use second as header, add `Account_Type` based on filename like 'GIA' or 'ISA').
   - Merge into `db/merged_trading.db` (overwriting any existing data).
   - Extract tickers in the background for unique securities (adds `Ticker` column; may take a few seconds).
   - Display a success message with total transactions and date range, plus a "Generate Charts" button.
4. Click "Generate Charts" to compute and display interactive graphs: histogram of net monthly contributions and line graph of actual portfolio value over time (including price changes).
5. Upload another batch to regenerate the database; click "Generate Charts" to update graphs.

To test merging without the web interface, run:
```
uv run python src/test_merge.py
```
This uses sample files in `trading_statements/` (add your CSVs there if needed).

To run ticker extraction standalone (e.g., on existing DB):
```
uv run python src/extract_tickers.py
```

## CSV Format Assumptions
- Row 1: Title (ignored).
- Row 2: Headers (Security / ISIN, Transaction Type, Quantity, Share Price, Total Trade Value, Trade Date/Time, Settlement Date, Broker).
- Data rows: Transactions with £ in currency fields (automatically cleaned to floats).
- Dates in format `dd/mm/yy` or `dd/mm/yy HH:MM:SS`.
- Account type extracted from filename prefix (e.g., 'GIA_' → 'GIA'; fallback to search in name).
- Security / ISIN format: e.g., "Vanguard FTSE Developed World UCITS ETF / ISIN GB00B3XXRP09" (regex extracts name and ISIN for ticker lookup).

## Graphs Explanation
- **Monthly Net Contributions Histogram**: Bars show net cash flow per month (positive for buys/investments, negative for sells/withdrawals). Combines GIA and ISA.
- **Actual Portfolio Value Line Graph**: Daily portfolio value over time, simulating holdings from trades and multiplying by historical closing prices from yfinance (includes asset price changes, dividends not included). Aggregates across accounts; invalid tickers skipped (value 0).
- Charts are interactive (zoom, hover) via Plotly and rendered client-side from server-computed data.

## Ticker Extraction
- For each unique "Security / ISIN", searches Yahoo Finance API by security name (prefers LSE/.L tickers, ETF/Equity types).
- Historical prices fetched via yfinance for valid tickers over the trade date range, cached in `prices` table in DB to avoid refetching.
- Fallback to ISIN search if name search fails.
- Results stored in `Ticker` column (e.g., "VUSA.L" or "Not found").
- Rate-limited by Yahoo; for many securities, it may take time (background task handles it post-upload).

## Development Notes
- Edit code in `src/` and restart/reload as needed (use `--reload` flag).
- yfinance usage: Fetches historical 'Close' prices; handles missing data with forward-fill; errors logged but computation continues with 0 for failed tickers.
- Portfolio computation: Holdings simulated by cumulative quantity adjustments on trade dates, forward-filled for non-trade days; value = sum(holdings_per_ticker * price_per_ticker) per day.
- Ruff is configured for linting/formatting in `.vscode/settings.json`.
- Database is overwritten on each upload; no versioning.
- For production: Remove `--reload`, use Gunicorn/HTTPS, and consider caching ticker lookups.
- Error handling: Invalid CSVs or API errors return JSON error messages displayed on the page.
- Debug prints in console for merging, computations, and ticker searches.

## Troubleshooting
- If dependencies fail: Run `uv sync --dev` or check `pyproject.toml` for versions.
- Server not starting: Ensure Python 3.8+ and UV installed. Check for port 8000 conflicts.
- CSV issues: Ensure files match the expected format (see sample in `trading_statements/`). Invalid columns or dates will raise errors.
- Ticker extraction fails: Check internet connection; Yahoo API may rate-limit. Run `src/extract_tickers.py` manually for debugging.
- No graphs: Ensure Plotly CDN loads; check browser console for JS errors.
- Database not created: Upload files first; `db/` directory auto-creates.
- Charts not generating: Ensure ticker extraction completed (wait a few seconds); check browser console for JS errors; verify yfinance fetches succeed (no internet issues).
- Linter errors in VSCode: Ruff/Basedpyright may flag type issues in dynamic code; functionality is unaffected.