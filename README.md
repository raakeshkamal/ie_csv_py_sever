# InvestEngine CSV Server

A Python web server for uploading batches of trading statement CSV files (e.g., GIA and ISA accounts), merging them into a single unified CSV with an added `Account_Type` column, and generating interactive graphs: a histogram of monthly net contributions (buys positive, sells negative) and a line graph of cumulative portfolio value over time. Each new upload overwrites the previous merged data to keep only the latest batch. Graphs are embedded directly in the page after upload (no download needed).

## Features
- Web interface for batch file uploads.
- Automatic parsing and merging using Pandas (skips header rows, cleans data like dates and currencies).
- Overwrites old merged data on new uploads.
- Generates and embeds Plotly graphs: monthly net contributions histogram and cumulative portfolio value line graph (combines GIA/ISA data).
- Uses FastAPI for the backend, UV for dependency management, and Plotly for interactive visualizations.

## Project Structure
```
.
├── pyproject.toml     # Dependencies managed by UV
├── README.md          # This file
├── src/
│   ├── __init__.py    # (Auto-created if needed)
│   ├── main.py        # FastAPI application
│   ├── merge_csv.py   # CSV merging logic
│   └── templates/
│       └── upload.html # Upload form
└── merged_trading.csv # Output file (generated after upload)
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
2. Select multiple CSV files (e.g., `GIA_Trading_statement_*.csv`, `ISA_Trading_statement_*.csv`).
3. Submit the form. The server will:
   - Parse each file (skip first 2 rows, add `Account_Type` based on filename).
   - Merge into `merged_trading.csv` (overwriting any existing file).
   - Display a success message with embedded interactive graphs: histogram of net monthly contributions and line graph of cumulative portfolio value.
4. Upload another batch to regenerate graphs with new data.

## CSV Format Assumptions
- Row 1: Title (ignored).
- Row 2: Headers (Security / ISIN, Transaction Type, Quantity, Share Price, Total Trade Value, Trade Date/Time, Settlement Date, Broker).
- Data rows: Transactions with £ in currency fields (automatically cleaned).
- Account type extracted from filename (e.g., 'GIA' from 'GIA_*.csv').

## Graphs Explanation
- **Monthly Net Contributions Histogram**: Bars show net cash flow per month (positive for buys/investments, negative for sells/withdrawals).
- **Cumulative Portfolio Value Line Graph**: Running total of net contributions over time (simple invested amount, no market valuation).
- Data combines GIA and ISA for overall portfolio view.

## Development Notes
- Edit code in `src/` and restart/reload as needed.
- For production: Remove `--reload` and use a process manager like Gunicorn.
- Error handling: Invalid CSVs return error messages on the page.

## Troubleshooting
- If dependencies fail: Run `uv sync --dev` or check `pyproject.toml`.
- Server not starting: Ensure Python 3.8+ and UV installed.
- CSV issues: Ensure files match the expected format (see example files in workspace).