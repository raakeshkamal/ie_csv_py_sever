# InvestEngine CSV Server

A Python web server for uploading batches of trading statement CSV files (e.g., GIA and ISA accounts), merging them into a unified SQLite database with an added `Account_Type` column and `Ticker` enrichment. The upload endpoint handles merging and saving to DB, while a separate endpoint computes and returns data for interactive graphs: a histogram of monthly net contributions (buys positive, sells negative) and a line graph of actual portfolio value over time (including asset price changes via historical prices from yfinance). Each new upload overwrites the previous merged data to keep only the latest batch. Graphs are embedded directly in the page after upload and chart generation (no download needed). Tickers are extracted in the background using Yahoo Finance API searches.

## Features

### Core Functionality
- **Web interface** for batch file uploads using FastAPI and HTMX
- **Automatic CSV parsing** and merging of trading statements using Pandas
- **CSV preprocessing**: Skips title rows, cleans currency formatting (£ and commas), parses dates, standardizes columns
- **Database storage** in SQLite (`db/merged_trading.db`) with overwrite-on-upload behavior

### Ticker Extraction & Price Data
- **Yahoo Finance integration** for ticker extraction using ISIN lookup
- **Smart ticker matching**: Prefers LSE exchange (.L suffix) and ETF/Equity types
- **Multi-currency support**: Handles GBP, GBp (LSE pence), USD, EUR with automatic conversion to GBP
- **FX rate handling**: Real-time FX rates for USD and EUR with intelligent caching
- **Parallel processing**: Concurrent ticker lookups and price fetching (5-thread worker pools)
- **Comprehensive caching**: Historical prices and FX rates cached in SQLite to minimize API calls

### Portfolio Calculations
- **Monthly net contributions**: Histogram showing net cash flow (buys positive, sells negative)
- **Daily portfolio values**: Line graph of actual portfolio value including asset price changes
- **Holdings simulation**: Cumulative holdings calculation with forward-fill for non-trade days
- **Real-time valuations**: Uses historical closing prices from yfinance converted to GBP
- **Precomputation**: Background processing of yfinance data for instant chart generation

### API Endpoints
- `GET /` - Upload form page
- `POST /reset/` - Reset database before new uploads
- `POST /upload/` - Process and merge CSV files (with ticker extraction and background processing)
- `GET /portfolio-values/` - Retrieve precomputed portfolio data (daily values + monthly contributions)
- `GET /export/trades/` - Export entire trades table as JSON
- `GET /export/prices/` - Export ticker prices with original and converted values

### Performance & Architecture
- **Background processing**: Asynchronous yfinance data collection triggered on upload
- **Intelligent caching**: Prices cached by (ticker, date) with OR REPLACE semantics
- **Precomputed tables**: Fast retrieval of portfolio values, monthly contributions, and ticker prices
- **Concurrent operations**: ThreadPoolExecutor for parallel API calls and data processing
- **State management**: Precomputation status tracking with completion monitoring

### Security
- **HTTP Basic Authentication**: All API endpoints require authentication (username/password)
- **Environment-based credentials**: Configure credentials via `AUTH_USERNAME` and `AUTH_PASSWORD` environment variables
- **HTTPS support**: Caddy reverse proxy provides automatic TLS certificates via Let's Encrypt
- **Security headers**: Includes HSTS, X-Content-Type-Options, X-Frame-Options, and X-XSS-Protection

## Project Structure

```
.
├── .gitignore                     # Git ignore rules (excludes venv, __pycache__, *.db, trading_statements/)
├── pyproject.toml                 # Dependencies managed by UV
├── uv.lock                        # UV lockfile (auto-generated)
├── README.md                      # This file
├── .vscode/
│   └── settings.json             # VSCode settings with Ruff linting/formatting
├── src/
│   ├── __init__.py               # Package init
│   ├── main.py                   # FastAPI application with all API endpoints
│   ├── database.py               # Database operations (CRUD, connections, caching)
│   ├── merge_csv.py              # CSV parsing, cleaning, and merging logic
│   ├── tickers.py                # Ticker extraction via Yahoo Finance API
│   ├── prices.py                 # Price fetching, currency detection, and FX conversion
│   ├── portfolio.py              # Portfolio value calculations and holdings simulation
│   ├── background_processor.py   # Asynchronous precomputation of yfinance data
│   ├── temp-tests/               # Testing and development utilities
│   │   ├── extract_tickers.py    # Standalone ticker extraction for existing DB
│   │   ├── test_merge.py         # Test merging without web interface
│   │   ├── get_historical_data.py # Fetch and inspect historical yfinance data
│   │   └── test_price_factors.py # Compare CSV vs yfinance prices for debugging
│   └── templates/
│       └── upload.html           # Upload form with HTMX and Plotly integration
├── tests/                        # Comprehensive test suite (see tests/README.md)
│   ├── conftest.py               # Pytest fixtures and configuration
│   ├── test_database.py          # Database operation unit tests
│   ├── test_merge_csv.py         # CSV merging unit tests
│   ├── test_tickers.py           # Ticker extraction unit tests
│   ├── test_prices.py            # Price fetching and currency conversion tests
│   ├── test_portfolio.py         # Portfolio calculation unit tests
│   ├── test_background_processor.py # Background processing tests
│   └── test_integration.py       # FastAPI endpoint integration tests
├── htmlcov/                      # Coverage reports (auto-generated by pytest)
├── db/                           # Database directory (auto-created on first run)
│   └── merged_trading.db         # SQLite with trades, prices, and precomputed tables
└── trading_statements/           # Sample CSV files (not committed, add your own)
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

3. **Configure Authentication** (Optional but recommended):
   Set environment variables for authentication credentials:
   ```bash
   export AUTH_USERNAME="your_username"
   export AUTH_PASSWORD="your_strong_password"
   ```
   Defaults are: username=`admin`, password=`password` (change these for production!)

4. **Run the Server**:
   ```
   uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   ```
   - `--reload` enables auto-reload for development.
   - Access the app at `http://localhost:8000`.
   - You'll be prompted for credentials when accessing any endpoint.

## Usage

### Initial Setup
1. Open `http://localhost:8000` in your browser to see the upload form.
2. **IMPORTANT**: Click the reset button or call `/reset/` endpoint to clear any existing data.
3. Select multiple CSV files (e.g., `GIA_Trading_statement_*.csv`, `ISA_Trading_statement_*.csv`) using the "Browse Files" button.

### Upload Process
The form submits automatically on file selection. The server will:
- Parse each file (skip first row title, use second as header, add `Account_Type` based on filename like 'GIA' or 'ISA')
- Merge into `db/merged_trading.db` (overwriting any existing data)
- Extract tickers for unique securities (queries Yahoo Finance, may take a few seconds)
- Start **background processing** of yfinance data for all tickers
- Display success message with transaction count, date range, and "Generate Charts" button

### Chart Generation
4. Click "Generate Charts" to view interactive graphs:
   - **Monthly Net Contributions Histogram**: Net cash flow per month (buys positive, sells negative)
   - **Portfolio Value Line Graph**: Daily portfolio value over time with price appreciation

### Data Export
Access these endpoints for data export:
- `/export/trades/` - Full trades table as JSON
- `/export/prices/` - Ticker prices with original and GBP-converted values

### Upload Another Batch
Click "Upload another batch" to reset the form and repeat the process with new files.

### Authentication
All API endpoints require HTTP Basic Authentication. You'll be prompted for credentials when accessing:
- The upload form at `/`
- Any API endpoint (`/reset/`, `/upload/`, `/portfolio-values/`, `/export/trades/`, `/export/prices/`)

**Configure credentials**:
```bash
export AUTH_USERNAME="your_username"
export AUTH_PASSWORD="your_secure_password"
```

Default credentials are `admin`/`password` - **change these for production use!**

## Database Schema

The SQLite database contains the following tables:

### Core Tables
**trades** - Merged transaction data from CSV uploads
- `Security / ISIN`, `Transaction Type`, `Quantity`, `Share Price`, `Total Trade Value`
- `Trade Date/Time`, `Settlement Date`, `Broker`, `Account_Type`, `Ticker`

**prices** - Cached historical prices and FX rates
- `ticker` (TEXT), `date` (DATE), `close` (REAL)
- Primary key: `(ticker, date)`
- Automatically populated during price fetching and FX rate queries

### Precomputed Tables (for performance)
**precomputed_portfolio_values** - Daily portfolio values
- `date` (DATE PRIMARY KEY), `daily_value` (REAL), `last_updated` (TIMESTAMP)

**precomputed_monthly_contributions** - Monthly net contributions
- `month` (TEXT PRIMARY KEY), `net_value` (REAL), `last_updated` (TIMESTAMP)

**precomputed_ticker_prices** - Ticker prices (original and converted)
- `ticker` (TEXT), `date` (DATE), `original_currency` (TEXT)
- `original_price` (REAL), `converted_price_gbp` (REAL), `last_updated` (TIMESTAMP)
- Primary key: `(ticker, date)`

**precompute_status** - Background processing metadata
- `id`, `status`, `started_at`, `completed_at`, `total_tickers`, `last_error`
- Tracks background yfinance data collection progress

## Test Utilities (src/temp-tests/)

The project includes several utility scripts in the `src/temp-tests/` directory for testing and development:

**test_merge.py**
```bash
uv run python src/temp-tests/test_merge.py
```
Test merging sample CSV files without web interface. Merges GIA and ISA files from `trading_statements/` and tests portfolio values endpoint.

**extract_tickers.py**
```bash
uv run python src/temp-tests/extract_tickers.py
```
Run ticker extraction standalone. Useful for reprocessing tickers without re-uploading CSV files.

**get_historical_data.py**
```bash
uv run python src/temp-tests/get_historical_data.py
```
Fetch and display historical price data for a specific ticker. Helpful for debugging price issues or checking data availability. Default ticker: CSH2.L

**test_price_factors.py**
```bash
uv run python src/temp-tests/test_price_factors.py
```
Advanced analysis tool that compares CSV prices to yfinance prices for the same dates to detect constant multiplication factors (useful for identifying unit discrepancies like pence vs pounds). Also tests FX conversion logic.

## Background Processing

The system includes a sophisticated background processing system that precomputes yfinance data:

### What Gets Precomputed
1. **Ticker Prices**: Fetches historical prices for all valid tickers
2. **Currency Detection**: Identifies reported currency for each ticker
3. **FX Rate Collection**: Gathers historical FX rates for non-GBP securities
4. **Portfolio Values**: Calculates daily portfolio values based on holdings × prices
5. **Monthly Contributions**: Computes net monthly cash flow

### How It Works
- Triggered automatically on successful file upload via `BackgroundTasks`
- Processes all unique tickers in parallel using ThreadPoolExecutor
- Stores both original prices (in reported currency) and GBP-converted prices
- Caches all fetched data to minimize future API calls
- Status tracked in `precompute_status` table

### Performance Benefits
- **Instant charts**: Precomputed data eliminates wait time when clicking "Generate Charts"
- **Reduced API calls**: Caching prevents redundant yfinance queries
- **Fresh data check**: System validates precomputed data is recent (<24 hours)
- **Fallback available**: If no precomputed data, falls back to live calculation

## Graphs Explanation

### Monthly Net Contributions Histogram
Bars show net cash flow per month (positive for buys/investments, negative for sells/withdrawals). Combines GIA and ISA data for a holistic view of contributions/distributions across all accounts.

### Actual Portfolio Value Line Graph
Daily portfolio value over time calculated using:
1. **Holdings Simulation**: Cumulative holdings from trades (buys increase shares, sells decrease them)
2. **Forward-Fill**: Holdings carried forward for non-trade days
3. **Price Conversion**: Historical closing prices from yfinance converted to GBP
4. **FX Rate Application**: Historic FX rates cached to avoid refetching
5. **Daily Valuation**: Σ(holdings × price) across all tickers and account types
6. **Invalid Handling**: Invalid or "Not found" tickers contribute 0 value

Charts are interactive (zoom, hover) via Plotly and rendered client-side from server-computed data.

## Ticker Extraction

### Extraction Process
For each unique "Security / ISIN", the system:
1. Parses security name and ISIN using regex pattern: `(# InvestEngine CSV Server

A Python FastAPI server for uploading and processing InvestEngine trading statement CSV files. It merges CSV batches into a SQLite database, enriches securities with ticker symbols via Yahoo Finance, fetches historical prices, and provides portfolio visualisations (monthly net contributions and daily portfolio value).

## Features
- Web interface for batch CSV upload (FastAPI + HTMX)
- Automatic CSV parsing, cleaning and merging
- ISIN → ticker mapping (Yahoo Finance API) with LSE and ETF handling
- Multi‑currency support with FX rate conversion to GBP
- Cached price and FX data in SQLite for fast subsequent queries
- Background pre‑computation of ticker prices and portfolio values
- API endpoints for data export and portfolio calculations
- Basic HTTP authentication (environment‑configurable)
- Docker Compose + Caddy reverse‑proxy for HTTPS deployment

## Project Structure
```
.
├── src/                 # Application source code
│   ├── main.py          # FastAPI app and routes
│   ├── merge_csv.py     # CSV merging logic
│   ├── security_parser.py
│   ├── tickers.py       # Ticker extraction
│   ├── prices.py        # Price fetching & FX conversion
│   ├── portfolio.py     # Portfolio calculations
│   ├── background_processor.py
│   ├── database.py      # SQLite helper functions
│   └── templates/upload.html
├── tests/               # pytest test suite
├── docker-compose.yml
├── Caddyfile
├── pyproject.toml
└── README.md           # This file
```

## Setup
1. Install **UV** (or use pip/conda):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # or: brew install uv
   ```
2. Install dependencies:
   ```bash
   uv sync
   ```
3. (Optional) Set authentication credentials:
   ```bash
   export AUTH_USERNAME=admin
   export AUTH_PASSWORD=password
   ```

## Running the Server
```bash
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` in a browser, reset the database via `/reset/`, then upload CSV files.

## API Endpoints
- `GET /` – upload form (HTML)
- `POST /reset/` – clear database
- `POST /upload/` – upload CSV batch (must call `/reset/` first)
- `GET /portfolio-values/` – monthly net contributions & daily portfolio values (pre‑computed when possible)
- `GET /export/trades/` – JSON export of all trades
- `GET /export/prices/` – JSON export of ticker prices (original & GBP)
- `POST /mapping/` – create/update ISIN → ticker mappings (JSON body)
- `GET /mapping/` – list all mappings

## Testing
```bash
# Install test extras
uv sync --extra test
# Run all tests with coverage
pytest --cov=src
```
The `tests/` directory contains unit, integration and background‑processor tests. Use markers to run subsets, e.g. `pytest -m unit`.

## Docker Compose (Production)
```bash
docker-compose up -d
```
Configure `Caddyfile` with your server IP and set strong `AUTH_USERNAME`/`AUTH_PASSWORD` environment variables.

## Security Notes
- Change default credentials before production use.
- HTTPS is provided via Caddy (self‑signed for IP addresses).
- Security headers (HSTS, X‑Content‑Type‑Options, etc.) are enabled.

---
Generated with Claude Code.
?)/\s*ISIN\s+([A-Z]{2}[A-Z0-9]{9}[0-9])`
2. Searches Yahoo Finance API by security name (primary method)
3. Prefers LSE exchange (.L suffix) and ETF/Equity quote types
4. Falls back to ISIN-based search if name search yields no results
5. Stores results in `Ticker` column (e.g., "VUSA.L" or "Not found")

### Multi-Currency Handling
- **Currency Detection**: Queries yfinance for each ticker's reported currency
- **GBp Conversion**: Automatically converts LSE pence prices (GBp) to GBP by dividing by 100
- **FX Rate Conversion**: Fetches historic FX rates (GBPUSD=X, EURGBP=X) for non-GBP securities
- **Caching Strategy**: All prices and FX rates cached in SQLite `prices` table by (ticker, date)

### API Rate Limiting
Yahoo Finance API may rate-limit requests. Extraction happens synchronously during upload, but subsequent price fetching runs in background to improve user experience.

## Development Notes

### Architecture & Performance

**Parallel Processing**: The system employs concurrent operations for optimal performance:
- Ticker lookups and currency detection run in parallel using `ThreadPoolExecutor` (5 workers default)
- Price fetching and currency conversion are parallelized for all unique securities
- FX rate fetching happens asynchronously for all needed currencies

**Caching System**: Optimized to reduce external API calls:
- Historical prices cached in SQLite `prices` table by (ticker, date) with OR REPLACE semantics
- FX rates (GBPUSD=X, EURGBP=X) cached using the same mechanism
- Cache hit query: Checks database before calling yfinance API
- Automatic forward-fill of cached data to handle weekends/holidays

**Database Schema**:
- `trades` table: Merged CSV data with columns including `Account_Type` and `Ticker`
- `prices` table: Historical prices and FX rates with primary key on (ticker, date)

### Development Workflow

- **Live Reload**: Edit code in `src/` and restart/reload as needed (FastAPI's `--reload` flag recommended for development)
- **Linting/Formatting**: Ruff is configured for linting/formatting in `.vscode/settings.json` (runs on save)
- **Database**: SQLite database is overwritten on each upload (no versioning)
- **Price Fetching**: yfinance fetches historical 'Close' prices; handles missing data with forward-fill
- **Error Handling**: Invalid CSVs or API errors return JSON error messages displayed in the browser
- **Logging**: Comprehensive logging to console for merging, computations, ticker searches, and price conversions

**Test Scripts**: Utility scripts in `src/temp-tests/` for debugging and development. See the "Test Utilities" section for detailed instructions:
- `test_merge.py`: Test merging without the web interface
- `extract_tickers.py`: Run ticker extraction standalone on existing DB
- `get_historical_data.py`: Fetch and inspect historical yfinance data
- `test_price_factors.py`: Analyze price ratios between CSV and yfinance data

### Production Deployment

#### Using Docker Compose (Recommended)

1. Edit `Caddyfile` and replace `192.168.1.100` with your server's actual IP address

2. Set secure authentication credentials:
   ```bash
   export AUTH_USERNAME="secure_user"
   export AUTH_PASSWORD="very_secure_password_123!"
   ```

3. Run with Docker Compose:
   ```bash
   docker-compose up -d
   ```

4. Access at `https://your-ip-address`
   - ⚠️ Browser will show "Not Secure" warning (click "Advanced" → "Proceed")
   - Your traffic is still encrypted, but certificates are self-signed

Caddy features:
- HTTPS with self-signed certificates for IP addresses
- Automatic HTTP to HTTPS redirect
- Security headers (HSTS, etc.)
- Request logging

**Note**: Since plain IP addresses can't use Let's Encrypt certificates, browsers show warnings even though traffic is encrypted. This is normal and acceptable for private use.

#### Manual Production Server
1. Set authentication credentials via environment variables
2. Use Gunicorn with Uvicorn workers:
   ```bash
   pip install gunicorn
   gunicorn src.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```
3. Configure a reverse proxy (Nginx, Caddy, etc.) for HTTPS termination
4. Set up SSL certificates (Let's Encrypt recommended)

#### Production Security Checklist
- ✅ Change default credentials (`admin`/`password`)
- ✅ Use HTTPS in production (via Caddy, Nginx, or similar)
- ✅ Set strong passwords
- ✅ Configure firewall to restrict access (limit to specific IPs if needed)
- ✅ Enable rate limiting on upload endpoint
- ✅ Monitor logs for suspicious activity
- ✅ Keep dependencies updated
- ✅ Use environment variables for all secrets (never hardcode credentials)
- ✅ Implement database backups
- ✅ Enable logging and monitoring

### Development vs Production

**Development**:
```bash
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```
- HTTP (no TLS)
- Basic auth enabled
- Auto-reload on code changes

**Production**:
```bash
docker-compose up -d
```
- HTTPS with automatic TLS
- Secure authentication
- Production-ready logging
- Containerized deployment

## Troubleshooting

**Dependency Issues**
- If dependencies fail to install: Run `uv sync --dev` or check `pyproject.toml` for version conflicts
- Ensure Python 3.8+ is installed and UV is available in PATH

**Server Problems**
- Server not starting: Check for port 8000 conflicts, verify UV installation
- Use `uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000` for detailed error output

**CSV Upload Issues**
- CSV issues: Ensure files match expected format (see CSV Format Assumptions)
- Invalid columns or malformed dates will raise errors displayed in the browser
- Check browser console for JavaScript errors related to file handling

**Ticker & Price Issues**
- Ticker extraction fails: Check internet connection; Yahoo Finance API may rate-limit
- Run `uv run python src/tests/extract_tickers.py` manually to debug ticker extraction
- No graphs appearing: Ensure ticker extraction completed (happens synchronously during upload)
- Verify yfinance can fetch data (check internet connectivity, ticker symbols are valid)

**Database & Caching**
- Database not created: Upload files first; `db/` directory auto-creates
- Old data persisting: Remember database overwrites on each upload (this is by design)
- Price caching: Check `prices` table in SQLite DB to verify caching is working

**Chart Generation**
- Charts not displaying: Check browser console for JavaScript errors; verify Plotly CDN loaded
- Empty chart data: Verify ticker extraction found valid tickers; check console logs for errors
- Mismatched data lengths: Check server logs; may indicate data validation issues

**Visual Studio Code**
- Linter errors: Ruff/Basedpyright may flag type issues in dynamic code; these don't affect runtime
- Ensure Ruff extension is installed and enabled
- Format on save is configured in `.vscode/settings.json`

**Performance**
- Slow ticker extraction: Normal for many securities due to Yahoo Finance API rate limiting
- Slow chart generation: First run fetches all historical data; subsequent runs use cache
- Consider running ticker extraction during off-hours for large portfolios