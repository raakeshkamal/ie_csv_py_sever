#!/usr/bin/env python3
"""
Manual test guide for background yfinance data collection
This script checks if the server is running and provides curl commands to test
"""

import sys

print("="*80)
print("BACKGROUND PROCESSOR - MANUAL TEST GUIDE")
print("="*80)
print()

# Check if server is running
try:
    import requests
    response = requests.get('http://localhost:8000/', timeout=2)
    if response.status_code == 200:
        print("✓ Server is running at http://localhost:8000")
    else:
        print("✗ Server responded but with error status")
        print(f"  Status code: {response.status_code}")
        sys.exit(1)
except ImportError:
    print("✗ requests module not available")
    print("  Please install: pip install requests")
    sys.exit(1)
except Exception as e:
    print(f"✗ Server not running or not accessible: {e}")
    print()
    print("  Please start the server first:")
    print("  $ uv run uvicorn src.main:app --reload")
    print()
    sys.exit(1)

print("\n" + "="*80)
print("TEST COMMANDS")
print("="*80)

print("""
Step 1: Reset the database
---------------------------
curl -X POST http://localhost:8000/reset/

Step 2: Upload CSV files
------------------------
Using web interface:
  - Open http://localhost:8000 in browser
  - Select CSV files from trading_statements/
  - Upload

Or using curl:
curl -X POST -F "files=@trading_statements/GIA_Trading_statement_2024.csv" \
     -F "files=@trading_statements/ISA_Trading_statement_2024.csv" \
     http://localhost:8000/upload/

Step 3: Check background processing status
-------------------------------------------
Wait 10-30 seconds for background processing to complete,
then test the portfolio-values endpoint:

curl http://localhost:8000/portfolio-values/

This should return data much faster than before (milliseconds instead of seconds)

Step 4: Export precomputed data
--------------------------------
curl http://localhost:8000/export/prices/

This returns all precomputed yfinance data including:
- Daily portfolio values with timestamps
- Monthly net contributions
- Background job status

Step 5: Verify speed improvement
---------------------------------
Time the portfolio-values endpoint:
time curl -s http://localhost:8000/portfolio-values/ > /dev/null

Should be very fast (< 1 second) after background processing completes.

""")

print("="*80)
print("EXPECTATIONS")
print("="*80)
print("""
• After upload, background processing starts automatically
• /portfolio-values/ returns precomputed data (fast!)
• If precomputed data not ready, falls back to live calculation
• Export endpoint shows all cached yfinance data
• Response format unchanged - fully backward compatible
""")

print("="*80)
print("TROUBLESHOOTING")
print("="*80)
print("""
If background processing fails:
  • Check server logs for errors
  • Verify tickers were extracted: curl http://localhost:8000/export/trades/
  • Try direct upload via web interface
  • Check that yfinance API is accessible

If precomputed data is not available:
  • System automatically falls back to live calculation
  • Wait a bit longer and try again
  • Check database 'precompute_status' table for errors
""")

print("="*80)
print("Test completed - use the commands above to verify functionality")
print("="*80)
