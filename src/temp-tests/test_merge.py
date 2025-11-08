import sys

sys.path.append("src")

import asyncio
import json
import sqlite3
import time

from extract_tickers import main as extract_tickers_main
from main import get_portfolio_values  # Import the function directly for testing
from merge_csv import merge_csv_files


# Read the existing CSV files
def read_csv_content(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


gia_content = read_csv_content(
    "trading_statements/GIA_Trading_statement_6_Jun_2022_to_7_Nov_2025.csv"
)
isa_content = read_csv_content(
    "trading_statements/ISA_Trading_statement_6_Jun_2022_to_7_Nov_2025.csv"
)

file_data = [("GIA.csv", gia_content), ("ISA.csv", isa_content)]

try:
    merged_df = merge_csv_files(file_data)
    print(f"Merged successfully! Shape: {merged_df.shape}")
    print(merged_df.head())
    print(f"Account types: {merged_df['Account_Type'].value_counts()}")
    print(
        f"Date range: {merged_df['Trade Date/Time'].min()} to {merged_df['Trade Date/Time'].max()}"
    )
    db_path = "db/merged_trading.db"
    conn = sqlite3.connect(db_path)
    merged_df.to_sql("trades", conn, if_exists="replace", index=False)
    conn.close()
    print("Saved to merged_trading.db")

    # Run ticker extraction
    extract_tickers_main()
    print("Tickers extracted.")

    # Wait a bit for background to complete (since it's sync now)
    time.sleep(2)

    # Test the new /portfolio-values/ endpoint by calling the function directly
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(get_portfolio_values())
        loop.close()

        content = str(response.body, "utf-8")
        data = json.loads(content)

        if data["success"]:
            print("Portfolio values computed successfully!")
            print(f"Monthly net data length: {len(data['monthly_net'])}")
            print(f"Daily dates length: {len(data['daily_dates'])}")
            print(f"Daily values length: {len(data['daily_values'])}")
            print(f"Sample daily value: {data['daily_values'][:5]}")
            # Verify lengths match
            if len(data["daily_dates"]) == len(data["daily_values"]):
                print("Verification: Daily dates and values lengths match.")
            else:
                print("Warning: Length mismatch in daily data.")
        else:
            print(f"Error in portfolio values: {data['error']}")
    except Exception as test_err:
        print(f"Test error: {test_err}")
        import traceback

        traceback.print_exc()

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
