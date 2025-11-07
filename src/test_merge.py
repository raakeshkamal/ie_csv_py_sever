import sys

sys.path.append("src")

from merge_csv import merge_csv_files
import pandas as pd
import sqlite3


# Read the existing CSV files
def read_csv_content(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


gia_content = read_csv_content("trading_statements/GIA_Trading_statement_6_Jun_2022_to_7_Nov_2025.csv")
isa_content = read_csv_content("trading_statements/ISA_Trading_statement_6_Jun_2022_to_7_Nov_2025.csv")

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
except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
