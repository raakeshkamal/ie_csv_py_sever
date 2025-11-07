import pandas as pd
import io
import re
from typing import List, Tuple
from datetime import datetime


def extract_account_type(filename: str) -> str:
    """
    Extract account type from filename, e.g., 'GIA' or 'ISA'.
    Assumes filename starts with 'GIA_' or 'ISA_'.
    """
    filename_upper = filename.upper()
    if filename_upper.startswith("GIA_"):
        return "GIA"
    elif filename_upper.startswith("ISA_"):
        return "ISA"
    else:
        # Fallback: try to find in name
        if "GIA" in filename_upper:
            return "GIA"
        elif "ISA" in filename_upper:
            return "ISA"
        else:
            return "Unknown"


def clean_currency(value: str) -> float:
    """
    Clean currency string: remove '£' and commas, convert to float.
    """
    if pd.isna(value):
        return 0.0
    value_str = str(value).replace("£", "").replace(",", "")
    try:
        return float(value_str)
    except ValueError:
        return 0.0


def parse_date(date_str: str) -> datetime:
    """
    Parse date string to datetime. Handles formats like '20/09/23 13:48:48' or '22/09/23'.
    """
    if pd.isna(date_str):
        return pd.NaT
    try:
        # Try full datetime format first
        return pd.to_datetime(date_str, format="%d/%m/%y %H:%M:%S", errors="coerce")
    except:
        try:
            # Try date only
            return pd.to_datetime(date_str, format="%d/%m/%y", errors="coerce")
        except:
            return pd.NaT


def merge_csv_files(file_data: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Merge list of (filename, csv_content_str) into a single DataFrame.

    Steps:
    - For each file: Load CSV skipping first 2 rows, add Account_Type.
    - Clean columns: Remove £ from Share Price, Total Trade Value; parse dates.
    - Concatenate all DFs.
    - Validate: Ensure consistent columns.
    - Sort by Trade Date/Time.

    Returns: Merged DataFrame.
    """
    dfs = []
    expected_columns = [
        "Security / ISIN",
        "Transaction Type",
        "Quantity",
        "Share Price",
        "Total Trade Value",
        "Trade Date/Time",
        "Settlement Date",
        "Broker",
    ]

    for filename, content_str in file_data:
        # Strip BOM if present
        content_str = content_str.lstrip("\ufeff")
        # Load CSV from string, skip first row (title), use second as header
        df = pd.read_csv(io.StringIO(content_str), skiprows=1)

        # Validate columns
        if not all(col in df.columns for col in expected_columns):
            raise ValueError(f"Invalid columns in {filename}: {list(df.columns)}")

        # Rename columns if needed (trim spaces)
        df.columns = df.columns.str.strip()

        # Add Account_Type
        account_type = extract_account_type(filename)
        df["Account_Type"] = account_type

        # Clean Quantity to float
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

        # Clean currency columns
        df["Share Price"] = df["Share Price"].apply(clean_currency)
        df["Total Trade Value"] = df["Total Trade Value"].apply(clean_currency)

        # Parse dates
        df["Trade Date/Time"] = df["Trade Date/Time"].apply(parse_date)
        df["Settlement Date"] = df["Settlement Date"].apply(parse_date)

        # Drop rows with invalid dates if needed
        df = df.dropna(subset=["Trade Date/Time"])

        dfs.append(df)

    if not dfs:
        raise ValueError("No valid CSV data to merge")

    # Concatenate
    merged_df = pd.concat(dfs, ignore_index=True)

    # Sort by Trade Date/Time
    merged_df = merged_df.sort_values("Trade Date/Time").reset_index(drop=True)

    # Reorder columns to put Account_Type after headers
    cols = expected_columns + ["Account_Type"]
    merged_df = merged_df[cols]

    return merged_df
