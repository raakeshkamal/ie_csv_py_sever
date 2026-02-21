import io
import re
from datetime import datetime
from typing import List, Tuple

import pandas as pd


def detect_file_type(filename: str) -> str:
    """
    Detect if file is 'trading' or 'cash' statement based on filename.
    Returns "trading" or "cash".
    """
    filename_upper = filename.upper()
    if "_CASH_" in filename_upper:
        return "cash"
    elif "_TRADING_" in filename_upper:
        return "trading"
    else:
        return "trading"


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

    # Try full datetime format first
    result = pd.to_datetime(date_str, format="%d/%m/%y %H:%M:%S", errors="coerce")
    if not pd.isna(result):
        return result

    # Try date only format
    result = pd.to_datetime(date_str, format="%d/%m/%y", errors="coerce")
    if not pd.isna(result):
        return result

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


def parse_cash_date(date_str: str) -> datetime:
    """
    Parse date string from cash statement format (DD/MM/YY).
    """
    if pd.isna(date_str):
        return pd.NaT

    date_str = str(date_str).strip()
    result = pd.to_datetime(date_str, format="%d/%m/%y", errors="coerce")
    return result


def extract_cash_flows_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter cash DataFrame to only external cash flow activities.
    Includes: Payment Received, Withdrawal, ISA Transfer In
    Excludes: Opening Balance, Purchase, Sale, Transfer, Dividend
    """
    cash_flow_activities = [
        "Payment Received",
        "Withdrawal",
        "ISA Transfer In",
    ]

    return df[df["Activity"].isin(cash_flow_activities)].copy()


def merge_cash_files(file_data: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Merge cash statement CSVs into a single DataFrame.

    Handles:
    - Multi-portfolio files (skip "Portfolio: Cash", only use "Portfolio: None")
    - Multiple header rows within one file
    - Adds Account_Type column (GIA/ISA)
    - Filters to external cash flows only

    Returns: DataFrame with Date, Activity, Credit, Debit, Balance, Account_Type, Net_Flow
    """
    dfs = []

    for filename, content_str in file_data:
        content_str = content_str.lstrip("\ufeff")
        lines = content_str.split("\n")

        current_df_lines = []
        current_account_type = extract_account_type(filename)
        headers = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Cash Statement:"):
                portfolio_match = re.search(r"Portfolio:\s*(\w+)", line)
                if portfolio_match:
                    portfolio_type = portfolio_match.group(1)
                    if portfolio_type == "Cash":
                        current_df_lines = []
                        headers = None
                        continue

                if current_df_lines and headers:
                    try:
                        csv_content = "\n".join([",".join(headers)] + current_df_lines)
                        temp_df = pd.read_csv(io.StringIO(csv_content))
                        temp_df["Account_Type"] = current_account_type
                        dfs.append(temp_df)
                    except Exception:
                        pass

                current_df_lines = []
                headers = None
                continue

            if headers is None:
                if line.startswith("Date,Activity"):
                    headers = ["Date", "Activity", "Credit", "Debit", "Balance"]
                    continue
            else:
                current_df_lines.append(line)

        if current_df_lines and headers:
            try:
                csv_content = "\n".join([",".join(headers)] + current_df_lines)
                temp_df = pd.read_csv(io.StringIO(csv_content))
                temp_df["Account_Type"] = current_account_type
                dfs.append(temp_df)
            except Exception:
                pass

    if not dfs:
        raise ValueError("No valid cash CSV data to merge")

    merged_df = pd.concat(dfs, ignore_index=True)

    merged_df["Credit"] = pd.to_numeric(merged_df["Credit"].fillna(0), errors="coerce")
    merged_df["Debit"] = pd.to_numeric(merged_df["Debit"].fillna(0), errors="coerce")
    merged_df["Balance"] = pd.to_numeric(
        merged_df["Balance"].fillna(0), errors="coerce"
    )

    merged_df["Date"] = merged_df["Date"].apply(parse_cash_date)
    merged_df = merged_df.dropna(subset=["Date"])

    merged_df["Net_Flow"] = merged_df["Credit"] - merged_df["Debit"]

    merged_df = extract_cash_flows_only(merged_df)

    merged_df = merged_df.sort_values("Date").reset_index(drop=True)

    cols = [
        "Date",
        "Activity",
        "Credit",
        "Debit",
        "Balance",
        "Account_Type",
        "Net_Flow",
    ]
    merged_df = merged_df[cols]

    return merged_df
