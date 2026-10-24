"""
etl/clean.py  –  Zomato Bangalore Dataset Cleaning Script
=========================================================
Reads the raw Kaggle Zomato CSV, applies cleaning & feature-engineering,
and saves a contract-compliant CSV to data/processed/zomato_clean.csv.

Run from the project root:
    python etl/clean.py
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0.  Paths (relative to project root, i.e. CWD when script is executed)
# ---------------------------------------------------------------------------
RAW_PATH       = os.path.join("data", "raw", "zomato.csv")
PROCESSED_PATH = os.path.join("data", "processed", "zomato_clean.csv")

# The exact column order we promise downstream scripts will see
CONTRACT_COLUMNS = [
    "restaurant_id", "name", "city", "location", "rest_type",
    "category", "cuisines", "primary_cuisine", "cuisine_count",
    "online_order", "book_table", "rate", "votes", "log_votes", "cost",
]


def main():
    # -------------------------------------------------------------------
    # 1.  Read raw CSV  (latin-1 handles special chars in this dataset)
    # -------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1 — Reading raw CSV …")
    df = pd.read_csv(RAW_PATH, encoding="latin-1")
    print(f"  Rows after read: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    # -------------------------------------------------------------------
    # 2.  Keep only needed columns & rename to contract names
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 — Selecting & renaming columns …")

    keep_rename = {
        "name":                        "name",
        "online_order":                "online_order",
        "book_table":                  "book_table",
        "rate":                        "rate",
        "votes":                       "votes",
        "location":                    "location",
        "cuisines":                    "cuisines",
        "rest_type":                   "rest_type",
        "approx_cost(for two people)": "cost",
        "listed_in(type)":             "category",
        "listed_in(city)":             "city",
        "address":                     "address",       # kept temporarily for dedup
    }

    df = df[list(keep_rename.keys())].rename(columns=keep_rename)
    print(f"  Rows after column selection: {len(df):,}")

    # -------------------------------------------------------------------
    # 3.  Clean `rate` — strip "/5", drop NEW / - / missing
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 — Cleaning 'rate' column …")

    # Convert to string first so we can do string ops safely
    df["rate"] = df["rate"].astype(str).str.strip()

    # Drop rows where rate is "NEW", "-", or was NaN (shows as "nan")
    invalid_rates = df["rate"].isin(["NEW", "-", "nan", ""])
    print(f"  Dropping {invalid_rates.sum():,} rows with invalid rate")
    df = df[~invalid_rates].copy()

    # Strip the "/5" suffix and convert to float
    df["rate"] = df["rate"].str.replace("/5", "", regex=False).astype(float)
    print(f"  Rows after rate cleaning: {len(df):,}")

    # -------------------------------------------------------------------
    # 4.  Clean `cost` — remove commas, convert to float, drop missing
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 — Cleaning 'cost' column …")

    df["cost"] = df["cost"].astype(str).str.replace(",", "", regex=False)
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")

    missing_cost = df["cost"].isna().sum()
    print(f"  Dropping {missing_cost:,} rows with missing cost")
    df = df.dropna(subset=["cost"]).copy()
    print(f"  Rows after cost cleaning: {len(df):,}")

    # -------------------------------------------------------------------
    # 5.  Map online_order / book_table  Yes→1  No→0
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5 — Encoding online_order & book_table …")

    yes_no_map = {"Yes": 1, "No": 0}
    df["online_order"] = df["online_order"].map(yes_no_map)
    df["book_table"]   = df["book_table"].map(yes_no_map)
    print(f"  online_order unique: {sorted(df['online_order'].dropna().unique())}")
    print(f"  book_table   unique: {sorted(df['book_table'].dropna().unique())}")
    print(f"  Rows: {len(df):,}")

    # -------------------------------------------------------------------
    # 6.  Deduplicate on (name, address), then drop address
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6 — Deduplicating on (name, address) …")

    before = len(df)
    df = df.drop_duplicates(subset=["name", "address"], keep="first").copy()
    print(f"  Removed {before - len(df):,} duplicate rows")
    print(f"  Rows after dedup: {len(df):,}")

    # Address is no longer needed
    df = df.drop(columns=["address"])

    # Handle missing text values gracefully
    df["cuisines"] = df["cuisines"].fillna("Unknown").astype(str).str.strip()
    df["rest_type"] = df["rest_type"].fillna("Unknown").astype(str).str.strip()
    df.loc[df["cuisines"] == "", "cuisines"] = "Unknown"
    df.loc[df["rest_type"] == "", "rest_type"] = "Unknown"

    # -------------------------------------------------------------------
    # 7.  Create restaurant_id  (1 … N)
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7 — Creating restaurant_id …")

    df = df.reset_index(drop=True)
    df["restaurant_id"] = df.index + 1
    print(f"  restaurant_id range: 1 … {df['restaurant_id'].max()}")

    # -------------------------------------------------------------------
    # 8.  primary_cuisine = first cuisine in the comma-separated list
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 8 — Extracting primary_cuisine …")

    df["primary_cuisine"] = (
        df["cuisines"]
        .str.split(",")
        .str[0]
        .str.strip()
    )
    df.loc[df["primary_cuisine"] == "", "primary_cuisine"] = "Unknown"
    print(f"  Top 5 primary cuisines:\n{df['primary_cuisine'].value_counts().head()}")

    # -------------------------------------------------------------------
    # 9.  cuisine_count = number of cuisines
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 9 — Computing cuisine_count …")

    df["cuisine_count"] = (
        df["cuisines"]
        .str.split(",")
        .apply(len)
    )
    print(f"  cuisine_count stats:\n{df['cuisine_count'].describe()}")

    # -------------------------------------------------------------------
    # 10. log_votes = log(1 + votes)
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 10 — Computing log_votes …")

    df["votes"]     = df["votes"].astype(int)
    df["log_votes"] = np.log1p(df["votes"])
    print(f"  log_votes stats:\n{df['log_votes'].describe()}")

    # -------------------------------------------------------------------
    # 11. Strip whitespace from all string columns
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 11 — Stripping whitespace from string columns …")

    str_cols = df.select_dtypes(include="object").columns.tolist()
    for col in str_cols:
        df[col] = df[col].str.strip()
    print(f"  Stripped columns: {str_cols}")

    # -------------------------------------------------------------------
    # 12. Reorder columns to contract & save
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 12 — Saving cleaned CSV …")

    df = df[CONTRACT_COLUMNS]

    # Make sure output directory exists
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"  Saved → {PROCESSED_PATH}")
    print(f"  Final shape: {df.shape}")

    # -------------------------------------------------------------------
    # 13. Summary statistics
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 13 — Summary statistics")
    print("\nNull counts:")
    print(df.isnull().sum())
    print("\nDescribe (numeric):")
    print(df.describe())

    # -------------------------------------------------------------------
    # 14. Assertions — contract checks
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 14 — Running contract assertions …")

    # 14-a  Columns match the contract exactly
    assert list(df.columns) == CONTRACT_COLUMNS, \
        f"Column mismatch!\n  Expected: {CONTRACT_COLUMNS}\n  Got:      {list(df.columns)}"
    print("  ✓ Columns match contract")

    # 14-b  restaurant_id is unique
    assert df["restaurant_id"].is_unique, "restaurant_id has duplicates!"
    print("  ✓ restaurant_id is unique")

    # 14-c  rate within [0, 5]
    assert df["rate"].between(0, 5).all(), \
        f"rate out of range! min={df['rate'].min()}, max={df['rate'].max()}"
    print("  ✓ rate is within 0–5")

    # 14-d  online_order and book_table are only 0 or 1
    assert set(df["online_order"].unique()).issubset({0, 1}), \
        f"online_order has unexpected values: {df['online_order'].unique()}"
    assert set(df["book_table"].unique()).issubset({0, 1}), \
        f"book_table has unexpected values: {df['book_table'].unique()}"
    print("  ✓ online_order & book_table ∈ {0, 1}")

    # 14-e  No nulls in critical columns
    critical = ["restaurant_id", "name", "rate", "votes", "cost",
                "online_order", "book_table"]
    for col in critical:
        assert df[col].notna().all(), f"Null found in critical column '{col}'!"
    print("  ✓ No nulls in critical columns")

    print("\n" + "=" * 60)
    print("✅  ETL cleaning complete — all assertions passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
