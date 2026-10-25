"""
warehouse/build_dw.py — Build the Zomato Data Warehouse (Star Schema)
=====================================================================

Reads  : data/processed/zomato_clean.csv
Writes : warehouse/zomato_dw.db  (SQLite)

Star-schema tables
──────────────────
fact_restaurant
    restaurant_id   INTEGER  PRIMARY KEY
    location_key    INTEGER  FK → dim_location
    cuisine_key     INTEGER  FK → dim_cuisine
    type_key        INTEGER  FK → dim_type
    rate            REAL
    votes           INTEGER
    cost            REAL
    online_order    INTEGER  (1 = Yes, 0 = No)
    book_table      INTEGER  (1 = Yes, 0 = No)
    cluster_id      INTEGER  NULLABLE  FK → dim_cluster

dim_location
    location_key    INTEGER  PRIMARY KEY AUTOINCREMENT
    city            TEXT
    location        TEXT

dim_cuisine
    cuisine_key     INTEGER  PRIMARY KEY AUTOINCREMENT
    primary_cuisine TEXT
    cuisine_count   INTEGER

dim_type
    type_key        INTEGER  PRIMARY KEY AUTOINCREMENT
    rest_type       TEXT
    category        TEXT

dim_cluster  (created empty — filled later by load_clusters.py)
    cluster_id      INTEGER  PRIMARY KEY
    label           TEXT
    description     TEXT
"""

import os
import sqlite3
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
CSV_PATH = os.path.join("data", "processed", "zomato_clean.csv")
DB_PATH  = os.path.join("warehouse", "zomato_dw.db")

# ===========================================================================
# 1. Read the clean CSV
# ===========================================================================
print("=" * 60)
print("BUILD DATA WAREHOUSE — Star Schema")
print("=" * 60)

print(f"\n[1/6] Reading clean CSV from {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH)
# Ensure text columns do not contain NaN
df["primary_cuisine"] = df["primary_cuisine"].fillna("Unknown")
df["rest_type"] = df["rest_type"].fillna("Unknown")
df["city"] = df["city"].fillna("Unknown")
df["location"] = df["location"].fillna("Unknown")
df["category"] = df["category"].fillna("Unknown")
print(f"      → {len(df):,} rows, {len(df.columns)} columns loaded.")
print(f"      Columns: {list(df.columns)}")

# ===========================================================================
# 2. Build dimension tables (unique combos)
# ===========================================================================
print("\n[2/6] Extracting dimension tables ...")

# --- dim_location: unique (city, location) combos ---
dim_location = (
    df[["city", "location"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
# Autoincrement keys start at 1
dim_location.index += 1
dim_location.index.name = "location_key"
dim_location = dim_location.reset_index()
print(f"      dim_location : {len(dim_location):,} unique city–location combos")

# --- dim_cuisine: unique (primary_cuisine, cuisine_count) combos ---
dim_cuisine = (
    df[["primary_cuisine", "cuisine_count"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
dim_cuisine.index += 1
dim_cuisine.index.name = "cuisine_key"
dim_cuisine = dim_cuisine.reset_index()
print(f"      dim_cuisine  : {len(dim_cuisine):,} unique cuisine combos")

# --- dim_type: unique (rest_type, category) combos ---
dim_type = (
    df[["rest_type", "category"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
dim_type.index += 1
dim_type.index.name = "type_key"
dim_type = dim_type.reset_index()
print(f"      dim_type     : {len(dim_type):,} unique type combos")

# ===========================================================================
# 3. Merge dimension keys back onto the fact table
# ===========================================================================
print("\n[3/6] Merging dimension keys into fact table ...")

fact = df.copy()

# Merge location_key
fact = fact.merge(dim_location, on=["city", "location"], how="left")

# Merge cuisine_key
fact = fact.merge(dim_cuisine, on=["primary_cuisine", "cuisine_count"], how="left")

# Merge type_key
fact = fact.merge(dim_type, on=["rest_type", "category"], how="left")

# Select only fact columns (keep the FK keys, drop the dimension text cols)
fact_cols = [
    "restaurant_id",
    "location_key",
    "cuisine_key",
    "type_key",
    "rate",
    "votes",
    "cost",
    "online_order",
    "book_table",
]
fact = fact[fact_cols].copy()

# cluster_id is NULL for now — will be filled by load_clusters.py
fact["cluster_id"] = None

print(f"      fact_restaurant : {len(fact):,} rows")

# ===========================================================================
# 4. Write to SQLite — drop & recreate tables each run
# ===========================================================================
print(f"\n[4/6] Writing star schema to {DB_PATH} ...")

# Ensure the warehouse directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()

# Enable foreign key enforcement
cur.execute("PRAGMA foreign_keys = ON;")

# Drop tables in dependency order (fact first, then dims)
for tbl in ["fact_restaurant", "dim_location", "dim_cuisine", "dim_type", "dim_cluster"]:
    cur.execute(f"DROP TABLE IF EXISTS {tbl};")
    print(f"      Dropped (if existed): {tbl}")

# --- Create dim_location ---
cur.execute("""
    CREATE TABLE dim_location (
        location_key  INTEGER PRIMARY KEY AUTOINCREMENT,
        city          TEXT NOT NULL,
        location      TEXT NOT NULL
    );
""")

# --- Create dim_cuisine ---
cur.execute("""
    CREATE TABLE dim_cuisine (
        cuisine_key     INTEGER PRIMARY KEY AUTOINCREMENT,
        primary_cuisine TEXT NOT NULL,
        cuisine_count   INTEGER NOT NULL
    );
""")

# --- Create dim_type ---
cur.execute("""
    CREATE TABLE dim_type (
        type_key   INTEGER PRIMARY KEY AUTOINCREMENT,
        rest_type  TEXT NOT NULL,
        category   TEXT NOT NULL
    );
""")

# --- Create dim_cluster (empty for now) ---
cur.execute("""
    CREATE TABLE dim_cluster (
        cluster_id   INTEGER PRIMARY KEY,
        label        TEXT,
        description  TEXT
    );
""")

# --- Create fact_restaurant ---
cur.execute("""
    CREATE TABLE fact_restaurant (
        restaurant_id  INTEGER PRIMARY KEY,
        location_key   INTEGER NOT NULL,
        cuisine_key    INTEGER NOT NULL,
        type_key       INTEGER NOT NULL,
        rate           REAL,
        votes          INTEGER,
        cost           REAL,
        online_order   INTEGER,
        book_table     INTEGER,
        cluster_id     INTEGER,
        FOREIGN KEY (location_key) REFERENCES dim_location(location_key),
        FOREIGN KEY (cuisine_key)  REFERENCES dim_cuisine(cuisine_key),
        FOREIGN KEY (type_key)     REFERENCES dim_type(type_key),
        FOREIGN KEY (cluster_id)   REFERENCES dim_cluster(cluster_id)
    );
""")

conn.commit()
print("      Tables created.")

# ===========================================================================
# 5. Insert data
# ===========================================================================
print("\n[5/6] Inserting data into tables ...")

# Insert dimensions with explicit primary keys
dim_location[["location_key", "city", "location"]].to_sql(
    "dim_location", conn, if_exists="append", index=False
)
dim_cuisine[["cuisine_key", "primary_cuisine", "cuisine_count"]].to_sql(
    "dim_cuisine", conn, if_exists="append", index=False
)
dim_type[["type_key", "rest_type", "category"]].to_sql(
    "dim_type", conn, if_exists="append", index=False
)

# Insert fact
fact.to_sql("fact_restaurant", conn, if_exists="append", index=False)

conn.commit()

# ===========================================================================
# 6. Print final row counts
# ===========================================================================
print("\n[6/6] Row counts per table:")
for tbl in ["dim_location", "dim_cuisine", "dim_type", "dim_cluster", "fact_restaurant"]:
    count = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    print(f"      {tbl:25s} : {count:,} rows")

conn.close()
print(f"\n[OK] Data warehouse saved to {DB_PATH}")
print("=" * 60)
