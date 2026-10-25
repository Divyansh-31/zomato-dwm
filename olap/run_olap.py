"""
olap/run_olap.py — OLAP Queries on the Zomato Data Warehouse
=============================================================

Reads  : warehouse/zomato_dw.db
Writes : outputs/olap/*.csv   (one CSV per query)
         figures/olap_rollup.png  (bar chart for roll-up)

Queries implemented
───────────────────
1. Roll-up   : Avg cost & rating by city
2. Drill-down: Avg cost & rating by location (within top city)
3. Slice     : Filter rest_type = 'Casual Dining'
4. Dice      : cost ∈ [300, 800] AND rate ≥ 4.0
5. Pivot     : Category vs city, avg rating
6. Top 10    : Top 10 primary cuisines by restaurant count
"""

import os
import sqlite3
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt

# Try a clean style; fall back gracefully
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass  # default style is fine

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DB_PATH   = os.path.join("warehouse", "zomato_dw.db")
OUT_DIR   = os.path.join("outputs", "olap")
FIG_DIR   = "figures"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------
print("=" * 60)
print("OLAP QUERIES — Zomato Data Warehouse")
print("=" * 60)

conn = sqlite3.connect(DB_PATH)
print(f"Connected to {DB_PATH}\n")

# ===== Helper =====
def save_and_show(df, filename, title):
    """Print a formatted table and save to CSV."""
    path = os.path.join(OUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    print(df.to_string(index=False))
    print(f"\n  → Saved to {path}\n")

# ===========================================================================
# 1. ROLL-UP: Average cost & rating by CITY
#    (Aggregate from location level up to city level)
# ===========================================================================
print("[1/6] Roll-Up: Average cost & rating by city")

rollup_sql = """
    SELECT  l.city,
            ROUND(AVG(f.cost), 2)  AS avg_cost,
            ROUND(AVG(f.rate), 2)  AS avg_rate,
            COUNT(*)               AS restaurant_count
    FROM    fact_restaurant f
    JOIN    dim_location    l ON f.location_key = l.location_key
    GROUP BY l.city
    ORDER BY restaurant_count DESC;
"""
df_rollup = pd.read_sql_query(rollup_sql, conn)
save_and_show(df_rollup, "rollup_city.csv", "ROLL-UP — Avg Cost & Rating by City")

# --- Bar chart for roll-up ---
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(df_rollup["city"], df_rollup["avg_cost"], color="steelblue", edgecolor="white")
ax.set_xlabel("City", fontsize=12)
ax.set_ylabel("Average Cost (INR)", fontsize=12)
ax.set_title("Roll-Up: Average Cost by City", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "olap_rollup.png")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"  [PLOT] Bar chart saved to {fig_path}\n")

# ===========================================================================
# 2. DRILL-DOWN: Avg cost & rating by LOCATION within the top city
# ===========================================================================
print("[2/6] Drill-Down: Avg cost & rating by location within top city")

# Find the city with the most restaurants
top_city = df_rollup.iloc[0]["city"]
print(f"      Top city by count: {top_city}")

drilldown_sql = """
    SELECT  l.city,
            l.location,
            ROUND(AVG(f.cost), 2)  AS avg_cost,
            ROUND(AVG(f.rate), 2)  AS avg_rate,
            COUNT(*)               AS restaurant_count
    FROM    fact_restaurant f
    JOIN    dim_location    l ON f.location_key = l.location_key
    WHERE   l.city = ?
    GROUP BY l.city, l.location
    ORDER BY restaurant_count DESC;
"""
df_drill = pd.read_sql_query(drilldown_sql, conn, params=[top_city])
save_and_show(df_drill, "drilldown_location.csv",
              f"DRILL-DOWN — Avg Cost & Rating by Location in {top_city}")

# ===========================================================================
# 3. SLICE: Filter rest_type = 'Casual Dining'
# ===========================================================================
print("[3/6] Slice: rest_type = 'Casual Dining'")

slice_sql = """
    SELECT  f.restaurant_id,
            l.city,
            l.location,
            t.rest_type,
            t.category,
            f.rate,
            f.cost,
            f.votes
    FROM    fact_restaurant f
    JOIN    dim_location    l ON f.location_key = l.location_key
    JOIN    dim_type        t ON f.type_key     = t.type_key
    WHERE   t.rest_type = 'Casual Dining'
    ORDER BY f.rate DESC
    LIMIT 500;
"""
df_slice = pd.read_sql_query(slice_sql, conn)
save_and_show(df_slice.head(20), "slice_casual_dining.csv",
              "SLICE — Casual Dining (showing top 20 of up to 500)")
# Save the full 500 rows
df_slice.to_csv(os.path.join(OUT_DIR, "slice_casual_dining.csv"), index=False)

# ===========================================================================
# 4. DICE: cost BETWEEN 300 AND 800 AND rate >= 4.0
# ===========================================================================
print("[4/6] Dice: cost ∈ [300, 800] AND rate ≥ 4.0")

dice_sql = """
    SELECT  f.restaurant_id,
            l.city,
            l.location,
            c.primary_cuisine,
            t.rest_type,
            f.rate,
            f.cost,
            f.votes
    FROM    fact_restaurant f
    JOIN    dim_location    l ON f.location_key = l.location_key
    JOIN    dim_cuisine     c ON f.cuisine_key  = c.cuisine_key
    JOIN    dim_type        t ON f.type_key     = t.type_key
    WHERE   f.cost BETWEEN 300 AND 800
      AND   f.rate >= 4.0
    ORDER BY f.rate DESC, f.votes DESC
    LIMIT 500;
"""
df_dice = pd.read_sql_query(dice_sql, conn)
save_and_show(df_dice.head(20), "dice_cost_rate.csv",
              "DICE — Cost 300–800 & Rate ≥ 4.0 (showing top 20 of up to 500)")
df_dice.to_csv(os.path.join(OUT_DIR, "dice_cost_rate.csv"), index=False)

# ===========================================================================
# 5. PIVOT: Category vs City — average rating
# ===========================================================================
print("[5/6] Pivot: Category vs City — average rating")

pivot_sql = """
    SELECT  t.category,
            l.city,
            f.rate
    FROM    fact_restaurant f
    JOIN    dim_type        t ON f.type_key     = t.type_key
    JOIN    dim_location    l ON f.location_key = l.location_key;
"""
df_pivot_raw = pd.read_sql_query(pivot_sql, conn)

# Build pandas pivot table
df_pivot = df_pivot_raw.pivot_table(
    values="rate",
    index="category",
    columns="city",
    aggfunc="mean"
).round(2)

pivot_path = os.path.join(OUT_DIR, "pivot_category_city.csv")
df_pivot.to_csv(pivot_path)
print(f"{'─' * 60}")
print("  PIVOT — Category × City (Average Rating)")
print(f"{'─' * 60}")
print(df_pivot.to_string())
print(f"\n  → Saved to {pivot_path}\n")

# ===========================================================================
# 6. TOP 10: Primary cuisines by restaurant count
# ===========================================================================
print("[6/6] Top 10: Primary cuisines by count")

top10_sql = """
    SELECT  c.primary_cuisine,
            COUNT(*)             AS restaurant_count,
            ROUND(AVG(f.rate), 2) AS avg_rate,
            ROUND(AVG(f.cost), 2) AS avg_cost
    FROM    fact_restaurant f
    JOIN    dim_cuisine     c ON f.cuisine_key = c.cuisine_key
    GROUP BY c.primary_cuisine
    ORDER BY restaurant_count DESC
    LIMIT 10;
"""
df_top10 = pd.read_sql_query(top10_sql, conn)
save_and_show(df_top10, "top10_cuisines.csv",
              "TOP 10 — Primary Cuisines by Restaurant Count")

# ---------------------------------------------------------------------------
conn.close()
print("=" * 60)
print("[OK] All OLAP queries completed.")
print("=" * 60)
