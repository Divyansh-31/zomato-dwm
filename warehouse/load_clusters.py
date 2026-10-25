"""
warehouse/load_clusters.py — Load clustering results into the Data Warehouse
=============================================================================

Run this AFTER Person B delivers the clustering outputs:
  • outputs/clustered_restaurants.csv  (columns: restaurant_id, cluster_id)
  • outputs/cluster_profiles.csv       (columns: cluster_id, label, size,
        avg_rate, avg_cost, avg_votes, top_cuisines)

Steps
─────
1. Read both CSVs.
2. Populate dim_cluster with profile info.
3. Update fact_restaurant.cluster_id for every matched restaurant.
4. Validate — assert all restaurant_ids in the clustering output
   exist in fact_restaurant; report mismatches.
5. Run cluster-based OLAP queries:
   a. Cluster × city counts
   b. Cluster × category counts
   c. Avg cost per cluster per location
   d. Underserved locations for the "budget & highly-rated" cluster
6. Save CSVs to outputs/olap/ and a grouped bar chart to
   figures/olap_cluster_distribution.png
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DB_PATH           = os.path.join("warehouse", "zomato_dw.db")
CLUSTERED_CSV     = os.path.join("outputs", "clustered_restaurants.csv")
PROFILES_CSV      = os.path.join("outputs", "cluster_profiles.csv")
OUT_DIR           = os.path.join("outputs", "olap")
FIG_DIR           = "figures"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 60)
print("LOAD CLUSTERS INTO DATA WAREHOUSE")
print("=" * 60)

# ===========================================================================
# 1. Read clustering outputs
# ===========================================================================
print(f"\n[1/5] Reading clustering CSVs ...")
df_clustered = pd.read_csv(CLUSTERED_CSV)
df_profiles  = pd.read_csv(PROFILES_CSV)
print(f"      clustered_restaurants : {len(df_clustered):,} rows")
print(f"      cluster_profiles     : {len(df_profiles):,} rows")
print(f"      Cluster IDs found    : {sorted(df_profiles['cluster_id'].unique())}")

# ===========================================================================
# 2. Populate dim_cluster
# ===========================================================================
print(f"\n[2/5] Populating dim_cluster ...")

conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()
cur.execute("PRAGMA foreign_keys = ON;")

# Clear existing dim_cluster data
cur.execute("DELETE FROM dim_cluster;")

# Build description from profile columns
for _, row in df_profiles.iterrows():
    desc = (
        f"Size: {int(row['size'])}, "
        f"Avg Rate: {row['avg_rate']:.2f}, "
        f"Avg Cost: {row['avg_cost']:.1f}, "
        f"Avg Votes: {row['avg_votes']:.0f}, "
        f"Top Cuisines: {row['top_cuisines']}"
    )
    cur.execute(
        "INSERT INTO dim_cluster (cluster_id, label, description) VALUES (?, ?, ?);",
        (int(row["cluster_id"]), str(row["label"]), desc),
    )
    print(f"      Cluster {int(row['cluster_id'])}: {row['label']}")

conn.commit()
count = cur.execute("SELECT COUNT(*) FROM dim_cluster").fetchone()[0]
print(f"      dim_cluster now has {count} rows.")

# ===========================================================================
# 3. Update fact_restaurant.cluster_id
# ===========================================================================
print(f"\n[3/5] Updating fact_restaurant.cluster_id ...")

# First, reset all cluster_ids to NULL
cur.execute("UPDATE fact_restaurant SET cluster_id = NULL;")

# Batch update using executemany
updates = list(zip(
    df_clustered["cluster_id"].astype(int).tolist(),
    df_clustered["restaurant_id"].astype(int).tolist(),
))
cur.executemany(
    "UPDATE fact_restaurant SET cluster_id = ? WHERE restaurant_id = ?;",
    updates,
)
conn.commit()

# Count how many were updated
assigned = cur.execute(
    "SELECT COUNT(*) FROM fact_restaurant WHERE cluster_id IS NOT NULL"
).fetchone()[0]
total = cur.execute("SELECT COUNT(*) FROM fact_restaurant").fetchone()[0]
print(f"      {assigned:,} / {total:,} restaurants assigned a cluster_id.")

# ===========================================================================
# 4. Validate — check for mismatches
# ===========================================================================
print(f"\n[4/5] Validating restaurant_id matches ...")

# Get all restaurant_ids from fact_restaurant
fact_ids = set(
    r[0] for r in cur.execute("SELECT restaurant_id FROM fact_restaurant").fetchall()
)
cluster_ids_set = set(df_clustered["restaurant_id"].astype(int).tolist())

# IDs in clustering output but NOT in the warehouse
missing = cluster_ids_set - fact_ids
if missing:
    print(f"      [WARN] {len(missing)} restaurant_id(s) in clustering output "
          f"NOT found in fact_restaurant:")
    # Show up to 10 examples
    for rid in sorted(missing)[:10]:
        print(f"         - {rid}")
    if len(missing) > 10:
        print(f"         ... and {len(missing) - 10} more.")
else:
    print(f"      [OK] All {len(cluster_ids_set):,} restaurant_ids matched.")

# IDs in fact but not clustered (informational)
not_clustered = fact_ids - cluster_ids_set
if not_clustered:
    print(f"      [INFO] {len(not_clustered):,} restaurant(s) in the warehouse "
          f"have no cluster assignment.")

# ===========================================================================
# 5. Cluster OLAP queries
# ===========================================================================
print(f"\n[5/5] Running cluster OLAP queries ...")

# Helper
def save_query(df, filename, title):
    path = os.path.join(OUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    print(df.to_string(index=False))
    print(f"  → Saved to {path}")

# ----- 5a. Cluster × City counts -----
sql_a = """
    SELECT  dc.label         AS cluster_label,
            dl.city,
            COUNT(*)         AS restaurant_count
    FROM    fact_restaurant  f
    JOIN    dim_cluster      dc ON f.cluster_id   = dc.cluster_id
    JOIN    dim_location     dl ON f.location_key  = dl.location_key
    GROUP BY dc.label, dl.city
    ORDER BY dc.label, restaurant_count DESC;
"""
df_a = pd.read_sql_query(sql_a, conn)
save_query(df_a, "cluster_city_counts.csv", "Cluster × City Counts")

# ----- 5b. Cluster × Category counts -----
sql_b = """
    SELECT  dc.label         AS cluster_label,
            dt.category,
            COUNT(*)         AS restaurant_count
    FROM    fact_restaurant  f
    JOIN    dim_cluster      dc ON f.cluster_id  = dc.cluster_id
    JOIN    dim_type         dt ON f.type_key    = dt.type_key
    GROUP BY dc.label, dt.category
    ORDER BY dc.label, restaurant_count DESC;
"""
df_b = pd.read_sql_query(sql_b, conn)
save_query(df_b, "cluster_category_counts.csv", "Cluster × Category Counts")

# ----- 5c. Average cost per cluster per location -----
sql_c = """
    SELECT  dc.label         AS cluster_label,
            dl.city,
            dl.location,
            ROUND(AVG(f.cost), 2) AS avg_cost,
            COUNT(*)              AS restaurant_count
    FROM    fact_restaurant  f
    JOIN    dim_cluster      dc ON f.cluster_id   = dc.cluster_id
    JOIN    dim_location     dl ON f.location_key  = dl.location_key
    GROUP BY dc.label, dl.city, dl.location
    ORDER BY dc.label, avg_cost DESC;
"""
df_c = pd.read_sql_query(sql_c, conn)
save_query(df_c, "cluster_location_avg_cost.csv",
           "Average Cost per Cluster per Location")

# ----- 5d. Underserved areas for the budget-and-highly-rated cluster -----
print(f"\n{'─' * 60}")
print("  Identifying the budget & highly-rated cluster ...")
print(f"{'─' * 60}")

# Find median rate across clusters
cluster_stats = pd.read_sql_query("""
    SELECT  dc.cluster_id,
            dc.label,
            AVG(f.rate)  AS avg_rate,
            AVG(f.cost)  AS avg_cost
    FROM    fact_restaurant f
    JOIN    dim_cluster     dc ON f.cluster_id = dc.cluster_id
    GROUP BY dc.cluster_id, dc.label;
""", conn)

median_rate = cluster_stats["avg_rate"].median()
print(f"  Median avg_rate across clusters: {median_rate:.2f}")

# Filter: above-median rate
above_median = cluster_stats[cluster_stats["avg_rate"] >= median_rate]

if len(above_median) > 0:
    # Pick the one with lowest avg_cost among above-median-rate clusters
    budget_cluster = above_median.loc[above_median["avg_cost"].idxmin()]
    budget_cid  = int(budget_cluster["cluster_id"])
    budget_label = budget_cluster["label"]
    print(f"  Budget & highly-rated cluster: id={budget_cid}, label='{budget_label}'")
    print(f"    avg_rate={budget_cluster['avg_rate']:.2f}, avg_cost={budget_cluster['avg_cost']:.1f}")

    # Locations where this cluster has the fewest restaurants
    sql_d = f"""
        SELECT  dl.city,
                dl.location,
                COUNT(*)  AS restaurant_count
        FROM    fact_restaurant  f
        JOIN    dim_location     dl ON f.location_key = dl.location_key
        WHERE   f.cluster_id = {budget_cid}
        GROUP BY dl.city, dl.location
        ORDER BY restaurant_count ASC
        LIMIT 20;
    """
    df_d = pd.read_sql_query(sql_d, conn)
    save_query(df_d, "underserved_locations.csv",
               f"Underserved Locations for '{budget_label}' Cluster (fewest restaurants)")
else:
    print("  [WARN] Could not determine a budget & highly-rated cluster (no data).")

# ----- Grouped bar chart: Cluster distribution by city -----
print(f"\n{'─' * 60}")
print("  Generating cluster distribution bar chart ...")
print(f"{'─' * 60}")

pivot_chart = df_a.pivot_table(
    values="restaurant_count",
    index="city",
    columns="cluster_label",
    aggfunc="sum",
    fill_value=0,
)

fig, ax = plt.subplots(figsize=(14, 7))
pivot_chart.plot(kind="bar", ax=ax, edgecolor="white", width=0.8)
ax.set_xlabel("City", fontsize=12)
ax.set_ylabel("Restaurant Count", fontsize=12)
ax.set_title("Cluster Distribution by City", fontsize=14, fontweight="bold")
ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
chart_path = os.path.join(FIG_DIR, "olap_cluster_distribution.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  [PLOT] Chart saved to {chart_path}")

# ---------------------------------------------------------------------------
conn.close()
print(f"\n{'=' * 60}")
print("[OK] Cluster loading and OLAP queries completed.")
print("=" * 60)
