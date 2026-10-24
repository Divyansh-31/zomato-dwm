# Segmentation of Restaurants Using K-Means Clustering on Zomato Bangalore Data
## A Data Warehousing and Mining (DWM) Mini Project

This README is the single source of truth for both team members. Read sections 1 to 6 together once, then follow only your own part (Section 7 for Person A, Section 8 for Person B), then meet again for Section 9 (integration).

## Table of Contents
1. Project overview
2. How the work is split (and why you can work at the same time)
3. The data contract (do not break this)
4. Folder structure
5. One-time setup (both members)
6. Git rules
7. PERSON A: Warehouse, ETL and OLAP
8. PERSON B: Clustering and analysis
9. Integration (day 3)
10. Report outline and who writes what
11. Viva preparation
12. Troubleshooting
13. Master checklist (one list for both)

---

## 1. Project overview
**Problem statement:** Zomato lists thousands of restaurants, but customers and restaurant owners have no clear way to see which types of restaurants exist in the market. Clustering groups restaurants automatically by cost, rating and popularity, so each segment can be understood and acted on.

### Who benefits
- **Customers:** faster choice (for example "budget but highly rated" places).
- **New restaurant owners:** which segments are crowded and which areas are underserved.
- **Existing restaurants:** compare against others in the same cluster.
- **The platform:** targeted promotions per segment.

**Why clustering:** There are no predefined labels for "budget" or "premium" restaurants, so the groups must come from the data (unsupervised learning).

**Honest note on originality:** Restaurant clustering on Zomato data already exists in tutorials and papers. Do not claim novelty. Our contribution is the proper data warehouse design with OLAP plus a comparison of three clustering algorithms with stability checks.

**Dataset:** Zomato Bangalore Restaurants (Kaggle), about 51,717 rows and 17 columns.

**Tech stack:** Python 3.11 or 3.12, pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, SQLite (built into Python), Git/GitHub, VS Code.

---

## 2. How the work is split

| | Person A: Warehouse and OLAP | Person B: Clustering and analysis |
|---|---|---|
| **Builds** | Cleaning/ETL, SQLite star schema, OLAP queries | K-Means, Hierarchical, DBSCAN, evaluation, plots |
| **Hands to B** | `data/processed/zomato_clean.csv` | |
| **Hands to A** | | `outputs/clustered_restaurants.csv`, `outputs/cluster_profiles.csv`, `outputs/metrics.csv` |
| **Report sections** | Problem, data, ETL, schema, OLAP | Algorithms, evaluation, interpretation, limitations |

### Can we work at the same time? Yes.
The two parts meet at only two handoff points:
1. **A gives B the real cleaned data.** B does not wait. B starts on mock data (prompt B1) with the same columns, then swaps in the real file when it arrives.
2. **B gives A the three output CSVs.** A does not wait. Prompts A1 to A3 need nothing from B. Only prompt A4 (the merge step) needs B's files.

So both are busy on days 1 and 2, and the merge happens on day 3.

---

## 3. The data contract (do not break this)
Column names, order and file names are fixed. If an AI tool renames anything, stop it. The contract wins. If you really need to change it, tell the other person first.

### 3.1 `data/processed/zomato_clean.csv` (made by A, used by B)
| Column | Type | Notes |
|---|---|---|
| `restaurant_id` | int | unique, 1..N |
| `name` | text | |
| `city` | text | from `listed_in(city)` |
| `location` | text | locality |
| `rest_type` | text | for example Casual Dining, Cafe |
| `category` | text | from `listed_in(type)` |
| `cuisines` | text | comma-separated |
| `primary_cuisine` | text | first cuisine in the list |
| `cuisine_count` | int | number of cuisines |
| `online_order` | int | 1 or 0 |
| `book_table` | int | 1 or 0 |
| `rate` | float | 0 to 5 |
| `votes` | int | |
| `log_votes` | float | log1p(votes) |
| `cost` | float | rupees for two people |

*Columns must appear in exactly this order.*

### 3.2 Files produced by B (used by A)
- `outputs/clustered_restaurants.csv`: `restaurant_id, cluster_id`
- `outputs/cluster_profiles.csv`: `cluster_id, label, size, avg_rate, avg_cost, avg_votes, top_cuisines`
- `outputs/metrics.csv`: `algorithm, n_clusters, silhouette, davies_bouldin, calinski_harabasz`

### 3.3 Important cleaning decision
Remove duplicates on `(name, address)`, not on `name` alone. Deduplicating on name alone deletes every branch of a chain (for example all Domino's outlets), which distorts the clusters. Address is dropped only after deduplication.

---

## 4. Folder structure
```
zomato-dwm/
  README.md
  DOCUMENTATION.md
  requirements.txt
  .gitignore
  data/
    raw/zomato.csv                  <- downloaded by each person (not in git)
    processed/zomato_clean.csv      <- made by A, sent to B (not in git)
  warehouse/
    build_dw.py                     <- A
    load_clusters.py                <- A
    zomato_dw.db                    <- generated (not in git)
  etl/
    clean.py                        <- A
  olap/
    run_olap.py                     <- A
  clustering/
    make_mock.py                    <- B
    choose_k.py                     <- B
    run_kmeans.py                   <- B
    compare_algorithms.py           <- B
    stability.py                    <- B
  outputs/
    clustered_restaurants.csv       <- B
    cluster_profiles.csv            <- B
    metrics.csv                     <- B
    stability.csv                   <- B
    olap/                           <- A (OLAP result CSVs)
  figures/                          <- both (different filenames)
```

### Ownership
- **A edits only:** `etl/`, `warehouse/`, `olap/`, `outputs/olap/`
- **B edits only:** `clustering/`, and the CSV files directly inside `outputs/`
- **Only A edits:** `requirements.txt`, `.gitignore`, `README.md` (B asks A if a new library is needed)

---

## 5. Star Schema Diagram

```
                dim_location
                (location_key, city, location)
                        |
dim_type ---- fact_restaurant ---- dim_cuisine
(type_key,    (restaurant_id, location_key,   (cuisine_key,
 rest_type,    cuisine_key, type_key, rate,    primary_cuisine,
 category)     votes, cost, online_order,      cuisine_count)
               book_table, cluster_id)
                        |
                dim_cluster
                (cluster_id, label, description)
```

---

## 6. Execution Pipeline

```bash
# 1. Clean raw data (Person A)
python etl/clean.py

# 2. Build star schema data warehouse (Person A)
python warehouse/build_dw.py

# 3. Run OLAP queries (Person A)
python olap/run_olap.py

# 4. Clustering pipeline (Person B)
python clustering/choose_k.py
python clustering/run_kmeans.py
python clustering/compare_algorithms.py
python clustering/stability.py

# 5. Integration: Load clusters into warehouse & run cluster OLAP (Person A)
python warehouse/load_clusters.py
```
