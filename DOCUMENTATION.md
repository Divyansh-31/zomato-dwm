# Segmentation of Restaurants Using K-Means Clustering on Zomato Bangalore Data

## A Data Warehousing and Mining (DWM) Approach — Project Documentation & Viva Reference

---

## Table of Contents
1. [Project Overview & Problem Statement](#1-project-overview--problem-statement)
2. [Data Contract & Schema Standards](#2-data-contract--schema-standards)
3. [Project Architecture & Directory Structure](#3-project-architecture--directory-structure)
4. [ETL Pipeline (`etl/clean.py`)](#4-etl-pipeline-etlcleanpy)
5. [Data Warehouse & Star Schema Design (`warehouse/build_dw.py`)](#5-data-warehouse--star-schema-design-warehousebuild_dwpy)
6. [OLAP Analytical Operations (`olap/run_olap.py`)](#6-olap-analytical-operations-olaprun_olappy)
7. [Integration & Cluster Loading (`warehouse/load_clusters.py`)](#7-integration--cluster-loading-warehouseload_clusterspy)
8. [Step-by-Step Execution Guide](#8-step-by-step-execution-guide)
9. [Comprehensive Viva Q&A Reference](#9-comprehensive-viva-qa-reference)
10. [Troubleshooting & Verification](#10-troubleshooting--verification)

---

## 1. Project Overview & Problem Statement

### Problem Statement
Zomato lists tens of thousands of restaurants in Bangalore across dozens of localities. However, customers, entrepreneurs, and platform administrators face major discovery hurdles:
- **Customers** have difficulty identifying restaurants that offer the ideal trade-off between price and quality (e.g., "affordable yet highly rated").
- **Prospective restaurant owners** cannot easily analyze market saturation or identify underserved areas for specific restaurant types.
- **Platform managers** require data-backed customer segmentation to design targeted promotions and optimize restaurant recommendations.

Because there are no ground-truth category labels for segments like "Budget Hidden Gems" or "High-End Fine Dining", **unsupervised machine learning (clustering)** is applied to segment restaurants based on their core numeric attributes: **cost, rating, and popularity (votes)**.

### Academic Contribution
While basic restaurant clustering exists in online tutorials, this project's unique contribution is:
1. **Formal Data Warehouse Engineering:** Implementation of a multidimensional **Star Schema** in SQLite.
2. **Comprehensive OLAP Implementation:** Practical execution of **Roll-Up, Drill-Down, Slice, Dice, and Pivot** operations on relational warehouse tables.
3. **Multi-Algorithm Comparative Analysis:** Benchmarking K-Means against Hierarchical (Agglomerative) and DBSCAN clustering with cluster stability validation.

### Tech Stack
| Component | Tool / Library | Role |
|---|---|---|
| Language | Python 3.11 / 3.12 | Primary programming language |
| Data Processing | `pandas`, `numpy` | ETL transformations, matrix operations |
| Warehouse Storage | `sqlite3` | Relational star schema data warehouse |
| Machine Learning | `scikit-learn`, `scipy` | Feature scaling, K-Means, hierarchical clustering, DBSCAN |
| Visualizations | `matplotlib`, `seaborn` | OLAP bar charts, elbow curve, PCA scatter plots |

---

## 2. Data Contract & Schema Standards

The **Data Contract** guarantees seamless interoperability between Person A (Warehouse/ETL/OLAP) and Person B (Clustering/Analysis). Neither party alters column names or ordering without mutual agreement.

### 2.1 Cleaned Dataset: `data/processed/zomato_clean.csv` (Person A → Person B)
Contains 15 standardized columns in exact sequence:

| # | Column Name | Data Type | Description & Engineering Rule | Example |
|---|---|---|---|---|
| 1 | `restaurant_id` | `int` | Surrogate primary key (1 .. N) | `1` |
| 2 | `name` | `text` | Cleaned restaurant name | `Jalsa` |
| 3 | `city` | `text` | Administrative territory (`listed_in(city)`) | `Banashankari` |
| 4 | `location` | `text` | Local neighborhood/suburb | `Banashankari` |
| 5 | `rest_type` | `text` | Facility format (e.g., Casual Dining, Quick Bites) | `Casual Dining` |
| 6 | `category` | `text` | Meal category (`listed_in(type)`: Dine-out, Delivery, etc.) | `Buffet` |
| 7 | `cuisines` | `text` | Comma-separated list of offered cuisines | `North Indian, Mughlai` |
| 8 | `primary_cuisine` | `text` | First cuisine in the list (main culinary specialty) | `North Indian` |
| 9 | `cuisine_count` | `int` | Number of distinct cuisines offered | `2` |
| 10 | `online_order` | `int` | Binary flag: `1` if ordering available, `0` otherwise | `1` |
| 11 | `book_table` | `int` | Binary flag: `1` if reservation available, `0` otherwise | `1` |
| 12 | `rate` | `float` | Customer rating normalized to 0.0 – 5.0 scale | `4.1` |
| 13 | `votes` | `int` | Raw user vote tally | `775` |
| 14 | `log_votes` | `float` | Log-transformed popularity: $\log(1 + \text{votes})$ | `6.654` |
| 15 | `cost` | `float` | Approximate cost for two people in INR | `800.0` |

### 2.2 Clustering Outputs (Person B → Person A)
- **`outputs/clustered_restaurants.csv`**: Contains `restaurant_id` (FK) and `cluster_id` (integer cluster assignment).
- **`outputs/cluster_profiles.csv`**: Contains summary statistics per segment: `cluster_id`, `label`, `size`, `avg_rate`, `avg_cost`, `avg_votes`, `top_cuisines`.
- **`outputs/metrics.csv`**: Evaluation comparison: `algorithm`, `n_clusters`, `silhouette`, `davies_bouldin`, `calinski_harabasz`.
- **`outputs/stability.csv`**: Adjusted Rand Index (ARI) over multiple random initializations.

---

## 3. Project Architecture & Directory Structure

```
DWM_PROJ/
├── README.md                           # Master project guide & checklist
├── DOCUMENTATION.md                    # In-depth technical & viva documentation
├── requirements.txt                    # Project library requirements
├── .gitignore                          # Git exclusions (datasets, sqlite db, venv)
│
├── data/
│   ├── raw/
│   │   └── zomato.csv                  # Raw Kaggle Bangalore dataset (~51,717 rows)
│   └── processed/
│       └── zomato_clean.csv            # Cleaned contract CSV (9,478 rows × 15 cols)
│
├── etl/
│   └── clean.py                        # [Person A] Data cleaning & feature engineering
│
├── warehouse/
│   ├── build_dw.py                     # [Person A] Star schema database builder
│   ├── load_clusters.py                # [Person A] Integration script for cluster profiles
│   └── zomato_dw.db                    # [Person A] SQLite Star Schema warehouse
│
├── olap/
│   └── run_olap.py                     # [Person A] SQL & pandas OLAP queries
│
├── clustering/                         # [Person B] Machine learning scripts
│   ├── make_mock.py                    # Mock data generator for independent testing
│   ├── choose_k.py                     # Elbow method & silhouette analysis
│   ├── run_kmeans.py                   # Final K-Means fitting & profile generation
│   ├── compare_algorithms.py           # KMeans vs Agglomerative vs DBSCAN
│   └── stability.py                    # Multi-seed ARI stability assessment
│
├── outputs/
│   ├── clustered_restaurants.csv       # [Person B] Restaurant-cluster mapping
│   ├── cluster_profiles.csv            # [Person B] Cluster centroid summaries
│   ├── metrics.csv                     # [Person B] Clustering score comparison
│   ├── stability.csv                   # [Person B] Stability scores
│   └── olap/                           # [Person A] Result CSVs from OLAP queries
│       ├── rollup_city.csv
│       ├── drilldown_location.csv
│       ├── slice_casual_dining.csv
│       ├── dice_cost_rate.csv
│       ├── pivot_category_city.csv
│       ├── top10_cuisines.csv
│       ├── cluster_city_counts.csv
│       ├── cluster_category_counts.csv
│       ├── cluster_location_avg_cost.csv
│       └── underserved_locations.csv
│
└── figures/                            # Visual artifacts for report & presentation
    ├── olap_rollup.png                 # [Person A] Average cost across cities
    ├── olap_cluster_distribution.png   # [Person A] Grouped cluster distribution
    ├── elbow.png                       # [Person B] K-Means inertia elbow curve
    ├── silhouette_by_k.png             # [Person B] Silhouette score vs k
    ├── clusters_pca.png                # [Person B] 2D PCA cluster projection
    ├── dendrogram.png                  # [Person B] Hierarchical dendrogram
    └── kdistance.png                   # [Person B] DBSCAN k-distance knee plot
```

---

## 4. ETL Pipeline (`etl/clean.py`)

ETL stands for **Extract, Transform, Load**. The pipeline reads the messy raw Zomato dataset, handles missing data and anomalies, creates engineered features, and enforces the data contract.

```
┌────────────────────────────────────────────────────────┐
│             EXTRACT: data/raw/zomato.csv               │
│          51,717 rows × 17 columns (Kaggle)             │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                      TRANSFORM                         │
│  1. Select 11 relevant columns + address               │
│  2. Clean rate: strip "/5", drop "NEW", "-", NaNs      │
│     (Removed 10,052 unrated/new listings)              │
│  3. Clean cost: strip commas, convert to float         │
│     (Removed 247 missing cost records)                 │
│  4. Binary Encode: Yes→1, No→0 for online/book         │
│  5. Deduplicate on (name, address)                     │
│     (Crucial: keeps chain outlets, drops duplicates)   │
│  6. Feature Engineering:                               │
│     - restaurant_id = 1..N                             │
│     - primary_cuisine = first item in cuisine list     │
│     - cuisine_count = count of listed cuisines         │
│     - log_votes = log(1 + votes)                       │
│  7. Impute missing categories/cuisines with "Unknown"  │
│  8. Strip whitespace from all text columns             │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│           LOAD: data/processed/zomato_clean.csv        │
│          9,478 clean rows × 15 contract columns        │
│          Passed 5 Automated Validation Assertions      │
└────────────────────────────────────────────────────────┘
```

### Key ETL Decisions & Row Progression
| ETL Stage | Resulting Rows | Rationale |
|---|---|---|
| Raw CSV loaded | 51,717 | Baseline Kaggle dataset |
| Invalid rate dropped | 41,665 | "NEW", "-" or missing ratings cannot be used in clustering |
| Missing cost dropped | 41,418 | Cost is a primary clustering axis |
| Deduplicated on `(name, address)` | **9,478** | Eliminates repeated listings of the same restaurant across different search categories, while preserving all unique physical branches of chains |

> [!IMPORTANT]
> **Why deduplicate on `(name, address)` instead of `name` alone?**
> Chains such as *Domino's*, *McDonald's*, or *Café Coffee Day* operate dozens of physical outlets across Bangalore. Each branch serves a specific neighborhood with distinct customer vote totals and service characteristics. Deduplicating by name alone would discard all but one branch, introducing severe geographical bias.

---

## 5. Data Warehouse & Star Schema Design (`warehouse/build_dw.py`)

### What is a Star Schema?
A **Star Schema** is the foundational dimensional data modeling pattern. It consists of:
- **One central Fact Table (`fact_restaurant`)**: Stores numerical, additive measures (`rate`, `votes`, `cost`, binary flags) and foreign keys referencing dimensions.
- **Surrounding Dimension Tables**: Store descriptive, contextual attributes (`dim_location`, `dim_cuisine`, `dim_type`, `dim_cluster`).

### Star Schema Entity-Relationship Diagram

```
                dim_location
                ┌──────────────────────────────────┐
                │ location_key (PK, INTEGER)       │
                │ city         (TEXT)              │
                │ location     (TEXT)              │
                └────────────────┬─────────────────┘
                                 │
                                 │ 1:N
                                 ▼
dim_type        fact_restaurant                    dim_cuisine
┌─────────────────────────┐      ┌─────────────────────────────┐   ┌───────────────────────────────┐
│ type_key (PK, INTEGER)  │ 1:N  │ restaurant_id  (PK, INTEGER)│N:1│ cuisine_key (PK, INTEGER)     │
│ rest_type(TEXT)         ├─────►│ location_key   (FK, INTEGER)│◄──┤ primary_cuisine (TEXT)        │
│ category (TEXT)         │      │ cuisine_key    (FK, INTEGER)│   │ cuisine_count   (INTEGER)     │
└─────────────────────────┘      │ type_key       (FK, INTEGER)│   └───────────────────────────────┘
                                 │ cluster_id     (FK, NULLABLE│
                                 │ rate           (REAL)       │
                                 │ votes          (INTEGER)    │
                                 │ cost           (REAL)       │
                                 │ online_order   (INTEGER)    │
                                 │ book_table     (INTEGER)    │
                                 └──────────────┬──────────────┘
                                                │
                                            N:1 │
                                                ▼
                                           dim_cluster
                                 ┌─────────────────────────────┐
                                 │ cluster_id   (PK, INTEGER)  │
                                 │ label        (TEXT)         │
                                 │ description  (TEXT)         │
                                 └─────────────────────────────┘
```

### Table Definitions & Warehouse Statistics
| Table | Type | Rows in Warehouse | Description |
|---|---|---|---|
| `dim_location` | Dimension | 333 | Unique `(city, location)` pairs across Bangalore |
| `dim_cuisine` | Dimension | 343 | Unique combinations of `(primary_cuisine, cuisine_count)` |
| `dim_type` | Dimension | 203 | Unique combinations of `(rest_type, category)` |
| `dim_cluster` | Dimension | *Populated in Day 3* | Machine learning cluster segments with descriptive labels |
| `fact_restaurant` | Fact Table | 9,478 | Central fact table containing core measures & foreign keys |

---

## 6. OLAP Analytical Operations (`olap/run_olap.py`)

**OLAP (Online Analytical Processing)** enables multidimensional queries for business intelligence and reporting.

### 6.1 Roll-Up (Aggregation: Location $\rightarrow$ City)
Climbs up the location hierarchy from fine-grained locations to broader administrative cities.
- **SQL Query:**
  ```sql
  SELECT  l.city,
          ROUND(AVG(f.cost), 2)  AS avg_cost,
          ROUND(AVG(f.rate), 2)  AS avg_rate,
          COUNT(*)               AS restaurant_count
  FROM    fact_restaurant f
  JOIN    dim_location    l ON f.location_key = l.location_key
  GROUP BY l.city
  ORDER BY restaurant_count DESC;
  ```
- **Business Question:** *Which Bangalore city areas contain the highest restaurant density, and how do average dining costs compare across regions?*
- **Key Insight:** BTM leads Bangalore with 1,122 restaurants (avg cost ₹497.70, avg rating 3.70), while Brigade Road represents a more premium hub (avg cost ₹774.99).

---

### 6.2 Drill-Down (Disaggregation: City $\rightarrow$ Specific Locality)
Navigates down the location hierarchy to analyze neighborhoods within a single high-volume city (`BTM`).
- **SQL Query:**
  ```sql
  SELECT  l.city,
          l.location,
          ROUND(AVG(f.cost), 2)  AS avg_cost,
          ROUND(AVG(f.rate), 2)  AS avg_rate,
          COUNT(*)               AS restaurant_count
  FROM    fact_restaurant f
  JOIN    dim_location    l ON f.location_key = l.location_key
  WHERE   l.city = 'BTM'
  GROUP BY l.city, l.location
  ORDER BY restaurant_count DESC;
  ```
- **Business Question:** *Within the high-density BTM territory, which sub-localities host the highest-rated dining spots?*
- **Key Insight:** Koramangala 5th Block achieves the highest local rating (3.92) with an average cost of ₹614.78 across 203 restaurants.

---

### 6.3 Slice (Filter along 1 Dimension)
Takes a rectangular slice of the data cube by fixing one dimension (`rest_type = 'Casual Dining'`).
- **SQL Query:**
  ```sql
  SELECT  f.restaurant_id, l.city, l.location, t.rest_type, f.rate, f.cost
  FROM    fact_restaurant f
  JOIN    dim_location    l ON f.location_key = l.location_key
  JOIN    dim_type        t ON f.type_key     = t.type_key
  WHERE   t.rest_type = 'Casual Dining'
  ORDER BY f.rate DESC;
  ```
- **Business Question:** *How do Casual Dining establishments perform in terms of customer satisfaction and pricing?*

---

### 6.4 Dice (Multi-Dimensional Windowing)
Selects a sub-cube by applying simultaneous boundary conditions across multiple dimensions:
`cost BETWEEN 300 AND 800` AND `rate >= 4.0`.
- **SQL Query:**
  ```sql
  SELECT  f.restaurant_id, l.city, l.location, c.primary_cuisine, f.rate, f.cost, f.votes
  FROM    fact_restaurant f
  JOIN    dim_location    l ON f.location_key = l.location_key
  JOIN    dim_cuisine     c ON f.cuisine_key  = c.cuisine_key
  WHERE   f.cost BETWEEN 300 AND 800 AND f.rate >= 4.0
  ORDER BY f.rate DESC, f.votes DESC;
  ```
- **Business Question:** *Where are Bangalore's "Sweet Spot" restaurants located—places with high customer satisfaction ($\ge 4.0$) at modest pricing (₹300 – ₹800)?*

---

### 6.5 Pivot (Matrix Cross-Tabulation)
Rotates the data perspective, tabulating average customer rating across two dimensions: **Category** (rows) vs **City** (columns).
- **Pandas Pivot Implementation:**
  ```python
  df_pivot = df.pivot_table(
      values="rate", index="category", columns="city", aggfunc="mean"
  ).round(2)
  ```
- **Business Question:** *How do dining formats (Buffet, Cafes, Delivery, Desserts, Dine-out, Pubs) vary in average rating across different city territories?*

---

### 6.6 Top 10 Primary Cuisines
Identifies Bangalore's market leaders in culinary offerings:
1. **North Indian** (2,272 restaurants, avg cost ₹586.22, avg rating 3.54)
2. **South Indian** (991 restaurants, avg cost ₹332.41, avg rating 3.57)
3. **Biryani** (578 restaurants, avg cost ₹441.78, avg rating 3.48)
4. **Cafe** (571 restaurants, avg cost ₹626.80, avg rating 3.77)
5. **Chinese** (550 restaurants, avg cost ₹594.73, avg rating 3.58)

---

## 7. Integration & Cluster Loading (`warehouse/load_clusters.py`)

When Person B finishes model training, `warehouse/load_clusters.py` executes the handoff:

1. **Populate `dim_cluster`**: Ingests `outputs/cluster_profiles.csv` and sets cluster metadata:
   - `cluster_id`
   - `label` (e.g., "Budget, Popular", "Premium Fine Dining", "Everyday Low-Cost")
   - `description` (computed string with profile averages and top cuisines)
2. **Update `fact_restaurant`**: Ingests `outputs/clustered_restaurants.csv` and assigns `cluster_id` to each matching `restaurant_id`.
3. **Integrity Validation**: Verifies that 100% of clustered restaurant IDs match existing fact records.
4. **Post-Clustering OLAP & Market Opportunity Discovery**:
   - **Cluster $\times$ City Distribution**: Identifies spatial geographic patterns of segments.
   - **Underserved Area Analysis**: Locates localities where the "Budget & Highly Rated" cluster has minimal representation—highlighting prime real estate opportunities for new restaurant entrants.

---

## 8. Step-by-Step Execution Guide

### End-to-End Pipeline Commands

```bash
# Ensure you are at the project root
cd /Users/divyansht/Downloads/DWM_PROJ

# ===================================================
# PHASE 1: Person A — ETL & Star Schema Warehouse
# ===================================================
# 1. Clean raw Kaggle data & generate zomato_clean.csv
python3 etl/clean.py

# 2. Build the SQLite Star Schema Data Warehouse
python3 warehouse/build_dw.py

# 3. Execute core OLAP operations & generate charts
python3 olap/run_olap.py

# ===================================================
# PHASE 2: Person B — Clustering (Parallel or Next)
# ===================================================
# 4. Determine optimal k (Elbow & Silhouette curves)
python3 clustering/choose_k.py

# 5. Fit K-Means, generate labels, and export cluster CSVs
python3 clustering/run_kmeans.py

# 6. Compare K-Means vs Agglomerative vs DBSCAN
python3 clustering/compare_algorithms.py

# 7. Test model stability across random seeds (ARI)
python3 clustering/stability.py

# ===================================================
# PHASE 3: Day 3 Integration (Both Team Members)
# ===================================================
# 8. Load cluster labels into warehouse & run cluster OLAP
python3 warehouse/load_clusters.py
```

---

## 9. Comprehensive Viva Q&A Reference

### Q1: Why use Clustering instead of Classification?
> **Answer:** Classification is a **supervised learning** technique that requires ground-truth labels during training (e.g., predicting whether a transaction is fraudulent). In the restaurant domain, there are no pre-existing labels defining what constitutes a "Budget Gem" versus an "Overpriced Tourist Spot". Clustering is an **unsupervised learning** technique that naturally partitions unlabelled data based on mathematical similarity.

### Q2: Why choose K-Means as the primary algorithm?
> **Answer:**
> 1. **Computational Efficiency:** K-Means converges in $\mathcal{O}(n \cdot k \cdot I)$ time, making it significantly faster on ~10,000+ data points than hierarchical clustering ($\mathcal{O}(n^2)$ memory and $\mathcal{O}(n^3)$ compute).
> 2. **Interpretability:** K-Means produces spherical clusters around explicit centroid coordinates $(\overline{\text{rate}}, \overline{\text{cost}}, \overline{\text{log\_votes}})$, directly translating into human-readable business personas.
> 3. **Scalability:** It integrates seamlessly with downstream SQL aggregations and production pipelines.

### Q3: Why is Feature Scaling (`StandardScaler`) mandatory?
> **Answer:** K-Means relies on Euclidean distance:
> $$d(p, q) = \sqrt{\sum (p_i - q_i)^2}$$
> If features are unscaled, `cost` (ranging from ₹40 to ₹6,000) would completely dominate distance calculations, rendering `rate` (scale 0.0 to 5.0) mathematically irrelevant. `StandardScaler` standardizes each feature to have a mean of $0$ and a standard deviation of $1$ ($z = \frac{x - \mu}{\sigma}$), ensuring equal weighting.

### Q4: Why apply a Log-Transformation to `votes`?
> **Answer:** Vote distributions exhibit extreme positive skewness (right-skewed Pareto distribution): the vast majority of restaurants have fewer than 100 votes, while iconic restaurants have over 16,000 votes. Without transformation, these extreme outliers would pull centroids toward themselves. Applying $\log_{1p}(\text{votes}) = \ln(1 + \text{votes})$ compresses the dynamic range, normalizes variance, and safely accommodates restaurants with 0 votes.

### Q5: What is a Star Schema, and why did we choose it over 3NF?
> **Answer:** A Star Schema organizes data into a central fact table surrounded by single-hop dimension tables. Third Normal Form (3NF) eliminates redundancy through extensive normalization, but requires complex multi-table joins. In Data Warehousing and OLAP, Star Schemas are preferred because:
> - Queries require only single-level joins between the fact and dimensions.
> - Aggregations and GROUP BY operations execute much faster.
> - The structure mirrors business dimensions (Who, Where, What format).

### Q6: Can you explain the difference between Roll-Up, Drill-Down, Slice, and Dice?
> **Answer:**
> - **Roll-Up:** Aggregates data up an architectural hierarchy (e.g., summarizing from `location` up to `city`).
> - **Drill-Down:** The inverse of roll-up; breaks down summary data into finer granularity (e.g., examining localities inside `BTM`).
> - **Slice:** Filters the data cube along a **single** dimension attribute (e.g., `rest_type = 'Casual Dining'`).
> - **Dice:** Filters the data cube along **multiple** dimensions simultaneously (e.g., `cost BETWEEN 300 AND 800` AND `rate >= 4.0`).
> - **Pivot:** Rotates axes to display cross-tabulations (e.g., meal category vs city).

### Q7: Why do we sample only 5,000 rows for Hierarchical Clustering?
> **Answer:** Agglomerative Hierarchical Clustering requires computing an $N \times N$ pairwise distance matrix. For $N = 9,478$, this matrix requires nearly $90\text{ million}$ float entries (~720 MB of RAM) and $\mathcal{O}(N^3)$ computational steps. Sampling 5,000 rows reduces memory pressure to manageable levels while preserving the underlying topological clustering structure for dendrogram analysis.

### Q8: What are the main limitations of this project?
> **Answer:**
> 1. **Temporal Horizon:** The dataset is a static snapshot; restaurant pricing, menu shifts, and ratings evolve over time.
> 2. **Geographic Specificity:** Data reflects Bangalore dining patterns and cannot be generalized to other metropolitan cities without recalibration.
> 3. **Geometric Assumption:** K-Means assumes spherical, isotropic clusters. If the natural market segments have irregular or elongated shapes, density-based approaches (DBSCAN) or Gaussian Mixture Models (GMM) may offer better boundaries.
> 4. **Exclusion of Textual Feedback:** The clustering uses numeric attributes; incorporating Natural Language Processing (NLP) sentiment scores from user reviews would create richer segmentations.

---

## 10. Troubleshooting & Verification

### Rapid Verification Commands
Run these commands to verify that all artifacts are correctly generated:

```bash
# 1. Verify Cleaned Dataset Structure
python3 -c "import pandas as pd; df = pd.read_csv('data/processed/zomato_clean.csv'); print('Shape:', df.shape); print('Nulls:', df.isna().sum().sum())"
# Expected: Shape: (9478, 15), Nulls: 0

# 2. Verify Warehouse SQLite Tables
python3 -c "import sqlite3; conn = sqlite3.connect('warehouse/zomato_dw.db'); cur = conn.cursor(); print('Tables:', [r[0] for r in cur.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()])"
# Expected: ['dim_location', 'dim_cuisine', 'dim_type', 'dim_cluster', 'fact_restaurant']

# 3. Verify OLAP Results Exist
ls -lh outputs/olap/
# Expected: 6+ CSV files generated

# 4. Verify Visual Figures Exist
ls -lh figures/
# Expected: olap_rollup.png and other generated figures
```

---
*Documentation compiled for the DWM Mini Project — Academic Year 2026.*
