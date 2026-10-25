"""
clustering/compare_algorithms.py -- Algorithm Benchmarking
==========================================================
Compares K-Means, Agglomerative (Hierarchical), and DBSCAN on the same
feature set.  Produces the contract metrics CSV and two diagnostic figures.

Notes
-----
- Agglomerative is fitted on a 5,000-row sample (O(n²) memory constraint).
- DBSCAN noise points (label == -1) are excluded from metric computation.
- All three algorithms use the same StandardScaler transform.

Outputs
-------
outputs/metrics.csv     -- algorithm, n_clusters, silhouette,
                          davies_bouldin, calinski_harabasz
figures/dendrogram.png  -- Hierarchical clustering dendrogram (500-row sub-sample)
figures/kdistance.png   -- DBSCAN k-distance knee plot (eps selection guide)
"""

import os
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# -- Constants ----------------------------------------------------------------
DATA_PATH      = "data/processed/zomato_clean.csv"
FEATURES       = ["rate", "log_votes", "cost"]
N_CLUSTERS     = 4
AGG_SAMPLE     = 5000    # max rows for Agglomerative (memory constraint)
DENDRO_SAMPLE  = 500     # rows used to draw dendrogram (visual clarity)
RANDOM_SEED    = 42
OUTPUTS_DIR    = "outputs"
FIGURES_DIR    = "figures"

# -- Bootstrap -----------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    print("[WARN]  Real data not found -- generating mock data ...")
    subprocess.run([sys.executable, "clustering/make_mock.py"], check=True)

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# -- Load & scale -------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"[OK] Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")

X = df[FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
n = len(X_scaled)

# ===============================================================================
# 1. K-Means  (full dataset)
# ===============================================================================
print("\n[1/3] Fitting K-Means ...")
km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
km_labels = km.fit_predict(X_scaled)
print(f"      Unique clusters: {len(set(km_labels))}")

# ===============================================================================
# 2. Agglomerative Clustering  (sampled subset)
# ===============================================================================
print(f"\n[2/3] Fitting Agglomerative (sample={min(AGG_SAMPLE, n):,}) ...")
rng = np.random.default_rng(RANDOM_SEED)
agg_idx = rng.choice(n, size=min(AGG_SAMPLE, n), replace=False)
X_agg = X_scaled[agg_idx]

agg = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward")
agg_labels = agg.fit_predict(X_agg)
print(f"      Unique clusters: {len(set(agg_labels))}")

# ===============================================================================
# 3. DBSCAN  (full dataset, eps tuned via k-distance)
# ===============================================================================
print("\n[3/3] Fitting DBSCAN ...")

# --- K-distance plot to guide eps selection ----------------------------------
k_nn = 5
nbrs = NearestNeighbors(n_neighbors=k_nn, algorithm="ball_tree")
nbrs.fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
kth_dist = np.sort(distances[:, -1])[::-1]   # sorted descending

# Knee heuristic: find the point of maximum curvature in kth_dist
diff1 = np.diff(kth_dist)
diff2 = np.diff(diff1)
knee_idx  = int(np.argmin(diff2)) + 2         # +2 for double-diff offset
eps_auto  = float(kth_dist[knee_idx])
eps_auto  = round(max(eps_auto, 0.3), 2)      # floor at 0.3 to avoid micro-clusters
print(f"      Auto-detected eps ≈ {eps_auto}  (k-distance knee at index {knee_idx})")

db = DBSCAN(eps=eps_auto, min_samples=k_nn)
db_labels = db.fit_predict(X_scaled)
n_noise   = int((db_labels == -1).sum())
n_db_clust = len(set(db_labels) - {-1})
print(f"      Unique clusters: {n_db_clust}  |  noise points: {n_noise:,}")

# Exclude noise for metrics (use core points only)
core_mask   = db_labels != -1
X_db_core   = X_scaled[core_mask]
db_core_lbl = db_labels[core_mask]

# ===============================================================================
# Evaluation Metrics
# ===============================================================================
print("\n📐 Computing evaluation metrics ...")

def safe_metrics(X_eval, labels, name):
    """Compute the three metrics; return NaN if fewer than 2 clusters."""
    u = set(labels)
    if len(u) < 2:
        print(f"  [WARN]  {name}: only {len(u)} cluster(s) found -- metrics undefined")
        return float("nan"), float("nan"), float("nan")
    sil = silhouette_score(X_eval, labels,
                           sample_size=min(5000, len(X_eval)),
                           random_state=RANDOM_SEED)
    db_s = davies_bouldin_score(X_eval, labels)
    ch   = calinski_harabasz_score(X_eval, labels)
    return sil, db_s, ch

km_sil,  km_db,  km_ch  = safe_metrics(X_scaled, km_labels,   "KMeans")
agg_sil, agg_db, agg_ch = safe_metrics(X_agg,    agg_labels,  "Agglomerative")
db_sil,  db_db,  db_ch  = safe_metrics(X_db_core, db_core_lbl, "DBSCAN")

rows = [
    {
        "algorithm":          "KMeans",
        "n_clusters":         N_CLUSTERS,
        "silhouette":         round(km_sil,  4),
        "davies_bouldin":     round(km_db,   4),
        "calinski_harabasz":  round(km_ch,   2),
    },
    {
        "algorithm":          "Agglomerative",
        "n_clusters":         N_CLUSTERS,
        "silhouette":         round(agg_sil, 4) if not np.isnan(agg_sil) else float("nan"),
        "davies_bouldin":     round(agg_db,  4) if not np.isnan(agg_db)  else float("nan"),
        "calinski_harabasz":  round(agg_ch,  2) if not np.isnan(agg_ch)  else float("nan"),
    },
    {
        "algorithm":          "DBSCAN",
        "n_clusters":         n_db_clust,
        "silhouette":         round(db_sil,  4) if not np.isnan(db_sil)  else float("nan"),
        "davies_bouldin":     round(db_db,   4) if not np.isnan(db_db)   else float("nan"),
        "calinski_harabasz":  round(db_ch,   2) if not np.isnan(db_ch)   else float("nan"),
    },
]

metrics_df = pd.DataFrame(rows, columns=["algorithm", "n_clusters", "silhouette",
                                          "davies_bouldin", "calinski_harabasz"])
m_path = os.path.join(OUTPUTS_DIR, "metrics.csv")
metrics_df.to_csv(m_path, index=False)
print(f"\n[SAVE] Saved → {m_path}")
print(metrics_df.to_string(index=False))

# ===============================================================================
# Figure 1: Dendrogram (Hierarchical)
# ===============================================================================
print("\n[PLOT] Generating dendrogram ...")
dend_idx  = rng.choice(len(X_agg), size=min(DENDRO_SAMPLE, len(X_agg)), replace=False)
X_dend    = X_agg[dend_idx]
Z         = linkage(X_dend, method="ward")

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#1a1d27")

dendrogram(Z, ax=ax, color_threshold=Z[-N_CLUSTERS, 2],
           above_threshold_color="#8b949e",
           no_labels=True,
           link_color_func=lambda k: (
               "#4fc3f7" if k <= N_CLUSTERS * 2
               else "#ff6f6f" if k <= N_CLUSTERS * 4
               else "#69f0ae" if k <= N_CLUSTERS * 6
               else "#ffcc02"
           ))

# Draw horizontal cut line
cut_height = (Z[-N_CLUSTERS, 2] + Z[-(N_CLUSTERS - 1), 2]) / 2
ax.axhline(cut_height, color="#ff6f6f", linestyle="--", linewidth=1.5,
           label=f"Cut for k={N_CLUSTERS} clusters")

ax.set_xlabel(f"Restaurant Index  (sub-sample of {len(X_dend):,})",
              color="#c9d1d9", fontsize=11)
ax.set_ylabel("Ward Linkage Distance", color="#c9d1d9", fontsize=11)
ax.set_title("Hierarchical Clustering Dendrogram  (Ward Linkage)",
             color="#e6edf3", fontsize=14, fontweight="bold", pad=12)

ax.tick_params(colors="#8b949e")
for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")
ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9",
          fontsize=10)

plt.tight_layout()
dend_path = os.path.join(FIGURES_DIR, "dendrogram.png")
plt.savefig(dend_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"[PLOT] Saved → {dend_path}")

# ===============================================================================
# Figure 2: k-Distance Plot (DBSCAN eps guide)
# ===============================================================================
print("[PLOT] Generating k-distance plot ...")
# Sub-sample for visual clarity
plot_n    = min(3000, n)
plot_idx  = rng.choice(n, size=plot_n, replace=False)
nbrs2     = NearestNeighbors(n_neighbors=k_nn, algorithm="ball_tree")
nbrs2.fit(X_scaled[plot_idx])
dist2, _  = nbrs2.kneighbors(X_scaled[plot_idx])
kd2       = np.sort(dist2[:, -1])[::-1]

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#1a1d27")

ax.plot(range(len(kd2)), kd2, color="#4fc3f7", linewidth=1.8, alpha=0.9)
ax.axhline(eps_auto, color="#ff6f6f", linestyle="--", linewidth=1.8,
           label=f"Selected eps = {eps_auto}")

ax.set_xlabel(f"Points sorted by {k_nn}-NN distance  (descending)",
              color="#c9d1d9", fontsize=11)
ax.set_ylabel(f"{k_nn}-NN Distance", color="#c9d1d9", fontsize=11)
ax.set_title(f"DBSCAN k-Distance Plot  (k={k_nn}) -- Knee ≈ eps={eps_auto}",
             color="#e6edf3", fontsize=14, fontweight="bold", pad=12)

ax.tick_params(colors="#8b949e")
for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")
ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9",
          fontsize=11)
ax.yaxis.grid(True, color="#21262d", linewidth=0.8, zorder=0)

plt.tight_layout()
kd_path = os.path.join(FIGURES_DIR, "kdistance.png")
plt.savefig(kd_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"[PLOT] Saved → {kd_path}")

print("\n[OK] compare_algorithms.py complete.")
