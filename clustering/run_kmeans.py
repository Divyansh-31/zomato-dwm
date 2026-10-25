"""
clustering/run_kmeans.py -- Final K-Means Fit & Profile Generation
=================================================================
Fits K-Means with the chosen k=4, assigns human-readable segment labels,
exports the two contract CSVs, and saves the PCA scatter figure.

Features used: rate, log_votes, cost  (StandardScaler-normalised)

Outputs
-------
outputs/clustered_restaurants.csv  -- restaurant_id, cluster_id
outputs/cluster_profiles.csv       -- cluster_id, label, size, avg_rate,
                                     avg_cost, avg_votes, top_cuisines
figures/clusters_pca.png           -- 2D PCA scatter coloured by cluster
"""

import os
import sys
import subprocess

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -- Constants ----------------------------------------------------------------
DATA_PATH    = "data/processed/zomato_clean.csv"
FEATURES     = ["rate", "log_votes", "cost"]
N_CLUSTERS   = 4
RANDOM_SEED  = 42
OUTPUTS_DIR  = "outputs"
FIGURES_DIR  = "figures"

# -- Bootstrap: fall back to mock data if real CSV is absent ------------------
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

# -- Fit K-Means ---------------------------------------------------------------
km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
df["cluster_id"] = km.fit_predict(X_scaled)
print(f"[OK] K-Means fitted  (k={N_CLUSTERS}, inertia={km.inertia_:.1f})")

# -- Assign human-readable labels ---------------------------------------------
# Sort cluster IDs by their centroid's *cost* (low → high) so labels are
# deterministic regardless of KMeans initialisation order.
centroids_orig = scaler.inverse_transform(km.cluster_centers_)   # shape (k, 3)
# centroids_orig columns: [rate, log_votes, cost]
cost_col = 2
sorted_by_cost = np.argsort(centroids_orig[:, cost_col])          # ascending cost

LABEL_NAMES = [
    "Everyday Low-Cost",   # cheapest centroid
    "Budget Hidden Gems",  # 2nd cheapest -- typically higher rated
    "Budget, Popular",     # 3rd -- moderate cost, many votes
    "Premium Fine Dining", # most expensive centroid
]

# Map original cluster id → sorted label index
label_map = {int(orig_id): LABEL_NAMES[rank]
             for rank, orig_id in enumerate(sorted_by_cost)}
df["label"] = df["cluster_id"].map(label_map)

print("\n[INFO] Cluster label assignment (by ascending avg cost):")
for orig_id, name in label_map.items():
    c = centroids_orig[orig_id]
    print(f"   cluster {orig_id} → '{name}'  "
          f"(avg_rate={c[0]:.2f}, avg_cost=₹{c[2]:.0f})")

# -- Export: clustered_restaurants.csv ----------------------------------------
out_cr = df[["restaurant_id", "cluster_id"]]
cr_path = os.path.join(OUTPUTS_DIR, "clustered_restaurants.csv")
out_cr.to_csv(cr_path, index=False)
print(f"\n[SAVE] Saved → {cr_path}  ({len(out_cr):,} rows)")

# -- Export: cluster_profiles.csv ---------------------------------------------
profiles = []
for cid in sorted(df["cluster_id"].unique()):
    sub = df[df["cluster_id"] == cid]
    top_c = (sub["primary_cuisine"].value_counts().head(3).index.tolist()
             if "primary_cuisine" in df.columns else [])
    profiles.append({
        "cluster_id":  cid,
        "label":       label_map[cid],
        "size":        len(sub),
        "avg_rate":    round(float(sub["rate"].mean()), 3),
        "avg_cost":    round(float(sub["cost"].mean()), 2),
        "avg_votes":   round(float(sub["votes"].mean()), 1),
        "top_cuisines": ", ".join(top_c) if top_c else "N/A",
    })

cp_df = pd.DataFrame(profiles,
                     columns=["cluster_id", "label", "size", "avg_rate",
                               "avg_cost", "avg_votes", "top_cuisines"])
cp_path = os.path.join(OUTPUTS_DIR, "cluster_profiles.csv")
cp_df.to_csv(cp_path, index=False)
print(f"[SAVE] Saved → {cp_path}")
print(cp_df.to_string(index=False))

# -- PCA scatter figure --------------------------------------------------------
pca = PCA(n_components=2, random_state=RANDOM_SEED)
coords = pca.fit_transform(X_scaled)
ev = pca.explained_variance_ratio_

PALETTE = ["#4fc3f7", "#ff6f6f", "#69f0ae", "#ffcc02"]

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#1a1d27")

for cid in sorted(df["cluster_id"].unique()):
    mask  = df["cluster_id"].values == cid
    color = PALETTE[cid % len(PALETTE)]
    ax.scatter(coords[mask, 0], coords[mask, 1],
               s=18, alpha=0.55, color=color,
               label=f"C{cid}: {label_map[cid]}", linewidths=0)

# Plot centroids projected into PCA space
centroid_pca = pca.transform(km.cluster_centers_)
ax.scatter(centroid_pca[:, 0], centroid_pca[:, 1],
           s=220, marker="*", color="#ffffff", zorder=5,
           edgecolors="#0f1117", linewidths=0.8, label="Centroids")

ax.set_xlabel(f"PC 1  ({ev[0]*100:.1f}% variance explained)",
              color="#c9d1d9", fontsize=11)
ax.set_ylabel(f"PC 2  ({ev[1]*100:.1f}% variance explained)",
              color="#c9d1d9", fontsize=11)
ax.set_title("K-Means Clusters -- 2D PCA Projection\n"
             "(features: rate · log_votes · cost)",
             color="#e6edf3", fontsize=14, fontweight="bold", pad=12)

ax.tick_params(colors="#8b949e")
for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")

ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9",
          fontsize=10, markerscale=1.6, loc="best")

plt.tight_layout()
pca_path = os.path.join(FIGURES_DIR, "clusters_pca.png")
plt.savefig(pca_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\n[PLOT] Saved → {pca_path}")

print("\n[OK] run_kmeans.py complete.")
