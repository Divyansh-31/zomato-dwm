"""
clustering/choose_k.py -- Elbow Method & Silhouette Analysis
============================================================
Sweeps k from 2 to 10, computes inertia (elbow) and silhouette score,
and saves two diagnostic figures to help choose the optimal number of clusters.

Features used: rate, log_votes, cost  (all StandardScaler-normalised)

Outputs
-------
figures/elbow.png           -- Inertia elbow curve
figures/silhouette_by_k.png -- Silhouette score vs k
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
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# -- Constants ----------------------------------------------------------------
DATA_PATH   = "data/processed/zomato_clean.csv"
FEATURES    = ["rate", "log_votes", "cost"]
K_RANGE     = range(2, 11)
FIGURES_DIR = "figures"
RANDOM_SEED = 42

# -- Bootstrap: fall back to mock data if real CSV is absent ------------------
if not os.path.exists(DATA_PATH):
    print("[WARN]  Real data not found -- generating mock data ...")
    subprocess.run([sys.executable, "clustering/make_mock.py"], check=True)

os.makedirs(FIGURES_DIR, exist_ok=True)

# -- Load & scale -------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"[OK] Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")

X = df[FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -- Sweep k ------------------------------------------------------------------
inertias    = []
silhouettes = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)),
                           random_state=RANDOM_SEED)
    silhouettes.append(sil)
    print(f"  k={k:2d}  inertia={km.inertia_:10.1f}  silhouette={sil:.4f}")

best_k = list(K_RANGE)[silhouettes.index(max(silhouettes))]
print(f"\n[BEST] Recommended k = {best_k}  (peak silhouette = {max(silhouettes):.4f})")

# -- Figure 1: Elbow Curve -----------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#1a1d27")

ks = list(K_RANGE)
ax.plot(ks, inertias, "o-", color="#4fc3f7", linewidth=2.5, markersize=7,
        markerfacecolor="#ffffff", markeredgecolor="#4fc3f7", markeredgewidth=1.5)

ax.axvline(best_k, color="#ff6f6f", linestyle="--", linewidth=1.5,
           label=f"Recommended k = {best_k}")

ax.set_xlabel("Number of Clusters  k", color="#c9d1d9", fontsize=12)
ax.set_ylabel("Inertia  (within-cluster SSE)", color="#c9d1d9", fontsize=12)
ax.set_title("Elbow Method -- K-Means Inertia vs k", color="#e6edf3",
             fontsize=14, fontweight="bold", pad=12)

ax.tick_params(colors="#8b949e")
for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")

ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9",
          fontsize=11)
ax.set_xticks(ks)

plt.tight_layout()
elbow_path = os.path.join(FIGURES_DIR, "elbow.png")
plt.savefig(elbow_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"[PLOT] Saved → {elbow_path}")

# -- Figure 2: Silhouette Score vs k ------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#1a1d27")

bar_colors = ["#ff6f6f" if k == best_k else "#4fc3f7" for k in ks]
bars = ax.bar(ks, silhouettes, color=bar_colors, edgecolor="#0f1117",
              linewidth=0.8, zorder=3)

for bar, val in zip(bars, silhouettes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{val:.3f}", ha="center", va="bottom", color="#c9d1d9", fontsize=9)

ax.set_xlabel("Number of Clusters  k", color="#c9d1d9", fontsize=12)
ax.set_ylabel("Silhouette Score", color="#c9d1d9", fontsize=12)
ax.set_title("Silhouette Score vs k  (higher = better separation)",
             color="#e6edf3", fontsize=14, fontweight="bold", pad=12)

ax.tick_params(colors="#8b949e")
for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")
ax.set_xticks(ks)
ax.set_ylim(0, max(silhouettes) * 1.15)
ax.yaxis.grid(True, color="#21262d", linewidth=0.8, zorder=0)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#ff6f6f", label=f"Best k = {best_k}"),
    Patch(facecolor="#4fc3f7", label="Other k values"),
]
ax.legend(handles=legend_elements, facecolor="#21262d", edgecolor="#30363d",
          labelcolor="#c9d1d9", fontsize=11)

plt.tight_layout()
sil_path = os.path.join(FIGURES_DIR, "silhouette_by_k.png")
plt.savefig(sil_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"[PLOT] Saved → {sil_path}")

print("\n[OK] choose_k.py complete.")
