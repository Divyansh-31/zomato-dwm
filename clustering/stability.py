"""
clustering/stability.py -- K-Means Stability Assessment (ARI)
=============================================================
Evaluates how stable K-Means cluster assignments are across different random
seeds by computing the Adjusted Rand Index (ARI) between a reference model
(seed=42) and 10 alternative seeds.

ARI = 1.0  →  perfect agreement (fully stable)
ARI ≥ 0.85 →  highly stable (publishable quality)
ARI < 0.50 →  unstable (consider changing k or feature set)

Outputs
-------
outputs/stability.csv -- seed, ari  (one row per alternative seed)
"""

import os
import sys
import subprocess

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# -- Constants ----------------------------------------------------------------
DATA_PATH   = "data/processed/zomato_clean.csv"
FEATURES    = ["rate", "log_votes", "cost"]
N_CLUSTERS  = 4
REF_SEED    = 42
ALT_SEEDS   = [0, 1, 7, 13, 21, 37, 55, 77, 88, 99]
OUTPUTS_DIR = "outputs"

# -- Bootstrap -----------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    print("[WARN]  Real data not found -- generating mock data ...")
    subprocess.run([sys.executable, "clustering/make_mock.py"], check=True)

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# -- Load & scale -------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"[OK] Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")

X = df[FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -- Reference model (seed=42) -------------------------------------------------
ref_km = KMeans(n_clusters=N_CLUSTERS, random_state=REF_SEED, n_init=10)
ref_labels = ref_km.fit_predict(X_scaled)
print(f"\n[KEY] Reference model fitted (seed={REF_SEED}, "
      f"inertia={ref_km.inertia_:.1f})")

# -- ARI across alternative seeds ---------------------------------------------
print(f"\n{'Seed':>6}  {'ARI':>8}  Interpretation")
print("-" * 40)

rows = []
for seed in ALT_SEEDS:
    km = KMeans(n_clusters=N_CLUSTERS, random_state=seed, n_init=10)
    labels = km.fit_predict(X_scaled)
    ari = adjusted_rand_score(ref_labels, labels)
    rows.append({"seed": seed, "ari": round(ari, 4)})

    if ari >= 0.95:
        interp = "[GREEN]Excellent"
    elif ari >= 0.85:
        interp = "[YELLOW]Good"
    elif ari >= 0.70:
        interp = "[ORANGE]Moderate"
    else:
        interp = "[RED]Unstable"

    print(f"{seed:>6}  {ari:>8.4f}  {interp}")

# -- Export CSV ----------------------------------------------------------------
stability_df = pd.DataFrame(rows, columns=["seed", "ari"])
s_path = os.path.join(OUTPUTS_DIR, "stability.csv")
stability_df.to_csv(s_path, index=False)
print(f"\n[SAVE] Saved → {s_path}")

# -- Summary statistics --------------------------------------------------------
aris = stability_df["ari"].values
mean_ari = aris.mean()
std_ari  = aris.std()
min_ari  = aris.min()

print("\n" + "=" * 45)
print(f"  Stability Summary  (k={N_CLUSTERS}, {len(ALT_SEEDS)} seeds)")
print("-" * 45)
print(f"  Mean ARI : {mean_ari:.4f}")
print(f"  Std  ARI : {std_ari:.4f}")
print(f"  Min  ARI : {min_ari:.4f}")
print("-" * 45)

if mean_ari >= 0.85:
    verdict = "[OK] STABLE -- clusters are highly reproducible across seeds."
elif mean_ari >= 0.70:
    verdict = "[WARN]  MODERATELY STABLE -- minor variation across seeds."
else:
    verdict = "[FAIL] UNSTABLE -- consider changing k or feature engineering."

print(f"  Verdict  : {verdict}")
print("=" * 45)

print("\n[OK] stability.py complete.")
