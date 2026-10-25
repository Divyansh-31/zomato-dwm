"""
run_pipeline.py — Master Orchestrator for Zomato DWM Mini Project
==================================================================
Coordinates the complete end-to-end pipeline:
  • Phase 1 (Person A): ETL, Star Schema Data Warehouse, Core OLAP
  • Phase 2 (Person B): Optimal K, K-Means fitting, Algorithm benchmarks, Stability
  • Phase 3 (Integration): Warehouse cluster ingestion & post-clustering OLAP
  • Verification: Star schema integrity checks & output asset audit
"""

import os
import sys
import time
import sqlite3
import subprocess
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)

def run_step(step_num, title, script_relpath):
    print(f"\n[{step_num}] Running {title} ({script_relpath}) ...")
    start = time.time()
    cmd = [sys.executable, "-X", "utf8", os.path.join(ROOT_DIR, script_relpath)]
    result = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace")
    duration = time.time() - start

    if result.returncode != 0:
        print(f"  [ERROR] {title} failed with exit code {result.returncode}!")
        print("  --- STDOUT ---")
        print(result.stdout[-1500:] if len(result.stdout) > 1500 else result.stdout)
        print("  --- STDERR ---")
        print(result.stderr[-1500:] if len(result.stderr) > 1500 else result.stderr)
        sys.exit(result.returncode)
    else:
        print(f"  [OK] Completed in {duration:.2f}s")
        # Print a short preview of the script's output
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        if lines:
            preview = lines[-3:]
            for l in preview:
                print(f"       {l}")

def main():
    print_header("Zomato DWM Mini Project: End-to-End Pipeline & Integration")
    overall_start = time.time()

    raw_path = os.path.join(ROOT_DIR, "data", "raw", "zomato.csv")
    proc_path = os.path.join(ROOT_DIR, "data", "processed", "zomato_clean.csv")

    # -------------------------------------------------------------------------
    # STEP 0: Data preparation
    # -------------------------------------------------------------------------
    if os.path.exists(raw_path):
        print(f"[INFO] Raw dataset detected at {raw_path}. Running ETL pipeline.")
        run_step("0/7", "ETL Cleaning", "etl/clean.py")
    elif os.path.exists(proc_path):
        print(f"[INFO] Processed dataset found at {proc_path}. Proceeding.")
    else:
        print("[INFO] No data file found. Bootstrapping mock data contract.")
        run_step("0/7", "Mock Data Generator", "clustering/make_mock.py")

    # -------------------------------------------------------------------------
    # STEP 1: Phase 1 — Star Schema Data Warehouse
    # -------------------------------------------------------------------------
    run_step("1/7", "Star Schema Warehouse Builder", "warehouse/build_dw.py")

    # -------------------------------------------------------------------------
    # STEP 2: Phase 1 — Pre-clustering OLAP
    # -------------------------------------------------------------------------
    run_step("2/7", "Core OLAP Queries", "olap/run_olap.py")

    # -------------------------------------------------------------------------
    # STEPS 3–6: Phase 2 — Person B Clustering Pipeline
    # -------------------------------------------------------------------------
    run_step("3/7", "Elbow & Silhouette Selection", "clustering/choose_k.py")
    run_step("4/7", "K-Means Fitting & Profile Generation", "clustering/run_kmeans.py")
    run_step("5/7", "Three-Algorithm Benchmarking", "clustering/compare_algorithms.py")
    run_step("6/7", "K-Means Stability Assessment", "clustering/stability.py")

    # -------------------------------------------------------------------------
    # STEP 7: Phase 3 — Integration (Handoff to Warehouse)
    # -------------------------------------------------------------------------
    run_step("7/7", "Cluster Loading & Post-Cluster OLAP", "warehouse/load_clusters.py")

    # -------------------------------------------------------------------------
    # INTEGRATION INTEGRITY AUDIT
    # -------------------------------------------------------------------------
    print_header("Integration Integrity & Quality Audit")

    db_path = os.path.join(ROOT_DIR, "warehouse", "zomato_dw.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. Table Counts
    print("\n[DB Audit] Table Record Counts:")
    tables = ["fact_restaurant", "dim_location", "dim_cuisine", "dim_type", "dim_cluster"]
    for tbl in tables:
        count = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  • {tbl:20s}: {count:,} rows")

    # 2. Cluster FK Coverage
    unassigned = cur.execute("SELECT COUNT(*) FROM fact_restaurant WHERE cluster_id IS NULL").fetchone()[0]
    total_facts = cur.execute("SELECT COUNT(*) FROM fact_restaurant").fetchone()[0]
    assigned = total_facts - unassigned
    print(f"\n[DB Audit] Fact Table Cluster Assignments: {assigned:,} / {total_facts:,} ({100 * assigned / total_facts:.1f}%)")
    assert unassigned == 0, f"Integrity Failure: {unassigned} restaurants have no cluster assignment!"

    # 3. Foreign Key Integrity Check
    cur.execute("PRAGMA foreign_keys = ON;")
    violations = cur.execute("PRAGMA foreign_key_check;").fetchall()
    print(f"[DB Audit] Foreign Key Violations: {len(violations)}")
    assert len(violations) == 0, f"Foreign Key Violations Found: {violations}"

    conn.close()

    # 4. File Deliverables Audit
    print("\n[Artifact Audit] Checking Deliverables:")
    expected_olap_csvs = [
        "rollup_city.csv", "drilldown_location.csv", "slice_casual_dining.csv",
        "dice_cost_rate.csv", "pivot_category_city.csv", "top10_cuisines.csv",
        "cluster_city_counts.csv", "cluster_category_counts.csv",
        "cluster_location_avg_cost.csv", "underserved_locations.csv"
    ]
    all_olap_ok = True
    for f in expected_olap_csvs:
        p = os.path.join(ROOT_DIR, "outputs", "olap", f)
        if not os.path.exists(p):
            print(f"  [MISSING] outputs/olap/{f}")
            all_olap_ok = False
    if all_olap_ok:
        print(f"  [OK] All {len(expected_olap_csvs)} OLAP CSVs present in outputs/olap/")

    expected_cluster_csvs = [
        "clustered_restaurants.csv", "cluster_profiles.csv",
        "metrics.csv", "stability.csv"
    ]
    all_cl_ok = True
    for f in expected_cluster_csvs:
        p = os.path.join(ROOT_DIR, "outputs", f)
        if not os.path.exists(p):
            print(f"  [MISSING] outputs/{f}")
            all_cl_ok = False
    if all_cl_ok:
        print(f"  [OK] All {len(expected_cluster_csvs)} Clustering CSVs present in outputs/")

    expected_figures = [
        "olap_rollup.png", "olap_cluster_distribution.png",
        "elbow.png", "silhouette_by_k.png", "clusters_pca.png",
        "dendrogram.png", "kdistance.png"
    ]
    all_fig_ok = True
    for f in expected_figures:
        p = os.path.join(ROOT_DIR, "figures", f)
        if not os.path.exists(p):
            print(f"  [MISSING] figures/{f}")
            all_fig_ok = False
    if all_fig_ok:
        print(f"  [OK] All {len(expected_figures)} Visual Figures present in figures/")

    # 5. Display Profiles & Metrics Summary
    print_header("Executive Summary: Cluster Profiles & Benchmarks")
    profiles_df = pd.read_csv(os.path.join(ROOT_DIR, "outputs", "cluster_profiles.csv"))
    print("\n--- Cluster Personas (outputs/cluster_profiles.csv) ---")
    print(profiles_df.to_string(index=False))

    metrics_df = pd.read_csv(os.path.join(ROOT_DIR, "outputs", "metrics.csv"))
    print("\n--- Algorithm Benchmark Metrics (outputs/metrics.csv) ---")
    print(metrics_df.to_string(index=False))

    stability_df = pd.read_csv(os.path.join(ROOT_DIR, "outputs", "stability.csv"))
    mean_ari = stability_df["ari"].mean()
    print(f"\n--- Model Stability: Mean ARI = {mean_ari:.4f} across 10 random seeds ---")

    underserved_df = pd.read_csv(os.path.join(ROOT_DIR, "outputs", "olap", "underserved_locations.csv"))
    print("\n--- Top 5 Underserved Locations for High-Quality Budget Dining ---")
    print(underserved_df.head(5).to_string(index=False))

    total_time = time.time() - overall_start
    print_header(f"All Steps Completed Successfully in {total_time:.2f}s!")

if __name__ == "__main__":
    main()
