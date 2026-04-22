"""
Baseline experiment — replicates exactly the Kechid et al. (2016) protocol.

Datasets:
  - Breast Cancer Wisconsin: k=2  (paper Figures 1-3)
  - COIL-100 images: n=720→5000, k=n//72  (paper Figures 4-8)

Metrics: intra-class inertia, inter-class inertia, NMI, runtime.
Baselines: k-means, PAM (k-medoids), k-MM.

Usage:
    python experiments/run_baseline.py
    python experiments/run_baseline.py --coil-dir /path/to/coil-100-parent
    python experiments/run_baseline.py --quick
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from algorithms import KMM, KMMPlusPlus, KMeans, KMeansPlusPlus, KMedoids
from evaluation.metrics import nmi_score
from data.loaders import load_breast_cancer_data, load_coil100_data

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def intra_inertia(X, labels, centers):
    return float(sum(
        np.sum(np.linalg.norm(X[labels == c] - centers[c], axis=1) ** 2)
        for c in range(len(centers)) if (labels == c).any()
    ))


def inter_inertia(X, labels, centers):
    mu = X.mean(axis=0)
    return float(sum(
        (labels == c).sum() * np.linalg.norm(centers[c] - mu) ** 2
        for c in range(len(centers))
    ))


def run_once(AlgoClass, X, y, k, seed):
    algo = AlgoClass(k=k, max_iter=100, random_state=seed)
    t0 = time.perf_counter()
    algo.fit(X)
    elapsed = time.perf_counter() - t0
    return {
        "intra_inertia": intra_inertia(X, algo.labels_, algo.centers_),
        "inter_inertia": inter_inertia(X, algo.labels_, algo.centers_),
        "nmi":           nmi_score(y, algo.labels_),
        "runtime_s":     elapsed,
        "n_iter":        algo.n_iter_,
    }


def run_experiment(name, X, y, k, n_runs, algorithms):
    rows = []
    for algo_name, AlgoClass in algorithms.items():
        print(f"    {algo_name} × {n_runs} seeds ...", end=" ", flush=True)
        for seed in range(n_runs):
            try:
                r = run_once(AlgoClass, X, y, k, seed)
                r.update({"algorithm": algo_name, "dataset": name,
                          "k": k, "n": len(X), "seed": seed})
                rows.append(r)
            except Exception as e:
                print(f"\n    [SKIP] {algo_name}: {e}")
                break
        print("ok")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coil-dir", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    n_runs = 3 if args.quick else args.runs
    algos = {
    "KMeans":      KMeans,
    "KMeans++":    KMeansPlusPlus,
    "PAM":         KMedoids,
    "KMM":         KMM,          # paper exact
    "KMM++":       KMMPlusPlus,  # your improvement
}

    all_dfs = []

    # ── Experiment 1: Breast Cancer (paper Figs 1-3) ─────────────────────────
    print("\n" + "="*52)
    print("EXP 1 — Breast Cancer Wisconsin  k=2")
    print("="*52)
    X_bc, y_bc = load_breast_cancer_data()
    df1 = run_experiment("breast_cancer", X_bc, y_bc, k=2, n_runs=n_runs, algorithms=algos)
    df1.to_csv(os.path.join(RESULTS_DIR, "baseline_breast_cancer.csv"), index=False)
    print(df1.groupby("algorithm")[["intra_inertia","inter_inertia","nmi","runtime_s"]].mean().round(4))
    all_dfs.append(df1)

    # ── Experiment 2: COIL-100 (paper Figs 4-8) ──────────────────────────────
    print("\n" + "="*52)
    print("EXP 2 — COIL-100  (varying n)")
    print("="*52)
    image_counts = [720, 1440] if args.quick else [720, 1440, 2160, 2880, 3600, 4320, 5000]
    coil_dfs = []

    for n_img in image_counts:
        k = max(2, n_img // 72)
        print(f"\n  n_images={n_img}  k={k}")
        X_c, y_c = load_coil100_data(max_images=n_img, data_dir=args.coil_dir)
        df_c = run_experiment(f"coil100_n{n_img}", X_c, y_c, k=k,
                              n_runs=n_runs, algorithms=algos)
        df_c["n_images_target"] = n_img
        coil_dfs.append(df_c)

    df2 = pd.concat(coil_dfs, ignore_index=True)
    df2.to_csv(os.path.join(RESULTS_DIR, "baseline_coil100.csv"), index=False)
    print("\nCOIL-100 summary (mean per algorithm × n):")
    print(df2.groupby(["algorithm","n"])[["intra_inertia","nmi","runtime_s"]].mean().round(4))
    all_dfs.append(df2)

    # ── Save combined ─────────────────────────────────────────────────────────
    pd.concat(all_dfs, ignore_index=True).to_csv(
        os.path.join(RESULTS_DIR, "baseline_all.csv"), index=False
    )
    print(f"\n Results saved to {os.path.abspath(RESULTS_DIR)}/")
    print("  baseline_breast_cancer.csv  ← replicate paper Figs 1-3")
    print("  baseline_coil100.csv        ← replicate paper Figs 4-8")
    print("  baseline_all.csv            ← combined (compare your hybrid here)")


if __name__ == "__main__":
    main()
