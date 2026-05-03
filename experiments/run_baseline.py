"""
Baseline experiment — centralized + distributed comparison.

Datasets:
  - Iris:             k=3   (sanity check)
  - Breast Cancer:    k=2   (paper Figs 1-3)
  - COIL-100:         k=n//72, varying n  (paper Figs 4-8)

Modes run back-to-back for every dataset:
  - Centralized:  KMeans, KMeans++, PAM, KMM, KMM++
  - Distributed:  same algorithms, 4-worker thread model

Output (results/):
  baseline_iris.csv
  baseline_breast_cancer.csv
  baseline_coil100.csv
  baseline_all.csv
  comparison_report.txt   ← human-readable centralized vs distributed summary

Usage:
    python experiments/run_baseline.py
    python experiments/run_baseline.py --quick
    python experiments/run_baseline.py --skip-coil
    python experiments/run_baseline.py --iris-csv path/to/iris.csv
"""
import sys, os, argparse, time, textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from algorithms import (
    KMeans, KMeansPlusPlus, KMedoids, KMM, KMMPlusPlus,
    DistributedKMeans, DistributedKMeansPlusPlus,
    DistributedKMedoids, DistributedKMM, DistributedKMMPlusPlus,
    N_WORKERS,
)
from evaluation.metrics import (
    inertia as calc_inertia,
    nmi_score, silhouette_score,
    davies_bouldin_score, calinski_harabasz_score,
    ari_score, purity_score,
)
from data.loaders import load_breast_cancer_data, load_coil100_data, load_iris_data, load_mnist_data

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Paired so the report can compare them side by side
CENTRALIZED_ALGOS = {
    "KMeans":   KMeans,
    "KMeans++": KMeansPlusPlus,
    "PAM":      KMedoids,
    "KMM":      KMM,
    "KMM++":    KMMPlusPlus,
}
DISTRIBUTED_ALGOS = {
    "Dist-KMeans":   DistributedKMeans,
    "Dist-KMeans++": DistributedKMeansPlusPlus,
    "Dist-PAM":      DistributedKMedoids,
    "Dist-KMM":      DistributedKMM,
    "Dist-KMM++":    DistributedKMMPlusPlus,
}
# Map centralized name → distributed name for side-by-side report
PAIR_MAP = {
    "KMeans":   "Dist-KMeans",
    "KMeans++": "Dist-KMeans++",
    "PAM":      "Dist-PAM",
    "KMM":      "Dist-KMM",
    "KMM++":    "Dist-KMM++",
}

ALL_ALGOS = {**CENTRALIZED_ALGOS, **DISTRIBUTED_ALGOS}


# ─── Metric helpers ───────────────────────────────────────────────────────────

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

    lbl = algo.labels_
    ctr = algo.centers_

    result = {
        "intra_inertia":     calc_inertia(X, lbl, ctr),
        "inter_inertia":     inter_inertia(X, lbl, ctr),
        "silhouette":        silhouette_score(X, lbl),
        "davies_bouldin":    davies_bouldin_score(X, lbl, ctr),
        "calinski_harabasz": calinski_harabasz_score(X, lbl, ctr),
        "runtime_s":         elapsed,
        "n_iter":            algo.n_iter_,
    }
    if y is not None:
        result["nmi"]    = nmi_score(y, lbl)
        result["ari"]    = ari_score(y, lbl)
        result["purity"] = purity_score(y, lbl)

    return result


# ─── Experiment runner ────────────────────────────────────────────────────────

def run_experiment(name, X, y, k, n_runs, algorithms):
    rows = []
    for algo_name, AlgoClass in algorithms.items():
        mode = "distributed" if algo_name.startswith("Dist-") else "centralized"
        print(f"    [{mode}] {algo_name} × {n_runs} seeds ...", end=" ", flush=True)
        for seed in range(n_runs):
            try:
                r = run_once(AlgoClass, X, y, k, seed)
                r.update({
                    "algorithm": algo_name,
                    "mode":      mode,
                    "dataset":   name,
                    "k":         k,
                    "n":         len(X),
                    "seed":      seed,
                })
                rows.append(r)
            except Exception as e:
                print(f"\n    [SKIP] {algo_name} seed={seed}: {e}")
                break
        print("ok")
    return pd.DataFrame(rows)


# ─── Report generation ────────────────────────────────────────────────────────

METRIC_COLS = [
    "intra_inertia", "inter_inertia", "silhouette",
    "davies_bouldin", "calinski_harabasz",
    "nmi", "ari", "purity",
    "runtime_s", "n_iter",
]

def build_report(all_df: pd.DataFrame, n_workers: int) -> str:
    lines = []
    sep  = "=" * 72
    sep2 = "-" * 72

    lines.append(sep)
    lines.append("  CENTRALIZED vs DISTRIBUTED CLUSTERING — COMPARISON REPORT")
    lines.append(f"  Workers: {n_workers}  |  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)

    for dataset in all_df["dataset"].unique():
        df_ds = all_df[all_df["dataset"] == dataset]
        k_val = df_ds["k"].iloc[0]
        n_val = df_ds["n"].iloc[0]

        lines.append(f"\n{'─'*72}")
        lines.append(f"  DATASET : {dataset}   (n={n_val}, k={k_val})")
        lines.append(f"{'─'*72}")

        present_metrics = [m for m in METRIC_COLS if m in df_ds.columns]
        means = df_ds.groupby("algorithm")[present_metrics].mean()

        for cent_name, dist_name in PAIR_MAP.items():
            if cent_name not in means.index or dist_name not in means.index:
                continue

            c_row = means.loc[cent_name]
            d_row = means.loc[dist_name]

            lines.append(f"\n  {cent_name:12s}  vs  {dist_name}")
            lines.append(f"  {'Metric':<22} {'Centralized':>14} {'Distributed':>14} {'Delta':>12} {'Better':>8}")
            lines.append(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*12} {'-'*8}")

            for m in present_metrics:
                if m not in c_row or m not in d_row:
                    continue
                cv = c_row[m]
                dv = d_row[m]
                delta = dv - cv
                delta_pct = (delta / abs(cv) * 100) if cv != 0 else float("nan")

                # Higher-is-better metrics
                higher_better = {"silhouette", "calinski_harabasz", "nmi", "ari",
                                 "purity", "inter_inertia"}
                # Lower-is-better metrics
                lower_better  = {"intra_inertia", "davies_bouldin", "runtime_s", "n_iter"}

                if m in higher_better:
                    better = "CENT" if cv > dv else ("DIST" if dv > cv else "TIE")
                elif m in lower_better:
                    better = "CENT" if cv < dv else ("DIST" if dv < cv else "TIE")
                else:
                    better = "—"

                lines.append(
                    f"  {m:<22} {cv:>14.4f} {dv:>14.4f} "
                    f"{delta_pct:>+11.1f}% {better:>8}"
                )

        # Summary table: mean runtime and NMI for all algorithms on this dataset
        lines.append(f"\n  {'─'*68}")
        lines.append(f"  SUMMARY — all algorithms on {dataset}")
        lines.append(f"  {'─'*68}")
        summary_metrics = [m for m in ["runtime_s", "intra_inertia", "nmi", "silhouette"] if m in means.columns]
        header = f"  {'Algorithm':<18}" + "".join(f"{m:>16}" for m in summary_metrics)
        lines.append(header)
        lines.append(f"  {'-'*18}" + "".join(f"{'-'*16}" for _ in summary_metrics))
        for algo in means.index:
            row_str = f"  {algo:<18}" + "".join(f"{means.loc[algo, m]:>16.4f}" for m in summary_metrics)
            lines.append(row_str)

    lines.append(f"\n{sep}")
    lines.append("  END OF REPORT")
    lines.append(sep)
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coil-dir",  default=None)
    parser.add_argument("--iris-csv",  default=None)
    parser.add_argument("--mnist-csv", default=None,
                    help="Path to mnist_train.csv (default: data/mnist_train.csv)")
    parser.add_argument("--quick",     action="store_true",
                        help="3 runs and small COIL subset for fast testing")
    parser.add_argument("--skip-coil", action="store_true")
    parser.add_argument("--runs",      type=int, default=10)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    n_runs = 3 if args.quick else args.runs

    all_dfs = []

    # ── Experiment 0: Iris ────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("EXP 0 — Iris  k=3")
    print("=" * 52)
    try:
        X_iris, y_iris = load_iris_data(csv_path=args.iris_csv)
        print(f"Loaded: shape={X_iris.shape}")
        df0 = run_experiment("iris", X_iris, y_iris, k=3, n_runs=n_runs, algorithms=ALL_ALGOS)
        df0.to_csv(os.path.join(RESULTS_DIR, "baseline_iris.csv"), index=False)
        _print_summary(df0)
        all_dfs.append(df0)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")

    # ── Experiment 1: Breast Cancer ───────────────────────────────────────────
    print("\n" + "=" * 52)
    print("EXP 1 — Breast Cancer Wisconsin  k=2")
    print("=" * 52)
    X_bc, y_bc = load_breast_cancer_data()
    print(f"Loaded: shape={X_bc.shape}")
    df1 = run_experiment("breast_cancer", X_bc, y_bc, k=2, n_runs=n_runs, algorithms=ALL_ALGOS)
    df1.to_csv(os.path.join(RESULTS_DIR, "baseline_breast_cancer.csv"), index=False)
    _print_summary(df1)
    all_dfs.append(df1)
    
    
    # # ── Experiment 1.5: MNIST digits ─────────────────────────────────────────
    # print("\n" + "=" * 52)
    # print("EXP 1.5 — MNIST digits  k=10  (n=10000, PCA=50)")
    # print("=" * 52)
    # try:
    #     X_mnist, y_mnist = load_mnist_data(n_samples=10000, n_components=50,
    #                                         csv_path=args.mnist_csv)
    #     print(f"Loaded: shape={X_mnist.shape}")
    #     mnist_algos_cent = {
    #         "KMeans":   KMeans,
    #         "KMeans++": KMeansPlusPlus,
    #     }
    #     mnist_algos_dist = {
    #         "Dist-KMeans":   DistributedKMeans,
    #         "Dist-KMeans++": DistributedKMeansPlusPlus,
    #         "Dist-PAM":      DistributedKMedoids,
    #         "Dist-KMM":      DistributedKMM,
    #         "Dist-KMM++":    DistributedKMMPlusPlus,
    #     }
    #     mnist_algos = {**mnist_algos_cent, **mnist_algos_dist}
    #     df_mnist = run_experiment(
    #         "mnist", X_mnist, y_mnist, k=10,
    #         n_runs=n_runs, algorithms=mnist_algos,
    #     )
    #     df_mnist.to_csv(os.path.join(RESULTS_DIR, "baseline_mnist.csv"), index=False)
    #     _print_summary(df_mnist)
    #     all_dfs.append(df_mnist)
    # except FileNotFoundError as e:
    #     print(f"  [SKIP] {e}")
    

    # ── Experiment 2: COIL-100 ────────────────────────────────────────────────
    if not args.skip_coil:
        print("\n" + "=" * 52)
        print("EXP 2 — COIL-100  (varying n)")
        print("=" * 52)
        image_counts = (
            [720, 1440] if args.quick
            else [720, 1440, 2160, 2880, 3600, 4320, 5000]
        )
        coil_dfs = []
        for n_img in image_counts:
            k = max(2, n_img // 72)
            print(f"\n  n_images={n_img}  k={k}")
            X_c, y_c = load_coil100_data(max_images=n_img, data_dir=args.coil_dir)
            df_c = run_experiment(
                f"coil100_n{n_img}", X_c, y_c, k=k,
                n_runs=n_runs, algorithms=ALL_ALGOS,
            )
            df_c["n_images_target"] = n_img
            coil_dfs.append(df_c)

        df2 = pd.concat(coil_dfs, ignore_index=True)
        df2.to_csv(os.path.join(RESULTS_DIR, "baseline_coil100.csv"), index=False)
        _print_summary(df2)
        all_dfs.append(df2)
    else:
        print("\n[SKIP] COIL-100 (--skip-coil)")

    # ── Save combined + report ────────────────────────────────────────────────
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(os.path.join(RESULTS_DIR, "baseline_all.csv"), index=False)

        report = build_report(combined, N_WORKERS)
        report_path = os.path.join(RESULTS_DIR, "comparison_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print("\n" + report)
        print(f"\nFiles saved to {os.path.abspath(RESULTS_DIR)}/")
        print("  baseline_iris.csv")
        print("  baseline_breast_cancer.csv")
        if not args.skip_coil:
            print("  baseline_coil100.csv")
        print("  baseline_all.csv")
        print("  comparison_report.txt  ← main comparison")


def _print_summary(df: pd.DataFrame):
    present = [m for m in ["runtime_s", "intra_inertia", "nmi", "silhouette"] if m in df.columns]
    print(df.groupby(["mode", "algorithm"])[present].mean().round(4).to_string())


if __name__ == "__main__":
    main()