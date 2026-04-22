"""
Benchmark runner.
Runs any BaseClusterer multiple times on a dataset, collects metrics,
and returns a tidy summary DataFrame.

Usage:
    from evaluation.benchmark import benchmark_algorithm, compare_algorithms

    results = compare_algorithms(
        algorithms={"KMM": KMM, "HybridA": HybridA},
        datasets={"iris": (X_iris, y_iris)},
        k_values=[3],
        n_runs=10,
    )
    print(results)
"""
import numpy as np
import pandas as pd
import time
from typing import Type, Optional

from algorithms.base import BaseClusterer
from evaluation.metrics import evaluate


def benchmark_algorithm(
    AlgorithmClass: Type[BaseClusterer],
    X: np.ndarray,
    k: int,
    labels_true: Optional[np.ndarray] = None,
    n_runs: int = 10,
    max_iter: int = 100,
    compute_silhouette: bool = True,
) -> pd.DataFrame:
    """
    Run an algorithm n_runs times with different random seeds.
    Returns a DataFrame where each row is one run.
    """
    rows = []
    for run in range(n_runs):
        algo = AlgorithmClass(k=k, max_iter=max_iter, random_state=run)
        algo.fit_timed(X)

        metrics = evaluate(
            X,
            algo.labels_,
            algo.centers_,
            labels_true=labels_true,
            compute_silhouette=compute_silhouette,
        )
        metrics["fit_time_s"] = algo.fit_time_
        metrics["n_iter"]     = algo.n_iter_
        metrics["run"]        = run
        metrics["algorithm"]  = AlgorithmClass.__name__
        metrics["k"]          = k
        metrics["n_samples"]  = len(X)
        rows.append(metrics)

    return pd.DataFrame(rows)


def compare_algorithms(
    algorithms: dict[str, Type[BaseClusterer]],
    datasets: dict[str, tuple],
    k_values: list[int],
    n_runs: int = 10,
    max_iter: int = 100,
    compute_silhouette: bool = True,
) -> pd.DataFrame:
    """
    Full comparison table.

    Parameters
    ----------
    algorithms : {"AlgoName": AlgoClass, ...}
    datasets   : {"dataset_name": (X, y) or (X, None), ...}
    k_values   : list of k to try
    n_runs     : how many seeds per (algo, dataset, k) combination
    """
    all_dfs = []

    for ds_name, (X, y) in datasets.items():
        for k in k_values:
            for algo_name, AlgoClass in algorithms.items():
                print(f"  [{ds_name}] k={k}  {algo_name} × {n_runs} runs ...")
                df = benchmark_algorithm(
                    AlgoClass, X, k,
                    labels_true=y,
                    n_runs=n_runs,
                    max_iter=max_iter,
                    compute_silhouette=compute_silhouette,
                )
                df["dataset"] = ds_name
                all_dfs.append(df)

    full = pd.concat(all_dfs, ignore_index=True)
    return full


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate mean ± std per (algorithm, dataset, k).
    Useful for the final comparison table in the paper.
    """
    metric_cols = ["inertia", "silhouette", "davies_bouldin",
                   "calinski_harabasz", "nmi", "ari", "purity", "fit_time_s"]
    present = [c for c in metric_cols if c in df.columns]

    agg = df.groupby(["algorithm", "dataset", "k"])[present].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()
