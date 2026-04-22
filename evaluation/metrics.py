"""
Evaluation metrics for clustering comparison.

All metrics used in the literature to compare k-MM against hybrid methods:
  - Internal (no ground truth needed): Inertia, Silhouette, Davies-Bouldin, Calinski-Harabasz
  - External (requires true labels):   NMI, ARI, Purity, Accuracy

Usage:
    from evaluation.metrics import evaluate

    results = evaluate(X, labels_pred, labels_true=y)
    print(results)
"""
import numpy as np
from typing import Optional
import time


# ─── Internal metrics (no labels needed) ─────────────────────────────────────

def inertia(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """Sum of squared distances from each point to its assigned center."""
    total = 0.0
    for c in range(len(centers)):
        mask = labels == c
        if mask.any():
            total += float(np.sum(np.linalg.norm(X[mask] - centers[c], axis=1) ** 2))
    return total


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Mean silhouette coefficient over all samples.
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    Range: [-1, 1]. Closer to 1 = better.
    """
    n = len(X)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    scores = []
    for i in range(n):
        own_cluster = labels[i]
        own_mask = (labels == own_cluster)
        own_mask[i] = False

        if own_mask.sum() == 0:
            scores.append(0.0)
            continue

        a = np.mean(np.linalg.norm(X[own_mask] - X[i], axis=1))

        b = np.inf
        for c in unique_labels:
            if c == own_cluster:
                continue
            other_mask = labels == c
            mean_dist = np.mean(np.linalg.norm(X[other_mask] - X[i], axis=1))
            b = min(b, mean_dist)

        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)

    return float(np.mean(scores))


def davies_bouldin_score(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """
    Davies-Bouldin index. Lower = better.
    DB = (1/k) * sum_i max_{j≠i} (s_i + s_j) / d(c_i, c_j)
    """
    k = len(centers)
    s = np.array([
        np.mean(np.linalg.norm(X[labels == c] - centers[c], axis=1))
        if (labels == c).any() else 0.0
        for c in range(k)
    ])

    db_sum = 0.0
    for i in range(k):
        max_ratio = 0.0
        for j in range(k):
            if i == j:
                continue
            d_ij = np.linalg.norm(centers[i] - centers[j])
            if d_ij > 0:
                ratio = (s[i] + s[j]) / d_ij
                max_ratio = max(max_ratio, ratio)
        db_sum += max_ratio

    return float(db_sum / k)


def calinski_harabasz_score(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """
    Calinski-Harabasz (Variance Ratio Criterion). Higher = better.
    """
    n = len(X)
    k = len(centers)
    if k <= 1 or n <= k:
        return 0.0

    global_mean = X.mean(axis=0)

    # Between-cluster scatter
    bss = sum(
        (labels == c).sum() * np.linalg.norm(centers[c] - global_mean) ** 2
        for c in range(k)
    )

    # Within-cluster scatter
    wss = sum(
        np.sum(np.linalg.norm(X[labels == c] - centers[c], axis=1) ** 2)
        for c in range(k)
        if (labels == c).any()
    )

    if wss == 0:
        return float("inf")

    return float((bss / (k - 1)) / (wss / (n - k)))


# ─── External metrics (ground truth required) ────────────────────────────────

def nmi_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Normalized Mutual Information. Range [0, 1]. Higher = better."""
    from math import log

    n = len(labels_true)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    # Contingency table
    cont = np.array([
        [(labels_true == c) & (labels_pred == k) for k in clusters]
        for c in classes
    ], dtype=float).sum(axis=2) if False else np.zeros((len(classes), len(clusters)))

    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            cont[i, j] = np.sum((labels_true == c) & (labels_pred == k))

    # Entropies
    def entropy(counts):
        probs = counts[counts > 0] / n
        return -np.sum(probs * np.log(probs))

    h_true = entropy(np.array([(labels_true == c).sum() for c in classes]))
    h_pred = entropy(np.array([(labels_pred == k).sum() for k in clusters]))

    # Mutual information
    mi = 0.0
    for i in range(len(classes)):
        for j in range(len(clusters)):
            if cont[i, j] > 0:
                mi += (cont[i, j] / n) * log(
                    (cont[i, j] * n) / (cont[i, :].sum() * cont[:, j].sum())
                )

    denom = (h_true + h_pred) / 2
    return float(mi / denom) if denom > 0 else 0.0


def ari_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Adjusted Rand Index. Range [-1, 1]. Higher = better."""
    n = len(labels_true)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    cont = np.zeros((len(classes), len(clusters)), dtype=int)
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            cont[i, j] = np.sum((labels_true == c) & (labels_pred == k))

    def comb2(x):
        return x * (x - 1) / 2

    sum_comb_c = sum(comb2(cont[i, :].sum()) for i in range(len(classes)))
    sum_comb_k = sum(comb2(cont[:, j].sum()) for j in range(len(clusters)))
    sum_comb = sum(comb2(cont[i, j]) for i in range(len(classes)) for j in range(len(clusters)))

    expected = sum_comb_c * sum_comb_k / comb2(n)
    max_val = (sum_comb_c + sum_comb_k) / 2

    denom = max_val - expected
    return float((sum_comb - expected) / denom) if denom > 0 else 0.0


def purity_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Purity: fraction of correctly assigned points (majority vote per cluster)."""
    total = 0
    for k in np.unique(labels_pred):
        mask = labels_pred == k
        if mask.any():
            total += np.max(np.bincount(labels_true[mask].astype(int)))
    return float(total / len(labels_true))


# ─── Master evaluation function ──────────────────────────────────────────────

def evaluate(
    X: np.ndarray,
    labels_pred: np.ndarray,
    centers: np.ndarray,
    labels_true: Optional[np.ndarray] = None,
    compute_silhouette: bool = True,
) -> dict:
    """
    Compute all metrics for a clustering result.

    Parameters
    ----------
    X            : data matrix (n, d)
    labels_pred  : cluster assignments (n,)
    centers      : cluster centers (k, d)
    labels_true  : ground truth labels (n,) — optional
    compute_silhouette : silhouette is O(n^2), can be slow for large n

    Returns
    -------
    dict of metric_name -> value
    """
    metrics = {
        "inertia":           inertia(X, labels_pred, centers),
        "davies_bouldin":    davies_bouldin_score(X, labels_pred, centers),
        "calinski_harabasz": calinski_harabasz_score(X, labels_pred, centers),
    }

    if compute_silhouette:
        metrics["silhouette"] = silhouette_score(X, labels_pred)

    if labels_true is not None:
        metrics["nmi"]     = nmi_score(labels_true, labels_pred)
        metrics["ari"]     = ari_score(labels_true, labels_pred)
        metrics["purity"]  = purity_score(labels_true, labels_pred)

    return metrics
