"""
K-medoids (PAM — Partitioning Around Medoids) — used as Phase 2 of k-MM.
Kaufman & Rousseeuw (1990).
"""
import numpy as np
from .base import BaseClusterer


class KMedoids(BaseClusterer):
    """
    Standard PAM: medoids are always actual data points.
    Complexity: O(k(n-k)^2) per iteration.
    """

    def fit(self, X: np.ndarray) -> "KMedoids":
        rng = np.random.default_rng(self.random_state)
        n = len(X)

        # Random initialization (will be replaced by KMM with warm start)
        medoid_indices = rng.choice(n, size=self.k, replace=False)
        return self._fit_from_indices(X, medoid_indices)

    def fit_from_centers(self, X: np.ndarray, warm_centers: np.ndarray) -> "KMedoids":
        """
        Warm-start: find the k nearest actual data points to warm_centers.
        This is the key step in k-MM: centroids → nearest real points → medoids.
        """
        medoid_indices = np.array([
            np.argmin(np.linalg.norm(X - c, axis=1))
            for c in warm_centers
        ])
        # Deduplicate: if two centroids map to the same point, pick next closest
        seen = set()
        unique_indices = []
        for i, idx in enumerate(medoid_indices):
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
            else:
                # Find next closest unused point
                dists = np.linalg.norm(X - warm_centers[i], axis=1)
                sorted_idx = np.argsort(dists)
                for candidate in sorted_idx:
                    if candidate not in seen:
                        seen.add(candidate)
                        unique_indices.append(candidate)
                        break

        return self._fit_from_indices(X, np.array(unique_indices))

    def _fit_from_indices(self, X: np.ndarray, medoid_indices: np.ndarray) -> "KMedoids":
        n = len(X)
        medoids = medoid_indices.copy()

        for iteration in range(self.max_iter):
            # Assignment: each point → nearest medoid
            dists = np.linalg.norm(X[:, None] - X[medoids][None, :], axis=2)
            labels = np.argmin(dists, axis=1)

            new_medoids = medoids.copy()
            improved = False

            for c in range(self.k):
                cluster_mask = labels == c
                if not cluster_mask.any():
                    continue

                cluster_pts = np.where(cluster_mask)[0]

                # Find the point in cluster that minimizes total intra-cluster distance
                best_cost = np.sum(np.linalg.norm(X[cluster_pts] - X[new_medoids[c]], axis=1))
                best_idx = new_medoids[c]

                for candidate in cluster_pts:
                    cost = np.sum(np.linalg.norm(X[cluster_pts] - X[candidate], axis=1))
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = candidate

                if best_idx != new_medoids[c]:
                    new_medoids[c] = best_idx
                    improved = True

            medoids = new_medoids
            self.n_iter_ = iteration + 1
            if not improved:
                break

        # Final assignment
        dists = np.linalg.norm(X[:, None] - X[medoids][None, :], axis=2)
        self.labels_ = np.argmin(dists, axis=1)
        self.centers_ = X[medoids]
        self.medoid_indices_ = medoids
        self.inertia_ = float(np.sum(np.min(dists, axis=1)))
        return self