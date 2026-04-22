import numpy as np
from .base import BaseClusterer


class KMeans(BaseClusterer):
    """Standard K-means with RANDOM initialization — as used in Kechid 2016."""

    def fit(self, X: np.ndarray) -> "KMeans":
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        indices = rng.choice(n, size=self.k, replace=False)
        centers = X[indices].astype(float)

        labels = np.zeros(n, dtype=int)
        for iteration in range(self.max_iter):
            dists = np.linalg.norm(X[:, None] - centers[None, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            new_centers = np.array([
                X[new_labels == c].mean(axis=0) if (new_labels == c).any()
                else centers[c]
                for c in range(self.k)
            ])
            if np.allclose(new_centers, centers) and np.array_equal(new_labels, labels):
                labels = new_labels
                centers = new_centers
                self.n_iter_ = iteration + 1
                break
            labels = new_labels
            centers = new_centers
        else:
            self.n_iter_ = self.max_iter

        self.labels_ = labels
        self.centers_ = centers
        self.inertia_ = float(np.sum(
            np.min(np.linalg.norm(X[:, None] - centers[None, :], axis=2) ** 2, axis=1)
        ))
        return self


class KMeansPlusPlus(BaseClusterer):
    """K-means with K-means++ initialization — improved version for comparison."""

    def fit(self, X: np.ndarray) -> "KMeansPlusPlus":
        rng = np.random.default_rng(self.random_state)
        centers = self._init_plus_plus(X, rng)

        labels = np.zeros(len(X), dtype=int)
        for iteration in range(self.max_iter):
            dists = np.linalg.norm(X[:, None] - centers[None, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            new_centers = np.array([
                X[new_labels == c].mean(axis=0) if (new_labels == c).any()
                else centers[c]
                for c in range(self.k)
            ])
            if np.allclose(new_centers, centers) and np.array_equal(new_labels, labels):
                labels = new_labels
                centers = new_centers
                self.n_iter_ = iteration + 1
                break
            labels = new_labels
            centers = new_centers
        else:
            self.n_iter_ = self.max_iter

        self.labels_ = labels
        self.centers_ = centers
        self.inertia_ = float(np.sum(
            np.min(np.linalg.norm(X[:, None] - centers[None, :], axis=2) ** 2, axis=1)
        ))
        return self

    def _init_plus_plus(self, X, rng):
        n = len(X)
        idx = rng.integers(n)
        centers = [X[idx].copy()]
        for _ in range(1, self.k):
            dists = np.array([
                min(np.linalg.norm(x - c) ** 2 for c in centers)
                for x in X
            ])
            probs = dists / dists.sum()
            idx = rng.choice(n, p=probs)
            centers.append(X[idx].copy())
        return np.array(centers)