"""
Distributed clustering algorithms — master/worker model using ThreadPoolExecutor.

Model:
  - Data is split into N_WORKERS shards (simulating distributed nodes).
  - For KMeans: each worker runs KMeans to local convergence on its shard
    using shared initial centers, then master does one weighted aggregation.
  - For KMedoids: each worker runs PAM to local convergence on its shard,
    returns local medoid points, master runs one final global PAM seeded
    with all candidate medoids.
  - For KMM/KMM++: Phase 1 = distributed KMeans, Phase 2 = distributed PAM
    warm-started from centroid->medoid mapping done by master.

All classes inherit BaseClusterer and expose the same interface as their
sequential counterparts so they drop into benchmark/run_baseline unchanged.
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from .base import BaseClusterer
from .kmeans import KMeans, KMeansPlusPlus
from .kmedoids import KMedoids

N_WORKERS = 4


def _split_data(X: np.ndarray, n_workers: int, random_state: int = 42):
    indices = np.random.default_rng(random_state).permutation(len(X))
    shards = np.array_split(indices, n_workers)
    return [(X[idx], idx) for idx in shards if len(idx) > 0]


# ─── Distributed KMeans ───────────────────────────────────────────────────────

def _worker_kmeans(X_shard, centers, max_iter, seed):
    """
    Worker: run KMeans to local convergence on shard, starting from given centers.
    Returns (local_centers, cluster_counts) — both needed for weighted aggregation.
    """
    k = len(centers)
    local_centers = centers.copy()

    labels = np.zeros(len(X_shard), dtype=int)
    for _ in range(max_iter):
        dists = np.linalg.norm(X_shard[:, None] - local_centers[None, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        new_centers = np.array([
            X_shard[new_labels == c].mean(axis=0) if (new_labels == c).any()
            else local_centers[c]
            for c in range(k)
        ])
        if np.allclose(new_centers, local_centers) and np.array_equal(new_labels, labels):
            labels = new_labels
            local_centers = new_centers
            break
        labels = new_labels
        local_centers = new_centers

    counts = np.array([(labels == c).sum() for c in range(k)], dtype=float)
    return local_centers, counts


class DistributedKMeans(BaseClusterer):
    """
    Distributed KMeans:
      1. Master initializes centers (random).
      2. Each worker runs KMeans to local convergence on its shard.
      3. Master aggregates via weighted average of local centers.
      4. One round only (workers converge locally, master aggregates once).
      5. Master does final global assignment.
    """

    def fit(self, X: np.ndarray) -> "DistributedKMeans":
        rng = np.random.default_rng(self.random_state)
        # Master initializes centers from full data
        init_idx = rng.choice(len(X), size=self.k, replace=False)
        centers = X[init_idx].astype(float)

        shards = _split_data(X, N_WORKERS)

        # Workers run to local convergence in parallel
        local_results = []
        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {
                pool.submit(_worker_kmeans, shard, centers, self.max_iter, self.random_state): i
                for i, (shard, _) in enumerate(shards)
            }
            for f in as_completed(futures):
                local_results.append(f.result())

        # Master aggregates: weighted average by cluster counts
        total_counts = np.zeros(self.k)
        weighted_sum = np.zeros((self.k, X.shape[1]))
        for local_centers, counts in local_results:
            weighted_sum += local_centers * counts[:, None]
            total_counts += counts

        # Avoid division by zero for empty clusters
        safe_counts = np.where(total_counts > 0, total_counts, 1)
        centers = weighted_sum / safe_counts[:, None]

        # Master does final global assignment
        dists = np.linalg.norm(X[:, None] - centers[None, :], axis=2)
        labels = np.argmin(dists, axis=1)

        # Recompute centers from final global assignment
        centers = np.array([
            X[labels == c].mean(axis=0) if (labels == c).any() else centers[c]
            for c in range(self.k)
        ])

        self.labels_ = labels
        self.centers_ = centers
        self.n_iter_ = 1  # 1 global round
        self.inertia_ = float(np.sum(
            np.min(np.linalg.norm(X[:, None] - centers[None, :], axis=2) ** 2, axis=1)
        ))
        return self


class DistributedKMeansPlusPlus(BaseClusterer):
    """
    Distributed KMeans++:
      Same as DistributedKMeans but master uses KMeans++ seeding on full data.
    """

    def fit(self, X: np.ndarray) -> "DistributedKMeansPlusPlus":
        rng = np.random.default_rng(self.random_state)

        # Master runs KMeans++ init on full data
        idx = rng.integers(len(X))
        centers = [X[idx].copy()]
        for _ in range(1, self.k):
            dists = np.array([
                min(np.linalg.norm(x - c) ** 2 for c in centers)
                for x in X
            ])
            probs = dists / dists.sum()
            idx = rng.choice(len(X), p=probs)
            centers.append(X[idx].copy())
        centers = np.array(centers)

        shards = _split_data(X, N_WORKERS)

        local_results = []
        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {
                pool.submit(_worker_kmeans, shard, centers, self.max_iter, self.random_state): i
                for i, (shard, _) in enumerate(shards)
            }
            for f in as_completed(futures):
                local_results.append(f.result())

        total_counts = np.zeros(self.k)
        weighted_sum = np.zeros((self.k, X.shape[1]))
        for local_centers, counts in local_results:
            weighted_sum += local_centers * counts[:, None]
            total_counts += counts

        safe_counts = np.where(total_counts > 0, total_counts, 1)
        centers = weighted_sum / safe_counts[:, None]

        dists = np.linalg.norm(X[:, None] - centers[None, :], axis=2)
        labels = np.argmin(dists, axis=1)
        centers = np.array([
            X[labels == c].mean(axis=0) if (labels == c).any() else centers[c]
            for c in range(self.k)
        ])

        self.labels_ = labels
        self.centers_ = centers
        self.n_iter_ = 1
        self.inertia_ = float(np.sum(
            np.min(np.linalg.norm(X[:, None] - centers[None, :], axis=2) ** 2, axis=1)
        ))
        return self


# ─── Distributed KMedoids (PAM) ───────────────────────────────────────────────

def _worker_pam(X_shard, init_medoid_points, max_iter):
    """
    Worker: run PAM to local convergence on shard.
    init_medoid_points: (k, d) array — starting medoid coordinates.
    Returns local medoid points (k, d) — actual data points from this shard.
    """
    k = len(init_medoid_points)
    n = len(X_shard)

    # Map init medoid coords to nearest actual points in this shard
    medoid_pts = np.array([
        X_shard[np.argmin(np.linalg.norm(X_shard - m, axis=1))]
        for m in init_medoid_points
    ])

    for _ in range(max_iter):
        dists = np.linalg.norm(X_shard[:, None] - medoid_pts[None, :], axis=2)
        labels = np.argmin(dists, axis=1)

        new_medoid_pts = medoid_pts.copy()
        improved = False
        for c in range(k):
            cluster_mask = labels == c
            if not cluster_mask.any():
                continue
            cluster_pts = X_shard[cluster_mask]
            # Best medoid = point minimizing total distance within local cluster
            costs = np.array([
                np.sum(np.linalg.norm(cluster_pts - p, axis=1))
                for p in cluster_pts
            ])
            best_local = cluster_pts[np.argmin(costs)]
            if not np.array_equal(best_local, new_medoid_pts[c]):
                new_medoid_pts[c] = best_local
                improved = True

        medoid_pts = new_medoid_pts
        if not improved:
            break

    return medoid_pts  # (k, d) real data points from this shard


class DistributedKMedoids(BaseClusterer):
    """
    Distributed PAM (CLARA-inspired):
      1. Master splits data, broadcasts initial medoid points to all workers.
      2. Each worker runs PAM to local convergence on its shard.
      3. Master collects all N_WORKERS × k candidate medoid points.
      4. Master maps each candidate to its nearest actual global data point.
      5. Master runs one final global PAM seeded with the best k candidates
         (chosen by their global cost).
    """

    def fit(self, X: np.ndarray) -> "DistributedKMedoids":
        rng = np.random.default_rng(self.random_state)
        # Master picks random initial medoids from full data
        init_idx = rng.choice(len(X), size=self.k, replace=False)
        init_medoid_pts = X[init_idx].astype(float)

        return self._fit_from_medoid_points(X, init_medoid_pts)

    def _fit_from_medoid_points(self, X: np.ndarray, init_medoid_pts: np.ndarray):
        shards = _split_data(X, N_WORKERS)

        # Workers run local PAM in parallel
        candidate_sets = []
        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {
                pool.submit(_worker_pam, shard, init_medoid_pts, self.max_iter): i
                for i, (shard, _) in enumerate(shards)
            }
            for f in as_completed(futures):
                candidate_sets.append(f.result())  # each is (k, d)

        # Master collects all candidates: shape (N_WORKERS * k, d)
        all_candidates = np.vstack(candidate_sets)

        # Map each candidate to the nearest actual global data point
        global_candidates = np.array([
            X[np.argmin(np.linalg.norm(X - pt, axis=1))]
            for pt in all_candidates
        ])

        # Deduplicate
        unique_candidates = np.unique(global_candidates, axis=0)

        # Pick best k seeds by global cost (each candidate as sole medoid of its cluster)
        # Use a greedy approach: pick the k candidates that together minimize global cost
        # Simple version: score each by its total distance to all points, pick best k
        scores = np.array([
            np.sum(np.min(
                np.linalg.norm(X[:, None] - unique_candidates[None, :], axis=2),
                axis=1
            ))
            for _ in [None]  # full pairwise score computed once below
        ])

        # Compute pairwise distances from X to all unique candidates (n, n_cand)
        dist_to_cands = np.linalg.norm(
            X[:, None] - unique_candidates[None, :], axis=2
        )  # (n, n_cand)

        # Greedily pick k candidates minimizing total assignment cost
        selected = []
        remaining = list(range(len(unique_candidates)))
        for _ in range(min(self.k, len(unique_candidates))):
            if not remaining:
                break
            if not selected:
                costs = dist_to_cands[:, remaining].min(axis=0)
                best = remaining[int(np.argmin(
                    [dist_to_cands[:, r].sum() for r in remaining]
                ))]
            else:
                current_best = dist_to_cands[:, selected].min(axis=1)
                best = remaining[int(np.argmin([
                    np.sum(np.minimum(current_best, dist_to_cands[:, r]))
                    for r in remaining
                ]))]
            selected.append(best)
            remaining.remove(best)

        seed_medoids = unique_candidates[selected]

        # Pad with random points if we don't have enough unique candidates
        if len(seed_medoids) < self.k:
            rng = np.random.default_rng(self.random_state + 999)
            extras = X[rng.choice(len(X), size=self.k - len(seed_medoids), replace=False)]
            seed_medoids = np.vstack([seed_medoids, extras])

        # Final global PAM from these seeds
        kmed = KMedoids(k=self.k, max_iter=self.max_iter, random_state=self.random_state)
        kmed.fit_from_centers(X, warm_centers=seed_medoids)

        self.labels_ = kmed.labels_
        self.centers_ = kmed.centers_
        self.medoid_indices_ = kmed.medoid_indices_
        self.inertia_ = kmed.inertia_
        self.n_iter_ = kmed.n_iter_ + 1  # local rounds + 1 global
        return self


# ─── Distributed KMM ──────────────────────────────────────────────────────────

class DistributedKMM(BaseClusterer):
    """
    Distributed k-MM:
      Phase 1: Distributed KMeans (random init) → global centroids.
      Bridge:  Master maps centroids → nearest real data points (medoid seeds).
      Phase 2: Distributed PAM warm-started from those seeds.
    """

    def fit(self, X: np.ndarray) -> "DistributedKMM":
        # Phase 1: distributed KMeans
        dkm = DistributedKMeans(k=self.k, max_iter=self.max_iter, random_state=self.random_state)
        dkm.fit(X)
        self.kmeans_centers_ = dkm.centers_.copy()
        self.kmeans_inertia_ = dkm.inertia_

        # Bridge: centroids → nearest real points
        medoid_seeds = np.array([
            X[np.argmin(np.linalg.norm(X - c, axis=1))]
            for c in dkm.centers_
        ])

        # Phase 2: distributed PAM warm-started
        dkmed = DistributedKMedoids(k=self.k, max_iter=self.max_iter, random_state=self.random_state)
        dkmed._fit_from_medoid_points(X, medoid_seeds)

        self.labels_ = dkmed.labels_
        self.centers_ = dkmed.centers_
        self.inertia_ = dkmed.inertia_
        self.n_iter_ = dkm.n_iter_ + dkmed.n_iter_
        self.medoid_indices_ = dkmed.medoid_indices_
        return self


class DistributedKMMPlusPlus(BaseClusterer):
    """
    Distributed k-MM++:
      Same as DistributedKMM but Phase 1 uses KMeans++ seeding.
    """

    def fit(self, X: np.ndarray) -> "DistributedKMMPlusPlus":
        dkm = DistributedKMeansPlusPlus(k=self.k, max_iter=self.max_iter, random_state=self.random_state)
        dkm.fit(X)
        self.kmeans_centers_ = dkm.centers_.copy()
        self.kmeans_inertia_ = dkm.inertia_

        medoid_seeds = np.array([
            X[np.argmin(np.linalg.norm(X - c, axis=1))]
            for c in dkm.centers_
        ])

        dkmed = DistributedKMedoids(k=self.k, max_iter=self.max_iter, random_state=self.random_state)
        dkmed._fit_from_medoid_points(X, medoid_seeds)

        self.labels_ = dkmed.labels_
        self.centers_ = dkmed.centers_
        self.inertia_ = dkmed.inertia_
        self.n_iter_ = dkm.n_iter_ + dkmed.n_iter_
        self.medoid_indices_ = dkmed.medoid_indices_
        return self