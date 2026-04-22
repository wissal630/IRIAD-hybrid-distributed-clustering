"""
TEMPLATE — Hybrid Algorithm (student 2 or 3).

Copy this file to hybrid_a.py or hybrid_b.py and implement your method.
Your algorithm MUST:
  1. Inherit from BaseClusterer
  2. Implement fit(X) → self
  3. Set self.labels_, self.centers_, self.inertia_, self.n_iter_
  4. Work with the existing benchmark infrastructure (no changes needed)

To run your method against the baseline:
    from algorithms.hybrid_a import HybridA
    from evaluation import compare_algorithms
    ...
"""
import numpy as np
from .base import BaseClusterer
from .kmeans import KMeans
from .kmedoids import KMedoids


class HybridTemplate(BaseClusterer):
    """
    Replace this with your algorithm name and description.

    Reference: [Your paper citation here]
    """

    def __init__(self, k: int, max_iter: int = 100, random_state: int = 42,
                 # Add your specific hyperparameters here:
                 # e.g., threshold: float = 0.5
                 ):
        super().__init__(k=k, max_iter=max_iter, random_state=random_state)
        # self.threshold = threshold  # store your hyperparams

    def fit(self, X: np.ndarray) -> "HybridTemplate":
        """
        Implement your hybrid algorithm here.

        Minimum required at the end of fit():
            self.labels_  = np.array of shape (n,)
            self.centers_ = np.array of shape (k, d)
            self.inertia_ = float
            self.n_iter_  = int
        """
        # ── Example: just call KMM as a starting point ───────────────────
        from .kmm import KMM
        baseline = KMM(k=self.k, max_iter=self.max_iter, random_state=self.random_state)
        baseline.fit(X)

        # TODO: replace with your actual hybrid logic
        self.labels_  = baseline.labels_
        self.centers_ = baseline.centers_
        self.inertia_ = baseline.inertia_
        self.n_iter_  = baseline.n_iter_

        return self
