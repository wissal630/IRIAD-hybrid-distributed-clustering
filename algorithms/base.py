"""
Base class for all clustering algorithms in this project.
Every hybrid method (KMM, hybrid-A, hybrid-B...) must inherit from this.
"""
from abc import ABC, abstractmethod
import numpy as np
import time


class BaseClusterer(ABC):
    """
    Abstract base class that enforces a common interface.
    All algorithms must implement fit() and expose labels_ and centers_.
    """

    def __init__(self, k: int, max_iter: int = 100, random_state: int = 42):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state

        # Set after fit()
        self.labels_: np.ndarray | None = None
        self.centers_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int = 0
        self.fit_time_: float = 0.0

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseClusterer":
        """
        Fit the clustering model to data X.
        Must set self.labels_, self.centers_, self.inertia_, self.n_iter_.
        Must return self.
        """
        ...

    def fit_timed(self, X: np.ndarray) -> "BaseClusterer":
        """Wrapper that records wall-clock time of fit()."""
        t0 = time.perf_counter()
        self.fit(X)
        self.fit_time_ = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each point in X to nearest center."""
        if self.centers_ is None:
            raise RuntimeError("Call fit() before predict().")
        dists = np.linalg.norm(X[:, None] - self.centers_[None, :], axis=2)
        return np.argmin(dists, axis=1)

    def __repr__(self):
        return f"{self.__class__.__name__}(k={self.k}, max_iter={self.max_iter})"