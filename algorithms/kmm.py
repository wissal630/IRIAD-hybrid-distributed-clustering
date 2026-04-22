import numpy as np
from .base import BaseClusterer
from .kmeans import KMeans, KMeansPlusPlus
from .kmedoids import KMedoids


class KMM(BaseClusterer):
    """
    k-MM exact as Kechid 2016 — uses standard random-init K-means.
    Phase 1: K-means (random init)
    Phase 2: centroid → nearest real point
    Phase 3: PAM from those points
    """

    def fit(self, X: np.ndarray) -> "KMM":
        km = KMeans(k=self.k, max_iter=self.max_iter, random_state=self.random_state)
        km.fit(X)
        self.kmeans_centers_ = km.centers_.copy()
        self.kmeans_inertia_ = km.inertia_
        self.kmeans_iters_   = km.n_iter_

        kmed = KMedoids(k=self.k, max_iter=self.max_iter, random_state=self.random_state)
        kmed.fit_from_centers(X, warm_centers=km.centers_)
        self.kmedoids_iters_ = kmed.n_iter_

        self.labels_          = kmed.labels_
        self.centers_         = kmed.centers_
        self.inertia_         = kmed.inertia_
        self.n_iter_          = self.kmeans_iters_ + self.kmedoids_iters_
        self.medoid_indices_  = kmed.medoid_indices_
        return self

    def summary(self):
        return {
            "kmeans_inertia": self.kmeans_inertia_,
            "final_inertia":  self.inertia_,
            "kmeans_iters":   self.kmeans_iters_,
            "kmedoids_iters": self.kmedoids_iters_,
            "total_iters":    self.n_iter_,
            "fit_time_s":     self.fit_time_,
        }


class KMMPlusPlus(BaseClusterer):
    """
    k-MM with K-means++ initialization — improved variant for comparison.
    Everything identical to KMM except Phase 1 uses K-means++.
    """

    def fit(self, X: np.ndarray) -> "KMMPlusPlus":
        km = KMeansPlusPlus(k=self.k, max_iter=self.max_iter, random_state=self.random_state)
        km.fit(X)
        self.kmeans_centers_ = km.centers_.copy()
        self.kmeans_inertia_ = km.inertia_
        self.kmeans_iters_   = km.n_iter_

        kmed = KMedoids(k=self.k, max_iter=self.max_iter, random_state=self.random_state)
        kmed.fit_from_centers(X, warm_centers=km.centers_)
        self.kmedoids_iters_ = kmed.n_iter_

        self.labels_         = kmed.labels_
        self.centers_        = kmed.centers_
        self.inertia_        = kmed.inertia_
        self.n_iter_         = self.kmeans_iters_ + self.kmedoids_iters_
        self.medoid_indices_ = kmed.medoid_indices_
        return self

    def summary(self):
        return {
            "kmeans_inertia": self.kmeans_inertia_,
            "final_inertia":  self.inertia_,
            "kmeans_iters":   self.kmeans_iters_,
            "kmedoids_iters": self.kmedoids_iters_,
            "total_iters":    self.n_iter_,
            "fit_time_s":     self.fit_time_,
        }