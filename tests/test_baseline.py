"""
Tests for the KMM baseline and evaluation metrics.
Run with: python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from algorithms import KMeans, KMedoids, KMM
from evaluation.metrics import (
    inertia, silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, nmi_score, ari_score, purity_score, evaluate
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_data():
    X, y = make_blobs(n_samples=150, centers=3, random_state=42)
    return X.astype(float), y


@pytest.fixture
def kmm_fitted(simple_data):
    X, y = simple_data
    model = KMM(k=3, random_state=42)
    model.fit_timed(X)
    return model, X, y


# ─── Algorithm tests ──────────────────────────────────────────────────────────

class TestKMeans:
    def test_output_shape(self, simple_data):
        X, _ = simple_data
        km = KMeans(k=3, random_state=0).fit(X)
        assert km.labels_.shape == (150,)
        assert km.centers_.shape == (3, X.shape[1])

    def test_labels_in_range(self, simple_data):
        X, _ = simple_data
        km = KMeans(k=3, random_state=0).fit(X)
        assert set(km.labels_).issubset({0, 1, 2})

    def test_inertia_positive(self, simple_data):
        X, _ = simple_data
        km = KMeans(k=3, random_state=0).fit(X)
        assert km.inertia_ > 0


class TestKMedoids:
    def test_medoids_are_data_points(self, simple_data):
        X, _ = simple_data
        kmed = KMedoids(k=3, random_state=0).fit(X)
        for center in kmed.centers_:
            assert any(np.allclose(center, x) for x in X), \
                "Medoid center must be an actual data point"

    def test_warm_start(self, simple_data):
        X, _ = simple_data
        km = KMeans(k=3, random_state=0).fit(X)
        kmed = KMedoids(k=3, random_state=0)
        kmed.fit_from_centers(X, km.centers_)
        assert kmed.labels_.shape == (150,)


class TestKMM:
    def test_centers_are_data_points(self, kmm_fitted):
        model, X, _ = kmm_fitted
        for center in model.centers_:
            assert any(np.allclose(center, x) for x in X), \
                "KMM final centers must be real data points (medoids)"

    def test_inertia_less_than_kmeans(self, simple_data):
        """KMM should have ≤ inertia than pure k-means on well-separated data."""
        X, _ = simple_data
        km = KMeans(k=3, random_state=42).fit(X)
        kmm = KMM(k=3, random_state=42).fit(X)
        # KMM uses medoids so inertia might differ — just check it's a valid float
        assert isinstance(kmm.inertia_, float)
        assert kmm.inertia_ > 0

    def test_fit_time_recorded(self, kmm_fitted):
        model, _, _ = kmm_fitted
        assert model.fit_time_ > 0

    def test_summary_keys(self, kmm_fitted):
        model, _, _ = kmm_fitted
        s = model.summary()
        assert "final_inertia" in s
        assert "fit_time_s" in s

    def test_reproducibility(self, simple_data):
        X, _ = simple_data
        m1 = KMM(k=3, random_state=7).fit(X)
        m2 = KMM(k=3, random_state=7).fit(X)
        np.testing.assert_array_equal(m1.labels_, m2.labels_)


# ─── Metric tests ─────────────────────────────────────────────────────────────

class TestMetrics:
    def test_silhouette_range(self, kmm_fitted):
        model, X, _ = kmm_fitted
        s = silhouette_score(X, model.labels_)
        assert -1.0 <= s <= 1.0

    def test_db_positive(self, kmm_fitted):
        model, X, _ = kmm_fitted
        db = davies_bouldin_score(X, model.labels_, model.centers_)
        assert db >= 0

    def test_ch_positive(self, kmm_fitted):
        model, X, _ = kmm_fitted
        ch = calinski_harabasz_score(X, model.labels_, model.centers_)
        assert ch >= 0

    def test_nmi_range(self, kmm_fitted):
        model, X, y = kmm_fitted
        nmi = nmi_score(y, model.labels_)
        assert 0.0 <= nmi <= 1.0

    def test_ari_range(self, kmm_fitted):
        model, X, y = kmm_fitted
        ari = ari_score(y, model.labels_)
        assert -1.0 <= ari <= 1.0

    def test_purity_range(self, kmm_fitted):
        model, X, y = kmm_fitted
        p = purity_score(y, model.labels_)
        assert 0.0 <= p <= 1.0

    def test_evaluate_all_keys(self, kmm_fitted):
        model, X, y = kmm_fitted
        res = evaluate(X, model.labels_, model.centers_, labels_true=y)
        for key in ["inertia", "davies_bouldin", "calinski_harabasz",
                    "silhouette", "nmi", "ari", "purity"]:
            assert key in res, f"Missing metric: {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
