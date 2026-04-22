"""
Dataset loaders — aligned with the exact datasets used in Kechid et al. (2016).

Paper datasets (Springer 2016):
  1. Breast Cancer Wisconsin (UCI) — 569 samples, 30 features, 2 classes
  2. COIL-100 (Columbia Object Image Library) — up to 5000 images, k=100

Journal extension (IOS Press 2016):
  3. Car Evaluation (UCI) — 1728 samples, 6 categorical features, 4 classes
  4. FTCDC — Firm-Teacher Clave-Direction Classification (UCI)

Metrics used in the paper: intra-class inertia, inter-class inertia, NMI, runtime.
Comparisons: k-MM vs k-means, PAM, CLARA, CLARANS.

All returns: (X: np.ndarray float64, y: np.ndarray int) normalized to [0,1].
"""
import os
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def _scale(X: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1] per feature."""
    return MinMaxScaler().fit_transform(X.astype(float))


# ─── 1. Breast Cancer Wisconsin (exact paper dataset) ────────────────────────

def load_breast_cancer_data():
    """
    Breast Cancer Wisconsin: 569 samples, 30 numeric features, 2 classes.
    Used in Kechid 2016: inertia, NMI, runtime figures 1-3.
    k = 2 (malignant / benign).
    Source: sklearn.datasets (mirrors UCI repository)
    """
    d = load_breast_cancer()
    return _scale(d.data), d.target


# ─── 2. COIL-100 image dataset ────────────────────────────────────────────────

def load_coil100_data(max_images: int = 5000, data_dir: str = None):
    """
    Columbia Object Image Library (COIL-100).
    100 objects × 72 shots = 7200 images total.
    Paper stops at 5000 images due to PAM memory overflow (mentioned explicitly).

    Download: http://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php
    Extract to data/coil-100/ directory.

    Pipeline: 128x128 RGB image → resize 32x32 → flatten → PCA(50) → MinMax scale
    Each image labelled by object id (0-99). k=100 in paper experiments.

    If data_dir is not provided or files not found, uses a surrogate
    for development (clearly marked in output).
    """
    if data_dir:
        coil_path = os.path.join(data_dir, "coil-100")
        if os.path.isdir(coil_path):
            return _load_coil100_real(coil_path, max_images)

    return _load_coil100_surrogate(max_images)


def _load_coil100_real(coil_dir: str, max_images: int):
    """Load actual COIL-100 PNG files."""
    from PIL import Image
    from sklearn.decomposition import PCA

    images, labels = [], []
    for obj_id in range(1, 101):
        for angle in range(0, 360, 5):
            # COIL-100 filenames: obj1__0.png, obj1__5.png, ...
            for fmt in [f"obj{obj_id}__{angle}.png",
                        f"obj{obj_id}___{angle}.png",
                        f"obj{obj_id}_{angle}.png"]:
                fpath = os.path.join(coil_dir, fmt)
                if os.path.exists(fpath):
                    img = np.array(
                        Image.open(fpath).convert("RGB").resize((32, 32))
                    ).flatten().astype(float)
                    images.append(img)
                    labels.append(obj_id - 1)
                    break
            if len(images) >= max_images:
                break
        if len(images) >= max_images:
            break

    X = np.array(images)
    y = np.array(labels)
    X = _scale(X)
    # PCA: reduce to 50 dimensions (makes inter-point distances meaningful)
    n_comp = min(50, X.shape[0] - 1, X.shape[1])
    X = _scale(PCA(n_components=n_comp, random_state=42).fit_transform(X))
    print(f"[COIL-100] Loaded {len(X)} real images → shape {X.shape}")
    return X, y


def _load_coil100_surrogate(max_images: int):
    """
    Synthetic surrogate for COIL-100 — for dev / CI only.
    Generates 100 Gaussian clusters (one per object) in 50D.
    Replace with real data for final paper experiments.
    """
    n = min(max_images, 7200)
    n_per_class = max(1, n // 100)
    rng = np.random.default_rng(42)

    X_list, y_list = [], []
    for obj_id in range(100):
        center = rng.uniform(-5, 5, size=50)
        samples = rng.normal(loc=center, scale=0.4, size=(n_per_class, 50))
        X_list.append(samples)
        y_list.extend([obj_id] * n_per_class)

    X = np.vstack(X_list)[:n]
    y = np.array(y_list)[:n]
    print(f"[COIL-100] *** SURROGATE *** ({n} samples, 50D). "
          f"Download real data: http://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php")
    return _scale(X), y


# ─── 3. Car Evaluation (journal extension) ────────────────────────────────────

def load_car_evaluation_data():
    """
    Car Evaluation: 1728 samples, 6 ordinal features, 4 classes.
    Used in journal version (IOS Press 2016). k = 4.
    """
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=19)
        X_raw = dataset.data.features
        y_raw = dataset.data.targets.values.ravel()
        X_enc = np.zeros((len(X_raw), X_raw.shape[1]))
        for i, col in enumerate(X_raw.columns):
            X_enc[:, i] = LabelEncoder().fit_transform(X_raw[col].astype(str))
        y = LabelEncoder().fit_transform(y_raw.astype(str))
        return _scale(X_enc), y
    except Exception as e:
        print(f"[Car Evaluation] Could not load: {e}. "
              f"Install ucimlrepo: pip install ucimlrepo")
        return None, None


# ─── 4. FTCDC (journal extension) ─────────────────────────────────────────────

def load_ftcdc_data():
    """
    Firm-Teacher Clave-Direction Classification (UCI id=315).
    Used in journal version of k-MM paper.
    """
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=315)
        X = dataset.data.features.values.astype(float)
        y = LabelEncoder().fit_transform(
            dataset.data.targets.values.ravel().astype(str)
        )
        return _scale(X), y
    except Exception as e:
        print(f"[FTCDC] Could not load: {e}. Install ucimlrepo: pip install ucimlrepo")
        return None, None


# ─── Master loaders ───────────────────────────────────────────────────────────

def load_paper_datasets(coil_max: int = 5000, coil_data_dir: str = None) -> dict:
    """
    Load the exact two datasets from Kechid 2016 Springer paper.
    These are MANDATORY for the baseline experiment.

    Args:
        coil_max: max COIL-100 images to load (paper uses up to 5000)
        coil_data_dir: path to directory containing coil-100/ folder
    """
    print("=" * 50)
    print("Loading paper datasets (Kechid et al. 2016)")
    print("=" * 50)

    bc_X, bc_y = load_breast_cancer_data()
    print(f"Breast Cancer: shape={bc_X.shape}, k=2")

    coil_X, coil_y = load_coil100_data(max_images=coil_max, data_dir=coil_data_dir)
    print(f"COIL-100:      shape={coil_X.shape}, k=100")

    return {
        "breast_cancer": (bc_X, bc_y),
        "coil100":       (coil_X, coil_y),
    }


def load_journal_datasets(coil_data_dir: str = None) -> dict:
    """
    Load all datasets from both the Springer paper and the journal extension.
    """
    datasets = load_paper_datasets(coil_data_dir=coil_data_dir)

    car_X, car_y = load_car_evaluation_data()
    if car_X is not None:
        print(f"Car Evaluation: shape={car_X.shape}, k=4")
        datasets["car_evaluation"] = (car_X, car_y)

    ftcdc_X, ftcdc_y = load_ftcdc_data()
    if ftcdc_X is not None:
        print(f"FTCDC:          shape={ftcdc_X.shape}")
        datasets["ftcdc"] = (ftcdc_X, ftcdc_y)

    return datasets
