"""
Dataset loaders — aligned with the exact datasets used in Kechid et al. (2016).

Paper datasets (Springer 2016):
  1. Breast Cancer Wisconsin (UCI) — 569 samples, 30 features, 2 classes
  2. COIL-100 (Columbia Object Image Library) — up to 5000 images, k=100

Journal extension (IOS Press 2016):
  3. Car Evaluation (UCI) — 1728 samples, 6 categorical features, 4 classes
  4. FTCDC — Firm-Teacher Clave-Direction Classification (UCI)

Extra:
  5. Iris (UCI) — 150 samples, 4 features, 3 classes

Metrics used in the paper: intra-class inertia, inter-class inertia, NMI, runtime.
Comparisons: k-MM vs k-means, PAM, CLARA, CLARANS.

All returns: (X: np.ndarray float64, y: np.ndarray int) normalized to [0,1].
"""
import os
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd


def _scale(X: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1] per feature."""
    return MinMaxScaler().fit_transform(X.astype(float))


# ─── 0. Iris ──────────────────────────────────────────────────────────────────

def load_iris_data(csv_path: str = None):
    """
    Iris: 150 samples, 4 numeric features, 3 classes.
    k = 3.
    Expects iris.csv in the data/ directory (last column = class label).
    """
    if csv_path is None:
        # Try next to this file, then next to the caller
        candidates = [
            os.path.join(os.path.dirname(__file__), "iris.csv"),
            os.path.join(os.path.dirname(__file__), "..", "data", "iris.csv"),
            "iris.csv",
        ]
        for path in candidates:
            if os.path.exists(path):
                csv_path = path
                break
        else:
            raise FileNotFoundError(
                "iris.csv not found. Place it in the data/ directory."
            )

    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values.astype(float)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    return _scale(X), y

def load_wine_data(csv_path: str = None):
    if csv_path is None:
        candidates = [
            os.path.join(os.path.dirname(__file__), "wine.csv"),
            os.path.join(os.path.dirname(__file__), "..", "data", "wine.csv"),
            "wine.csv",
        ]
        for path in candidates:
            if os.path.exists(path):
                csv_path = path
                break
        else:
            raise FileNotFoundError("wine.csv not found.")

    df = pd.read_csv(csv_path)

    # ✅ label is FIRST column
    y = LabelEncoder().fit_transform(df.iloc[:, 0].values)

    # ✅ features are the rest
    X = df.iloc[:, 1:].values.astype(float)

    return _scale(X), y

def load_mnist_data(n_samples: int = 10000, n_components: int = 50,
                    random_state: int = 42, csv_path: str = None):
    """
    MNIST digits: 784 features (28x28 pixels), 10 classes (0-9).
    Expects mnist_train.csv with first column 'label', rest pixel values.
    Takes n_samples via stratified sampling (equal per digit class).
    Reduces to n_components via PCA, then MinMax scales.
    k = 10.
    """
    from sklearn.decomposition import PCA

    if csv_path is None:
        candidates = [
            os.path.join(os.path.dirname(__file__), "mnist_train.csv"),
            os.path.join(os.path.dirname(__file__), "..", "data", "mnist_train.csv"),
            "mnist_train.csv",
        ]
        for path in candidates:
            if os.path.exists(path):
                csv_path = path
                break
        else:
            raise FileNotFoundError(
                "mnist_train.csv not found. Download from "
                "https://www.kaggle.com/datasets/oddrationale/mnist-in-csv "
                "and place it in the data/ directory."
            )

    print(f"[MNIST] Loading from {csv_path} ...")
    df = pd.read_csv(csv_path)

    # First column is label
    y_raw = df.iloc[:, 0].values.astype(int)
    X_raw = df.iloc[:, 1:].values.astype(float)

    # Stratified subsample — equal samples per digit class
    rng = np.random.default_rng(random_state)
    per_class = n_samples // 10
    indices = []
    for digit in range(10):
        idx = np.where(y_raw == digit)[0]
        chosen = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        indices.append(chosen)
    indices = np.concatenate(indices)
    rng.shuffle(indices)
    X_raw, y_raw = X_raw[indices], y_raw[indices]

    # PCA reduction (784 → n_components)
    X_scaled = _scale(X_raw)
    n_comp = min(n_components, X_scaled.shape[0] - 1, X_scaled.shape[1])
    print(f"[MNIST] Running PCA ({X_scaled.shape[1]} → {n_comp} components) ...")
    X_pca = PCA(n_components=n_comp, random_state=random_state).fit_transform(X_scaled)

    print(f"[MNIST] Ready: {len(X_pca)} samples, shape={X_pca.shape}, k=10")
    return _scale(X_pca), y_raw

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