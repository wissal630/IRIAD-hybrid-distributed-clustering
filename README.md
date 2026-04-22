````markdown
# IRIAD — Hybrid Distributed Clustering

Implémentation et comparaison d'hybridations K-means / K-médoïdes dans un contexte distribué.
Baseline : **k-MM** (Drias, Cherif, Kechid — Springer 2016).

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```
````

---

## Structure du projet

```
algorithms/
  base.py             ← classe abstraite BaseClusterer (tous les algos héritent de ça)
  kmeans.py           ← KMeans (random init) + KMeansPlusPlus (smart init)
  kmedoids.py         ← PAM avec warm-start depuis des centroïdes
  kmm.py              ← KMM (Kechid 2016 exact) + KMMPlusPlus (amélioration)
  hybrid_template.py  ← COPIER CE FICHIER pour implémenter un nouvel hybride

evaluation/
  metrics.py          ← intra/inter inertia, silhouette, DB, CH, NMI, ARI, purity
  benchmark.py        ← compare_algorithms() → DataFrame prêt pour analyse

data/
  loaders.py          ← Breast Cancer (UCI) + COIL-100 (surrogate ou réel)

experiments/
  run_baseline.py     ← script principal, lance tous les algos sur les 2 datasets

results/              ← CSV générés automatiquement (ne pas commiter les gros fichiers)

tests/
  test_baseline.py    ← 17 tests pytest
```

---

## Lancer le baseline

```bash
# Run rapide (3 seeds, 2 tailles COIL-100) — pour vérifier que tout marche
python experiments/run_baseline.py --quick

# Run complet (10 seeds, 7 tailles COIL-100) — pour les résultats finaux
python experiments/run_baseline.py
```

Résultats sauvegardés dans `results/` :

- `baseline_breast_cancer.csv` — réplique les figures 1-3 du papier
- `baseline_coil100.csv` — réplique les figures 4-8 du papier
- `baseline_all.csv` — tableau combiné pour comparer vos hybrides

---

## Algorithmes implémentés

| Algo             | Description                                   | Init                             |
| ---------------- | --------------------------------------------- | -------------------------------- |
| `KMeans`         | K-means standard                              | aléatoire (comme dans le papier) |
| `KMeansPlusPlus` | K-means amélioré                              | K-means++                        |
| `PAM` (KMedoids) | K-médoïdes pur                                | aléatoire                        |
| `KMM`            | **Baseline Kechid 2016** — K-means → médoïdes | aléatoire                        |
| `KMMPlusPlus`    | KMM avec meilleure init                       | K-means++                        |

---

## Ajouter un hybride

1. Copier `algorithms/hybrid_template.py` → `algorithms/hybrid_a.py` (ou `hybrid_b.py`)
2. Renommer la classe, implémenter `fit(X)` — doit setter `labels_`, `centers_`, `inertia_`, `n_iter_`
3. Ajouter l'import dans `algorithms/__init__.py`
4. L'ajouter dans le dict `algos` dans `experiments/run_baseline.py`
5. Lancer `pytest tests/ -v` pour vérifier
6. PR vers `develop`

Contrainte : **votre classe doit hériter de `BaseClusterer`**. C'est la seule règle — le reste du framework (metrics, benchmark, tests) fonctionne automatiquement.

---

## Métriques

| Métrique             | Type        | Meilleur |
| -------------------- | ----------- | -------- |
| Inertia intra-classe | interne     | min      |
| Inertia inter-classe | interne     | max      |
| Silhouette           | interne     | max → 1  |
| Davies-Bouldin       | interne     | min      |
| Calinski-Harabasz    | interne     | max      |
| NMI                  | externe     | max → 1  |
| ARI                  | externe     | max → 1  |
| Purity               | externe     | max → 1  |
| Runtime (s)          | performance | min      |

```

```
