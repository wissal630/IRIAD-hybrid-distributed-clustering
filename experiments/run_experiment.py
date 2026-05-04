"""
Baseline experiment — centralized + distributed comparison.

Datasets:
  - Iris:             k=3   (sanity check)
  - Breast Cancer:    k=2   (paper Figs 1-3)
  - COIL-100:         k=n//72, varying n  (paper Figs 4-8)

Modes run back-to-back for every dataset:
  - Centralized:  KMeans, KMeans++, PAM, KMM, KMM++
  - Distributed:  same algorithms, 4-worker thread model

Output:
  results/
    baseline_iris.csv
    baseline_breast_cancer.csv
    baseline_coil100.csv
    baseline_all.csv
    worker_scaling.csv
    comparison_report.txt
    comparison_report.html
  plots/
    runtime_s_iris.png
    intra_inertia_iris.png
    runtime_s_breast_cancer.png
    intra_inertia_breast_cancer.png
    runtime_s_coil100.png
    intra_inertia_coil100.png
    worker_scaling_runtime_s.png

Usage:
    python experiments/run_baseline.py
    python experiments/run_baseline.py --quick
    python experiments/run_baseline.py --skip-coil
    python experiments/run_baseline.py --iris-csv path/to/iris.csv
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

from algorithms import (
    KMeans, KMeansPlusPlus, KMedoids, KMM, KMMPlusPlus,
    DistributedKMeans, DistributedKMeansPlusPlus,
    DistributedKMedoids, DistributedKMM, DistributedKMMPlusPlus,
    N_WORKERS,
)
from evaluation.metrics import (
    inertia as calc_inertia,
    nmi_score, silhouette_score,
    davies_bouldin_score, calinski_harabasz_score,
    ari_score, purity_score,
)
from data.loaders import (
    load_breast_cancer_data, load_coil100_data,
    load_iris_data, load_mnist_data,
)

RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR     = os.path.join(os.path.dirname(__file__), "..", "plots")
WORKER_COUNTS = [3, 5, 8]

# ── Consistent visual identity across every plot ───────────────────────────────
ALGO_COLORS = {
    "KMeans":        "#4C72B0",
    "KMeans++":      "#DD8452",
    "PAM":           "#55A868",
    "KMM":           "#C44E52",
    "KMM++":         "#8172B2",
    "Dist-KMeans":   "#4C72B0",
    "Dist-KMeans++": "#DD8452",
    "Dist-PAM":      "#55A868",
    "Dist-KMM":      "#C44E52",
    "Dist-KMM++":    "#8172B2",
}
ALGO_HATCH = {
    "KMeans":        "",
    "KMeans++":      "",
    "PAM":           "",
    "KMM":           "",
    "KMM++":         "",
    "Dist-KMeans":   "///",
    "Dist-KMeans++": "///",
    "Dist-PAM":      "///",
    "Dist-KMM":      "///",
    "Dist-KMM++":    "///",
}

CENTRALIZED_ALGOS = {
    "KMeans":   KMeans,
    "KMeans++": KMeansPlusPlus,
    "PAM":      KMedoids,
    "KMM":      KMM,
    "KMM++":    KMMPlusPlus,
}
DISTRIBUTED_ALGOS = {
    "Dist-KMeans":   DistributedKMeans,
    "Dist-KMeans++": DistributedKMeansPlusPlus,
    "Dist-PAM":      DistributedKMedoids,
    "Dist-KMM":      DistributedKMM,
    "Dist-KMM++":    DistributedKMMPlusPlus,
}
PAIR_MAP = {
    "KMeans":   "Dist-KMeans",
    "KMeans++": "Dist-KMeans++",
    "PAM":      "Dist-PAM",
    "KMM":      "Dist-KMM",
    "KMM++":    "Dist-KMM++",
}
ALL_ALGOS = {**CENTRALIZED_ALGOS, **DISTRIBUTED_ALGOS}

METRIC_COLS = [
    "intra_inertia", "inter_inertia", "silhouette",
    "davies_bouldin", "calinski_harabasz",
    "nmi", "ari", "purity",
    "runtime_s", "n_iter",
]
HIGHER_BETTER = {"silhouette", "calinski_harabasz", "nmi", "ari",
                 "purity", "inter_inertia"}
LOWER_BETTER  = {"intra_inertia", "davies_bouldin", "runtime_s", "n_iter"}


# ─── Metric helpers ────────────────────────────────────────────────────────────

def inter_inertia(X, labels, centers):
    mu = X.mean(axis=0)
    return float(sum(
        (labels == c).sum() * np.linalg.norm(centers[c] - mu) ** 2
        for c in range(len(centers))
    ))


def run_once(AlgoClass, X, y, k, seed):
    algo = AlgoClass(k=k, max_iter=100, random_state=seed)
    t0 = time.perf_counter()
    algo.fit(X)
    elapsed = time.perf_counter() - t0

    lbl = algo.labels_
    ctr = algo.centers_

    result = {
        "intra_inertia":     calc_inertia(X, lbl, ctr),
        "inter_inertia":     inter_inertia(X, lbl, ctr),
        "silhouette":        silhouette_score(X, lbl),
        "davies_bouldin":    davies_bouldin_score(X, lbl, ctr),
        "calinski_harabasz": calinski_harabasz_score(X, lbl, ctr),
        "runtime_s":         elapsed,
        "n_iter":            getattr(algo, "n_iter_", np.nan),
    }
    if y is not None:
        result["nmi"]    = nmi_score(y, lbl)
        result["ari"]    = ari_score(y, lbl)
        result["purity"] = purity_score(y, lbl)

    return result


# ─── Experiment runner ─────────────────────────────────────────────────────────

def run_experiment(name, X, y, k, n_runs, algorithms):
    rows = []
    for algo_name, AlgoClass in algorithms.items():
        mode = "distributed" if algo_name.startswith("Dist-") else "centralized"
        print(f"    [{mode}] {algo_name} × {n_runs} seeds ...", end=" ", flush=True)
        for seed in range(n_runs):
            try:
                r = run_once(AlgoClass, X, y, k, seed)
                r.update({
                    "algorithm": algo_name,
                    "mode":      mode,
                    "dataset":   name,
                    "k":         k,
                    "n":         len(X),
                    "seed":      seed,
                })
                algo_rows.append(r)
            except Exception as e:
                print(f"\n      [WARN] seed={seed}: {e}")
        rows.extend(algo_rows)
        if algo_rows:
            mean_rt = np.mean([r["runtime_s"] for r in algo_rows])
            print(f"ok  (mean runtime = {mean_rt:.4f}s over "
                  f"{len(algo_rows)}/{n_runs} seeds)")
        else:
            print("FAILED  (0 seeds succeeded)")
    return pd.DataFrame(rows)


# ─── Worker-scaling experiment ─────────────────────────────────────────────────

def run_worker_scaling(X, y, k, n_runs, worker_counts=WORKER_COUNTS):
    rows = []
    for nw in worker_counts:
        print(f"\n  --- Worker count = {nw} ---")
        for algo_name, AlgoClass in DISTRIBUTED_ALGOS.items():
            print(f"    {algo_name} × {n_runs} seeds ...", end=" ", flush=True)
            algo_rows = []
            for seed in range(n_runs):
                _orig_class_attr = None
                try:
                    # Strategy 1 — constructor kwarg
                    try:
                        algo = AlgoClass(k=k, max_iter=100,
                                         random_state=seed, n_workers=nw)
                    except TypeError:
                        algo = AlgoClass(k=k, max_iter=100, random_state=seed)

                    # Strategy 2 — instance attribute
                    if hasattr(algo, "n_workers"):
                        algo.n_workers = nw
                    # Strategy 3 — class-level monkey-patch
                    elif hasattr(AlgoClass, "N_WORKERS"):
                        _orig_class_attr = AlgoClass.N_WORKERS
                        AlgoClass.N_WORKERS = nw

                    t0 = time.perf_counter()
                    algo.fit(X)
                    elapsed = time.perf_counter() - t0

                    lbl, ctr = algo.labels_, algo.centers_
                    r = {
                        "algorithm":     algo_name,
                        "n_workers":     nw,
                        "seed":          seed,
                        "runtime_s":     elapsed,
                        "intra_inertia": calc_inertia(X, lbl, ctr),
                        "n_iter":        getattr(algo, "n_iter_", np.nan),
                    }
                    if y is not None:
                        r["nmi"] = nmi_score(y, lbl)
                    algo_rows.append(r)

                except Exception as e:
                    print(f"\n      [WARN] {algo_name} nw={nw} seed={seed}: {e}")

                finally:
                    if _orig_class_attr is not None:
                        AlgoClass.N_WORKERS = _orig_class_attr

            rows.extend(algo_rows)
            if algo_rows:
                print(f"ok  ({len(algo_rows)}/{n_runs} seeds succeeded)")
            else:
                print(f"FAILED  (0/{n_runs} seeds succeeded)")

    if not rows:
        print("\n  [ERROR] Worker-scaling produced no results.")
        print("          Ensure DistributedXxx classes expose a worker count via")
        print("          n_workers= kwarg, .n_workers attribute, or .N_WORKERS.")
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ─── Plot helpers ──────────────────────────────────────────────────────────────

def plot_metric_per_algo(df, metric, ylabel, title_prefix, dataset_label, plots_dir):
    """
    Bar chart of mean ± std of `metric` for every algorithm on one dataset.
    Each bar is labelled with its numeric value just above the error cap.
    Centralized = solid fill; distributed = hatched.
    """
    if metric not in df.columns:
        print(f"  [SKIP] plot_metric_per_algo: '{metric}' not in DataFrame")
        return

    agg        = df.groupby("algorithm")[metric].agg(["mean", "std"])
    means      = agg["mean"].to_dict()
    stds       = agg["std"].fillna(0).to_dict()
    algo_order = list(ALL_ALGOS.keys())
    algos      = [a for a in algo_order if a in means]
    values     = [means[a] for a in algos]
    errors     = [stds.get(a, 0.0) for a in algos]
    colors     = [ALGO_COLORS.get(a, "#888888") for a in algos]
    hatch      = [ALGO_HATCH.get(a, "") for a in algos]

    fig, ax = plt.subplots(figsize=(12, 5))
    x    = np.arange(len(algos))
    bars = ax.bar(
        x, values, yerr=errors, capsize=5,
        color=colors, edgecolor="black", linewidth=0.8,
        error_kw={"elinewidth": 1.2, "ecolor": "black"},
    )
    for bar, h in zip(bars, hatch):
        bar.set_hatch(h)

    # ── Value labels on top of each bar ───────────────────────────────────────
    # First pass: draw at provisional positions so we can read the y-axis range
    ax.set_ylim(0, max(v + e for v, e in zip(values, errors)) * 1.18)
    for bar, val, err in zip(bars, values, errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + err + ax.get_ylim()[1] * 0.012,
            f"{val:.4g}",
            ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f"{title_prefix} per Algorithm — {dataset_label}  "
        f"(mean ± std, {df['seed'].nunique()} runs)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4g"))
    ax.grid(axis="y", linestyle="--", alpha=0.45)

    legend_handles = [
        Patch(facecolor="white", edgecolor="black",
              label="Centralized (solid fill)"),
        Patch(facecolor="white", edgecolor="black", hatch="///",
              label="Distributed (hatched)"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right")

    fig.tight_layout()
    fname     = f"{metric}_{dataset_label}.png"
    save_path = os.path.join(plots_dir, fname)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✔ Saved → {os.path.relpath(save_path)}")


def plot_worker_scaling(scaling_df, plots_dir):
    """
    Line plot: x = worker count, y = mean runtime ± std,
    one line per distributed algorithm.
    Only runtime_s is plotted.
    """
    metric = "runtime_s"

    # ── Guards ────────────────────────────────────────────────────────────────
    if scaling_df is None or scaling_df.empty:
        print(f"  [SKIP] plot_worker_scaling: DataFrame is empty")
        return
    for col in ("algorithm", "n_workers", metric):
        if col not in scaling_df.columns:
            print(f"  [SKIP] plot_worker_scaling: column '{col}' missing")
            return

    dist_algos = list(DISTRIBUTED_ALGOS.keys())
    fig, ax    = plt.subplots(figsize=(9, 5))
    plotted    = 0

    for algo in dist_algos:
        sub = scaling_df[scaling_df["algorithm"] == algo]
        if sub.empty:
            continue
        grp = (sub.groupby("n_workers")[metric]
                  .agg(["mean", "std"])
                  .reset_index())
        grp["std"] = grp["std"].fillna(0)
        ax.errorbar(
            grp["n_workers"], grp["mean"], yerr=grp["std"],
            marker="o", linewidth=2, capsize=5,
            label=algo,
            color=ALGO_COLORS.get(algo, None),
        )
        plotted += 1

    if plotted == 0:
        print(f"  [SKIP] plot_worker_scaling: no algorithm had data")
        plt.close(fig)
        return

    worker_ticks = sorted(scaling_df["n_workers"].unique())
    ax.set_xlabel("Number of Workers", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title(
        "Effect of Worker Count on Runtime (Breast Cancer, k=2)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xticks(worker_ticks)
    ax.set_xticklabels([str(w) for w in worker_ticks], fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4g"))
    ax.legend(fontsize=9, loc="best", title="Algorithm", title_fontsize=9)
    ax.grid(linestyle="--", alpha=0.45)

    fig.tight_layout()
    save_path = os.path.join(plots_dir, "worker_scaling_runtime_s.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✔ Saved → {os.path.relpath(save_path)}")


# ─── TXT report ────────────────────────────────────────────────────────────────

def build_report(all_df: pd.DataFrame, n_workers: int) -> str:
    lines = []
    sep   = "=" * 72

    lines.append(sep)
    lines.append("  CENTRALIZED vs DISTRIBUTED CLUSTERING — COMPARISON REPORT")
    lines.append(f"  Workers: {n_workers}  |  "
                 f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("  All values are the mean of 10 independent runs "
                 "(different random seeds).")
    lines.append(sep)

    for dataset in all_df["dataset"].unique():
        df_ds = all_df[all_df["dataset"] == dataset]
        k_val = df_ds["k"].iloc[0]
        n_val = df_ds["n"].iloc[0]

        lines.append(f"\n{'─'*72}")
        lines.append(f"  DATASET : {dataset}   (n={n_val}, k={k_val})")
        lines.append(f"{'─'*72}")

        present = [m for m in METRIC_COLS if m in df_ds.columns]
        means   = df_ds.groupby("algorithm")[present].mean()
        stds    = df_ds.groupby("algorithm")[present].std().fillna(0)

        for cent, dist in PAIR_MAP.items():
            if cent not in means.index or dist not in means.index:
                continue

            c_row, c_std = means.loc[cent], stds.loc[cent]
            d_row, d_std = means.loc[dist], stds.loc[dist]

            lines.append(f"\n  {cent:12s}  vs  {dist}")
            lines.append(
                f"  {'Metric':<22} {'Cent mean':>12} {'±std':>8} "
                f"{'Dist mean':>12} {'±std':>8} {'Δ%':>8} {'Better':>8}"
            )
            lines.append(
                f"  {'-'*22} {'-'*12} {'-'*8} {'-'*12} {'-'*8} {'-'*8} {'-'*8}"
            )

            for m in present:
                cv, cs = c_row[m], c_std[m]
                dv, ds = d_row[m], d_std[m]
                dpct   = (dv - cv) / abs(cv) * 100 if cv != 0 else float("nan")

                if m in HIGHER_BETTER:
                    better = "CENT" if cv > dv else ("DIST" if dv > cv else "TIE")
                elif m in LOWER_BETTER:
                    better = "CENT" if cv < dv else ("DIST" if dv < cv else "TIE")
                else:
                    better = "—"

                dpct_str = f"{dpct:+.1f}%" if not np.isnan(dpct) else "—"
                lines.append(
                    f"  {m:<22} {cv:>12.4f} {cs:>8.4f} "
                    f"{dv:>12.4f} {ds:>8.4f} {dpct_str:>8} {better:>8}"
                )

        s_metrics = [m for m in ["runtime_s", "intra_inertia", "nmi", "silhouette"]
                     if m in means.columns]
        lines.append(f"\n  {'─'*68}")
        lines.append(f"  SUMMARY — all algorithms on {dataset}")
        lines.append(f"  {'─'*68}")
        lines.append(f"  {'Algorithm':<18}" +
                     "".join(f"{m:>16}" for m in s_metrics))
        lines.append(f"  {'-'*18}" + "".join(f"{'-'*16}" for _ in s_metrics))
        for algo in means.index:
            lines.append(
                f"  {algo:<18}" +
                "".join(f"{means.loc[algo, m]:>16.4f}" for m in s_metrics)
            )

    lines.append(f"\n{sep}")
    lines.append("  END OF REPORT")
    lines.append(sep)
    return "\n".join(lines)


# ─── HTML report ───────────────────────────────────────────────────────────────

def build_html_report(all_df: pd.DataFrame, n_workers: int,
                      plots_dir: str, scaling_df: pd.DataFrame) -> str:

    rel_plots = os.path.relpath(plots_dir, RESULTS_DIR)

    css = """
<style>
  body  { font-family:'Segoe UI',Arial,sans-serif; margin:2em 3em;
          background:#f8f9fa; color:#212529; }
  h1    { color:#1a237e; border-bottom:3px solid #1a237e; padding-bottom:.3em; }
  h2    { color:#283593; margin-top:2em; }
  h3    { color:#37474f; }
  table { border-collapse:collapse; width:100%; margin:1em 0;
          font-size:.85em; background:white;
          box-shadow:0 1px 4px rgba(0,0,0,.12); }
  th    { background:#1a237e; color:white; padding:8px 10px;
          text-align:right; }
  th:first-child { text-align:left; }
  td    { padding:6px 10px; text-align:right;
          border-bottom:1px solid #e0e0e0; }
  td:first-child { text-align:left; font-weight:500; }
  tr:nth-child(even) { background:#f3f4f6; }
  tr:hover { background:#e8eaf6; }
  .cent { color:#1565c0; font-weight:bold; }
  .dist { color:#2e7d32; font-weight:bold; }
  .tie  { color:#888; }
  .box  { background:white; border-radius:6px; padding:1.5em;
          margin:1.5em 0; box-shadow:0 2px 6px rgba(0,0,0,.12); }
  .plots { display:flex; flex-wrap:wrap; gap:1em; margin:1em 0; }
  .plots figure { margin:0; width:calc(50% - .5em); min-width:380px; }
  .plots img    { width:100%; border:1px solid #ddd; border-radius:4px; }
  figcaption    { text-align:center; font-size:.8em; color:#555;
                  margin-top:.3em; }
  .meta  { font-size:.8em; color:#666; margin-top:.5em; }
  .note  { background:#e8f5e9; border-left:4px solid #43a047;
           padding:.6em 1em; margin:1em 0; border-radius:3px;
           font-size:.9em; }
</style>"""

    def _cls(better):
        return {"CENT": "cent", "DIST": "dist", "TIE": "tie"}.get(better, "")

    ts  = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    out = [f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Clustering Comparison Report</title>
  {css}
</head>
<body>
<h1>Centralized vs Distributed Clustering — Comparison Report</h1>
<p class="meta">Workers: {n_workers} &nbsp;|&nbsp; Generated: {ts} &nbsp;|&nbsp;
All values are the <strong>mean of 10 independent runs</strong>.</p>
<div class="note">
  Solid bars = centralized &nbsp;·&nbsp; Hatched bars = distributed.<br>
  Error bars / ± columns represent one standard deviation across runs.
</div>
"""]

    for dataset in all_df["dataset"].unique():
        df_ds   = all_df[all_df["dataset"] == dataset]
        k_val   = df_ds["k"].iloc[0]
        n_val   = df_ds["n"].iloc[0]
        present = [m for m in METRIC_COLS if m in df_ds.columns]
        means   = df_ds.groupby("algorithm")[present].mean()
        stds    = df_ds.groupby("algorithm")[present].std().fillna(0)

        out.append(f'<div class="box">')
        out.append(f'<h2>Dataset: {dataset} '
                   f'<small style="font-weight:normal">'
                   f'(n={n_val}, k={k_val})</small></h2>')

        # Runtime + inertia plots side by side
        out.append('<div class="plots">')
        for metric_name, caption in [("runtime_s",     "Runtime (s)"),
                                     ("intra_inertia", "Intra-cluster Inertia")]:
            img = f"{rel_plots}/{metric_name}_{dataset}.png"
            out.append(f'<figure><img src="{img}" alt="{caption}">'
                       f'<figcaption>{caption}</figcaption></figure>')
        out.append('</div>')

        # Per-pair comparison tables
        for cent, dist in PAIR_MAP.items():
            if cent not in means.index or dist not in means.index:
                continue
            c_row, c_std = means.loc[cent], stds.loc[cent]
            d_row, d_std = means.loc[dist], stds.loc[dist]

            out.append(f'<h3>{cent} &nbsp;vs&nbsp; {dist}</h3>')
            out.append('<table><tr>'
                       '<th>Metric</th>'
                       '<th>Cent mean</th><th>±std</th>'
                       '<th>Dist mean</th><th>±std</th>'
                       '<th>Δ%</th><th>Better</th></tr>')

            for m in present:
                cv, cs = c_row[m], c_std[m]
                dv, ds = d_row[m], d_std[m]
                dpct   = (dv - cv) / abs(cv) * 100 if cv != 0 else float("nan")
                if m in HIGHER_BETTER:
                    better = "CENT" if cv > dv else ("DIST" if dv > cv else "TIE")
                elif m in LOWER_BETTER:
                    better = "CENT" if cv < dv else ("DIST" if dv < cv else "TIE")
                else:
                    better = "—"
                dpct_str = f"{dpct:+.1f}%" if not np.isnan(dpct) else "—"
                cls = _cls(better)
                out.append(
                    f'<tr><td>{m}</td>'
                    f'<td>{cv:.4f}</td><td>{cs:.4f}</td>'
                    f'<td>{dv:.4f}</td><td>{ds:.4f}</td>'
                    f'<td class="{cls}">{dpct_str}</td>'
                    f'<td class="{cls}">{better}</td></tr>'
                )
            out.append('</table>')

        # Summary table
        s_metrics = [m for m in ["runtime_s", "intra_inertia", "nmi", "silhouette"]
                     if m in means.columns]
        out.append('<h3>Summary — all algorithms</h3><table>')
        out.append('<tr><th>Algorithm</th>' +
                   ''.join(f'<th>{m}</th>' for m in s_metrics) + '</tr>')
        for algo in means.index:
            out.append(
                '<tr><td>' + algo + '</td>' +
                ''.join(f'<td>{means.loc[algo, m]:.4f}</td>'
                        for m in s_metrics) +
                '</tr>'
            )
        out.append('</table></div>')

    # ── Worker-scaling section (runtime only) ──────────────────────────────────
    out.append('<div class="box">')
    out.append('<h2>Worker Scaling — Runtime '
               '<small style="font-weight:normal">'
               '(Distributed Algorithms, Breast Cancer k=2)</small></h2>')

    if scaling_df is not None and not scaling_df.empty:
        img = f"{rel_plots}/worker_scaling_runtime_s.png"
        out.append(f'<div class="plots">'
                   f'<figure style="width:100%;max-width:700px">'
                   f'<img src="{img}" alt="Runtime vs Worker Count">'
                   f'<figcaption>Runtime (s) vs. Number of Workers</figcaption>'
                   f'</figure></div>')

        # Summary table
        if "algorithm" in scaling_df.columns:
            grp = (scaling_df.groupby(["algorithm", "n_workers"])["runtime_s"]
                             .agg(["mean", "std"])
                             .reset_index())
            grp.columns = ["Algorithm", "Workers", "Mean Runtime (s)", "±std"]
            out.append('<h3>Mean runtime by algorithm and worker count</h3>')
            out.append('<table><tr>' +
                       ''.join(f'<th>{c}</th>' for c in grp.columns) +
                       '</tr>')
            for _, row in grp.iterrows():
                out.append('<tr>' +
                           ''.join(
                               f'<td>{v:.4f}</td>'
                               if isinstance(v, float) else f'<td>{v}</td>'
                               for v in row
                           ) + '</tr>')
            out.append('</table>')
    else:
        out.append('<p style="color:#c62828">Worker-scaling data unavailable — '
                   'see console output for details.</p>')

    out.append('</div>')
    out.append('</body></html>')
    return "\n".join(out)


# ─── Utility ───────────────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame):
    present = [m for m in ["runtime_s", "intra_inertia", "nmi", "silhouette"]
               if m in df.columns]
    print(df.groupby(["mode", "algorithm"])[present].mean().round(4).to_string())


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coil-dir",  default=None)
    parser.add_argument("--iris-csv",  default=None)
    parser.add_argument("--mnist-csv", default=None)
    parser.add_argument("--quick",     action="store_true",
                        help="3 runs + small COIL subset for fast testing")
    parser.add_argument("--skip-coil", action="store_true")
    parser.add_argument("--runs",      type=int, default=10,
                        help="Independent seeds per algorithm (default: 10)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    n_runs   = 3 if args.quick else args.runs
    all_dfs  = []
    df_scale = pd.DataFrame()

    # ── Experiment 0: Iris ────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("EXP 0 — Iris  k=3")
    print("=" * 52)
    try:
        X_iris, y_iris = load_iris_data(csv_path=args.iris_csv)
        print(f"Loaded: shape={X_iris.shape}")
        df0 = run_experiment("iris", X_iris, y_iris, k=3,
                             n_runs=n_runs, algorithms=ALL_ALGOS)
        df0.to_csv(os.path.join(RESULTS_DIR, "baseline_iris.csv"), index=False)
        _print_summary(df0)
        plot_metric_per_algo(df0, "runtime_s",     "Runtime (seconds)",
                             "Mean Runtime",  "iris", PLOTS_DIR)
        plot_metric_per_algo(df0, "intra_inertia", "Intra-cluster Inertia",
                             "Mean Inertia",  "iris", PLOTS_DIR)
        all_dfs.append(df0)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")

    # ── Experiment 1: Breast Cancer ───────────────────────────────────────────
    print("\n" + "=" * 52)
    print("EXP 1 — Breast Cancer Wisconsin  k=2")
    print("=" * 52)
    X_bc, y_bc = load_breast_cancer_data()
    print(f"Loaded: shape={X_bc.shape}")
    df1 = run_experiment("breast_cancer", X_bc, y_bc, k=2,
                         n_runs=n_runs, algorithms=ALL_ALGOS)
    df1.to_csv(os.path.join(RESULTS_DIR, "baseline_breast_cancer.csv"), index=False)
    _print_summary(df1)
    plot_metric_per_algo(df1, "runtime_s",     "Runtime (seconds)",
                         "Mean Runtime",  "breast_cancer", PLOTS_DIR)
    plot_metric_per_algo(df1, "intra_inertia", "Intra-cluster Inertia",
                         "Mean Inertia",  "breast_cancer", PLOTS_DIR)
    all_dfs.append(df1)

    # ── Worker-scaling experiment ─────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("WORKER SCALING TEST — Breast Cancer  k=2")
    print(f"  Worker counts : {WORKER_COUNTS}  ×  {n_runs} seeds each")
    print("=" * 52)
    df_scale = run_worker_scaling(X_bc, y_bc, k=2, n_runs=n_runs,
                                  worker_counts=WORKER_COUNTS)

    if not df_scale.empty:
        df_scale.to_csv(os.path.join(RESULTS_DIR, "worker_scaling.csv"),
                        index=False)
        print(f"\n  worker_scaling.csv saved  "
              f"({len(df_scale)} rows, algorithms: "
              f"{df_scale['algorithm'].unique().tolist()})")
        print("\n  Generating worker-scaling plot …")
        plot_worker_scaling(df_scale, PLOTS_DIR)
    else:
        print("\n  [WARN] Worker-scaling produced no data — plot skipped.")
        print("         Ensure DistributedXxx classes expose the worker count via")
        print("         n_workers= kwarg, .n_workers attribute, or .N_WORKERS.")

    # ── Experiment 2: COIL-100 ────────────────────────────────────────────────
    if not args.skip_coil:
        print("\n" + "=" * 52)
        print("EXP 2 — COIL-100  (varying n)")
        print("=" * 52)
        image_counts = (
            [720, 1440] if args.quick
            else [720, 1440, 2160, 2880, 3600, 4320, 5000]
        )
        coil_dfs = []
        for n_img in image_counts:
            k = max(2, n_img // 72)
            print(f"\n  n_images={n_img}  k={k}")
            X_c, y_c = load_coil100_data(max_images=n_img,
                                         data_dir=args.coil_dir)
            df_c = run_experiment(
                f"coil100_n{n_img}", X_c, y_c, k=k,
                n_runs=n_runs, algorithms=ALL_ALGOS,
            )
            df_c["n_images_target"] = n_img
            coil_dfs.append(df_c)

        df2 = pd.concat(coil_dfs, ignore_index=True)
        df2.to_csv(os.path.join(RESULTS_DIR, "baseline_coil100.csv"), index=False)
        _print_summary(df2)
        df2_last = df2[df2["n_images_target"] == image_counts[-1]]
        plot_metric_per_algo(df2_last, "runtime_s",
                             "Runtime (seconds)",
                             "Mean Runtime",  "coil100", PLOTS_DIR)
        plot_metric_per_algo(df2_last, "intra_inertia",
                             "Intra-cluster Inertia",
                             "Mean Inertia",  "coil100", PLOTS_DIR)
        all_dfs.append(df2)
    else:
        print("\n[SKIP] COIL-100 (--skip-coil)")

    # ── Save combined CSV + both reports ──────────────────────────────────────
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(os.path.join(RESULTS_DIR, "baseline_all.csv"), index=False)

        report   = build_report(combined, N_WORKERS)
        txt_path = os.path.join(RESULTS_DIR, "comparison_report.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report)
        print("\n" + report)

        html_report = build_html_report(combined, N_WORKERS, PLOTS_DIR, df_scale)
        html_path   = os.path.join(RESULTS_DIR, "comparison_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_report)

        print(f"\n{'='*52}")
        print(f"Results  →  {os.path.abspath(RESULTS_DIR)}/")
        for fname in [
            "baseline_iris.csv", "baseline_breast_cancer.csv",
            *(["baseline_coil100.csv"] if not args.skip_coil else []),
            "baseline_all.csv", "worker_scaling.csv",
            "comparison_report.txt", "comparison_report.html",
        ]:
            path = os.path.join(RESULTS_DIR, fname)
            mark = "✔" if os.path.exists(path) else "✘"
            print(f"  {mark}  {fname}")

        print(f"\nPlots    →  {os.path.abspath(PLOTS_DIR)}/")
        if os.path.isdir(PLOTS_DIR):
            for fname in sorted(f for f in os.listdir(PLOTS_DIR)
                                if f.endswith(".png")):
                print(f"  ✔  {fname}")


if __name__ == "__main__":
    main()