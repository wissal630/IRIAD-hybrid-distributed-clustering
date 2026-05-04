"""
Baseline experiment — centralized + distributed comparison.
Datasets:
  - Iris:             k=3
  - Breast Cancer:    k=2
  - MNIST:            k=10  (n=2000, PCA=30)
  - COIL-100:         k=n//72, varying n
Output (results/):
  baseline_iris.csv, baseline_breast_cancer.csv,
  baseline_mnist.csv, baseline_coil100.csv,
  baseline_all.csv, worker_scaling.csv,
  comparison_report.txt, report.html
Output (plots/):
  runtime_s_iris.png, intra_inertia_iris.png,
  nmi_iris.png, silhouette_iris.png,
  runtime_s_breast_cancer.png, intra_inertia_breast_cancer.png,
  nmi_breast_cancer.png, silhouette_breast_cancer.png,
  runtime_s_mnist.png, intra_inertia_mnist.png,
  nmi_mnist.png, silhouette_mnist.png,
  worker_scaling_runtime_s.png
Usage:
    python experiments/run_baseline.py
    python experiments/run_baseline.py --quick --skip-coil
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
# ── Execution log ─────────────────────────────────────────────────────────────
_LOG_LINES = []
def _log(msg: str):
    ts   = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    _LOG_LINES.append(line)
# ── Visual identity ───────────────────────────────────────────────────────────
ALGO_COLORS = {
    "KMeans":        "#4C72B0", "KMeans++":      "#DD8452",
    "PAM":           "#55A868", "KMM":           "#C44E52",
    "KMM++":         "#8172B2",
    "Dist-KMeans":   "#4C72B0", "Dist-KMeans++": "#DD8452",
    "Dist-PAM":      "#55A868", "Dist-KMM":      "#C44E52",
    "Dist-KMM++":    "#8172B2",
}
ALGO_HATCH = {
    "KMeans": "", "KMeans++": "", "PAM": "", "KMM": "", "KMM++": "",
    "Dist-KMeans": "///", "Dist-KMeans++": "///",
    "Dist-PAM": "///", "Dist-KMM": "///", "Dist-KMM++": "///",
}
CENTRALIZED_ALGOS = {
    "KMeans": KMeans, "KMeans++": KMeansPlusPlus,
    "PAM": KMedoids, "KMM": KMM, "KMM++": KMMPlusPlus,
}
DISTRIBUTED_ALGOS = {
    "Dist-KMeans":   DistributedKMeans,
    "Dist-KMeans++": DistributedKMeansPlusPlus,
    "Dist-PAM":      DistributedKMedoids,
    "Dist-KMM":      DistributedKMM,
    "Dist-KMM++":    DistributedKMMPlusPlus,
}
PAIR_MAP = {
    "KMeans": "Dist-KMeans", "KMeans++": "Dist-KMeans++",
    "PAM": "Dist-PAM", "KMM": "Dist-KMM", "KMM++": "Dist-KMM++",
}
ALL_ALGOS = {**CENTRALIZED_ALGOS, **DISTRIBUTED_ALGOS}
METRIC_COLS = [
    "intra_inertia", "inter_inertia", "silhouette",
    "davies_bouldin", "calinski_harabasz",
    "nmi", "ari", "purity", "runtime_s", "n_iter",
]
HIGHER_BETTER = {"silhouette", "calinski_harabasz", "nmi", "ari",
                 "purity", "inter_inertia"}
LOWER_BETTER  = {"intra_inertia", "davies_bouldin", "runtime_s", "n_iter"}
# ─── Metric helpers ───────────────────────────────────────────────────────────
def inter_inertia(X, labels, centers):
    mu = X.mean(axis=0)
    return float(sum(
        (labels == c).sum() * np.linalg.norm(centers[c] - mu) ** 2
        for c in range(len(centers))
    ))
def run_once(AlgoClass, X, y, k, seed):
    algo = AlgoClass(k=k, max_iter=100, random_state=seed)
    t0   = time.perf_counter()
    algo.fit(X)
    elapsed = time.perf_counter() - t0
    lbl, ctr = algo.labels_, algo.centers_
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
# ─── Experiment runner ────────────────────────────────────────────────────────
def run_experiment(name, X, y, k, n_runs, algorithms):
    rows = []
    for algo_name, AlgoClass in algorithms.items():
        mode = "distributed" if algo_name.startswith("Dist-") else "centralized"
        _log(f"  Running [{mode}] {algo_name} on {name} × {n_runs} seeds ...")
        algo_rows = []
        for seed in range(n_runs):
            try:
                r = run_once(AlgoClass, X, y, k, seed)
                r.update({
                    "algorithm": algo_name, "mode": mode,
                    "dataset": name, "k": k, "n": len(X), "seed": seed,
                })
                algo_rows.append(r)
            except Exception as e:
                print(f"\n    [WARN] seed={seed}: {e}")
        rows.extend(algo_rows)
        if algo_rows:
            mean_rt = np.mean([r["runtime_s"] for r in algo_rows])
            print(f"    ok  (mean {mean_rt:.4f}s, {len(algo_rows)}/{n_runs} seeds)")
        else:
            print(f"    FAILED (0 seeds succeeded)")
    return pd.DataFrame(rows)
# ─── Worker scaling ───────────────────────────────────────────────────────────
def run_worker_scaling(X, y, k, n_runs, dataset_name="dataset", worker_counts=WORKER_COUNTS):
    """Run distributed algos with different worker counts on a given dataset."""
    import algorithms.distributed as dist_module
    rows = []
    orig = dist_module.N_WORKERS
    for nw in worker_counts:
        print(f"\n  --- Workers = {nw} ---")
        dist_module.N_WORKERS = nw
        for algo_name, AlgoClass in DISTRIBUTED_ALGOS.items():
            print(f"    {algo_name} × {n_runs} seeds ...", end=" ", flush=True)
            algo_rows = []
            for seed in range(n_runs):
                try:
                    r = run_once(AlgoClass, X, y, k, seed)
                    r.update({
                        "algorithm": algo_name, "n_workers": nw,
                        "seed": seed,
                    })
                    algo_rows.append(r)
                except Exception as e:
                    print(f"\n      [WARN] nw={nw} seed={seed}: {e}")
            rows.extend(algo_rows)
            if algo_rows:
                mean_rt = np.mean([r["runtime_s"] for r in algo_rows])
                print(f"ok  ({mean_rt:.4f}s mean)")
            else:
                print("FAILED")
    dist_module.N_WORKERS = orig
    return pd.DataFrame(rows)
# ─── Matplotlib plots ─────────────────────────────────────────────────────────
def plot_metric_per_algo(df, metric, ylabel, title_prefix, dataset_label, plots_dir):
    if metric not in df.columns:
        return
    agg        = df.groupby("algorithm")[metric].agg(["mean", "std"])
    algo_order = list(ALL_ALGOS.keys())
    algos      = [a for a in algo_order if a in agg.index]
    if not algos:
        return
    values = [agg.loc[a, "mean"] for a in algos]
    errors = [agg.loc[a, "std"]  if not np.isnan(agg.loc[a, "std"]) else 0 for a in algos]
    colors = [ALGO_COLORS.get(a, "#888") for a in algos]
    hatch  = [ALGO_HATCH.get(a, "")     for a in algos]
    fig, ax = plt.subplots(figsize=(12, 7))
    x    = np.arange(len(algos))
    bars = ax.bar(
        x, values, yerr=errors, capsize=5,
        color=colors, edgecolor="black", linewidth=0.8,
        error_kw={"elinewidth": 1.2, "ecolor": "black"},
    )
    for bar, h in zip(bars, hatch):
        bar.set_hatch(h)
    ax.set_ylim(0, max(v + e for v, e in zip(values, errors)) * 1.18)
    for bar, val, err in zip(bars, values, errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + err + ax.get_ylim()[1] * 0.012,
            f"{val:.4g}", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#222",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f"{title_prefix} — {dataset_label}  (mean ± std, {df['seed'].nunique()} runs)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4g"))
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.legend(handles=[
        Patch(facecolor="white", edgecolor="black", label="Centralized (solid)"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="Distributed (hatched)"),
    ], fontsize=9, loc="upper right")
    fig.tight_layout()
    path = os.path.join(plots_dir, f"{metric}_{dataset_label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✔ plot → {os.path.relpath(path)}")
def plot_worker_scaling(scaling_df, plots_dir, dataset_name="dataset", k=None):
    if scaling_df is None or scaling_df.empty:
        return
    if not all(c in scaling_df.columns for c in ("algorithm", "n_workers", "runtime_s")):
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    colors_cycle = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    plotted = 0
    for i, algo in enumerate(DISTRIBUTED_ALGOS.keys()):
        sub = scaling_df[scaling_df["algorithm"] == algo]
        if sub.empty:
            continue
        grp = sub.groupby("n_workers")["runtime_s"].agg(["mean", "std"]).reset_index()
        grp["std"] = grp["std"].fillna(0)
        ax.errorbar(
            grp["n_workers"], grp["mean"], yerr=grp["std"],
            marker="o", linewidth=2, capsize=5,
            label=algo, color=colors_cycle[i % len(colors_cycle)],
        )
        plotted += 1
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Number of Workers", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    k_str = f", k={k}" if k is not None else ""
    ax.set_title(f"Effect of Worker Count on Runtime ({dataset_name}{k_str})",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(sorted(scaling_df["n_workers"].unique()))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4g"))
    ax.legend(fontsize=9, loc="best", title="Algorithm", title_fontsize=9)
    ax.grid(linestyle="--", alpha=0.45)
    fig.tight_layout()
    path = os.path.join(plots_dir, f"worker_scaling_runtime_s_{dataset_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✔ plot → {os.path.relpath(path)}")
# ─── TXT report ──────────────────────────────────────────────────────────────
def build_report(all_df: pd.DataFrame, n_workers: int) -> str:
    lines = []
    sep   = "=" * 72
    lines.append(sep)
    lines.append("  CENTRALIZED vs DISTRIBUTED CLUSTERING — COMPARISON REPORT")
    lines.append(f"  Workers: {n_workers}  |  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)
    for dataset in all_df["dataset"].unique():
        df_ds   = all_df[all_df["dataset"] == dataset]
        k_val   = df_ds["k"].iloc[0]
        n_val   = df_ds["n"].iloc[0]
        present = [m for m in METRIC_COLS if m in df_ds.columns]
        means   = df_ds.groupby("algorithm")[present].mean()
        lines.append(f"\n{'─'*72}")
        lines.append(f"  DATASET : {dataset}   (n={n_val}, k={k_val})")
        lines.append(f"{'─'*72}")
        for cent, dist in PAIR_MAP.items():
            if cent not in means.index or dist not in means.index:
                continue
            c_row, d_row = means.loc[cent], means.loc[dist]
            lines.append(f"\n  {cent:12s}  vs  {dist}")
            lines.append(f"  {'Metric':<22} {'Centralized':>14} {'Distributed':>14} {'Delta':>12} {'Better':>8}")
            lines.append(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*12} {'-'*8}")
            for m in present:
                cv, dv = c_row[m], d_row[m]
                dpct   = (dv - cv) / abs(cv) * 100 if cv != 0 else float("nan")
                if m in HIGHER_BETTER:
                    better = "CENT" if cv > dv else ("DIST" if dv > cv else "TIE")
                elif m in LOWER_BETTER:
                    better = "CENT" if cv < dv else ("DIST" if dv < cv else "TIE")
                else:
                    better = "—"
                dpct_str = f"{dpct:+.1f}%" if not np.isnan(dpct) else "—"
                lines.append(f"  {m:<22} {cv:>14.4f} {dv:>14.4f} {dpct_str:>12} {better:>8}")
        s_metrics = [m for m in ["runtime_s", "intra_inertia", "nmi", "silhouette"] if m in means.columns]
        lines.append(f"\n  {'─'*68}")
        lines.append(f"  SUMMARY — all algorithms on {dataset}")
        lines.append(f"  {'─'*68}")
        lines.append(f"  {'Algorithm':<18}" + "".join(f"{m:>16}" for m in s_metrics))
        lines.append(f"  {'-'*18}" + "".join(f"{'-'*16}" for _ in s_metrics))
        for algo in means.index:
            lines.append(f"  {algo:<18}" + "".join(f"{means.loc[algo, m]:>16.4f}" for m in s_metrics))
    lines.append(f"\n{sep}\n  END OF REPORT\n{sep}")
    return "\n".join(lines)
# ─── HTML report ─────────────────────────────────────────────────────────────
def build_html_report(all_df: pd.DataFrame, n_workers: int,
                      log_lines: list, scaling_df: pd.DataFrame,
                      plots_dir: str) -> str:
    METRIC_LABELS = {
        "intra_inertia": "Intra-cluster Inertia ↓",
        "inter_inertia": "Inter-cluster Inertia ↑",
        "silhouette":    "Silhouette ↑",
        "davies_bouldin":    "Davies-Bouldin ↓",
        "calinski_harabasz": "Calinski-Harabasz ↑",
        "nmi": "NMI ↑", "ari": "ARI ↑", "purity": "Purity ↑",
        "runtime_s": "Runtime (s) ↓", "n_iter": "Iterations ↓",
    }
    KPI_METRICS = ["nmi", "silhouette", "intra_inertia", "runtime_s"]
    KPI_LABELS  = {"nmi": "NMI", "silhouette": "Silhouette",
                   "intra_inertia": "Intra-Inertia", "runtime_s": "Runtime (s)"}
    datasets   = all_df["dataset"].unique().tolist()
    ts         = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    rel_plots  = os.path.relpath(plots_dir, RESULTS_DIR).replace("\\", "/")

    # ── KPI cards ─────────────────────────────────────────────────────────────
    def kpi_cards(df_ds):
        def best(mode, metric):
            sub = df_ds[df_ds["mode"] == mode]
            if sub.empty or metric not in sub.columns:
                return "—", "—"
            g    = sub.groupby("algorithm")[metric].mean()
            algo = g.idxmin() if metric in LOWER_BETTER else g.idxmax()
            return algo, f"{g[algo]:.4f}"
        html = '<div class="kpi-row">'
        for m in KPI_METRICS:
            if m not in df_ds.columns:
                continue
            c_algo, c_val = best("centralized", m)
            d_algo, d_val = best("distributed", m)
            try:
                cent_wins = (
                    (m in LOWER_BETTER  and float(c_val) <= float(d_val)) or
                    (m in HIGHER_BETTER and float(c_val) >= float(d_val))
                )
            except ValueError:
                cent_wins = True
            html += f"""
            <div class="kpi-card">
              <div class="kpi-label">{KPI_LABELS.get(m, m)}</div>
              <div class="kpi-row-inner">
                <div class="kpi-half {"kpi-winner" if cent_wins else ""}">
                  <span class="kpi-mode cent-text">Centralized</span>
                  <span class="kpi-val">{c_val}</span>
                  <span class="kpi-algo">{c_algo}</span>
                </div>
                <div class="kpi-half {"kpi-winner" if not cent_wins else ""}">
                  <span class="kpi-mode dist-text">Distributed</span>
                  <span class="kpi-val">{d_val}</span>
                  <span class="kpi-algo">{d_algo}</span>
                </div>
              </div>
            </div>"""
        return html + '</div>'

    # ── Per-dataset panel ─────────────────────────────────────────────────────
    def dataset_panel(dataset):
        df_ds   = all_df[all_df["dataset"] == dataset]
        k_val   = int(df_ds["k"].iloc[0])
        n_val   = int(df_ds["n"].iloc[0])
        present = [m for m in METRIC_COLS if m in df_ds.columns]
        stats   = df_ds.groupby("algorithm")[present].agg(["mean", "std"])

        # All 4 matplotlib image plots, 2 per row
        img_html = '<div class="plots">'
        for metric, caption in [("runtime_s",     "Runtime (s)"),
                                 ("intra_inertia", "Intra-cluster Inertia"),
                                 ("nmi",           "NMI"),
                                 ("silhouette",    "Silhouette")]:
            img_path = f"{rel_plots}/{metric}_{dataset}.png"
            img_html += (f'<figure><img src="{img_path}" alt="{caption}">'
                         f'<figcaption>{caption}</figcaption></figure>')
        img_html += '</div>'

        # Comparison tables
        tables_html = ""
        for cent, dist in PAIR_MAP.items():
            if cent not in stats.index or dist not in stats.index:
                continue
            tables_html += f'<h3 class="pair-title">{cent} &nbsp;vs&nbsp; {dist}</h3>'
            tables_html += '''<table><tr>
              <th>Metric</th><th>Cent mean</th><th>±std</th>
              <th>Dist mean</th><th>±std</th><th>Δ%</th><th>Better</th></tr>'''
            for m in present:
                cv  = stats.loc[cent, (m, "mean")]
                cs  = stats.loc[cent, (m, "std")]
                dv  = stats.loc[dist, (m, "mean")]
                ds_ = stats.loc[dist, (m, "std")]
                dpct = (dv - cv) / abs(cv) * 100 if cv != 0 else float("nan")
                if m in HIGHER_BETTER:
                    better = "CENT" if cv > dv else ("DIST" if dv > cv else "TIE")
                elif m in LOWER_BETTER:
                    better = "CENT" if cv < dv else ("DIST" if dv < cv else "TIE")
                else:
                    better = "—"
                css      = "cent" if better == "CENT" else ("dist" if better == "DIST" else "tie")
                sign     = "+" if dpct > 0 else ""
                dpct_str = f"{sign}{dpct:.1f}%" if not np.isnan(dpct) else "—"
                tables_html += (
                    f'<tr><td>{METRIC_LABELS.get(m, m)}</td>'
                    f'<td>{cv:.4f}</td><td>{cs:.4f}</td>'
                    f'<td>{dv:.4f}</td><td>{ds_:.4f}</td>'
                    f'<td class="{css}">{dpct_str}</td>'
                    f'<td class="{css}">{better}</td></tr>'
                )
            tables_html += "</table>"

        # Summary table
        s_metrics    = [m for m in ["runtime_s", "intra_inertia", "nmi", "silhouette"]
                        if m in df_ds.columns]
        means_all    = df_ds.groupby("algorithm")[s_metrics].mean()
        summary_html = '<h3>Summary — all algorithms</h3><table><tr><th>Algorithm</th>'
        for m in s_metrics:
            summary_html += f"<th>{METRIC_LABELS.get(m, m)}</th>"
        summary_html += "</tr>"
        for algo in means_all.index:
            css = "dist-row" if algo.startswith("Dist-") else "cent-row"
            summary_html += f'<tr class="{css}"><td>{algo}</td>'
            for m in s_metrics:
                summary_html += f"<td>{means_all.loc[algo, m]:.4f}</td>"
            summary_html += "</tr>"
        summary_html += "</table>"

        return (
            f'<p class="ds-meta">n = {n_val} &nbsp;|&nbsp; k = {k_val} &nbsp;|&nbsp; '
            f'{len(df_ds["algorithm"].unique())} algorithms</p>'
            + kpi_cards(df_ds)
            + img_html
            + tables_html
            + summary_html
        )

    # ── Worker scaling panel ──────────────────────────────────────────────────
    def worker_panel():
        has_data = (scaling_df is not None and not scaling_df.empty
                    and "n_workers" in scaling_df.columns)
        if not has_data:
            return '<p style="color:#888;padding:2em">No worker scaling data available.</p>'

        # matplotlib image — full width
        ws_datasets = scaling_df["dataset"].unique() if "dataset" in scaling_df.columns else []
        img_html = '<div class="plots plots-full">'
        for ds in ws_datasets:
            img_path = f"{rel_plots}/worker_scaling_runtime_s_{ds}.png"
            img_html += (f'<figure><img src="{img_path}" alt="Worker Scaling {ds}">'
                         f'<figcaption>Runtime (s) vs Workers — {ds}</figcaption></figure>')
        img_html += '</div>'

        # Table
        grp = (scaling_df.groupby(["algorithm", "n_workers"])["runtime_s"]
                          .agg(["mean", "std"]).reset_index())
        table_html = '''<h3 style="margin:1em 0 .5em;color:#37474f">
            Mean runtime by algorithm and worker count</h3>
            <table><tr><th>Algorithm</th><th>Workers</th>
            <th>Mean Runtime (s)</th><th>±std</th></tr>'''
        for _, row in grp.iterrows():
            table_html += (f'<tr><td>{row["algorithm"]}</td>'
                           f'<td>{int(row["n_workers"])}</td>'
                           f'<td>{row["mean"]:.4f}</td>'
                           f'<td>{row["std"]:.4f}</td></tr>')
        table_html += "</table>"
        return img_html + table_html

    # ── Assemble tabs ─────────────────────────────────────────────────────────
    tab_buttons = ""
    tab_panels  = ""
    for i, ds in enumerate(datasets):
        active       = "active" if i == 0 else ""
        tab_buttons += (f'<button class="tab-btn {active}" '
                        f'onclick="showTab(\'{ds}\')" id="btn-{ds}">{ds}</button>')
        tab_panels  += (f'<div class="tab-panel {active}" id="panel-{ds}">'
                        f'{dataset_panel(ds)}</div>')
    tab_buttons += ('<button class="tab-btn" onclick="showTab(\'workers\')" '
                    'id="btn-workers">⚙ Worker Scaling</button>')
    tab_panels  += f'<div class="tab-panel" id="panel-workers">{worker_panel()}</div>'

    log_html = "".join(f'<div class="log-line">{l}</div>' for l in log_lines)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Clustering Comparison Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5;
          color: #212529; height: 100vh; display: flex; flex-direction: column; }}
  .header {{ background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
             color: white; padding: 1.5em 3em; flex-shrink: 0; }}
  .header h1 {{ font-size: 1.5em; font-weight: 700; margin-bottom: .3em; }}
  .header .meta {{ font-size: .82em; opacity: .85; margin-top: .4em; }}
  .badge {{ display: inline-block; background: rgba(255,255,255,.2);
            border-radius: 20px; padding: .2em .9em; margin-right: .5em;
            font-size: .78em; }}
  .main {{ display: flex; flex: 1; overflow: hidden; }}
  .sidebar {{ width: 190px; min-width: 190px; background: #1e2a4a;
              display: flex; flex-direction: column;
              overflow-y: auto; flex-shrink: 0; }}
  .nav-label {{ color: #90a4ae; font-size: .7em; font-weight: 700;
                text-transform: uppercase; letter-spacing: .1em;
                padding: 1em 1.2em .4em; }}
  .tab-btn {{ display: block; width: 100%; text-align: left;
              padding: .65em 1.2em; border: none; background: none;
              color: #b0bec5; cursor: pointer; font-size: .88em;
              transition: all .2s; border-left: 3px solid transparent; }}
  .tab-btn:hover {{ background: rgba(255,255,255,.07); color: white; }}
  .tab-btn.active {{ background: rgba(255,255,255,.12); color: white;
                     border-left-color: #64b5f6; font-weight: 600; }}
  .sidebar-footer {{ margin-top: auto; padding: 1em 1.2em;
                     border-top: 1px solid rgba(255,255,255,.08); }}
  .legend-item {{ display: flex; align-items: center; gap: .5em;
                  font-size: .8em; color: #b0bec5; margin: .35em 0; }}
  .legend-dot {{ width: 11px; height: 11px; border-radius: 3px; flex-shrink: 0; }}
  .content {{ flex: 1; overflow-y: auto; padding: 1.8em 2em; }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}
  .ds-meta {{ color: #666; font-size: .85em; margin-bottom: 1.2em; }}
  /* KPI */
  .kpi-row {{ display: flex; flex-wrap: wrap; gap: .8em; margin-bottom: 1.5em; }}
  .kpi-card {{ background: white; border-radius: 8px; padding: .9em 1.1em;
               box-shadow: 0 2px 8px rgba(0,0,0,.08); flex: 1; min-width: 175px; }}
  .kpi-label {{ font-size: .72em; color: #888; font-weight: 700;
                text-transform: uppercase; letter-spacing: .05em; margin-bottom: .5em; }}
  .kpi-row-inner {{ display: flex; gap: .5em; }}
  .kpi-half {{ flex: 1; padding: .45em .5em; border-radius: 5px;
               background: #f8f9fa; text-align: center; }}
  .kpi-winner {{ background: #e8f5e9; border: 1px solid #a5d6a7; }}
  .kpi-mode {{ display: block; font-size: .65em; font-weight: 700;
               text-transform: uppercase; margin-bottom: .15em; }}
  .kpi-val  {{ display: block; font-size: 1.05em; font-weight: 700; color: #212529; }}
  .kpi-algo {{ display: block; font-size: .65em; color: #888; margin-top: .1em;
               white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .cent-text {{ color: #1565c0; }}
  .dist-text {{ color: #2e7d32; }}
  /* Matplotlib images — 2 per row, full-width variant */
  .plots {{ display: flex; flex-wrap: wrap; gap: 1em; margin: 1em 0 1.5em; }}
  .plots figure {{ margin: 0; width: calc(50% - .5em); min-width: 380px; }}
  .plots.plots-full figure {{ width: 100%; min-width: unset; }}
  .plots img {{ width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  figcaption {{ text-align: center; font-size: .8em; color: #555; margin-top: .3em; }}
  /* Tables */
  .pair-title {{ color: #283593; margin: 1.5em 0 .5em; padding-bottom: .3em;
                 border-bottom: 2px solid #e8eaf6; font-size: 1em; }}
  table {{ border-collapse: collapse; width: 100%; margin: .5em 0 1.5em;
           font-size: .85em; background: white;
           box-shadow: 0 1px 4px rgba(0,0,0,.12); }}
  th {{ background: #1a237e; color: white; padding: 8px 10px; text-align: right; }}
  th:first-child {{ text-align: left; }}
  td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid #e0e0e0; }}
  td:first-child {{ text-align: left; font-weight: 500; }}
  tr:nth-child(even) td {{ background: #f3f4f6; }}
  tr:hover td {{ background: #e8eaf6; }}
  tr:last-child td {{ border-bottom: none; }}
  .cent {{ color: #1565c0; font-weight: bold; }}
  .dist {{ color: #2e7d32; font-weight: bold; }}
  .tie  {{ color: #888; }}
  .cent-row td:first-child {{ color: #1565c0; }}
  .dist-row td:first-child {{ color: #2e7d32; font-style: italic; }}
  /* Note */
  .note {{ background: #e8f5e9; border-left: 4px solid #43a047;
           padding: .6em 1em; margin: 0 0 1.2em; border-radius: 3px;
           font-size: .85em; color: #2e7d32; }}
  /* Log */
  .log-section {{ background: #1e2a2e; color: #a5d6a7; border-radius: 8px;
                  padding: 1.2em 1.5em; font-family: monospace; font-size: .78em;
                  line-height: 1.7; margin-top: 2em; max-height: 260px; overflow-y: auto; }}
  .log-line {{ border-bottom: 1px solid rgba(255,255,255,.04); padding: .1em 0; }}
  .log-line:last-child {{ border-bottom: none; }}
</style>
</head>
<body>
<div class="header">
  <h1>Centralized vs Distributed Clustering — Comparison Report</h1>
  <div class="meta">
    <span class="badge">⚙ Workers: {n_workers}</span>
    <span class="badge">📅 {ts}</span>
    <span class="badge">📊 {len(datasets)} datasets</span>
    <span class="badge">Mean of 10 independent runs</span>
  </div>
</div>
<div class="main">
  <nav class="sidebar">
    <div class="nav-label">Datasets</div>
    {tab_buttons}
    <div class="sidebar-footer">
      <div class="nav-label" style="padding:0 0 .5em">Legend</div>
      <div class="legend-item">
        <div class="legend-dot" style="background:#4C72B0"></div>Centralized
      </div>
      <div class="legend-item">
        <div class="legend-dot" style="background:#4C72B0;opacity:.5;
             background: repeating-linear-gradient(45deg,#4C72B0 0,#4C72B0 2px,
             transparent 0,transparent 50%) center/6px 6px"></div>Distributed
      </div>
      <div style="font-size:.72em;color:#78909c;margin-top:.8em;line-height:1.5">
        Solid = centralized<br>Hatched = distributed
      </div>
    </div>
  </nav>
  <div class="content">
    <div class="note">
      Solid bars / CENT = centralized &nbsp;·&nbsp;
      Hatched bars / DIST = distributed &nbsp;·&nbsp;
      Highlighted KPI = better mode &nbsp;·&nbsp;
      Error bars = ±1 std across runs.
    </div>
    {tab_panels}
    <div class="log-section">
      <div style="color:#80cbc4;font-weight:700;margin-bottom:.6em">▶ Execution Log</div>
      {log_html}
    </div>
  </div>
</div>
<script>
function showTab(name) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  document.getElementById('btn-' + name).classList.add('active');
}}
</script>
</body>
</html>"""
# ─── Utility ─────────────────────────────────────────────────────────────────
def _print_summary(df: pd.DataFrame):
    present = [m for m in ["runtime_s", "intra_inertia", "nmi", "silhouette"]
               if m in df.columns]
    print(df.groupby(["mode", "algorithm"])[present].mean().round(4).to_string())
# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coil-dir",  default=None)
    parser.add_argument("--iris-csv",  default=None)
    parser.add_argument("--mnist-csv", default=None)
    parser.add_argument("--quick",     action="store_true")
    parser.add_argument("--skip-coil", action="store_true")
    parser.add_argument("--runs",      type=int, default=10)
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)
    n_runs     = 3 if args.quick else args.runs
    all_dfs    = []
    scaling_df = pd.DataFrame()
    # ── Experiment 0: Iris ────────────────────────────────────────────────────
    _log("EXP 0 — Iris k=3 — starting")
    try:
        X_iris, y_iris = load_iris_data(csv_path=args.iris_csv)
        _log(f"Iris loaded: shape={X_iris.shape}")
        df0 = run_experiment("iris", X_iris, y_iris, k=3,
                             n_runs=n_runs, algorithms=ALL_ALGOS)
        df0.to_csv(os.path.join(RESULTS_DIR, "baseline_iris.csv"), index=False)
        _print_summary(df0)
        plot_metric_per_algo(df0, "runtime_s",     "Runtime (seconds)",
                             "Mean Runtime",    "iris", PLOTS_DIR)
        plot_metric_per_algo(df0, "intra_inertia", "Intra-cluster Inertia",
                             "Intra-Inertia",   "iris", PLOTS_DIR)
        plot_metric_per_algo(df0, "nmi",           "NMI",
                             "NMI",             "iris", PLOTS_DIR)
        plot_metric_per_algo(df0, "silhouette",    "Silhouette",
                             "Silhouette",      "iris", PLOTS_DIR)
        all_dfs.append(df0)
        _log(f"EXP 0 — Iris done ({len(df0)} rows)")
    except FileNotFoundError as e:
        _log(f"  [SKIP] Iris: {e}")
    # ── Experiment 1: Breast Cancer ───────────────────────────────────────────
    _log("EXP 1 — Breast Cancer k=2 — starting")
    X_bc, y_bc = load_breast_cancer_data()
    _log(f"Breast Cancer loaded: shape={X_bc.shape}")
    df1 = run_experiment("breast_cancer", X_bc, y_bc, k=2,
                         n_runs=n_runs, algorithms=ALL_ALGOS)
    df1.to_csv(os.path.join(RESULTS_DIR, "baseline_breast_cancer.csv"), index=False)
    _print_summary(df1)
    plot_metric_per_algo(df1, "runtime_s",     "Runtime (seconds)",
                         "Mean Runtime",    "breast_cancer", PLOTS_DIR)
    plot_metric_per_algo(df1, "intra_inertia", "Intra-cluster Inertia",
                         "Intra-Inertia",   "breast_cancer", PLOTS_DIR)
    plot_metric_per_algo(df1, "nmi",           "NMI",
                         "NMI",             "breast_cancer", PLOTS_DIR)
    plot_metric_per_algo(df1, "silhouette",    "Silhouette",
                         "Silhouette",      "breast_cancer", PLOTS_DIR)
    all_dfs.append(df1)
    _log(f"EXP 1 — Breast Cancer done ({len(df1)} rows)")
    # ── Worker scaling ────────────────────────────────────────────────────────
    # ── Worker scaling — Iris + Breast Cancer + MNIST (if available) ──────────
    _log(f"Worker scaling — Iris + Breast Cancer + MNIST, workers={WORKER_COUNTS}")
    scaling_datasets = []

    # Iris (already loaded above into X_iris / y_iris, wrapped in try)
    try:
        _log("  Worker scaling: Iris k=3")
        df_ws_iris = run_worker_scaling(X_iris, y_iris, k=3, n_runs=n_runs,
                                        dataset_name="iris",
                                        worker_counts=WORKER_COUNTS)
        if not df_ws_iris.empty:
            df_ws_iris["dataset"] = "iris"
            scaling_datasets.append(df_ws_iris)
            plot_worker_scaling(df_ws_iris, PLOTS_DIR, dataset_name="iris", k=3)
    except NameError:
        _log("  [SKIP] Iris not loaded — skipping worker scaling for iris")

    # Breast Cancer (always loaded)
    _log("  Worker scaling: Breast Cancer k=2")
    df_ws_bc = run_worker_scaling(X_bc, y_bc, k=2, n_runs=n_runs,
                                  dataset_name="breast_cancer",
                                  worker_counts=WORKER_COUNTS)
    if not df_ws_bc.empty:
        df_ws_bc["dataset"] = "breast_cancer"
        scaling_datasets.append(df_ws_bc)
        plot_worker_scaling(df_ws_bc, PLOTS_DIR, dataset_name="breast_cancer", k=2)

    # MNIST (loaded later — handled after MNIST experiment block)
    _mnist_for_scaling = None  # filled in after EXP 1.5 below

    scaling_df = pd.concat(scaling_datasets, ignore_index=True) if scaling_datasets else pd.DataFrame()
    if not scaling_df.empty:
        scaling_df.to_csv(os.path.join(RESULTS_DIR, "worker_scaling.csv"), index=False)
    _log(f"Worker scaling (iris+bc) done")
    # ── Experiment 1.5: MNIST ────────────────────────────────────────────────
    _log("EXP 1.5 — MNIST k=10 — starting")
    try:
        X_mnist, y_mnist = load_mnist_data(n_samples=2000, n_components=30,
                                            csv_path=args.mnist_csv)
        _log(f"MNIST loaded: shape={X_mnist.shape}")
        mnist_algos = {
            "KMeans":        KMeans,
            "KMeans++":      KMeansPlusPlus,
            "Dist-KMeans":   DistributedKMeans,
            "Dist-KMeans++": DistributedKMeansPlusPlus,
            "Dist-PAM":      DistributedKMedoids,
            "Dist-KMM":      DistributedKMM,
            "Dist-KMM++":    DistributedKMMPlusPlus,
        }
        df_mnist = run_experiment("mnist", X_mnist, y_mnist, k=10,
                                  n_runs=n_runs, algorithms=mnist_algos)
        df_mnist.to_csv(os.path.join(RESULTS_DIR, "baseline_mnist.csv"), index=False)
        _print_summary(df_mnist)
        plot_metric_per_algo(df_mnist, "runtime_s",     "Runtime (seconds)",
                             "Mean Runtime",    "mnist", PLOTS_DIR)
        plot_metric_per_algo(df_mnist, "intra_inertia", "Intra-cluster Inertia",
                             "Intra-Inertia",   "mnist", PLOTS_DIR)
        plot_metric_per_algo(df_mnist, "nmi",           "NMI",
                             "NMI",             "mnist", PLOTS_DIR)
        plot_metric_per_algo(df_mnist, "silhouette",    "Silhouette",
                             "Silhouette",      "mnist", PLOTS_DIR)
        all_dfs.append(df_mnist)
        _log(f"EXP 1.5 — MNIST done ({len(df_mnist)} rows)")
        # Worker scaling for MNIST
        _log("  Worker scaling: MNIST k=10")
        df_ws_mnist = run_worker_scaling(X_mnist, y_mnist, k=10, n_runs=n_runs,
                                         dataset_name="mnist",
                                         worker_counts=WORKER_COUNTS)
        if not df_ws_mnist.empty:
            df_ws_mnist["dataset"] = "mnist"
            plot_worker_scaling(df_ws_mnist, PLOTS_DIR, dataset_name="mnist", k=10)
            if not scaling_df.empty:
                scaling_df = pd.concat([scaling_df, df_ws_mnist], ignore_index=True)
                scaling_df.to_csv(os.path.join(RESULTS_DIR, "worker_scaling.csv"), index=False)
            _log(f"  Worker scaling MNIST done ({len(df_ws_mnist)} rows)")
    except FileNotFoundError as e:
        _log(f"  [SKIP] MNIST: {e}")
    # ── Experiment 2: COIL-100 ────────────────────────────────────────────────
    if not args.skip_coil:
        _log("EXP 2 — COIL-100 — starting")
        image_counts = [720, 1440] if args.quick else [720,1440,2160,2880,3600,4320,5000]
        coil_dfs = []
        for n_img in image_counts:
            k = max(2, n_img // 72)
            _log(f"  COIL-100 n_images={n_img} k={k}")
            X_c, y_c = load_coil100_data(max_images=n_img, data_dir=args.coil_dir)
            df_c = run_experiment(f"coil100_n{n_img}", X_c, y_c, k=k,
                                  n_runs=n_runs, algorithms=ALL_ALGOS)
            df_c["n_images_target"] = n_img
            coil_dfs.append(df_c)
        df2 = pd.concat(coil_dfs, ignore_index=True)
        df2.to_csv(os.path.join(RESULTS_DIR, "baseline_coil100.csv"), index=False)
        _print_summary(df2)
        df2_last = df2[df2["n_images_target"] == image_counts[-1]]
        plot_metric_per_algo(df2_last, "runtime_s",     "Runtime (seconds)",
                             "Mean Runtime",  "coil100", PLOTS_DIR)
        plot_metric_per_algo(df2_last, "intra_inertia", "Intra-cluster Inertia",
                             "Intra-Inertia", "coil100", PLOTS_DIR)
        plot_metric_per_algo(df2_last, "nmi",           "NMI",
                             "NMI",           "coil100", PLOTS_DIR)
        plot_metric_per_algo(df2_last, "silhouette",    "Silhouette",
                             "Silhouette",    "coil100", PLOTS_DIR)
        all_dfs.append(df2)
        _log(f"EXP 2 — COIL-100 done ({len(df2)} rows)")
    else:
        _log("COIL-100 skipped (--skip-coil)")
    # ── Save everything ───────────────────────────────────────────────────────
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(os.path.join(RESULTS_DIR, "baseline_all.csv"), index=False)
        txt = build_report(combined, N_WORKERS)
        with open(os.path.join(RESULTS_DIR, "comparison_report.txt"),
                  "w", encoding="utf-8") as f:
            f.write(txt)
        html = build_html_report(combined, N_WORKERS, _LOG_LINES,
                                 scaling_df, PLOTS_DIR)
        html_path = os.path.join(RESULTS_DIR, "report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        _log("All done.")
        print(f"\nResults  → {os.path.abspath(RESULTS_DIR)}/")
        print(f"Plots    → {os.path.abspath(PLOTS_DIR)}/")
        print(f"Report   → {html_path}")
if __name__ == "__main__":
    main()