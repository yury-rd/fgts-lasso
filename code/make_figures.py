"""Generate paper figures from results/synthetic.json and results/power.json.

Produces:

  figures/synth_regret_grid.{pdf,png} -- 2x2 grid of cumulative-regret-vs-t
      curves with 95% CI bands, configs exp_50_5 / exp_100_5 / exp_200_5 /
      exp_500_5, all 5 algos, log-y, single legend.
  figures/power_regret.{pdf,png}      -- cumulative regret vs t with 95% CI
      for the IEEE 33-bus power-systems experiment.
  figures/power_voltage.{pdf,png}     -- per-step voltage-deviation
      trajectory (RMS / running mean) for the top-3 algos by final regret.
  figures/regret_vs_d.{pdf,png}       -- final regret vs d at fixed s=5,
      with error bars.

Also writes the results/tables_*.tex tables.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Plot styling
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "lines.linewidth": 1.4,
})

ALGO_COLORS = {
    "FGTSLasso": "#d62728",                # red (proposed)
    "LassoBandit": "#1f77b4",              # blue
    "SALassoBandit": "#2ca02c",            # green
    "ThresholdedLassoBandit": "#9467bd",   # purple
    "LinUCB": "#ff7f0e",                   # orange
}
ALGO_LABEL = {
    "FGTSLasso": "FGTS-Lasso (ours)",
    "LassoBandit": "Lasso Bandit (BB20)",
    "SALassoBandit": "SA-Lasso (OIZ21)",
    "ThresholdedLassoBandit": "Thresh-Lasso (AAP22)",
    "LinUCB": "LinUCB (CLRS11)",
}
ALGO_ORDER = ["FGTSLasso", "LassoBandit", "SALassoBandit",
              "ThresholdedLassoBandit", "LinUCB"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(path: Path) -> List[dict]:
    payload = json.loads(path.read_text())
    return payload["results"]


def _group_by_config_algo(results: List[dict]) -> Dict[str, Dict[str, List[dict]]]:
    out: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        out[r["config_tag"]][r["algo"]].append(r)
    return out


def _aggregate_traj(runs: List[dict], key: str = "regret_trajectory"):
    """Aggregate trajectories across seeds.

    Returns (steps, mean, lo, hi) where lo/hi are the 95% CI of the mean
    using the sample-std-over-sqrt(n).
    """
    arr = np.asarray([r[key] for r in runs], dtype=float)  # (n_seeds, n_steps)
    steps = np.asarray(runs[0]["regret_traj_steps"], dtype=int)
    mean = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else 0
    z = 1.96
    lo = mean - z * sem
    hi = mean + z * sem
    return steps, mean, lo, hi


# ---------------------------------------------------------------------------
# Synthetic 2x2 regret grid
# ---------------------------------------------------------------------------

def make_synth_grid(synth_path=RESULTS_DIR / "synthetic.json",
                    out_pdf=FIG_DIR / "synth_regret_grid.pdf",
                    out_png=FIG_DIR / "synth_regret_grid.png"):
    if not synth_path.exists():
        print(f"[synth_grid] missing {synth_path}, skipping")
        return
    data = _group_by_config_algo(_load(synth_path))
    panels = ["exp_50_5", "exp_100_5", "exp_200_5", "exp_500_5"]
    panels = [p for p in panels if p in data]
    if not panels:
        # Fall back to whatever s=5 configs are present.
        panels = sorted(c for c in data if c.endswith("_5"))[:4]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), sharex=True)
    axes = axes.flatten()
    for ax_i, tag in enumerate(panels):
        ax = axes[ax_i]
        for algo in ALGO_ORDER:
            if algo not in data[tag]:
                continue
            runs = data[tag][algo]
            steps, mean, lo, hi = _aggregate_traj(runs)
            ax.plot(steps, mean, color=ALGO_COLORS[algo], label=ALGO_LABEL[algo])
            ax.fill_between(steps, np.maximum(lo, 1e-3), hi,
                            color=ALGO_COLORS[algo], alpha=0.18, linewidth=0)
        d, s = tag.split("_")[1:]
        ax.set_title(f"d={d}, s={s}")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        if ax_i // 2 == 1:
            ax.set_xlabel("rounds t")
        if ax_i % 2 == 0:
            ax.set_ylabel("cumulative regret (log)")
    # Hide unused panels
    for k in range(len(panels), len(axes)):
        axes[k].axis("off")
    # Single legend on first panel
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"  wrote {out_pdf} and {out_png}")


# ---------------------------------------------------------------------------
# Power experiment: regret + voltage trajectories
# ---------------------------------------------------------------------------

def make_power_regret(power_path=RESULTS_DIR / "power.json",
                      out_pdf=FIG_DIR / "power_regret.pdf",
                      out_png=FIG_DIR / "power_regret.png"):
    if not power_path.exists():
        print(f"[power_regret] missing {power_path}, skipping")
        return
    data = _group_by_config_algo(_load(power_path))
    tag = "power_ieee33"
    if tag not in data:
        print(f"[power_regret] no power_ieee33 entries")
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for algo in ALGO_ORDER:
        if algo not in data[tag]:
            continue
        runs = data[tag][algo]
        steps, mean, lo, hi = _aggregate_traj(runs)
        ax.plot(steps, mean, color=ALGO_COLORS[algo], label=ALGO_LABEL[algo])
        ax.fill_between(steps, lo, hi,
                        color=ALGO_COLORS[algo], alpha=0.18, linewidth=0)
    ax.set_xlabel("rounds t")
    ax.set_ylabel("cumulative pseudo-regret")
    ax.set_title("IEEE 33-bus voltage-regulation bandit")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"  wrote {out_pdf} and {out_png}")


def make_power_voltage(power_path=RESULTS_DIR / "power.json",
                       out_pdf=FIG_DIR / "power_voltage.pdf",
                       out_png=FIG_DIR / "power_voltage.png",
                       n_top: int = 3):
    if not power_path.exists():
        print(f"[power_voltage] missing {power_path}, skipping")
        return
    data = _group_by_config_algo(_load(power_path))
    tag = "power_ieee33"
    if tag not in data:
        return
    # Rank algos by mean final regret (lower = top)
    ranked = sorted(
        ALGO_ORDER,
        key=lambda a: np.mean([r["final_regret"] for r in data[tag].get(a, [])])
        if data[tag].get(a) else float("inf"),
    )
    top = [a for a in ranked if a in data[tag]][:n_top]

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for algo in top:
        runs = data[tag][algo]
        # Each run has voltage_dev_traj (downsampled). Compute running mean
        # over seeds.
        arr = np.asarray([r["voltage_dev_traj"] for r in runs])
        steps = np.asarray(runs[0]["regret_traj_steps"], dtype=int)
        mean = arr.mean(axis=0)
        # Smooth with a moving average over the trajectory axis to make
        # qualitative trends visible (stochastic per-step values are noisy).
        if mean.size > 20:
            w = 11
            kernel = np.ones(w) / w
            mean_smooth = np.convolve(mean, kernel, mode="same")
        else:
            mean_smooth = mean
        ax.plot(steps, mean_smooth, color=ALGO_COLORS[algo],
                label=ALGO_LABEL[algo])
    ax.set_xlabel("rounds t")
    ax.set_ylabel("|V deviation| at chosen-arm bus (pu)")
    ax.set_title(f"Voltage-deviation magnitude (top {len(top)} algos)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"  wrote {out_pdf} and {out_png}")


# ---------------------------------------------------------------------------
# Final regret vs d at fixed s=5
# ---------------------------------------------------------------------------

def make_regret_vs_d(synth_path=RESULTS_DIR / "synthetic.json",
                     out_pdf=FIG_DIR / "regret_vs_d.pdf",
                     out_png=FIG_DIR / "regret_vs_d.png"):
    if not synth_path.exists():
        print(f"[regret_vs_d] missing {synth_path}, skipping")
        return
    data = _group_by_config_algo(_load(synth_path))
    # Pick configs with s=5
    s5_tags = [t for t in data if t.split("_")[2] == "5"]
    s5_tags = sorted(s5_tags, key=lambda t: int(t.split("_")[1]))
    if not s5_tags:
        print("[regret_vs_d] no s=5 configs found")
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for algo in ALGO_ORDER:
        ds, means, sems = [], [], []
        for tag in s5_tags:
            if algo not in data[tag]:
                continue
            runs = data[tag][algo]
            finals = np.asarray([r["final_regret"] for r in runs])
            d = int(tag.split("_")[1])
            ds.append(d)
            means.append(finals.mean())
            sems.append(finals.std(ddof=1) / np.sqrt(len(finals)) if len(finals) > 1 else 0)
        if not ds:
            continue
        ax.errorbar(ds, means, yerr=1.96 * np.asarray(sems),
                    color=ALGO_COLORS[algo], marker="o", capsize=3,
                    label=ALGO_LABEL[algo])
    ax.set_xscale("log")
    ax.set_xlabel("ambient dimension d")
    ax.set_ylabel(f"final cum. regret @ T")
    ax.set_title("Final regret vs d (s=5)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"  wrote {out_pdf} and {out_png}")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def _fmt(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "---"
    return f"{mean:.1f} $\\pm$ {std:.1f}"


def make_synth_table(synth_path=RESULTS_DIR / "synthetic.json",
                     out_tex=RESULTS_DIR / "tables_synth_regret.tex",
                     out_txt=RESULTS_DIR / "tables.txt") -> str:
    if not synth_path.exists():
        print(f"[synth_table] missing {synth_path}, skipping")
        return ""
    data = _group_by_config_algo(_load(synth_path))
    configs = sorted(
        data.keys(),
        key=lambda t: (int(t.split("_")[1]), int(t.split("_")[2])),
    )
    # Build mean/std table, configs as rows, algos as cols.
    means = np.full((len(configs), len(ALGO_ORDER)), np.nan)
    stds = np.full_like(means, np.nan)
    for i, tag in enumerate(configs):
        for j, algo in enumerate(ALGO_ORDER):
            runs = data[tag].get(algo, [])
            if not runs:
                continue
            finals = np.asarray([r["final_regret"] for r in runs])
            means[i, j] = finals.mean()
            stds[i, j] = finals.std(ddof=1) if len(finals) > 1 else 0.0
    # Bold lowest mean per row, restricted to *sparse-aware* methods --
    # LinUCB is a non-sparse reference and is excluded from the bold pick
    # so the table caption matches its scope.
    sparse_idx = [j for j, a in enumerate(ALGO_ORDER) if a != "LinUCB"]
    means_sparse = means[:, sparse_idx]
    best_in_sparse_local = np.nanargmin(means_sparse, axis=1)
    best_per_row = np.array([sparse_idx[k] for k in best_in_sparse_local])
    # LaTeX
    head = ["config"] + [ALGO_LABEL[a].replace("(", "{(").replace(")", ")}") for a in ALGO_ORDER]
    lines = []
    lines.append("\\begin{tabular}{l" + "c" * len(ALGO_ORDER) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(head) + " \\\\")
    lines.append("\\midrule")
    for i, tag in enumerate(configs):
        d, s = tag.split("_")[1:]
        cells = [f"{d},{s}"]
        for j, algo in enumerate(ALGO_ORDER):
            cell = _fmt(means[i, j], stds[i, j])
            if j == best_per_row[i] and np.isfinite(means[i, j]):
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    tex = "\n".join(lines)
    out_tex.write_text(tex)

    # Plain-text summary
    text_lines = ["Synthetic final cumulative regret (mean +- std over seeds)\n"]
    text_lines.append(f"{'config':>10}  " + "  ".join(f"{a:>22}" for a in ALGO_ORDER))
    for i, tag in enumerate(configs):
        row = [f"{tag:>10}"]
        for j in range(len(ALGO_ORDER)):
            cell = f"{means[i,j]:7.1f}+-{stds[i,j]:5.1f}"
            if j == best_per_row[i]:
                cell = "*" + cell
            else:
                cell = " " + cell
            row.append(f"{cell:>22}")
        text_lines.append("  ".join(row))
    out_txt.write_text("\n".join(text_lines) + "\n")
    print(f"  wrote {out_tex} and {out_txt}")
    return tex


def make_power_table(power_path=RESULTS_DIR / "power.json",
                     out_tex=RESULTS_DIR / "tables_power.tex") -> str:
    if not power_path.exists():
        print(f"[power_table] missing {power_path}, skipping")
        return ""
    data = _group_by_config_algo(_load(power_path))
    tag = "power_ieee33"
    if tag not in data:
        return ""
    rows = []
    for algo in ALGO_ORDER:
        runs = data[tag].get(algo, [])
        if not runs:
            rows.append((algo, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        finals = np.asarray([r["final_regret"] for r in runs])
        # mean per-step voltage deviation (over downsampled traj, then over seeds)
        v_traj = np.asarray([r["voltage_dev_traj"] for r in runs])
        v_mean_per_run = v_traj.mean(axis=1)
        runtimes = np.asarray([r["runtime_s"] for r in runs])
        T = runs[0]["T"]
        per_round_us = 1e6 * runtimes.mean() / T
        rows.append((algo, finals.mean(), finals.std(ddof=1),
                     v_mean_per_run.mean(), v_mean_per_run.std(ddof=1),
                     per_round_us))

    # Bold the lowest values column by column. For final regret we mark
    # *both* the overall winner and the best sparse-aware method, since
    # the latter is the relevant comparison for the paper's claims.
    finals = [r[1] for r in rows]
    runtimes_us = [r[5] for r in rows]
    sparse_indices = [i for i, r in enumerate(rows) if r[0] != "LinUCB"]
    best_overall = int(np.nanargmin(finals))
    sparse_finals = [finals[i] if i in sparse_indices else np.inf for i in range(len(rows))]
    best_sparse = int(np.nanargmin(sparse_finals))
    best_runtime = int(np.nanargmin(runtimes_us))

    lines = []
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("algorithm & final regret & mean $|\\Delta V|$ (pu) & runtime ($\\mu$s/round) \\\\")
    lines.append("\\midrule")
    for i, (algo, fm, fs, vm, vs, rt) in enumerate(rows):
        label = ALGO_LABEL[algo]
        cell_reg = _fmt(fm, fs) if np.isfinite(fm) else "---"
        cell_v = f"{vm:.4f} $\\pm$ {vs:.4f}" if np.isfinite(vm) else "---"
        cell_rt = f"{rt:.0f}" if np.isfinite(rt) else "---"
        if i == best_overall:
            cell_reg = f"\\textbf{{{cell_reg}}}"
        elif i == best_sparse:
            cell_reg = f"\\underline{{{cell_reg}}}"
        if i == best_runtime:
            cell_rt = f"\\textbf{{{cell_rt}}}"
        lines.append(f"{label} & {cell_reg} & {cell_v} & {cell_rt} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    tex = "\n".join(lines)
    out_tex.write_text(tex)
    print(f"  wrote {out_tex}")
    return tex


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    make_synth_grid()
    make_power_regret()
    make_power_voltage()
    make_regret_vs_d()
    make_synth_table()
    make_power_table()


if __name__ == "__main__":
    main()
