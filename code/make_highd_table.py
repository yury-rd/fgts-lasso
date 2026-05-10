"""Generate LaTeX table for high-d sweep."""
from __future__ import annotations
import json, collections, statistics as stats
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
data = json.loads((RES / "highd.json").read_text())
by_reg = collections.defaultdict(list); by_rt = collections.defaultdict(list)
for r in data["results"]:
    by_reg[(r["d"], r["algo"])].append(r["final_regret"])
    by_rt[(r["d"], r["algo"])].append(r["runtime_s"] * 1000.0 / r["T"])

algo_order = ["FGTSLasso","LassoBandit","SALassoBandit","ThresholdedLassoBandit","LinUCB"]
labels = {
    "FGTSLasso": r"FGTS-LASSO (ours)",
    "LassoBandit": r"LASSO-Bandit~\cite{bastani2020online}",
    "SALassoBandit": r"SA-LASSO~\cite{oh2021sparsity}",
    "ThresholdedLassoBandit": r"Thresh-LASSO~\cite{ariu2022thresholded}",
    "LinUCB": r"LinUCB~\cite{chu2011contextual}",
}
ds = sorted(set(r["d"] for r in data["results"]))

# Best (lowest) per column among sparse-aware methods only.
# Use small tolerance to handle ties where rounded means coincide.
sparse = ["FGTSLasso","LassoBandit","SALassoBandit","ThresholdedLassoBandit"]
TOL = 0.05
best_reg_sparse = {d: min(stats.mean(by_reg[(d, a)]) for a in sparse) for d in ds}
best_rt = {d: min(stats.mean(by_rt[(d, a)]) for a in algo_order) for d in ds}

lines = []
lines.append(r"\begin{tabular}{l" + "c" * len(ds) + "}")
lines.append(r"\toprule")
lines.append("algorithm & " + " & ".join(f"$d{{=}}{d}$" for d in ds) + r" \\")
lines.append(r"\midrule")
lines.append(rf"\multicolumn{{{len(ds)+1}}}{{l}}{{\textit{{Cumulative regret at $T{{=}}1000$ (mean $\pm$ std, 20 seeds)}}}} \\")
for a in algo_order:
    cells = []
    for d in ds:
        v = by_reg[(d, a)]
        m = stats.mean(v); sd = stats.stdev(v)
        cell = f"{m:.1f}$\\pm${sd:.1f}"
        if a in sparse and abs(m - best_reg_sparse[d]) < TOL:
            cell = r"\textbf{" + cell + "}"
        cells.append(cell)
    lines.append(f"{labels[a]} & " + " & ".join(cells) + r" \\")
lines.append(r"\midrule")
lines.append(rf"\multicolumn{{{len(ds)+1}}}{{l}}{{\textit{{Runtime per round (ms)}}}} \\")
for a in algo_order:
    cells = []
    for d in ds:
        m = stats.mean(by_rt[(d, a)])
        cell = f"{m:.2f}"
        if abs(m - best_rt[d]) < 0.01:
            cell = r"\textbf{" + cell + "}"
        cells.append(cell)
    lines.append(f"{labels[a]} & " + " & ".join(cells) + r" \\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")

out = RES / "tables_highd.tex"
out.write_text("\n".join(lines))
print(f"wrote {out}")
