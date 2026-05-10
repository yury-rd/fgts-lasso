"""Generate LaTeX table comparing nonlinear baselines (KernelUCB, GP-TS) vs.
FGTS-LASSO and LinUCB on representative configs.
"""
from __future__ import annotations
import json, collections, statistics as stats
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"

# Pull from synthetic.json (FGTS-LASSO, LinUCB), kernel.json (KernelUCB, GPTS),
# power.json (FGTS-LASSO, LinUCB on power).
def load(p):
    return json.loads(Path(p).read_text())["results"] if Path(p).exists() else []

synth = load(RES / "synthetic.json")
power = load(RES / "power.json")
kernel = load(RES / "kernel.json")

CONFIGS = ["exp_50_5", "exp_100_5", "exp_200_5", "power_ieee33"]
LABELS = {
    "exp_50_5": "synth, $d{=}50$, $s{=}5$",
    "exp_100_5": "synth, $d{=}100$, $s{=}5$",
    "exp_200_5": "synth, $d{=}200$, $s{=}5$",
    "power_ieee33": "IEEE 33-bus",
}
ALGOS = [("FGTSLasso", "FGTS-LASSO (ours)"),
         ("LinUCB", "LinUCB"),
         ("KernelUCB", "KernelUCB"),
         ("GPTS", "GP-TS")]

# Aggregate
agg = collections.defaultdict(list)
for r in synth + power + kernel:
    agg[(r["config_tag"], r["algo"])].append(r["final_regret"])

lines = [r"\begin{tabular}{l" + "c" * len(CONFIGS) + "}", r"\toprule"]
lines.append("algorithm & " + " & ".join(LABELS[c] for c in CONFIGS) + r" \\")
lines.append(r"\midrule")
# Compute best (lowest mean) per column among the four algos
best_per_col = {}
for c in CONFIGS:
    valid = [(a, stats.mean(agg[(c, a)])) for a, _ in ALGOS if agg.get((c, a))]
    if valid:
        best_per_col[c] = min(valid, key=lambda t: t[1])[0]
for a, lbl in ALGOS:
    cells = []
    for c in CONFIGS:
        v = agg.get((c, a))
        if not v:
            cells.append("---")
            continue
        m = stats.mean(v); sd = stats.stdev(v) if len(v) > 1 else 0.0
        cell = f"{m:.1f}$\\pm${sd:.1f}"
        if best_per_col.get(c) == a:
            cell = r"\textbf{" + cell + "}"
        cells.append(cell)
    lines.append(f"{lbl} & " + " & ".join(cells) + r" \\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")

out = RES / "tables_kernel.tex"
out.write_text("\n".join(lines))
print(f"wrote {out}")
print("\n".join(lines))
