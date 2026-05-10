"""Empirical conformal-coverage demonstration for FGTS-LASSO.

We run the bandit normally; after the forced phase, we maintain a per-arm
calibration set of held-out residuals (absolute reward prediction error) and,
at each round, form a split-conformal prediction interval for the chosen-arm
reward at level 1 - alpha. We then track empirical marginal coverage as the
horizon advances.

Reference: Vovk-Gammerman-Shafer (2005); Gibbs & Candes (2021) for the
adaptive online variant. Here we use plain split conformal with a rolling
window so the calibration set adapts to drift in policy-induced data.
"""
from __future__ import annotations
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from algos import FGTSLasso, _ridge_solve
from bandit_env import SyntheticEnv, PowerSystemEnv

RES = HERE.parent / "results"
RES.mkdir(exist_ok=True)

ALPHA = 0.10           # target miscoverage; aim for ~90% empirical coverage
CAL_FRAC = 0.30        # held-out fraction used as conformal calibration set
WIN = 400              # rolling window for adaptive calibration set
T = 5000
K = 5
SEEDS = 30
CONFIGS = [
    ("synth", 200, 5),
    ("synth", 500, 5),
    ("power", 32, 5),
]


def split_conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """Standard split-conformal correction: ceil((n+1)(1-alpha))/n quantile of |residuals|."""
    if residuals.size == 0:
        return 0.0
    n = residuals.size
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(np.abs(residuals), q_level))


def run_one(args):
    env_kind, d, s, seed = args
    if env_kind == "synth":
        env = SyntheticEnv(d=d, s=s, K=K, T=T)
        tag = f"exp_{d}_{s}"
    else:
        env = PowerSystemEnv(T=T)
        tag = "power_ieee33"
    algo = FGTSLasso(d=env.d, K=env.K, T=T, seed=seed, s_hint=s)
    rng = np.random.default_rng(10_000 + seed)
    x = env.reset(seed=seed)

    # Per-arm rolling calibration buffers
    cal_residuals = [[] for _ in range(env.K)]
    covered = []   # 1 if y_t in conformal interval, else 0
    interval_widths = []

    for t in range(T):
        a = algo.act(x)
        # Predict mean reward for chosen arm using current ridge mean
        S = algo._S_hat[a]
        mu_full = algo._mu[a]
        if S.size > 0:
            y_hat = float(x[S] @ mu_full[S])
        else:
            y_hat = 0.0
        # Conformal half-width at level (1-alpha)
        q = split_conformal_quantile(np.asarray(cal_residuals[a]), ALPHA)
        interval_widths.append(2.0 * q)

        x_next, y, _info = env.step(a)
        # Decide: was this point assigned to calibration or to model-update?
        if rng.random() < CAL_FRAC and t > algo.tau_0:
            # Calibration point: add residual, do not update algo
            cal_residuals[a].append(abs(y - y_hat))
            if len(cal_residuals[a]) > WIN:
                cal_residuals[a] = cal_residuals[a][-WIN:]
        else:
            algo.update(x, a, y)

        # Record coverage event (after forced phase, otherwise interval=0 trivially)
        if t > algo.tau_0:
            in_int = abs(y - y_hat) <= q
            covered.append(1.0 if in_int else 0.0)
        else:
            covered.append(np.nan)

        x = x_next

    covered = np.asarray(covered)
    widths = np.asarray(interval_widths)
    # Rolling coverage (window = 400 rounds)
    valid = ~np.isnan(covered)
    rc = np.full_like(covered, np.nan)
    for i in range(covered.size):
        lo = max(0, i - 400)
        seg = covered[lo:i + 1]
        seg_v = seg[~np.isnan(seg)]
        if seg_v.size >= 25:
            rc[i] = float(seg_v.mean())
    # Final marginal coverage
    final_cov = float(covered[valid].mean()) if valid.any() else float("nan")
    final_width = float(widths[-1])
    # Down-sample trajectories for storage
    n_keep = 250
    idx = np.linspace(0, T - 1, n_keep, dtype=int)
    return {
        "env_kind": env_kind, "config_tag": tag, "seed": seed,
        "d": env.d, "s": s, "K": env.K, "T": T,
        "alpha": ALPHA, "final_coverage": final_cov, "final_width": final_width,
        "rolling_coverage": rc[idx].tolist(),
        "interval_widths": widths[idx].tolist(),
        "steps": idx.tolist(),
    }


def main():
    specs = [(k, d, s, seed) for (k, d, s) in CONFIGS for seed in range(SEEDS)]
    print(f"conformal sweep: {len(specs)} runs", flush=True)
    t0 = time.time()
    with Pool(processes=8) as pool:
        out = []
        for i, r in enumerate(pool.imap_unordered(run_one, specs, chunksize=2)):
            out.append(r)
            if (i + 1) % max(1, len(specs)//10) == 0 or i + 1 == len(specs):
                print(f"  {i+1}/{len(specs)}  elapsed={time.time()-t0:.1f}s", flush=True)
    fp = RES / "conformal.json"
    fp.write_text(json.dumps({"results": out, "alpha": ALPHA, "total_runtime_s": time.time()-t0}))
    print(f"wrote {fp}")
    # Summary
    import collections, statistics as st
    by = collections.defaultdict(list)
    for r in out:
        by[r["config_tag"]].append(r["final_coverage"])
    for cfg, v in by.items():
        print(f"  {cfg}: empirical coverage mean={st.mean(v):.3f}+-{st.stdev(v) if len(v)>1 else 0:.3f}  (target {1-ALPHA})")


if __name__ == "__main__":
    main()
