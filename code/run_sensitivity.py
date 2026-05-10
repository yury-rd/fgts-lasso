"""FGTS-LASSO sensitivity ablation: lambda multiplier and tau_0 multiplier
sweeps. Run on representative configs (d=200, s=5) and (d=500, s=5).
"""
from __future__ import annotations
import json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from algos import FGTSLasso, _safe_lasso_fit, _ridge_solve
from bandit_env import SyntheticEnv

RES = HERE.parent / "results"
T = 5000
K = 5
S = 5
SEEDS = 20
LAMBDA_MULTS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
TAU_FACTORS = [0.5, 1.0, 2.0, 4.0, 8.0]
CONFIGS = [(100, 5), (500, 5)]


def make_fgts(d, K, T, seed, lam_mult, tau_factor):
    algo = FGTSLasso(d=d, K=K, T=T, seed=seed, s_hint=S)
    # Override tau_0 (default = c1*s_hint*log(d) with c1=2)
    algo.tau_0 = max(K, int(np.ceil(tau_factor * 2.0 * S * np.log(max(d, 2)))))
    # Monkey-patch _refit_arm to use lam_mult
    orig_self = algo
    def _refit(arm: int) -> None:
        X, y = orig_self._arm_data(arm)
        if X is None or X.shape[0] < 2:
            return
        n_a = X.shape[0]
        lam = lam_mult * orig_self.sigma * np.sqrt(np.log(max(orig_self.d, 2)) / max(n_a, 1))
        coef = _safe_lasso_fit(X, y, lam)
        S_set = np.where(np.abs(coef) > orig_self.coef_thr)[0]
        if S_set.size == 0:
            order = np.argsort(np.abs(coef))[::-1]
            S_set = np.sort(order[: min(orig_self.s_hint, orig_self.d)])
        X_S = X[:, S_set]
        mu_S, V_inv_S = _ridge_solve(X_S, y, orig_self.rho)
        full_mu = np.zeros(orig_self.d); full_mu[S_set] = mu_S
        full_V_inv = np.zeros((orig_self.d, orig_self.d))
        full_V_inv[np.ix_(S_set, S_set)] = V_inv_S
        orig_self._S_hat[arm] = S_set
        orig_self._mu[arm] = full_mu
        orig_self._V_inv[arm] = full_V_inv
    algo._refit_arm = _refit
    return algo


def run_one(args):
    kind, d, s, seed, lam_mult, tau_factor = args
    env = SyntheticEnv(d=d, s=s, K=K, T=T)
    algo = make_fgts(d, K, T, seed, lam_mult, tau_factor)
    x = env.reset(seed=seed)
    for _ in range(T):
        a = algo.act(x); xn, r, _i = env.step(a); algo.update(x, a, r); x = xn
    return {
        "kind": kind, "d": d, "s": s, "seed": seed,
        "lam_mult": lam_mult, "tau_factor": tau_factor,
        "final_regret": float(env.regret_so_far()),
    }


def main():
    specs = []
    # Sweep lam_mult at fixed tau_factor=1
    for d, s in CONFIGS:
        for lm in LAMBDA_MULTS:
            for sd in range(SEEDS):
                specs.append(("lam", d, s, sd, lm, 1.0))
    # Sweep tau_factor at fixed lam_mult=0.1
    for d, s in CONFIGS:
        for tf in TAU_FACTORS:
            for sd in range(SEEDS):
                specs.append(("tau", d, s, sd, 0.1, tf))
    print(f"sensitivity sweep: {len(specs)} runs", flush=True)
    t0 = time.time()
    with Pool(processes=8) as pool:
        out = []
        for i, r in enumerate(pool.imap_unordered(run_one, specs, chunksize=2)):
            out.append(r)
            if (i + 1) % max(1, len(specs)//10) == 0 or i+1 == len(specs):
                print(f"  {i+1}/{len(specs)}  elapsed={time.time()-t0:.1f}s", flush=True)
    fp = RES / "sensitivity.json"
    fp.write_text(json.dumps({"results": out, "total_runtime_s": time.time()-t0}))
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()
