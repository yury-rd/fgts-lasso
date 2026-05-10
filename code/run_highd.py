"""High-dimensional sweep: T < d regime where sparse methods dominate on runtime.

Configs: d in {500, 1000, 2000, 4000}, T = 1000, K=5, s=5.
"""

from __future__ import annotations
import json, sys, time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from algos import ALGO_REGISTRY
from bandit_env import SyntheticEnv

ROOT = HERE.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

D_LIST = [500, 1000, 2000, 4000]
T = 1000
K = 5
S = 5
SEEDS = 20
ALGO_NAMES = ["FGTSLasso", "LassoBandit", "SALassoBandit", "ThresholdedLassoBandit", "LinUCB", "LinTS"]


def run_one(args):
    d, algo_name, seed = args
    # LinTS has O(d^3) Cholesky per round and is intractable at d>=2000;
    # we skip it on those configs and report N/A in tables.
    if algo_name == "LinTS" and d >= 2000:
        return {
            "d": d, "algo": algo_name, "seed": seed, "T": T, "K": K, "s": S,
            "final_regret": float("nan"), "runtime_s": float("nan"),
        }
    env = SyntheticEnv(d=d, s=S, K=K, T=T)
    cls = ALGO_REGISTRY[algo_name]
    kw = dict(d=d, K=K, T=T, seed=seed)
    if algo_name in ("FGTSLasso", "ThresholdedLassoBandit"):
        kw["s_hint"] = S
    algo = cls(**kw)
    x = env.reset(seed=seed)
    t0 = time.time()
    for _ in range(T):
        a = algo.act(x)
        xn, r, _info = env.step(a)
        algo.update(x, a, r)
        x = xn
    runtime = time.time() - t0
    return {
        "d": d, "algo": algo_name, "seed": seed, "T": T, "K": K, "s": S,
        "final_regret": env.regret_so_far(), "runtime_s": runtime,
    }


def main():
    specs = [(d, a, s) for d in D_LIST for a in ALGO_NAMES for s in range(SEEDS)]
    print(f"highd sweep: {len(specs)} runs")
    t0 = time.time()
    with Pool(processes=8) as pool:
        results = list(pool.imap_unordered(run_one, specs, chunksize=1))
    payload = {"n_runs": len(results), "results": results, "total_runtime_s": time.time() - t0}
    out = RESULTS_DIR / "highd.json"
    out.write_text(json.dumps(payload))
    print(f"wrote {out} in {payload['total_runtime_s']:.1f}s")

    # Quick summary
    import collections, statistics as stats
    by = collections.defaultdict(list)
    by_rt = collections.defaultdict(list)
    for r in results:
        by[(r["d"], r["algo"])].append(r["final_regret"])
        by_rt[(r["d"], r["algo"])].append(r["runtime_s"])
    print("\nFinal regret (mean +- std):")
    for d in D_LIST:
        row = [f"d={d}"]
        for a in ALGO_NAMES:
            v = by[(d, a)]
            row.append(f"{a}={stats.mean(v):.1f}+-{stats.stdev(v) if len(v)>1 else 0:.1f}")
        print("  ", "  ".join(row))
    print("\nRuntime per algo (mean s/run):")
    for d in D_LIST:
        row = [f"d={d}"]
        for a in ALGO_NAMES:
            v = by_rt[(d, a)]
            row.append(f"{a}={stats.mean(v):.2f}s")
        print("  ", "  ".join(row))


if __name__ == "__main__":
    main()
