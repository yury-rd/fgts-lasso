"""Add nonlinear baselines (KernelUCB, GP-TS) on representative configs.

Subset configs: synthetic (50,5), (100,5), (200,5) at T=5000; power-systems.
20 seeds. Uses existing run_experiments harness via direct call.
"""
from __future__ import annotations
import json, sys, time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from algos import ALGO_REGISTRY
from bandit_env import SyntheticEnv, PowerSystemEnv

RES = HERE.parent / "results"
RES.mkdir(exist_ok=True)

CONFIGS = [
    ("synth", 50, 5), ("synth", 100, 5), ("synth", 200, 5),
    ("power", 32, 5),
]
ALGOS = ["KernelUCB", "GPTS"]
T = 5000
K = 5
SEEDS = 20


def run_one(args):
    env_kind, d, s, algo_name, seed = args
    if env_kind == "synth":
        env = SyntheticEnv(d=d, s=s, K=K, T=T)
        tag = f"exp_{d}_{s}"
    else:
        env = PowerSystemEnv(T=T)
        tag = "power_ieee33"
    cls = ALGO_REGISTRY[algo_name]
    algo = cls(d=env.d, K=env.K, T=T, seed=seed)
    x = env.reset(seed=seed)
    t0 = time.time()
    for _ in range(T):
        a = algo.act(x); xn, r, _i = env.step(a); algo.update(x, a, r); x = xn
    runtime = time.time() - t0
    return {
        "env_kind": env_kind, "config_tag": tag, "algo": algo_name, "seed": seed,
        "d": env.d, "s": s, "K": K, "T": T,
        "final_regret": float(env.regret_so_far()), "runtime_s": runtime,
    }


def main():
    specs = []
    for env_kind, d, s in CONFIGS:
        for a in ALGOS:
            for seed in range(SEEDS):
                specs.append((env_kind, d, s, a, seed))
    print(f"kernel sweep: {len(specs)} runs")
    t0 = time.time()
    with Pool(processes=8) as pool:
        out = []
        for i, r in enumerate(pool.imap_unordered(run_one, specs, chunksize=1)):
            out.append(r)
            if (i + 1) % max(1, len(specs)//10) == 0 or (i + 1) == len(specs):
                print(f"  {i+1}/{len(specs)}  elapsed={time.time()-t0:.1f}s", flush=True)
    payload = {"n_runs": len(out), "results": out, "total_runtime_s": time.time()-t0}
    fp = RES / "kernel.json"
    fp.write_text(json.dumps(payload))
    print(f"wrote {fp}")

    import collections, statistics as st
    by = collections.defaultdict(list)
    for r in out:
        by[(r["config_tag"], r["algo"])].append(r["final_regret"])
    for (cfg, a), vs in sorted(by.items()):
        print(f"  {cfg:<20} {a:<10} regret={st.mean(vs):.1f}+-{st.stdev(vs) if len(vs)>1 else 0:.1f}")


if __name__ == "__main__":
    main()
