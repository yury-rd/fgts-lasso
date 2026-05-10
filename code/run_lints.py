"""LinTS-only sweep, merge into existing synthetic.json and power.json."""
from __future__ import annotations
import json, sys, time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from algos import LinTS
from bandit_env import SyntheticEnv, PowerSystemEnv

RES = HERE.parent / "results"
SYNTH_CONFIGS = [(10,2),(50,3),(50,5),(100,2),(100,5),(200,2),(200,5),(500,3),(500,5)]
T = 5000
K = 5
SEEDS = 30


def run_one(args):
    kind, d, s, seed = args
    if kind == "synth":
        env = SyntheticEnv(d=d, s=s, K=K, T=T)
        tag = f"exp_{d}_{s}"
    else:
        env = PowerSystemEnv(T=T)
        tag = "power_ieee33"
    algo = LinTS(d=env.d, K=env.K, T=T, seed=seed)
    x = env.reset(seed=seed)
    t0 = time.time()
    voltage = []
    for _ in range(T):
        a = algo.act(x); xn, r, info = env.step(a); algo.update(x, a, r); x = xn
        if kind == "power":
            voltage.append(info.get("voltage_dev_arm", 0.0))
    runtime = time.time() - t0
    traj = env.regret_history()
    import numpy as np
    n_keep = 250
    idx = np.linspace(0, traj.size-1, n_keep, dtype=int)
    result = {
        "env_kind": kind, "config_tag": tag, "algo": "LinTS", "seed": seed,
        "d": env.d, "s": s, "K": K, "T": T,
        "final_regret": float(env.regret_so_far()), "runtime_s": runtime,
        "regret_trajectory": traj[idx].tolist(),
        "regret_traj_steps": (idx + 1).tolist(),
    }
    if kind == "power":
        v = np.asarray(voltage)
        result["voltage_dev_traj"] = v[idx].tolist()
    return result


def main():
    specs = [("synth", d, s, seed) for d, s in SYNTH_CONFIGS for seed in range(SEEDS)]
    specs += [("power", 32, 5, seed) for seed in range(SEEDS)]
    print(f"LinTS sweep: {len(specs)} runs")
    t0 = time.time()
    with Pool(processes=8) as pool:
        out = []
        for i, r in enumerate(pool.imap_unordered(run_one, specs, chunksize=1)):
            out.append(r)
            if (i+1) % max(1, len(specs)//10) == 0 or i+1 == len(specs):
                print(f"  {i+1}/{len(specs)}  elapsed={time.time()-t0:.1f}s", flush=True)

    # Merge into existing synthetic.json and power.json
    for kind, fp in [("synth", RES / "synthetic.json"), ("power", RES / "power.json")]:
        if not fp.exists():
            continue
        existing = json.loads(fp.read_text())
        keep = [r for r in existing["results"] if r.get("algo") != "LinTS"]
        new_runs = [r for r in out if r["env_kind"] == kind]
        existing["results"] = keep + new_runs
        existing["n_runs"] = len(existing["results"])
        fp.write_text(json.dumps(existing))
        print(f"  merged {len(new_runs)} LinTS runs into {fp}")


if __name__ == "__main__":
    main()
