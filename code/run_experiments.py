"""Experiment runner for the sparse-bandits paper.

Two scripts in one file:

* ``run_synthetic`` -- 9 (d, s) configs x 5 algorithms x 30 seeds, T=5000.
* ``run_power`` -- IEEE 33-bus LinDistFlow env, 5 algorithms x 30 seeds, T=5000.

Seed-level work is parallelised via ``multiprocessing.Pool`` capped at 8
workers. Results are written to ``results/synthetic.json`` and
``results/power.json`` with cumulative-regret trajectories, final regret,
and runtime per (config, algo, seed).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

# Ensure local imports work from any cwd
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from algos import ALGO_REGISTRY  # noqa: E402
from bandit_env import PowerSystemEnv, SyntheticEnv  # noqa: E402

ROOT = HERE.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Configurations to sweep -----------------------------------------------------
SYNTH_CONFIGS: List[Tuple[int, int]] = [
    (10, 2), (50, 3), (50, 5),
    (100, 2), (100, 5),
    (200, 2), (200, 5),
    (500, 3), (500, 5),
]
DEFAULT_T = 5000
DEFAULT_K = 5
DEFAULT_SEEDS = 30
ALGO_NAMES = ["FGTSLasso", "LassoBandit", "SALassoBandit", "ThresholdedLassoBandit", "LinUCB", "LinTS"]

# How many regret-trajectory samples to keep per run (downsample to save space)
TRAJ_SAMPLES = 250  # store every T/250 ~= 20 steps of cum regret


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _make_algo(name: str, d: int, K: int, T: int, seed: int, s_hint: int):
    cls = ALGO_REGISTRY[name]
    kwargs = dict(d=d, K=K, T=T, seed=seed)
    if name in ("FGTSLasso", "ThresholdedLassoBandit"):
        kwargs["s_hint"] = s_hint
    return cls(**kwargs)


def _downsample(traj: np.ndarray, n_keep: int) -> np.ndarray:
    if traj.size <= n_keep:
        return traj.astype(float)
    idx = np.linspace(0, traj.size - 1, n_keep, dtype=int)
    return traj[idx].astype(float)


@dataclass
class RunSpec:
    """Specification of a single experiment run (one algo / seed)."""
    env_kind: str           # 'synth' or 'power'
    algo_name: str
    seed: int
    d: int
    s: int                  # s_hint for algos that accept it
    K: int
    T: int
    config_tag: str


def _run_one(spec: RunSpec) -> dict:
    """Run a single (env-instance x algorithm x seed) trajectory."""
    t0 = time.time()
    if spec.env_kind == "synth":
        env = SyntheticEnv(d=spec.d, s=spec.s, K=spec.K, T=spec.T)
    elif spec.env_kind == "power":
        env = PowerSystemEnv(T=spec.T)
    else:
        raise ValueError(f"unknown env_kind {spec.env_kind}")

    algo = _make_algo(spec.algo_name, env.d, env.K, spec.T, spec.seed, spec.s)
    x = env.reset(seed=spec.seed)

    voltage_devs: List[float] = []  # only for power env
    for _t in range(spec.T):
        a = algo.act(x)
        x_next, r, info = env.step(a)
        algo.update(x, a, r)
        x = x_next
        if spec.env_kind == "power":
            voltage_devs.append(info.get("voltage_dev_arm", 0.0))

    runtime = time.time() - t0
    traj = env.regret_history()
    final_regret = float(traj[-1]) if traj.size else 0.0
    out = {
        "env_kind": spec.env_kind,
        "config_tag": spec.config_tag,
        "algo": spec.algo_name,
        "seed": spec.seed,
        "d": spec.d,
        "s": spec.s,
        "K": spec.K,
        "T": spec.T,
        "final_regret": final_regret,
        "runtime_s": runtime,
        "regret_trajectory": _downsample(traj, TRAJ_SAMPLES).tolist(),
        "regret_traj_steps": _downsample(np.arange(1, traj.size + 1), TRAJ_SAMPLES).astype(int).tolist(),
    }
    if spec.env_kind == "power":
        out["voltage_dev_traj"] = _downsample(np.asarray(voltage_devs), TRAJ_SAMPLES).tolist()
    return out


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def _make_synth_specs(seeds: List[int], configs=SYNTH_CONFIGS) -> List[RunSpec]:
    specs = []
    for d, s in configs:
        tag = f"exp_{d}_{s}"
        for seed in seeds:
            for algo_name in ALGO_NAMES:
                specs.append(RunSpec(
                    env_kind="synth", algo_name=algo_name, seed=seed,
                    d=d, s=s, K=DEFAULT_K, T=DEFAULT_T, config_tag=tag,
                ))
    return specs


def _make_power_specs(seeds: List[int]) -> List[RunSpec]:
    specs = []
    for seed in seeds:
        for algo_name in ALGO_NAMES:
            specs.append(RunSpec(
                env_kind="power", algo_name=algo_name, seed=seed,
                d=32, s=5, K=DEFAULT_K, T=DEFAULT_T, config_tag="power_ieee33",
            ))
    return specs


def run_synthetic(seeds: int = DEFAULT_SEEDS, n_workers: int = 8,
                  out_path: Path = RESULTS_DIR / "synthetic.json",
                  configs=SYNTH_CONFIGS) -> dict:
    seeds_list = list(range(seeds))
    specs = _make_synth_specs(seeds_list, configs=configs)
    print(f"[synthetic] {len(specs)} runs across {len(configs)} configs, "
          f"{seeds} seeds, {len(ALGO_NAMES)} algos", flush=True)
    return _execute(specs, n_workers, out_path)


def run_power(seeds: int = DEFAULT_SEEDS, n_workers: int = 8,
              out_path: Path = RESULTS_DIR / "power.json") -> dict:
    seeds_list = list(range(seeds))
    specs = _make_power_specs(seeds_list)
    print(f"[power] {len(specs)} runs across 1 config, {seeds} seeds, "
          f"{len(ALGO_NAMES)} algos", flush=True)
    return _execute(specs, n_workers, out_path)


def _execute(specs: List[RunSpec], n_workers: int, out_path: Path) -> dict:
    n_workers = max(1, min(int(n_workers), 8, cpu_count()))
    t0 = time.time()
    print(f"  using {n_workers} workers", flush=True)
    with Pool(processes=n_workers) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(_run_one, specs, chunksize=1)):
            results.append(res)
            if (i + 1) % max(1, len(specs) // 20) == 0 or (i + 1) == len(specs):
                elapsed = time.time() - t0
                print(f"    {i+1}/{len(specs)} done   elapsed={elapsed:.1f}s",
                      flush=True)
    payload = {
        "n_runs": len(results),
        "results": results,
        "total_runtime_s": time.time() - t0,
    }
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(payload))
    print(f"  wrote {out_path}  ({len(results)} runs, "
          f"{payload['total_runtime_s']:.1f}s total)", flush=True)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["synth", "power", "both"], default="both")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reduced", action="store_true",
                        help="Reduce to 5 most informative configs + power expt.")
    args = parser.parse_args(argv)

    configs = SYNTH_CONFIGS
    if args.reduced:
        # Five "most informative" configs spanning the d-axis at fixed s=5
        # plus the small-(d,s) point as an easy-regime sanity check.
        configs = [(50, 5), (100, 5), (200, 5), (500, 5), (50, 3)]

    if args.which in ("synth", "both"):
        run_synthetic(seeds=args.seeds, n_workers=args.workers, configs=configs)
    if args.which in ("power", "both"):
        run_power(seeds=args.seeds, n_workers=args.workers)


if __name__ == "__main__":
    main()
