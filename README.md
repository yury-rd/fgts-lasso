# FGTS-LASSO

Reference implementation for *"Conformal Sparse Bandits for Online Voltage
Regulation in Distribution Grids"* by E. Suraveikin, A. Golubev, R. Sultimov,
A. Volkov, and Y. Maximov (IEEE Control Systems Letters, 2026).

## Algorithms

| Class | Reference |
|---|---|
| `FGTSLasso` | Proposed: forced-exploration + LASSO support recovery + feature-Gaussian Thompson sampling on the recovered subspace |
| `LassoBandit` | Bastani & Bayati, *Oper. Res.*, 2020 |
| `SALassoBandit` | Oh, Iyengar & Zeevi, ICML, 2021 |
| `ThresholdedLassoBandit` | Ariu, Abe & Proutiere, ICML, 2022 |
| `LinUCB` | Chu et al., AISTATS, 2011 |
| `KernelUCB` | Valko et al., UAI, 2013 |
| `GPTS` | Srinivas et al., ICML, 2010 |
| `LinTS` | Agrawal & Goyal, ICML, 2013 |

## Environments

* `SyntheticEnv` — sparse linear bandit with random ±1 supports
* `PowerSystemEnv` — IEEE 33-bus radial feeder; LinDistFlow voltage-sensitivity oracle; arms target individual buses, sparsity = depth in the feeder topology

## Layout

```
code/
  lindistflow.py          IEEE 33-bus LinDistFlow oracle
  bandit_env.py           Synthetic + power-systems environments
  algos.py                8 algorithms with a unified API
  run_experiments.py      Synthetic + Volt/VAR sweeps (T = 5000, 30 seeds)
  run_highd.py            High-dimensional sweep (d up to 4000)
  run_kernel.py           KernelUCB + GP-TS subset
  run_lints.py            Ambient-space Thompson sampling
  run_sensitivity.py      lambda + tau_0 ablation
  run_conformal.py        Split-conformal coverage demonstration
  make_figures.py         Plot + table generation
  make_highd_table.py     High-d LaTeX table
  make_kernel_table.py    Kernel-baselines LaTeX table
```

## Reproducing the experiments

Dependencies: numpy, scipy, scikit-learn, matplotlib.

```bash
cd code
python3 run_experiments.py    # ~25 min on 8 cores
python3 run_highd.py
python3 run_kernel.py
python3 run_sensitivity.py
python3 run_conformal.py
python3 make_figures.py
python3 make_highd_table.py
python3 make_kernel_table.py
```

Results land in `results/` (created on first run).

## License

MIT.
