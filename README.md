# FGTS-LASSO: Decoupled Sparse Thompson Sampling for High-Dimensional Linear Contextual Bandits

Reference implementation and reproducibility artefacts for the IEEE Control
Systems Letters submission *"Sparse Linear Bandits with LASSO Support Recovery for Online Voltage
Regulation"* by E. Suraveikin, A. Golubev, R. Sultimov, A. Volkov, and Y. Maximov.

## Contents

```
main.tex                    Paper source (IEEEtran journal class)
refs.bib                    Bibliography
figures/                    PDF/PNG figures used in the paper
results/                    Aggregated experiment results (JSON + LaTeX tables)
code/                       Python implementation
  lindistflow.py            IEEE 33-bus LinDistFlow oracle
  bandit_env.py             Synthetic and power-systems bandit environments
  algos.py                  FGTS-LASSO + 7 baseline algorithms
  run_experiments.py        Synthetic and Volt/VAR sweeps
  run_highd.py              High-dimensional sweep (d up to 4000)
  run_kernel.py             Nonlinear baselines (KernelUCB, GP-TS)
  run_sensitivity.py        lambda / tau_0 ablation
  run_lints.py              Ambient-space Thompson Sampling baseline
  make_figures.py           Plot + table generators
  make_highd_table.py       High-d LaTeX table
  make_kernel_table.py      Kernel-baselines LaTeX table
```

## Reproducing the experiments

```bash
# Dependencies: numpy, scipy, scikit-learn, matplotlib
cd code
python3 run_experiments.py   # synthetic + IEEE 33-bus, ~25 min on 8 cores
python3 run_highd.py         # d in {500, 1000, 2000, 4000}
python3 run_kernel.py        # KernelUCB, GP-TS on selected configs
python3 run_sensitivity.py   # lambda + tau_0 ablation
python3 make_figures.py
python3 make_highd_table.py
python3 make_kernel_table.py
```

## Building the paper

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Algorithms implemented

* `FGTSLasso`        Proposed: forced-exploration + LASSO support recovery + feature-Gaussian Thompson Sampling on the recovered subspace.
* `LassoBandit`      Bastani & Bayati (2020). Forced-sample + all-sample LASSO.
* `SALassoBandit`    Oh, Iyengar & Zeevi (2021). Sparsity-agnostic LASSO bandit.
* `ThresholdedLassoBandit`  Ariu, Abe & Proutiere (2022).
* `LinUCB`           Chu et al. (2011). Non-sparse ridge UCB.
* `KernelUCB`        Valko et al. (2013). RBF-kernel UCB.
* `GPTS`             Srinivas et al. (2010). Gaussian-process Thompson Sampling.
* `LinTS`            Agrawal & Goyal (2013). Ambient-space linear Thompson Sampling.

## Citation

```bibtex
@article{suraveikin2026fgts,
  author  = {Suraveikin, Egor and Golubev, Alexey and Sultimov, Roman and Volkov, Alexander and Maximov, Yury},
  title   = {Sparse Linear Bandits with {LASSO} Support Recovery for Online Voltage Regulation},
  journal = {IEEE Control Systems Letters},
  year    = {2026}
}
```

## License

MIT.
