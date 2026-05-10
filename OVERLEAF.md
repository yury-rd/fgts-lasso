# Overleaf submission instructions

This repo is configured for direct Overleaf import. Three options:

## Option 1 — Import from GitHub (easiest, requires Overleaf paid plan)

1. Overleaf → New Project → **Import from GitHub**
2. Authorise Overleaf for GitHub if needed.
3. Paste repo URL: `https://github.com/yury-rd/fgts-lasso`
4. Open `main.tex`. Set compiler to **pdfLaTeX** (Menu → Compiler).
5. Set BibTeX as the bibliography backend (Menu → BibTeX).

## Option 2 — Upload zipped bundle (works on free plan)

1. Download `fgts-lasso-overleaf.zip` from the repo root (or build from `overleaf_bundle/`).
2. Overleaf → New Project → **Upload Project** → select the zip.
3. Open `main.tex`. Compiler: pdfLaTeX. Bibliography: BibTeX.
4. Click *Recompile*; output should be 7 pages.

## Option 3 — Manual upload

Upload these files into a fresh Overleaf project, preserving folder structure:

```
main.tex
refs.bib
ieeecolor.cls
lcsys.sty
LOGO-lcsys-web.eps
figures/
  synth_regret_grid.pdf
  power_regret.pdf
  power_voltage.pdf
  regret_vs_d.pdf
  highd_regret_runtime.pdf
  sensitivity.pdf
  conformal_coverage.pdf
results/
  tables_synth_regret.tex
  tables_power.tex
  tables_highd.tex
  tables_kernel.tex
```

## Compile settings (all options)

* **Compiler**: pdfLaTeX
* **Bibliography**: BibTeX
* **TeX Live**: 2023 or later (default Overleaf)

Compile order: `pdflatex → bibtex → pdflatex → pdflatex`.

## Expected output

* 7 pages
* IEEE Control Systems Letters template (blue accents, IEEE CSS logo)
* Title: *"Conformal Sparse Bandits for Online Voltage Regulation in Distribution Grids"*
* 5 authors listed: E. Suraveikin, A. Golubev, R. Sultimov, A. Volkov, Y. Maximov
