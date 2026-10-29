# WP5 item 5: read-only N17 diagnosis (m_t_std under-coverage). No engine change, no fix.

Design and reading rules R1 to R3 were committed before the run (`tests/PREDECLARED_CRITERIA_WP5.md`, amendment A9, commit 28808ef). Script: `integration_report/wp5/scripts/n17_diag.py` (RUN, about 5 minutes, deterministic: two runs gave identical numbers). Raw output: `wp5_evidence/n17_diag.txt`, per-seed rows `n17_diag_rows.json`. Statistic: share of true latent values inside +-2 `m_t_std`, per latent dimension (nominal 0.9545), with the N17 in-sample per-regime affine alignment for fitted arms; `truth` = engine run at the generator's true parameters, no alignment. Seeds 5001..5024 were unused before.

## Numbers

| | personal | shrinkage (main prior, n0 = 5) | truth parameters |
|---|---|---|---|
| cell A (seed 202), dim 0 / 1 | 0.883 / 0.967 | 1.000 / 0.933 | 0.933 / 0.933 |
| cell B (seed 404), dim 0 / 1 | 0.828 / 0.938 | 0.969 / 0.984 | 0.938 / 0.969 |
| 24 new seeds, mean coverage dim 0 / 1 | **0.675 / 0.677** | 0.928 / 0.933 | 0.951 / 0.950 |
| across-seed sd (binomial sd for independent z: 0.027) | 0.184 / 0.217 | 0.084 / 0.072 | 0.041 / 0.046 |
| median coverage, seeds below 0.80 (of 24), dim 0 | 0.694, 19 | 0.954, 2 | 0.960, 0 |
| mean z^2 (1 if calibrated), dim 0 / 1 | 353 / 41 | 2.1 / 1.6 | 2.7 / 2.7 |
| lag-1 autocorrelation of z, dim 0 / 1 | 0.24 / 0.24 | 0.08 / 0.00 | 0.07 / 0.09 |

Paired differences of mean coverage (24 seeds): personal minus truth -0.276 / -0.273 (se 0.038 / 0.044); shrinkage minus personal +0.253 / +0.256 (se 0.033 / 0.042). Cell B dim 0 (0.828) sits at the 79th percentile of the 24 personal values: **the two N17 cells are milder than a typical seed**, so the N17 failing numbers understate the problem for plain personal fits.

GPB2 against the exact oracle (T = 6, Dmax = 3, 24 random tiny problems, 216 values): median std ratio 0.991, only 46% within 0.95..1.05, min 0.24, max 2.17; smoothed-mean gap median 0.12 exact-std, max 5.8.

## Reading (by the pre-declared rules)

* **R1 (GPB2 collapse too inaccurate): does not fire.** `truth` coverage 0.951 / 0.950 is within 1 se of nominal, and the median GPB2/exact ratio is inside 0.95..1.05. The wide spread of the individual ratios at T = 6 is reported, but it is not what the rule tests, and it is a tiny random problem, not T = 60, Dmax = 25.
* **R2 (missing parameter uncertainty): fires.** `truth` is at nominal, `personal` is 6 to 7 se below `truth`, and `shrinkage` is 7 to 8 se above `personal`.
* **R3 (alignment plus autocorrelated errors): does not fire.** Its first condition (personal at nominal over 24 seeds) fails: 0.675. Lag-1 autocorrelation is 0.24, below the declared 0.3. The across-seed dispersion of `personal` is 7 to 8 times the binomial one, but the `truth` arm already has 1.5 to 1.7 with the same alignment-free statistic, and the personal number is not separable from bad fits (mean z^2 = 353 means a few very large errors against a tiny `m_t_std`).

**The numbers favour "the reported `m_t_std` conditions on the fitted parameters as if they were the truth" (N17 cause 2).** With the true parameters the same smoother is calibrated, so the smoother and the GPB2 collapse are not the main culprit at this size.

## What this does not show

* It cannot separate "parameter variance not propagated" from "personal fits land on a poorly identified or wrongly segmented solution": both give narrow `m_t_std` around a wrong path, and both are cured, here, by a prior that is **centred on the generator** (the population generator is built around the same parameters as the N17 generator: d = -2, +2, dwell 15, K = 2, F = 4, L = 2). So "the prior raises coverage from 0.68 to 0.93" is a favourable-case number, not a general one.
* `shrinkage` itself is slightly below nominal (0.928 / 0.933 vs 0.951 / 0.950 for truth; 2 seeds under 0.80). Both shrinkage cells stopped at MAX_ITER_REACHED.
* The prior does not add parameter uncertainty to `m_t_std` (`prior_uncertainty_in_m_t_std = False`); it changes calibration only by moving the fit. N17 is not fixed and C1 and C3 stay `xfail(strict=True)` (their numbers were not changed by WP5).
* One generator, K = 2, F = 4, L = 2, 20 iterations; no check at other sizes or on real data. GPB2 at the production size (T = 60, Dmax = 25) was not compared with an exact smoother (no oracle exists for it).
* Possible fixes, for WP6 and not tried: propagate parameter uncertainty (bootstrap or Laplace over the M-step parameters, or an inflation factor learned on held-out data); report coverage warnings when the personal fit is degraded or poorly identified; use the shared prior to fix the frame before reporting `m_t_std`.
