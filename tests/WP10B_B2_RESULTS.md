# WP10B B2 results: opt-in fixes for the dual engine, development only (2026-11-27)

Every number: Python 3.12.3, repo pins, the development 2-core AVX-512 machine. Design, levels and predictions: `tests/PREDECLARED_CRITERIA_WP10B.md` Part 4 (section 10), committed before the first B2 fit. Users 11201..11236 (12 per stratum), setting S0 (identity streams, K = 3, F = 7, train days [0, 120)), 144 variant fits, each paired with the same user's S0 fit of B1; 0 failed to run, 0 stopped by the wall-clock guard. **Development seeds only: no claim about unseen users rests on this note**; the frozen variant meets evaluation seeds once, in the B3 S0 arm (owner decision W10B-1). Analysis: `wp10b_evidence/b2_dev_analysis.txt`, `.json`. Freeze: `tests/DESIGN_FREEZE_WP10B_B2.txt`.

## Result (RUN, development)
| variant | publishable of 36 (S0: 4) | gained / lost | exact one-sided sign p | paired held-out change vs S0, 95% lower bound (level -0.10) | class | median fit time vs S0 |
|---|---|---|---|---|---|---|
| D60 (Dmax 60) | 3 | 0 / 1 | 1.00 | +0.034, -0.0006 | NO EFFECT DEMONSTRATED | 1.11 |
| **I600** (600 iterations, 900 s guard) | **21** (58%, Wilson 42% to 73%) | **17 / 0** | 7.6e-6 | -0.027 (se 0.017), -0.055 | **HELPED** | 3.18 |
| REG (duration prior, 1 pseudo-segment) | 6 | 4 / 2 | 0.34 | -0.021 (se 0.055), -0.112 | HARMED | 1.10 |
| ALL | 22 | 19 / 1 | 2e-5 | -0.027 (se 0.069), -0.140 | HELPED BUT HARMFUL | 3.12 |
By stratum, I600 publishable (S0 to I600): hard 2 to 8, medium 2 to 8, easy 0 to 5 (gains 6, 6, 5; no losses). Mean held-out change I600 minus S0: hard -0.040, medium -0.005, easy -0.036.
Non-publishable under I600 (15; a fit can carry several reasons): rejected update 7, iteration cap 7, degenerate duration 2. Under S0 it was 28 at the iteration cap and 5 rejected updates.

## What it means (INFERENCE where marked)
- **The cause named in B1 is the iteration budget, not the durations.** 32 of 36 S0 fits stop on the cap or on a rejected update, and a larger budget alone turns 17 of them into publishable fits with no loss (the converged S0 fits stay converged). Dmax = 60 and the duration prior, which the brief's WP7-R population numbers suggested, do nothing useful on S0 identity streams (1 of 108 regimes was degenerate in B1), and the prior harms held-out prediction on the declared level. Prediction 10.4 for REG (|g - l| <= 1) failed (observed 2); the others held.
- **"Publishable" is convergence, not usefulness.** I600 lowers the paired held-out mean by 0.027 nats per step on average (se 0.017); the declared "not harmed" level is met (lower bound -0.055 against -0.10), but this is no improvement in prediction. Whether a published dual fit predicts well is B3's and B4's question, on evaluation seeds.
- **SQUAREM-S3 was not built** (declared trigger: at least 10 fits still at the cap and a time ratio of at least 3; observed 7 and 3.18). The 7 fits that still hit the cap and the 7 rejected updates are not addressed by this package.
- **Cost:** about 3.2 times the S0 fit time per dual fit (median), which matters for the B3 S0 arm (about 3 times the declared 75 minutes at most) and for anything that refits (the certificate refits three times).
- **Not shown:** development users only; one setting (S0); one budget level (600) and one prior strength (1.0), not tuned; cross-machine reproducibility is not claimed; the effect on real users is UNKNOWN. Default untouched: the dual engine stays research-only and every switch is opt-in.

## Bit-identity of the default path
PENDING at the time of writing (this section is replaced by a dated addendum with the `dual_battery.py` result before the item is exported). The defaults of the dual engine are not changed by B2; the opt-in switches are class attributes and fit arguments set only by `dual_lib.VARIANTS`.

## Tests and mutation controls (earlier, at the B2 commits)
`tests/test_wp10b_b2.py`, `tests/test_wp10b_b2_analyze.py` (11), `tests/test_wp10b_dual_lib.py` (21); mutation sets `mutation_wp10b_b2.py` (19/19) and `mutation_wp10b_b2_analyze.py` (22/22); rerun at the end of the package.
