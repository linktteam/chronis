# WP7-R Item 4: N17, a practical route for `m_t_std` (results)

Criteria: `tests/PREDECLARED_CRITERIA_WP7R.md` section 4 (committed before any seed 9101..9112 ran; nothing in it was edited after the results). Scripts: `integration_report/wp7r/scripts/item4_n17_route.py` (coverage / bootstrap / cost), `analyze_item4.py` (prints the words and the recommendation by the pre-declared rule). Evidence: `wp7r_evidence/item4_*`. All numbers below are **RUN** at this package's head (engine unchanged since `d3e0028`, i.e. with the Item 5 rules in place) unless labelled otherwise. **Nothing is integrated. The N17 xfail stays.**

**Looked at before declaring** (section 4 header): WP5 and WP6 N17 documents (READ), the prototype code. No seed 9101..9112 had been run.

## 4.1 Harness control
`T0` (engine at the generator's true parameters): `c2` 0.944, `c1` 0.700, both inside the bands (+-2: [0.9145, 0.9945], +-1: [0.6427, 0.7227]). The harness is valid; every number below was read.

## 4.2 Coverage of shrinkage-fitted fits that are publishable (12 seeds, public path, K = 2, F = 4, L = 2)

| arm | fits | withheld | `c2` all fits | `c1` all fits | seeds with `c2` < 0.80 | publishable n | `c2` / `c1` on publishable |
|---|---|---|---|---|---|---|---|
| `T0` (true parameters) | 12 | 0 | 0.944 | 0.700 | 0 | 12 | 0.944 / 0.700 |
| `shrink5`, 200 iterations | 12 | **1** | 0.933 (se 0.024) | 0.681 | 1 | **11** | **0.934 / 0.681** (both in band) |
| `shrink5`, 20 iterations | 12 | 12 (all `MAX_ITER_REACHED`) | 0.932 | 0.684 | 1 | 0 | not applicable |
| `personal`, 200 iterations | 12 | 12 | 0.589 | 0.363 | 10 | 0 | not applicable |
| `personal`, 20 iterations | 12 | 12 | 0.733 | 0.479 | 7 | 0 | not applicable |

**Word (4.2): "coverage of publishable shrinkage-fitted fits reaches the band"** (11 >= 8 publishable fits, `c2` 0.934 and `c1` 0.681 both inside).

Things the word does not say, all from the same run:
- **The WP5 number is a favourable case and so is this one.** The shared prior is the WP5 prior (centred on the generator's population). Nothing here tests a prior that is off-centre (**UNKNOWN** for N17; WP5 ran shifted and scrambled priors for the score criteria, not for coverage).
- **The gate does not guarantee coverage.** The one seed below 0.80 (9102, `c2` 0.685) is a PUBLISHABLE fit (`CONVERGED`, 92 iterations, not withheld). The one withheld shrinkage fit (9101) was withheld by the new Item 5 duration rule (`degenerate_duration: regime(s) [0]`) although it is `CONVERGED`; its `c2` is 0.920, so the rule removed a fit that was not badly calibrated here (one fit; no conclusion).
- **The coverage comes from the prior, not from convergence.** `shrink5` at 20 iterations (always withheld) has the same coverage as at 200 (0.932 vs 0.933); the personal fits, which have no prior, are far below the band and get WORSE with more iterations (0.733 at 20, 0.589 at 200), and none of them is publishable.
- Per seed `c2` of the publishable shrinkage fits: 0.685, 0.993, 0.982, 0.971, 0.973, 0.969, 0.960, 0.903, 0.980, 0.934, 0.925 (see `item4_analysis.txt` and the jsonl for `c1`). Eleven seeds: the mean over seeds has se 0.026, so "in the band" (half-width 0.04) is a statement about a mean with a standard error of about 0.03, not a tight one.

## 4.3 Parametric bootstrap (seeds 9101..9106, `personal` at 20 iterations, B = 30, warm start)

| seed | T | replicates ok / failed | `E0` `c2` | `EB` `c2` | `E0` `c1` | `EB` `c1` | mean std ratio |
|---|---|---|---|---|---|---|---|
| 9101 | 81 | 30 / 0 | 0.883 | 0.988 | 0.599 | 0.698 | 1.45 |
| 9102 | 65 | 30 / 0 | 0.700 | 0.838 | 0.446 | 0.515 | 1.36 |
| 9103 | 67 | 30 / 0 | 0.821 | 0.933 | 0.522 | 0.813 | 2.33 |
| 9104 | 56 | 30 / 0 | 0.375 | 0.482 | 0.170 | 0.241 | 1.79 |
| 9105 | 51 | 30 / 0 | 0.873 | 0.980 | 0.627 | 0.794 | 1.72 |
| 9106 | 37 | **0 / 30** | 0.635 | none | 0.405 | none | none |

Over the five seeds with a bootstrap: `E0` `c2` 0.730 -> `EB` 0.844 (paired difference +0.114, paired se 0.006, improved on 5 of 5), `c1` 0.473 -> 0.612, mean ratio `m_t_std_boot / m_t_std` 1.73. **Word (4.3): "improves but does not reach"** (`EB` `c2` 0.844 is below 0.9145; counting 9106 as not improved gives 5 of 6, which meets the rule's "at least 5 of 6"; leaving it out changes nothing).

- **Seed 9106 failed for a reason fixed in the design, not by a bug:** the output-space frame removal needs at least `3 (L + 1) = 9` time steps in every regime (section 4.1); the original fit's MAP labels give regimes of 8 and 29 steps (T = 37), so `per_regime_affine_prediction` refuses every replicate. The script records only that a replicate failed, not why; the cause was established afterwards by re-fitting seed 9106 and calling the function (label sizes [8, 29], the ValueError text above). A bootstrap on short series with a small regime therefore returns nothing; no fallback was invented.
- The improvement is uneven: three seeds overshoot to 0.93..0.99, two stay well below (0.84, 0.48). The mean ratio is 1.73 and the variance is large, so the route widens the bands a lot without making them uniformly right.
- These fits are `MAX_ITER_REACHED` and not publishable (the question was only whether the route can close the calibration gap).
- Six seeds is a screening size (pre-declared): even a "reaches" would not have been a pass.

## 4.4 Cost (MEASURED unless stated)

| | P = 50 (K 2, F 4, L 2) | P = 204 (K 3, F 10, L 3) |
|---|---|---|
| one 20-iteration fit | 6.95 s | 5.78 s |
| forward pass / `estimate_states` | 0.095 s / 0.137 s | 0.167 s / 0.247 s |
| numeric Hessian: passes needed | 1351 | 21 217 |
| numeric Hessian: time | **128.2 s measured** (Jacobian 13.2 s measured) | **3551 s EXTRAPOLATED** (passes x per-pass time; Jacobian 101 s extrapolated) |
| bootstrap B = 30 per fit | **87 s measured** in the bootstrap run (4.0 s fit + 30 x 2.9 s warm replicates); 208 s if every replicate cost a full fit | 173 s extrapolated (B x fit time) |
| bootstrap B = 100 per fit | 695 s if every replicate cost a full fit | 578 s extrapolated |

The Hessian route grows with P squared and is an hour per fit at P = 204 (extrapolated); the bootstrap route grows with the fit time only, and both bootstrap numbers are well below the 15 minute line of rule 4.5(1). The 87 s at P = 50 was measured under two-worker load on a 2-core machine, so it is an upper-side figure for that machine.

## 4.5 Recommendation (the rule, applied in order)
1. Bootstrap "reaches the band"? No (0.844). Not taken.
2. 4.2 "reaches the band" with n >= 8? **Yes** -> **"shrinkage-only uncertainty", with the favourable-case caveat.**

What that means for the owner, in plain words: of the routes tried, the only one that reaches the band is the one that comes with the prior (a fit anchored to a shared prior, which then converges within 200 iterations on 11 of 12 seeds). It is a favourable case (a prior centred on the data's own generator), it is not a calibration of the personal fits, and a publishable fit can still undercover (seed 9102). The bootstrap helps (mean +0.114, 5 of 5 seeds that ran) but not enough, and fails outright on short series with a small regime. **Not integrated; the xfail on the N17 cells stays** (the xfail floor is unchanged and no route was run against it). Before anything is integrated, a later package would need: (a) shrinkage coverage on seeds never touched, with a prior that is NOT centred on the generator; (b) a rule for what a user sees when a fit is publishable but its prior-anchored uncertainty has no calibration evidence; (c) the owner's decision on the `n0`/withheld behaviour from Item 3. Numeric confidence stays hidden (contract document 9.4) in the meantime.

## Not verified
- No fresh-seed evaluation of the shrinkage coverage; no off-centre prior; K = 2, F = 4, L = 2 only (P = 204 was timed, not scored).
- The bootstrap used `personal` fits at 20 iterations only, B = 30, six seeds, one of which produced nothing.
- Cost at P = 204 for the Hessian and bootstrap is extrapolated.
