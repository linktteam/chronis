# WP10B C1(2) results: does the dual certificate refuse a MILDLY wrong prior? (2026-11-27)

Every number: Python 3.12.3, repo pins (numpy 2.2.6, scipy 1.15.3), the development 2-core AVX-512 machine (`tests/WP10B_C1_CPU_FEATURES.md`). Design and rule: `tests/PREDECLARED_CRITERIA_WP10B.md` section 4 and Part 1b (section 7), frozen in `tests/DESIGN_FREEZE_WP10B_C1.txt` after the development run and before the evaluation. Development users 11001..11012 (36 fits per candidate; no claim rests on them). Evaluation users 11021..11060 (40 users, 3 series lengths 30 / 60 / 120, 120 fits per arm, 360 fits in all), **analysed once** (`wp10b_evidence/c1_eval_analysis.txt`, `.json`). Owner decision W10B-1 applies: the evaluation was read once and nothing was changed after it. The pre-declared extension (seeds 11061..11080) was NOT needed and not run.

## The question
WP10A showed that the certificate refuses a prior that is wrong by +16 on every intercept (H3, norm 45). Is that a real refusal or only a refusal of an obviously bad prior? C1(2) builds a milder wrong prior, checks on development users that it really harms (fits score below the same user's single Gaussian) and that those harmed fits reach the certificate stage, and then asks how many of them are certified.

## Verdict (declared rule E1)
**MET.** Frozen candidate M1 (every intercept + 8, norm 22.63; milder than H3 by 2x). Among the 120 M1 fits, 47 score below the single Gaussian on the held-out days; 13 of those are eligible for the certificate (dual fit, EM state REJECTED_UPDATE or MAX_ITER_REACHED, no other withhold reason). The blind count of 13 meets the required 12. **0 of the 13 are certified** (`CERTIFIED_PRIOR_ANCHORED`); exact one-sided 95% upper bound on the certification rate **20.6%** (0 of 12 would give 22%; the declared "refusal" statement is therefore a bound, not a rate of zero).
The wording the rule allows: **"the certificate refuses a mild harmful prior in this setting"**, never "refuses any harmful prior".

## Table (RUN; evaluation set)
| item | value |
|---|---|
| arm M (M1) fits, with a usable single-Gaussian score | 120, 120 |
| worse than the single Gaussian (held-out one-step mean log density) | 47 (length 30: 31, 60: 13, 120: 3) |
| of those eligible for the certificate (the blind count) | 13 (length 30: 10, 60: 3, 120: 0) |
| E1: certified among the 13 | 0; bound 20.6%; MET |
| E3 (reported): arm-M fits that score below the single Gaussian and are published after the certificate | 0 of 47 |
| E0 (reported): arm T (the true prior), eligible / certified | 31 / 13 (42%); the refusal test is not vacuous |

## What it means (INFERENCE where marked)
- **The refusal was tested where the harm lives:** 10 of the 13 eligible harmful fits are 30-day series, 3 are 60-day, none is a 120-day series (at 120 days the data override the prior: 3 harmful fits, none eligible). So the result is about short series. At length 120 there is nothing to refuse.
- **The certificate is not simply refusing everything:** with the true prior 13 of 31 eligible fits are certified (E0), and 5 of the 39 eligible M1 fits were certified, all of length 120 and none of them worse than the single Gaussian (descriptive count from the same file, not used for any verdict).
- **A bound, not a proof:** 13 fits and 0 events do not exclude a certification rate up to about 21%. The development run points the same way but is thinner: of the 6 eligible harmful M1 fits there, 1 was certified (a development record only; no claim rests on it, and it was recorded in the design freeze as a reason why E1 NOT MET was a live possibility). I do not pool development and evaluation (the development users chose the candidate).
- **What it does not show:** one mechanism of mildness (a constant shift of the intercepts), one setting (K = 2, F = 4 shared prior, N18 generator, identity streams are not used here), three lengths. M3 (a regime swap, norm 1.73) did not harm on development (3 of 36 fits below the single Gaussian) and was not evaluated, so a prior that is wrong by exchanging regimes is NOT covered. "A prior from a different generator family" and "a very different regime count" cannot be built here without a new generator (said in the design). Not tested: any prior from real users. Cross-machine reproducibility is not claimed.

## Tests and mutation controls (earlier in this package)
`tests/test_wp10b_c1.py` pins the rules on synthetic row sets (validity (b) and (c), the mildest-wins choice, the blind count, the extension trigger, the verdict wording); the mutation set `mutation_wp10b_c1.py` and the cross-machine tests were run at their commits and are rerun at the end of the package.
