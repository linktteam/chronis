# WP7-R item 2 results: K selection with the collapse rule (baseline model)

RUN on 2026-11-24 from `wp7r_evidence/item2_kselect.jsonl` (48 datasets: 24 seeds 7801..7824 x 2 designs, each a full `fit_hssm(candidate_ks=(2,3,4), n_initializations=10, max_iter=60)`; every fit ended `CONVERGED`, no errors). Words and thresholds: `tests/PREDECLARED_CRITERIA_WP7R.md` section 2, committed (7020e65) before any seed ran. Analysis output: `wp7r_evidence/item2_analysis.txt` (`integration_report/wp7r/scripts/analyze_k_select.py`). Validity gate (2.4): the offline re-derivation at 4 days reproduces the actual selected K, collapse flags, BICs and collapsed-withheld status in **48 of 48** datasets, so the sensitivity numbers below are claimed.

## 1. Planted-K recovery (served K equals planted K; Wilson 95% interval)
| Design | Overall | planted 2 | planted 3 | planted 4 | Pre-declared word |
|---|---|---|---|---|---|
| easy (means 2.5, noise 0.5) | 20/24 (0.64 to 0.93) | 8/8 | 7/8 | 5/8 | **BIC recovers the planted K** (needed 20/24 and 5/8 each: met exactly) |
| hard (means 1.25, noise 0.8) | 8/24 (0.18 to 0.53) | 8/8 | 0/8 | 0/8 | **BIC does not reliably recover the planted K** |
Selected K by planted K, hard design: planted 3 gives K = 2 in 8 of 8; planted 4 gives K = 2 in 7 and K = 3 in 1. Easy design: the 4 misses are all under-selections (3 as 2; 4 as 2, 2 and 3). **BIC never over-selected K in any of the 48 datasets**; it errs toward fewer regimes, and when regimes overlap it errs a lot.

**Post-hoc check (not a declared word; found after the results, flagged as a design flaw of my generator):** the planted path is cut at T = 75 rows and cycles through regimes, so in the easy and hard designs alike 4 of the 24 datasets do not contain all planted regimes (planted K = 3 with 2 regimes present: 1 dataset; planted K = 4 with 3 or 2 present: 3 datasets; seeds 7815, 7819, 7822, 7824). Among the 20 datasets in which every planted regime occurs, easy recovery is **20 of 20**, and the selected K equals the number of regimes actually present in 23 of 24 datasets; hard recovery is 8 of 20 (the 8 are the planted-K = 2 datasets). So the easy-design "misses" are mostly not BIC failures, and the hard-design failure is real. The declared word for the easy design was reached with the planted label and is reported as declared.

## 2. Collapse handling
* Hard criterion (i), served collapsed fits: **0** (both designs). Hard criterion (ii), a collapsed candidate K chosen while a healthy one exists: **0**.
* (iii) **Never triggered.** No candidate K had a collapsed selected model at the 4-day threshold in any of the 48 datasets (smallest minimum effective days over all candidate fits: 4.04 days, K = 4 easy; 4.95 hard), so the demotion rule changed no choice and neither helped nor harmed. This study therefore **does not exercise the demotion path**; the WP6 unit tests do, this study does not add evidence either way. (At 6 days one hard dataset has a collapsed candidate, still without effect.)
* Withheld rate: **0 of 48** (all `CONVERGED`, none collapsed). Every fit, including the wrong-K ones, was served.

## 3. Served ARI at the selected K (argmax of the filtered posterior against the planted path)
Easy: median **1.000**, minimum 0.669 (n = 24, all served) -> pre-declared word **"served ARI is adequate"**. Hard: median 0.732, minimum 0.349 (reported only, as declared). **A fit with the wrong K is served with no flag in this study**: in the hard design 16 of 24 are served with the wrong K.

## 4. Sensitivity to the 4-day rule (measurement only; the rule is unchanged)
| Threshold | Recovery easy | Recovery hard | Withheld by collapse (easy, hard) | Selected K differs from the 4-day choice |
|---|---|---|---|---|
| 3 days | 20/24 | 8/24 | 0, 0 | 0 of 48 |
| 4 days | 20/24 | 8/24 | 0, 0 | (reference) |
| 6 days | 20/24 | 8/24 | 0, 0 | 0 of 48 |
The rule is not exercised by these data at any of the three thresholds, so this study cannot say whether 3, 4 or 6 days is better; it says the 4-day rule does not interfere with K selection or withholding on well-populated regimes (T = 75). The ARI at 3 and 6 days is not reported (alternative restarts' labels are not stored).

## 5. P1-01: is the BIC parameter count right?
**No** (derived from the code before running, `PREDECLARED_CRITERIA_WP7R.md` 2.1): `n_params() = (K-1) + K(K-1) + 2KF + 2K`; the M-step keeps a zero transition diagonal, so the free transition entries are `K(K-2)`; the code over-counts by **K** (F = 7: 35 / 56 / 79 counted against 33 / 53 / 75 free for K = 2 / 3 / 4), and the penalty by `K ln(n)` (ln 75 = 4.3). The structural count of a fitted model equals the free count (test). **Effect:** recomputing every dataset's BIC with the free count changes the selected K in **0 of 48** datasets (recovery 20/24 and 8/24 unchanged); the smallest gap between the two lowest BICs is 14.6 (easy) and 4.8 (hard), against a possible shift of 4.3 per K step; one hard dataset is within 8.6. Declared word: **"the count does not move selection in this study."** Change made: an opt-in class attribute `GaussianHSMM.n_params_convention` (`"legacy"` default, `"free"`), documented in the contract doc (9.8); 6 tests, 13 mutants all killed; default numbers bit-identical (3 of 3 dataset records, covering 90 restart fits, identical between `git archive c95cb5d` and this tree). **Owner decision needed only if you want the default switched to `"free"`: it is demonstrably more correct and did not change a single choice here, but it would change model selection in principle.**

## 6. Reading for the owner
1. On well-separated regimes with all regimes present the baseline selects the planted K in 20 of 20 datasets and serves ARI 1.0 at the median.
2. When regimes overlap, BIC under-selects (served K = 2 for planted K = 3 and 4 in 15 of 16 datasets) and the result is served without any flag. That is a model-selection limit, not a collapse problem, so the collapse rule does not catch it. If a regime count is ever shown to a person, it should not be presented as established (contract doc 9.8).
3. The P1-01 count error is real, small and harmless in this study.

## What was not verified
* The collapse demotion and the 4-day threshold are not exercised by this study (no collapsed candidates); a design with rare or short regimes would be needed (not run; not declared).
* One generator, F = 7, T = 75, 8 datasets per planted K and design; 4 of the 24 planted paths do not contain every planted regime (flagged above); no real data; the dual model is not touched.
