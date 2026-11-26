# WP9 A1 (seed stability) and A2 (gate and coverage): results

Design, levels and rules: `tests/PREDECLARED_CRITERIA_WP9.md` sections 6 to 10 and Parts 3 and 8 (committed before any development or evaluation fit); frozen candidate: `tests/DESIGN_FREEZE_WP9_A12.txt`. Labels: **RUN** = executed here, output in `wp9_evidence/`; **READ**; **UNKNOWN**. Everything below is synthetic data (K = 3 planted regimes, seven features, strata hard / medium / easy that are assumptions of the study); how many real people look like each stratum is **UNKNOWN**. No target or level was moved after a result was seen.

## What was built (READ)
- **A1 rule C** (`backbone/hssm/fit_stability.py`, `fitting._restart_summary`, `lineage`): a baseline fit is `FIT_UNSTABLE` when fewer than 3 healthy restarts exist or when the best 3 healthy restarts (by log-likelihood) do not all agree at adjusted Rand index >= 0.99. The flag blocks "same as before" (`fit_unstable`, `anchor_fit_unstable`); it does not withhold insights and does not change the selected model. A flagged anchor is replaced by the first later stable fit (Part 3, dated before any result).
- **A2 gate** chosen by the frozen rule on the development users 9301..9360: `AlignmentConfig.min_separation` 4.0 -> **2.5**, plus a posterior-sharpness floor 0.95 (the FIT_UNSTABLE reason `low_posterior_sharpness`, `config.FIT_MIN_POSTERIOR_SHARPNESS`). Both are changes of a default and are listed in the CHANGELOG and the report.

## Development (RUN, `wp9_evidence/a2_dev_analysis.txt`, users 9301..9360, 960 fits)
- Rule of section 8: calibration half (even users) zero wrong-and-unflagged, then highest calibration coverage; validation half <= 1%. **Chosen h = 2.5, s = 0.95, flag on** (calibration coverage 42.6%, validation 49.3%, 0 wrong-and-unflagged on both halves). The WP8 default (h = 4.0) had pooled coverage 41.3%; the chosen candidate 45.9%.
- Development rule C check (14-day chains, two fit seeds): identical lineage among served 121/142 = 85.2% at h = 2.5; fit level 153/156 = 98.1% same partition among unflagged pairs. Reported, not used to tune.

## Evaluation (RUN once at the frozen candidate, users 9401..9448, 768 fits, `wp9_evidence/a2_eval_analysis.txt`)
Primary = the first 10 restarts (the production default); secondary = the first 20.

| level (declared before any fit) | 10 restarts | 20 restarts | verdict |
|---|---|---|---|
| A2 coverage >= 80% (accepted refits ending ALIGNED) | 214/432 = **49.5%** (CI 44.8-54.2%) | 251/432 = 58.1% (CI 53.4-62.7%) | **NOT MET** |
| A2 wrong-and-unflagged <= 1% of alignments | 0/432 = 0.0% (Wilson upper 0.9%) | 0/432 (upper 0.9%) | **MET** |
| A1-1 identical lineage among served >= 95% (strict: both seeds ALIGNED, >= 99% of days named identically) | 92/136 = **67.6%** (CI 59.4-74.9%) | 128/155 = 82.6% (CI 75.8-87.7%) | **NOT MET** |
| A1-2 ALIGNED steps with a flagged fit or anchor = 0 | 0 of 230 | 0 of 284 | **MET** |
| A1-3 served share (reported, no threshold) | 49.5% | 58.1% | reported |
| A1-4 bit-identity of previously healthy fits vs WP8 (7887118) | 12/12 (fast battery) and 12/12 (full battery) | | **MET** |
| A1-5 mutants killed | see "Mutation checks" below | | |

Per stratum (10 restarts, coverage / wrong-and-unflagged): hard 21/144 = 14.6% / 0, medium 86/144 = 59.7% / 0, easy 107/144 = 74.3% / 0. At 20 restarts: hard 23.6%, medium 66.0%, easy **84.7%** (all with 0 wrong-and-unflagged). Only the easy stratum at 20 restarts meets both numbers, and only on the point estimate (the 95% interval for coverage, 78.0-89.7%, reaches below 80%).

Correct among served: 214/214 at 10 restarts and 251/251 at 20 (no served "same as before" was wrong). What the gate and the flag withheld: 172 right and 46 wrong matches at 10 restarts (pooled), so the flag and gate stopped 46 wrong claims at the price of 172 right ones.

### Fit-level agreement (RUN)
When both seeds' fits of the same window are unflagged, the selected partitions are the same (ARI >= 0.99) in 128/130 = 98.5% (10 restarts) and 178/179 = 99.4% (20 restarts) of windows. The consensus rule therefore does what it was designed to do at fit level. The failure of A1-1 is not in the partitions.

### Why A1-1 is not met (RUN, `wp9_evidence/a1_eval_identity_split.txt`)
Of the 136 strict cases (10 restarts), 92 were served by both seeds with the same names, **2 were served by both with different names**, and **42 were served by one seed only** (the other seed said "cannot confirm"). At 20 restarts: 128 / 1 / 26 of 155. So the criterion fails because two runs of the same user can disagree on whether the fit is good enough to serve (a flag decided by three restarts out of ten), not because they name regimes differently when both serve. That matters for the product: the cost is inconsistency of the *offer* of continuity, not wrong names, but 2 of 136 (1.5%) cases with different names when both serve is still above zero.

### Flag on versus flag off (RUN, reported, nothing chosen from it; `wp9_evidence/a2_eval_flagoff.txt`, `wp9_evidence/a1_eval_flagoff_identity.txt`)
| gate h, 10 restarts, pooled | flag ON (s = 0.95): coverage / wrong-and-unflagged | flag OFF: coverage / wrong-and-unflagged |
|---|---|---|
| 0.0 | 221/432 = 51.2% / 0 | 411/432 = 95.1% / 34 (7.9%) |
| 2.0 | 216/432 = 50.0% / 0 | 384/432 = 88.9% / 17 (3.9%) |
| **2.5 (frozen)** | **214/432 = 49.5% / 0** | **333/432 = 77.1% / 0** |
| 3.0 | 210/432 = 48.6% / 0 | 296/432 = 68.5% / 0 |
| 4.0 | 183/432 = 42.4% / 0 | 238/432 = 55.1% / 0 |

At 20 restarts: flag on h = 2.5 58.1% / 0; flag off h = 2.5 73.8% / 0; flag off h = 2.0 88.0% / 14 wrong.

**Seed-to-seed agreement, flag on versus off (RUN, `wp9_evidence/a1_eval_flagoff_identity.txt`, frozen gate h = 2.5, strict cases = 14-day windows where at least one of two fit seeds ends ALIGNED):**

| 10 restarts | strict cases | same names (both serve) | both serve, **different names** | one seed only serves |
|---|---|---|---|---|
| flag ON (s = 0.95) | 136 | 92 (67.6%) | **2** | 42 |
| flag OFF | 190 | 150 (78.9%) | **27** (14.2%) | 13 |
| flag ON, 20 restarts | 155 | 128 (82.6%) | 1 | 26 |
| flag OFF, 20 restarts | 190 | 162 (85.3%) | 13 (6.8%) | 15 |

So the flag trades cases where two runs of the same user would give different names (27 of 190 without it, 2 of 136 with it) for cases where one run says "cannot confirm" (13 without it, 42 with it). The flag-off agreement of 78.9% is also below the 95% level; **neither configuration meets A1-1**. By stratum (10 restarts, flag off): easy 73/80 = 91.2%, medium 53/71 = 74.6%, hard 24/39 = 61.5% same names.

## The 80% / 1% question, stated plainly
- **On the pooled mix of this study (a third hard, a third medium, a third easy), 80% coverage and at most 1% wrong-and-unflagged are not achieved together at the frozen design.** Coverage is 49.5% (10 restarts) with 0 wrong-and-unflagged. With the consensus flag on, no gate value reaches 80% (the highest, at h = 0, is 51.2%, and at 20 restarts 63.2% with 2 wrong).
- With the flag **off**, the wrong-and-unflagged rate is 0 at h = 2.5 and above, and coverage there is 77.1% (10 restarts), still below 80%; at h = 2.0 coverage is 88.9% but 3.9% of alignments are wrong. The candidate grid has nothing between 2.0 and 2.5; a gate between them with the flag off might reach both numbers on these 48 users, but picking it from the evaluation numbers would be choosing on the evaluation users, so it is **not claimed** and would need new users (WP10).
- **By stratum the answer is different:** the easy stratum meets both on the point estimate (flag on at 20 restarts 84.7% / 0; with the flag off at h = 2.5, 144/144 = 100% / 0), the medium stratum with the flag off (123/144 = 85.4% / 0, 10 restarts) and not with it on (59.7%), and the hard stratum meets coverage in neither configuration (14.6% flag on, 45.8% flag off at h = 2.5; 66/144) because fits of weakly separated users do not recover the planted regimes and cannot be trusted to name them.
- The 80% target is therefore a statement about the share of identifiable people; **whether 80% of real users are easy or medium is UNKNOWN** and cannot be settled by synthetic data. The hard stratum is the reason the pooled number is low, and it is low by design: a fit that cannot tell its regimes apart must not claim continuity.

## Open owner decision (new; not decided by me)
The A1 consensus flag costs about 28 coverage points at the frozen gate (49.5% on versus 77.1% off, both 0 of 432 wrong-and-unflagged against the planted truth on these users) and does not by itself deliver the 95% identical-lineage level (67.6% / 82.6%). What it buys, measured on the same fits: with it, two runs of the same user gave different names in 2 of 136 served-by-at-least-one cases; without it, in 27 of 190. Options for the owner:
1. keep the flag on (the default I leave in place: the declared rule chose it, and the evaluation is not used to choose);
2. switch it off: about 28 points more coverage on this mix, but seed-dependent names for roughly one served case in seven (10 restarts);
3. more restarts (20 restarts, flag on: 58.1% coverage, 82.6% identical lineage, twice the EM time);
4. a different consensus rule or a gate between 2.0 and 2.5 tested on new users (WP10).

## What this does not show
- The identical-lineage level and the coverage level are judged on 48 synthetic users (16 per stratum); intervals are wide per stratum, and the stratum mix is an assumption.
- Real users have drift, missing data and regime counts that change; none of that is in these 48 chains beyond the generator's own settings.
- `FIT_UNSTABLE` was tuned for stability of the partition, not for correctness: a fit that is stable and wrong is not caught by it (0 such cases among served in this study).

## Mutation checks (RUN on the final head)
- A1 consensus function and its use in `fitting` and `lineage` (`mutation_a1.py`, `wp9_evidence/mutation_a1.txt`): **28 of 28 killed** (target: at least 15, all killed). The first re-run on the final head applied only 25 because three text patterns no longer matched after the A3 change to the same lines (`k_change=` argument); the three patterns were repaired without changing what they mutate, and the re-run applied and killed all 28.
- A2 scoring and selection rule (`mutation_a12_scoring.py`, `wp9_evidence/mutation_a12_scoring.txt`): **26 of 26 killed**.
- A3 (`mutation_a3.py`, `wp9_evidence/mutation_a3.txt`): **34 of 34 killed** (one pattern had to be repaired the same way).

## Artifacts
`wp9_evidence/a12_eval_fits.jsonl` (768 fits), `a2_eval_analysis.txt`, `a2_eval_curve_n10.json`, `a2_eval_curve_n20.json`, `a2_eval_flagoff.txt`, `a1_eval_identity_split.txt`, `a1_eval_flagoff_identity.txt`, `a1_battery_fast.txt`, `a1_battery_full.txt`, `a2_dev_analysis.txt`, `a2_dev_curve.json`; scripts in `integration_report/wp9/scripts/` (`a12_run.py`, `a12_analyze.py`, `a12_eval_flagoff.py`, `a12_eval_identity_split.py`, `a12_eval_flagoff_identity.py`); tests `tests/test_wp9_a1_*.py`, `tests/test_wp9_a2_sharpness.py`, `tests/test_wp9_a12_analyze.py`.
