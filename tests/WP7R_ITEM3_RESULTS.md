# WP7-R item 3: shrinkage (cold start) under fail-closed. Results

Pre-declaration: `tests/PREDECLARED_CRITERIA_WP7R.md` section 3 (committed in 4acf9d6, before any scored seed; the study script was committed after its smoke test on development seed 7603 and before the run). Script: `integration_report/wp7r/scripts/item3_shrinkage_study.py`; analysis: `analyze_item3.py` with the pure functions of `tests/utils/item3_stats.py` (8 tests, 22 mutants, all killed). Evidence: `wp7r_evidence/item3_stageA.jsonl`, `item3_stageB.jsonl`, `item3_analysis.txt` (full tables). Seeds 7901..7924, 3 lengths (30, 60, 120 training steps = 2, 4, 8 training segments), 24 users each, nothing excluded. Labels: RUN unless stated. Development seeds touched: 7602 (route-equivalence and timing probe) and 7603 (smoke test of the script); neither is part of any number below.

## UPDATE 2026-11-24 (section 3.9): the same study under the FINAL engine. THIS IS THE HEADLINE; the original reading below is kept for the record

Stage A and B ran before the Item 5 rules (dual collapsed-regime R1, degenerate-duration R2) existed, so their "publishable" counts describe an engine that is no longer the final one. After Items 4 and 5 I refitted, with the identical call and seeds, every stage B fit that had been publishable (38 `shrink5` and 2 `personal` fits; pre-declared in 3.9 before the run; `integration_report/wp7r/scripts/item3_restatus.py`). **All 40 refits reproduced the stored record exactly (fit state, iterations run, held-out score), so only the status changed: 18 of the 40 are now withheld** (16 `shrink5`, 2 `personal`). Stage A cannot change (every fit was already withheld). Evidence: `wp7r_evidence/item3_stageB_final_engine.jsonl`, `item3_analysis_final_engine.txt`.

| (stage B, `max_iter` 200, 24 users) | length 30 | length 60 | length 120 |
|---|---|---|---|
| `shrink5` withheld, original engine | 10 (42%) | 11 (46%) | 13 (54%) |
| **`shrink5` withheld, final engine** | **16 (67%)** | **15 (62%)** | **19 (79%)** |
| reasons added by the new rules (some fits carry several) | 4 `degenerate_duration`, 3 `collapsed_regime` | 4, 1 | 6, 0 |
| **publishable `shrink5` fits** | **8** | **9** | **5** |
| gain over the single Gaussian on the publishable fits (mean, paired se) | +1.111 (0.239) | +0.626 (0.302) | +1.284 (0.122) |
| **`mean - 3 se`** | **+0.393** (judged, 8 of 8 favour) | **-0.280** (judged, 8 of 9 favour) | +0.918 (**not judged: 5 < 8**, 5 of 5 favour) |
| risk-coverage: coverage, gain on all 24, `mean - 3 se` | 33%, +0.370, -0.029 | 38%, +0.235, -0.143 | 21%, +0.267, -0.066 |
| `personal` publishable | 0 of 24 | 0 of 24 | 0 of 24 |

* **Pre-declared word: "does not survive"** (length 60 is judged and its bound is -0.280, against -0.001 in the original reading; length 120 can no longer be judged). **Net deployment gain: "not shown", now at every length** (the original reading missed only at length 60): with the new rules the share of users who get a prediction is 21 to 38%, and the single-Gaussian fallback for the others leaves the lower bound below zero everywhere, although the mean is positive everywhere. **The cold-start feature is not reported as working.** The conclusion of the original reading stands and is stronger.
* On the publishable fits `shrink5` still beats the shared-only model on every seed (+3.85, +5.35, +6.43 nats; 8/8, 9/9, 5/5): the personal-prior mixture is better than the prior alone whenever it is publishable.
* The new rules remove 6, 4, 6 `shrink5` fits and the 2 publishable `personal` fits. Post hoc (not pre-declared): the withheld shrinkage fits still predict better than the publishable ones (+1.27, +1.16, +1.34 against +1.11, +0.63, +1.28), the same pattern as before.
* The production gate reading is unchanged except for counts: at length 30, 15 of 24 training sets pass the 30-row cold-start gate and 4 of those are publishable; at 60 and 120 all 24 pass and 9 and 5 are publishable.

## The answer, plainly (ORIGINAL reading: the engine before the Item 5 rules)

**The WP5 cold-start gain does NOT survive on publishable fits by the pre-declared rule, and under the WP5 design (20 EM iterations) nothing is publishable at all.**

* **Stage A (the WP5 design, `max_iter = 20`): 100% of `personal` and of `shrink5` fits are withheld** at all three lengths, so the publishable-fit comparison cannot be judged (0 publishable fits) and there is no deployment gain to show. The WP5 number itself reproduces if withheld fits are counted (shrinkage minus single Gaussian: +1.221, +0.962, +1.327 nats per held-out step at 30/60/120; 23, 23, 24 of 24 seeds favour shrinkage), but none of those fits would be served by `fit_hssm` today. This was the expectation written down in 3.4.
* **Stage B (`max_iter = 200`, the `fit_hssm` default): shrinkage fits are withheld in 42%, 46% and 54% of cases (lengths 30, 60, 120); personal fits in 96%, 100% and 96%.** On the PUBLISHABLE shrinkage fits the gain over the single Gaussian is positive at every length, but the pre-declared floor is missed at one length: 

| length | publishable / 24 | mean held-out score, shrink5 / single Gaussian | gain (mean) | paired se | **mean - 3 se** | seeds favouring shrinkage |
|---|---|---|---|---|---|---|
| 30 | 14 | -4.842 / -5.836 | +0.995 | 0.167 | **+0.494** | 13 of 14 |
| 60 | 13 | -5.086 / -5.713 | +0.627 | 0.209 | **-0.001** | 12 of 13 |
| 120 | 11 | -4.096 / -5.249 | +1.153 | 0.092 | **+0.877** | 11 of 11 |

  Pre-declared word (`survives`: every length judged, every `mean - 3 se > 0`, strictly): **"does not survive"**. The miss at length 60 is 0.001 nats; the point estimate there is +0.63 and 12 of 13 seeds favour shrinkage. I report the word the rule gives and do not soften it; I also do not read the miss as a refutation: with 13 publishable fits the margin is about the size of the rounding. No criterion is changed.
* **Risk-coverage (a withheld fit counts as "no prediction", the single Gaussian is served instead):** coverage 58%, 54%, 46%; gain over the single Gaussian on all 24 seeds +0.580, +0.340, +0.528 nats per step, `mean - 3 se` = +0.160, **-0.047**, +0.149. **Net deployment gain: "not shown"** (length 60 again). The mean gain is positive everywhere.
* Therefore, by the pre-declared sentence: **the cold-start feature is not reported as working** (it needed both "survives" and "positive").

## Withheld rates and reasons (Stage B; Stage A in `item3_analysis.txt`)

| arm | length | withheld | by fit state (withheld fits) | reason prefixes | iterations run (median, min to max) |
|---|---|---|---|---|---|
| shrink5 | 30 | 10 of 24 (42%) | 9 `REJECTED_UPDATE`, 1 `MAX_ITER_REACHED` | all `fit_state_not_converged` | 75 (2 to 200) |
| shrink5 | 60 | 11 of 24 (46%) | 10 `REJECTED_UPDATE`, 1 `MAX_ITER_REACHED` | all `fit_state_not_converged` | 41 (2 to 200) |
| shrink5 | 120 | 13 of 24 (54%) | 9 `REJECTED_UPDATE`, 4 `MAX_ITER_REACHED` | all `fit_state_not_converged` | 106 (2 to 200) |
| personal | 30 | 23 of 24 (96%) | 23 `MAX_ITER_REACHED` | all `fit_state_not_converged` | 200 |
| personal | 60 | 24 of 24 (100%) | 24 `MAX_ITER_REACHED` | all `fit_state_not_converged` | 200 |
| personal | 120 | 23 of 24 (96%) | 23 `MAX_ITER_REACHED` | all `fit_state_not_converged` | 200 |

Stage A: `personal` 24 of 24 `MAX_ITER_REACHED` at each length; `shrink5` 16/14/16 `MAX_ITER_REACHED` and 8/10/8 `REJECTED_UPDATE`. No fit in either stage was withheld for a numerical-health reason (0 degraded fits, 0 exceptions). Publishable shrinkage fits end by `abs_tol` (9, 10, 7 fits) or by the window certification at the cap (5, 3, 4). The shared-only arm is not a fit and is never withheld (it is NOT a publishable model: its held-out score is -10.1, -10.3, -10.6 against -6.1, -5.6, -5.4 for the single Gaussian, and `shrink5` beats it on every publishable seed: +4.82, +4.78, +5.75 nats, 14/14, 13/13, 11/11).

## The other arms

* **personal-only** is publishable in 1 of 24 fits at lengths 30 and 120 and in none at 60, so it cannot be compared with anything under fail-closed. (For all fits, WP5 statistic: -286 and -21 nats per step at lengths 30 and 60 because of a few explosive fits, +1.09 at length 120.)
* **single Gaussian:** the reference; never withheld.
* **Production gate (descriptive):** the cold-start gate (30 present sessions) would refuse the fit before any prior is used. At length 30, 15 of the 24 training sets have at least 30 rows (range 16 to 46 rows) and 8 of those are publishable shrinkage fits; at lengths 60 and 120 all 24 are eligible (13 and 11 publishable). The gate was switched off for this study only (`ColdStartConfig(min_present_sessions = 1)`, default untouched). So the "30 training steps" case is not reachable through `fit_hssm` for 9 of 24 users whatever the prior does.

## POST HOC, not pre-declared (labelled so; no word is derived from it)

Splitting the shrinkage fits of Stage B by whether they were withheld (shrink5 minus single Gaussian, mean and se):

| length | publishable | withheld |
|---|---|---|
| 30 | +0.995 (0.167), n = 14 | +1.524 (0.233), n = 10 |
| 60 | +0.627 (0.209), n = 13 | +1.350 (0.177), n = 11 |
| 120 | +1.153 (0.092), n = 11 | +1.479 (0.130), n = 13 |

The withheld fits predict BETTER than the publishable ones at every length. Most withheld fits are `REJECTED_UPDATE` fits that stopped after 2 or 3 EM iterations (median iterations 2, 2, 3), i.e. they stay close to the prior; the publishable ones ran a median of 148, 99, 151 iterations. Reading (a reading, not a finding): the fail-closed rule removes the shrinkage fits that stay closest to the population prior, and those are the best predictors in this design; the converged fits have moved away from the prior and gain less. Whether a different certification for prior-anchored fits would be sound is a design question for the owner and is NOT answered here (no change was made).

## What was not done / limits

* `personal` uses ONE restart (the WP5 design); the production default is 10 restarts with a preference for a converged restart, which was not tested, so the 96 to 100% withheld rate of `personal` at 200 iterations is for a single restart.
* The prior is the WP5 prior (users 8001..8020, built at the WP5 code), not rebuilt at this head; n0 is not re-tuned; population users only (one generator, centred on the prior's generator: a favourable case, as WP5 already said); the atypical-user check of WP5 criterion 3 is not re-run (not in the brief).
* The held-out comparison on publishable fits conditions on the fit outcome (selection); the paired design compares each seed with the single Gaussian on the same user, which removes the user effect but not a possible link between "converges" and "easy user". The risk-coverage figure is the one that does not condition.
* 24 users, one data-generating process, K = 2, F = 4, L = 2, Dmax = 25.
