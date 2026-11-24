# WP7-R item 1 results: does more EM make the dual model predict worse?

All numbers RUN on 2026-11-17 (and re-analysed 2026-11-24) from `wp7r_evidence/item1_iter.jsonl` (28 seeds), `item1_s3_dmax60.jsonl` (9 seeds), `item1_faithfulness.txt`; the words are those of `tests/PREDECLARED_CRITERIA_WP7R.md` section 1, committed (bfc6b0f) before any scored seed ran. Output of the analysis script: `wp7r_evidence/item1_analysis.txt` (`integration_report/wp7r/scripts/analyze_item1.py`). Nothing was excluded; all 28 fits ended `MAX_ITER_REACHED` at 160 iterations (none stopped by `abs_tol`), all held-out scores are finite.

## Verdict (pre-declared word, section 1.3): **NO CONSISTENT TREND**
Pooled per-seed slope of the held-out score on log2(iterations): -0.026 nats/step per doubling, se 0.021, t = -1.23; negative in 15 of 28 seeds (the declared "lowers" word needed t <= -2, at least 20 of 28 negative and a negative mean in both groups). The mean slope is -0.112 in the N18 group (6 of 8 negative, t = -1.72) and +0.008 in the D-pop group (9 of 20 negative, t = +0.94). So the decline the lost session saw on the 8 N18 seeds is real in that design, it does not generalise to the D-pop design, and inside the N18 group it is carried by two seeds (202: -0.46 and 404: -0.35 per doubling; the other six lie between -0.05 and +0.02). Consequence fixed in advance (1.5): **Item 5c (the dual publish gate) is not built.**

## Held-out score (mean one-step-ahead predictive log density per held-out step), by number of EM iterations
Paired differences are later minus earlier; "better" = seeds whose held-out score rose; training log-likelihood per row rose in **every** seed at every step (28 of 28 each time).

| Group | n | 20 it | 40 it | 80 it | 160 it |
|---|---|---|---|---|---|
| N18 held-out | 8 | -5.740 | -5.959 | -6.009 | -6.095 |
| N18 train ll/row | 8 | -3.113 | -3.094 | -3.079 | -3.061 |
| D-pop held-out | 20 | -3.971 | -3.976 | -3.962 | -3.950 |
| D-pop train ll/row | 20 | -3.499 | -3.481 | -3.463 | -3.445 |
| Pooled held-out | 28 | -4.477 | -4.542 | -4.546 | -4.563 |
| Pooled train ll/row | 28 | -3.389 | -3.370 | -3.353 | -3.335 |

| Group | step | held-out diff: mean | paired se | n | better | train ll/row diff |
|---|---|---|---|---|---|---|
| N18 | 20 to 40 | -0.219 | 0.129 | 8 | 1 | +0.020 |
| N18 | 40 to 80 | -0.050 | 0.043 | 8 | 2 | +0.015 |
| N18 | 80 to 160 | -0.087 | 0.066 | 8 | 3 | +0.017 |
| N18 | 20 to 160 | -0.356 | 0.205 | 8 | 2 | +0.052 |
| D-pop | 20 to 40 | -0.005 | 0.014 | 20 | 13 | +0.018 |
| D-pop | 40 to 80 | +0.014 | 0.009 | 20 | 13 | +0.018 |
| D-pop | 80 to 160 | +0.012 | 0.009 | 20 | 12 | +0.018 |
| D-pop | 20 to 160 | +0.021 | 0.025 | 20 | 11 | +0.054 |
| Pooled | 20 to 40 | -0.066 | 0.041 | 28 | 14 | +0.019 |
| Pooled | 40 to 80 | -0.004 | 0.014 | 28 | 15 | +0.017 |
| Pooled | 80 to 160 | -0.016 | 0.021 | 28 | 15 | +0.018 |
| Pooled | 20 to 160 | -0.086 | 0.067 | 28 | 13 | +0.053 |

No step has a difference beyond 2 paired standard errors in either direction (the largest, N18 20 to 160, is 1.7 se).

## Descriptive facts that Item 5 and the report need (not part of the declared words)
* **Dual against the single Gaussian (held-out):** on the D-pop seeds the dual beats the Gaussian in 20 of 20 seeds at every checkpoint (mean +1.40 to +1.42 nats/step, already at 20 iterations); on the N18 seeds it is worse than the Gaussian on average (-0.68 at 20 iterations, -1.04 at 160) and beats it in 3, 2, 2, 2 of 8 seeds at 20/40/80/160 iterations. So "beats a single Gaussian" is a property of the design, not of the iteration count.
* **Degenerate durations (outside the WP6 box) at the checkpoint:** N18 1 of 8 at every checkpoint (seed 202); D-pop 9, 9, 8, 8 of 20 at 20/40/80/160 iterations.
* **Post-hoc (exploratory, NOT a declared word):** leaving out the 9 seeds that are degenerate at 160 iterations, the pooled mean slope is -0.018 (se 0.021, 11 of 19 negative): still no trend.

## (F) Faithfulness of the checkpoint design (RUN, `item1_faithfulness.txt`)
7 of the 8 N18 seeds give a **bit-identical** held-out score between the shipped path (`fit(n_iter = 20)` with the box refit on; for these 7 also `engine_scores("real", seed)`) and the 20-iteration checkpoint. The one that differs is seed 202 (-6.1889 shipped against -6.1813 checkpoint), the one where the duration-box refit fired, as expected.

## Salvage-note facts
| Fact | Status | Evidence (RUN) |
|---|---|---|
| EM creeps at about 0.995 per iteration; about 1,330 iterations to |dll| < 1e-4 | **RE-VERIFIED** (with different numbers) | Median per-seed rate over iterations 100 to 160 = 0.9923 (28 seeds, none stopped early); median extrapolated iterations to |dll| < 1e-4 = 863 (n = 27; range 430 to 2,670). Declared range was rate 0.99 to 0.999 and count 700 to 2,500. The note's 0.995 and 1,330 are not reproduced to the digit; the creep itself is. |
| About 99% of the remaining gain is in the observation block | **NOT RE-VERIFIED** | Not attempted: needs the Q-block decomposition of the M-step (declared in advance, 1.4). |
| Segments longer than Dmax go with degenerate durations | **RE-VERIFIED** | 28 seeds: degenerate and long 7, degenerate and not long 2, not degenerate and long 2, neither 17; Fisher two-sided p = 0.0009, odds ratio 29.8 (the note's p = 0.015 came from different data and is not claimed). "Long" = a planted training segment above 25 rows. |
| Dmax = 60 cured 8 of 9 | **RE-VERIFIED** (by the declared box rule) | Of the 9 seeds degenerate at 160 iterations with Dmax = 25 (202, 5203, 5204, 5207, 5208, 5211, 5212, 5213, 5215), 8 are inside the box with Dmax = 60; the one that is not is 5215 (its longest segment is exactly 25 rows, so it is not a "long segment" case; one duration at mu = -111, sigma = 10). Two caveats, both post hoc: the box rule only bounds sigma from above, and N18 seed 202 counts as cured although its regime 1 ends at sigma = 0.0 and its held-out score falls from -7.60 to -13.68; on the 8 D-pop seeds the held-out score rises in 7 (mean +0.32 over the 8 non-202 seeds). So raising Dmax removes the box violation for long-segment cases, it is not a general cure, and it does not by itself make a fit publishable (all 9 still end `MAX_ITER_REACHED`). |

## What was not verified
* The Dmax = 60 follow-up covers only the 9 seeds that were degenerate at Dmax = 25; I did not run Dmax = 60 on the other 19 seeds, so I cannot say what it does to fits that were fine.
* One model configuration (K = 2, F = 4, L = 2, Dmax = 25) and two generators; no real-data fit. The checkpoints are pure EM with the duration-box refit off (faithful to the shipped path in 7 of 8 checked seeds; the refit changes a fit only when it fires).
* Held-out scores here use 2 held-out segments (about 24 to 60 steps) per seed; the seed-to-seed spread is large relative to the small per-step differences, which is why no step reaches 2 se.
