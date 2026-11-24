# WP7-R Item 6b: n0 = 1 and 2 against the frozen n0 = 5 (results)

Criteria: `tests/PREDECLARED_CRITERIA_WP7R.md` section 6 (committed before any fit of this item; not edited). Script: `integration_report/wp7r/scripts/item6b_n0_study.py`. Evidence: `wp7r_evidence/item6b_dev.jsonl`, `item6b_dev_analysis.txt`. **RUN** at the package head (`0eab6af`); development seeds 6001..6012, the WP5 design (class path, 20 EM iterations, population users of 30, 60, 120 steps, all fits counted, fail-closed NOT applied: this item applies the WP5 rule, as declared).

## Result
Shrinkage minus single Gaussian, `mean - 3 se` (the WP5 statistic), and the n0 score = the minimum over the three lengths:

| n0 | length 30 | length 60 | length 120 | score |
|---|---|---|---|---|
| 1 | -0.391 | +0.156 | +0.851 | -0.391 |
| 2 | +0.092 | +0.483 | +0.920 | +0.092 |
| **5 (re-run at this head)** | +0.106 | +0.569 | +0.885 | **+0.106** |

Means (seeds favouring): n0 1: +0.540 / +0.682 / +1.317 (8, 11, 12 of 12); n0 2: +0.928 / +0.931 / +1.343 (11, 12, 12); n0 5: +1.018 / +1.131 / +1.329 (11, 12, 12).

**Rule 6b.1: neither n0 = 1 nor n0 = 2 has a strictly larger score than n0 = 5. Nothing further was run: seeds 7501..7524 stay unused and the default `SHARED_PRIOR_DEFAULT_N0 = 5.0` stands (it was never going to change in this item).**

## Reading, and what it does not say
- The rule's answer is clear for n0 = 1 (its length-30 bound is negative) and close for n0 = 2 (+0.092 against +0.106 at the binding length 30, a difference of 0.014 on a standard error of about 0.3): on this evidence n0 = 2 is no better, and not detectably worse, than 5. The data cannot separate them; the frozen value was kept because the rule requires a strictly larger score, not because 2 was shown worse.
- Reproduction of WP5: the re-run n0 = 5 bounds are +0.1063 / +0.5686 / +0.8854, the stored WP5 values +0.1046 / +0.5686 / +0.8857. They agree to 0.002, not bit for bit (the WP5 numbers came from the WP5 worktree; other WP6/WP7-R changes sit between). The cause was not investigated. The rule used the re-run numbers, as declared.
- All 108 shrinkage fits (3 n0 x 3 lengths x 12 seeds) are `MAX_ITER_REACHED` at 20 iterations, so every one is withheld under the current fail-closed rule (class-path proxy: not `CONVERGED` or degraded), the same finding as Item 3 stage A. This item is therefore about the WP5 score statistic only; it says nothing about what a user would be shown.
- Twelve development seeds, one population, one prior (the WP5 prior, centred on the generator): a statement about this grid and this prior only.
