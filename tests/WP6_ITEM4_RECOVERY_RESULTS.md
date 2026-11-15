# WP6 item 4: recovery study, before and after (RUN)

Design and criteria S1 to S7: `tests/PREDECLARED_CRITERIA_WP6.md` section 6 (written before the script). BEFORE = `git archive 871a3ab` (WP5 head), AFTER = WP6 head at a7bb04e (no engine change since). 24 datasets per arm, seeds 7201..7224, one script (`integration_report/wp6/scripts/recovery_study.py`) run on both trees. Raw lines: `wp6_evidence/recovery_{before,after}.jsonl`; table: `wp6_evidence/summary_item4_recovery.txt`. B-F2 and B-F3 are the baseline arms (K = 3 fixed by design, 10 restarts), D-pop is the dual arm (one fit, 200 iterations, the WP5 population generator).

## Before / after, all arms

| arm | | CONVERGED | no fit | degenerate | degenerate AND served | withheld | served | ARI (served) | s per dataset |
|---|---|---|---|---|---|---|---|---|---|
| B-F2 | BEFORE | 23 | 1 | 6 | 6 | 0 | 23 | 0.917 | 3.6 |
| B-F2 | AFTER | 24 | 0 | 0 | 0 | 0 | 24 | 0.940 | 5.9 |
| B-F3 | BEFORE | 23 | 1 | 1 | 1 | 0 | 23 | 0.725 | 3.9 |
| B-F3 | AFTER | 24 | 0 | 0 | 0 | 0 | 24 | 0.728 | 5.2 |
| D-pop | BEFORE | 0 (24 MAX_ITER_REACHED) | 0 | 7 | 7 | 0 | 24 | 0.772 | 58.9 |
| D-pop | AFTER | 2 (22 MAX_ITER_REACHED) | 0 | 0 | 0 | 22 | 2 | 0.005 (n = 2) | 76.3 |

D-pop BEFORE served all 24 fits although none had converged and 7 had durations outside the box; AFTER 22 of 24 (92%) are withheld because the fit is not converged at 200 iterations (the withheld rate in this population is 92%), 2 are certified by the window rule, 7 were refitted inside the duration box and none is degenerate. The ARI of all dual fits (withheld ones included) is 0.772 before and 0.772 after. Mean A_fast error (eigenvalue moduli) 0.1337 before, 0.1356 after. Wall time is indicative only (other jobs were running on the machine); the dual refit costs about 2x for the 7 of 24 fits (29%) that needed it.

## Criteria

* **S1 identity: PASS.** Restart level: B-F2 144 restarts CONVERGED before and not collapsed, 0 with a different log-likelihood; B-F3 164, 0. Dual: 17 BEFORE fits with durations inside the box, 0 with parameters not bit-identical. Reported separately as pre-declared: the served model of a dataset changed in 12 of 24 (B-F2) and 7 of 24 (B-F3) datasets, 6 and 6 of them were "healthy before". Cause: restarts that were REJECTED_UPDATE before (86 in B-F2, 66 in B-F3) now converge and some win on log-likelihood. The owner decision "previously healthy fits stay bit-identical" therefore holds per fit (restart), not per served dataset; flagged for the owner. ARI on the datasets whose served model changed: +0.038 (se 0.079, n = 12) and -0.030 (se 0.047, n = 7).
* **S2 nothing degenerate is served: PASS** (0 in every arm; before 6, 1 and 7).
* **S3 accuracy: first part PASS, second part FAIL.** Healthy before: paired ARI after minus before +0.054 (se 0.036, n = 17) and +0.004 (se 0.005, n = 22); the dual arm has no healthy-before fit (nothing to test). Failing before and served after: B-F2 **-0.077** (se 0.116, n = 6) and B-F3 **-0.295** (n = 1) miss the pre-declared minimum of -0.01; D-pop -0.0015 (n = 2) passes. Reading (checked afterwards, not a criterion): in five of the six B-F2 cases and the B-F3 case the planted path visits only 2 of the 3 regimes (the generator's path can do that), so the BEFORE fit gave the third regime 1 to 4 days, which the collapse rule calls degenerate but which describes the data (seed 7208 before: p_t counts 44 / 1 / 30 against planted 45 / 30 / 0, ARI 0.97, log-likelihood higher than after in five of the six), whereas the AFTER fit is not allowed a regime under 4 effective days and splits a real regime (ARI 0.70 to 0.79). The one true failure (seed 7218: ARI 0.50 before, 1.00 after; seed 7203 had no fit before and ARI 1.0 after) improved. This is the cost of the collapse rule when K exceeds the number of regimes present; K selection by BIC over candidate_ks is the intended remedy and this study fixed K = 3 by design. Nothing was tuned.
* **S4 served share: PASS** (24/24 and 24/24; dual 2/24 = 8%, no minimum, expectation was about 10%).
* **S5 certified dual fits do not move: NOT TESTED.** Only 2 fits were certified (fewer than 5). The continuation harness did not reproduce the certification (both continued fits ended MAX_ITER_REACHED at 200 iterations when refit directly), so its numbers (label change 0 and 0.4%, largest m_t change 0.29 and 2.24 sd) say nothing about certified fits. Why the direct refit differs from the `fit_hssm` path is UNKNOWN; not investigated.
* **S6 N18 held-out scores: PASS** (earlier, `wp6_evidence/n18_check_summary.txt`: the 7 seeds with the duration inside the box are bit-identical, the eighth is listed).
* **S7 cost, reported:** see the table; refit share 7/24 for the dual arm.

Not measured here: MODEL_SELECTION_CONTESTED (K is fixed in this study, so no model selection happens); accuracy of the dual arm at more iterations than 200.
