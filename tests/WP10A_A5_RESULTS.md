# WP10A A5: the monitoring primitives (B5) redesigned: result

Environment label for every number: Python 3.12, repo pins (numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, scikit-learn 1.9.1), generated users, production `fit_hssm`.
Pre-declaration: `tests/PREDECLARED_CRITERIA_WP10A.md` Part 4 and Part 4b (committed before any A5 fit). Seeds: development users 10401..10448 (48), evaluation users 10451..10546 (96). Freeze: `tests/DESIGN_FREEZE_WP10A_A5.txt`.

## Answer
**Not met.** None of the three candidate monitors reached the three levels together on the development users, so the declared fallback froze the one with the smallest shortfall (MB, the transition-excluded score). On the 96 fresh evaluation users MB lowered the false alarms (51% -> 38%) and raised the regime-shift detection (46% -> 60%) against the production monitor, **but both stay far from the levels (false alarm <= 10%, regime-shift detection >= 70%)**. The sensor-change result is kept (88% against the 70% level, no regression). The production primitives stay as in WP9; **no monitor is used as an alarm**; `confidence_display_gate` stays `early_estimate`.

## RUN: development (48 users, `wp10a_evidence/a5_dev_analysis.txt`)
| monitor | false alarm (<= 10%) | regime shift (>= 70%) | sensor change (>= 70%) |
|---|---|---|---|
| M0 production | 19/48 = 39.6% | 21/48 = 43.8% | 41/48 = 85.4% |
| MA block-bootstrap se | 18/48 = 37.5% | 21/48 = 43.8% | 41/48 = 85.4% |
| MB transition-excluded | 15/48 = 31.2% | 21/48 = 43.8% | 44/47 = 93.6% |
| MC regime-standardised | 17/48 = 35.4% | 21/48 = 43.8% | 42/48 = 87.5% |
None eligible; total shortfalls MA 0.537, MB 0.475, MC 0.517 -> MB frozen.

## RUN: evaluation, once (96 users, `wp10a_evidence/a5_eval_analysis.txt`)
| monitor | false alarm (<= 10%) | regime shift (>= 70%) | sensor change (>= 70%) |
|---|---|---|---|
| MB (frozen) | 36/94 = 38.3% (Wilson 29.1-48.4%) NOT MET | 56/94 = 59.6% (Wilson 49.5-68.9%), median latency 18 days, events before day 135: 0, NOT MET | 83/94 = 88.3% (Wilson 80.2-93.3%), median latency 14, events before day 135: 0, MET |
| M0 production, same users | 49/96 = 51.0% (41.2-60.8%) NOT MET | 44/96 = 45.8% (36.2-55.8%), latency 12, events before day 135: 12, NOT MET | 81/96 = 84.4% (75.8-90.3%), latency 5, MET |
MB's denominators are 94: two users have too few scoreable days once the transition days are removed (status `not_enough_scores`; they count in neither numerator nor denominator, as declared). The WP9 sensor-change result was 12/12 on 12 users (Wilson 76-100%); 83/94 = 88% (Wilson 80-93%) is consistent with it: no regression is claimed beyond "still above the 70% level".

## RUN, POST HOC (development users only, `wp10a_evidence/a5_diag_posthoc.txt`; labelled, not part of any declared study)
- All **19 of 19** alarms of the production monitor on the stable development streams have a decoded regime change inside the 14-day window or the 28-day reference: the clustering by transitions found in WP9 is confirmed.
- The one-step scores are heavy-tailed (median kurtosis 8.2 against 3 for a normal series) and the frozen model's held-out mean score (-10.7) is far below its in-sample training score (-6.6): the score series carries model misfit that no change of the window statistic removes. I did not test a further variant (each look at the development users would need new development users; a new design would be a new Part).
- Reading (INFERENCE): a monitor built on the 14-day mean of one-step scores of a frozen regime model cannot reach the false-alarm level on these streams whatever its standard error; a monitor on a different statistic (for example the regime posterior itself, or a re-fitted reference) would be a new design.

## Mutation check
`integration_report/wp10a/scripts/mutation_wp10a_a5.py` (52 mutants of the candidate monitors, the development rule and the additive forward pass of `monitoring.py`) against `tests/test_wp10a_a5_monitors.py` (44 tests): first run 49/52 (survivors N09, N18, N24, each then covered by a new test and verified killed: 3/3); the full rerun on the final head is in the report. `baseline_forward_scores` is additive; its increments are bit-identical to `baseline_predictive_increments` (tested; the 12 + 12 batteries are in the report).

## What this does not show
Real users (UNKNOWN). That a different monitor could meet the levels (UNKNOWN). The levels are WP9's and were not moved.
