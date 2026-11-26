# WP9 B5 (monitoring primitives) and B6 (periodic re-certification): results

Design and levels: `tests/PREDECLARED_CRITERIA_WP9.md` Part 9 (sections 23-25), Part 9b (the monitor statistics were replaced after a check on pure noise, before any B5 seed was generated) and Part 9c (a supplementary B6 arm, added after the declared arm was vacuous). Labels: **RUN** = executed here, output in `wp9_evidence/`; **READ**; **UNKNOWN**. All users are generated; whether real drops look like the planted ones is **UNKNOWN**.

## B5: what was built (READ, `backbone/hssm/monitoring.py`, pure functions, nothing calls them from the fit path)
- `predictability_monitor(scores)`: events when the 14-day mean of a per-day held-out score falls below the 28-day trailing mean by at least 0.75 reference sd and 3.5 standard errors (reference sd inflated for the lag-1 autocorrelation); one event per 14 days. `alert_on_drop` sends one developer alert per event (hashed user id, numbers only, via `alerts.emit`).
- `confidence_display_gate(calibrated_confidence, calibration_evidence)`: `early_estimate` unless N17 passed (`config.N17_PASSED`, **False**), calibration evidence exists for the class of fit and the confidence is a number in [0, 1] of at least 0.85 (never lowered; the integer percentage is rounded down). **While N17 is not passed the gate returns `early_estimate` for every input** (tested over a grid that includes 1.0, NaN, strings, booleans).
- `baseline_predictive_increments(model, X)`: held-out one-step scores of a fitted baseline model (tested: the increments sum to the model's log-likelihood and a prefix gives the same increments).
- Tests: 45 (`tests/test_wp9_b5_monitoring.py`); mutants 26/26 killed on the final head (`wp9_evidence/mutation_b5.txt`; three first-run survivors were fixed with new tests).

## B5: simulation (RUN, seeds 9901..9912, 12 users, `wp9_evidence/b5_sim_analysis.txt`)
K = 3 baseline model fitted on days 0-89 and frozen; the held-out score of days 90-179 on three streams that are identical before day 135 (none; regime shift = every regime mean moved by 2.0 within-regime sd; sensor change = noise of three features x4). Primary series S1 (raw score against the user's trailing window).

| level (declared before any run) | result | verdict |
|---|---|---|
| detection of the regime shift >= 70% | 5/12 = 41.7% (Wilson 19-68%), median latency 4 days | **NOT MET** (declared expectation, Part 9b: a 2-sd shift is about 1 score-sd) |
| detection of the sensor change >= 70% | 12/12 = 100% (Wilson 76-100%), median latency 3 days | MET |
| false alarm <= 10% of the none streams | 5/12 = 41.7% (Wilson 19-68%) | **NOT MET** |

S2 (gain over a single Gaussian, no level): false alarm 10/12, regime shift 6/12, sensor change 8/12; the single-Gaussian reference is **worse** than the user's own trailing window.
**Reading, stated plainly:** as declared, the monitor detects a large sensor change quickly but raises a false alarm on 5 of 12 stable users within 49 scoreable days, so it cannot yet be used as an alarm. The checks on pure noise (Part 9b) had assumed score noise without regime structure; the real scores of a regime model are clustered by regime transitions (a 14-day window holds about one transition, a 28-day reference two or three), which makes window means vary more than the autocorrelation correction allows. I did not tune on these seeds. A longer window or a reference that knows the regime (score given the current regime) is the obvious next design; it is not tried here because that would be a second look at the evaluation users (**UNKNOWN** whether it works). What the owner decision C1 asks for (early estimate plus a developer alert on a drop) is delivered as primitives; **the false-alarm rate is a reason to treat the alert as a developer hint, not as a user-facing statement**.

## B6: what was built (READ, `backbone/hssm/recertify.py`, nothing calls it from the fit path)
`recertify(X, new_rows, metadata=...)` refits with the production `fit_hssm` on the stored plus the new days, with the earlier settings, and returns `PUBLISHABLE` only when the new result's own `insights_withheld` is False. It has no parameter that loosens a rule; it refuses settings that would (a fast-test fit, fewer than 10 restarts, a lowered cold-start gate, unknown settings) and never touches an earlier published state. `recertify_due` and `recertification_sweep` are the schedule hook (the refit policy, weekly by default). Tests: 22; mutants 23/25 killed (`wp9_evidence/mutation_b6.txt`, first run; the survivor R16 (row order) got a test, R22 is an equivalent mutant: a boolean count of restarts is refused anyway by `n < 10`).

## B6: simulation (RUN, seeds 9801..9812, `wp9_evidence/b6_sim_analysis.txt`, `wp9_evidence/b6s_sim_analysis.txt`)
- **Declared arm** (baseline, candidates (2, 3, 4), 10 restarts): **0 of 12** fits withheld at day 30 (all published; fresh window fits at 60 and 90 days 12/12 published). The re-certification rates are therefore **too few to read** (declared wording). The no-relaxation check covered the 36 published outputs: **0 violations (MET)**.
- **Supplementary arm S** (added after seeing that; not part of the declared design): the same users with K forced to 5, which gives collapsed regimes in short windows. **6 of 12** withheld at day 30 (`collapsed_regime`, 1-2 effective days); after re-certification with the days to day 60, **6 of 6 publishable**; a fresh fit on the same matrix agrees 6 of 6 (same procedure); fresh window fits with their own z-scoring: 11/12 published at day 60, 12/12 at day 90. No-relaxation check on 35 published outputs: **0 violations (MET)**. What this shows: more days cure a collapsed regime on these generated users and `recertify` publishes exactly when the production rule does. It does **not** show how many real withheld fits more days would cure (the dual engine's withheld fits, mostly non-convergence, are a different population; B1 below).

## What could not be established
- Whether the B5 thresholds suit real score noise (**UNKNOWN**); whether real withheld fits are cured by more days (**UNKNOWN**).
- B4 (shrinkage uncertainty on fresh and unusual people) was not run in this package unless the report says otherwise.
