# WP8 Item 3: ordering score ladder (results)

Labels: RUN = executed in this container and the output is in `wp8_evidence/`; READ = read in code or docs; REPORTED = taken from an earlier package; UNKNOWN = not established.
Seeds used in this item: development **8201..8212** only (baseline fits of 12 users on 5 windows each = 60 fits; dual fits of 8201..8204). Evaluation seeds 8301..8324 were not touched.

## What was built (READ)
- `backbone/hssm/ordering.py` (new, additive): `LadderConfig`, `order_regimes(raw, labels, K, registry, config)` -> `OrderingResult(order, activity_rung, verified, reason, rungs)`; every rung's order, weights, scores, smallest gap, modal frequency and modal order are returned and are JSON-ready (`as_dict`).
- Owner decision A2 as implemented: score of a regime = sum over the rung's features of `w_f * direction_f * z`, `z` = the regime's raw-unit mean in units of the window's own mean and sd. `direction_f` comes from the feature registry (a domain prior, never learned). `w_f` is proportional to `coverage_f * SNR_f / (1 + SNR_f)`, SNR = between-regime over pooled within-regime variance over the **decoded** regimes. No outcome label or planted truth enters (test: the signature has no such argument, the result is deterministic and independent of regime names and day order).
- Rungs: 1 all features with a direction, 2 `movement` + `time_moving`, 3 `movement`. A rung is reliable iff its modal block-bootstrap ordering (B = 200, 7-day blocks, labels fixed, weights and scores recomputed in every resample, no refit) equals the point ordering with frequency >= 0.90 **and** the smallest adjacent gap >= `min_gap`. The highest reliable rung names the regimes if no other reliable rung disagrees; otherwise (none reliable, or a conflict) `verified = False`, `activity_rung = 0` and the ids follow first-fit order (`0..K-1`), with no activity claim.
- Not done here (Item 4): applying the ordering only at the first accepted fit of a lineage and inheriting names on later fits by alignment.
- Tagged `EXPERT_REVIEW(identity-distance)` at the rung roles, the config thresholds, the weight function and the rung choice (Item 7 builds the register).

## Tests and mutants (RUN)
- `tests/test_wp8_item3_ordering.py`: **18 tests** (independent re-computation of weights and scores for every rung on 4 seeds; weights sum to 1 and zero outside the rung; reliability and coverage raise the weight; no dependence on regime names or day order; clear data verified at rung 1 with the planted order; flipping every registry direction reverses the order; no signal gives unverified first-fit; conflicting reliable rungs are not served; the highest reliable rung names the regimes when rung 1 is not reliable; the gap condition and the frequency condition each make an ordering unverified on their own; a point order that is not the modal order is not reliable; a rung without usable data is not available; no label argument and determinism; invalid input and config raise; the default config equals the frozen design).
- Mutants: **29 of 29 killed** (`wp8_evidence/item3_mutation_results.txt`, script `integration_report/wp8/scripts/mutation_item3.py`), covering weights (coverage, SNR, saturation, normalisation), directions, rung roles, order direction, each reliability condition, agreement check, rung preference, first-fit fallback, bootstrap (blocks, label resampling, seeding, stride), unweighted grand mean and between variance, validation, the frozen defaults.

## Development study (RUN, `wp8_evidence/item3_analysis.txt`, `item3_ladder_dev.jsonl`, `item3_dev_fits.jsonl`; script `item3_analyze.py`)
60 baseline fits (12 users x windows (0,90), (0,120), (0,150), (30,90), (60,120); `fit_hssm`, K = 3, 10 restarts, `max_iter` 60); all 60 accepted. The 5 windows of one user are not independent, so every count below is a count of fits, not of users.
- Fits whose regimes map one-to-one to planted regimes by day overlap (**recoverable**): 52 of 60 (86.7%).
- Each rung alone, point ordering equal to the planted order (a fit that is not recoverable counts as wrong): rung 1 51/60, rung 2 52/60, rung 3 50/60.
- **g_min by the declared rule** (4.2): pooled (fit, rung) results with modal frequency >= 0.90: 137, correct 128. The largest smallest-gap among the 9 wrong ones is 0.624, so the rule gives **g_min = 0.7** (<= 1.0, so no fallback). Reported next to it, not used: **all 9 wrong results come from the 4 fits that are not recoverable**; among recoverable fits all 128 pooled results are correct. So 0.7 is driven by fits that no ordering rule could name correctly, and the conditional rule would have given 0.0 and hence the declared fallback 0.5. I keep the rule as declared (0.7): it is the more cautious value and moving the threshold after seeing the numbers is exactly what the pre-declaration is meant to prevent.
- Ladder at modal frequency 0.90, g_min 0.7: **verified 45/60 (75.0%), correct among verified 45/45**; per rung 1: 35, 2: 7, 3: 3; conflicts 0. Without the agreement check the result is identical on these data (the check is untested by real data; the tests cover it). Rung 3 alone would verify only 34/60 (56.7%); rungs 1 and 3 only 44/60.
- Sensitivity: modal frequency 0.80 / 0.90 / 0.95 -> verified 80% / 75% / 65%, all 100% correct among verified. g_min 0 / 0.25 / 0.5 / 0.75 / 1.0 -> verified 86.7 / 85.0 / 80.0 / 70.0 / 33.3%, correct among verified 92.3 / 94.1 / 95.8 / 100 / 100%.
- By planted separation of the user: sep < 0.8: verified 9/20; 0.8..1.3: 28/30; >= 1.3: 8/10. Low separation is where the ladder abstains, which is the intended behaviour.
- **Caveat (not hidden):** the threshold was chosen on these 60 fits and 100% on them is therefore not an estimate of the ladder's accuracy; that number comes only from Item 8 (criterion L, evaluation seeds, first fits of window 90). The 8 non-recoverable fits are not caught by the ladder at all (4 of them passed the frequency condition): a fit that has merged or split planted regimes has no planted order, and the ladder says what the fitted regimes mean in activity terms, not whether the fit is right. The fit-identifiability gate (Item 4) is the part that addresses this.

## Dual engine settings (RUN, development seeds 8201..8204, window (0,90), `max_iter` 200)
8 dual fits (4 users x {1, 2} restarts, `fit_hssm(model_type="dual")`, `max_iter` 200, window 0..90; `item3_dev_fits.jsonl`, group D):
| user | restarts | fit state | publishable | seconds | regimes recoverable |
|---|---|---|---|---|---|
| 8201 | 1 | MAX_ITER_REACHED | no | 48 | no |
| 8201 | 2 | MAX_ITER_REACHED | no | 193 | no |
| 8202 | 1 | CONVERGED | **yes** | 60 | no |
| 8202 | 2 | CONVERGED | **yes** | 160 | no |
| 8203 | 1 | REJECTED_UPDATE | no | 42 | yes |
| 8203 | 2 | REJECTED_UPDATE | no | 92 | yes |
| 8204 | 1 | CONVERGED | no (`degenerate_duration`, a regime at a wall of the duration box) | 83 | yes |
| 8204 | 2 | CONVERGED | no (same reason) | 186 | yes |
- 2 of 8 dual fits are publishable, both of the same user, and that user's fit does not recover the planted regimes. A second restart gave the identical log-likelihood for 3 of the 4 users (8201 improved from -596.4 to -579.6 and is still withheld), changed no publish decision, and cost 2.3 to 4 times the time. So the dual benchmark is expected to have few accepted alignments (the declared 20-alignment rule decides whether a dual criterion is evaluated or reported UNKNOWN).
- Consequence (frozen in the design-freeze section): dual benchmark fits use `max_iter = 200`, **one** restart. This deviates from the declared "two restarts" in one respect (the declaration allowed raising `max_iter`, not dropping a restart); the reason is compute and the evidence is the table above.
- Secondary measurement (declared): the dual lineage is also measured on all dual fits that did not fail numerically (including `MAX_ITER_REACHED` and `degenerate_duration`), clearly labelled as not the criterion.

## Design freeze: see the dated section of `tests/PREDECLARED_CRITERIA_WP8.md` (written before any evaluation seed is touched).
