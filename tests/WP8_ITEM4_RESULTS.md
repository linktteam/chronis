# WP8 Item 4: regime identity across refits (results)

Labels: RUN = executed in this container and the output is in `wp8_evidence/`; READ = read in code or docs; REPORTED = taken from an earlier package; UNKNOWN = not established.
Seeds used in this item: development **8201..8212** only (960 fits) and test-only seeds 8791..8794. Evaluation seeds (8301..8324, 8401..8424, 8501..8512, 8601..8612) were **not touched** (RUN: no record with such a user seed in `wp8_evidence/*.jsonl`).

## What was built (READ)
- `backbone/hssm/lineage.py` (new, additive): `FitSummary` / `make_summary` / `summarize_fit` (either engine, regime means in raw physical units with the fit's own scaler, A6), `separation_min` (the identifiability gate statistic), `alignment` (Hungarian on the squared distance after dividing by the anchor's per-feature scale; best and second-best total cost; `margin_diff = (C2 - C1) / K`, `margin_ratio = C1 / C2`), `margin_ok`, `step` (state machine ALIGNED / CONTINUITY_UNVERIFIED / K_CHANGED / NEW_LINEAGE; the anchor moves only on ALIGNED and NEW_LINEAGE), `split_merge_candidates`, `apply_user_choice` (A3: `None` = `KEEP_OLD`, `NEW_NAMES` starts a new epoch and keeps the old-to-new mapping), `may_claim_same_as_before` (True only for ALIGNED), `annotate_result` (`HSSMResult.lineage`, additive field, A4: the raw outputs are not touched).
- Record columns of the brief: `stable_regime_id`, `lineage_id`, `lineage_epoch`, `continuity_state`, `alignment_margin`, `activity_rung`, `scaler_mean`, `scaler_std`, `expert_review_status` (always "PENDING"), plus `margin_diff`, `margin_ratio`, per-regime distances, `proposed_new_names`, `user_choice`, `message`, `split_merge_candidates`, `reasons`, `threshold`.
- `label_switching.canonicalize_labels`: docstring and constant-name claim corrected (the raw order is a label convention, **not** an activity ordering; the constant value still works; no behaviour change).
- **Design freeze (this commit):** `AlignmentConfig` defaults set from the development study; new `PairThreshold.calibrated` flag and `AlignmentConfig.dual_engine`; the classes without a calibrated threshold are never ALIGNED.

## Development study (RUN; seeds 8201..8212; scripts `item4_run.py`, `item4_analyze.py`, `item4_margin_report.py`; evidence `item4_dev_fits.jsonl`, `item4_analysis.txt`, `item4_alignments.jsonl`, `item4_thresholds.json`, `item4_margin_analysis.txt`)
960 fits: baseline 720 (12 users x drift 0, 1, 2.5 x protocols G and R, 20 fits each), dual 240 (12 users x one drift x G and R; frozen dual settings: 1 restart, `max_iter` 200; only 31 of the 240 were accepted by the publish rule, so the dual alignments use every dual fit that did not fail numerically, as pre-declared). Alignments: baseline 648, dual 216, cross-engine (the same window by the two engines, both directions) 480.
| class | alignments | both fits recover the planted regimes | permutation errors among those | strict-correct overall |
|---|---|---|---|---|
| baseline | 648 | 549 (84.7%) | 4 | 545 (84.1%) |
| dual | 216 | 54 (25.0%) | 9 | 45 (20.8%) |
| cross-engine | 480 | 168 (35.0%) | 28 | 140 (29.2%) |

**The gate (declared rule): h = 4.0.** Strict-wrong among baseline alignments whose two fits both have separation >= h: 103 (h = 0), 95 (1.5), 62 (2.0), 19 (2.5), 3 (3.0), 1 (3.5), **0 (4.0)**. Pass share at 4.0: baseline 242/648 (37.3%), dual 4/216 (1.9%), cross-engine 16/480 (3.3%).

**Thresholds (declared rule, clarification applied).** Baseline: raw, z-space excluded as a negative control, raw + duration 0.5 and 1.0 tie on coverage, so raw (declared tie order); acceptance (a) and (c) are vacuous (smallest tau = 0) and excluded; (b) with rho = 0.5. Coverage on development 242/648 = **37.3%**, strict-wrong served 0 (drift 0: 78/216 = 36.1%; drift 1: 82/216 = 38.0%; drift 2.5: 82/216 = 38.0%). **Dual and cross-engine: UNKNOWN** (4 and 16 passing alignments, fewer than the declared 20), so those classes are not calibrated and are never served as ALIGNED.

**Shortfall against the declared 80% target: coverage 37.3% (baseline), 0% (dual, cross-engine).** Reported, criteria not loosened. Sensitivity of the gate (reported, not used to choose): h = 0: coverage 90.3%, 43 strict-wrong served; 1.5: 90.1%, 42; 2.0: 86.6%, 27; 2.5: 78.1%, 11; 3.0: 63.3%, 3; 3.5: 51.2%, 1; 4.0: 37.3%, 0.

**Per-alignment margin analysis (baseline, raw distance; `item4_margin_analysis.txt`).**
- Among the 242 alignments that pass the gate, all are strict-correct, in every `margin_ratio` bucket up to 0.33 (maximum 0.332); median 0.014 to 0.020 per drift level.
- Among all 648 alignments, the strict-wrong ones (103; almost all involve a fit that did not recover the planted regimes) occur from `margin_ratio` 0.013 upward and `margin_diff` up to 3.2: **a good margin does not show that a fit is identifiable**; a fit that merged two planted regimes can match its neighbour cleanly. This is why the gate, not the margin, carries the safety.
- Permutation errors among alignments of two recoverable fits: 4, all at `margin_ratio` >= 0.48 (0.48, 0.57, 0.61, 0.91) and `margin_diff` <= 0.89. So **the matching itself starts to be wrong at about `margin_ratio` 0.48, close to the production rule 0.5**; the frozen ratio threshold 0.5 sits at that boundary on baseline data (a tighter threshold would cost almost no coverage among the passing alignments, whose largest ratio is 0.33; this is left to the expert, A7).
- Dual: 9 permutation errors at `margin_ratio` 0.34 to 1.0, in every one of them the smaller separation of the two fits is 0.95 to 2.43; cross-engine: 28 permutation errors, all at ratio >= 0.58 and in pairs where one of the two fits has separation < 4.

## Tests and mutants (RUN)
- Tests of Item 4: `test_wp8_item4_lineage.py` 56, `test_wp8_item4_engines.py` 7, `test_wp8_item4_label_convention.py` 15, `test_wp8_item4_freeze.py` 15 (new at the freeze: frozen numbers pinned and compared with `item4_thresholds.json`; uncalibrated thresholds accept nothing; clear dual and cross-engine pairs are not ALIGNED by default but are with a calibrated threshold; the gate at exactly 4.0; each class of pair uses its own threshold; the threshold dict stays additive). Two earlier tests were adapted because dual-to-dual pairs now have their own class.
- Mutants: **64 mutants, 64 killed** (the first full run reported 63/64: L47's search text did not match the source comment, so it was NOT APPLIED, not a survivor; the pattern was corrected and L47 re-run alone, KILLED, `wp8_evidence/item4_mutation_L47_rerun.txt`) (`wp8_evidence/item4_mutation_results.txt`, script `integration_report/wp8/scripts/mutation_item4.py`; the runner now keeps `item4_thresholds.json` in the throw-away copy because a test reads it). L33 and L14 were re-pointed at the changed code; L45..L56 are new: gate default, baseline rule, calibrated defaults of the two classes, the calibration check in `margin_ok` (removed, and placed after the single-regime shortcut), the reason string, class selection for dual pairs, the `calibrated` key of the record, the duration default, the gate boundary (`<` vs `<=`), and the validation of rho.
- Full suite: **738 passed, 5 xfailed** of 743 collected (RUN, 1430 s, `wp8_evidence/full_suite_wp8_item4.txt`); 555 collected at the base `27c591c`, so the 555 older tests are among them and pass.

## Limits (not hidden)
- The thresholds were chosen on 960 development fits of 12 synthetic users; criterion rates on the evaluation seeds are the estimates, these are not.
- The gate was chosen by the declared rule on **baseline** alignments and applied unchanged to the dual engine and to engine switches; the dual development data say the gate is far too strict for the dual fits as they are (1.9% pass), which is a statement about dual fit quality (WP10), not evidence about the gate.
- The synthetic users are K = 3 with planted separation; real-user separation is UNKNOWN (WP9).
- The matching distance, the gate and every threshold are awaiting expert review (A7): `expert_review_status = "PENDING"` on every record.
