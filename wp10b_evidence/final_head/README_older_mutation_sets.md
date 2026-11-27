# Older mutation sets rerun at the WP10B final head (Python 3.12.3, repo pins)

Sets that patch files WP10B changed (`lineage.py`, `k_change.py`, `alerts.py`, `dual_model.py`) or whose targets moved since their last run were rerun with the unchanged runners and rules of their own packages (WP8 runner: any non-zero exit counts as killed; WP9 runner: as in WP9). Run 1 (`*_run1_patterns_not_applied.txt`) found mutants that did NOT APPLY because F3 (lineage, alerts) or WP10A (monitoring, the gate default 2.5) had rewritten the patched lines; a not-applied mutant proves nothing, so the patterns were rewritten to the current lines (same planted defects) and the sets were rerun. One mutant also SURVIVED in run 1:

| mutant | run 1 | cause | action | rerun |
|---|---|---|---|---|
| item 4: L14, L45, L51 | not applied | F3 changed the margin check to `margin_passes = margin_ok(th, al)` inside a guard; the gate default is 2.5 since WP10A (L45 now moves 2.5 to 3.0); the `elif` of the calibrated check is now `elif al is not None:` | patterns rewritten | 64/64 killed |
| item 5: C25 | not applied | same `elif` as L51 | pattern rewritten | 25/25 killed |
| item 5: C13 (mismatched categories accepted) | SURVIVED | under F3 a width mismatch of 3 against 2 columns still fails closed without the explicit check, because numpy raises by itself; the planted defect only matters where numpy broadcasts (a one-column profile) | NEW TEST `test_a_single_category_context_is_not_broadcast_against_a_full_profile_in_either_direction` (both directions); no production change | killed |
| WP9 A1: A24, A26, A27, A28 | not applied | `make_summary(...)` call and the anchor-replacement condition gained arguments (`assessment_crashes=...`, `and not crashes`) | patterns rewritten | 28/28 killed |
| WP9 B3: B17, B18 | not applied | `alerts.emit` now binds `except Exception as exc` and has a second `logger.warning("CHRONIS_DEV_ALERT ...")` line (F3) | patterns rewritten (B18 anchored on the `%s` format of the event line) | 23/23 killed |
| WP9 B5: M24, M25 | not applied | WP10A A5 added `baseline_forward_scores`, which repeats the two patched lines (pattern count 2) | patterns anchored on the whole block of `baseline_predictive_increments` | 26/26 killed |
Not rerun here: WP9 A3 (rerun after F3: `wp10b_evidence/mutation_a3_after_f3.txt`, 34/34 and 4/4 live), WP10A sets (their target code is unchanged by WP10B).
Also at the final head: the 14 WP10B sets (`mutation_wp10b_*.txt`, every mutant caught, control first), WP8 item 0e 15/15 and item 7 25/25. The 12 + 12 baseline bit-identity batteries against `git archive 95dd125` print PASS (`baseline_battery_fast.txt`, `baseline_battery_full.txt`).
