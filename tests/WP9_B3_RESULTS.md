# WP9 B3: the "unsure number of phases" tag (`k_adequacy` / `k_unsure`), results

Design and levels: `tests/PREDECLARED_CRITERIA_WP9.md` Part 4 (sections 12 and 13, committed before any B3 fit); frozen thresholds: `tests/DESIGN_FREEZE_WP9_B3.txt`. Labels: **RUN** = executed here, output in `wp9_evidence/`; **READ**; **UNKNOWN**. All data are generated (the WP7-R generator, planted K = 2, 3 or 4, F = 7, T = 75, easy and hard separation); whether real people's regime counts behave like this is **UNKNOWN**. No level was moved after a result was seen.

## What was built (READ, `backbone/hssm/k_adequacy.py`, `alerts.py`, `fitting.py`)
After the BIC choice of K, `HSSMResult.k_adequacy` (baseline engine only) records signals of the **selected** fit and the property `k_unsure` is true when one of them fires: (1) `regimes_overlap` (smallest pairwise distance between regime means below `d0` pooled within-regime sd), (2) `k_not_clearly_best` (another candidate K within `delta` BIC points), (3) `selected_k_is_largest_candidate` (off), (4) `restart_disagreement` (A1 `FIT_UNSTABLE`). **The best guess is still shown, the selected K never changes** (`best_guess_shown = True`; tested, and covered by the bit-identity batteries). When `k_unsure` is true, one developer alert is emitted (`alerts.emit("k_unsure", ...)`, user id hashed, numbers only). Owner decision B3: show the best guess with an "unsure" tag and flag the developer: this is that.

## Development (RUN, seeds 9531..9554, 48 datasets, `wp9_evidence/b3_dev_analysis.txt`)
Declared rule (grid d0 x delta x signal 3; false alarm on right-K easy fits at most 25%, then highest detection): **d0 = 1.5, delta = 0, signal (3) off.** Development numbers: detection 15/19 = 78.9% of wrong-K fits, false alarm 0/20 on right-K easy fits, 9/29 = 31% on all right-K fits. With delta = 0 signal (2) can never fire (the selected K has the lowest BIC, so the gap to the runner-up is never negative): the tag is in practice "regimes overlap" or "restarts disagree".

## Evaluation (RUN once, seeds 9501..9524, 48 datasets, `wp9_evidence/b3_eval_analysis.txt`)
| level (declared before any fit) | result | verdict |
|---|---|---|
| (a) detection of wrong-K fits >= 60% (tagged / wrong-K) | 13/22 = **59.1%** (Wilson 38.7-76.7%) | **NOT MET** (by one fit: 14 of 22 were needed) |
| (b) false alarm on right-K fits of the easy design <= 25% | 2/18 = 11.1% (Wilson 3.1-32.8%) | met on the point estimate, NOT DEMONSTRATED (the interval reaches 33%) |
| (c) hard design (reported, no level) | false alarm 7/8 = 87.5% (Wilson 52.9-97.8%); detection 13/16 = 81.2% (Wilson 57.0-93.4%) | reported |

Detection fell from 78.9% on development to 59.1% on evaluation; the interval is wide (about 39% to 77%), so the evaluation does not say whether the true detection rate is above or below 60%; it says the declared level was not reached.

Other numbers reported (no level): BIC-selected K correct 26/48 (easy 18/24, hard 8/24; WP7-R reported 20/24 and 8/24); tagged fits 22/48; all right-K fits tagged 9/26 = 34.6%; reasons that fired: `regimes_overlap` 21, `restart_disagreement` 9 (a fit can have both). By design and planted K: easy 0/8 tagged at K = 2, 2/8 at K = 3, 0/8 at K = 4; hard 7/8, 5/8, 8/8.

## Reading, stated plainly
- **Where the tag works:** on the hard design it tags 13 of 16 wrong-K fits (and the BIC is right in 0 of 16 of the planted K = 3 and K = 4 fits there), which is what the owner wanted: "unsure" when the count is probably wrong.
- **Where it does not:** on the easy design **0 of the 6 wrong-K fits were tagged** (the declared blind spot of section 12: a confident, well-separated fit with a wrong K, for example the fit that merged two regimes into one, shows neither overlap nor restart disagreement). Nine of the 22 wrong-K fits had no signal at all: all six easy ones and three hard ones.
- **What it costs:** on the hard design it is nearly always on (7 of 8 right-K fits tagged), so for weakly separated people the tag says "unsure" whether or not K is right; I think that is a reasonable answer there (declared in section 13: a tag on overlapping regimes is partly the point), but it means the tag carries almost no information within that group.
- **Recomputation check (RUN):** the tag recomputed offline from the stored fits equals the production field `k_adequacy.k_unsure` in all 48 evaluation fits (no mismatch; the development analysis did not print this check).
- **B3 detection is NOT achieved at the declared level (59.1% against 60%).** I did not change a threshold. What ships is a tag that is right in the direction intended for weakly separated users, silent on easy users (which includes silence on their wrong K), and which never changes the selected K. A different kind of signal (for example a held-out likelihood test of K + 1 versus K, a bootstrap of the BIC gap, or a check against the posterior occupancy of small regimes) is WP10 work and has not been tried on these seeds.

## What this does not show
Whether a real person's regime count is 3 (UNKNOWN); how often real fits are "easy" in this sense (UNKNOWN); it shows how often these signals coincide with a wrong BIC choice on this generator.

## Mutation checks (RUN on the final head)
`mutation_b3.py` -> `wp9_evidence/mutation_b3.txt`: MUTATION_B3_PLACEHOLDER

## Artifacts
`wp9_evidence/b3_dev_fits.jsonl`, `b3_dev_analysis.txt/.json`, `b3_eval_fits.jsonl` (48 fits), `b3_eval_analysis.txt/.json`, `mutation_b3.txt`; scripts `integration_report/wp9/scripts/b3_run.py`, `b3_analyze.py`, `mutation_b3.py`; tests `tests/test_wp9_b3_k_adequacy.py`.
