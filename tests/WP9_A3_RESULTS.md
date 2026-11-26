# WP9 A3: detecting a change of the number of regimes (K change), results

Design and levels: `tests/PREDECLARED_CRITERIA_WP9.md` Part 7 (committed before any A3 fit). Labels: **RUN** = executed here, output in `wp9_evidence/`; **READ**; **UNKNOWN**. Everything below is synthetic data (planted split and merge at day 90, 12 users per stage, K = 3); whether real changes look like these is **UNKNOWN**.

## What was built (READ, `backbone/hssm/k_change.py`, opt-in)
At a refit the caller may fit the last 45 days at K-1, K and K+1 (production `fit_hssm`, 10 restarts) and compare BIC; a suspected change keeps the old names and the state `CONTINUITY_UNVERIFIED` with the reason `k_change_suspected` (or `K_CHANGED` when the new fit has another K). Split and merge candidates come from the posterior overlap with the old regimes. Nothing in the fit path calls it by default.

## Development (RUN, users 9221..9232, `wp9_evidence/a3_dev_analysis.txt`)
The declared rule set the two BIC thresholds: `up` = 0 (no-change false alarm 0/12), `down` = 40. **No grid value met the 10% false-alarm cap for `down`** (4/12 even at 40; 11/12 at 0 to 10): the largest value was taken and the shortfall reported (frozen in `tests/DESIGN_FREEZE_WP9_A3.txt`).

## Evaluation (RUN once, users 9241..9252, 36 streams, `wp9_evidence/a3_eval_analysis.txt`)
| level (declared before any fit) | result | verdict |
|---|---|---|
| (a) split detection >= 50% (an `up` signal at some refit) | 0/12 = 0.0% (95% CI 0.0-24.2%) | **NOT MET** |
| (b) false alarm <= 10% per direction on no-change streams | `up` 0/12 = 0.0%; `down` 1/12 = 8.3% | met on the point estimate, NOT DEMONSTRATED (CI reaches 24% and 35%) |
| (c) split-candidate precision >= 80% among `up` signals of split streams | 0/0 | UNKNOWN (no `up` signal occurred) |
| (d) a K-change record is never served as "same as before" and carries the reason | 0 violations in 6 `k_change_suspected` + 216 `K_CHANGED` records | **MET** |

Reported without a level: merge detection (`down` signal in a merge stream) 1/12 = 8.3%; merge-candidate precision 1/1; old nearest-mean candidates 0/1 for the merge; wrong-direction signals: `down` in split streams 2/12, `up` in merge streams 0/12.

## Against WP8
WP8 criterion 6: K selection on the growing window saw the change in 3 of 24 streams (12.5%), candidate precision 3 of 38 (7.9%), 38 of 38 K-differing refits `K_CHANGED` without "same as before". On the same kind of streams (this run, K selection (2, 3, 4) on [0, 150)): split 0/12, merge 4/12 = 33.3%. **The recent-window signal did not improve on K selection for splits (0/12 against 0/12) and its detection of merges (1/12) is below K selection's (4/12).** The only part that holds is the safety part: no K-change record can claim "same as before" (d), and false alarms are low because the signal almost never fires.

## Reading, stated plainly
- Why: 45 rows and 7 features give a K+1 regime too little data to win on BIC; a planted split has the child regime in about half of the later parent segments, so a 45-day window often holds few child days. Merging cannot be distinguished from a regime that is simply not visited in 45 days (declared expectation in section 19, borne out).
- **A3 detection is NOT achieved.** What ships is the lineage rule (a suspected change or a different K never claims "same as before"; owner decision A3: old names kept, "cannot confirm this is the same as before") and an opt-in signal that is quiet. The signal is not claimed to detect a split. A different design (longer recent window, an explicit test of a new cluster, or a change-point model) is WP10 work; I have not tried one on these seeds (that would be a second look at the evaluation users).
- Level (a) was not moved; the levels, deltas and seeds are as declared.

Mutation check (RUN on the final head): `wp9_evidence/mutation_a3.txt` (see the report for the count).
