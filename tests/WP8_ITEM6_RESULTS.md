# WP8 Item 6: refit cadence (results)

Labels: RUN = executed in this container and the output is in `wp8_evidence/`; READ = read in code or docs; REPORTED = taken from an earlier package; UNKNOWN = not established.
Seeds used in this item: development **8201..8204** only (baseline engine, drift 0, windows [0, t) for t = 90..150). Evaluation seeds (8301..8324 and later ranges) were **not touched**.

## What was built (READ)
- `backbone/hssm/refit_policy.py` (new, additive): `RefitPolicy(refit_interval_days=7)` (owner decision A5: weekly by default, changeable), bounds **1 <= interval <= 28** days, pure `should_refit(last_refit_day, today, policy)`, `next_refit_day`, `refit_schedule(first_day, end_day, policy)`, `RefitPolicy.from_config(...)` (refuses unknown keys). Changing the cadence is configuration only; nothing else in the package reads it. The reasons for the bounds are in the module docstring and in `tests/PREDECLARED_CRITERIA_WP8.md` 4.6: below one day there is no new data per refit; above four weeks the gap to the last anchor grows and the stable ids can no longer be tracked through the drift of the study (**the upper bound is a declared limit; it is not tested by the development study below, which has no drift**).
- Tagged `EXPERT_REVIEW(identity-distance)` in Item 7 (the cadence sets how far apart two matched fits are).

## Tests and mutants (RUN)
- `tests/test_wp8_item6_refit_policy.py`: **73 tests** (the default and the bounds; refusal of out-of-range, non-integer, boolean and non-numeric intervals; `should_refit` exactly follows every allowed interval 1..28; a user without an earlier refit is due; purity and immutability; invalid days fail closed; the schedule is an arithmetic progression for every interval; changing the interval needs a config change only; the contract documents the default and the bounds).
- Mutants: **21 of 21 killed** (`wp8_evidence/item6_mutation_results.txt`, `integration_report/wp8/scripts/mutation_item6.py`).
- The study code is tested by `tests/test_wp8_item8_scoring.py` (21 tests, scoring and statistics) with **43 of 43 mutants killed** (`wp8_evidence/item8_scoring_mutation_results.txt`, `mutation_item8_scoring.py`). A Wilson interval that printed `-0.0%` was clamped to [0, 1] with a test and two mutants.

## Cadence study (RUN, development seeds 8201..8204; `integration_report/wp8/scripts/item6_study.py`; fits `wp8_evidence/item6_dev_fits.jsonl` and `item4_dev_fits.jsonl`; output `wp8_evidence/item6_analysis.txt`)
The same four 150-day streams (baseline engine, K = 3, fit seed = user seed, drift 0, protocol G windows [0, t)) were refitted **daily** (t = 90..150, 61 fits per user), **weekly** (t = 90, 97, ..., 146, 150, 10 fits) and **every 14 days** (t = 90, 104, ..., 146, 150, 6 fits), 204 new fits plus 40 shared with Item 4. Every schedule was replayed through the production `lineage.step` with the **frozen defaults** (Item 4 design freeze) and every alignment scored against the planted truth (permutation-correct, both fits recoverable).
| schedule | fits fed | alignments | ALIGNED | ALIGNED and correct | wrong and unflagged |
|---|---|---|---|---|---|
| daily | 244 | 240 | 120/240 = 50.0% (Wilson 95% 43.7-56.3%) | 120/120 | 0/240 (0.0-1.6%) |
| weekly | 40 | 36 | 18/36 = 50.0% (34.5-65.5%) | 18/18 | 0/36 (0.0-9.6%) |
| every 14 days | 24 | 20 | 10/20 = 50.0% (29.9-70.1%) | 10/10 | 0/20 (0.0-16.1%) |
- **The interval changed nothing on these four users.** Two users (8202, 8204; separation of their fits 4.65 and 8.5 at day 120) were ALIGNED at every step of every schedule; two (8201, 8203; separation 3.44 and 3.02) were CONTINUITY_UNVERIFIED at every step of every schedule, because their fits stay below the frozen gate h = 4.0 (they would pass a gate of 3.0, and one of them only by 0.02). Coverage is therefore a property of the user's fit quality, not of the cadence, here; 50% is two of four users, not a rate.
- **Stable ids at day 150 agree across the schedules: 1.000 for every pair of schedules and every user (4/4 users at or above the declared 0.95).** For the two ALIGNED users the chain of 60, 9 or 5 alignments gave the same names at the end; for the other two the old (first-fit) names were kept throughout.
- Not shown by these data (UNKNOWN until Item 8 criteria 5 and 8): whether the cadence matters when regimes **drift** (this study is drift 0), whether more frequent refits keep more users ALIGNED when fits differ more, and anything about the dual engine (not refitted daily; about 60 s per fit). With four users, 95% intervals on the rates are wide (see the table); the 100% correct among ALIGNED is a development observation, not an estimate.

## Limits (not hidden)
- Development only, four users, drift 0; the cadence criteria are evaluated on 8301..8324 (weekly, 14-day) and 8301..8312 (daily) in Item 8.
- The identity gate (frozen at 4.0) decides who is ALIGNED; the cadence cannot rescue a user whose fits never pass it.
