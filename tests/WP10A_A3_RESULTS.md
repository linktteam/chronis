# WP10A A3: three new metrics, the gate between 2.0 and 2.5, and the safety demonstration: result

Label for every number: Python 3.12, repo pins, generated users, production engine (K = 3, 60 EM iterations, 10 restarts, every restart stored), replayed through the production lineage at the frozen operating point. Pre-declaration: `tests/PREDECLARED_CRITERIA_WP10A.md` section 3 and Part 7 (the freeze, committed before any EVAL or SAFE-EXTRA fit existed). Real users: UNKNOWN.

## Answer
**At the frozen operating point (consensus flag ON with the sharpness floor 0.95, gate h = 2.5) the 1% safety bound is demonstrated in every stratum, and coverage and names-agreement stay below their targets.**
- **Safety (RUN): wrong-and-unflagged 0 of 972 alignments** (108 users, seed-A weekly chains). By the rule declared in advance (zero wrong and the one-sided exact 95% upper bound at most 1%, which needs n >= 299): **demonstrated in hard, medium and easy (0 of 324 each, bound 0.92%) and pooled (bound 0.31%)**. The two-sided Wilson upper bound is 1.17% per stratum; by the general verdict wording of section 2 (Wilson-based) the per-stratum lines therefore read "met on the point estimate, not demonstrated". The declared safety rule is the specific one for the 1% claim, and I report both.
- **Coverage >= 80%: NOT MET pooled (528/972 = 54.3%, 95% CI 51.2-57.4%).** By stratum: hard 30/324 = 9.3% (6.6-12.9%), medium 221/324 = 68.2% (63.0-73.0%), **easy 277/324 = 85.5% (81.2-88.9%): MET**. The easy stratum meets both levels (coverage and safety); the stratum that drives the shortfall is hard, then medium.
- **(a) names agree when both seeds serve: 233/241 = 96.7% (93.6-98.3%): NOT MET** (target >= 98%; the point estimate is below it).
- **(b) serve/withhold agreement: 464/540 = 85.9% (82.7-88.6%)**; no target here (reported). Below the 90% I declared for WP10B in Part 7.
- **(c) days to first served lineage: reported below.** **Old criterion** (strict identical lineage among served, target 95%): 233/317 = 73.5% (68.4-78.1%): NOT MET.

## The numbers (pooled evaluation population = EVAL users 10101..10148 + SAFE-EXTRA users 10151..10210, 36 per stratum; the 48-user EVAL-only readout is beside it in `a3_eval_analysis.txt`)
| | pooled (108 users) | hard (36) | medium (36) | easy (36) |
|---|---|---|---|---|
| coverage (served alignments / alignments) | 528/972 = 54.3% (51.2-57.4) | 30/324 = 9.3% (6.6-12.9) | 221/324 = 68.2% (63.0-73.0) | 277/324 = 85.5% (81.2-88.9) |
| wrong-and-unflagged | 0/972 | 0/324 | 0/324 | 0/324 |
| one-sided exact 95% upper bound (the declared test) | 0.31% | 0.92% | 0.92% | 0.92% |
| (a) names agree when both serve | 233/241 = 96.7% (93.6-98.3) | 8/8 = 100% (67.6-100) | 94/97 = 96.9% (91.3-98.9) | 131/136 = 96.3% (91.7-98.4) |
| (b) serve/withhold agreement (both serve / both withhold / exactly one serves) | 464/540 = 85.9% (82.7-88.6); 241 / 223 / 76 | 171/180 = 95.0% (90.8-97.3); 8 / 163 / 9 | 141/180 = 78.3% (71.8-83.7); 97 / 44 / 39 | 152/180 = 84.4% (78.4-89.0); 136 / 16 / 28 |
| (b) where at least one seed serves | 241/317 = 76.0% (71.0-80.4) | 8/17 = 47.1% (26.2-69.0) | 97/136 = 71.3% (63.2-78.3) | 136/164 = 82.9% (76.4-87.9) |
| (c) seed A weekly: served by day 14 / 28 / 60; never (of n) | 58 / 71 / 76; never 32 (of 108) | 2 / 5 / 8; never 28 (of 36) | 24 / 30 / 32; never 4 | 32 / 36 / 36; never 0 |
| (c) seed A: median days among served (censored users counted as 61: quartiles) | 7 (7 / 14 / 61) | 21 (61 / 61 / 61) | 7 (7 / 14 / 28) | 7 (7 / 7 / 7) |
| (c) seed B 14-day: served by day 14 / 28 / 60; never | 45 / 57 / 70; never 38 | 2 / 2 / 4; never 32 | 15 / 22 / 30; never 6 | 28 / 33 / 36; never 0 |
Reading of (b): in the hard stratum 95% agreement is mostly **both seeds withholding** (163 of 180), not both serving; where at least one seed serves, hard agrees on 8 of 17. The product-relevant gap is the 76 windows (14%) on which exactly one of the two seeds serves.

## The operating point and the curve (REPORTED; nothing was chosen from the evaluation)
Development (48 users) froze flag ON, h = 2.5 (the three flag-ON candidates tied; Part 7). On the evaluation population the six candidates give:
| candidate | pooled coverage | wrong-and-unflagged | hard / medium / easy wrong |
|---|---|---|---|
| flag ON, h = 2.0 | 535/972 = 55.0% | 0 | 0 / 0 / 0 |
| flag ON, h = 2.25 | 532/972 = 54.7% | 0 | 0 / 0 / 0 |
| **flag ON, h = 2.5 (frozen)** | 528/972 = 54.3% | 0 | 0 / 0 / 0 |
| flag OFF, h = 2.0 | 795/972 = 81.8% (79.2-84.1) | 33/972 = 3.4% | 32 / 1 / 0 |
| flag OFF, h = 2.25 | 716/972 = 73.7% | 7/972 | 6 / 1 / 0 |
| flag OFF, h = 2.5 | 679/972 = 69.9% (66.9-72.7) | 1/972 | 0 / 1 / 0 |
- **The gate between 2.0 and 2.5 does not matter with the flag ON** (55.0 / 54.7 / 54.3%): the sharpness floor and the consensus flag decide. It matters only with the flag OFF, where h = 2.0 reaches the 80% pooled coverage but makes 33 wrong-and-unflagged alignments (3.4%; 32 of them in the hard stratum), which breaks the 1% level; the same line on development had 14 of 432.
- **No tested candidate reaches 80% pooled coverage and 1% wrong-and-unflagged together.** Flag OFF buys coverage with wrong answers in hard; flag ON buys safety with withholding in hard and medium. The hard stratum is the driver.
- Flag OFF at h = 2.5 (secondary line, no verdict): 1 wrong in 972 (medium), but names agree 334/363 = 92.0% (88.8-94.4%), worse than flag ON (96.7%). The wrong-and-unflagged counts of the flag-OFF candidates moved a lot between development (11 of 432 at h = 2.5, 10 in hard) and evaluation (1 of 972): they are not a stable property I can state.

## Declared expectations (section 3.4) and failure conditions
Declared in advance: hard fails coverage (it does: 9.3%); the gate in 2.0..2.5 does not by itself reach 80% pooled (it does not); wrong-and-unflagged stays 0 for the flag-ON candidates (it does: 0 of 972); names-agree close to 98% and not demonstrable unless at least 188 cases show no disagreement (96.7%, 8 disagreements in 241). **Failure conditions that occurred:** names-agree below 98% on the point estimate; coverage below 80% in hard and medium. Not occurred: any wrong-and-unflagged alignment; "no stratum meets both levels" (easy does).

## Seeds, order and what was looked at first
DEV users 10001..10048 (both chains, 768 fits), EVAL 10101..10148 (768 fits), SAFE-EXTRA 10151..10210 (960 fits); fit seed = user (chain A) and user + 1000 (chain B). Read first: the development readout (to freeze the point). The evaluation was read once, after all 1,728 fits existed. Dated criterion changes: Part 7 only (the freeze, the WP10B target for (b), the A4 closure); none after an evaluation result.

## What this does not say
Generated users only. The 1% safety claim is about gate-eligible alignments of seed-A weekly chains with 10 restarts at this operating point and population; it is not a claim about real users (UNKNOWN), about other chains or about the flag-OFF candidates. The consensus flag stays on by owner decision W9-1; I changed no default.

## Files
`wp10a_evidence/a3_dev_fits.jsonl`, `a3_dev_analysis.txt/.json`, `a3_eval_fits.jsonl` (1,728 fits), `a3_eval_analysis.txt/.json`; `tests/DESIGN_FREEZE_WP10A_A3.txt`; scripts `integration_report/wp10a/scripts/a3_run.py`, `a3_analyze.py`; tests `tests/test_wp10a_a3_analyze.py` and `tests/test_wp10a_a3_wiring.py` (36 in all).

## Mutation check
`mutation_wp10a_a3.py`: 40 mutants of the replay, the operating-point rule, the metrics (a) to (c), the safety bound and the freeze reading: **40/40 killed on the final head** (`wp10a_evidence/mutation_wp10a_a3_final.txt`; the third run on the earlier head was also 40/40). No survivors.
