# WP10B B5: dual-to-dual and cross-engine lineage thresholds (2026-11-27)

Every number: Python 3.12.3, repo pins, development fits only (B1 users 11201..11236 at S0, B2 users at the frozen configuration I600). Rule: `tests/PREDECLARED_CRITERIA_WP10B.md` Part 10 (section 16), committed (223eafc) before the B2 analysis and before any number below, with the expectation stated in advance. **No B5 fit was run and no B5 seed (11601..11799) was used.** Script and output: `integration_report/wp10b/scripts/b5_feasibility.py`, `wp10b_evidence/b5_feasibility.txt`.

## Verdict
**NOT DEMONSTRATED (infeasible in this package)** for dual-to-dual and for baseline-to-dual alignments, in every stratum. This is the declared outcome of the rule, not a failure. The two pair types stay `calibrated = False`: never ALIGNED, as before (RUN: `tests/test_wp10b_b5_feasibility.py::test_the_production_code_keeps_both_pair_types_uncalibrated_and_the_dual_without_a_flag`; the default-configuration behaviour is pinned by `tests/test_wp8_item4_freeze.py::test_a_clear_dual_pair_is_not_aligned_with_the_defaults_but_is_with_a_calibrated_threshold`). No production code was changed for B5.

## Why (RUN, the three conditions of the rule)
| | S0 (B1) | I600 (B2, frozen) |
|---|---|---|
| publishable share of a dual fit: hard / medium / easy | 0.17 / 0.17 / 0.00 | 0.67 / 0.67 / 0.42 |
| median dual fit time | 60 s | 191 s |
| alignments needed for a one-sided 95% bound of at most 1% with zero wrong | 299 per stratum | 299 per stratum |
| evaluation fits for 299 per stratum, every window pair counted (10 windows per chain, 9 alignments) | 1,020 (34 chains per stratum) | 1,020 |
| the same on two cores | 8.5 h | 27.0 h |
| only pairs of fits the production policy publishes | impossible (easy 0%); hard and medium 2,000 fits each | 1,800 fits (hard 500, medium 500, easy 800), 47.7 h |
| (a)+(b) favourable cost against the 6 h budget | FAILS | FAILS |
| alignments per stratum the 6 h budget buys (every pair counted) and the bound with zero wrong | 207; 1.44% | 63; 4.64% |
| (c) at least 20 development alignments pass the gate per pair type, and a dual stability signal | FAILS | FAILS |
Condition (c): WP8 development found 4 of 216 dual-to-dual and 16 of 480 cross-engine alignments passing the gate (READ, `backbone/hssm/lineage.py`), and the dual engine has no counterpart of the multi-start consensus flag that the A3 "wrong and unflagged" count rests on (RUN: `fit_stability_flag` of a dual result is `(False, None)`). A threshold alone cannot supply that flag.

## What it means (INFERENCE where marked)
- **The achievable n is below the 1% level in every case here:** about 63 alignments per stratum with the frozen configuration (bound 4.6% at zero wrong), about 207 at the old S0 fit time (1.4%). Better fitting (I600) makes more fits publishable but each fit about 3 times dearer, so it does not buy a 1% bound inside this package.
- **What would change the verdict (owner's decision, not in this package):** a design for a dual stability signal (what "unflagged" means for a dual fit); a cheaper fit (SQUAREM was not built because its trigger was not met); a much larger compute budget; or a lower level than 1% (the owner's W10A level is unchanged).
- **Not shown:** nothing about the correctness of dual or cross-engine alignments; the numbers say only that the A3-style demonstration cannot be made here. The cost model counts evaluation fits only (development fits would add), counts every window pair as an alignment (the favourable count), and uses the development publishable rates (36 users, one setting).

## Tests and mutation controls (RUN)
`tests/test_wp10b_b5_feasibility.py` (12 tests, no evidence file needed); mutation set `mutation_wp10b_b5.py` (21 mutants; run 1 was 15 of 18 caught: H03 and H15 crashed instead of failing by assertion, H17 survived because the stored facts already fail condition (c); the tests were strengthened and the next run caught 18 of 18 (`wp10b_evidence/mutation_wp10b_b5_run1.txt`, `..._run2_18of18.txt`); the achievable-n function and mutants H19 to H21 were added afterwards: **21 of 21 caught** in the final file `wp10b_evidence/mutation_wp10b_b5.txt`).
