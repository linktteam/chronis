# WP10B B3 results: the dual predictive certificate (2026-11-27)

Every number: Python 3.12.3, repo pins, the development 2-core AVX-512 machine. Rules and levels: `tests/PREDECLARED_CRITERIA_WP10B.md` Part 5 (section 11), committed before any B3 scored run. Study library `integration_report/wp10b/scripts/dual_cert_lib.py` (production `cold_start_cert.py` unchanged; nothing is switched on). Configuration: the B2-frozen **I600** (`tests/DESIGN_FREEZE_WP10B_B2.txt`), setting S0 (identity streams, K = 3, F = 7, train days [0, 120), test days [120, 150)). **Owner decision W10B-1:** the frozen version met the evaluation users once, the data were analysed once (`wp10b_evidence/b3_analysis.txt`, `.json`), nothing was changed after. Natural arm users 11421..11456 (12 per stratum) plus the pre-declared extension 11457..11480 (24 users): the blind count of eligible harmful fits in the first 36 was **0**, below 14, so the extension ran as declared (`wp10b_evidence/b3_blind_count.txt`); pooled 60 users, 0 fits failed to run. Wrong-engine arm: 80 fits (N18 generator, seeds 11501..11550), 0 failed.

## Verdict (declared words; B3 as a whole: NOT MET)
| claim | result | word |
|---|---|---|
| R1 refusal, natural S0 fits | 13 eligible fits (certificate asked), 2 harmful, 0 certified; exact bound 77.6% | **NOT DEMONSTRATED** (fewer than 14 harmful) |
| R2 refusal, wrong engine `stops` | 30 harmful, 0 certified; bound 9.5% | **MET** |
| R2 refusal, wrong engine `B2` | 14 harmful (4 strictly), 0 certified; bound 19.3% | **MET** |
| R3 useful share, easy + medium, natural | 8 good eligible fits, 0 certified (Wilson 0.00 to 0.32) | **NOT DEMONSTRATED** (fewer than 12 good) |
| reported, no verdict: all strata | 11 good, 0 certified (Wilson 0.00 to 0.26); strictly good 4, 0 certified | – |
| reported, no verdict: real engine, N18, certificate forced | 19 good fits, 5 certified = 0.26 (Wilson 0.12 to 0.49) | – (the analysis printout carries a "NOT MET" label for this line; the rule gives it no verdict) |
| R4 reported: certified fits that also beat the baseline refits | 0 of 0 | – |
"B3 as a whole is MET iff R1, R2 and R3 are MET": R1 and R3 are NOT DEMONSTRATED, so **B3 is not MET**. Two B2-engine fits were certified; both have a held-out gain above the single Gaussian, so they are not harmful controls (reported, not counted, rule 11.5).

## Per stratum (RUN; 20 users each)
| stratum | converged-publishable | eligible (certificate asked) | certified | published | dual minus Gaussian (se) | dual minus baseline engine (se) |
|---|---|---|---|---|---|---|
| hard | 14 | 5 | 0 | 14 | -0.033 (0.148) | -0.311 (0.171) |
| medium | 18 | 1 | 0 | 18 | +0.095 (0.276) | -0.607 (0.293) |
| easy | 13 | 7 | 0 | 13 | +0.462 (0.155) | -0.321 (0.294) |
Held-out one-step mean log density per user, paired; "published" = converged-publishable or certified.

## What it means (INFERENCE where marked)
- **The certificate refuses the planted harmful engines.** For the two wrong engines (0 of 30 and 0 of 14 harmful fits certified) the declared refusal level is MET. This is the same rule WP10A A6 and C1(2) tested with priors; here it is applied to the dual fit with the engine itself wrong.
- **On natural S0 fits it was never asked a hard question.** With I600 most fits converge (31 of 40 easy + medium, 45 of 60 overall) and need no certificate; only 13 were eligible and only 2 were harmful. R1 and R3 are NOT DEMONSTRATED for lack of n, as I said before the run, even after the extension. I did not add users beyond the declared extension (W10B-1; more would need the spare seeds 11901..11999 and the owner's go-ahead).
- **The certificate certified none of the 13 natural eligible fits, good or not**, and 5 of 19 good fits of the real engine on the N18 generator. INFERENCE: with three origins at 60, 78 and 96 rows a margin of 3 standard errors is hard to pass for a K = 3, F = 7 fit; it is conservative rather than selective. Whether that is the right trade for a fail-closed gate is the owner's call.
- **A converged fit is not a good fit.** The published dual fits (converged-publishable, nothing else passes) score below the baseline engine on the held-out days in every stratum (see the B4 memo). The certificate never touched them.
- **Not shown:** one setting (S0), one configuration (I600), 20 users per stratum, synthetic streams, no real users; no claim about the dual on other generators. Cross-machine reproducibility is not claimed.

## Tests and mutation controls
`tests/test_wp10b_dual_cert.py` (20), `tests/test_wp10b_b3_run.py` (9), `tests/test_wp10b_b3_analyze.py` (16); mutation sets `mutation_wp10b_b3.py` (28) and `mutation_wp10b_b3_analyze.py` (36), run at their commits and rerun at the final head (see the report).
