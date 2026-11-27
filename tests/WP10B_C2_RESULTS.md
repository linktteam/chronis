# WP10B C2 results: the hard-stratum ceiling, oracle against the production fit (2026-11-27)

Every number: Python 3.12.3, repo pins, the development 2-core AVX-512 machine. Design and rule: `tests/PREDECLARED_CRITERIA_WP10B.md` sections 5 and 8 (Part 2), frozen in `tests/DESIGN_FREEZE_WP10B_C2.txt` after the development run and before the evaluation. Development users 11101..11136 (360 fits; no claim rests on them); evaluation users 11141..11188 (16 per stratum, 480 production fits, 0 failed), **analysed once** (`wp10b_evidence/c2_eval_analysis.txt`, `.json`). Owner decision W10B-1 applies: the evaluation was read once and nothing was changed after it.

## The question
Is the hard stratum withheld because it cannot be served by design (even the TRUE regimes are too close for the identity gate), or because the production fit is worse than it could be? The oracle quantity is `lineage.separation_min` computed with the true labels of the window's days and the sample means of the true regimes (arm O; arm G uses the generator's true means); the production quantity is `separation_min` of the production-selected restart. The gate passes at `separation_min >= 2.5` (h of the chosen operating point). An evaluable refit has every one of the 3 true regimes on at least 3 days with an observed value per feature. Intervals: user-level bootstrap, 2000 resamples, seed 11999.

## Verdict (declared rule 5.3, hard stratum, evaluation set)
**NOT SERVABLE BY DESIGN, on the point estimate.** pO = 0.431 (the share of evaluable hard refits whose oracle separation passes the gate; user-bootstrap 95% interval 0.206 to 0.669; naive Wilson over refits 0.357 to 0.509) is below the 0.50 line, but the interval reaches 0.50, so the word "demonstrated" is not allowed. The "needs better fitting" branch (pO >= 0.50 and a gap of at least 0.20 with a lower interval end above 0) is not reached: the gap oracle minus production on the separation alone is **-0.050** [-0.131, 0.025] (the production fit's separation is not lower than the oracle's).

## The table (RUN; evaluation set)
| stratum | users, refits (evaluable) | median separation (gate units): oracle / generator means / production | share passing the gate: oracle | production, separation only | production, separation and not flagged | users with the oracle passing at least once | users with the production full gate passing at least once |
|---|---|---|---|---|---|---|---|
| hard | 16, 160 (160) | 2.38 [1.79, 3.44] / 2.42 / 2.43 | 0.431 [0.206, 0.669] | 0.481 [0.287, 0.681] | 0.194 [0.069, 0.344] | 8 of 16 | 7 of 16 |
| medium | 16, 160 (159) | 4.69 [3.65, 6.34] / 4.53 / 4.48 | 0.937 [0.810, 1.000] | 0.899 [0.748, 1.000] | 0.560 [0.354, 0.755] | 15 of 16 | 12 of 16 |
| easy | 16, 160 (160) | 7.68 [6.77, 8.61] / 7.55 / 7.06 | 1.000 | 1.000 | 0.944 [0.869, 0.994] | 16 of 16 | 16 of 16 |
Gap oracle minus production (separation and not flagged): hard 0.238 [0.113, 0.381]; medium 0.377 [0.196, 0.579]; easy 0.056 [0.006, 0.131]. Medium and easy are reported with no pass/fail (rule 5.3).

## What it means (INFERENCE where marked)
- **For the hard stratum the limit is the data, not the fit:** the median separation of the TRUE regimes in gate units is 2.38, below the gate of 2.5, and the production fit's separation (2.43) is as large as the oracle's. Withholding hard users is what the gate is designed to do. A better fit of the means cannot raise the oracle ceiling by construction: the oracle already uses the true labels.
- **What the production fit loses is the instability flag, not the separation:** in the hard stratum the share passing the separation alone is 0.481 and drops to 0.194 once fits flagged as unstable (consensus flag and sharpness floor) are removed. The oracle has no flag, so the 0.238 gap is the price of refusing fits whose partition is not reproducible; this is the safety mechanism working, and it is consistent with WP10A (28 of 36 hard users never served).
- **The evaluation is a boundary case:** the development users gave pO about 0.54, the evaluation users 0.43; with 16 hard users the interval is wide (0.21 to 0.67). The word is "on the point estimate" and I do not move the 0.50 line or add users after seeing it (owner decision W10B-1). A second look would need new seeds and the owner's go-ahead.
- **In medium the production loses about 0.38 of refits to the flag** against an oracle that passes 94%: this is where better fitting could pay, and it is the owner's call whether a different level applies to medium (rule 5.3 states the rule only for hard). Per decision W10B-1 no hard-tier-specific tuning follows from this result; any generic fitting change is a hardening-track item, developed on development seeds and evaluated once.

## What it does not show
The oracle gate quantity is the separation of the regimes, not the whole identity decision: the margin rule of the matching, the alignment itself, the sharpness of the posterior, and the correctness of the names are not part of it. It does not say hard users would be served correctly if the gate were lowered (WP10A: the gate cannot be lowered safely; unchanged). Cross-machine reproducibility is not claimed. Synthetic streams only.

## Tests and mutation controls (earlier in this package)
`tests/test_wp10b_c2.py` (41 tests, pass again now), mutation set `mutation_wp10b_c2.py` (27/27 caught, `wp10b_evidence/mutation_wp10b_c2.txt`, run at the C2 commit; not rerun for this note).
