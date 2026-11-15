# WP6 part B: N17 (parameter uncertainty in `m_t_std`): NEGATIVE RESULT, engine not changed

Design, constants and criteria B1 to B6: `tests/PREDECLARED_CRITERIA_WP6.md` section 7 (7.1 to 7.4 before any run, 7.5 amendment after one development seed, 7.6 frozen choices before the evaluation). Scripts: `integration_report/wp6/scripts/n17_proto.py`, `n17_study.py`, `n17_diag_truth.py`. Evidence: `wp6_evidence/n17_dev*.jsonl`, `n17_eval.jsonl`, `n17_cells.jsonl`, `n17_diag_iter100.jsonl` and the `*_summary.txt` files. Labels: RUN unless stated. There is NO engine code for this part: the term lives only in the prototype script, `backbone/` is unchanged by part B, so B4 (identity) holds trivially and B5 (mutants) has nothing to test.

## What was built (prototype, script level)
Delta method on the smoothed latent mean: parameters = the 50 continuous parameters (K = 2, L = 2, F = 4), information = minus the numeric Hessian of the engine's own log-likelihood (90 s per fit), J = numeric Jacobian of the engine's own smoothed mean (9 s), the exact latent-frame directions removed in parameter space, negative eigenvalues reflected (policy R, frozen on development), and, after the first development seed showed intervals 5 times too wide (amendment 7.5), each regime's own affine latent frame removed in OUTPUT space using the MAP regime labels of the same call. `m_t` is untouched.

## Evaluation (seeds 7401..7424, once; N17 generator, 20 iterations as cell B)
Mean coverage over seeds and the two latent dimensions (nominal 0.9545 at +-2, 0.6827 at +-1; standard error across seeds in brackets):

| arm | +-2 | +-1 |
|---|---|---|
| E0 estimated parameters, unchanged `m_t_std` | 0.716 (0.038) | 0.460 (0.031) |
| **E1 estimated parameters, with the term (policy R)** | **0.818 (0.030)** | **0.597 (0.033)** |
| E1 with policy C (drop negative eigenvalues) | 0.812 | 0.591 |
| E1 without the output projection (the 7.1 version) | 0.977 (0.008) | 0.910 (0.023) |
| T0 true parameters, unchanged (control) | 0.950 (0.006) | 0.675 (0.009) |
| T1 true parameters, term computed at the truth | 0.969 (0.006) | 0.765 (0.013) |

Development seeds 7301..7312 gave the same picture (E0 0.781 / 0.495, E1R 0.859 / 0.642). Mean z^2 (nominal 1): 11.5 without the term, 5.7 with it (heavy tails). No seed got worse with the term (0 of 24); 8 of 24 reach at least 0.9145 on average over the two dimensions. All 24 fits end MAX_ITER_REACHED at 20 iterations (the N17 setting).

## Criteria
* **B1 (E1 calibrated): FAIL.** 0.818 against the band 0.9145 to 0.9945 at +-2, and 0.597 against 0.6427 to 0.7227 at +-1. The term closes about half of the gap at +-2 and a bit over half at +-1.
* **B2 (not a blanket widening): PASS.** T1 0.969 and 0.765 (ceilings 0.99 and 0.85); E1 is below T1 at both levels.
* **B3 (the two cells): the prototype passes both cells at the unchanged floors** (RUN, `n17_cells_summary.txt`): cell A (seed 202, T = 60, floor 0.8738) 0.883 -> 0.950 in dim 0 and 0.967 -> 1.000 in dim 1; cell B (seed 404, T = 64, floor 0.8764) 0.828 -> 0.906 in dim 0 and 0.938 -> 0.984 in dim 1. The cells are milder than a typical seed (WP5: cell B at the 79th percentile), so passing them says less than B1 does. Because B1 fails, the pre-declared rule applies: **the xfail on cell B dim 0 is left in place, no engine code is added.**
* **B4 identity: holds trivially (no engine change). B5 mutants: not applicable (no engine code). B6 cost, reported:** 90 s Hessian plus 9 s Jacobian per fit for P = 50; the cost grows with P^2 (about 4 minutes at P = 90, impractical at P of 200), no negative eigenvalue fell below the numerical tolerance (0 excluded), 2 to 6 negative eigenvalues per fit were reflected.

## Diagnostic that is not a criterion (RUN, development seeds 7301..7308, `n17_diag_iter100_summary.txt`)
Hypothesis tested: the remaining gap is under-fitting (all fits stop at 20 iterations). Result: the hypothesis is wrong. With 100 iterations the log-likelihood rises by 3.4 nats on average, and coverage gets WORSE: E0 0.826 -> 0.692 and E1 0.902 -> 0.807 at +-2 (fits still MAX_ITER_REACHED). The better the in-sample fit, the larger the shortfall, which fits missing parameter-estimation error and not optimisation error.

## Reading
* The earlier WP5 reading (missing parameter uncertainty is the cause; true-parameter coverage is nominal) is supported: adding a parameter term lowers the coverage of none of the 24 seeds (mean +0.10 at +-2) and moves the true-parameter arm only from 0.950 to 0.969.
* This particular term is not enough: it reaches 0.82, not 0.95. Candidate reasons, NOT tested here: the Laplace (quadratic) approximation of a likelihood with 50 parameters on 60 to 70 rows; the discrete parameters (switch probabilities, durations) and the regime labels are held fixed in the term; the delta method takes the information at a point that is not an exact maximum (2 to 6 negative eigenvalues); the size of the term depends strongly on the latent-frame projection (ratio 1.7 with each regime's frame removed, 7.1 without, coverage 0.82 against 0.98), so the answer is sensitive to a modelling choice that the data cannot settle. A parametric bootstrap (about 20 warm-started refits per fit) would avoid the quadratic approximation but needs a regime-wise frame alignment of each replicate; not tried.
* Product consequence: `m_t_std` of the dual model should not be read as a calibrated interval in this setting (the old +-2 intervals cover about 72% on average, about 95% when the parameters are known). The cost of an honest term (about 100 s per fit at P = 50) is also a product decision.

## Not done
Engine integration, tests and mutants for the term (conditional on B1), docs, the optional n0 = 1 and 2 grid (6001..6012, 7501..7524; the default n0 is unchanged).
