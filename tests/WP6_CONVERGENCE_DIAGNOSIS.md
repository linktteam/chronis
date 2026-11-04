# WP6 item 2: convergence and M-step diagnosis (read-only)

Written and committed BEFORE any fix. Questions, definitions and the words each statement may use were fixed earlier in `tests/PREDECLARED_CRITERIA_WP6.md` section 3 (commit d880c74). Nothing in the engine changed in this item. Labels: RUN (I ran it), READ (I read code), REPORTED (told to me, not reproduced). Scripts: `integration_report/wp6/scripts/diag_baseline.py`, `diag_dual.py`, `diag_baseline_ari.py`, `diag_dual_cont.py` and the `analyze_*.py` summaries; every result below is in `wp6_evidence/` (`summary_*.txt`, `diag_*.jsonl`). Tree: `ledger/wp6` at 3abf49c (item 1 only; the model code is the WP5 code).

## 0. In ten lines

1. RUN. The owner's repro is reproduced exactly: 8 of 10 restarts `CONVERGED`, 2 `REJECTED_UPDATE` (drop 0.132 and 0.133 nats at iteration 8).
2. RUN. Baseline restarts end `REJECTED_UPDATE` in 20% (repro), 36% (F2, 200 fits) and 32% (F3, 200 fits) of restarts. All 138 events came from the DURATION block: its expected complete-data log-likelihood fell in 138 of 138 events (rule "the decrease is the M-step": satisfied). 0 of 138 drops were at floating-point noise level (rule "the decrease is numerical noise": not satisfied); drops are 1e-4 to 0.48 nats (median 6.1e-4 in F2, 2.5e-3 in F3).
3. READ + RUN. Mechanism of the duration drop: the M-step starts L-BFGS-B at the UNTRUNCATED moment estimate (not at the previous iterate), runs at most 5 iterations with a loose tolerance, and in 1.6% to 2.1% of M-steps falls back to the moment estimate itself, which ignores the truncation. Nothing in the code makes the new duration parameters at least as good as the old ones under the same E-step statistics.
4. RUN. The baseline never ended `MAX_ITER_REACHED` at 200 iterations (0 of 410 fits) and convergence is rarely premature (2 of 272 `CONVERGED` fits, 0.7%; both in F3, one gaining 16 nats). The "premature" rule (above 5%) does not fire.
5. RUN. Collapsed baseline fits (a variance at the floor, or a regime under 4 effective days) are 15% (F2) and 48% (F3) of fits and often have a higher log-likelihood than every healthy restart of the same dataset; the library's selection (best log-likelihood among `CONVERGED`) picks a collapsed restart in 4 of 20 (F2) and 13 of 20 (F3) datasets; in F3, 6 of 20 datasets have no healthy converged restart at all. The floor explains the higher log-likelihood for 29 of 38 F3 fits and for 2 of 10 F2 fits (rule: at least 80% of the gap; the other F2 cases have regimes under 4 days, "not shown").
6. RUN (added question Q2b, dated below). The collapse flag does not by itself mark a worse labelling: among `CONVERGED` fits, floor-only fits have the same adjusted Rand index (ARI) against the true regimes as healthy ones (F3: 0.605 against 0.574); fits with a regime under 4 effective days are lower on average (F2 0.774 against 0.835, F3 0.453 against 0.574). Choosing restarts to avoid regimes under 4 days changes mean ARI by -0.0015 (standard error 0.0087, 40 datasets; 5 datasets differ).
7. RUN. Dual model: 0 `REJECTED_UPDATE` and 0 silent drops (1e-4 to 1e-3) in 44 fits, so the dual guard question is empty; 44 of 44 fits end `MAX_ITER_REACHED` (100, 30 and 200 iterations) and every one is still rising (median gain 0.195 nats over the last 10 of 200 iterations). Also: the dual sets `n_iter_ = 0` on `MAX_ITER_REACHED` (READ and seen in the output).
8. RUN. Dual duration degeneracy: 10 of 40 regimes in 8 of 20 users (NEW), 5 of 24 in 4 of 12 (DEV, 100 iterations), 4 of 24 in 3 of 12 (WP5 recipe, 30 iterations). It is NOT in general a ridge (KL to the nearest ordinary pmf below 1e-3 in 3 of 10, 1 of 5, 1 of 4: rule "ridge" not satisfied); seven of the ten have sigma at the wall 10 and mu from -165 to +146 (a pmf close to 1/d, insensitive to mu), three have mu about 10 and sigma 1.6 to 2.2 (a pmf rising to the lattice end; these three ARE reparametrisations of ordinary pmfs). It is strongly associated with segments longer than Dmax = 25 in the data (NEW: 7 of 10 users with such a segment against 1 of 10 without, Fisher p = 0.020; pooled with DEV 9 of 13 against 3 of 19, p = 0.0035). Seed 6010 (WP5): (mu, sigma) = (2.42, 0.65) at iteration 1, (16.95, 4.5) at iteration 2, (20.07, 4.94) at iteration 3 and frozen until iteration 29 (Nelder-Mead jumped onto a flat region of the objective and stayed).
9. RUN. No dual regime has under 4 effective days (smallest 12.0), no observation variance at the floor, no state-noise eigenvalue at the floor, 0 degraded fits: the dual model shows no collapse in these populations.
10. READ. Consequence for the next item: with the rule "a fit that is not `CONVERGED` is not publishable" the dual model at its default settings would never publish (20 of 20 users end `MAX_ITER_REACHED`). The stopping rule for the dual therefore needs its own evidence (addendum Q4b) before item 3 fixes it.

## 1. Deviations and additions (stated, not hidden)

* The pre-declared instrumented copy of the EM loop reproduces the library bit for bit in every fit where it ran: baseline 410 of 410, dual 23 of 23 (DEV 12, WP5-recipe 3, NEW 8). For dual fits other than DEV, REJECTED or degenerate ones the analysis reads the library fit directly (the instrumented loop doubles the run time); this was declared in the script header before the run. It is not a deviation from the questions: no dual REJECTED fit exists.
* Two questions were ADDED after seeing results, each stated in its script before it ran: Q2b (does the collapse flag mark worse labels? script `diag_baseline_ari.py`) and Q4b (how far does a dual fit move after the 200 iterations that `fit_hssm` allows? script `diag_dual_cont.py`). Both are descriptive; no criterion of section 3 was edited.
* The "flicker" regimes (a regime whose duration pmf concentrates at d = 1 and that has at least 4 entries, not degenerate by the declared definition) were noticed in the dual output: 2 of 40 regimes in NEW (users 5108, 5111) and 2 of 24 in DEV. They are listed, not analysed further.
* The F1 family was run at 60 iterations (as the repro is written) and at 200; the results are identical.

## 2. Q1 REJECTED_UPDATE

| family | fits | `REJECTED_UPDATE` | events | drop min / median / max (nats) | at most 1e-6 | 1e-6 to 1e-3 | above 1e-3 | block with dQ < 0 | previous M-step used the moment fallback |
|---|---|---|---|---|---|---|---|---|---|
| F1 owner repro | 10 | 2 (20%) | 2 | 0.132 / 0.132 / 0.133 | 0 | 0 | 2 | duration 2 of 2 | 0 |
| F2 repro construction, 20 datasets | 200 | 72 (36%) | 72 | 0.0001 / 0.00061 / 0.391 | 0 | 38 | 34 | duration 72 of 72 | 24 |
| F3 correlated bounded features | 200 | 64 (32%) | 64 | 0.00012 / 0.0025 / 0.482 | 0 | 25 | 39 | duration 64 of 64 | 12 |
| dual (DEV, WP5 recipe, NEW) | 44 | 0 | 0 | | | | | | |

Events happen at iteration 5 to 29 (F2, median 12) and 4 to 69 (F3, median 11); at the event the total Q also fell in every case (138 of 138), so the E-step approximation is not involved. The emission, transition and initial blocks never had dQ < 0. READ: the three components of the duration M-step that allow a decrease are (a) the start at the moment estimate instead of the previous iterate, (b) `maxiter=5, ftol=1e-2`, (c) the fallback to the moment estimate when the optimiser reports no success.

## 3. Q2 collapse (baseline) and Q2b

* Collapsed fits: F1 1 of 10; F2 30 of 200 (29 `CONVERGED`, 1 `REJECTED_UPDATE`; 7 with a variance at the floor, 30 with a regime under 4 effective days); F3 97 of 200 (62 `CONVERGED`, 35 `REJECTED_UPDATE`; 95 with a floor, 9 under 4 days).
* Collapsed fits with a higher log-likelihood than the best healthy converged restart of the same dataset: F2 10 fits in 4 datasets (gaps 7.1 to 13.9 nats), F3 38 fits in 8 datasets (gaps up to 455 nats). Floor-explained share at least 80%: F2 2 of 10 fits, F3 29 of 38 fits. For the other 8 F2 fits (regimes under 4 days, no variance at the floor) the floor does not explain it; the likely cause (a variance estimated from one to three days) is not shown by a measurement here.
* What the library selects (best log-likelihood among `CONVERGED`): a collapsed restart in 4 of 20 (F2) and 13 of 20 (F3) datasets, none in F1; in F3, 6 of 20 datasets have no healthy converged restart.
* Q2b, ARI against the generating regimes, `CONVERGED` fits:

| family | healthy | floor only | regime under 4 days |
|---|---|---|---|
| F2 | n 99, mean 0.835 | none | n 29, mean 0.774 |
| F3 | n 74, mean 0.574 | n 53, mean 0.605 | n 9, mean 0.453 |

  `REJECTED_UPDATE` fits in F2 have a MEAN ARI of 0.939 (n 71 healthy): they stop at the best parameters seen before the drop. This matters for item 3: once a repaired fit continues past the event, these fits will end elsewhere. Selection rules, mean ARI over datasets: library 0.864 (F2) / 0.531 (F3); best log-likelihood among unflagged converged restarts 0.846 / 0.526; among restarts without a regime under 4 days 0.846 / 0.545; oracle (best ARI among converged restarts) 0.888 / 0.669. Paired over the 40 datasets, "without a regime under 4 days" minus library = -0.0015 (se 0.0087), differing in 5 datasets (-0.115, -0.110, -0.102, -0.021, +0.286). So: avoiding collapsed restarts is NOT shown to improve labels; the floor flag alone carries no signal about labels; a regime under 4 effective days is associated with a lower ARI.

## 4. Q3 dual duration degeneracy

See point 8 of the summary. Counts by the declared definition (`dur_sigma >= 3` or `exp(dur_mu) > 10 * Dmax`): NEW 10 of 40 regimes (7 with sigma at the wall, 3 "other" with mu 9.0 to 10.9 and sigma 1.56 to 2.15), DEV 5 of 24, WP5 recipe 4 of 24. Nelder-Mead reported success in every degenerate regime. The pmfs at sigma 10 are within KL 0.0013 to 0.0061 nats of the best ordinary pmf: not a ridge by the declared rule, but flat enough that the optimiser has no signal along mu. Mechanism proposed by the data, NOT proven: a regime whose entry mass is split between very short visits (d = 1, the model re-entering the regime because a segment longer than Dmax cannot be represented and self-transitions are not allowed) and long ones would be best fitted by a flat pmf; a 1/d-like pmf with sigma at the wall is what such an objective favours (not verified by a direct computation here). Nothing in the data distinguishes this from a plain optimiser failure for seed 6010, whose trajectory (iteration 2 jump, then frozen) points to the optimiser.

## 5. Q4 stopping

| model | setting | fits | `CONVERGED` | `MAX_ITER_REACHED` | still rising (above 0.01 nats over the last 10) | premature among `CONVERGED` (above 0.01 nats in 50 more) |
|---|---|---|---|---|---|---|
| baseline | 200 its, F1/F2/F3 | 410 | 272 | 0 | 0 | 2 of 272 (0.7%; F3 1.5%, max gain 16 nats; F2 and F1 0) |
| dual | 30 its (WP5 recipe) | 12 | 0 | 12 | 12, median 1.11 nats | not assessable |
| dual | 100 its (DEV) | 12 | 0 | 12 | 12, median 0.375 | not assessable |
| dual | 200 its (NEW) | 20 | 0 | 20 | 20, median 0.195 (min 0.026, max 0.805) | not assessable |

Baseline iterations to convergence: median 12 to 18, 90th percentile 22 to 25, maximum 146. The baseline duration optimiser fell back to the moment estimate in 0 of 164 (F1), 45 of 2863 (F2, 1.6%, 39 fits) and 65 of 3113 (F3, 2.1%, 25 fits) M-steps. The dual `fit` leaves `n_iter_ = 0` when the cap is reached.

## 6. Q5 candidate fixes and their risks

| cause | candidate | what it fixes | risks |
|---|---|---|---|
| inexact duration M-step (baseline `REJECTED_UPDATE`) | A1: when the guard would fire, redo that M-step's duration block with a safeguarded update (candidates: previous iterate, the old optimiser result, a tight L-BFGS-B started at the previous iterate; keep the one with the best Q), continue; keep the safeguard for the rest of that fit | the cause (the M-step could lower its own Q); fits that never reach the guard run the same code as before (bit-identical) | fits that used to stop at the event now run on and may end `CONVERGED` at a different, possibly collapsed, optimum (see section 3, F2 `REJECTED_UPDATE` ARI 0.939); extra cost only at events |
| same | A2: always use the safeguarded update | cleaner code | changes the numbers of every fit; excluded by the owner's rule |
| same | A3: raise the guard tolerance | fewer rejections | hides real decreases (0 of 138 are noise); excluded |
| collapsed regime wins on log-likelihood (baseline) | B1: report a collapse measure per fit; among `CONVERGED` restarts prefer ones without a regime under 4 effective days; if all have one, return the best and withhold | the failure the owner saw when a collapsed restart is selected | measured effect on label accuracy is nil (-0.0015, se 0.0087); more fits withheld; after A1 more collapsed restarts become eligible (their rejection used to hide them) |
| same | B2: a relative variance prior or a higher floor | removes the unbounded likelihood | changes healthy fits; excluded |
| same | B3: reduce K when a regime collapses | model selection | outside this item |
| dual `MAX_ITER_REACHED` | C1: raise the cap | converge | cost: about 0.6 s per iteration; slow sublinear EM, so unclear |
| same | C2: certify at the cap when the log-likelihood gain over the last window is below a threshold; otherwise stay `MAX_ITER_REACHED` | a fit that has stopped moving is called converged; a moving one is not | the threshold needs evidence (addendum Q4b); a rule that stops earlier than the old one would change fits that used to end `CONVERGED`, so it is applied only at the cap |
| same | C3: acceleration (SQUAREM etc.) | speed | a large change; out of scope |
| dual degenerate durations | D1: box on the duration M-step (sigma at most 3 and mu in [-5, ln(10 Dmax)], the declared degeneracy limits) and flag the regime when the optimum is at a bound | stops the jump onto the flat region and the unbounded mu | a regime whose true duration pmf needs a larger sigma loses a little log-likelihood (flagged); healthy regimes (sigma 0.1 to 1.4) are unaffected |
| same | D2: flag only | transparent | leaves the degenerate fit as it is |
| same | D3: warn when a segment in the data may exceed Dmax | the data-side cause | the engine does not see segments, only rows; out of scope as a hard check |
| multi-start stability (ledger N1) | E: report the number of restarts that converged, that are healthy, the log-likelihood gap to the best distinct solution, and the ARI between the selected labelling and the other converged restarts | the caller can see a fragile solution | metadata only; its value as a predictor of label quality is not shown here |
| non-`CONVERGED` fits reach users | F: `withheld_reason` also fires when `fit_state` is not `CONVERGED` (and for a selected collapsed fit) | P0-07, fails closed | the dual model at the default cap would never publish unless C2 or a longer cap makes its fits `CONVERGED`; `MODEL_SELECTION_CONTESTED` results (a different, existing state) are also affected |

## 7. What this diagnosis did not show (UNKNOWN)

* Why the real integration case (73 days, K = 3) ended with ARI 0.514 against 1.000 for a Gaussian mixture is not reproduced by F1, F2 or F3: F1 selects a healthy converged restart and the repro does not carry the owner's data. UNKNOWN whether its healthy converged solution is a poor local optimum of the HSMM or a model-misfit effect; the multi-start stability metadata (E) is aimed at making that visible.
* Whether the dual duration degeneracy is caused by segments longer than Dmax (associated, p = 0.0035, not shown causal) or by the optimiser alone (seed 6010 points at the optimiser).
* Premature convergence of the dual model (no dual fit reaches `CONVERGED` in 200 iterations).

## Addendum Q4b (dual continuation to 400 iterations), dated 2026-11-01

Filled in on 2026-11-03/04 from `wp6_evidence/summary_dual_CONT.txt` (script `diag_dual_cont.py`; users 5101..5108, plain EM continued to 400 iterations, reference = the fit at iteration 200). READ from that file:

* The log-likelihood keeps rising: median gain 1.51 nats from iteration 200 to 400 (maximum 6.94); median gain 1.63 from 100 to 200; the gain over 200 more iterations is 0.82 of the gain over the previous 100 to 200 (median ratio). The EM is sublinear, not converged at 200 iterations for any of the eight users.
* The MAP regime labels barely move after iteration 200: median 0 of the rows changed, maximum 0.86% of the rows (user 5102).
* The continuous state `m_t` moves more: largest change over the record 0.05 to 0.62 standard deviations of `m_t` (median 0.21), mean absolute change 0.012 to 0.18 sd (median 0.044).
* The movement follows the remaining gain. With the trailing 10-iteration gain per row below 2.0e-4 at iteration 200 (users 5107 and 5108: 1.87e-4 and 1.11e-4 per row) the largest `m_t` change was 0.08 and 0.05 sd and the label change at most 0.62% of the rows; above 3.6e-4 per row it was 0.13 sd or more.
* First iteration at which the trailing-10 gain falls below x nats (users 5101..5108): 0.50 nats at iterations 131, 69, 56, 51, 38, 246, 44, 50; 0.20 nats at 281, 147, 104, 88, 185, 361, 69, 70; 0.10 nats at none, 246, 187, 144, 384, none, 101, 97; 0.05 nats and below for most users not within 400 iterations.
* Of the 20 NEW users (5101..5120) at 200 iterations, 2 (10%) have a trailing gain per row below 2.0e-4.

Consequence used by item 3 part 2 (tests/PREDECLARED_CRITERIA_WP6.md section 5.1): certify at the iteration limit when the trailing 10-iteration gain per row is below 2.0e-4; about 90% of dual fits at the default 200 iterations stay `MAX_ITER_REACHED` and are withheld. A larger `max_iter` or an accelerated EM would raise the share (out of scope for WP6).
