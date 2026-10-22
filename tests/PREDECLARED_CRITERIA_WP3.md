# Pre-declared criteria: work package 3 (N18, hold-back switch, D12 notice, dependencies, close-out)

Written and committed BEFORE any new WP3 test was run (brief section 0, rule 4). Criteria below are not edited after results are
seen. If one turns out wrong it is reported as a finding and a NEW dated section is added below; the original text stays.
Branch `ledger/wp3`, created from `ledger/wp2` (head `08a4c76`).

## 0. What I had already seen before writing this file (so the thresholds are not presented as blind)

* From WP2 (RUN): the A2 matrix (`A2_matrix.txt`): per-regime-aligned R² 0.974 / 0.941; coverage 0.8281 on cell B dim 0 (N17); B1 not
  detected by the A2 tests, B2 detected in one calibration cell only; fallback firing counts 0 everywhere except the forced tests.
* This session, BEFORE writing this file, I measured only fit durations (no scores, no predictive densities, no N18 comparison of any kind),
  on a throw-away seed (7), `max_duration=25`, `n_iter=20`: real engine 4.3 s (T=55) and 6.0 s (T=79); the "stops learning" variant
  0.3 s / 0.5 s (it reaches `CONVERGED` after 2 iterations); B2 2.1 s / 3.6 s (it ends in `REJECTED_UPDATE`, i.e. the monotonicity guard
  rolls back); one forward pass 0.12 s (T=55). These timings fixed the number of seeds below.
* I READ `data/synthetic_dual.py`, `tests/utils/synthetic_dual_fits.py`, `DualTimescaleHSSM._forward_pass` and `_init_params`,
  `HSSMResult` and both places that construct it (`fitting.py` l.605 and l.738), and `fit_dual_with_random_restarts`.
* I have NOT seen any held-out predictive log density of any engine, nor any result of the B1-against-oracle run.

## 1. Item 1: N18, held-out prediction test

### 1.1 Data (same family as the A2 cells)
* Generator `data.synthetic_dual.generate_synthetic_dual_timescale_data(n_regimes=2, n_features=4, latent_dim=2, n_segments=4, mean_dwell=15.0, seed=s)`
  gives the TRAINING series (T about 55 to 64, exactly the A2 construction; seeds 202 and 404 give the A2 cells A and B).
* HELD-OUT series: the same generative process continued for 2 more segments (about 30 steps, 16 to 60 possible) from the generator's own
  random stream, so for seeds 202 and 404 the training part is bit-identical to the A2 data. The continuation helper replays the generator's draws;
  a test requires its training part to equal the generator's output exactly (`np.array_equal` on X, regimes, latent). If it does not, the helper raises.
  Regime sequence continues as the generator does it (the regime after the 4th segment is the one the generator drew and did not use).
* Seeds: `(202, 404, 505, 606, 707, 808, 909, 1010)`, n = 8. Reason: 8 fits of each of three engines cost about 1 minute (timings above), within
  the "new tests add under about 3 minutes" budget; a multi-seed study with more seeds belongs in a script, not in pytest.
* Fit settings per seed: `DualTimescaleHSSM(n_regimes=2, n_features=4, latent_dim=2, max_duration=25, seed=s)`, `fit(X_train, n_iter=n)` with n = 25 for
  seed 202 and n = 20 for every other seed (A2 cell A and cell B settings).

### 1.2 Score
* Parameters are frozen after `fit`. The forward filter (`_forward_pass`) is run on train + held-out; the one-step-ahead predictive log density is
  `lp_t = logsumexp(alpha[t]) - logsumexp(alpha[t-1])` (`alpha` is the joint of regime, remaining duration and y_1:t, so this is `log p(y_t | y_1:t-1)` under
  the model's own filter; it uses no future data). Held-out score of a seed = mean of `lp_t` over the held-out steps. Cross-check required in a test:
  that mean times T_test equals `LL(full) - LL(train)` within 1e-9.
* Health gate: every fit and every scoring pass must have `cholesky_fallback_count_` unchanged by scoring and `fit_degraded_ == False`;
  otherwise the seed's score contains the magic constant -1e6 and the test FAILS (it does not skip the seed).
* Why the mean and not the median: the mean log score is the strictly proper scoring rule; the median per-step difference is reported as information only.

### 1.3 Opponents (same data, same constructor seed unless stated)
* `baseline`: one Gaussian with the training mean and covariance (`np.cov`, ddof 1), time-invariant.
* `stops`: "stops learning" = `DualTimescaleHSSM` subclass whose `_update_discrete_params` and `_m_step_continuous` are no-ops (parameters stay at initialisation).
* `B2`: `tests.utils.synthetic_dual_fits.EngineADrivenToZero` (A_fast multiplied by 0.1 after each continuous M-step).

### 1.4 Statistic and criteria of the TEST
* Per seed s and opponent O: d_s(O) = score_real(s) - score_O(s). Over n = 8 seeds: mean d, standard error se = std(d, ddof=1)/sqrt(8) (paired).
* Criterion for an opponent: **d - 3.0 * se > 0** (the engine under test beats it by at least 3 paired standard errors).
  Reason for 3.0: under no true difference and roughly normal d_s the one-sided false-pass probability is P(t_7 > 3) = 0.0099 (about 1%); a false pass means a
  broken engine would be accepted, which is the dangerous error. If se = 0 and d = 0 (identical scores) the criterion fails.
* Required outcomes: **C1** real beats `baseline`; **C2** real beats `stops`; **C3** real beats `B2`. If the real engine fails one of them, that is a finding about the
  ENGINE (reported with the numbers); the criterion is not changed and no xfail is added.
* Test of the test (mutation check). Engine under test E replaced by a breaker with a DIFFERENT random initialisation than the opponent copy
  (constructor seed s + 1000): `S_alt` (stops-learning) and `B2_alt`. The conjunction C1 and C2 and C3 must FAIL for each. Reason: substituting the identical fit of an opponent
  fails by construction (difference exactly 0), which proves nothing, so that variant is run and reported but not counted.
* Reported, not criteria: per-seed scores, the sign count (seeds where real > opponent), median per-step difference, and the real engine's absolute score on the two N17 cells
  (seeds 202 and 404). These cell numbers are the baseline for the later N17 fix; the engine is not changed here.
* Also tested (metric tests, no engine): the aggregator accepts a consistently positive d, rejects zero-mean noise and rejects an all-zero d;
  the continuation reproduces the generator's training part exactly; the per-step increments sum to LL(full) - LL(train).

### 1.5 B1 (what N18 cannot see)
* B1 (smoother aggregated with the filtered posterior) changes `estimate_states`, not the forward filter, so N18 will not detect it; this is not claimed.
* Check (RUN, script): apply B1 as a class-level patch to `DualTimescaleHSSM._switching_smoother` and run the existing oracle-based files
  (the same ten files as the earlier mutation harness, `evidence/scripts/mutation_real.py` TESTS). B1 counts as caught iff at least one test that passes
  unmutated fails with B1. PREDICTION written now: caught (the earlier harness lists the same mutation, M3, as caught: READ `G6_mutation_check.log`).
  If it is NOT caught, I add a small oracle-based smoother check with its own mutation check (pre-declared design: lossless-collapse configs, compare
  `smoothed_m` and `smoothed_P` with the exact oracle, tolerances 1e-6 as in the existing file) and state that it is outside N18.

## 2. Item 2: hold-back switch and health plumbing

* `HSSMResult` gains `fit_degraded: bool = False`, `fit_cholesky_fallbacks: int = 0`, `fit_exceptions: list = []` (carried from the fitted model's `fit_degraded_`,
  `fit_cholesky_fallbacks_`, `fit_exceptions_`; safe defaults on every other path) and the derived, non-constructor fields `insights_withheld: bool`, `withheld_reason: str | None`.
* **Rule (pre-declared):** `insights_withheld = fit_degraded or (fit_state == FitState.NUMERICAL_FAILURE)`. `withheld_reason` is None when not withheld; otherwise it names
  every cause that applies (`fit_degraded: ...` and/or `fit_state is NUMERICAL_FAILURE`). `fit_state_` semantics are not changed.
* `publishable_outputs()` returns `None` when withheld, otherwise a dict with `p_t_estimate`, `p_t_posterior`, `m_t_posterior_mean`, `m_t_uncertainty`. Raw fields stay available.
* No user-facing notice (nothing is appended to `warnings`/`limitations`, no `warnings.warn`). On withholding `fit_hssm` emits exactly ONE `logging` WARNING on logger
  `backbone.hssm.fitting` containing the reason, the fallback count, the exception type names, `random_seed`, `user_id` and the dataset hash.
* Tests (each with a control): clean fit -> not withheld, outputs returned and equal to the raw fields, no log record; forced negative R (WP2 method: `_init_params` patched) through
  `fit_hssm` -> degraded, withheld, accessor returns None, raw fields still present, exactly one WARNING with the fallback count in it; injected M-step failure -> model `fit_state_`
  NUMERICAL_FAILURE and `fit_exceptions_` carried into the result, withheld; all restarts failing -> `FittingConvergenceError` (no result exists).
  Baseline (non-dual) path: defaults present, not withheld on a normal fit.
* Mutation checks (each must make at least one test fail): flag ignored; rule inverted; NUMERICAL_FAILURE not considered; accessor returns data anyway; log removed;
  log emitted twice; `fit_degraded` not carried (hard-coded False); `withheld_reason` None when withheld.
* Consumers: grep (READ) for direct reads of `p_t_posterior`, `p_t_estimate`, `m_t_posterior_mean`, `m_t_uncertainty`, `.p_t`, `.m_t`, `.regime_posterior` outside `fitting.py`; list them, do not rewrite them.
* Known limit that I state now, not as a result: the rule covers firings during `fit()` only (that is what `fit_degraded_` counts); firings during the later `estimate_states` pass are in the
  cumulative counter but not in the rule.

## 3. Item 3: D12 weakness notice
* Replace the current `UPDATE (ledger D12 ...)` comment in `tests/test_dynamics_statistics.py` by the owner's block, with the two bracketed owner fields left exactly as written.
  Verify every number against `D12_gap_table.txt` / the D12 mutation log first; a number that is wrong or imprecise is corrected and the correction is reported.
  The assertion and tolerances are not touched. Check: `git diff` shows comment lines only.

## 4. Item 4: dependencies
* `requirements.txt` UTF-16 -> UTF-8, pins unchanged. Proof: decode old and new, compare the sorted `name==version` lists: they must be identical (empty diff), saved as evidence.
* Add `scikit-learn==1.9.1` to `requirements.txt` and to `sbom/chronis-ml.cdx.json` (same structure as the existing components). The versions I ran under: Python 3.13.16 (this
  session) and, REPORTED by the brief, Python 3.12 by the reviewer; the repo's own target Python is looked up in the repo and reported as UNKNOWN if absent.
* N20 stays NOT RUN (no PyPI); the report records the reviewer's run as REPORTED with the numbers the brief gives.

## 5. Item 5: close-out
* Grep for tests that capture or assert on warning locations or filename/lineno of warnings, and `filterwarnings` / `-W` settings in the pytest config; run those tests; report.
* Full suite once at the end. Expected (not a criterion): the same three xfails as WP2, no failures. Added wall time of all new WP3 tests (from `--durations`) must be under 180 s.
* Ledger wording for A2 power, D12 and the health flag is updated as a separate text file with only claims that were RUN in WP2/WP3.

## 6. Stop conditions in force
The repo contradicts the brief for an item (stop that item, report, continue); a test that would need an existing assertion changed; a wish to tune a threshold; a result that looks too good.
