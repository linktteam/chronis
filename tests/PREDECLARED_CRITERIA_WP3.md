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

## 7. Amendments (dated; the text above is unchanged)

### 2026-10-22, after the FIRST run of the N18 tests (`wp3_evidence/N18_first_run.log`)
* The first run gave: C2 passes; **C1 fails** (real minus single Gaussian: mean -0.7087 nats per held-out step, paired se 0.4920, mean - 3 se = -2.1846; 3 of 8 seeds favour the real engine);
  **C3 fails narrowly** (real minus B2: mean +2.2200, se 0.7444, mean - 3 se = -0.0131; 8 of 8 seeds favour the real engine, 2.98 se); both breaker controls pass. The criteria are NOT changed. These are findings about the engine
  (reported in the WP3 report), the two tests stay failing, and no xfail is added (brief section 0 rule 5).
* A sanity assertion in the continuation test required T_train >= 55 (my informal "about 55 to 64" taken from the two A2 cells). Seeds 909 and 1010 give T_train = 50, which made that test fail. The bounds are corrected to the generator's logical ranges
  (training 32..120, held-out 16..60). This is a bug in a sanity assertion, not an engine criterion; no engine criterion was touched.
* Post-hoc diagnostics (NOT pass criteria, labelled as such everywhere): per-step diagnosis of the losses (`n18_diagnose.py`) and a sensitivity study (`n18_sensitivity.py`: more seeds; training length; split inside a segment).
  All variants are reported whatever they show; none replaces the official criteria above.

### 2026-10-22, B1 against the existing oracle-based tests, and a corrected design for the fallback check (written BEFORE the new check was run)
* RUN (`wp3_evidence/B1_oracle_unmutated.log`, `B1_oracle_B1.log`, `B1_oracle_verdict.txt`): the ten existing oracle/smoother test files give `57 passed` unmutated and `1 failed, 56 passed` with B1 applied. The one test that fails is
  `tests/test_switching_smoother.py::test_smoother_uses_smoothed_not_filtered_posterior`. That test is structural (it compares `smoothed_m` with weighted mixes of its own per-regime means); it is NOT an exact-oracle comparison.
  **No exact-oracle comparison fails under B1**, including `test_switching_smoother_matches_oracle_tiny_case` and the 13 lossless-collapse configs. My PREDICTION in section 1.5 ("caught, probably by the lossless oracle tests") was right about "caught" only in the structural sense and wrong about where.
* RUN (`wp3_evidence/B1_design_scan.txt`, part a): in all 13 lossless-collapse configs the per-regime smoothed means differ by at most 5.6e-16 and the filtered and smoothed regime posteriors differ by at most 1.8e-15, so the aggregation weights cannot matter
  there. The fallback design written in section 1.5 (lossless-collapse configs) cannot see B1 by construction. It is REPLACED by the design below; section 1.5 is otherwise left as written.
* New check, outside N18 (it tests the smoother, not prediction). Configs: K=2, F=1, L=1, Dmax=2, T=5, `make_tiny_model(seed=s)`, `make_tiny_data(seed=s+100)`; the first six seeds in ascending order from 0..199 that satisfy the SELECTION conditions
  D1 = max |filtered - smoothed regime posterior| >= 0.05 and D2 = max_t |smoothed_m_k[t,0] - smoothed_m_k[t,1]| >= 0.1 (the scan printed these values only, no oracle gap and no B1 outcome): **seeds 0, 1, 2, 3, 4, 5**.
  Claim tested on each config: `max_t |smoothed_m - oracle.smoothed_m| < max_t |filtered_mix - oracle.smoothed_m|`, where `filtered_mix[t] = sum_k filtered_post[t,k] * smoothed_m_k[t,k]` uses the engine's own per-regime means with the FILTERED weights.
  Reason: the exact smoothed mean is a mixture of per-regime means weighted by the SMOOTHED regime posterior; the engine's per-regime means approximate the oracle's, so the smoothed weights should land closer to the oracle than the filtered ones.
  Margin: strict inequality; the ratios are reported. Control (inside the test file): an engine of class `EngineFilteredSmoother` (B1) returns `smoothed_m` equal to the filtered mix, so the same predicate must fail for it (ratio exactly 1). Mutation check (script): the B1 class-level patch plugin must make the new test fail.
  If the real engine fails the claim on some config that is a finding about the engine or the claim; the claim is not changed.

### 2026-10-22, DESIGN 2 of the B1-against-oracle check (written after design 1 failed on the real engine; ONE revision only)
* Design 1 result (RUN, `wp3_evidence/B1_new_check_first_run.log`): the claim failed on the real engine on 2 of the 6 pre-declared configs (seed 0: ratio 1.002; seed 5: ratio 1.242); selection conditions held on all six; the B1 control behaved as expected (ratio 1).
  Reading: the engine's per-regime mean error can exceed the effect of the aggregation weights, so comparing `smoothed_m` with the oracle cannot separate B1 from the smoother's own approximation error at these sizes. Design 1 is kept as a diagnostic outside the suite
  (`tests/diagnostics/design1_smoother_closer_to_oracle_than_filtered_mix.py`) with its result; it is not edited.
* Design 2 isolates the weights. Same six configs (no re-selection). `mix_oracle[t] = sum_k oracle.smoothed_regime_posterior[t,k] * smoothed_m_k[t,k]` (the engine's own per-regime means with the EXACT smoothed weights);
  `g_engine = max_t |smoothed_m - mix_oracle|`, `g_filtered = max_t |filtered_mix - mix_oracle|`. **Claim: `g_engine <= 0.25 * g_filtered`.**
  Reason: the engine's weights are its own smoothed regime posterior, which should match the oracle's up to the discrete approximation error, while the filtered weights differ from the oracle's smoothed weights by D1 >= 0.05 on these configs; 0.25 means "at least four times closer";
  B1 gives ratio exactly 1. The number 0.25 was chosen before any design-2 number was seen. Control: the B1 engine class must violate the claim. Mutation check: the B1 class-level patch plugin must make the main test fail on every config.
* STOP RULE: if design 2 also fails on the real engine, that is reported as a finding and no third design is tried; the owner decides.
