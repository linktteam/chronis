# Pre-declared criteria — ledger items A2, A6, A7, N21, D12, N20

Written and committed BEFORE any of the new tests in this work package was run
(ledger §2.2 rule 12). `git log` shows this file's commit; every test log in the report
carries its own timestamp. Criteria below are not edited after results are seen. If a
criterion turns out to be wrong, that is reported as a finding and a NEW dated section is
added; the original text stays.

Base: branch `ledger/wp2`, parent commit = `hssm_integration.patch` applied to `bf031b0`
(tree identical to `d48af35`).

## 0. What I had already seen before writing these criteria (so the thresholds are not presented as blind)

* RUN (earlier session, on this code): old test 39 gives latent R² 0.101 (global affine) and old
  test 41's statistic gives per-dimension 0.875 / 0.8125 on the ported tree (0.859 / 0.891 on the
  baseline); fitted duration sigma on seed 202 is 0.03 and 2.21.
* REPORTED (ledger §4.5, reviewer): per-regime-aligned R² 0.974 (seed 202) and 0.941 (seed 404)
  on the ported tree; coverage of ±2·m_t_std in fitted coordinates 0.883 / 0.967 (seed 202) and
  0.828 / 0.938 (seed 404).
* I had NOT written or run any per-regime alignment code, any calibration code, any of the new
  tests, or any of the A6/A7/N21/D12 code before this commit.
* The thresholds in §1 were chosen from first principles stated next to each, not by trying
  values. Where I name a number I had read in the ledger, I say so.

## 1. A2 — replace tests 39 and 41

### 1.1 Data and fits (identical to the old tests, so the comparison is like for like)
* Generator: `data.synthetic_dual.generate_synthetic_dual_timescale_data(n_regimes=2,
  n_features=4, latent_dim=2, n_segments=4, mean_dwell=15.0, seed=s)`.
* Cell A: s = 202 (old test 39's data): `DualTimescaleHSSM(2, 4, 2, max_duration=25, seed=202)`, `fit(X, n_iter=25)`.
* Cell B: s = 404 (old test 41's data): same constructor with `seed=404`, `fit(X, n_iter=20)`.
* Outputs read from `estimate_states(X)`: `m_t` (smoothed mean), `m_t_std`.

### 1.2 New test "per-regime-aligned latent recovery" (replaces test 39's assumption)
* Alignment: for each TRUE regime r (generator's `true_p_t`), ordinary least squares WITH
  intercept from fitted `m_t` (L=2) to the true latent `true_m_t` (L=2), fitted IN SAMPLE
  (documented: in-sample alignment flatters R²; chance level for p regressors is about p/(n_r-1)).
  Each true regime must have at least 3·(L+1) = 9 time steps; otherwise the test ERRORS (it
  never skips or silently drops a regime).
* Statistic: pooled R² = 1 − Σ_r Σ_{t in r} ||true_m_t − Ŵ_rᵀ[m̂_t;1]||² / Σ_t ||true_m_t − mean_t(true_m)||²
  (global mean over all t).
* Criterion: **R² ≥ 0.80 in BOTH cells.**
  Reason for 0.80: the old bar was 0.50 for a map the model class cannot satisfy; with the
  model-class-correct map the chance level is about 0.1, and 0.80 leaves at most 20% of true
  latent variance unexplained. I chose it knowing the reviewer's REPORTED values (0.941–0.974
  ported, 0.825–0.948 baseline); it sits below them on purpose, not above.
* Reported but NOT criteria: per-regime R², per-dimension R², global-affine R² (to show the old
  statistic next to the new one).

### 1.3 New test "calibration of m_t_std" (replaces test 41; actually reads `m_t_std`)
* Alignment into FITTED coordinates (so `m_t_std` is read in its own coordinates, no covariance
  propagation needed): for each true regime r, OLS with intercept regressing fitted `m_t` on
  `true_m_t`, in sample. Error e_t = m̂_t − prediction. Standardised z_{t,d} = e_{t,d} / m_t_std_{t,d}.
* Statistic: coverage_d = mean_t( |z_{t,d}| ≤ 2 ), per cell and per dimension d (4 cells in total).
* Criterion per (cell, dimension): coverage ≥ p0 − 3·se(T), where p0 = P(|N(0,1)| ≤ 2) = 0.954500
  and se(T) = sqrt(p0 (1 − p0) / T) with T = len(X) (T = 60 and 64, so the lower bounds are
  about 0.874 and 0.876). No upper bound: with n = 60 a coverage of 1.0 has probability
  p0^60 ≈ 0.06 under perfect calibration, so over-wide intervals cannot be detected at this
  sample size (a stated limitation of this test).
* Reason for 3·se: about 0.13% one-sided false-failure rate per cell if errors were independent;
  autocorrelation makes the true false-failure rate higher, and the in-sample alignment makes
  coverage look better than it is. The criterion is therefore not generous in one direction and
  strict in the other by design; it is a coarse screen, not a calibration certificate.
* Also required: `m_t_std` finite and > 0 everywhere.
* Parametrised per (cell, dimension). If a cell fails, that is **finding N17**. The criterion is
  NOT loosened. Failing cells are then marked `xfail(strict=True)` with the observed numbers
  (strict so a later engine fix flips them to a visible failure). The xfail marks are added in a
  separate commit after the first run; owner may prefer to leave such cells red (decision listed in the report).
  From the ledger I expect cell B, dimension 0 (REPORTED 0.828) to fail. The criterion was set
  by the formula above, not by that number.

### 1.4 Old tests 39 and 41
* Kept, assertions untouched, marked `@pytest.mark.xfail(strict=False, reason=...)` with the numbers
  (R² 0.101 vs >0.50; coverage 0.8125 vs >0.85 on the ported tree; baseline passes) and a pointer to the replacements.
  `strict=False` on purpose: if another library version makes one of them pass, it shows as XPASS
  (visible in the summary) instead of failing the suite.

### 1.5 Helper (metric) tests — positive and negative controls on the metrics themselves, no engine involved
* H1: synthetic fitted coordinates = a DIFFERENT random rotation+offset of the true latent in
  each regime, plus noise sd 0.05: per-regime-aligned R² ≥ 0.95 AND global-affine R² < 0.50
  (shows the old assumption is wrong for this model class while the new check is right).
* H2: fitted coordinates = time-permuted copy of the true latent (fixed seed): per-regime-aligned R² < 0.30.
* H3: iid Gaussian errors with the stated std (n = 20,000 time points, 1 dimension, one regime):
  coverage within 0.9545 ± 0.01; with std × 0.25 the coverage < 0.50.

### 1.6 Negative controls with deliberately broken ENGINES (run through the real fit/estimate_states)
Engines (defined once, in the test module):
* B1 "smoother aggregated with the FILTERED posterior": `_switching_smoother` is called with
  `regime_posterior_smoothed` replaced by `fwd.regime_post_filtered` (this is the repo's mutation M3).
* B2 "A_fast driven toward 0": `_m_step_continuous` runs, then `A_fast ← 0.1 · A_fast`.
* Post-hoc output breakers (use the real fitted engine, corrupt one output): G (m_t time-permuted,
  rng seed 0) and S (m_t_std × 0.25).

Required outcomes, declared now:
* G must violate the R² criterion in both cells. (REQUIRED)
* S must violate the calibration criterion in every (cell, dimension). (REQUIRED)
* B1 and B2 are run against BOTH new tests in both cells and the 2×2×2 result is reported
  as it comes out. Whether they violate the criteria is UNKNOWN to me now. If a broken engine does
  NOT violate a criterion, the report says so plainly: that test cannot detect that fault. I do not
  alter a criterion or choose a different broken engine to make this look better. (The ledger asks
  that B1 and B2 make the tests fail; if they do not, that is a finding about the tests' power.)

## 2. A7 — flag when the Cholesky −1e6 fallback fires
* Implementation: an integer counter incremented inside `_kalman_update_regime`'s
  `except np.linalg.LinAlgError` branch only. Return values unchanged.
* Pass criteria:
  1. A/B bit-identity (`np.array_equal`) of `_kalman_update_regime` outputs (valid inputs, NaN
     patterns, and the failing input) between the pre-change snapshot and the changed code, ≥ 160 valid cases + the failing case.
  2. A/B bit-identity of a 15-iteration `fit()` (parameters and log-likelihood history) on a planted
     K=3 dataset, pre-change snapshot vs changed code.
  3. Forced firing (regime with negative R): counter increases by exactly the number of failing calls;
     a clean model: counter stays 0; after `fit()` with a forced firing inside the loop
     `fit_degraded_` is True and `fit_cholesky_fallbacks_ > 0`; clean fit: False and 0.
  4. The forcing tests FAIL on the pre-change code (attribute missing) and PASS after.
  5. Counts of firings, reported (expected 0, not a pass/fail criterion): oracle runs = reference battery (10) +
     K=3/K=4 lossless configs + K=3 statistics configs; recovery runs = G8 truth-start (T=300 seeds 1,2; T=900 seed 1) and
     G9 real-init 40-iteration fits, seeds 0–3. A non-zero count is reported as a finding, not tuned away.
  6. Mutants against the new tests, all must be caught: counter not incremented; counter incremented on every call;
     `fit_degraded_` hard-coded False; `fit_degraded_` hard-coded True.

## 3. A6 — `estimate_states`
* Consumers of `m_t` / `m_t_std` / `estimate_states` outside `dual_model.py`: grep over `backbone/`, `data/`, `docs/`, `tests/`;
  the full command and every hit go into the report.
* Identical-prefix test (T = 60 fitted model from §1.1 cell A; prefixes n = 20 and n = 35):
  `regime_post` and `p_t` of `estimate_states(X[:n])` equal the first n rows of `estimate_states(X)`
  (`np.allclose`, atol = 1e-12, rtol = 0), while `smoothed_m[:n]` differs between them by more than 1e-6 in max-abs.
* Negative controls, must be caught: engine returning the SMOOTHED regime posterior as `regime_post`
  (fails the prefix equality); engine whose `smoothed_m` is the FILTERED-weights/no-future version is not
  constructible without re-deriving the smoother, so the control for "smoothed_m uses future data" is an engine that
  returns the filtered per-regime mean mixture `sum_k filtered_post[t,k] * m_filt[t,k]` as `smoothed_m` (fails the "differs" check).
* Documentation: `estimate_states` docstring states that `smoothed_m` and `m_t_std` use the whole series while
  `regime_post` / `p_t` use only observations up to t.

## 4. N21 — exceptions caught in `fit()`
* Implementation: every `(LinAlgError, ValueError)` caught in the E-step or M-step appends a record
  `{iteration, stage, exception_type, message, location}` to `fit_exceptions_`; the list is reset at the start of each `fit()`.
  Exception types outside that tuple still propagate.
* Pass criteria: injected `LinAlgError` in the E-step and in the M-step, and an injected plain `ValueError`
  (imitating a shape bug) are each recorded exactly once with the right stage, iteration, type and message, and
  `fit_state_ == NUMERICAL_FAILURE`; a clean fit has `fit_exceptions_ == []`; an injected `RuntimeError` propagates and is not recorded;
  a second `fit()` call resets the list. Tests FAIL on the pre-change code and PASS after.
* Mutants, must be caught: record not appended in the E-step branch; record not appended in the M-step branch; list not reset at fit start.
* NOT done (out of scope, reported as an owner decision): propagation into `HSSMResult` (`fitting.py`, P0-10); the non-finite
  log-likelihood path (not an exception) keeps ending in NUMERICAL_FAILURE without a recorded reason.

## 5. D12 — leakage-aware tolerance for the K=3 concentrated statistics test
* Criterion: relative gap ≤ max(1e-6, 2 × leakage), where leakage = 1 − (largest posterior mass on one complete regime sequence)
  computed by the exact oracle (summing over duration paths). The config list is unchanged; the failing config is not replaced.
  The test text states that this is a new criterion chosen after seeing the result (owner ruling D12, option (a), ASSUMED APPROVED, to be confirmed in writing).
* Pass criteria: all six pre-declared K=3 concentrated configs satisfy it; the old-pairing negative control (> 0.1) still passes;
  the dynamics-statistics mutants N2–N6 still make the K=3 concentrated tests fail, and on the seed-1 config specifically their gap
  exceeds that config's tolerance.

## 6. N20 — pinned library versions
* Pinned (decoded from the UTF-16 `requirements.txt`): numpy 2.2.6, scipy 1.15.3, pandas 2.3.3 (and pytest 9.1.1).
* PASS requires: the interpreter that runs pytest prints exactly those versions, all 238 + new tests collected and run,
  failures listed. If the pinned versions cannot be installed in this environment the item is reported NOT RUN with the exact error,
  and the suite is run in the available environment (versions printed) as a clearly labelled substitute, not as N20.

## 7. Stop conditions in force
Repo contradicts the ledger in a way that changes the plan; any test that would need its assertion changed (other than the owner-ruled D12
criterion and the xfail decorators above); a result that looks too good; an oracle comparison that fails and three debugging passes do not find the cause.
