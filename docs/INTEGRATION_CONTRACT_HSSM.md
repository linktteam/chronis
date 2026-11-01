# HSSM integration contract (WP6 item 1)

For the owner's integrated pipeline. It states what `backbone.hssm` accepts, returns and raises, and what the product layer may read. Labels: **READ** = taken from the code, **RUN** = checked by a test or a run. Module paths under `backbone/hssm/` are stable; do not rename or move them. Companion: `docs/backbone/HSSM_RESULT_SCHEMA.md` (field-by-field).

## 1. Entry points

| Function | Module | Use |
|---|---|---|
| `fit_hssm(matrix, ...)` | `backbone.hssm.fitting` | The canonical fit; returns an `HSSMResult`; never raises for withheld insights. |
| `fit_hssm_gated_result(X, ...)` | `backbone.hssm.gating` | Same fit, legacy argument names, returns the `HSSMResult`. |
| `fit_hssm_gated(X, ...)` | `backbone.hssm.gating` | Legacy `(model, run_log)`; raises `InsightsWithheldError` when withheld; baseline model only. |
| `fit_with_random_restarts`, `fit_dual_with_random_restarts`, `select_k_by_bic` | `backbone.hssm.fitting` | Building blocks; the same input checks apply. |

## 2. Inputs (`fit_hssm`)

* `matrix`: `(T, F)` float array or DataFrame, one row per day (or per session), **z-scored per person by the caller** (the engine does not standardise; its initial variances come from the data and its variance floor is absolute). NaN is allowed: a row with at least 50% of its features observed is a *present session*; a row with every feature NaN is marginalised (contributes nothing, never imputed). `F >= 1`. (The legacy `fit_hssm_model` refuses NaN.)
* `timestamps` (optional): **calendar DAYS**, one value per row, finite, strictly increasing, fractional days allowed but no two rows on the same calendar day after rounding to whole days, span at most `MAX_TIMESTAMP_SPAN_DAYS` = 20,000 days. Seconds or milliseconds raise `ValueError` before any allocation (RUN: `tests/test_wp6_item1_timestamp_guard.py`). A unit mistake under the limit (hours over a month) is not detectable: the caller owns the unit. The baseline model uses the timestamps to put rows on a daily lattice (gaps become NaN rows; durations are in days); the dual-timescale model does **not** build a lattice: it works in row steps and uses the timestamps only to label `duration_unit` (READ).
* `candidate_ks` (default `(2, 3, 4)`), `n_initializations` (default 10, minimum 10 unless `allow_fast_test_fit=True`), `max_iter` (default 200), `max_duration` = Dmax (default 45, in lattice steps), `random_seed` (restart seeds are `random_seed + k*1000 + i`, and `+ i*31` for the dual model; READ), `model_type` (`"baseline"` or `"dual_timescale"`), `latent_dim` (dual only, default 2).
* `user_id`, `dataset_hash`: opaque strings copied into the result. The library neither computes nor verifies the hash (`HSSMFitDataset` computes one). `n_present_sessions`: optional override of the count used for the gate and for BIC.
* `config`: `ColdStartConfig(min_present_sessions=30)`.
* `shared_prior`, `shared_prior_n0`: optional, dual model only (WP5; schema section 3b).

## 3. Cold-start gate

Fewer than 30 present sessions: `ColdStartError`, nothing is fitted, no output (the correct silent state, not a low-confidence result). The timestamp check runs first, so a malformed argument raises `ValueError` even for a too-short record. The threshold is not changed by anything in WP5 or WP6.

## 4. Outputs

`HSSMResult` (schema sections 2 to 3b). The fields a pipeline needs: `fit_state` (`CONVERGED`, `MAX_ITER_REACHED`, `REJECTED_UPDATE`, `NUMERICAL_FAILURE`, `FAILED`, `MODEL_SELECTION_CONTESTED`), `K`, `BIC`, `AIC`, `p_t_estimate` / `p_t_posterior` (`(T, K)`; baseline: computed on the rows as given), `m_t_posterior_mean` / `m_t_uncertainty` (dual only; `None` for the baseline), `duration_parameters`, `fit_degraded`, `run_log` (one dict per restart), `convergence_metadata`, `eligible_session_count`, `calendar_start` / `calendar_end`, `used_shared_prior`.

## 5. Hold-back contract

The product layer reads regime and latent outputs **only** through `result.publishable_outputs()` (or the aliases `p_t`, `regime_posterior`, `m_t`), which return `None` when `result.insights_withheld`. The raw fields stay readable for diagnostics and must not reach a user. `result.withheld_reason` lists every cause. Rules in force (schema section 3): `fit_degraded`, `fit_state == NUMERICAL_FAILURE`, a fallback firing in the `estimate_states` pass, a fallback firing after the result was built. (WP6 item 3 adds the non-`CONVERGED` rule; the section "Changes in WP6 item 3" below is completed there.)

## 6. Exceptions a caller can see

| Exception | Raised by | When |
|---|---|---|
| `ValueError` | `fit_hssm`, `fit_with_random_restarts`, `fit_dual_with_random_restarts`, `GaussianHSMM` | `F <= 0`; fewer than 10 restarts without `allow_fast_test_fit`; invalid `timestamps` (not numeric, wrong length, NaN or infinite, not strictly increasing, span over the limit, two rows on one calendar day); a shared prior that does not match the request, is not aggregate-only, or a strength without a prior; empty or non-positive `candidate_ks`; NaN in `fit_hssm_model`; K, F or L not positive when a model is built. |
| `ColdStartError` | `fit_hssm`, gated functions | Fewer than `config.min_present_sessions` present sessions. |
| `FittingConvergenceError` | `fit_with_random_restarts`, `select_k_by_bic`, `fit_hssm` | Baseline: none of the restarts of any candidate K ended `CONVERGED`. Dual: no restart finished (a shared-prior fit that did not finish). This is what the owner's run hit on 73 days. |
| `InsightsWithheldError` | `fit_hssm_gated` only | The fit finished but its insights are withheld; carries `reason` and the full `result`. |
| `InternalStateError` | engine | An internal invariant was violated (a log-likelihood that is `None` after a pass; canonicalisation postcondition). A bug, not an input problem. |
| `NotFittedError` | models | A fitted-only attribute is read before `fit()`. |
| `MemoryError` | numpy | Not covered by any guard other than the timestamp span; the lattice is `T_grid x K x Dmax` float64 per array. |

Warnings: a `UserWarning` is emitted when the duration optimiser falls back; a `print` line when the baseline monotonicity guard fires (READ). Internal log: one WARNING per withheld result on logger `backbone.hssm.fitting`.

## 7. Known limitations that matter to an integrator (READ, not hidden)

1. The two models read time differently (section 2). For the dual model `duration_unit_source` says `"calendar_days"` when timestamps are passed although its durations are in row steps.
2. For the baseline model with timestamps, `fit_hssm` computes `p_t` on the rows as given, not on the daily lattice the model was fitted on.
3. `MODEL_SELECTION_CONTESTED` (two BICs within 2.0) is a `fit_state` of its own and is not a convergence failure.
4. The reported `m_t_uncertainty` omits parameter uncertainty (N17: coverage of +-2 sd 0.68 against 0.95 nominal in the WP5 study); it must not be presented as a calibrated interval.
5. Cold-start personal fits are weak on short records; the optional shared prior (WP5) helps on synthetic populations only.
