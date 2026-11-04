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

The product layer reads regime and latent outputs **only** through `result.publishable_outputs()` (or the aliases `p_t`, `regime_posterior`, `m_t`), which return `None` when `result.insights_withheld`. The raw fields stay readable for diagnostics and must not reach a user. `result.withheld_reason` lists every cause. Rules in force (schema section 3): `fit_degraded`, `fit_state == NUMERICAL_FAILURE`, a fallback firing in the `estimate_states` pass, a fallback firing after the result was built, and, since WP6 item 3 (P0-07), two more:

* **`fit_state_not_converged`**: the fit state is `MAX_ITER_REACHED`, `REJECTED_UPDATE`, `FAILED` or `RUNNING`. A fit that did not converge is never publishable. `MODEL_SELECTION_CONTESTED` is a model-selection state of a fit that converged: it is judged by the fit's own state in `convergence_metadata["fit_state"]` (a result without that key counts as converged), so a contested result stays publishable. The dual-timescale model at the default 200 iterations did not converge in any of the 20 diagnosis users (`tests/WP6_CONVERGENCE_DIAGNOSIS.md`), so its results are withheld by this rule unless its stopping rule certifies them (see "Changes in WP6 item 3").
* **`collapsed_regime`**: the selected baseline fit has a regime with fewer than `COLLAPSE_MIN_EFFECTIVE_DAYS` = 4 effective days (`fit_collapsed`, details in `collapse_report`). A variance at the variance floor is reported (`n_variance_at_floor`) but does not by itself withhold: in the diagnosis such fits label no worse than healthy ones.

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

## 8. Changes in WP6 item 3 (convergence and M-step health)

* **Baseline guard repair.** The EM monotonicity guard used to end a fit as `REJECTED_UPDATE` whenever the log-likelihood fell by more than 1e-4. In the diagnosis the cause was the duration M-step in 138 of 138 events (it could lower its own expected log-likelihood). Now, at the first such event the previous M-step is redone with a safeguarded duration update (it cannot lower that block's expected log-likelihood), the fit goes on, and the update is used for the rest of that fit. The event is recorded: `model.guard_repairs_` / `result.guard_repairs` (count) / `convergence_metadata["guard_repairs"]` (iteration and drop). If the repaired update still lowers the log-likelihood the fit ends `REJECTED_UPDATE` as before. Fits that never meet the guard condition are bit-identical to the WP5 code. The guard tolerance is unchanged (1e-4 baseline, 1e-3 dual).
* **Collapse.** `collapse_report` (`result.collapse_report`, `convergence_metadata["collapse"]`): effective days per regime (in the model's canonical regime order), the minimum, the number of variances at the floor, the threshold, `collapsed`. Restart selection (`fit_with_random_restarts`) takes the best converged restart that is not collapsed; only if every converged restart is collapsed does it return the best one, flagged and withheld. `select_k_by_bic` demotes a collapsed candidate K like a degraded one. This changes the selected fit only where the old best-log-likelihood restart (or BIC-best K) was collapsed; the diagnosis found no change in label accuracy from it (paired -0.0015, se 0.0087), so it is a safety rule, not an accuracy rule.
* **Multi-start summary (ledger N1).** `result.restart_summary` / `convergence_metadata["restart_summary"]` / `model.restart_summary_`: restarts run, converged, rejected, at the iteration limit, collapsed among the converged; how many converged restarts lie within 1.0 nat of the selected one; the adjusted Rand index of the other converged non-collapsed restarts' labellings against the selected one (list, median, minimum), how many are the same solution (index at least 0.9), and the log-likelihood gap to the best restart that is a different solution (`None` when there is none, and all comparison fields are empty when the selected fit is collapsed). Descriptive: nothing is withheld because of them. The run log carries `collapsed`, `min_effective_days`, `n_variance_at_floor`, `guard_repairs` for every restart.
* **Dual-timescale restart selection** prefers a converged restart to a non-converged one with a higher log-likelihood (the latter would be withheld).
* **Dual stopping rule at the iteration limit (C2).** The existing rule (`|ll - previous| < tol`) is unchanged and ends the fit `CONVERGED` with `convergence_criterion_ = "abs_tol"`. If the iteration limit is reached instead, the fit is certified `CONVERGED` (`convergence_criterion_ = "window_gain_at_cap"`) only if the log-likelihood gained less than `DUAL_CAP_CERT_GAIN_PER_ROW` = 2.0e-4 nats per row over the last `DUAL_CAP_WINDOW` = 10 iterations (`window_gain_at_cap_` holds the gain, also when it does not certify); otherwise it stays `MAX_ITER_REACHED` and the result is withheld (P0-07). `n_iter_` is now the number of iterations run (it used to stay 0 at the limit). In the diagnosis 10% of dual fits at the default 200 iterations met the certification rule, so **about 90% of dual fits at the default `max_iter` are withheld**; a larger `max_iter` or an accelerated EM would raise that share (not done in WP6). Evidence: `tests/WP6_CONVERGENCE_DIAGNOSIS.md`, addendum Q4b: a certified fit kept its regime labels (at most 0.62% of rows changed) and its `m_t` within 0.1 sd over 200 further iterations.
* **Dual duration box (D1).** The duration update could run to a flat, meaningless optimum (sigma at the optimiser wall 10, mu from -165 to 146, in 10 of 40 regimes of the diagnosis fits). `DualTimescaleHSSM.fit` now runs the fit as before and, only if it ENDS with a regime duration outside the box (mu in [-5, ln(10 Dmax)], sigma at most 3, finite), runs it once more from the same starting point with the box active (an update that leaves the box is redone from the previous iterate with a penalty outside the box). A fit that ends inside the box is never touched (bit-identical). The first fit is described in `model.duration_box_refit_` and `convergence_metadata["duration_box"]["first_fit"]`; `convergence_metadata["duration_box"]` also carries `refit`, `n_applications` and `applied` (iteration, regime, from and to values of every redo of the final fit); `result.duration_at_bound[k]` is True when regime k's final duration is within 1e-3 of a wall of the box (a regime whose best unconstrained pmf needs sigma above 3 loses a little log-likelihood; that is what the flag reports). A fit that is refitted costs about twice the time. Nothing is withheld because of the box. Switch: `DualTimescaleHSSM.DURATION_BOX_REFIT` (class attribute, default True).
