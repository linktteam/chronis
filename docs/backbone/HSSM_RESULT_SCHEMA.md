# HSSM Result Schema & Specification Contract

**Document ID**: `CHRONIS-HSSM-SCHEMA-001`  
**Version**: `1.0.0`  
**Acceptance State**: `SEALED_YELLOW` (FULL_MODEL_SCAFFOLD_READY)

---

## 1. Overview

The `HSSMResult` dataclass in `backbone.hssm.fitting` is the single canonical return type for all HSSM fitting procedures. It enforces complete metadata tracking across 40+ fields required by Section Q of the Chronis Behavioral Backbone Directive.

---

## 2. Field Definitions

| Field | Type | Description | Baseline Model Value | Full Dual Model Value |
|---|---|---|---|---|
| `model` | `Any` | Fitted model instance | `GaussianHSMM` | `DualTimescaleHSSM` |
| `user_id` | `str` | User identifier | String | String |
| `model_identity` | `str` | Formal class identifier | `"ExplicitDurationSwitchingBaselineV1"` | `"ChronisDualTimescaleHSMMV1"` |
| `model_family` | `str` | Algorithmic family | `"slow_regime_only"` / `"GaussianHSMM"` | `"fast_slow_dual_timescale"` |
| `model_capability` | `dict` | Capability dictionary | `{"fast_state_supported": False, "is_baseline_model": True}` | `{"fast_state_supported": True, ...}` |
| `is_baseline_model` | `bool` | Flag indicating baseline | `True` | `False` |
| `fast_state_supported`| `bool` | Whether $m_t$ is supported | `False` | `True` |
| `p_t_posterior` | `ndarray (T, K)` | Posterior regime probabilities | Sums to 1.0 across rows | Sums to 1.0 across rows |
| `p_t_estimate` | `ndarray (T,)` | MAP regime sequence | Array of ints $\{0..K-1\}$ | Array of ints $\{0..K-1\}$ |
| `m_t_posterior_mean`| `ndarray (T, L)`| Continuous latent mean | `None` (Never faked) | Array of floats |
| `m_t_uncertainty` | `ndarray (T, L)`| Latent state std error | `None` (Never faked) | Array of floats $\ge 0$ |
| `duration_parameters`| `dict` | $\mu, \sigma$ arrays | Dict of lists | Dict of lists |
| `duration_model_identity`| `str` | Duration distribution | `"explicit_truncated_gaussian"` | `"explicit_truncated_lognormal"` |
| `Dmax` | `int` | Maximum duration truncation | Integer ($> 0$) | Integer ($> 0$) |
| `duration_unit_source`| `str` | Source of duration time | `"session_index"` or `"calendar_days"` | `"calendar_days"` |
| `calendar_start` | `float` | Start timestamp | Float or `None` | Float or `None` |
| `calendar_end` | `float` | End timestamp | Float or `None` | Float or `None` |
| `observation_density`| `float` | Fraction of present sessions | Float in $[0, 1]$ | Float in $[0, 1]$ |
| `model_version` | `str` | Code version | `"1.0.0"` | `"1.0.0"` |
| `feature_schema_version`| `str` | Feature schema version | `"1.0.0"` | `"1.0.0"` |
| `alignment_version` | `str` | Label switching version | `"1.0.0"` | `"1.0.0"` |
| `eligible_session_count`| `int` | Count of eligible rows | Integer $\ge 30$ | Integer $\ge 30$ |
| `min_present_sessions`| `int` | Minimum required sessions | `30` | `30` |
| `fit_dataset_hash` | `str` | Deterministic SHA-256 | 64-char hex string | 64-char hex string |
| `K` | `int` | Selected regime count | Integer in $\{2, 3, 4\}$ | Integer in $\{2, 3, 4\}$ |
| `BIC` | `float` | Bayesian Information Criterion| Float | Float |
| `AIC` | `float` | Akaike Information Criterion | Float | Float |
| `heldout_predictive_log_likelihood` | `float` | Chronological test LL | Float or `None` | Float or `None` |
| `fit_state` | `FitState` | Terminal state enum | `FitState.CONVERGED` etc. | `FitState.CONVERGED` etc. |
| `optimizer_success` | `bool` | All optimizers converged | Boolean | Boolean |
| `fallback_used` | `bool` | Truncation fallback flag | Boolean | Boolean |
| `run_log` | `list[dict]` | Restart audit trail | List of dicts | List of dicts |
| `warnings` | `list[str]` | Fitting warnings | List of strings | List of strings |
| `limitations` | `list[str]` | Model boundary statements | List of strings | List of strings |
| `neural_residual_enabled`| `bool` | Neural residual flag | `False` | `False` (by default) |
| `residual_mode` | `str` | Mode of neural residual | `"DISABLED"` | `"DISABLED"` / `"SHADOW"` |
| `residual_weight` | `float` | Additive blending factor | `0.0` | `0.0` |
| `fit_degraded` | `bool` | The -1e6 Cholesky fallback fired during `fit()` (ledger A7) | `False` (no such flag) | `True` iff it fired during the fit |
| `fit_cholesky_fallbacks` | `int` | How many times it fired during that `fit()` | `0` | Integer $\ge 0$ |
| `fit_exceptions` | `list[dict]` | One dict per exception caught inside `fit()` (type, message, stage, location) | `[]` | List of dicts |
| `states_pass_fallbacks` | `int` | Firings inside the `estimate_states` pass of `fit_hssm`, which runs after `fit()` and before the result is built (WP4) | `0` | Integer $\ge 0$ |
| `fallback_count_at_build` | `int` or `None` | The model's cumulative fallback count when the result was built; `None` if not recorded (WP4) | `None` | Integer $\ge 0$ |
| `used_shared_prior` | `bool` | The fit started at a shared prior and was regularised toward it (WP5; section 3b) | `False` | `True` iff `fit_hssm(..., shared_prior=<SharedPrior>)` was used |
| `shared_prior_n0` | `float` or `None` | Strength of that prior in pseudo-steps per regime (WP5; section 3b) | `None` | Float $\ge 0$ when a prior was used (default `SHARED_PRIOR_DEFAULT_N0` = 5.0), else `None` |

---

## 3. Hold-back of insights (ledger A7 / decision D3; closed in WP4)

Computed read-only properties and one method, derived from the CURRENT field values and the CURRENT model count (nothing is stored, so nothing goes stale):

| Name | Kind | Meaning |
|---|---|---|
| `late_fallbacks` | `int` property | Firings of the fallback on `result.model` after the result was built: the model's current `cholesky_fallback_count_` minus `fallback_count_at_build`. `0` if the count was not recorded or the model has no counter (baseline). |
| `withheld_reason` | `str` or `None` property | `None` when the insights may be used; otherwise every cause that applies, joined by `"; "`. |
| `insights_withheld` | `bool` property | `withheld_reason is not None`. |
| `publishable_outputs()` | method | `None` when withheld, otherwise a dict with the four output arrays (`p_t_estimate`, `p_t_posterior`, `m_t_posterior_mean`, `m_t_uncertainty`). The only supported way for the product layer to read them. |

RULE: the insights are withheld when `fit_degraded` is `True`, or `fit_state == FitState.NUMERICAL_FAILURE`, or `states_pass_fallbacks > 0`, or `late_fallbacks > 0`.

Internal log: one WARNING per result on logger `backbone.hssm.fitting` (at build time, or the first time a result that became withheld later is read through `publishable_outputs()` or an alias). Nothing user-facing is added to the result.

Entry points:
- `fit_hssm(...)` returns the result; it never raises for withheld insights.
- `fit_hssm_gated_result(...)` (`backbone.hssm.gating`) is the gated fit returning the full result, with `model_type="baseline"` (default) or `"dual_timescale"`; it never raises for withheld insights.
- `fit_hssm_gated(...)` keeps the legacy return `(model, run_log)` and raises `InsightsWithheldError` (attributes `reason`, `result`) when the insights are withheld. It fits the baseline model only.
- `fit_dual_with_random_restarts(...)` never returns a degraded restart while a healthy one exists. If every usable restart is degraded it returns the best one by log-likelihood, and the result built from it is withheld.

Not covered: the choice of K by BIC across candidate Ks does not look at degradation.

---

## 3b. Shared prior for the cold start (WP5, N18-v2; optional, off by default)

`fit_hssm(..., model_type="dual_timescale", shared_prior=None, shared_prior_n0=None)` and `fit_dual_with_random_restarts(..., shared_prior=None, shared_prior_n0=None)`. With the default `None` nothing changes (checked bit for bit against `16c3431`, see the WP5 report).

With a `backbone.hssm.shrinkage.SharedPrior` (aggregate parameters of a population of users; no data):
- the fit starts at the shared parameters (so it stays in the shared latent frame) and its M-step is regularised toward them with strength `shared_prior_n0` pseudo-steps per regime (default `backbone.hssm.config.SHARED_PRIOR_DEFAULT_N0` = 5.0, frozen on development seeds, `tests/N0_FROZEN_WP5.md`);
- exactly one fit is run (a deterministic starting point makes restarts identical); the minimum of 10 random initialisations does not apply; `n_initializations` is ignored;
- K is the prior's K: it must be in `candidate_ks`; no BIC search over K is made; `bic_by_k` has that one entry; BIC and AIC use the unpenalised log-likelihood at the shrunk parameters;
- the result records `used_shared_prior=True`, `shared_prior_n0`, `model_class="ShrinkageDualHSSM"`, and `convergence_metadata["shared_prior"] = {n0, n_users, k_fixed_by_prior, n_fits_run, prior_uncertainty_in_m_t_std: False, bic_note}`; the default result's metadata has no `shared_prior` key;
- the prior's uncertainty is NOT included in `m_t_uncertainty` (the engine's `estimate_states` is unchanged; N17 is not addressed);
- the EM monotonicity guard compares the unpenalised log-likelihood, so a regularised fit can stop early with `fit_state == REJECTED_UPDATE` (about a third of the fits in the WP5 study); the hold-back rule of section 3 is unchanged;
- the cold-start gate (30 present sessions) is NOT changed by a prior; a caller may pass a `config` with a lower minimum, which is a product decision;
- fail closed: a prior whose (K, F, L) differ from the request, a prior that is not aggregate-only (attribute whitelist), a negative or non-finite strength, a strength without a prior, a prior with `model_type="baseline"`, or a fit that did not finish raises (`ValueError` / `FittingConvergenceError`); there is no silent fall-back to a prior-free fit.

---

## 4. Backward Compatibility Aliases

The following read properties maintain backward compatibility with legacy Sprint 3/4 consumers. The first three return `None` when the insights are withheld (section 3); the raw fields stay readable for internal diagnostics:
- `result.p_t` $\rightarrow$ returns `result.p_t_estimate` (`None` if withheld)
- `result.regime_posterior` $\rightarrow$ returns `result.p_t_posterior` (`None` if withheld)
- `result.m_t` $\rightarrow$ returns `result.m_t_posterior_mean` (`None` if withheld; also `None` for the baseline model)
- `result.k_selected` $\rightarrow$ returns `result.K`
- `result.bic_by_k` $\rightarrow$ returns BIC map from convergence metadata
- `result.duration_info` $\rightarrow$ returns duration parameter dictionary
