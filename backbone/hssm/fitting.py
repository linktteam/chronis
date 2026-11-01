"""
Sprint 3, Day 8 — EM fitting harness + K selection via BIC.

Spec:
  - hard minimum of 10 random initializations per fit
  - select the CONVERGED run with the highest log-likelihood, never "best-looking"
  - K selection by BIC ONLY (K in {2,3,4})

Label-switching canonicalization lives in label_switching.py (separated per
Palash review) and is applied here as the final step before returning.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, cast

from backbone.hssm.model import GaussianHSMM, KimHSSMModel, FitState, NotFittedError, FittingConvergenceError, InternalStateError
from backbone.hssm.dual_model import DualTimescaleHSSM
from backbone.hssm.label_switching import canonicalize_labels
from backbone.hssm.config import DEFAULT_FIT_CONFIG, HSSMFitConfig, DEFAULT_COLD_START_CONFIG, ColdStartConfig, SHARED_PRIOR_DEFAULT_N0, MAX_TIMESTAMP_SPAN_DAYS
from backbone.hssm.gating import ColdStartError, count_present_sessions

if TYPE_CHECKING:     # the prototype module is imported lazily, only when a shared prior is passed: the default path never loads it
    from backbone.hssm.shrinkage import SharedPrior

logger = logging.getLogger(__name__)   # internal log only; the hold-back (WP3 item 2) never produces a user-facing notice


def validate_timestamps_days(timestamps, n_rows: int | None = None) -> np.ndarray:
    """Check that `timestamps` is a usable vector of calendar DAYS (WP6 item 1); return it as a float array.

    `timestamps` are CALENDAR DAYS (fractional days allowed, one row per calendar day): the baseline model places row i on a daily lattice at
    `round(timestamps[i] - timestamps[0])`, so the span of the record is the length of the lattice. Raises `ValueError` (the message names the unit
    mistake) when the input is not a 1-D array of numbers, its length differs from `n_rows`, any value is NaN or infinite, the values are not strictly
    increasing, the span exceeds `MAX_TIMESTAMP_SPAN_DAYS`, or two timestamps fall on the same calendar day after rounding (the engine would otherwise
    move one of them to the next day without saying so). Nothing is allocated before this check passes.
    """
    try:
        ts = np.asarray(timestamps, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"timestamps must be a 1-D array of numbers in calendar DAYS ({exc})") from exc
    if ts.ndim != 1:
        raise ValueError(f"timestamps must be 1-dimensional (one calendar-day value per row); got shape {ts.shape}")
    if n_rows is not None and len(ts) != n_rows:
        raise ValueError(f"timestamps has {len(ts)} values but the matrix has {n_rows} rows")
    if len(ts) == 0:
        raise ValueError("timestamps is empty")
    if not np.all(np.isfinite(ts)):
        raise ValueError("timestamps contains NaN or infinite values")
    if np.any(np.diff(ts) <= 0):
        raise ValueError("timestamps must be strictly increasing (one row per calendar day; sort the rows and drop duplicate days)")
    span = float(ts[-1] - ts[0])
    if span > MAX_TIMESTAMP_SPAN_DAYS:
        raise ValueError(
            f"timestamps span {span:g} days ({span / 365.25:.0f} years), more than the limit of {MAX_TIMESTAMP_SPAN_DAYS} days. Timestamps are calendar "
            f"DAYS; a span this large is the signature of seconds or milliseconds (seconds passed as days gave a 6.4 GiB lattice). Convert to days."
        )
    if np.any(np.diff(np.round(ts - ts[0]).astype(int)) <= 0):
        raise ValueError("two timestamps fall on the same calendar day after rounding to whole days; timestamps are calendar DAYS, one row per day")
    return ts


def fit_with_random_restarts(
    X: np.ndarray,
    n_regimes: int,
    n_features: int,
    n_init: int = DEFAULT_FIT_CONFIG.n_init,
    n_iter: int = DEFAULT_FIT_CONFIG.n_iter,
    max_duration: int = DEFAULT_FIT_CONFIG.max_duration,
    base_seed: int = 0,
    verbose: bool = False,
    bypass_init_gate: bool = False,
    timestamps: np.ndarray | None = None,
) -> tuple[GaussianHSMM, list[dict]]:
    """Fit with >= n_init random initializations, return the converged run
    with highest log-likelihood, canonicalized via the MP-02 label-switching
    fix. run_log records every attempt for auditability."""
    if not bypass_init_gate:
        if n_init < 10:
            raise ValueError("Directive Day 8 requires a HARD MINIMUM of 10 random initializations.")
    else:
        if n_init < 1:
            raise ValueError("Must run at least 1 random initialization.")

    if timestamps is not None:
        validate_timestamps_days(timestamps, len(X))          # WP6 item 1: calendar DAYS, checked before anything is allocated

    run_log = []
    best_model: GaussianHSMM | None = None
    best_ll = -np.inf

    for i in range(n_init):
        model = GaussianHSMM(n_regimes=n_regimes, n_features=n_features,
                              max_duration=max_duration, seed=base_seed + i)
        if timestamps is not None:
            model.fit(X, n_iter=n_iter, verbose=False, timestamps=timestamps)
        else:
            model.fit(X, n_iter=n_iter, verbose=False)

        run_log.append({
            "init": i,
            "seed": base_seed + i,
            "converged": bool(model.converged_),
            "fit_state": model.fit_state_,
            "n_iter": model.n_iter_,
            "log_likelihood": float(model.log_likelihood_) if model.log_likelihood_ is not None else None,
            "monotonic_ll": model.is_log_likelihood_monotonic(),
            "duration_fallback_used": getattr(model, "duration_fallback_used_", False),
            "duration_optimizer_success": getattr(model, "duration_optimizer_success_", {}),
        })
        if verbose:
            ll_val = model.log_likelihood_ if model.log_likelihood_ is not None else 0.0
            print(f"  init {i}: fit_state={model.fit_state_}, converged={model.converged_}, ll={ll_val:.3f}")

        if model.fit_state_ == FitState.CONVERGED:
            ll = model.log_likelihood_
            if ll is None:
                raise InternalStateError("Log likelihood is None for converged model")
            if ll > best_ll:
                best_ll = ll
                best_model = model

    if best_model is None:
        raise FittingConvergenceError(
            f"None of {n_init} random-init EM runs converged. Real failure, "
            f"not something to paper over — check data quality, n_iter, or "
            f"max_duration before proceeding."
        )

    convergence_rate = sum(1.0 if r["converged"] else 0.0 for r in run_log) / len(run_log)
    selected_model = best_model
    canonicalized_model = canonicalize_labels(selected_model)
    if canonicalized_model is None:
        raise InternalStateError("canonicalize_labels violated its postcondition: returned None")
    if canonicalized_model is not selected_model:
        raise InternalStateError(
            "canonicalize_labels violated in-place canonicalization contract"
        )
    best_model = cast(GaussianHSMM, canonicalized_model)
    best_model.convergence_rate_ = convergence_rate

    return best_model, run_log


def fit_dual_with_random_restarts(
    X: np.ndarray,
    n_regimes: int,
    n_features: int,
    latent_dim: int = 2,
    n_init: int = DEFAULT_FIT_CONFIG.n_init,
    n_iter: int = 50,
    max_duration: int = DEFAULT_FIT_CONFIG.max_duration,
    base_seed: int = 0,
    verbose: bool = False,
    bypass_init_gate: bool = False,
    timestamps: np.ndarray | None = None,
    shared_prior: "SharedPrior | None" = None,
    shared_prior_n0: float | None = None,
) -> tuple[DualTimescaleHSSM, list[dict]]:
    """Fit DualTimescaleHSSM with multiple initializations, return highest-LL model.

    `timestamps` (WP6 item 1) are CALENDAR DAYS and are validated by `validate_timestamps_days` (finite, strictly increasing, span at most
    `MAX_TIMESTAMP_SPAN_DAYS`); this model does not build a calendar lattice, it labels `duration_unit` and works in row steps.

    Shared prior (WP5, off by default). With `shared_prior=None` (the default) nothing below changes. With a `SharedPrior` the fit is NOT a
    random-restart fit: it starts at the shared parameters (so it stays in the shared latent frame) and its M-step is regularised toward them with
    strength `shared_prior_n0` pseudo-steps per regime (default `SHARED_PRIOR_DEFAULT_N0`); restarts from one deterministic starting point would be
    identical, so exactly one fit is run and the `n_init` minimum is not enforced. The prior's (K, F, L) must equal (n_regimes, n_features, latent_dim).

    Selection (WP4 item 1b, ledger A7 / decision D3): a restart whose fit was degraded (`fit_degraded_`: the -1e6 Cholesky
    fallback fired) is never chosen while a healthy restart exists, because that fit's log-likelihood contains the magic
    constant and can look better than a healthy one. Among healthy restarts the highest log-likelihood wins, first maximum on
    ties (unchanged). If every usable restart is degraded, the highest-log-likelihood one is returned and the HSSMResult built
    from it is withheld.
    """
    if timestamps is not None:
        validate_timestamps_days(timestamps, len(X))          # WP6 item 1: calendar DAYS, checked before anything is allocated

    if shared_prior is None:
        if shared_prior_n0 is not None:
            raise ValueError("shared_prior_n0 was given without a shared_prior")
    else:
        return _fit_dual_from_shared_prior(X, n_regimes, n_features, latent_dim, n_iter, max_duration, base_seed, verbose, timestamps, shared_prior, shared_prior_n0)

    if not bypass_init_gate:
        if n_init < 10:
            raise ValueError("Directive Day 8 requires a HARD MINIMUM of 10 random initializations.")
    else:
        if n_init < 1:
            raise ValueError("Must run at least 1 random initialization.")

    run_log = []
    best_ll = -np.inf
    best_model: DualTimescaleHSSM | None = None
    best_degraded_ll = -np.inf
    best_degraded_model: DualTimescaleHSSM | None = None

    for i in range(n_init):
        seed = base_seed + i * 31
        model = DualTimescaleHSSM(
            n_regimes=n_regimes,
            n_features=n_features,
            latent_dim=latent_dim,
            max_duration=max_duration,
            seed=seed,
        )
        model.fit(X, n_iter=n_iter, timestamps=timestamps, verbose=verbose)

        run_info = {
            "init_index": i,
            "seed": seed,
            "fit_state": model.fit_state_,
            "converged": model.converged_,
            "log_likelihood": model.log_likelihood_,
            "n_iter": model.n_iter_,
        }
        run_log.append(run_info)

        if model.fit_state_ in (FitState.CONVERGED, FitState.MAX_ITER_REACHED, FitState.REJECTED_UPDATE):
            ll = model.log_likelihood_
            if ll is not None:
                if getattr(model, "fit_degraded_", False):
                    if ll > best_degraded_ll:
                        best_degraded_ll = ll
                        best_degraded_model = model
                elif ll > best_ll:
                    best_ll = ll
                    best_model = model

    if best_model is None:
        best_model = best_degraded_model     # only reached when no healthy restart exists; the result is then withheld
    if best_model is None:
        raise FittingConvergenceError("None of the random restarts converged for DualTimescaleHSSM")

    convergence_rate = sum(1.0 if r["converged"] else 0.0 for r in run_log) / len(run_log)
    best_model.convergence_rate_ = convergence_rate
    return best_model, run_log


def _fit_dual_from_shared_prior(X, n_regimes, n_features, latent_dim, n_iter, max_duration, base_seed, verbose, timestamps, shared_prior, shared_prior_n0):
    """The shared-prior branch of fit_dual_with_random_restarts (WP5). Fails closed: a prior that is not aggregate-only, or whose (K, F, L) differ from the
    request, raises ValueError; a fit that did not finish raises FittingConvergenceError. No silent fall-back to a prior-free fit."""
    from backbone.hssm.shrinkage import ShrinkageDualHSSM        # lazy: see the TYPE_CHECKING note at the top of this module
    n0 = SHARED_PRIOR_DEFAULT_N0 if shared_prior_n0 is None else float(shared_prior_n0)
    if (shared_prior.K, shared_prior.F, shared_prior.L) != (n_regimes, n_features, latent_dim):
        raise ValueError(f"shared prior has (K, F, L) = {(shared_prior.K, shared_prior.F, shared_prior.L)}, the fit asks for {(n_regimes, n_features, latent_dim)}")
    try:
        shared_prior.assert_aggregate_only()
    except AssertionError as exc:
        raise ValueError(f"shared prior refused: {exc}") from exc
    model = ShrinkageDualHSSM(n_regimes=n_regimes, n_features=n_features, latent_dim=latent_dim, max_duration=max_duration, seed=base_seed,
                              shared_prior=shared_prior, n0=n0)
    model.fit(X, n_iter=n_iter, timestamps=timestamps, verbose=verbose)
    run_log = [{
        "init_index": 0, "seed": base_seed, "fit_state": model.fit_state_, "converged": model.converged_, "log_likelihood": model.log_likelihood_,
        "n_iter": model.n_iter_, "shared_prior": True, "shared_prior_n0": n0,
    }]
    if model.fit_state_ not in (FitState.CONVERGED, FitState.MAX_ITER_REACHED, FitState.REJECTED_UPDATE) or model.log_likelihood_ is None:
        raise FittingConvergenceError(f"The shared-prior fit did not finish for DualTimescaleHSSM (fit_state={model.fit_state_})")
    model.convergence_rate_ = 1.0 if model.converged_ else 0.0
    return model, run_log


def select_k_by_bic(
    X: np.ndarray,
    n_features: int,
    k_candidates: tuple[int, ...] = DEFAULT_FIT_CONFIG.k_candidates,
    n_init: int = DEFAULT_FIT_CONFIG.n_init,
    n_iter: int = DEFAULT_FIT_CONFIG.n_iter,
    max_duration: int = DEFAULT_FIT_CONFIG.max_duration,
    base_seed: int = 0,
    verbose: bool = False,
    bypass_init_gate: bool = False,
    timestamps: np.ndarray | None = None,
) -> tuple[GaussianHSMM, dict]:
    """Fit each candidate K with the full random-restart harness, select by
    BIC only — never because a K 'gives more interesting regimes'.

    WP5 item 0a: a candidate whose selected model is degraded (`fit_degraded_` true: the -1e6 Cholesky fallback fired) is not chosen
    while any healthy candidate exists; if every candidate is degraded the lowest BIC wins as before (and the result built from it is
    withheld by the hold-back rule). Models without the flag (the baseline GaussianHSMM) count as healthy, so for them nothing changes.
    """
    if not k_candidates or len(k_candidates) == 0:
        raise ValueError("k_candidates must be a non-empty sequence of positive integers")
    if any(k <= 0 for k in k_candidates):
        raise ValueError("All candidate K values must be strictly positive (K > 0)")

    results = {}
    best_k = None
    best_bic = np.inf
    best_model: GaussianHSMM | None = None
    best_degraded_k = None
    best_degraded_bic = np.inf
    best_degraded_model: GaussianHSMM | None = None

    for k in k_candidates:
        if verbose:
            print(f"Fitting K={k}...")
        try:
            model, run_log = fit_with_random_restarts(
                X, n_regimes=k, n_features=n_features, n_init=n_init, n_iter=n_iter,
                max_duration=max_duration, base_seed=base_seed + k * 1000, verbose=False,
                bypass_init_gate=bypass_init_gate, timestamps=timestamps,
            )
            if not model.converged_:
                raise InternalStateError("Selected model is not marked as converged")
            if model.log_likelihood_ is None or not np.isfinite(model.log_likelihood_):
                raise InternalStateError(
                    "Selected model has an invalid log_likelihood_"
                )
            n_obs = count_present_sessions(X) if count_present_sessions(X) > 0 else X.shape[0]
            bic = model.bic(n_observations=n_obs)
            results[k] = {
                "bic": float(bic),
                "log_likelihood": float(model.log_likelihood_),
                "convergence_rate": model.convergence_rate_,
                "run_log": run_log,
            }
            if verbose:
                print(f"  K={k}: BIC={bic:.2f}, conv_rate={model.convergence_rate_:.2f}")

            if getattr(model, "fit_degraded_", False):
                if bic < best_degraded_bic:
                    best_degraded_bic = bic
                    best_degraded_k = k
                    best_degraded_model = model
            elif bic < best_bic:
                best_bic = bic
                best_k = k
                best_model = model
        except FittingConvergenceError:
            results[k] = {
                "bic": float("inf"),
                "log_likelihood": float("-inf"),
                "convergence_rate": 0.0,
                "run_log": [],
            }
            if verbose:
                print(f"  K={k}: No random restart converged")

    if best_model is None:                 # no healthy candidate: choose among the degraded ones by BIC (the result is then withheld)
        best_k, best_model = best_degraded_k, best_degraded_model
    if best_model is None:
        raise FittingConvergenceError("None of the candidate Ks converged across random restarts")
    selection_report = {
        "selected_k": best_k,
        "bic_by_k": {k: results[k]["bic"] for k in results},
        "loglik_by_k": {k: results[k]["log_likelihood"] for k in results},
        "convergence_rate_by_k": {k: results[k]["convergence_rate"] for k in results},
        "selection_rule": "min BIC across K candidates — never chosen for 'interesting regimes'",
        "run_log": results[best_k]["run_log"],
    }
    return best_model, selection_report


def fit_hssm_model(
    observations: np.ndarray | pd.DataFrame,
    candidate_ks: Sequence[int] = (2, 3, 4),
    n_initializations: int = 10,
    max_iter: int = 200,
    tol: float = 1e-4,
    random_seed: Optional[int] = None,
    allow_fast_test_fit: bool = False,
) -> Tuple[KimHSSMModel, Dict[str, Any]]:
    """Legacy wrapper for backward compatibility with existing test suites.

    If allow_fast_test_fit is False (default), n_initializations < 10 will raise.
    Set to True only for fast unit testing.
    """
    if isinstance(observations, pd.DataFrame):
        X = observations.to_numpy(dtype=float)
    else:
        X = np.asarray(observations, dtype=float)

    n_features = X.shape[1]

    # Explicitly check for NaNs as expected by some legacy tests
    if np.isnan(X).any():
        raise ValueError("Observation matrix contains NaN values; missingness must be handled upstream.")

    model, report = select_k_by_bic(
        X,
        n_features=n_features,
        k_candidates=tuple(candidate_ks),
        n_init=n_initializations,
        n_iter=max_iter,
        base_seed=random_seed or 0,
        bypass_init_gate=allow_fast_test_fit,
    )

    compat_report = {
        "k_selected": report["selected_k"],
        "bic_by_k": report["bic_by_k"],
        "loglik_by_k": report["loglik_by_k"],
        "convergence_rate_by_k": report["convergence_rate_by_k"],
        "selected_loglik": model.log_likelihood_,
        "candidate_ks": tuple(candidate_ks),
        "n_initializations": n_initializations,
        "max_iter": max_iter,
        "tol": tol,
    }

    # Convert GaussianHSMM to KimHSSMModel for compatibility (using genuine EM-integrated durations)
    if model.var is None or model.A is None or model.mu is None or model.dur_mu is None or model.dur_sigma is None:
        raise InternalStateError("Fitted model has uninitialized parameters")

    covariances = [np.diag(v) for v in model.var]
    fitted_model = KimHSSMModel(
        n_regimes=model.K,
        n_features=model.F,
        transition_matrix=model.A,
        emission_means=model.mu,
        emission_covariances=covariances,
        duration_mu=model.dur_mu,
        duration_sigma=model.dur_sigma,
        duration_prior="lognormal",
        max_duration=model.Dmax,
        seed=random_seed,
    )
    fitted_model.log_likelihood_ = model.log_likelihood_
    fitted_model.log_likelihood_history_ = model.log_likelihood_history_
    fitted_model.fit_state_ = model.fit_state_
    fitted_model.converged_ = model.converged_
    fitted_model.n_iter_ = model.n_iter_

    return fitted_model, compat_report


def compute_dmax_tail_diagnostic(model: GaussianHSMM) -> dict[int, float]:
    """Ticket S34.4: Compute posterior mass at truncation boundary Dmax for each regime."""
    model._require_fitted()
    dur_logpmf = model._duration_logpmf()
    tail_mass = {}
    for k in range(model.K):
        tail_mass[k] = float(np.exp(dur_logpmf[k, -1]))
    return tail_mass


def sweep_dmax_sensitivity(
    X: np.ndarray,
    n_regimes: int,
    n_features: int,
    dmax_candidates: tuple[int, ...] = (30, 45, 60),
    n_init: int = 10,
    base_seed: int = 0,
) -> dict[int, dict[str, Any]]:
    """Ticket S34.4: Sweep across candidate Dmax values and report log-likelihood,
    BIC, and tail mass at Dmax for diagnostic inspection."""
    report = {}
    for dmax in dmax_candidates:
        model, _ = fit_with_random_restarts(
            X,
            n_regimes=n_regimes,
            n_features=n_features,
            n_init=n_init,
            max_duration=dmax,
            base_seed=base_seed,
        )
        tail_mass = compute_dmax_tail_diagnostic(model)
        n_obs = count_present_sessions(X) if count_present_sessions(X) > 0 else X.shape[0]
        report[dmax] = {
            "log_likelihood": float(model.log_likelihood_) if model.log_likelihood_ is not None else 0.0,
            "bic": float(model.bic(n_observations=n_obs)),
            "max_tail_mass": max(tail_mass.values()),
            "tail_mass_by_regime": tail_mass,
        }
    return report


@dataclass
class HSSMResult:
    """Canonical result contract for fit_hssm — directive Section Q.

    All fields required by the directive are present. For the baseline model,
    fast-state fields (m_t, m_t_uncertainty) are explicitly None — never faked.
    Neural residual fields default to disabled/None.
    """
    # --- Core model ---
    model: Any

    # --- Identity ---
    user_id: str = ""
    model_identity: str = ""
    model_family: str = ""
    model_capability: dict = field(default_factory=dict)
    is_baseline_model: bool = True
    fast_state_supported: bool = False

    # --- Slow state p_t ---
    p_t_posterior: np.ndarray | None = None  # shape (T, K) posterior probabilities
    p_t_estimate: np.ndarray | None = None  # shape (T,) MAP regime assignment

    # --- Fast state m_t ---
    m_t_posterior_mean: np.ndarray | None = None  # shape (T, latent_dim)
    m_t_uncertainty: Any = None  # shape (T, latent_dim) or (T, latent_dim, latent_dim)

    # --- Duration ---
    duration_parameters: dict[str, Any] = field(default_factory=dict)
    duration_model_identity: str = ""
    Dmax: int = 0
    duration_unit_source: str = ""

    # --- Temporal coverage ---
    calendar_start: Optional[float] = None
    calendar_end: Optional[float] = None
    observation_density: float = 0.0

    # --- Versioning ---
    model_version: str = "1.0.0"
    feature_schema_version: str = "1.0.0"
    alignment_version: str = "1.0.0"

    # --- Eligibility ---
    eligible_session_count: int = 0
    min_present_sessions: int = 30

    # --- Data hash ---
    fit_dataset_hash: str = ""

    # --- Model selection ---
    K: int = 0
    BIC: float = float("inf")
    AIC: float = float("inf")
    heldout_predictive_log_likelihood: Optional[float] = None

    # --- Fit state ---
    fit_state: FitState = FitState.FAILED
    optimizer_success: bool = False
    fallback_used: bool = False
    run_log: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    # --- Numerical health of the fit (ledger A7 / N21, WP3 item 2) ---
    # Carried from the fitted DualTimescaleHSSM (fit_degraded_, fit_cholesky_fallbacks_, fit_exceptions_).
    # Safe defaults for every other path (the baseline model has no such flags).
    fit_degraded: bool = False                 # the -1e6 Cholesky fallback fired during fit()
    fit_cholesky_fallbacks: int = 0            # how many times it fired during that fit()
    fit_exceptions: list = field(default_factory=list)   # one dict per exception caught inside fit()
    # WP4 item 1c: firings AFTER fit() finished. `states_pass_fallbacks` = firings inside the estimate_states pass of fit_hssm (they happen
    # before the result exists); `fallback_count_at_build` = the model's cumulative count when the result was built (None: not recorded).
    states_pass_fallbacks: int = 0
    fallback_count_at_build: Optional[int] = None

    # --- Shared prior (WP5, N18-v2): off by default ---
    used_shared_prior: bool = False                      # the fit started at a shared prior and was regularised toward it
    shared_prior_n0: Optional[float] = None              # its strength, in pseudo-steps per regime; None when no prior was used

    # --- Neural residual ---
    neural_residual_enabled: bool = False
    residual_mode: str = "DISABLED"
    residual_weight: float = 0.0
    residual_model_version: str = ""
    residual_ablation_status: str = "NOT_APPLICABLE"

    # --- Legacy backward-compat fields (kept for existing callers) ---
    duration_info: dict[str, Any] = field(default_factory=dict)
    duration_unit: str = "sessions"
    selected_k: int = 0
    model_class: str = ""
    convergence_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # private once-flag for the internal WARNING (not a dataclass field: not compared, not in asdict, not in the schema)
        self._holdback_logged = False
        self._log_seed: Optional[int] = None

    # --- Hold-back (WP3 item 2, owner decision D3; closed in WP4 item 1). Derived from the CURRENT field values and the CURRENT model
    # count, never stored, so it cannot go stale. ---
    @property
    def late_fallbacks(self) -> int:
        """Firings of the -1e6 Cholesky fallback on the model AFTER this result was built (for example a later `estimate_states` call
        on `self.model`). 0 when the count at build was not recorded or the model has no counter (baseline model)."""
        at_build = self.fallback_count_at_build
        now = getattr(self.model, "cholesky_fallback_count_", None)
        if at_build is None or not isinstance(now, (int, np.integer)) or isinstance(now, bool):
            return 0
        return max(int(now) - int(at_build), 0)

    @property
    def withheld_reason(self) -> Optional[str]:
        """None when insights may be used; otherwise every cause that applies. RULE: withheld when `fit_degraded` is True,
        or `fit_state == FitState.NUMERICAL_FAILURE`, or `states_pass_fallbacks > 0` (the fallback fired in the estimate_states pass
        that produced the outputs), or `late_fallbacks > 0` (it fired on the model after this result was built)."""
        reasons = []
        if self.fit_degraded:
            reasons.append(f"fit_degraded: the -1e6 Cholesky fallback fired {self.fit_cholesky_fallbacks} time(s) during fit")
        if self.fit_state == FitState.NUMERICAL_FAILURE:
            reasons.append("fit_state is NUMERICAL_FAILURE")
        if self.states_pass_fallbacks > 0:
            reasons.append(f"states_pass: the -1e6 Cholesky fallback fired {self.states_pass_fallbacks} time(s) in the estimate_states pass")
        late = self.late_fallbacks
        if late > 0:
            reasons.append(f"late_fallbacks: the -1e6 Cholesky fallback fired {late} time(s) on the model after this result was built")
        return "; ".join(reasons) if reasons else None

    @property
    def insights_withheld(self) -> bool:
        return self.withheld_reason is not None

    def _withheld_on_read(self) -> bool:
        """True when the insights are withheld. The first time this is seen for a result, one internal WARNING is logged (once per
        result, shared with the build-time log in `_log_holdback`)."""
        if not self.insights_withheld:
            return False
        _log_holdback(self, getattr(self, "_log_seed", None))
        return True

    def publishable_outputs(self) -> dict[str, Any] | None:
        """The ONLY supported way for the product layer to read regime/latent outputs.

        Returns None when the insights are withheld (see `withheld_reason`), otherwise the four output arrays. The raw fields
        (`p_t_estimate`, `p_t_posterior`, `m_t_posterior_mean`, `m_t_uncertainty`) stay readable for internal diagnostics, but anything
        that reaches a user must come through this accessor. The legacy aliases `p_t`, `regime_posterior`, `m_t` follow the same rule.
        The arrays are the stored ones, not copies: do not modify them.
        """
        if self._withheld_on_read():
            return None
        return {
            "p_t_estimate": self.p_t_estimate,
            "p_t_posterior": self.p_t_posterior,
            "m_t_posterior_mean": self.m_t_posterior_mean,
            "m_t_uncertainty": self.m_t_uncertainty,
        }

    # Legacy aliases. WP4 item 1d: they return None when the insights are withheld (the raw fields above stay readable).
    @property
    def p_t(self) -> np.ndarray | None:
        """Legacy alias: MAP regime assignment. None when the insights are withheld."""
        return None if self._withheld_on_read() else self.p_t_estimate

    @property
    def regime_posterior(self) -> np.ndarray | None:
        """Legacy alias: full regime posterior. None when the insights are withheld."""
        return None if self._withheld_on_read() else self.p_t_posterior

    @property
    def m_t(self) -> np.ndarray | None:
        """Legacy alias: fast state posterior mean. None when the insights are withheld."""
        return None if self._withheld_on_read() else self.m_t_posterior_mean

    @property
    def k_selected(self) -> int:
        return self.selected_k or self.K

    @property
    def bic_by_k(self) -> dict[int, float]:
        return self.convergence_metadata.get("bic_by_k", {})

    @property
    def loglik_by_k(self) -> dict[int, float]:
        return self.convergence_metadata.get("loglik_by_k", {})

    @property
    def convergence_rate_by_k(self) -> dict[int, float]:
        return self.convergence_metadata.get("convergence_rate_by_k", {})

    @property
    def selected_loglik(self) -> float | None:
        return self.convergence_metadata.get("selected_loglik")

    @property
    def duration_fallback_used(self) -> bool:
        """True if ANY regime's duration optimizer fell back to the untruncated estimate."""
        return self.fallback_used or bool(self.convergence_metadata.get("duration_fallback_used", False))

    @property
    def duration_optimizer_success(self) -> bool:
        """Aggregate 'all-clear' boolean: True only if NO regime fell back."""
        return self.optimizer_success and bool(self.convergence_metadata.get("duration_optimizer_success", True))

    @property
    def per_regime_optimizer_status(self) -> dict[int, bool]:
        """Mapping of regime index -> boolean for duration optimization success."""
        return self.convergence_metadata.get("per_regime_optimizer_status", {})


def _compute_heldout_predictive_loglik(
    X: np.ndarray,
    model: GaussianHSMM,
    holdout_fraction: float = 0.2,
) -> Optional[float]:
    """Chronological heldout predictive log-likelihood.

    Uses the last holdout_fraction of the data as test set.
    The model is already fit on the full data (this is a diagnostic, not a refit).
    We compute the emission log-likelihood of the held-out portion under
    the fitted model's regime posterior, as a predictive quality metric.
    """
    T = X.shape[0]
    split_idx = max(1, int(T * (1.0 - holdout_fraction)))
    if split_idx >= T - 1:
        return None  # Not enough data for holdout

    X_test = X[split_idx:]
    try:
        emission_ll = model._emission_loglik(X_test)  # (T_test, K)
        # Marginal emission LL under uniform regime prior (conservative)
        from scipy.special import logsumexp as _logsumexp
        log_pi = np.log(np.clip(model.pi, 1e-300, None)) if model.pi is not None else np.zeros(model.K)
        marginal_ll = _logsumexp(emission_ll + log_pi[None, :], axis=1)  # (T_test,)
        return float(np.sum(marginal_ll))
    except Exception:
        return None


def _dual_health_fields(model: DualTimescaleHSSM) -> dict[str, Any]:
    """Numerical-health fields of a fitted DualTimescaleHSSM, as HSSMResult keyword arguments."""
    return dict(
        fit_degraded=bool(getattr(model, "fit_degraded_", False)),
        fit_cholesky_fallbacks=int(getattr(model, "fit_cholesky_fallbacks_", 0)),
        fit_exceptions=[dict(e) for e in getattr(model, "fit_exceptions_", [])],
    )


def _log_holdback(result: HSSMResult, random_seed: Optional[int]) -> None:
    """One internal WARNING per result when its insights are withheld (at build time, or the first time a result that became withheld
    later is read). Nothing user-facing is added to the result."""
    if random_seed is not None:
        result._log_seed = random_seed
    if not result.insights_withheld or getattr(result, "_holdback_logged", False):
        return
    result._holdback_logged = True
    exception_types = sorted({str(e.get("exception_type", "?")) for e in result.fit_exceptions})
    logger.warning(
        "HSSM insights withheld: reason=[%s]; fit_cholesky_fallbacks=%d; exception_types=%s; random_seed=%s; user_id=%s; dataset_hash=%s; model_class=%s",
        result.withheld_reason, result.fit_cholesky_fallbacks, exception_types, random_seed, result.user_id,
        result.fit_dataset_hash, result.model_class,
    )


def fit_hssm(
    matrix: np.ndarray | pd.DataFrame,
    candidate_ks: Sequence[int] = (2, 3, 4),
    n_initializations: int = 10,
    max_iter: int = 200,
    random_seed: Optional[int] = None,
    allow_fast_test_fit: bool = False,
    timestamps: Optional[np.ndarray] = None,
    config: ColdStartConfig = DEFAULT_COLD_START_CONFIG,
    n_present_sessions: Optional[int] = None,
    max_duration: int = 45,
    user_id: str = "",
    dataset_hash: str = "",
    model_type: str = "baseline",
    latent_dim: int = 2,
    shared_prior: "SharedPrior | None" = None,
    shared_prior_n0: float | None = None,
) -> HSSMResult:
    """Canonical fitting entrypoint for backbone.hssm per directive.

    This is the ONLY public fitting entrypoint. All legacy wrappers
    (fit_hssm_model, fit_hssm_gated) delegate to this function.

    Enforces cold-start gate (VF-14): requires >= config.min_present_sessions
    eligible sessions or raises ColdStartError.

    Parameters:
      matrix: (T, F) float matrix, one row per day or session, z-scored per person; NaN rows (missing days) are allowed and are marginalised.
      timestamps: optional, CALENDAR DAYS (WP6 item 1): one value per row, finite, strictly increasing, fractional days allowed but no two rows on the
                  same calendar day after rounding, and a span (last - first) of at most `MAX_TIMESTAMP_SPAN_DAYS` (20,000). Seconds or milliseconds
                  raise `ValueError` before anything is allocated. The baseline model places the rows on a daily lattice with these offsets (gaps
                  become NaN rows); the dual-timescale model uses them only to label `duration_unit` and works in row steps. See
                  docs/INTEGRATION_CONTRACT_HSSM.md.
      model_type: 'baseline' (GaussianHSMM, slow-regime only) or
                  'dual_timescale' (DualTimescaleHSSM, fast m_t + slow p_t).
      shared_prior, shared_prior_n0 (WP5, off by default): only for the dual-timescale model. With a `SharedPrior` the fit starts at the shared
                  parameters and is regularised toward them with strength `shared_prior_n0` pseudo-steps per regime (default
                  `SHARED_PRIOR_DEFAULT_N0`). K is the prior's K (it must be among `candidate_ks`; no BIC search over K), one fit is run (see
                  `fit_dual_with_random_restarts`), and the result records `used_shared_prior=True` and `shared_prior_n0`. The cold-start gate below
                  is NOT changed by a prior (a caller may pass a `config` with a lower minimum). Default `None`: identical to before.
    """
    if isinstance(matrix, pd.DataFrame):
        X = matrix.to_numpy(dtype=float)
    else:
        X = np.asarray(matrix, dtype=float)

    if X.ndim < 2 or X.shape[1] <= 0:
        raise ValueError("Feature count must be strictly greater than zero (F > 0)")

    n_features = X.shape[1]

    if timestamps is not None:
        validate_timestamps_days(timestamps, X.shape[0])      # WP6 item 1: first, so a malformed argument raises ValueError even for a too-short record

    if shared_prior is None:
        if shared_prior_n0 is not None:
            raise ValueError("shared_prior_n0 was given without a shared_prior")
    else:
        if model_type not in ("dual_timescale", "dual", "full_hssm"):
            raise ValueError("a shared prior applies to the dual-timescale model only (model_type='dual_timescale')")
        if shared_prior.K not in tuple(candidate_ks):
            raise ValueError(f"the shared prior has K={shared_prior.K}, which is not among candidate_ks={tuple(candidate_ks)}")

    if n_present_sessions is None:
        n_present_sessions = count_present_sessions(X)

    if n_present_sessions < config.min_present_sessions:
        raise ColdStartError(
            f"{n_present_sessions} present sessions < cold-start minimum "
            f"({config.min_present_sessions}). No HSSM output produced. This "
            f"is the correct, silent, no-output state per Bible 5.1 doctrine, "
            f"not a low-confidence result."
        )

    # --- Dual Timescale Model Fitting Branch ---
    if model_type in ("dual_timescale", "dual", "full_hssm"):
        best_k = None
        best_bic = float("inf")
        best_dual_model: DualTimescaleHSSM | None = None
        best_degraded_k = None                 # WP5 item 0a: degraded candidates are only used when no healthy candidate exists
        best_degraded_bic = float("inf")
        best_degraded_model: DualTimescaleHSSM | None = None
        bic_by_k = {}
        loglik_by_k = {}
        conv_by_k = {}
        full_run_log = []

        for k in (candidate_ks if shared_prior is None else (shared_prior.K,)):
            dual_m, rlog = fit_dual_with_random_restarts(
                X,
                n_regimes=k,
                n_features=n_features,
                latent_dim=latent_dim,
                n_init=n_initializations,
                n_iter=max_iter,
                max_duration=max_duration,
                base_seed=(random_seed or 0) + k * 1000,
                bypass_init_gate=allow_fast_test_fit,
                timestamps=timestamps,
                shared_prior=shared_prior,
                shared_prior_n0=shared_prior_n0,
            )
            n_obs = n_present_sessions if n_present_sessions > 0 else X.shape[0]
            bic = dual_m.bic(n_observations=n_obs)
            bic_by_k[k] = float(bic)
            loglik_by_k[k] = float(dual_m.log_likelihood_ or 0.0)
            conv_by_k[k] = dual_m.convergence_rate_
            full_run_log.extend(rlog)

            if getattr(dual_m, "fit_degraded_", False):
                if bic < best_degraded_bic:
                    best_degraded_bic = bic
                    best_degraded_k = k
                    best_degraded_model = dual_m
            elif bic < best_bic:
                best_bic = bic
                best_k = k
                best_dual_model = dual_m

        if best_dual_model is None:            # every candidate K is degraded: lowest BIC, the result is withheld by the hold-back rule
            best_k, best_bic, best_dual_model = best_degraded_k, best_degraded_bic, best_degraded_model

        if best_dual_model is None or best_k is None:
            raise FittingConvergenceError("None of the candidate Ks converged across random restarts")

        # WP4 item 1c: the estimate_states pass runs AFTER fit() finished and BEFORE the result exists, so a firing inside it is not
        # in `fit_degraded`. Count it, and remember the model's cumulative count so that later firings (late_fallbacks) are visible too.
        count_before_states = int(best_dual_model.cholesky_fallback_count_)
        m_t, m_t_std, p_t, p_t_post = best_dual_model.estimate_states(X)
        count_after_states = int(best_dual_model.cholesky_fallback_count_)
        aic_val = float(best_dual_model.aic())
        used_prior = bool(getattr(best_dual_model, "used_shared_prior", False))
        prior_n0 = float(best_dual_model.n0) if used_prior else None

        dual_result = HSSMResult(
            model=best_dual_model,
            user_id=user_id,
            model_identity=best_dual_model.model_identity,
            model_family=best_dual_model.model_family,
            model_capability=best_dual_model.model_capability,
            is_baseline_model=False,
            fast_state_supported=True,
            p_t_posterior=p_t_post,
            p_t_estimate=p_t,
            m_t_posterior_mean=m_t,
            m_t_uncertainty=m_t_std,
            duration_parameters={
                "duration_mu": best_dual_model.dur_mu.tolist() if best_dual_model.dur_mu is not None else [],
                "duration_sigma": best_dual_model.dur_sigma.tolist() if best_dual_model.dur_sigma is not None else [],
            },
            duration_model_identity=best_dual_model.duration_model_identity,
            Dmax=best_dual_model.Dmax,
            duration_unit_source=best_dual_model.duration_unit_source,
            calendar_start=float(timestamps[0]) if timestamps is not None and len(timestamps) > 0 else None,
            calendar_end=float(timestamps[-1]) if timestamps is not None and len(timestamps) > 0 else None,
            observation_density=float(n_present_sessions / max(X.shape[0], 1)),
            model_version="1.0.0",
            feature_schema_version="1.0.0",
            alignment_version="1.0.0",
            eligible_session_count=n_present_sessions,
            min_present_sessions=config.min_present_sessions,
            fit_dataset_hash=dataset_hash,
            K=best_k,
            BIC=best_bic,
            AIC=aic_val,
            heldout_predictive_log_likelihood=None,
            fit_state=best_dual_model.fit_state_,
            optimizer_success=True,
            fallback_used=best_dual_model.duration_fallback_used_,
            run_log=full_run_log,
            warnings=[],
            limitations=[],
            **_dual_health_fields(best_dual_model),
            states_pass_fallbacks=count_after_states - count_before_states,
            fallback_count_at_build=count_after_states,
            neural_residual_enabled=False,
            residual_mode="DISABLED",
            residual_weight=0.0,
            residual_model_version="",
            residual_ablation_status="NOT_APPLICABLE",
            duration_info={},
            duration_unit=best_dual_model.duration_unit,
            selected_k=best_k,
            model_class=best_dual_model.__class__.__name__,
            convergence_metadata={
                "bic_by_k": bic_by_k,
                "loglik_by_k": loglik_by_k,
                "convergence_rate_by_k": conv_by_k,
                "selected_loglik": best_dual_model.log_likelihood_,
                "fit_state": best_dual_model.fit_state_,
                "converged": best_dual_model.converged_,
                "n_iter": best_dual_model.n_iter_,
                "run_log": full_run_log,
            },
            used_shared_prior=used_prior,
            shared_prior_n0=prior_n0,
        )
        if used_prior:      # recorded only when a prior was used: the default result's metadata is unchanged
            dual_result.convergence_metadata["shared_prior"] = {
                "n0": prior_n0, "n_users": int(best_dual_model.shared_prior.n_users), "k_fixed_by_prior": True, "n_fits_run": 1,
                "prior_uncertainty_in_m_t_std": False,
                "bic_note": "BIC and AIC use the unpenalised log-likelihood at the shrunk parameters",
            }
        _log_holdback(dual_result, random_seed)
        return dual_result

    # --- Baseline Model Fitting Branch ---
    model, report = select_k_by_bic(
        X,
        n_features=n_features,
        k_candidates=tuple(candidate_ks),
        n_init=n_initializations,
        n_iter=max_iter,
        max_duration=max_duration,
        base_seed=random_seed or 0,
        bypass_init_gate=allow_fast_test_fit,
        timestamps=timestamps,
    )

    if timestamps is not None:
        model.duration_unit = "calendar_days"

    # Compute posterior quantities for the selected model
    regime_post, _, _ = model._forward_backward(X)
    p_t = np.argmax(regime_post, axis=1)

    # Compute information criteria
    n_obs = n_present_sessions if n_present_sessions > 0 else X.shape[0]
    bic_val = float(model.bic(n_observations=n_obs))
    aic_val = float(model.aic())

    # Compute heldout predictive log-likelihood
    heldout_ll = _compute_heldout_predictive_loglik(X, model)

    # Detect MODEL_SELECTION_CONTESTED: BIC and heldout LL disagree on best K
    fit_state = model.fit_state_
    if len(report.get("bic_by_k", {})) > 1 and heldout_ll is not None:
        bic_values = list(report["bic_by_k"].values())
        if len(bic_values) >= 2:
            sorted_bics = sorted(bic_values)
            if len(sorted_bics) >= 2 and abs(sorted_bics[0] - sorted_bics[1]) < 2.0:
                fit_state = FitState.MODEL_SELECTION_CONTESTED

    # Build duration info (legacy compat)
    duration_info = {
        "duration_mu": model.duration_mu,
        "duration_sigma": model.duration_sigma,
        "max_duration": model.Dmax,
        "duration_optimizer_success": getattr(model, "duration_optimizer_success_", {}),
        "duration_fallback_used": getattr(model, "duration_fallback_used_", False),
    }

    # Build convergence metadata (legacy compat)
    convergence_metadata = {
        "bic_by_k": report["bic_by_k"],
        "loglik_by_k": report["loglik_by_k"],
        "convergence_rate_by_k": report["convergence_rate_by_k"],
        "selected_loglik": model.log_likelihood_,
        "fit_state": model.fit_state_,
        "converged": model.converged_,
        "n_iter": model.n_iter_,
        "run_log": report.get("run_log", []),
        "duration_optimizer_success": not getattr(model, "duration_fallback_used_", False),
        "duration_fallback_used": getattr(model, "duration_fallback_used_", False),
        "per_regime_optimizer_status": getattr(model, "duration_optimizer_success_", {}),
    }

    # Collect warnings and limitations
    result_warnings = []
    result_limitations = [
        "Baseline model: fast continuous state m_t is not supported.",
        "Duration units are session-index unless timestamps provided.",
    ]
    if getattr(model, "duration_fallback_used_", False):
        result_warnings.append(
            "Duration optimizer fallback used for one or more regimes. "
            "Untruncated moment estimates used as initialization — not the "
            "final truncated model."
        )

    baseline_result = HSSMResult(
        model=model,
        # Identity
        user_id=user_id,
        model_identity=model.model_identity,
        model_family=model.model_family,
        model_capability=model.model_capability,
        is_baseline_model=model.is_baseline_model,
        fast_state_supported=model.fast_state_supported,
        # Slow state
        p_t_posterior=regime_post,
        p_t_estimate=p_t,
        # Fast state (None for baseline)
        m_t_posterior_mean=None,
        m_t_uncertainty=None,
        # Duration
        duration_parameters={
            "duration_mu": model.duration_mu.tolist() if model.dur_mu is not None else [],
            "duration_sigma": model.duration_sigma.tolist() if model.dur_sigma is not None else [],
        },
        duration_model_identity=model.duration_model_identity,
        Dmax=model.Dmax,
        duration_unit_source=model.duration_unit_source,
        # Temporal
        calendar_start=float(timestamps[0]) if timestamps is not None and len(timestamps) > 0 else None,
        calendar_end=float(timestamps[-1]) if timestamps is not None and len(timestamps) > 0 else None,
        observation_density=float(n_present_sessions / max(X.shape[0], 1)),
        # Versioning
        model_version="1.0.0",
        feature_schema_version="1.0.0",
        alignment_version="1.0.0",
        # Eligibility
        eligible_session_count=n_present_sessions,
        min_present_sessions=config.min_present_sessions,
        # Data hash
        fit_dataset_hash=dataset_hash,
        # Model selection
        K=report["selected_k"],
        BIC=bic_val,
        AIC=aic_val,
        heldout_predictive_log_likelihood=heldout_ll,
        # Fit state
        fit_state=fit_state,
        optimizer_success=not getattr(model, "duration_fallback_used_", False),
        fallback_used=getattr(model, "duration_fallback_used_", False),
        run_log=report.get("run_log", []),
        warnings=result_warnings,
        limitations=result_limitations,
        # Neural residual (disabled for baseline)
        neural_residual_enabled=False,
        residual_mode="DISABLED",
        residual_weight=0.0,
        residual_model_version="",
        residual_ablation_status="NOT_APPLICABLE",
        # Legacy compat fields
        duration_info=duration_info,
        duration_unit=model.duration_unit,
        selected_k=report["selected_k"],
        model_class=model.__class__.__name__,
        convergence_metadata=convergence_metadata,
    )
    _log_holdback(baseline_result, random_seed)
    return baseline_result
