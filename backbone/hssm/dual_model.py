"""
Dual-Timescale Fast+Slow HSSM: Kim (1994)-style Switching Linear Dynamical System
with Semi-Markov Explicit Duration Slow Regimes.

Theoretical Architecture:
  1. Fast continuous latent state m_t in R^L (minute-level / momentary dynamics):
       m_t = A_k m_{t-1} + b_k + w_t,   w_t ~ N(0, Q_k)
  2. Slow discrete regime state p_t in {0, ..., K-1} (macro life-phases):
       d_t ~ LogNormal(mu_k, sigma_k), d in {1, ..., Dmax}
       p_t transitions governed by transition matrix A_slow (zero diagonal)
  3. Observation emission model conditional on continuous state m_t and regime p_t=k:
       y_t = C_k m_t + d_k + v_t,       v_t ~ N(0, R_k)
       where R_k is diagonal covariance.
  4. Kim filter with GPB2 (Generalized Pseudo-Bayes of order 2) collapse and
     Kim backward smoother for continuous state trajectory and covariance.
  5. Missing values (NaNs) handled by marginalization in the Kalman update.
  6. Generalized EM with backtracking rollback ensuring monotonic log-likelihood.
"""

from __future__ import annotations

import os
import traceback
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import lognorm

from backbone.hssm.config import (
    DUAL_CAP_CERT_GAIN_PER_ROW,
    DUAL_CAP_WINDOW,
    DUAL_DURATION_BOUND_TOL,
    DUAL_DURATION_MU_MAX_DMAX_FACTOR,
    DUAL_DURATION_MU_MIN,
    DUAL_DURATION_SIGMA_MAX,
)
from backbone.hssm.model import (
    FitState,
    HSSMError,
    NotFittedError,
    InternalStateError,
    FittingConvergenceError,
    NEG_INF,
)


def _safe_softmax(log_w: np.ndarray) -> np.ndarray:
    """Softmax in log space; falls back to uniform (never NaN) if everything is ~-inf."""
    if np.all(log_w <= NEG_INF / 2):
        return np.full_like(log_w, 1.0 / len(log_w))
    m = np.max(log_w)
    w = np.exp(log_w - m)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full_like(log_w, 1.0 / len(log_w))
    return w / s


def _mixture_moments(means: np.ndarray, covs: np.ndarray, weights: np.ndarray):
    """Moment-match a Gaussian mixture into a single Gaussian (standard GPB2 collapse)."""
    mean = np.sum(weights[:, None] * means, axis=0)
    L = means.shape[1]
    cov = np.zeros((L, L))
    for i in range(len(weights)):
        diff = means[i] - mean
        cov += weights[i] * (covs[i] + np.outer(diff, diff))
    return mean, cov


@dataclass
class ForwardPassResult:
    regime_post_filtered: np.ndarray   # (T, K)  P(k_t=k | y_1:t)
    entry_mass: np.ndarray             # (T, K, D)  see NOTE (a) in _forward_pass
    m_filt: np.ndarray                 # (T, K, L)
    P_filt: np.ndarray                 # (T, K, L, L)
    m_pred: np.ndarray                 # (T, K, L)  diagnostic only
    P_pred: np.ndarray                 # (T, K, L, L)
    emis_loglik: np.ndarray            # (T, K)  diagnostic only, not used for inference
    ll_continuation: np.ndarray        # (T, K)  NEG_INF at t=0
    ll_entry: np.ndarray               # (T, K)  predecessor-marginalized; at t=0 the initial Kalman ll
    ll_branch: np.ndarray              # (T, K, K) [t,i,j] uncollapsed per-predecessor entry likelihood;
                                        #   needed by the discrete backward, see NOTE (c) in _discrete_backward
    alpha: np.ndarray                  # (T, K, D)
    J_cont: np.ndarray                 # (T, K, L, L) P_pred of the (k,k) branch, cached
    log_likelihood: float


@dataclass
class BackwardPassResult:
    beta: np.ndarray                       # (T, K, D)
    regime_posterior_smoothed: np.ndarray  # (T, K)   P(k_t=k | y_1:T)
    gamma: np.ndarray                      # (T, K, D)
    entry_posterior: np.ndarray            # (T, K, D)
    xi_counts: np.ndarray                  # (K, K)
    xi_t: np.ndarray                       # (T-1, K, K)
    future_entry_val: np.ndarray           # (T, K)


@dataclass
class SmootherResult:
    smoothed_m_k: np.ndarray     # (T, K, L)   E[m_t | k_t=k, Y]
    smoothed_P_k: np.ndarray     # (T, K, L, L)
    smoothed_m: np.ndarray       # (T, L)
    smoothed_P: np.ndarray       # (T, L, L)
    # Pair-conditioned quantities for t = 0..T-2, indexed [t, i, k] for the pair
    # (k_t = i, k_{t+1} = k). Needed for the dynamics M-step: regime k's dynamics at time
    # t+1 relate m_{t+1} to m_t, and m_t's distribution depends on which regime i it was
    # generated in (at a segment start i != k). See NOTE (e) in _dynamics_sufficient_stats.
    pair_m: np.ndarray           # (T-1, K, K, L)     E[m_t | k_t=i, k_{t+1}=k, Y]
    pair_P: np.ndarray           # (T-1, K, K, L, L)  Cov[m_t | k_t=i, k_{t+1}=k, Y]
    pair_J: np.ndarray           # (T-1, K, K, L, L)  RTS gain; Cov(m_{t+1}, m_t | pair) = J @ smoothed_P_k[t+1,k]
    pair_w: np.ndarray           # (T-1, K, K)        P(k_t=i, k_{t+1}=k | Y)


class DualTimescaleHSSM:
    """Production-grade Dual-Timescale Switching Linear Dynamical System (HSSM).

    Combines:
      - Fast continuous state m_t in R^L (momentary behavioral dynamics)
      - Slow discrete regime p_t in {0..K-1} with explicit semi-Markov duration
      - Kim (1994) GPB2 filter and backward smoother
      - Full EM with analytical M-step for linear-Gaussian state/observation dynamics
      - Generalized EM monotonicity backtracking
    """

    def __init__(
        self,
        n_regimes: int = 3,
        n_features: int = 2,
        latent_dim: int = 2,
        max_duration: int = 40,
        seed: int | None = None,
        duration_unit: str = "sessions",
    ):
        if n_regimes <= 0:
            raise ValueError("Number of regimes must be strictly positive (K > 0)")
        if n_features <= 0:
            raise ValueError("Number of features must be strictly positive (F > 0)")
        if latent_dim <= 0:
            raise ValueError("Latent dimension must be strictly positive (L > 0)")

        self.K = n_regimes
        self.F = n_features
        self.L = latent_dim
        self.Dmax = max_duration
        self.duration_unit = duration_unit
        self.rng = np.random.default_rng(seed)

        # --- Slow discrete regime parameters ---
        self.pi: np.ndarray | None = None          # (K,) initial regime distribution
        self.A_slow: np.ndarray | None = None      # (K, K) regime transition matrix (0 diagonal)
        self.dur_mu: np.ndarray | None = None      # (K,) log-normal duration mu
        self.dur_sigma: np.ndarray | None = None   # (K,) log-normal duration sigma

        # --- Fast continuous state dynamics conditional on regime k ---
        # m_t = A_k m_{t-1} + b_k + w_t,  w_t ~ N(0, Q_k)
        self.A_fast: np.ndarray | None = None      # (K, L, L) state transition matrices
        self.b_fast: np.ndarray | None = None      # (K, L) state biases
        self.Q_fast: np.ndarray | None = None      # (K, L, L) process noise covariances
        self.m0: np.ndarray | None = None          # (K, L) initial latent means
        self.P0: np.ndarray | None = None          # (K, L, L) initial latent covariances

        # --- Observation emission model conditional on continuous state and regime k ---
        # y_t = C_k m_t + d_k + v_t,  v_t ~ N(0, R_k)
        self.C_obs: np.ndarray | None = None       # (K, F, L) observation loading matrices
        self.d_obs: np.ndarray | None = None       # (K, F) observation intercepts
        self.R_obs: np.ndarray | None = None       # (K, F) observation noise variances (diagonal)

        # --- Fitting diagnostics ---
        self.log_likelihood_: float | None = None
        self.log_likelihood_history_: list[float] = []
        self.fit_state_: FitState = FitState.FAILED
        self.n_iter_: int = 0
        self.convergence_rate_: float = 0.0
        self._is_fitted: bool = False
        self._n_observations: int = 0
        self.duration_fallback_used_: bool = False
        self.duration_optimizer_success_: dict[int, bool] = {}
        self.transition_fallback_rows_: set[int] = set()

        # --- WP6 item 3 part 2 bookkeeping (reported, never fed back into inference; not part of the parameter snapshot) ---
        # how the last fit() ended: "abs_tol" (|ll - prev| < tol), "window_gain_at_cap" (iteration limit reached with a small trailing gain), or None
        self.convergence_criterion_: Optional[str] = None
        # log-likelihood gained over the last DUAL_CAP_WINDOW iterations when the iteration limit was reached (None otherwise)
        self.window_gain_at_cap_: Optional[float] = None
        # one dict (iteration, regime, from_mu, from_sigma) per box redo of the duration update in the last fit()
        self.duration_box_applied_: list[dict[str, Any]] = []
        self._em_iteration: int = 0

        # --- Numerical-health bookkeeping (ledger A7 and N21). Bookkeeping only: nothing
        # here feeds back into inference, and none of it is part of the parameter snapshot,
        # so a rollback never erases the record that something went wrong. ---
        # cumulative number of times the -1e6 Cholesky fallback in _kalman_update_regime fired
        # since construction (E-steps, estimate_states, and anything else that calls it)
        self.cholesky_fallback_count_: int = 0
        # firings during the most recent fit() only, and the flag derived from it
        self.fit_cholesky_fallbacks_: int = 0
        self.fit_degraded_: bool = False
        # every exception caught inside the most recent fit(); one dict per exception with
        # keys iteration, stage, exception_type, message, location
        self.fit_exceptions_: list[dict[str, Any]] = []

    # ---------- Model Identity & Capabilities ----------

    @property
    def model_identity(self) -> str:
        return "DualTimescaleKimHSSMV1"

    @property
    def model_family(self) -> str:
        return "SwitchingLinearDynamicalSystem"

    @property
    def is_baseline_model(self) -> bool:
        return False

    @property
    def fast_state_supported(self) -> bool:
        return True

    @property
    def duration_model_identity(self) -> str:
        return "explicit_truncated_gaussian"

    @property
    def duration_unit_source(self) -> str:
        if self.duration_unit == "calendar_days":
            return "calendar_days"
        return "session_index"

    @property
    def model_capability(self) -> dict[str, Any]:
        return {
            "supports_fast_state_m_t": True,
            "supports_slow_state_p_t": True,
            "supports_explicit_duration": True,
            "supports_neural_residual": False,
            "is_production_grade_full_hssm": True,
            "is_baseline_model": False,
            "fast_state_supported": True,
            "latent_dimension": self.L,
        }

    @property
    def converged_(self) -> bool:
        return self.fit_state_ == FitState.CONVERGED

    @property
    def duration_at_bound_(self) -> dict[int, bool]:
        """Per regime: True when the final duration parameters lie within `DUAL_DURATION_BOUND_TOL` of a wall of the duration box
        (mu at its lower or upper limit, sigma at its upper limit). Empty before the first fit."""
        if self.dur_mu is None or self.dur_sigma is None:
            return {}
        mu_lo, mu_hi, sig_hi = self._duration_box()
        tol = DUAL_DURATION_BOUND_TOL
        return {
            int(k): bool(abs(float(self.dur_mu[k]) - mu_lo) <= tol or abs(float(self.dur_mu[k]) - mu_hi) <= tol or abs(float(self.dur_sigma[k]) - sig_hi) <= tol)
            for k in range(self.K)
        }

    def _duration_box(self) -> tuple[float, float, float]:
        """(mu lower limit, mu upper limit, sigma upper limit) of the duration box (WP6 item 3 part 2)."""
        return (DUAL_DURATION_MU_MIN, float(np.log(DUAL_DURATION_MU_MAX_DMAX_FACTOR * self.Dmax)), DUAL_DURATION_SIGMA_MAX)

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError("Model has not been fitted yet — call .fit() first")

    # ---------- Initialization ----------

    def _init_params(self, X: np.ndarray) -> None:
        """Initialize all fast+slow state-space parameters."""
        K, F, L = self.K, self.F, self.L
        
        # Slow initial distribution and transition matrix
        self.pi = np.full(K, 1.0 / K)
        A_raw = self.rng.dirichlet(np.ones(K - 1) * 2.0, size=K) if K > 1 else np.ones((1, 1))
        self.A_slow = np.zeros((K, K))
        for k in range(K):
            others = [j for j in range(K) if j != k]
            for idx, j in enumerate(others):
                self.A_slow[k, j] = A_raw[k, idx] if K > 1 else 1.0

        # Duration parameters
        self.dur_mu = np.full(K, np.log(10.0)) + self.rng.normal(scale=0.2, size=K)
        self.dur_sigma = np.full(K, 0.5) + self.rng.uniform(0.0, 0.1, size=K)
        self.duration_optimizer_success_ = {k: True for k in range(K)}
        self.duration_fallback_used_ = False

        # Fast continuous state dynamics
        # Stable AR(1) dynamics with slight persistence (A ~ 0.8 * I)
        self.A_fast = np.zeros((K, L, L))
        self.b_fast = np.zeros((K, L))
        self.Q_fast = np.zeros((K, L, L))
        self.m0 = np.zeros((K, L))
        self.P0 = np.zeros((K, L, L))

        for k in range(K):
            # Diagonally dominant stable transition
            self.A_fast[k] = 0.7 * np.eye(L) + self.rng.normal(scale=0.05, size=(L, L))
            self.b_fast[k] = self.rng.normal(scale=0.1, size=L)
            self.Q_fast[k] = 0.5 * np.eye(L)
            self.m0[k] = np.zeros(L)
            self.P0[k] = 1.0 * np.eye(L)

        # Observation emission parameters
        # C maps L latent dims to F observed features
        self.C_obs = np.zeros((K, F, L))
        self.d_obs = np.zeros((K, F))
        self.R_obs = np.zeros((K, F))

        # Initialize observation mean/variance from data
        feat_means = np.nanmean(X, axis=0) if not np.isnan(X).all() else np.zeros(F)
        feat_vars = np.nanvar(X, axis=0) if not np.isnan(X).all() else np.ones(F)
        feat_means = np.nan_to_num(feat_means, nan=0.0)
        feat_vars = np.nan_to_num(feat_vars, nan=1.0)
        feat_vars = np.clip(feat_vars, 1e-3, None)

        for k in range(K):
            # Random orthogonal-like loading matrix
            rand_proj = self.rng.normal(size=(F, L))
            q, _ = np.linalg.qr(rand_proj)
            self.C_obs[k] = q * np.sqrt(feat_vars)[:, None]
            self.d_obs[k] = feat_means + self.rng.normal(scale=0.5 * np.sqrt(feat_vars), size=F)
            self.R_obs[k] = feat_vars * 0.5

    def _duration_logpmf(self) -> np.ndarray:
        if self.dur_sigma is None or self.dur_mu is None:
            raise NotFittedError("Duration parameters are not initialized")
        d = np.arange(1, self.Dmax + 1)
        logpmf = np.zeros((self.K, self.Dmax))
        for k in range(self.K):
            pdf = lognorm.pdf(d, s=self.dur_sigma[k], scale=np.exp(self.dur_mu[k]))
            pdf = np.clip(pdf, 1e-300, None)
            pdf = pdf / pdf.sum()
            logpmf[k] = np.log(pdf)
        return logpmf

    # ---------- Kim Filter (GPB2 Collapse) & Semi-Markov Integration ----------

    def _forward_pass(self, X: np.ndarray) -> ForwardPassResult:
        T, K, L, D = X.shape[0], self.K, self.L, self.Dmax
        dur_logpmf = self._duration_logpmf()
        logA = np.log(np.clip(self.A_slow, 1e-300, None))
        log_pi = np.log(np.clip(self.pi, 1e-300, None))

        m_filt = np.zeros((T, K, L))
        P_filt = np.zeros((T, K, L, L))
        m_pred = np.zeros((T, K, L))
        P_pred = np.zeros((T, K, L, L))
        emis_loglik = np.zeros((T, K))
        ll_continuation = np.full((T, K), NEG_INF)
        ll_entry = np.full((T, K), NEG_INF)
        ll_branch = np.full((T, K, K), NEG_INF)
        J_cont_input_Ppred = np.zeros((T, K, L, L))

        alpha = np.full((T, K, D), NEG_INF)
        entry_mass = np.full((T, K, D), NEG_INF)

        # t = 0: single source per regime, no collapse needed
        for k in range(K):
            m_pred[0, k], P_pred[0, k] = self.m0[k], self.P0[k]
            m_up, P_up, ll_k = self._kalman_update_regime(X[0], self.m0[k], self.P0[k], k)
            m_filt[0, k], P_filt[0, k], emis_loglik[0, k] = m_up, P_up, ll_k
            # t=0 is inherently an entry; ll_entry carries the single-source likelihood so
            # downstream code has a uniform (T,K) contract. ll_continuation stays NEG_INF.
            ll_entry[0, k] = ll_k
            J_cont_input_Ppred[0, k] = self.P0[k]

        entry_mass[0] = log_pi[:, None] + dur_logpmf
        alpha[0] = entry_mass[0] + emis_loglik[0][:, None]

        for t in range(1, T):
            cont = np.full((K, D), NEG_INF)
            cont[:, : D - 1] = alpha[t - 1, :, 1:D]
            completing = alpha[t - 1, :, 0]  # (K,) P(k_{t-1}=i, r_{t-1}=1, y_1:t-1)

            # NOTE (a): entry_mass is dur_logpmf alone for t >= 1; the
            # transition prior is folded into ll_entry through the likelihood-weighted
            # marginalization over predecessor i below. Adding trans_in here as well
            # would double count it.
            entry_mass[t] = dur_logpmf

            for j in range(K):
                # continuation branch: single deterministic source i = j
                m_pred_cont = self.A_fast[j] @ m_filt[t - 1, j] + self.b_fast[j]
                P_pred_cont = self.A_fast[j] @ P_filt[t - 1, j] @ self.A_fast[j].T + self.Q_fast[j]
                m_up_cont, P_up_cont, ll_cont = self._kalman_update_regime(X[t], m_pred_cont, P_pred_cont, j)
                ll_continuation[t, j] = ll_cont
                J_cont_input_Ppred[t, j] = P_pred_cont

                # entry branch: GPB2 collapse over predecessors i (i=j vanishes via logA)
                log_w_raw = np.full(K, NEG_INF)
                m_up_ij = np.zeros((K, L))
                P_up_ij = np.zeros((K, L, L))
                for i in range(K):
                    if not np.isfinite(logA[i, j]) or logA[i, j] <= NEG_INF / 2 or i == j:
                        continue
                    m_pred_ij = self.A_fast[j] @ m_filt[t - 1, i] + self.b_fast[j]
                    P_pred_ij = self.A_fast[j] @ P_filt[t - 1, i] @ self.A_fast[j].T + self.Q_fast[j]
                    m_up, P_up, ll_ij = self._kalman_update_regime(X[t], m_pred_ij, P_pred_ij, j)
                    m_up_ij[i], P_up_ij[i] = m_up, P_up
                    log_w_raw[i] = completing[i] + logA[i, j] + ll_ij
                    ll_branch[t, i, j] = ll_ij

                ll_entry[t, j] = logsumexp(log_w_raw)
                w = _safe_softmax(log_w_raw)
                m_up_entry, P_up_entry = _mixture_moments(m_up_ij, P_up_ij, w)

                alpha[t, j, :] = np.logaddexp(
                    cont[j, :] + ll_continuation[t, j],
                    entry_mass[t, j, :] + ll_entry[t, j],
                )

                # final collapse: continuation vs entry into m_filt[t, j]
                cont_mass_r = logsumexp(cont[j, :]) if np.any(cont[j, :] > NEG_INF / 2) else NEG_INF
                log_mass_cont = cont_mass_r + ll_continuation[t, j] if cont_mass_r > NEG_INF / 2 else NEG_INF
                log_mass_entry = ll_entry[t, j]
                pi_cont, pi_entry = _safe_softmax(np.array([log_mass_cont, log_mass_entry]))

                means = np.stack([m_up_cont, m_up_entry])
                covs = np.stack([P_up_cont, P_up_entry])
                m_filt[t, j], P_filt[t, j] = _mixture_moments(means, covs, np.array([pi_cont, pi_entry]))

                entry_m_pred_by_i = np.array([self.A_fast[j] @ m_filt[t - 1, i] + self.b_fast[j] for i in range(K)])
                m_pred[t, j] = pi_cont * m_pred_cont + pi_entry * np.sum(w[:, None] * entry_m_pred_by_i, axis=0)
                P_pred[t, j] = pi_cont * P_pred_cont + pi_entry * np.sum(
                    w[:, None, None] * np.array([
                        self.A_fast[j] @ P_filt[t - 1, i] @ self.A_fast[j].T + self.Q_fast[j] for i in range(K)
                    ]), axis=0)

                emis_loglik[t, j] = pi_cont * ll_continuation[t, j] + pi_entry * ll_entry[t, j]  # diagnostic only

        log_likelihood = float(logsumexp(alpha[-1]))
        regime_post_filtered = np.zeros((T, K))
        for t in range(T):
            alpha_t_k = logsumexp(alpha[t], axis=1)
            regime_post_filtered[t] = np.exp(alpha_t_k - logsumexp(alpha_t_k))

        return ForwardPassResult(
            regime_post_filtered=regime_post_filtered,
            entry_mass=entry_mass,
            m_filt=m_filt, P_filt=P_filt, m_pred=m_pred, P_pred=P_pred,
            emis_loglik=emis_loglik,
            ll_continuation=ll_continuation, ll_entry=ll_entry, ll_branch=ll_branch,
            alpha=alpha,
            J_cont=J_cont_input_Ppred,
            log_likelihood=log_likelihood,
        )

    def _kalman_update_regime(
        self,
        yt: np.ndarray,
        m_prior: np.ndarray,
        P_prior: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Kalman measurement update for regime k with NaN marginalization.

        If the innovation covariance cannot be Cholesky-factorised the function returns the
        prior unchanged with log-density -1e6 (a magic number, not a likelihood). That
        behaviour is kept, but every firing is now COUNTED in `self.cholesky_fallback_count_`
        (ledger A7 / decision D3 interim); `fit()` turns the count into `fit_degraded_`.
        Returned values are identical to the previous version in every case.
        """
        L, F = self.L, self.F
        is_obs = ~np.isnan(yt)
        if not np.any(is_obs):
            # All features missing: no update, zero log-density
            return m_prior.copy(), P_prior.copy(), 0.0

        y_obs = yt[is_obs]
        C_obs = self.C_obs[k][is_obs, :]
        d_obs = self.d_obs[k][is_obs]
        R_diag = self.R_obs[k][is_obs]

        # Innovation
        y_pred = C_obs @ m_prior + d_obs
        nu = y_obs - y_pred

        # Innovation covariance: S = C P C^T + R
        S = C_obs @ P_prior @ C_obs.T + np.diag(R_diag)
        S = 0.5 * (S + S.T) + 1e-6 * np.eye(len(y_obs))

        try:
            chol = np.linalg.cholesky(S)
            # Kalman gain K = P C^T S^{-1}
            # Solve S x = C P -> x = S^{-1} C P
            CP = C_obs @ P_prior
            # K_gain = (S^{-1} CP)^T
            solve_val = np.linalg.solve(chol, CP)
            K_gain = np.linalg.solve(chol.T, solve_val).T

            # Updated state and covariance
            m_up = m_prior + K_gain @ nu
            I_KC = np.eye(L) - K_gain @ C_obs
            P_up = I_KC @ P_prior @ I_KC.T + K_gain @ np.diag(R_diag) @ K_gain.T
            P_up = 0.5 * (P_up + P_up.T) + 1e-6 * np.eye(L)

            # Log-likelihood of innovation
            log_det = 2.0 * np.sum(np.log(np.diag(chol)))
            inv_nu = np.linalg.solve(chol, nu)
            quad = float(np.sum(inv_nu**2))
            ll = -0.5 * (len(y_obs) * np.log(2.0 * np.pi) + log_det + quad)
            return m_up, P_up, ll
        except np.linalg.LinAlgError:
            self.cholesky_fallback_count_ += 1
            return m_prior.copy(), P_prior.copy(), -1e6

    # ---------- Discrete Backward Pass & Kim Switching Smoother ----------

    def _discrete_backward(self, fwd: ForwardPassResult) -> BackwardPassResult:
        T, K, D = fwd.alpha.shape
        dur_logpmf = self._duration_logpmf()
        logA = np.log(np.clip(self.A_slow, 1e-300, None))
        alpha = fwd.alpha
        ll_continuation, ll_entry = fwd.ll_continuation, fwd.ll_entry
        log_likelihood = fwd.log_likelihood

        # NOTE (c): reusing the predecessor-marginalized ll_entry in beta would make alpha
        # and beta non-dual (checked against brute force on a pure-discrete mock). beta needs
        # the source-specific branch likelihood, which the forward pass already computes
        # before collapsing: ll_branch[t, i, j].
        ll_branch = fwd.ll_branch

        beta = np.full((T, K, D), NEG_INF)
        beta[-1] = 0.0
        future_entry_val = np.full((T, K), NEG_INF)

        for t in range(T - 2, -1, -1):
            inner = logsumexp(beta[t + 1] + dur_logpmf, axis=1)  # (K,) per destination j
            future_entry_val[t + 1] = inner

            cont_term = np.full((K, D), NEG_INF)
            cont_term[:, 1:D] = beta[t + 1, :, : D - 1] + ll_continuation[t + 1][:, None]

            fut_kj = logA + ll_branch[t + 1] + inner[None, :]  # [k,j]
            trans_out = logsumexp(fut_kj, axis=1)              # (K,) per source k
            end_term = np.full((K, D), NEG_INF)
            end_term[:, 0] = trans_out

            beta[t] = np.logaddexp(cont_term, end_term)

        # Per-timestep normalizer Z_t = logsumexp(alpha[t] + beta[t]). With a correct dual
        # this equals log_likelihood for every t; normalizing per t keeps gamma a proper
        # distribution regardless. entry_posterior uses the same normalizer.
        gamma = np.zeros((T, K, D))
        entry_posterior = np.zeros((T, K, D))
        for t in range(T):
            Z_t = logsumexp((alpha[t] + beta[t]).ravel())
            gamma[t] = np.exp(np.clip(alpha[t] + beta[t] - Z_t, -700, 0))
            ep_log = fwd.entry_mass[t] + beta[t] + ll_entry[t][:, None] - Z_t
            entry_posterior[t] = np.exp(np.clip(ep_log, -700, 0))
        regime_posterior_smoothed = gamma.sum(axis=2)

        # xi[t,i,j] = P(k_t=i, r_t=1, k_{t+1}=j | Y): uses the source-specific ll_branch
        # (i is fixed here, so the i-marginalized ll_entry would be wrong).
        xi_counts = np.zeros((K, K))
        xi_t = np.zeros((max(T - 1, 0), K, K))
        for t in range(T - 1):
            completing_t = alpha[t, :, 0]
            if np.all(completing_t <= NEG_INF / 2) or np.all(future_entry_val[t + 1] <= NEG_INF / 2):
                continue
            xi_log_t = (completing_t[:, None] + logA + ll_branch[t + 1]
                        + future_entry_val[t + 1][None, :] - log_likelihood)
            xi_t[t] = np.exp(np.clip(xi_log_t, -700, 0))
            xi_counts += xi_t[t]

        return BackwardPassResult(
            beta=beta,
            regime_posterior_smoothed=regime_posterior_smoothed,
            gamma=gamma,
            entry_posterior=entry_posterior,
            xi_counts=xi_counts,
            xi_t=xi_t,
            future_entry_val=future_entry_val,
        )

    # NOTE (d): the Kim/GPB2 smoother uses the already-collapsed smoothed_m_k[t+1, j] in every
    # (k, j) correction, which is itself a moment match over predecessors. Its continuous
    # moments are therefore approximate even at T = 2 whenever regimes have distinct
    # dynamics (exact when the collapse is lossless, i.e. identical continuous parameters).
    def _switching_smoother(self, fwd: ForwardPassResult, bwd: BackwardPassResult) -> SmootherResult:
        T, K, L = fwd.m_filt.shape
        D = self.Dmax
        m_filt, P_filt = fwd.m_filt, fwd.P_filt
        xi_t, gamma = bwd.xi_t, bwd.gamma
        regime_posterior_smoothed = bwd.regime_posterior_smoothed

        smoothed_m_k = np.zeros((T, K, L))
        smoothed_P_k = np.zeros((T, K, L, L))
        n_pair = max(T - 1, 0)
        pair_m = np.zeros((n_pair, K, K, L))
        pair_P = np.zeros((n_pair, K, K, L, L))
        pair_J = np.zeros((n_pair, K, K, L, L))
        pair_w = np.zeros((n_pair, K, K))

        smoothed_m_k[-1] = m_filt[-1]
        smoothed_P_k[-1] = P_filt[-1]

        for t in range(T - 2, -1, -1):
            mass_continue = gamma[t, :, 1:D].sum(axis=1)  # P(r_t > 1, k_t=k | Y)

            for k in range(K):
                m_smooth_per_j = np.zeros((K, L))
                P_smooth_per_j = np.zeros((K, L, L))
                raw_weight = np.zeros(K)

                for j in range(K):
                    P_pred_kj = self.A_fast[j] @ P_filt[t, k] @ self.A_fast[j].T + self.Q_fast[j]
                    P_pred_kj = P_pred_kj + 1e-9 * np.eye(L)
                    J_kj = P_filt[t, k] @ self.A_fast[j].T @ np.linalg.inv(P_pred_kj)
                    m_pred_kj = self.A_fast[j] @ m_filt[t, k] + self.b_fast[j]
                    m_smooth_per_j[j] = m_filt[t, k] + J_kj @ (smoothed_m_k[t + 1, j] - m_pred_kj)
                    P_smooth_per_j[j] = P_filt[t, k] + J_kj @ (smoothed_P_k[t + 1, j] - P_pred_kj) @ J_kj.T

                    # joint posterior mass of the pair (k_t=k, k_{t+1}=j): the continuation
                    # mass on the diagonal (r_t > 1 forces k_{t+1} = k), the pairwise
                    # transition marginal off it
                    raw_weight[j] = mass_continue[k] if j == k else xi_t[t, k, j]

                    # cache the pair-conditioned branch quantities for the M-step
                    pair_m[t, k, j] = m_smooth_per_j[j]
                    pair_P[t, k, j] = P_smooth_per_j[j]
                    pair_J[t, k, j] = J_kj
                    pair_w[t, k, j] = raw_weight[j]

                total = raw_weight.sum()
                if total <= 1e-12:
                    # ~zero smoothed mass for k at t: keep the filtered belief (documented
                    # fallback; avoids dividing by ~0)
                    smoothed_m_k[t, k] = m_filt[t, k]
                    smoothed_P_k[t, k] = P_filt[t, k]
                    continue

                m_out, P_out = _mixture_moments(m_smooth_per_j, P_smooth_per_j, raw_weight / total)
                smoothed_m_k[t, k] = m_out
                smoothed_P_k[t, k] = P_out

        # aggregate with the SMOOTHED marginal (the old code used the filtered one)
        smoothed_m = np.zeros((T, L))
        smoothed_P = np.zeros((T, L, L))
        for t in range(T):
            smoothed_m[t], smoothed_P[t] = _mixture_moments(
                smoothed_m_k[t], smoothed_P_k[t], regime_posterior_smoothed[t])

        return SmootherResult(
            smoothed_m_k=smoothed_m_k, smoothed_P_k=smoothed_P_k,
            smoothed_m=smoothed_m, smoothed_P=smoothed_P,
            pair_m=pair_m, pair_P=pair_P, pair_J=pair_J, pair_w=pair_w,
        )

    # ---------- EM M-Step ----------

    def _update_discrete_params(self, bwd: BackwardPassResult):
        K = self.K
        N = bwd.xi_counts
        A_new = np.zeros((K, K))
        self.transition_fallback_rows_ = set()
        for i in range(K):
            row_total = N[i].sum()
            if row_total <= 1e-10:
                A_new[i] = self.A_slow[i]  # explicit fallback: keep previous row
                self.transition_fallback_rows_.add(i)
            else:
                A_new[i] = N[i] / row_total
                A_new[i, i] = 0.0
        self.A_slow = A_new

        # duration parameters: weighted log-normal MLE through the SAME truncated and
        # renormalized PMF used everywhere else
        d_grid = np.arange(1, self.Dmax + 1)
        new_mu = self.dur_mu.copy()
        new_sigma = self.dur_sigma.copy()
        for k in range(K):
            N_j_d = bwd.entry_posterior[:, k, :].sum(axis=0)
            if N_j_d.sum() <= 1e-10:
                self.duration_optimizer_success_[k] = False
                self.duration_fallback_used_ = True
                continue

            def neg_expected_ll(params, N_j_d=N_j_d, d_grid=d_grid):
                mu, log_sigma = params
                sigma = np.exp(log_sigma)
                if sigma <= 1e-6 or sigma > 10 or not np.isfinite(mu):
                    return 1e12
                pdf = lognorm.pdf(d_grid, s=sigma, scale=np.exp(mu))
                pdf = np.clip(pdf, 1e-300, None)
                pdf = pdf / pdf.sum()
                return -float(np.sum(N_j_d * np.log(pdf)))

            x0 = [self.dur_mu[k], np.log(max(self.dur_sigma[k], 1e-3))]
            try:
                res = minimize(neg_expected_ll, x0, method="Nelder-Mead")
                mu_hat, sigma_hat = res.x[0], float(np.exp(res.x[1]))
                if not (res.success and np.isfinite(mu_hat) and np.isfinite(sigma_hat) and sigma_hat > 0):
                    raise RuntimeError("optimizer did not converge to a valid result")
                # WP6 item 3 part 2 (D1, event-triggered): a result the old rule would accept but that lies outside the duration box is
                # redone from the previous iterate inside the box. A result inside the box is accepted exactly as before.
                mu_lo, mu_hi, sig_hi = self._duration_box()
                if mu_hat < mu_lo or mu_hat > mu_hi or sigma_hat > sig_hi:
                    mu_hat, sigma_hat = self._boxed_duration_update(k, N_j_d, d_grid, neg_expected_ll)
                new_mu[k], new_sigma[k] = mu_hat, sigma_hat
                self.duration_optimizer_success_[k] = True
            except Exception:
                new_mu[k], new_sigma[k] = self.dur_mu[k], self.dur_sigma[k]
                self.duration_optimizer_success_[k] = False
                self.duration_fallback_used_ = True

        self.dur_mu, self.dur_sigma = new_mu, new_sigma

        self.pi = np.clip(bwd.regime_posterior_smoothed[0], 1e-4, None)
        self.pi = self.pi / self.pi.sum()

    def _boxed_duration_update(self, k: int, N_j_d: np.ndarray, d_grid: np.ndarray, neg_expected_ll) -> tuple[float, float]:
        """Redo the duration update of regime k inside the duration box (WP6 item 3 part 2, D1).

        Starts at the previous iterate clipped into the box and minimises the same objective with a penalty (1e12) outside the box. Returns
        (mu, sigma) inside the box; raises RuntimeError when the optimiser does not succeed, so the caller takes the old fallback path
        (previous iterate kept, `duration_optimizer_success_[k] = False`). Records the application in `duration_box_applied_`.
        """
        mu_lo, mu_hi, sig_hi = self._duration_box()
        from_mu, from_sigma = float(self.dur_mu[k]), float(self.dur_sigma[k])

        def boxed(params):
            mu, log_sigma = params
            if not np.isfinite(mu) or not np.isfinite(log_sigma) or mu < mu_lo or mu > mu_hi or np.exp(log_sigma) > sig_hi:
                return 1e12
            return neg_expected_ll(params)

        x0 = [float(np.clip(from_mu, mu_lo, mu_hi)), float(np.log(np.clip(max(from_sigma, 1e-3), 1e-3, sig_hi)))]
        res = minimize(boxed, x0, method="Nelder-Mead")
        mu_hat, sigma_hat = float(res.x[0]), float(np.exp(res.x[1]))
        if not (res.success and np.isfinite(mu_hat) and np.isfinite(sigma_hat) and sigma_hat > 0
                and mu_lo <= mu_hat <= mu_hi and sigma_hat <= sig_hi):
            raise RuntimeError("boxed duration update did not converge to a valid result")
        self.duration_box_applied_.append({
            "iteration": int(self._em_iteration), "regime": int(k), "from_mu": from_mu, "from_sigma": from_sigma,
            "to_mu": mu_hat, "to_sigma": sigma_hat,
        })
        return mu_hat, sigma_hat

    def _dynamics_sufficient_stats(self, smoother: SmootherResult):
        """Expected sufficient statistics for each regime's fast dynamics
        m_t = A_k m_{t-1} + b_k + noise, applied at every t where k_t = k.

        NOTE (e): the transition into time t+1 under regime k relates m_{t+1}
        to m_t, and the distribution of m_t depends on the regime i it was generated in.
        Within a segment i == k; at a segment start i != k, and E[m_t | k_t=k, Y] is the
        wrong quantity there (it conditions on a regime m_t was probably not generated in).
        So the statistics are accumulated over predecessor regimes i, weighted by the joint
        posterior P(k_t=i, k_{t+1}=k | Y), using the pair-conditioned smoothed moments of
        m_t and the matching lag covariance Cov(m_{t+1}, m_t) = J_{ik} @ P^s_{t+1,k}.

        Returns G (K,L+1,L+1), H (K,L+1,L), S (K,L,L), W (K,) such that the weighted
        least-squares problem for Theta = [A | b] is  G Theta = H  and the residual
        second-moment sum is  S - Theta^T H.
        """
        K, L = self.K, self.L
        n_pair = smoother.pair_w.shape[0]
        G = np.zeros((K, L + 1, L + 1))
        H = np.zeros((K, L + 1, L))
        S = np.zeros((K, L, L))
        W = np.zeros(K)
        for t in range(n_pair):
            for k in range(K):
                m_cur = smoother.smoothed_m_k[t + 1, k]
                E_cc = smoother.smoothed_P_k[t + 1, k] + np.outer(m_cur, m_cur)
                for i in range(K):
                    w = smoother.pair_w[t, i, k]
                    if w <= 1e-12:
                        continue
                    m_prev = smoother.pair_m[t, i, k]
                    E_pp = smoother.pair_P[t, i, k] + np.outer(m_prev, m_prev)
                    # E[m_{t+1} m_t^T] = Cov(m_{t+1}, m_t) + outer(m_{t+1}, m_t)
                    E_cp = smoother.pair_J[t, i, k] @ smoother.smoothed_P_k[t + 1, k] + np.outer(m_cur, m_prev)

                    W[k] += w
                    G[k, :L, :L] += w * E_pp
                    G[k, :L, L] += w * m_prev
                    G[k, L, :L] += w * m_prev
                    G[k, L, L] += w
                    H[k, :L, :] += w * E_cp.T
                    H[k, L, :] += w * m_cur
                    S[k] += w * E_cc
        return G, H, S, W

    def _m_step_continuous(self, X: np.ndarray, bwd: BackwardPassResult, smoother: SmootherResult):
        T, K, L, F = X.shape[0], self.K, self.L, self.F
        smoothed_m_k, smoothed_P_k = smoother.smoothed_m_k, smoother.smoothed_P_k
        gamma_tk = bwd.regime_posterior_smoothed

        # observation model: only m_t is involved, so the regime-k moments at time t
        # (conditioned on k_t = k) weighted by gamma_t[k] are the right statistics
        Ex = smoothed_m_k
        Exx = np.zeros((T, K, L, L))
        for t in range(T):
            for k in range(K):
                Exx[t, k] = smoothed_P_k[t, k] + np.outer(smoothed_m_k[t, k], smoothed_m_k[t, k])

        G_dyn, H_dyn, S_dyn, W_dyn = self._dynamics_sufficient_stats(smoother)

        A_new, b_new, Q_new = self.A_fast.copy(), self.b_fast.copy(), self.Q_fast.copy()
        for k in range(K):
            G, H, sum_Exx_t, W = G_dyn[k], H_dyn[k], S_dyn[k], W_dyn[k]

            if W <= 1e-8:
                continue  # no evidence for this regime's dynamics this iteration
            theta = np.linalg.solve(G + 1e-6 * np.eye(L + 1), H)  # (L+1, L)
            A_k, b_k = theta[:L, :].T, theta[L, :]

            # weighted-LS residual identity: at the optimum the residual is sum_Exx - Theta^T H
            Q_k = (sum_Exx_t - theta.T @ H) / W
            Q_k = 0.5 * (Q_k + Q_k.T)
            eigvals, eigvecs = np.linalg.eigh(Q_k)
            Q_k = eigvecs @ np.diag(np.clip(eigvals, 1e-6, None)) @ eigvecs.T

            A_new[k], b_new[k], Q_new[k] = A_k, b_k, Q_k

        self.A_fast, self.b_fast, self.Q_fast = A_new, b_new, Q_new

        C_new, d_new, R_new = self.C_obs.copy(), self.d_obs.copy(), self.R_obs.copy()
        for k in range(K):
            for f in range(F):
                valid = ~np.isnan(X[:, f])
                G = np.zeros((L + 1, L + 1))
                H = np.zeros((L + 1, 1))
                sum_y2 = 0.0
                W = 0.0
                for t in range(T):
                    if not valid[t]:
                        continue
                    w = gamma_tk[t, k]
                    if w <= 1e-12:
                        continue
                    W += w
                    y = X[t, f]
                    G[:L, :L] += w * Exx[t, k]
                    G[:L, L] += w * Ex[t, k]
                    G[L, :L] += w * Ex[t, k]
                    G[L, L] += w
                    H[:L, 0] += w * Ex[t, k] * y
                    H[L, 0] += w * y
                    sum_y2 += w * y * y

                if W <= 1e-8:
                    continue
                theta_f = np.linalg.solve(G + 1e-6 * np.eye(L + 1), H)
                C_new[k, f, :] = theta_f[:L, 0]
                d_new[k, f] = theta_f[L, 0]
                R_new[k, f] = max((sum_y2 - (theta_f.T @ H).item()) / W, 1e-6)

        self.C_obs, self.d_obs, self.R_obs = C_new, d_new, R_new

    # ---------- Fitting & Generalized EM ----------

    def fit(
        self,
        X: np.ndarray,
        n_iter: int = 50,
        tol: float = 1e-4,
        verbose: bool = False,
        timestamps: np.ndarray | None = None,
    ) -> "DualTimescaleHSSM":
        """Fit dual-timescale model using generalized EM with backtracking rollback.

        Besides the fit itself this records, for the call that just ran (ledger A7 / N21):
          fit_cholesky_fallbacks_ : how many times the -1e6 Cholesky fallback fired in the loop
          fit_degraded_           : True iff that number is > 0 (the likelihood contains the magic constant)
          fit_exceptions_         : one dict per exception caught inside the loop
        These are set even if an exception that is NOT caught here propagates out of the loop.
        """
        n_fallbacks_before = self.cholesky_fallback_count_
        self.fit_exceptions_ = []
        self.fit_cholesky_fallbacks_ = 0
        self.fit_degraded_ = False
        try:
            return self._fit_impl(X, n_iter=n_iter, tol=tol, verbose=verbose, timestamps=timestamps)
        finally:
            self.fit_cholesky_fallbacks_ = self.cholesky_fallback_count_ - n_fallbacks_before
            self.fit_degraded_ = self.fit_cholesky_fallbacks_ > 0

    def _record_fit_exception(self, exc: BaseException, stage: str, iteration: int) -> None:
        """Remember an exception that fit() caught, instead of discarding it (ledger N21).

        The `except (LinAlgError, ValueError)` clauses in the EM loop also catch ValueErrors that
        come from programming errors (for example a shape mismatch). Recording the type, message
        and raising location keeps those distinguishable from genuine numerical failures.
        """
        frames = traceback.extract_tb(exc.__traceback__)
        last = frames[-1] if frames else None
        self.fit_exceptions_.append({
            "iteration": int(iteration),
            "stage": stage,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "location": f"{os.path.basename(last.filename)}:{last.lineno} in {last.name}" if last else None,
        })

    def _fit_impl(
        self,
        X: np.ndarray,
        n_iter: int = 50,
        tol: float = 1e-4,
        verbose: bool = False,
        timestamps: np.ndarray | None = None,
    ) -> "DualTimescaleHSSM":
        self.fit_state_ = FitState.RUNNING
        self.log_likelihood_history_ = []
        self._n_observations = len(X)
        self.convergence_criterion_ = None
        self.window_gain_at_cap_ = None
        self.duration_box_applied_ = []
        self._em_iteration = 0

        if timestamps is not None:
            self.duration_unit = "calendar_days"
        else:
            self.duration_unit = "sessions"

        self._init_params(X)

        prev_params = None
        prev_ll = None

        for it in range(n_iter):
            self._em_iteration = it
            # Save snapshot for monotonicity rollback. It covers every array the M-step
            # changes (pi, A_slow, dur_mu, dur_sigma, A_fast, b_fast, Q_fast, C_obs, d_obs,
            # R_obs) AND the duration/transition status fields, so a rollback restores both.
            snapshot = self._snapshot_params()

            # E-step: two-level GPB2 forward pass, discrete backward pass, switching smoother
            try:
                fwd = self._forward_pass(X)
                bwd = self._discrete_backward(fwd)
                smoother = self._switching_smoother(fwd, bwd)
            except (np.linalg.LinAlgError, ValueError) as exc:
                self._record_fit_exception(exc, stage="E-step", iteration=it)
                if prev_params is not None:
                    self._restore_params(prev_params)
                self.log_likelihood_ = prev_ll
                self.fit_state_ = FitState.NUMERICAL_FAILURE
                self.n_iter_ = it
                self._is_fitted = True
                return self

            ll = fwd.log_likelihood
            self.log_likelihood_ = ll
            if ll is None or np.isnan(ll) or np.isinf(ll):
                self.fit_state_ = FitState.NUMERICAL_FAILURE
                self.n_iter_ = it
                self._is_fitted = True
                return self

            self.log_likelihood_history_.append(ll)

            if verbose and it % 5 == 0:
                print(f"  [DualTimescaleHSSM] iter {it}: log-likelihood = {ll:.3f}")

            # Convergence check
            if it > 0 and prev_ll is not None and abs(ll - prev_ll) < tol:
                self.fit_state_ = FitState.CONVERGED
                self.convergence_criterion_ = "abs_tol"
                self.n_iter_ = it
                break

            # Generalized EM Monotonicity Guard (Backtracking)
            if it > 0 and prev_ll is not None and ll < prev_ll - 1e-3:
                if prev_params is not None:
                    warnings.warn(
                        f"[DualTimescaleHSSM] Monotonicity violation at iter {it}: "
                        f"LL dropped from {prev_ll:.4f} to {ll:.4f}. Backtracking.",
                        UserWarning,
                        stacklevel=3,  # fit() -> _fit_impl(): attribute the warning to the caller of fit()
                    )
                    self._restore_params(prev_params)
                    self.log_likelihood_ = prev_ll
                    self.fit_state_ = FitState.REJECTED_UPDATE
                    self.n_iter_ = it
                    break

            prev_params = snapshot
            prev_ll = ll

            # M-step: discrete parameters (A_slow, durations, pi), then continuous
            # linear-Gaussian parameters from pair-conditioned sufficient statistics
            try:
                self._update_discrete_params(bwd)
                self._m_step_continuous(X, bwd, smoother)
            except (np.linalg.LinAlgError, ValueError) as exc:
                self._record_fit_exception(exc, stage="M-step", iteration=it)
                self._restore_params(snapshot)
                self.fit_state_ = FitState.NUMERICAL_FAILURE
                self.n_iter_ = it
                self._is_fitted = True
                return self

        if self.fit_state_ == FitState.RUNNING:
            # The iteration limit was reached without the |ll - prev| < tol rule firing (WP6 item 3 part 2, C2). The fit is certified CONVERGED
            # only if the log-likelihood gained less than DUAL_CAP_CERT_GAIN_PER_ROW nats per row over the last DUAL_CAP_WINDOW iterations;
            # otherwise it stays MAX_ITER_REACHED (and the result is withheld by the non-convergence rule).
            hist = self.log_likelihood_history_
            self.n_iter_ = len(hist)
            if len(hist) > DUAL_CAP_WINDOW:
                self.window_gain_at_cap_ = float(hist[-1] - hist[-1 - DUAL_CAP_WINDOW])
                if self.window_gain_at_cap_ < DUAL_CAP_CERT_GAIN_PER_ROW * self._n_observations:
                    self.fit_state_ = FitState.CONVERGED
                    self.convergence_criterion_ = "window_gain_at_cap"
            if self.fit_state_ == FitState.RUNNING:
                self.fit_state_ = FitState.MAX_ITER_REACHED

        self._is_fitted = True
        return self

    def _snapshot_params(self):
        """Copy every parameter the EM loop can change, plus the status fields that
        _update_discrete_params sets, so a rollback restores both."""
        names = ["pi", "A_slow", "dur_mu", "dur_sigma", "A_fast", "b_fast", "Q_fast",
                 "m0", "P0", "C_obs", "d_obs", "R_obs"]
        snap = {n: np.array(getattr(self, n), copy=True) for n in names}
        snap["duration_optimizer_success_"] = dict(self.duration_optimizer_success_)
        snap["duration_fallback_used_"] = bool(self.duration_fallback_used_)
        snap["transition_fallback_rows_"] = set(self.transition_fallback_rows_)
        return snap

    def _restore_params(self, snapshot):
        for n, v in snapshot.items():
            if isinstance(v, np.ndarray):
                setattr(self, n, np.array(v, copy=True))
            elif isinstance(v, dict):
                setattr(self, n, dict(v))
            elif isinstance(v, set):
                setattr(self, n, set(v))
            else:
                setattr(self, n, v)

    # ---------- Inference & Estimation ----------

    def estimate_states(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Estimate fast continuous trajectory and slow regime assignments.

        Returns:
          m_t: (T, L) continuous state trajectory posterior mean (switching-smoother output)
          m_t_std: (T, L) continuous state posterior standard error (uncertainty)
          p_t: (T,) MAP discrete regime sequence, from the FILTERED regime posterior
          p_t_post: (T, K) FILTERED regime posterior probabilities (not the smoothed ones)

        INFORMATION SETS (ledger A6, owner decision D2). The four outputs do NOT use the same data:
          * m_t and m_t_std are SMOOTHED: row t uses EVERY observation in X, including those after t.
          * p_t and p_t_post are FILTERED: row t uses only X[0..t] (causal).
        Consequence: appending observations to X can change m_t / m_t_std of EARLIER rows but never
        changes p_t / p_t_post of earlier rows (tests/test_dual_model_estimate_states_information_sets.py).
        Do not display m_t beside p_t as if they came from one information set.
        This method does not modify self.log_likelihood_ or any parameter.
        """
        self._require_fitted()
        fwd = self._forward_pass(X)
        bwd = self._discrete_backward(fwd)
        smoother = self._switching_smoother(fwd, bwd)

        regime_post = fwd.regime_post_filtered
        p_t = np.argmax(regime_post, axis=1)

        smoothed_m, smoothed_P = smoother.smoothed_m, smoother.smoothed_P

        # Standard error along diagonal of covariance
        m_t_std = np.zeros_like(smoothed_m)
        for t in range(len(smoothed_m)):
            m_t_std[t] = np.sqrt(np.maximum(np.diag(smoothed_P[t]), 1e-8))

        return smoothed_m, m_t_std, p_t, regime_post

    # ---------- Information Criteria ----------

    def n_params(self) -> int:
        self._require_fitted()
        K, F, L = self.K, self.F, self.L
        slow_params = (K - 1) + K * (K - 1) + 2 * K
        fast_params = K * (L * L + L + L * (L + 1) // 2)
        obs_params = K * (F * L + F + F)
        return slow_params + fast_params + obs_params

    def bic(self, n_observations: int | None = None) -> float:
        self._require_fitted()
        if self.log_likelihood_ is None:
            raise InternalStateError("Log likelihood is None for fitted model")
        n_obs = n_observations if n_observations is not None else getattr(self, "_n_observations", 100)
        return -2 * self.log_likelihood_ + self.n_params() * np.log(max(n_obs, 1))

    def aic(self, n_observations: int | None = None) -> float:
        self._require_fitted()
        if self.log_likelihood_ is None:
            raise InternalStateError("Log likelihood is None for fitted model")
        return -2 * self.log_likelihood_ + 2 * self.n_params()
