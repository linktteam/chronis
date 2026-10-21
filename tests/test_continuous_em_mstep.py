"""Negative controls for the continuous EM M-step (sections 7.3 / 7.5).

The K=1 test checks production's M-step against an independent, from-scratch textbook
(non-switching) LDS EM, with K=1 and Dmax >= T so no discrete switching is in play. It is
the most load-bearing test here: if it fails, the other M-step tests are not trustworthy.

The reference is written twice over one code path (`mirror_regularization`):
  - False: pure textbook math, no numerical jitter.
  - True : adds the same small diagonal jitter the UNCHANGED `_kalman_update_regime`
           applies (S += 1e-6 I, P_up += 1e-6 I) and the smoother's 1e-9 I in its gain
           inverse. This isolates formula errors from deliberate regularization.
"""
import numpy as np

from tests.utils.model_factory import make_tiny_model


def _textbook_lds_em_mstep(X, A, b, Q, C, d, R, m0, P0, mirror_regularization=False):
    """Single-regime Kalman filter + RTS smoother + EM M-step. Shares no code with
    backbone/hssm/dual_model.py."""
    T, F = X.shape
    L = A.shape[0]
    eps_S = 1e-6 if mirror_regularization else 0.0
    eps_P = 1e-6 if mirror_regularization else 0.0
    eps_J = 1e-9 if mirror_regularization else 0.0

    m_filt = np.zeros((T, L)); P_filt = np.zeros((T, L, L))
    for t in range(T):
        if t == 0:
            m_p, P_p = m0, P0
        else:
            m_p = A @ m_filt[t - 1] + b
            P_p = A @ P_filt[t - 1] @ A.T + Q
        S = C @ P_p @ C.T + np.diag(R)
        S = 0.5 * (S + S.T) + eps_S * np.eye(F)
        K_gain = P_p @ C.T @ np.linalg.inv(S)
        nu = X[t] - (C @ m_p + d)
        m_filt[t] = m_p + K_gain @ nu
        I_KC = np.eye(L) - K_gain @ C
        P_up = I_KC @ P_p @ I_KC.T + K_gain @ np.diag(R) @ K_gain.T
        P_filt[t] = 0.5 * (P_up + P_up.T) + eps_P * np.eye(L)

    m_smooth = np.zeros((T, L)); P_smooth = np.zeros((T, L, L))
    m_smooth[-1], P_smooth[-1] = m_filt[-1], P_filt[-1]
    P_lag = np.zeros((T, L, L))  # P_lag[t] = Cov(m_t, m_{t-1} | Y), t >= 1
    for t in range(T - 2, -1, -1):
        m_pred_next = A @ m_filt[t] + b
        P_pred_next = A @ P_filt[t] @ A.T + Q + eps_J * np.eye(L)
        J = P_filt[t] @ A.T @ np.linalg.inv(P_pred_next)
        m_smooth[t] = m_filt[t] + J @ (m_smooth[t + 1] - m_pred_next)
        P_smooth[t] = P_filt[t] + J @ (P_smooth[t + 1] - P_pred_next) @ J.T
        P_lag[t + 1] = J @ P_smooth[t + 1]

    Ex = m_smooth
    Exx = np.array([P_smooth[t] + np.outer(m_smooth[t], m_smooth[t]) for t in range(T)])
    Exx_lag = np.array([P_lag[t] + np.outer(m_smooth[t], m_smooth[t - 1]) if t >= 1 else np.zeros((L, L))
                         for t in range(T)])

    G = np.zeros((L + 1, L + 1)); H = np.zeros((L + 1, L)); sum_Exx = np.zeros((L, L))
    for t in range(1, T):
        G[:L, :L] += Exx[t - 1]; G[:L, L] += Ex[t - 1]; G[L, :L] += Ex[t - 1]; G[L, L] += 1
        H[:L, :] += Exx_lag[t].T; H[L, :] += Ex[t]
        sum_Exx += Exx[t]
    theta = np.linalg.solve(G + 1e-6 * np.eye(L + 1), H)
    A_new, b_new = theta[:L, :].T, theta[L, :]
    Q_new = (sum_Exx - theta.T @ H) / (T - 1)
    Q_new = 0.5 * (Q_new + Q_new.T)

    C_new, d_new, R_new = np.zeros_like(C), np.zeros_like(d), np.zeros_like(R)
    for f in range(F):
        Gf = np.zeros((L + 1, L + 1)); Hf = np.zeros((L + 1, 1)); sy2 = 0.0
        for t in range(T):
            ex_t, exx_t, y = Ex[t], Exx[t], X[t, f]
            Gf[:L, :L] += exx_t; Gf[:L, L] += ex_t; Gf[L, :L] += ex_t; Gf[L, L] += 1
            Hf[:L, 0] += ex_t * y; Hf[L, 0] += y
            sy2 += y * y
        theta_f = np.linalg.solve(Gf + 1e-6 * np.eye(L + 1), Hf)
        C_new[f], d_new[f] = theta_f[:L, 0], theta_f[L, 0]
        R_new[f] = max((sy2 - (theta_f.T @ Hf).item()) / T, 1e-6)

    return A_new, b_new, Q_new, C_new, d_new, R_new


def _k1_setup(L=2, F=2, T=12, seed=42):
    rng = np.random.default_rng(seed)
    A = 0.8 * np.eye(L) + 0.05 * rng.standard_normal((L, L))
    b = 0.1 * rng.standard_normal(L)
    Qc = 0.1 * rng.standard_normal((L, L)); Q = Qc @ Qc.T + 0.05 * np.eye(L)
    C = rng.standard_normal((F, L)); d = 0.1 * rng.standard_normal(F)
    R = rng.uniform(0.1, 0.3, size=F)
    m0 = np.zeros(L); P0 = np.eye(L)
    m = np.zeros((T, L)); X = np.zeros((T, F))
    for t in range(T):
        m[t] = (m0 if t == 0 else A @ m[t - 1] + b) + (
            rng.multivariate_normal(np.zeros(L), P0) if t == 0 else rng.multivariate_normal(np.zeros(L), Q))
        X[t] = C @ m[t] + d + rng.normal(0, np.sqrt(R))
    return X, (A, b, Q, C, d, R, m0, P0)


def _run_production_k1(X, params, T):
    A, b, Q, C, d, R, m0, P0 = params
    L, F = A.shape[0], C.shape[0]
    # Dmax >= T so the discrete machinery never forces a completion (gamma_t[0] == 1 arises
    # naturally, it is asserted below rather than hardcoded)
    model = make_tiny_model(K=1, F=F, L=L, Dmax=T + 2, seed=0)
    model.A_fast[0], model.b_fast[0], model.Q_fast[0] = A, b, Q
    model.C_obs[0], model.d_obs[0], model.R_obs[0] = C, d, R
    model.m0[0], model.P0[0] = m0, P0
    fwd = model._forward_pass(X)
    bwd = model._discrete_backward(fwd)
    assert np.allclose(bwd.regime_posterior_smoothed[:, 0], 1.0, atol=1e-8), (
        "K=1 with Dmax>=T should give gamma_t[0]=1 for every t with no forcing")
    smoother = model._switching_smoother(fwd, bwd)
    model._m_step_continuous(X, bwd, smoother)
    return model


def _max_param_gap(model, ref):
    ref_A, ref_b, ref_Q, ref_C, ref_d, ref_R = ref
    return {
        "A": np.max(np.abs(model.A_fast[0] - ref_A)),
        "b": np.max(np.abs(model.b_fast[0] - ref_b)),
        "Q": np.max(np.abs(model.Q_fast[0] - ref_Q)),
        "C": np.max(np.abs(model.C_obs[0] - ref_C)),
        "d": np.max(np.abs(model.d_obs[0] - ref_d)),
        "R": np.max(np.abs(model.R_obs[0] - ref_R)),
    }


# Tolerance for the regularization-mirrored reference: only float round-off should remain.
K1_MIRRORED_TOL = 1e-8


def test_k1_em_matches_reference_lds_em():
    """Formula-correctness check: with the reference mirroring the production code's
    documented diagonal jitter, the two M-steps must agree to round-off."""
    T = 12
    X, params = _k1_setup(T=T)
    model = _run_production_k1(X, params, T)
    ref = _textbook_lds_em_mstep(X, *params, mirror_regularization=True)
    gaps = _max_param_gap(model, ref)
    for name, g in gaps.items():
        assert g < K1_MIRRORED_TOL, f"{name} gap {g:.3e} exceeds {K1_MIRRORED_TOL} (all gaps: {gaps})"


def test_k1_regularization_effect_is_small():
    """Against the PURE textbook reference (no jitter), the gap is the effect of the
    unchanged Kalman update's 1e-6 jitter, so it should be small but is not float-exact.
    This bounds how much that regularization moves the learned parameters."""
    T = 12
    X, params = _k1_setup(T=T)
    model = _run_production_k1(X, params, T)
    ref = _textbook_lds_em_mstep(X, *params, mirror_regularization=False)
    gaps = _max_param_gap(model, ref)
    for name, g in gaps.items():
        assert g < 1e-4, f"{name} gap {g:.3e} vs pure textbook EM is larger than the jitter can explain (all: {gaps})"


def test_Q_update_uses_posterior_covariance():
    """If Q_fast were computed from point estimates alone (outer(m,m) instead of
    P_smooth + outer(m,m)) it would systematically underestimate process noise. Use very
    noisy observations so the smoother posterior covariance is large."""
    L, F, T = 1, 1, 10
    rng = np.random.default_rng(7)
    model = make_tiny_model(K=1, F=F, L=L, Dmax=T + 2, seed=1)
    model.R_obs[0] = np.array([5.0])
    X = rng.normal(size=(T, F)) * 2.0

    fwd = model._forward_pass(X)
    bwd = model._discrete_backward(fwd)
    sm = model._switching_smoother(fwd, bwd)

    m = sm.smoothed_m_k[:, 0, :]
    A, b = model.A_fast[0], model.b_fast[0]
    resid = m[1:] - (m[:-1] @ A.T + b)
    Q_point_estimate_only = (resid.T @ resid) / (T - 1)

    model._m_step_continuous(X, bwd, sm)
    Q_correct = model.Q_fast[0]

    assert not np.allclose(Q_correct, Q_point_estimate_only, atol=1e-3), (
        "Q_fast matches the point-estimate-only computation: posterior covariance term was dropped")
    assert Q_correct[0, 0] >= Q_point_estimate_only[0, 0] - 1e-8
