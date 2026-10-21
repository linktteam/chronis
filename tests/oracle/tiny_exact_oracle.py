"""Tiny brute-force exact oracle for the dual-timescale HSSM.

Reference-only. O(K^T * D^T) enumeration: never call outside tests, never at T > ~6 or
K > ~4 (size guard below: T<=6, K<=4, Dmax<=3 AND an enumerated-path cap). If the fast implementation disagrees with this oracle on a tiny well-conditioned
example, the fast implementation is wrong until proven otherwise.

Copied close to verbatim from section 4.4 of the fix package. One change: the dead
placeholder loop (`for w, path in zip(...): pass`) the package flags as leftover
scaffolding has been deleted, per its own instruction.
"""
from __future__ import annotations

from dataclasses import dataclass

from functools import lru_cache

import numpy as np
from scipy.special import logsumexp

# Size guard. The packaged oracle allowed K<=2; the real-repo integration relaxes it to K<=4
# (T<=6 and Dmax<=3 unchanged) and adds a cap on the NUMBER of enumerated legal paths so a
# future call cannot silently run away. Largest allowed case, K=4 T=6 Dmax=3, has 10,668 paths.
MAX_T, MAX_K, MAX_DMAX = 6, 4, 3
MAX_ORACLE_PATHS = 20_000


def count_legal_paths(T: int, K: int, Dmax: int) -> int:
    """Exact number of legal (k,r) paths enumerate_legal_paths would yield, without
    enumerating them (checked equal to brute-force enumeration for several sizes)."""
    @lru_cache(None)
    def f(t, r, first):
        if t == T:
            return 1
        if r is not None and r > 1:
            return f(t + 1, r - 1, False)
        n_regimes = K if first else K - 1
        return n_regimes * sum(f(t + 1, d, False) for d in range(1, Dmax + 1))
    return f(0, None, True)


def _check_size(T: int, K: int, Dmax: int) -> None:
    assert T <= MAX_T and K <= MAX_K and Dmax <= MAX_DMAX, (
        f"oracle is brute-force only: keep the problem tiny (T<={MAX_T}, K<={MAX_K}, Dmax<={MAX_DMAX}); "
        f"got T={T}, K={K}, Dmax={Dmax}")
    n = count_legal_paths(T, K, Dmax)
    assert n <= MAX_ORACLE_PATHS, f"oracle would enumerate {n} paths (> {MAX_ORACLE_PATHS})"


@dataclass
class OracleResult:
    log_likelihood: float
    filtered_regime_posterior: np.ndarray     # (T, K)
    smoothed_regime_posterior: np.ndarray     # (T, K)
    pairwise_transition_marginal: np.ndarray  # (T-1, K, K): [t,i,j] = P(k_t=i, k_{t+1}=j, boundary at t | Y)
    smoothed_m: np.ndarray                    # (T, L)
    smoothed_P: np.ndarray                    # (T, L, L)


def enumerate_legal_paths(T: int, K: int, Dmax: int):
    """Yield every legal (k_1..k_T, r_1..r_T) path as a list of (k, r) tuples."""
    def rec(t, k_prev, r_prev, path):
        if t == T:
            yield list(path)
            return
        if r_prev is not None and r_prev > 1:
            path.append((k_prev, r_prev - 1))
            yield from rec(t + 1, k_prev, r_prev - 1, path)
            path.pop()
        else:
            candidates = range(K) if k_prev is None else [j for j in range(K) if j != k_prev]
            for k_new in candidates:
                for d in range(1, Dmax + 1):
                    path.append((k_new, d))
                    yield from rec(t + 1, k_new, d, path)
                    path.pop()
    yield from rec(0, None, None, [])


def path_log_prior(path, log_pi, logA, dur_logpmf):
    """Prior log-probability of a (k,r) path, before seeing any data."""
    lp = 0.0
    for t, (k, r) in enumerate(path):
        if t == 0:
            lp += log_pi[k] + dur_logpmf[k, r - 1]
        else:
            k_prev, r_prev = path[t - 1]
            if r_prev > 1:
                pass  # deterministic continuation contributes 0
            else:
                lp += logA[k_prev, k] + dur_logpmf[k, r - 1]
    return lp


def path_kalman_likelihood_and_smoother(path, X, model, return_lag=False):
    """Exact (non-switching) Kalman filter + RTS smoother along the fixed regime sequence
    in `path`. Returns (log_likelihood, smoothed_m (T,L), smoothed_P (T,L,L)); with
    return_lag=True also returns P_lag (T,L,L) where P_lag[t] = Cov(m_t, m_{t-1} | path, Y)
    for t >= 1 (row 0 unused)."""
    T, L = len(path), model.L
    m_filt = np.zeros((T, L)); P_filt = np.zeros((T, L, L))
    m_pred = np.zeros((T, L)); P_pred = np.zeros((T, L, L))
    loglik = 0.0
    for t, (k, _r) in enumerate(path):
        if t == 0:
            m_p, P_p = model.m0[k], model.P0[k]
        else:
            m_p = model.A_fast[k] @ m_filt[t - 1] + model.b_fast[k]
            P_p = model.A_fast[k] @ P_filt[t - 1] @ model.A_fast[k].T + model.Q_fast[k]
        m_pred[t], P_pred[t] = m_p, P_p
        m_u, P_u, ll = model._kalman_update_regime(X[t], m_p, P_p, k)
        m_filt[t], P_filt[t] = m_u, P_u
        loglik += ll

    smoothed_m = np.zeros((T, L)); smoothed_P = np.zeros((T, L, L))
    P_lag = np.zeros((T, L, L))
    smoothed_m[-1], smoothed_P[-1] = m_filt[-1], P_filt[-1]
    for t in range(T - 2, -1, -1):
        k_next = path[t + 1][0]
        J = P_filt[t] @ model.A_fast[k_next].T @ np.linalg.inv(P_pred[t + 1] + 1e-9 * np.eye(L))
        smoothed_m[t] = m_filt[t] + J @ (smoothed_m[t + 1] - m_pred[t + 1])
        smoothed_P[t] = P_filt[t] + J @ (smoothed_P[t + 1] - P_pred[t + 1]) @ J.T
        P_lag[t + 1] = J @ smoothed_P[t + 1]
    if return_lag:
        return loglik, smoothed_m, smoothed_P, P_lag
    return loglik, smoothed_m, smoothed_P


def run_tiny_oracle_dynamics_stats(model, X: np.ndarray):
    """EXACT expected sufficient statistics for each regime's fast dynamics
    m_t = A_k m_{t-1} + b_k + noise, summed over every t >= 1 with k_t = k, under the exact
    posterior over all legal paths. Same layout as DualTimescaleHSSM._dynamics_sufficient_stats:
    G (K,L+1,L+1), H (K,L+1,L), S (K,L,L), W (K,)."""
    T, K, Dmax, L = X.shape[0], model.K, model.Dmax, model.L
    _check_size(T, K, Dmax)
    log_pi = np.log(np.clip(model.pi, 1e-300, None))
    logA = np.log(np.clip(model.A_slow, 1e-300, None))
    dur_logpmf = model._duration_logpmf()

    paths, log_joints, per_path = [], [], []
    for path in enumerate_legal_paths(T, K, Dmax):
        loglik, sm_m, sm_P, P_lag = path_kalman_likelihood_and_smoother(path, X, model, return_lag=True)
        paths.append(path)
        log_joints.append(path_log_prior(path, log_pi, logA, dur_logpmf) + loglik)
        per_path.append((sm_m, sm_P, P_lag))
    log_joints = np.array(log_joints)
    weights = np.exp(log_joints - logsumexp(log_joints))

    G = np.zeros((K, L + 1, L + 1)); H = np.zeros((K, L + 1, L))
    S = np.zeros((K, L, L)); W = np.zeros(K)
    for w, path, (sm_m, sm_P, P_lag) in zip(weights, paths, per_path):
        for t in range(1, T):
            k = path[t][0]
            m_prev, m_cur = sm_m[t - 1], sm_m[t]
            E_pp = sm_P[t - 1] + np.outer(m_prev, m_prev)
            E_cc = sm_P[t] + np.outer(m_cur, m_cur)
            E_cp = P_lag[t] + np.outer(m_cur, m_prev)  # E[m_t m_{t-1}^T]
            W[k] += w
            G[k, :L, :L] += w * E_pp
            G[k, :L, L] += w * m_prev
            G[k, L, :L] += w * m_prev
            G[k, L, L] += w
            H[k, :L, :] += w * E_cp.T
            H[k, L, :] += w * m_cur
            S[k] += w * E_cc
    return G, H, S, W


def oracle_regime_sequence_leakage(model, X: np.ndarray) -> float:
    """Exact posterior mass NOT on the single most probable complete regime sequence k_1..k_T
    (mass of all duration paths that share a regime sequence is summed first). 0 when the data pin
    down one regime sequence; larger when several remain plausible.

    Added for ledger D12: the gap between the production dynamics statistics and the oracle's is
    expected to track this leaked mass (the Gaussian collapse merges branches that the exact
    posterior still separates), so the K=3 concentrated-posterior test scales its tolerance by it.
    New function; nothing above it changed."""
    T, K, Dmax = X.shape[0], model.K, model.Dmax
    _check_size(T, K, Dmax)
    log_pi = np.log(np.clip(model.pi, 1e-300, None))
    logA = np.log(np.clip(model.A_slow, 1e-300, None))
    dur_logpmf = model._duration_logpmf()
    log_joints, sequences = [], []
    for path in enumerate_legal_paths(T, K, Dmax):
        loglik, _, _ = path_kalman_likelihood_and_smoother(path, X, model)
        log_joints.append(path_log_prior(path, log_pi, logA, dur_logpmf) + loglik)
        sequences.append(tuple(k for k, _r in path))
    log_joints = np.array(log_joints)
    weights = np.exp(log_joints - logsumexp(log_joints))
    mass: dict = {}
    for w, seq in zip(weights, sequences):
        mass[seq] = mass.get(seq, 0.0) + float(w)
    return max(0.0, 1.0 - max(mass.values()))


def run_tiny_exact_oracle(model, X: np.ndarray) -> OracleResult:
    T, K, Dmax, L = X.shape[0], model.K, model.Dmax, model.L
    _check_size(T, K, Dmax)
    log_pi = np.log(np.clip(model.pi, 1e-300, None))
    logA = np.log(np.clip(model.A_slow, 1e-300, None))
    dur_logpmf = model._duration_logpmf()

    log_joints, path_regimes, path_smoothed_m, path_smoothed_P = [], [], [], []
    for path in enumerate_legal_paths(T, K, Dmax):
        log_prior = path_log_prior(path, log_pi, logA, dur_logpmf)
        loglik, sm_m, sm_P = path_kalman_likelihood_and_smoother(path, X, model)
        log_joints.append(log_prior + loglik)
        path_regimes.append([k for k, _r in path])
        path_smoothed_m.append(sm_m)
        path_smoothed_P.append(sm_P)

    log_joints = np.array(log_joints)
    log_marginal = logsumexp(log_joints)
    weights = np.exp(log_joints - log_marginal)

    smoothed_regime = np.zeros((T, K))
    for w, regs in zip(weights, path_regimes):
        for t, k in enumerate(regs):
            smoothed_regime[t, k] += w

    filtered_regime = np.zeros((T, K))
    for t in range(T):
        log_joints_t, regs_t = [], []
        for path in enumerate_legal_paths(t + 1, K, Dmax):
            log_prior = path_log_prior(path, log_pi, logA, dur_logpmf)
            loglik, _, _ = path_kalman_likelihood_and_smoother(path, X[: t + 1], model)
            log_joints_t.append(log_prior + loglik)
            regs_t.append(path[-1][0])
        log_joints_t = np.array(log_joints_t)
        w_t = np.exp(log_joints_t - logsumexp(log_joints_t))
        for w, k in zip(w_t, regs_t):
            filtered_regime[t, k] += w

    pairwise = np.zeros((max(T - 1, 0), K, K))
    for path in enumerate_legal_paths(T, K, Dmax):
        log_prior = path_log_prior(path, log_pi, logA, dur_logpmf)
        loglik, _, _ = path_kalman_likelihood_and_smoother(path, X, model)
        w = np.exp(log_prior + loglik - log_marginal)
        for t in range(T - 1):
            k_t, r_t = path[t]
            k_t1, _ = path[t + 1]
            if r_t == 1 and k_t1 != k_t:
                pairwise[t, k_t, k_t1] += w

    smoothed_m = np.zeros((T, L))
    smoothed_P = np.zeros((T, L, L))
    for w, sm_m in zip(weights, path_smoothed_m):
        smoothed_m += w * sm_m
    for w, sm_m, sm_P in zip(weights, path_smoothed_m, path_smoothed_P):
        for t in range(T):
            diff = sm_m[t] - smoothed_m[t]
            smoothed_P[t] += w * (sm_P[t] + np.outer(diff, diff))

    return OracleResult(
        log_likelihood=float(log_marginal),
        filtered_regime_posterior=filtered_regime,
        smoothed_regime_posterior=smoothed_regime,
        pairwise_transition_marginal=pairwise,
        smoothed_m=smoothed_m,
        smoothed_P=smoothed_P,
    )
