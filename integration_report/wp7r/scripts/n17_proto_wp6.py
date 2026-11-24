"""WP6 Part B (N17), PROTOTYPE of the parameter-uncertainty term for m_t_std. Development only; the design is in tests/PREDECLARED_CRITERIA_WP6.md section 7.
Method (delta method on the smoothed latent mean, gauge-projected):
  theta = unconstrained continuous parameters (A, b, log-Cholesky Q, C, d, log R), per regime.
  I = - Hessian of log p(X | theta) (central differences of the engine's own forward pass).
  G = gauge tangent: the latent frame change m' = S m + c moves (A, b, Q, C, d) without changing what the model says about X; those directions are NOT uncertainty.
  Z = orthonormal basis of the complement of span(G);  Sigma = Z (Z' I Z)^-1 Z'  (if Z' I Z is positive definite; see policy in section 7).
  J = d m_t / d theta by central differences of the engine's own smoother;  Var_param(m_t) = diag(J Sigma J').
  m_t_std_total^2 = m_t_std^2 + Var_param.
"""
import copy

import numpy as np

# ---------------------------------------------------------------- parameter vector
def _tril_idx(L):
    return np.tril_indices(L)


def pack(model):
    K, L, F = model.K, model.L, model.F
    parts = [model.A_fast.ravel(), model.b_fast.ravel()]
    ti = _tril_idx(L)
    for k in range(K):
        Lc = np.linalg.cholesky(model.Q_fast[k])
        d = np.log(np.diag(Lc))
        off = Lc[ti]
        off = off.copy()
        off[np.where(ti[0] == ti[1])[0]] = d
        parts.append(off)
    parts += [model.C_obs.ravel(), model.d_obs.ravel(), np.log(model.R_obs).ravel()]
    return np.concatenate(parts)


def unpack(model, theta):
    """Returns a deep copy of `model` with the continuous parameters set from `theta`."""
    K, L, F = model.K, model.L, model.F
    m = copy.deepcopy(model)
    i = 0
    m.A_fast = theta[i:i + K * L * L].reshape(K, L, L).copy(); i += K * L * L
    m.b_fast = theta[i:i + K * L].reshape(K, L).copy(); i += K * L
    ti = _tril_idx(L)
    diag_pos = np.where(ti[0] == ti[1])[0]
    Q = np.zeros((K, L, L))
    nl = L * (L + 1) // 2
    for k in range(K):
        v = theta[i:i + nl].copy(); i += nl
        v[diag_pos] = np.exp(v[diag_pos])
        Lc = np.zeros((L, L)); Lc[ti] = v
        Q[k] = Lc @ Lc.T
    m.Q_fast = Q
    m.C_obs = theta[i:i + K * F * L].reshape(K, F, L).copy(); i += K * F * L
    m.d_obs = theta[i:i + K * F].reshape(K, F).copy(); i += K * F
    m.R_obs = np.exp(theta[i:i + K * F]).reshape(K, F).copy(); i += K * F
    assert i == len(theta)
    return m


def apply_gauge(model, S, c):
    """The same model written in the latent frame m' = S m + c (exact transformation of the continuous parameters)."""
    m = copy.deepcopy(model)
    Si = np.linalg.inv(S)
    for k in range(model.K):
        Ak = S @ model.A_fast[k] @ Si
        m.A_fast[k] = Ak
        m.b_fast[k] = S @ model.b_fast[k] + c - Ak @ c
        m.Q_fast[k] = S @ model.Q_fast[k] @ S.T
        m.C_obs[k] = model.C_obs[k] @ Si
        m.d_obs[k] = model.d_obs[k] - model.C_obs[k] @ Si @ c
    return m


def gauge_tangent(model, eps=1e-6):
    L = model.L
    th0 = pack(model)
    cols = []
    for a in range(L):
        for b in range(L):
            E = np.zeros((L, L)); E[a, b] = 1.0
            cols.append((pack(apply_gauge(model, np.eye(L) + eps * E, np.zeros(L))) - pack(apply_gauge(model, np.eye(L) - eps * E, np.zeros(L)))) / (2 * eps))
    for a in range(L):
        e = np.zeros(L); e[a] = 1.0
        cols.append((pack(apply_gauge(model, np.eye(L), eps * e)) - pack(apply_gauge(model, np.eye(L), -eps * e))) / (2 * eps))
    return np.array(cols).T  # (P, L*L+L)


# ---------------------------------------------------------------- derivatives
def loglik(model, X, theta):
    return float(unpack(model, theta)._forward_pass(X).log_likelihood)


def numeric_hessian(model, X, theta, h=1e-3):
    P = len(theta)
    f0 = loglik(model, X, theta)
    fp = np.zeros(P); fm = np.zeros(P)
    E = np.eye(P) * h
    for i in range(P):
        fp[i] = loglik(model, X, theta + E[i]); fm[i] = loglik(model, X, theta - E[i])
    H = np.zeros((P, P))
    for i in range(P):
        H[i, i] = (fp[i] - 2 * f0 + fm[i]) / h ** 2
        for j in range(i + 1, P):
            fpp = loglik(model, X, theta + E[i] + E[j])
            H[i, j] = H[j, i] = (fpp - fp[i] - fp[j] + f0) / h ** 2
    return H


def smoothed_mean(model, X, theta):
    return unpack(model, theta).estimate_states(X)[0]


def jacobian_m(model, X, theta, h=1e-4):
    P = len(theta)
    T, L = len(X), model.L
    J = np.zeros((T, L, P))
    for i in range(P):
        e = np.zeros(P); e[i] = h
        J[:, :, i] = (smoothed_mean(model, X, theta + e) - smoothed_mean(model, X, theta - e)) / (2 * h)
    return J


def parameter_variance(model, X, h_hess=1e-3, h_jac=1e-4, return_parts=False):
    th = pack(model)
    G = gauge_tangent(model)
    Qg, _ = np.linalg.qr(G)
    Zfull = np.eye(len(th)) - Qg @ Qg.T
    w, V = np.linalg.eigh(Zfull)
    Z = V[:, w > 0.5]
    H = numeric_hessian(model, X, th, h_hess)
    info = -0.5 * (H + H.T)
    IZ = Z.T @ info @ Z
    ev, evec = np.linalg.eigh(IZ)
    J = jacobian_m(model, X, th, h_jac)
    out = dict(theta=th, G=G, Z=Z, info=info, IZ_eigs=ev, J=J)
    return out


# ---------------------------------------------------------------- the term, both policies from one Hessian / Jacobian
EPS = np.finfo(float).eps


def parameter_variance_terms(model, X, h_hess=1e-3, h_jac=1e-4):
    """Returns dict(var_R, var_C, n_excluded, n_neg_kept_R, n_neg_C_dropped, n_pos, tol, seconds_hess, seconds_jac, ll)."""
    import time
    th = pack(model)
    G = gauge_tangent(model)
    Qg, _ = np.linalg.qr(G)
    w, V = np.linalg.eigh(np.eye(len(th)) - Qg @ Qg.T)
    Z = V[:, w > 0.5]
    t0 = time.time(); H = numeric_hessian(model, X, th, h_hess); t_h = time.time() - t0
    info = -0.5 * (H + H.T)
    ev, U = np.linalg.eigh(Z.T @ info @ Z)
    ll = loglik(model, X, th)
    tol = 4000.0 * EPS * max(abs(ll), 1.0) / h_hess ** 2
    t0 = time.time(); J = jacobian_m(model, X, th, h_jac); t_j = time.time() - t0
    out = dict(n_excluded=int(np.sum(np.abs(ev) < tol)), n_neg=int(np.sum(ev <= -tol)), n_pos=int(np.sum(ev >= tol)), tol=tol, seconds_hess=t_h, seconds_jac=t_j, ll=ll, eig=ev.tolist())
    mt, _, labels, _ = model.estimate_states(X)
    T, L = mt.shape
    Pperp = np.eye(T * L) - _frame_basis(mt, labels) @ _frame_basis(mt, labels).T
    Jm = J.reshape(T * L, -1)
    Jp = Pperp @ Jm
    out["n_labels"] = int(len(np.unique(labels))); out["rank_frame"] = int(np.linalg.matrix_rank(_frame_basis(mt, labels)))
    for name, keep, scale in (("R", np.abs(ev) >= tol, np.abs(ev)), ("C", ev >= tol, ev)):
        Sz = (U[:, keep] / scale[keep]) @ U[:, keep].T
        Sig = Z @ Sz @ Z.T
        out["var_" + name + "_plain"] = np.einsum("tlp,pq,tlq->tl", J, Sig, J)
        out["var_" + name] = np.einsum("ip,pq,iq->i", Jp, Sig, Jp).reshape(T, L)
    return out


def _frame_basis(mt, labels):
    """Orthonormal basis (T*L, r) of the per-regime frame changes in OUTPUT space: for each label r present, rows of regime r with delta m[:, a] = m[:, b] and delta m[:, a] = 1."""
    T, L = mt.shape
    cols = []
    for r in np.unique(labels):
        idx = labels == r
        for a in range(L):
            for b in range(L):
                D = np.zeros((T, L)); D[idx, a] = mt[idx, b]; cols.append(D.ravel())
            D = np.zeros((T, L)); D[idx, a] = 1.0; cols.append(D.ravel())
    Q, Rr = np.linalg.qr(np.array(cols).T)
    keep = np.abs(np.diag(Rr)) > 1e-10 * max(np.abs(np.diag(Rr)).max(), 1e-300)
    return Q[:, keep]
