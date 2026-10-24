"""
Population and atypical-user generators (WP5, N18-v2). Criteria: tests/PREDECLARED_CRITERIA_WP5.md section 2.

A user is the CENTRE of the existing generator (data/synthetic_dual.py: A = 0.75 I, b = 0, Q = 0.3 I, loadings 1.5 on feature f -> latent f mod L,
d_k = (k - (K-1)/2) * 4, R = 0.05, durations lognormal(ln 15, 0.3) clipped to 8..30, regimes alternate) times a user-specific perturbation.
The perturbation comes from its own random generator `default_rng([seed, 1])`, separate from the data stream `default_rng(seed)`, so with all
spreads set to 0 the data of a user equal `tests.utils.heldout_prediction.continued_synthetic(seed)` exactly (this is a test).

The perturbation sizes are ASSUMPTIONS (there are no real population data here): they are meant to make users clearly different but of one structure.
`generate_atypical_user` multiplies every standard-normal draw by `spread_factor` (3.0 in the pre-declared study).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class PopulationSpec:
    n_regimes: int = 2
    n_features: int = 4
    latent_dim: int = 2
    mean_dwell: float = 15.0
    sd_intercept: float = 0.6        # additive, per regime and feature (the regime separation is 4.0)
    sd_log_loading: float = 0.15     # multiplicative on the loading of each feature
    sd_A: float = 0.05               # additive on the AR(1) coefficient (A = a I), a clipped to [0.4, 0.9]
    sd_log_Q: float = 0.25           # multiplicative on the latent innovation variance (Q = q I)
    sd_log_R: float = 0.25           # multiplicative on the observation variance (scalar per user)
    sd_log_dwell: float = 0.2        # multiplicative on the mean dwell
    intercept_shift: float = 0.0     # added to EVERY intercept (the "shifted population" variant of criterion 6)

    def with_zero_spread(self) -> "PopulationSpec":
        return replace(self, sd_intercept=0.0, sd_log_loading=0.0, sd_A=0.0, sd_log_Q=0.0, sd_log_R=0.0, sd_log_dwell=0.0)


@dataclass
class PopulationUser:
    seed: int
    spread_factor: float
    X_train: np.ndarray
    X_test: np.ndarray
    p_train: np.ndarray
    p_test: np.ndarray
    truth: dict                      # the user's true generating parameters (for diagnostics only, never given to an arm)
    eps: np.ndarray                  # the standard-normal perturbation draws, already multiplied by spread_factor

    @property
    def X_full(self) -> np.ndarray:
        return np.vstack([self.X_train, self.X_test])


def draw_perturbation(seed: int, spec: PopulationSpec, spread_factor: float = 1.0) -> tuple[dict, np.ndarray]:
    """The user's parameters and the flat vector of draws (multiplied by spread_factor)."""
    K, F, L = spec.n_regimes, spec.n_features, spec.latent_dim
    rng = np.random.default_rng([int(seed), 1])
    e_d = rng.standard_normal((K, F)) * spread_factor
    e_c = rng.standard_normal(F) * spread_factor
    e_a, e_q, e_r, e_w = (rng.standard_normal(4) * spread_factor)
    C = np.zeros((F, L))
    for f in range(F):
        C[f, f % L] = 1.5 * np.exp(spec.sd_log_loading * e_c[f])
    base_d = [np.full(F, (k - (K - 1) / 2.0) * 4.0) for k in range(K)]
    d = [base_d[k] + spec.sd_intercept * e_d[k] + spec.intercept_shift for k in range(K)]
    a = float(np.clip(0.75 + spec.sd_A * e_a, 0.4, 0.9))
    truth = dict(
        A=a * np.eye(L), C=C, d=np.array(d),
        q=float(0.3 * np.exp(spec.sd_log_Q * e_q)), r=float(0.05 * np.exp(spec.sd_log_R * e_r)),
        mean_dwell=float(spec.mean_dwell * np.exp(spec.sd_log_dwell * e_w)),
    )
    eps = np.concatenate([e_d.ravel(), e_c, [e_a, e_q, e_r, e_w]])
    return truth, eps


def generate_population_user(seed: int, n_train_segments: int, n_test_segments: int = 2, spec: PopulationSpec = PopulationSpec(),
                             spread_factor: float = 1.0) -> PopulationUser:
    K, F, L = spec.n_regimes, spec.n_features, spec.latent_dim
    truth, eps = draw_perturbation(seed, spec, spread_factor)
    A, C, d = truth["A"], truth["C"], truth["d"]
    Q_scale = np.sqrt(np.diag(truth["q"] * np.eye(L)))
    R_scale = np.sqrt(np.full(F, truth["r"]))
    rng = np.random.default_rng(int(seed))

    def segments(n, current):
        regimes = []
        for _ in range(n):
            dur = int(np.clip(rng.lognormal(mean=np.log(truth["mean_dwell"]), sigma=0.3), 8, 30))
            regimes.extend([current] * dur)
            current = rng.choice([k for k in range(K) if k != current])
        return np.array(regimes, dtype=int), current

    p_tr, nxt = segments(n_train_segments, 0)
    T_tr = len(p_tr)
    m_tr = np.zeros((T_tr, L))
    m_curr = rng.normal(scale=0.2, size=L)
    m_tr[0] = m_curr
    for t in range(1, T_tr):
        m_curr = A @ m_curr + rng.normal(scale=Q_scale, size=L)
        m_tr[t] = m_curr
    X_tr = np.zeros((T_tr, F))
    for t in range(T_tr):
        X_tr[t] = C @ m_tr[t] + d[p_tr[t]] + rng.normal(scale=R_scale, size=F)

    p_te, _ = segments(n_test_segments, nxt)
    T_te = len(p_te)
    m_te = np.zeros((T_te, L))
    for t in range(T_te):
        m_curr = A @ m_curr + rng.normal(scale=Q_scale, size=L)
        m_te[t] = m_curr
    X_te = np.zeros((T_te, F))
    for t in range(T_te):
        X_te[t] = C @ m_te[t] + d[p_te[t]] + rng.normal(scale=R_scale, size=F)
    return PopulationUser(seed=int(seed), spread_factor=float(spread_factor), X_train=X_tr, X_test=X_te, p_train=p_tr, p_test=p_te, truth=truth, eps=eps)


def generate_atypical_user(seed: int, n_train_segments: int, n_test_segments: int = 2, spec: PopulationSpec = PopulationSpec(),
                           spread_factor: float = 3.0) -> PopulationUser:
    """A user whose perturbation draws are `spread_factor` (3) times those of a population user: about 3 SD from the centre in every coordinate."""
    return generate_population_user(seed, n_train_segments, n_test_segments, spec, spread_factor=spread_factor)
