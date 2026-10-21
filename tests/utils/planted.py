"""Planted-parameter model and simulator for recovery tests.

Taken from the reviewer's recovery_study_repro.py (same parameters, same generative
process), with T made a parameter. Latent state is a scalar AR(1) whose coefficient
depends on the current regime; the latent is NOT reset at regime changes. Observations
are regime-specific linear functions of the latent plus regime-specific offsets.
"""
import numpy as np

from tests.utils.model_factory import make_tiny_model

K, F, L, D = 3, 2, 1, 6
TRUE_A_FAST = np.array([0.9, 0.6, 0.3])


def planted_model():
    m = make_tiny_model(K=K, F=F, L=L, Dmax=D, seed=0)
    m.A_slow = np.array([[0, .8, .2], [.3, 0, .7], [.6, .4, 0]])
    m.dur_mu = np.log([3., 4., 2.5])
    m.dur_sigma = np.array([.3, .3, .3])
    m.A_fast = TRUE_A_FAST.reshape(K, 1, 1).copy()
    m.b_fast = np.zeros((K, L))
    m.Q_fast = np.full((K, 1, 1), 0.05)
    m.C_obs = np.array([[[1.], [0.5]], [[1.], [-0.5]], [[1.], [1.]]])
    m.d_obs = np.array([[3, 0], [-3, 0], [0, 3.]])
    m.R_obs = np.full((K, F), 0.1)
    m.m0 = np.zeros((K, L))
    m.P0 = np.stack([np.eye(L)] * K)
    return m


def simulate(m, T, rng):
    """Draw T observations from the model; also returns the true regime and latent paths."""
    pmf = np.exp(m._duration_logpmf())
    X, regimes, latents = [], [], []
    k = rng.integers(K)
    z = np.zeros(L)
    while len(X) < T:
        d = rng.choice(D, p=pmf[k]) + 1
        for _ in range(d):
            z = m.A_fast[k] @ z + m.b_fast[k] + rng.normal(0, np.sqrt(m.Q_fast[k][0, 0]), L)
            X.append(m.C_obs[k] @ z + m.d_obs[k] + rng.normal(0, np.sqrt(m.R_obs[k])))
            regimes.append(k)
            latents.append(z.copy())
        k = rng.choice(K, p=m.A_slow[k])
    return np.array(X[:T]), np.array(regimes[:T]), np.array(latents[:T])
