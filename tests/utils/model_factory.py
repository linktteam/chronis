"""Shared helpers for building small DualTimescaleHSSM instances and data.

PORT NOTE. The tests in this tree were written against a standalone package whose model had
a `random_init()` method and a constructor keyword `random_state`. The real class has neither:
it has `seed=` and a data-aware `_init_params(X)`. To keep every test's numerical setup
IDENTICAL to the one its tolerances were frozen against, `make_tiny_model` builds a real
`DualTimescaleHSSM`, runs the real `_init_params` (so all real-class state exists), and then
overwrites the parameter arrays with exactly the values the package's `random_init()` drew for
the same seed (`_package_random_init` is a verbatim copy of that arithmetic, drawing from its own
`default_rng(seed)`; bit-identity against the package was checked when this was ported).

Only SETUP is adapted here. No assertion or tolerance in any ported test was changed.

One deliberate neutralisation: the real `_init_params` pre-sets
`duration_optimizer_success_ = {k: True}`. A test that asserts those flags are TRUE after
`_update_discrete_params` would then pass even if the update never touched them, so the factory
resets the status fields to the package's starting state ({} / False). Tests that start from the
real init state are in test_dual_model_port_contract.py.
"""
import numpy as np

from backbone.hssm.dual_model import DualTimescaleHSSM


def _package_random_init(model: DualTimescaleHSSM, random_state: int) -> DualTimescaleHSSM:
    rng = np.random.default_rng(random_state)
    K, F, L, Dmax = model.K, model.F, model.L, model.Dmax

    model.pi = np.full(K, 1.0 / K)

    A_full = np.zeros((K, K))
    if K > 1:
        A_off = rng.dirichlet(np.full(K - 1, 2.0), size=K)
        for i in range(K):
            off = [j for j in range(K) if j != i]
            for idx, j in enumerate(off):
                A_full[i, j] = A_off[i, idx]
    model.A_slow = A_full

    model.dur_mu = rng.normal(np.log(max(2.0, Dmax / 2.0)), 0.25, size=K)
    model.dur_sigma = rng.uniform(0.25, 0.45, size=K)

    model.A_fast = np.zeros((K, L, L))
    model.b_fast = np.zeros((K, L))
    model.Q_fast = np.zeros((K, L, L))
    for k in range(K):
        A = 0.7 * np.eye(L) + 0.08 * rng.standard_normal((L, L))
        max_abs = np.max(np.abs(np.linalg.eigvals(A)))
        if max_abs > 0.9:
            A *= 0.9 / max_abs
        model.A_fast[k] = A
        model.b_fast[k] = 0.05 * rng.standard_normal(L)
        Qc = 0.05 * rng.standard_normal((L, L))
        model.Q_fast[k] = Qc @ Qc.T + 0.02 * np.eye(L)

    model.m0 = 0.1 * rng.standard_normal((K, L))
    model.P0 = np.stack([np.eye(L) for _ in range(K)])

    model.C_obs = 0.8 * rng.standard_normal((K, F, L))
    model.d_obs = 0.05 * rng.standard_normal((K, F))
    model.R_obs = rng.uniform(0.05, 0.2, size=(K, F))
    return model


def make_tiny_model(K=2, F=1, L=1, Dmax=2, seed=0):
    model = DualTimescaleHSSM(n_regimes=K, n_features=F, latent_dim=L, max_duration=Dmax, seed=seed)
    model._init_params(np.zeros((2, F)))     # real init: establishes all real-class state
    _package_random_init(model, seed)        # then the package's parameter values for this seed
    model.duration_optimizer_success_ = {}   # neutral status, see module docstring
    model.duration_fallback_used_ = False
    return model


def make_tiny_data(model, T, seed=0, nan_prob=0.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, model.F))
    if nan_prob > 0:
        mask = rng.random(size=X.shape) < nan_prob
        for t in range(T):  # never blank out an entire timestep across all features
            if mask[t].all():
                mask[t, rng.integers(model.F)] = False
        X[mask] = np.nan
    return X
