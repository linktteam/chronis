"""Tests for fit()'s EM loop: monotonicity and the backtracking guard.

fit() is new supporting code (the spec's quoted scope does not include it), needed to
exercise the fixed E-step and M-step components end to end.
"""
import numpy as np

from backbone.hssm.model import FitState
from tests.utils.model_factory import make_tiny_model


def _two_regime_data(T, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((T, 2))
    regime, t = 0, 0
    while t < T:
        mean = [2.0, 2.0] if regime == 0 else [-2.0, -2.0]
        for _ in range(rng.integers(2, 5)):
            if t >= T:
                break
            X[t] = mean + 0.3 * rng.standard_normal(2)
            t += 1
        regime = 1 - regime
    return X


def test_fit_loglik_monotone_until_stop():
    X = _two_regime_data(25, seed=1)
    model = make_tiny_model(K=2, F=2, L=1, Dmax=3, seed=2)
    model.fit(X, n_iter=20)

    hist = model.log_likelihood_history_
    assert len(hist) >= 1
    assert np.all(np.diff(hist) >= -1e-6), f"log-likelihood history is not monotone: {hist}"
    assert model.fit_state_ in (FitState.CONVERGED, FitState.MAX_ITER_REACHED,
                                  FitState.REJECTED_UPDATE, FitState.NUMERICAL_FAILURE)


def test_backtracking_guard_rejects_decrease():
    """Force a corrupting M-step and confirm fit() detects the decrease and reverts."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(15, 2))
    model = make_tiny_model(K=2, F=2, L=1, Dmax=3, seed=4)

    call_count = {"n": 0}
    original_m_step = model._m_step_continuous

    def sabotaged_m_step(X_, bwd, sm):
        call_count["n"] += 1
        if call_count["n"] == 2:
            model.R_obs = model.R_obs * 1e6
            return
        return original_m_step(X_, bwd, sm)

    model._m_step_continuous = sabotaged_m_step
    model.fit(X, n_iter=10)

    assert model.fit_state_ == FitState.REJECTED_UPDATE, (
        f"expected REJECTED_UPDATE after a sabotaged M-step, got {model.fit_state_}")
    assert not np.any(model.R_obs > 1e5), "corrupted parameters were not reverted by the guard"


def test_fit_state_is_set():
    model = make_tiny_model(K=2, F=1, L=1, Dmax=2, seed=5)
    X = np.random.default_rng(5).normal(size=(10, 1))
    model.fit(X, n_iter=5)
    assert isinstance(model.fit_state_, FitState)
