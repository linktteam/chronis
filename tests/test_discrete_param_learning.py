"""Negative controls for A_slow / duration parameter learning (section 7.2)."""
import numpy as np
from unittest.mock import patch

from tests.utils.model_factory import make_tiny_model, make_tiny_data


def _three_regime_data(T, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((T, 2))
    means = {0: [3.0, 3.0], 1: [-3.0, -3.0], 2: [3.0, -3.0]}
    regime, t = 0, 0
    while t < T:
        for _ in range(rng.integers(2, 5)):
            if t >= T:
                break
            X[t] = means[regime] + 0.3 * rng.standard_normal(2)
            t += 1
        regime = (regime + 1) % 3
    return X


def test_A_slow_row_normalize_and_fallback():
    model = make_tiny_model(K=3, F=1, L=1, Dmax=2, seed=1)
    X = make_tiny_data(model, T=8, seed=101)
    fwd = model._forward_pass(X)
    bwd = model._discrete_backward(fwd)
    model._update_discrete_params(bwd)

    assert np.allclose(model.A_slow.sum(axis=1), 1.0, atol=1e-8), f"rows: {model.A_slow.sum(axis=1)}"
    assert np.allclose(np.diag(model.A_slow), 0.0), "A_slow has a nonzero diagonal"
    assert np.all(model.A_slow >= 0)


def test_A_slow_fallback_row_is_recorded_when_no_evidence():
    """Zero transition-out evidence for a regime must trigger the explicit fallback (keep
    the old row) and be recorded, not silently produce NaN or an arbitrary row."""
    model = make_tiny_model(K=2, F=1, L=1, Dmax=2, seed=2)
    X = make_tiny_data(model, T=3, seed=202)
    fwd = model._forward_pass(X)
    bwd = model._discrete_backward(fwd)
    bwd.xi_counts[0, :] = 0.0
    old_row0 = model.A_slow[0].copy()
    model._update_discrete_params(bwd)
    assert 0 in model.transition_fallback_rows_
    assert np.allclose(model.A_slow[0], old_row0), "fallback row should equal the previous row"
    assert np.all(np.isfinite(model.A_slow))


def test_A_slow_moves_from_seeded_initialization():
    """A_slow must actually learn. Uses K=3: with K=2, zero diagonal plus row-sum 1 pins
    the off-diagonal to exactly 1.0, which is not a meaningful test."""
    model = make_tiny_model(K=3, F=2, L=1, Dmax=3, seed=3)
    X = _three_regime_data(45, seed=303)
    # PORT (setup only): the real fit() always re-runs _init_params(X), so "before" must be the
    # A_slow that fit() itself initialised, not the factory's. Record it when fit() draws it;
    # without this the comparison would pass even if the M-step never touched A_slow.
    captured = {}
    real_init = model._init_params

    def spy_init(X_):
        real_init(X_)
        captured["A_slow"] = model.A_slow.copy()

    model._init_params = spy_init
    model.fit(X, n_iter=15)
    A_slow_before = captured["A_slow"]
    assert not np.allclose(model.A_slow, A_slow_before, atol=1e-3), (
        "A_slow did not move from its random initialization after fit()")


def test_duration_optimizer_status_reflects_real_attempt():
    """duration_optimizer_success_ must reflect a real per-regime attempt, not a hardcoded
    True. Force minimize to fail and confirm the fallback fires."""
    model = make_tiny_model(K=2, F=1, L=1, Dmax=3, seed=4)
    X = make_tiny_data(model, T=10, seed=404)
    fwd = model._forward_pass(X)
    bwd = model._discrete_backward(fwd)
    old_mu, old_sigma = model.dur_mu.copy(), model.dur_sigma.copy()

    class _FakeResult:
        success = False
        x = np.array([0.0, 0.0])

    with patch("backbone.hssm.dual_model.minimize", return_value=_FakeResult()):
        model._update_discrete_params(bwd)

    assert model.duration_fallback_used_ is True
    assert all(v is False for v in model.duration_optimizer_success_.values())
    assert np.allclose(model.dur_mu, old_mu) and np.allclose(model.dur_sigma, old_sigma), (
        "fallback should keep the previous duration parameters")


def test_duration_optimizer_succeeds_normally():
    """Guards against a fallback path that is ALWAYS silently taken."""
    model = make_tiny_model(K=2, F=1, L=1, Dmax=3, seed=5)
    X = make_tiny_data(model, T=12, seed=505)
    fwd = model._forward_pass(X)
    bwd = model._discrete_backward(fwd)
    model._update_discrete_params(bwd)
    assert any(model.duration_optimizer_success_.values()), (
        "no regime's duration optimizer succeeded on ordinary data")
