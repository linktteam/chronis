"""Contract tests for the REAL-class wiring of the ported inference/EM code.

The ported package tests exercise the algorithms through a factory that bypasses the real
`fit()` and `_init_params`. These tests cover what is specific to the real class:
  - estimate_states keeps returning the FILTERED regime posterior (an owner decision, not
    changed by the port) while the continuous mean/std come from the new smoother;
  - the rollback snapshot covers every array the M-step changes AND the duration/transition
    status fields;
  - the status fields start from the real `_init_params` state ({k: True}) and are flipped by a
    real optimizer failure;
  - A_slow, durations and pi really move through the real fit() (the old M-step never touched
    A_slow, dur_mu, dur_sigma);
  - numerical failures inside the new E/M steps end in FitState.NUMERICAL_FAILURE rather than
    an exception escaping fit() (callers such as fit_dual_with_random_restarts rely on that).
"""
import inspect
from unittest.mock import patch

import numpy as np

from backbone.hssm.dual_model import DualTimescaleHSSM
from backbone.hssm.model import FitState


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


def _real_model(K=3, F=2, L=1, D=6, seed=0):
    return DualTimescaleHSSM(n_regimes=K, n_features=F, latent_dim=L, max_duration=D, seed=seed)


def test_public_signatures_are_unchanged():
    sig = inspect.signature(DualTimescaleHSSM.fit)
    assert list(sig.parameters) == ["self", "X", "n_iter", "tol", "verbose", "timestamps"]
    assert sig.parameters["n_iter"].default == 50
    assert sig.parameters["tol"].default == 1e-4
    assert sig.parameters["verbose"].default is False
    assert sig.parameters["timestamps"].default is None
    sig = inspect.signature(DualTimescaleHSSM.__init__)
    assert list(sig.parameters) == ["self", "n_regimes", "n_features", "latent_dim", "max_duration",
                                    "seed", "duration_unit"]
    assert sig.parameters["max_duration"].default == 40 and sig.parameters["duration_unit"].default == "sessions"


def test_estimate_states_returns_filtered_posterior_and_smoother_mean():
    X = _three_regime_data(30, seed=11)
    model = _real_model(seed=3).fit(X, n_iter=6)
    m_t, m_t_std, p_t, p_t_post = model.estimate_states(X)

    fwd = model._forward_pass(X)
    bwd = model._discrete_backward(fwd)
    sm = model._switching_smoother(fwd, bwd)

    T, K, L = len(X), model.K, model.L
    assert m_t.shape == (T, L) and m_t_std.shape == (T, L) and p_t.shape == (T,) and p_t_post.shape == (T, K)
    # the regime outputs are the FILTERED posterior, exactly
    assert np.array_equal(p_t_post, fwd.regime_post_filtered)
    assert np.array_equal(p_t, np.argmax(fwd.regime_post_filtered, axis=1))
    # ... and on this data they genuinely differ from the smoothed posterior, so the test
    # would fail if the smoothed one were returned instead
    assert not np.allclose(p_t_post, bwd.regime_posterior_smoothed, atol=1e-6)
    # the continuous outputs come from the switching smoother
    assert np.array_equal(m_t, sm.smoothed_m)
    assert np.allclose(m_t_std, np.sqrt(np.maximum(np.stack([np.diag(P) for P in sm.smoothed_P]), 1e-8)))


def test_estimate_states_does_not_overwrite_log_likelihood():
    X = _three_regime_data(30, seed=11)
    model = _real_model(seed=3).fit(X, n_iter=6)
    ll = model.log_likelihood_
    X_other = X + np.random.default_rng(0).normal(scale=0.5, size=X.shape)
    model.estimate_states(X_other)
    assert model.log_likelihood_ == ll, "estimate_states changed the fitted log_likelihood_"


def test_fit_moves_A_slow_durations_and_pi_through_the_real_fit():
    X = _three_regime_data(45, seed=303)
    model = _real_model(seed=3)
    captured = {}
    real_init = model._init_params

    def spy_init(X_):
        real_init(X_)
        captured.update(A_slow=model.A_slow.copy(), dur_mu=model.dur_mu.copy(),
                        dur_sigma=model.dur_sigma.copy(), pi=model.pi.copy())

    model._init_params = spy_init
    model.fit(X, n_iter=15)
    assert not np.allclose(model.A_slow, captured["A_slow"], atol=1e-3), "A_slow never moved"
    assert not (np.allclose(model.dur_mu, captured["dur_mu"], atol=1e-3)
                and np.allclose(model.dur_sigma, captured["dur_sigma"], atol=1e-3)), "durations never moved"
    assert not np.allclose(model.pi, captured["pi"], atol=1e-3), "pi never moved"


def test_duration_status_starts_from_real_init_and_flips_on_forced_failure():
    X = _three_regime_data(30, seed=5)
    model = _real_model(K=3, F=2, L=1, D=4, seed=2)
    model._init_params(X)
    # real init state (NOT neutralised): all True / no fallback
    assert model.duration_optimizer_success_ == {0: True, 1: True, 2: True}
    assert model.duration_fallback_used_ is False
    fwd = model._forward_pass(X)
    bwd = model._discrete_backward(fwd)
    old_mu, old_sigma = model.dur_mu.copy(), model.dur_sigma.copy()

    class _FailedResult:
        success = False
        x = np.array([0.0, 0.0])

    with patch("backbone.hssm.dual_model.minimize", return_value=_FailedResult()):
        model._update_discrete_params(bwd)

    assert model.duration_fallback_used_ is True
    assert model.duration_optimizer_success_ == {0: False, 1: False, 2: False}
    assert np.allclose(model.dur_mu, old_mu) and np.allclose(model.dur_sigma, old_sigma)


def test_rollback_restores_params_and_status_fields():
    """Force an accepted-then-bad M-step. After REJECTED_UPDATE every array the M-step touches
    and every status field must be what it was BEFORE that M-step."""
    X = _three_regime_data(30, seed=9)
    model = _real_model(seed=1)
    assert model.transition_fallback_rows_ == set()           # attribute exists on a fresh model
    state = {"calls": 0, "expected": None}
    orig_discrete = model._update_discrete_params
    orig_cont = model._m_step_continuous

    def discrete_then_corrupt(bwd):
        state["calls"] += 1
        if state["calls"] == 2:
            state["expected"] = dict(
                A_slow=model.A_slow.copy(), dur_mu=model.dur_mu.copy(), dur_sigma=model.dur_sigma.copy(),
                pi=model.pi.copy(), status=dict(model.duration_optimizer_success_),
                fallback=model.duration_fallback_used_, rows=set(model.transition_fallback_rows_))
            orig_discrete(bwd)
            # corrupt every field the rollback is supposed to repair
            model.A_slow = model.A_slow[::-1].copy()
            model.dur_mu = model.dur_mu + 5.0
            model.dur_sigma = model.dur_sigma * 2.0
            model.pi = model.pi[::-1] * 0 + np.array([0.98, 0.01, 0.01])
            model.duration_optimizer_success_ = {0: False, 1: False, 2: False}
            model.duration_fallback_used_ = True
            model.transition_fallback_rows_ = {0, 1, 2}
            return
        return orig_discrete(bwd)

    def cont_sabotage(X_, bwd, sm):
        if state["calls"] == 2:
            model.R_obs = model.R_obs * 1e6
            return
        return orig_cont(X_, bwd, sm)

    model._update_discrete_params = discrete_then_corrupt
    model._m_step_continuous = cont_sabotage
    with patch("warnings.warn"):
        model.fit(X, n_iter=10)

    assert model.fit_state_ == FitState.REJECTED_UPDATE, model.fit_state_
    exp = state["expected"]
    assert exp is not None
    assert np.array_equal(model.A_slow, exp["A_slow"])
    assert np.array_equal(model.dur_mu, exp["dur_mu"]) and np.array_equal(model.dur_sigma, exp["dur_sigma"])
    assert np.array_equal(model.pi, exp["pi"])
    assert model.duration_optimizer_success_ == exp["status"]
    assert model.duration_fallback_used_ == exp["fallback"]
    assert model.transition_fallback_rows_ == exp["rows"]
    assert not np.any(model.R_obs > 1e5)


def test_numerical_failure_in_E_step_does_not_escape_fit():
    X = _three_regime_data(20, seed=4)
    model = _real_model(seed=2)
    with patch.object(DualTimescaleHSSM, "_switching_smoother", side_effect=np.linalg.LinAlgError("boom")):
        model.fit(X, n_iter=5)           # must not raise
    assert model.fit_state_ == FitState.NUMERICAL_FAILURE
    assert model._is_fitted is True


def test_numerical_failure_in_M_step_restores_pre_step_parameters():
    X = _three_regime_data(20, seed=4)
    model = _real_model(seed=2)
    seen = {"n": 0, "before": None}
    orig = model._m_step_continuous

    def boom_on_second(X_, bwd, sm):
        seen["n"] += 1
        if seen["n"] == 2:
            seen["before"] = (model.A_fast.copy(), model.C_obs.copy())
            model.A_fast = model.A_fast + 9.0        # half-applied update, then failure
            raise np.linalg.LinAlgError("boom")
        return orig(X_, bwd, sm)

    model._m_step_continuous = boom_on_second
    model.fit(X, n_iter=6)
    assert model.fit_state_ == FitState.NUMERICAL_FAILURE
    assert np.array_equal(model.A_fast, seen["before"][0]), "half-applied update was not rolled back"
