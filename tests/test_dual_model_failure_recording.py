"""Ledger A7 (flag the silent -1e6 Cholesky fallback) and N21 (record every exception caught in fit()).

Criteria were pre-declared in tests/PREDECLARED_CRITERIA_WP2.md (commit made before this file existed).

What each test would catch if it were wrong:
  * a hard-coded `fit_degraded_ = False`  -> the forced-firing tests fail
  * a hard-coded `fit_degraded_ = True`   -> the clean-fit tests fail
  * a counter that is not incremented / incremented on every call -> the exact-count test fails
  * an `except` that swallows without recording                    -> the injected-failure tests fail
  * an `except` that records but never clears between fits          -> the reset test fails
The new-behaviour tests also fail on the pre-change code (the attributes do not exist there).
"""
from unittest.mock import patch

import numpy as np
import pytest

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


def _model(seed=0):
    return DualTimescaleHSSM(n_regimes=3, n_features=2, latent_dim=1, max_duration=6, seed=seed)


def _fails_on_nth_call(original, n, exc):
    """Wrapper that behaves like `original` except that its n-th call raises `exc`."""
    calls = {"n": 0}

    def wrapper(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == n:
            raise exc
        return original(self, *args, **kwargs)
    return wrapper


# ------------------------------------------------------------------------------------------
# A7: the Cholesky fallback
# ------------------------------------------------------------------------------------------
def test_fallback_keeps_its_return_values_and_is_counted_exactly():
    X = _three_regime_data(20, 0)
    m = _model()
    m._init_params(X)
    m.R_obs[1] = -5.0  # invalid on purpose: S = C P C^T + R is not positive definite for regime 1
    m_prior, P_prior = np.zeros(1), np.eye(1)
    assert m.cholesky_fallback_count_ == 0

    for n_calls in (1, 2, 3):
        m_up, P_up, ll = m._kalman_update_regime(X[0], m_prior, P_prior, 1)
        assert ll == -1e6                                   # numerics unchanged
        assert np.array_equal(m_up, m_prior) and np.array_equal(P_up, P_prior)
        assert m.cholesky_fallback_count_ == n_calls        # exactly one count per firing

    before = m.cholesky_fallback_count_
    _, _, ll_ok = m._kalman_update_regime(X[0], m_prior, P_prior, 0)   # valid regime
    assert ll_ok > -1e5 and m.cholesky_fallback_count_ == before       # a valid call never counts

    # all-features-missing is not a fallback either
    m._kalman_update_regime(np.full(2, np.nan), m_prior, P_prior, 1)
    assert m.cholesky_fallback_count_ == before


def test_forward_pass_counts_firings_only_when_a_regime_is_invalid():
    X = _three_regime_data(20, 1)
    clean = _model()
    clean._init_params(X)
    clean._forward_pass(X)
    assert clean.cholesky_fallback_count_ == 0

    broken = _model()
    broken._init_params(X)
    broken.R_obs[2] = -5.0
    broken._forward_pass(X)
    assert broken.cholesky_fallback_count_ > 0


def test_fit_flags_the_fit_as_degraded_when_the_fallback_fires_inside_the_loop():
    X = _three_regime_data(30, 2)
    m = _model()
    real_init = DualTimescaleHSSM._init_params

    def init_then_break_one_regime(self, X_):
        real_init(self, X_)
        self.R_obs[1] = -5.0

    with patch.object(DualTimescaleHSSM, "_init_params", init_then_break_one_regime):
        m.fit(X, n_iter=4)
    assert m.fit_cholesky_fallbacks_ > 0
    assert m.fit_degraded_ is True
    assert m.fit_cholesky_fallbacks_ == m.cholesky_fallback_count_   # fresh model: all firings were inside fit()


def test_clean_fit_is_not_degraded_and_the_flag_clears_on_the_next_clean_fit():
    X = _three_regime_data(30, 2)
    clean = _model()
    clean.fit(X, n_iter=4)
    assert clean.fit_cholesky_fallbacks_ == 0 and clean.fit_degraded_ is False and clean.cholesky_fallback_count_ == 0

    # a degraded fit followed by a clean fit on the SAME object: the per-fit flag must not stay stale
    m = _model()
    real_init = DualTimescaleHSSM._init_params

    def init_then_break_one_regime(self, X_):
        real_init(self, X_)
        self.R_obs[1] = -5.0

    with patch.object(DualTimescaleHSSM, "_init_params", init_then_break_one_regime):
        m.fit(X, n_iter=3)
    assert m.fit_degraded_ is True
    cumulative = m.cholesky_fallback_count_
    m.fit(X, n_iter=3)                       # real _init_params again: clean
    assert m.fit_degraded_ is False and m.fit_cholesky_fallbacks_ == 0
    assert m.cholesky_fallback_count_ == cumulative   # the cumulative counter keeps the history


def test_oracle_battery_models_never_fire_the_fallback():
    """Expected-zero count on every model of the 10-config reference battery (forward pass + exact oracle,
    which share the Kalman primitive). A non-zero count would mean a test above compared a substituted likelihood."""
    from tests.oracle.generate_reference_battery import CONFIGS
    from tests.oracle.tiny_exact_oracle import run_tiny_exact_oracle
    from tests.utils.model_factory import make_tiny_model, make_tiny_data
    counts = []
    for (K, F, L, Dmax, T, seed, nan_prob) in CONFIGS:
        model = make_tiny_model(K=K, F=F, L=L, Dmax=Dmax, seed=seed)
        X = make_tiny_data(model, T, seed=seed + 100, nan_prob=nan_prob)
        fwd = model._forward_pass(X)
        bwd = model._discrete_backward(fwd)
        model._switching_smoother(fwd, bwd)
        run_tiny_exact_oracle(model, X)
        counts.append(model.cholesky_fallback_count_)
    assert counts == [0] * len(CONFIGS), counts


# ------------------------------------------------------------------------------------------
# N21: exceptions caught inside fit()
# ------------------------------------------------------------------------------------------
def test_injected_linalg_error_in_E_step_is_recorded():
    X = _three_regime_data(30, 3)
    m = _model()
    wrapper = _fails_on_nth_call(DualTimescaleHSSM._switching_smoother, 3, np.linalg.LinAlgError("injected e-step failure"))
    with patch.object(DualTimescaleHSSM, "_switching_smoother", wrapper):
        m.fit(X, n_iter=8)
    assert m.fit_state_ == FitState.NUMERICAL_FAILURE
    assert len(m.fit_exceptions_) == 1
    rec = m.fit_exceptions_[0]
    assert rec["stage"] == "E-step" and rec["iteration"] == 2
    assert rec["exception_type"] == "LinAlgError" and rec["message"] == "injected e-step failure"
    assert isinstance(rec["location"], str) and rec["location"]


def test_injected_linalg_error_in_M_step_is_recorded():
    X = _three_regime_data(30, 3)
    m = _model()
    wrapper = _fails_on_nth_call(DualTimescaleHSSM._m_step_continuous, 2, np.linalg.LinAlgError("injected m-step failure"))
    with patch.object(DualTimescaleHSSM, "_m_step_continuous", wrapper):
        m.fit(X, n_iter=8)
    assert m.fit_state_ == FitState.NUMERICAL_FAILURE
    assert len(m.fit_exceptions_) == 1
    rec = m.fit_exceptions_[0]
    assert rec["stage"] == "M-step" and rec["iteration"] == 1
    assert rec["exception_type"] == "LinAlgError" and rec["message"] == "injected m-step failure"


def test_a_genuine_shape_bug_raised_as_ValueError_is_not_hidden_behind_NUMERICAL_FAILURE():
    """The except clause also catches ValueError, which numpy raises for ordinary shape bugs. The recorded
    message and location are what lets a reader tell such a bug from a numerical failure."""
    X = _three_regime_data(30, 3)
    m = _model()

    def buggy_m_step(self, X_, bwd, smoother):
        return np.ones((2, 3)) @ np.ones((4, 5))   # real numpy ValueError (core-dimension mismatch)

    with patch.object(DualTimescaleHSSM, "_m_step_continuous", buggy_m_step):
        m.fit(X, n_iter=5)
    assert m.fit_state_ == FitState.NUMERICAL_FAILURE
    assert len(m.fit_exceptions_) == 1
    rec = m.fit_exceptions_[0]
    assert rec["exception_type"] == "ValueError" and rec["stage"] == "M-step"
    assert "mismatch" in rec["message"] or "dimension" in rec["message"]
    assert "buggy_m_step" in rec["location"]          # points at the frame that raised


def test_exception_types_outside_the_caught_tuple_still_propagate_and_are_not_recorded():
    X = _three_regime_data(30, 3)
    m = _model()
    wrapper = _fails_on_nth_call(DualTimescaleHSSM._forward_pass, 2, RuntimeError("not a numerical failure"))
    with patch.object(DualTimescaleHSSM, "_forward_pass", wrapper):
        with pytest.raises(RuntimeError, match="not a numerical failure"):
            m.fit(X, n_iter=5)
    assert m.fit_exceptions_ == []
    # the per-fit bookkeeping was still finalised by the `finally`
    assert m.fit_cholesky_fallbacks_ == 0 and m.fit_degraded_ is False


def test_clean_fit_records_no_exceptions_and_the_list_is_reset_by_the_next_fit():
    X = _three_regime_data(30, 3)
    m = _model()
    m.fit(X, n_iter=3)
    assert m.fit_exceptions_ == []

    wrapper = _fails_on_nth_call(DualTimescaleHSSM._m_step_continuous, 1, np.linalg.LinAlgError("injected"))
    with patch.object(DualTimescaleHSSM, "_m_step_continuous", wrapper):
        m.fit(X, n_iter=3)
    assert len(m.fit_exceptions_) == 1

    m.fit(X, n_iter=3)                      # clean again: the earlier record must not linger
    assert m.fit_exceptions_ == []
    assert m.fit_state_ != FitState.NUMERICAL_FAILURE
