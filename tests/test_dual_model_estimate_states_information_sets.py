"""Ledger A6: `estimate_states` mixes two information sets, and that must be stated, tested, and not changed by accident.

  * p_t, p_t_post (regime) are FILTERED: row t uses only X[0..t].
  * m_t, m_t_std (fast state) are SMOOTHED: row t uses every observation in X.

Pre-declared criteria: tests/PREDECLARED_CRITERIA_WP2.md section 3.
  identical-prefix test: for n in (20, 35), estimate_states(X[:n]) regime outputs equal the first n rows of
  estimate_states(X) (allclose, atol 1e-12, rtol 0), while smoothed_m[:n] differs by more than 1e-6 (max abs).

Negative controls (must make the same checks FAIL): an engine that returns the SMOOTHED regime posterior, and an
engine whose `m_t` is built from filtered quantities only (no future information).
"""
import copy

import numpy as np
import pytest

from tests.utils.synthetic_dual_fits import fitted

PREFIXES = (20, 35)
ATOL_REGIME = 1e-12
MIN_SMOOTHED_DIFFERENCE = 1e-6


class _EngineSmoothedRegimePosterior:
    """Control 1: regime outputs come from the SMOOTHED posterior (uses future data)."""
    @staticmethod
    def estimate_states(model, X):
        m_t, m_std, _, _ = type(model).estimate_states(model, X)
        fwd = model._forward_pass(X)
        bwd = model._discrete_backward(fwd)
        post = bwd.regime_posterior_smoothed
        return m_t, m_std, np.argmax(post, axis=1), post


class _EngineFilteredMean:
    """Control 2: m_t is the filtered mixture mean sum_k P(k_t=k | y_1:t) * m_filt[t,k] (no future data)."""
    @staticmethod
    def estimate_states(model, X):
        _, m_std, p_t, post = type(model).estimate_states(model, X)
        fwd = model._forward_pass(X)
        m_t = np.einsum("tk,tkl->tl", fwd.regime_post_filtered, fwd.m_filt)
        return m_t, m_std, p_t, post


def _regime_prefix_difference(estimate, model, X, n):
    _, _, p_full, post_full = estimate(model, X)
    _, _, p_pre, post_pre = estimate(model, X[:n])
    return float(np.max(np.abs(post_pre - post_full[:n]))), bool(np.array_equal(p_pre, p_full[:n]))


def _smoothed_prefix_difference(estimate, model, X, n):
    m_full, _, _, _ = estimate(model, X)
    m_pre, _, _, _ = estimate(model, X[:n])
    return float(np.max(np.abs(m_pre - m_full[:n])))


def _real(model, X):
    return type(model).estimate_states(model, X)


@pytest.fixture(scope="module")
def fitted_model_and_data():
    data, model, *_ = fitted("real", "A")
    return copy.deepcopy(model), data.X


@pytest.mark.parametrize("n", PREFIXES)
def test_regime_outputs_use_only_past_data(fitted_model_and_data, n):
    model, X = fitted_model_and_data
    diff, same_map = _regime_prefix_difference(_real, model, X, n)
    assert diff <= ATOL_REGIME, f"prefix n={n}: filtered regime posterior changed by {diff:.3e} when later rows were appended"
    assert same_map


@pytest.mark.parametrize("n", PREFIXES)
def test_fast_state_mean_uses_future_data(fitted_model_and_data, n):
    model, X = fitted_model_and_data
    diff = _smoothed_prefix_difference(_real, model, X, n)
    assert diff > MIN_SMOOTHED_DIFFERENCE, f"prefix n={n}: smoothed m_t did not change when later rows were appended (max abs {diff:.3e})"


@pytest.mark.parametrize("n", PREFIXES)
def test_control_smoothed_regime_posterior_is_caught_by_the_prefix_check(fitted_model_and_data, n):
    model, X = fitted_model_and_data
    diff, _ = _regime_prefix_difference(_EngineSmoothedRegimePosterior.estimate_states, model, X, n)
    assert diff > ATOL_REGIME, f"prefix n={n}: a regime output that uses the future would still pass the prefix check (diff {diff:.3e})"


@pytest.mark.parametrize("n", PREFIXES)
def test_control_filtered_mean_is_caught_by_the_uses_future_check(fitted_model_and_data, n):
    model, X = fitted_model_and_data
    diff = _smoothed_prefix_difference(_EngineFilteredMean.estimate_states, model, X, n)
    assert diff <= MIN_SMOOTHED_DIFFERENCE, (
        f"prefix n={n}: a causal m_t changed by {diff:.3e} between prefix and full series; "
        "the 'uses future data' check would not tell it from a smoother")


def test_estimate_states_documents_the_mixed_information_sets():
    from backbone.hssm.dual_model import DualTimescaleHSSM
    doc = DualTimescaleHSSM.estimate_states.__doc__
    assert "SMOOTHED" in doc and "FILTERED" in doc and "INFORMATION SETS" in doc
