"""WP3 item 2: numerical-health plumbing into HSSMResult and the hold-back switch (owner decision D3).

Rule (pre-declared in tests/PREDECLARED_CRITERIA_WP3.md section 2): insights are withheld when `fit_degraded` is True or `fit_state == NUMERICAL_FAILURE`.
No user-facing notice (nothing is added to `warnings` / `limitations`); one internal logging WARNING; the product layer must read outputs through
`HSSMResult.publishable_outputs()`.

What each test would catch if it were wrong:
  * flag ignored / rule inverted / NUMERICAL_FAILURE ignored   -> truth-table test, forced-fit test, injected-failure test
  * accessor returns data although withheld                    -> forced-fit test, injected-failure test, truth-table test
  * accessor withholds a clean fit                             -> clean-fit test
  * log removed, emitted twice, or emitted for a clean fit     -> forced-fit and clean-fit tests
  * health fields not carried from the model into the result   -> forced-fit and injected-failure tests
"""
import dataclasses
import logging
from unittest.mock import patch

import numpy as np
import pytest

from backbone.hssm.dual_model import DualTimescaleHSSM
from backbone.hssm.fitting import HSSMResult, _dual_health_fields, fit_hssm
from backbone.hssm.model import FitState, FittingConvergenceError
from data.synthetic_dual import generate_synthetic_dual_timescale_data

LOGGER_NAME = "backbone.hssm.fitting"


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _fit_dual_capturing_log(data, force_degraded=False):
    handler, lg = _ListHandler(), logging.getLogger(LOGGER_NAME)
    old_level = lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.DEBUG)
    real_init = DualTimescaleHSSM._init_params

    def init_then_break_one_regime(self, X_):          # the WP2 forcing method: regime 1 gets a negative observation variance
        real_init(self, X_)
        self.R_obs[1] = -5.0

    try:
        kwargs = dict(matrix=data.X, candidate_ks=(2,), n_initializations=2, max_iter=4, random_seed=42, allow_fast_test_fit=True,
                      timestamps=data.timestamps, user_id="user_hb", dataset_hash="hash_hb", model_type="dual_timescale", latent_dim=2)
        if force_degraded:
            with patch.object(DualTimescaleHSSM, "_init_params", init_then_break_one_regime):
                result = fit_hssm(**kwargs)
        else:
            result = fit_hssm(**kwargs)
    finally:
        lg.removeHandler(handler)
        lg.setLevel(old_level)
    return result, [r for r in handler.records if r.levelno >= logging.WARNING]


@pytest.fixture(scope="module")
def dual_data():
    return generate_synthetic_dual_timescale_data(n_regimes=2, n_features=3, latent_dim=2, n_segments=4, mean_dwell=15.0, seed=505)


@pytest.fixture(scope="module")
def clean(dual_data):
    return _fit_dual_capturing_log(dual_data)


@pytest.fixture(scope="module")
def forced(dual_data):
    return _fit_dual_capturing_log(dual_data, force_degraded=True)


# ------------------------------------------------------------------------------------------
def test_clean_dual_fit_carries_health_fields_is_not_withheld_and_returns_outputs(clean):
    result, warnings_logged = clean
    assert result.fit_degraded is False and result.fit_cholesky_fallbacks == 0 and result.fit_exceptions == []
    assert result.insights_withheld is False and result.withheld_reason is None
    out = result.publishable_outputs()
    assert out is not None and set(out) == {"p_t_estimate", "p_t_posterior", "m_t_posterior_mean", "m_t_uncertainty"}
    assert out["p_t_estimate"] is result.p_t_estimate and out["m_t_posterior_mean"] is result.m_t_posterior_mean
    assert out["m_t_posterior_mean"].shape == (len(result.p_t_estimate), 2)
    assert warnings_logged == []                      # no log noise on a healthy fit


def test_forced_negative_R_fit_is_degraded_withheld_and_logged_exactly_once(forced):
    result, warnings_logged = forced
    assert result.fit_degraded is True and result.fit_cholesky_fallbacks > 0
    assert result.insights_withheld is True
    assert "fit_degraded" in result.withheld_reason and str(result.fit_cholesky_fallbacks) in result.withheld_reason
    assert result.publishable_outputs() is None
    # raw fields stay available for internal diagnostics
    assert result.p_t_estimate is not None and result.p_t_posterior is not None and result.m_t_posterior_mean is not None
    # fit_state_ semantics are unchanged by this work: a degraded fit still ends as a normal state
    assert result.fit_state in (FitState.CONVERGED, FitState.MAX_ITER_REACHED, FitState.REJECTED_UPDATE)
    # exactly one internal log line, carrying the facts; nothing user-facing on the result
    assert len(warnings_logged) == 1 and warnings_logged[0].name == LOGGER_NAME and warnings_logged[0].levelno == logging.WARNING
    msg = warnings_logged[0].getMessage()
    assert "withheld" in msg and f"fit_cholesky_fallbacks={result.fit_cholesky_fallbacks}" in msg
    assert "random_seed=42" in msg and "user_id=user_hb" in msg and "dataset_hash=hash_hb" in msg and "exception_types=[]" in msg
    assert result.warnings == [] and result.limitations == []


def _fails_on_nth_call(original, n, exc):
    calls = {"n": 0}

    def wrapper(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == n:
            raise exc
        return original(self, *args, **kwargs)
    return wrapper


def test_injected_m_step_failure_is_carried_into_the_result_and_withheld(dual_data):
    m = DualTimescaleHSSM(n_regimes=2, n_features=3, latent_dim=2, max_duration=25, seed=3)
    wrapper = _fails_on_nth_call(DualTimescaleHSSM._m_step_continuous, 2, np.linalg.LinAlgError("injected m-step failure"))
    with patch.object(DualTimescaleHSSM, "_m_step_continuous", wrapper):
        m.fit(dual_data.X, n_iter=6)
    assert m.fit_state_ == FitState.NUMERICAL_FAILURE
    result = HSSMResult(model=m, fit_state=m.fit_state_, **_dual_health_fields(m))
    assert result.insights_withheld is True and "NUMERICAL_FAILURE" in result.withheld_reason
    assert result.publishable_outputs() is None
    assert len(result.fit_exceptions) == 1 and result.fit_exceptions[0]["stage"] == "M-step"
    assert result.fit_exceptions[0]["exception_type"] == "LinAlgError"
    assert result.fit_exceptions[0] is not m.fit_exceptions_[0]       # copied, not shared


def test_every_restart_failing_produces_no_result_at_all(dual_data):
    def always(self, *a, **k):
        raise np.linalg.LinAlgError("always")
    with patch.object(DualTimescaleHSSM, "_m_step_continuous", always):
        with pytest.raises(FittingConvergenceError):
            fit_hssm(matrix=dual_data.X, candidate_ks=(2,), n_initializations=2, max_iter=3, random_seed=1, allow_fast_test_fit=True,
                     timestamps=dual_data.timestamps, model_type="dual_timescale", latent_dim=2)


@pytest.mark.parametrize("degraded,state,withheld", [
    (False, FitState.CONVERGED, False),
    (True, FitState.CONVERGED, True),
    (False, FitState.NUMERICAL_FAILURE, True),
    (True, FitState.NUMERICAL_FAILURE, True),
])
def test_rule_truth_table(degraded, state, withheld):
    r = HSSMResult(model=None, fit_state=state, fit_degraded=degraded, fit_cholesky_fallbacks=2 if degraded else 0,
                   p_t_estimate=np.array([0, 1]), p_t_posterior=np.eye(2))
    assert r.insights_withheld is withheld
    assert (r.withheld_reason is not None) is withheld
    assert (r.publishable_outputs() is None) is withheld
    if degraded:
        assert "fit_degraded" in r.withheld_reason
    if state == FitState.NUMERICAL_FAILURE:
        assert "NUMERICAL_FAILURE" in r.withheld_reason


def test_withholding_follows_the_current_field_values_not_a_stored_copy():
    ok = HSSMResult(model=None, fit_state=FitState.CONVERGED)
    assert ok.insights_withheld is False
    assert dataclasses.replace(ok, fit_degraded=True, fit_cholesky_fallbacks=1).insights_withheld is True
    ok.fit_state = FitState.NUMERICAL_FAILURE          # mutated after construction
    assert ok.insights_withheld is True and ok.publishable_outputs() is None


def test_baseline_path_has_safe_defaults_and_is_not_withheld():
    np.random.seed(0)
    X = np.vstack([np.random.normal(0, 0.5, (30, 2)), np.random.normal(3, 0.5, (30, 2))])
    handler, lg = _ListHandler(), logging.getLogger(LOGGER_NAME)
    lg.addHandler(handler)
    try:
        # same call as tests/test_canonical_api.py (n_initializations=2 gave FittingConvergenceError for this baseline data: not a rule matter)
        r = fit_hssm(matrix=X, candidate_ks=(2,), n_initializations=10, max_iter=50, random_seed=42, allow_fast_test_fit=True,
                     timestamps=np.linspace(100.0, 160.0, 60), user_id="u", dataset_hash="h")
    finally:
        lg.removeHandler(handler)
    assert r.is_baseline_model is True
    assert r.fit_degraded is False and r.fit_cholesky_fallbacks == 0 and r.fit_exceptions == []
    assert r.insights_withheld is False and r.withheld_reason is None
    out = r.publishable_outputs()
    assert out is not None and out["m_t_posterior_mean"] is None and out["p_t_estimate"] is r.p_t_estimate
    assert [x for x in handler.records if x.levelno >= logging.WARNING] == []
