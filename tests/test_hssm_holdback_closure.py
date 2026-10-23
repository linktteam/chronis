"""WP4 item 1: closing the hold-back gaps (ledger A7 / decision D3). Criteria: tests/PREDECLARED_CRITERIA_WP4.md section 4.

What each test would catch if it were wrong (the mutation set in integration_report/wp4/scripts/mutation_wp4.py checks that):
  * gated call changes a healthy fit                           -> test_gated_healthy_baseline_call_is_bit_identical_to_the_old_function
  * gated call hands on a degraded model silently              -> test_gated_call_raises_on_a_degraded_dual_fit_and_not_on_a_healthy_one
  * result function raises / ignores model_type                -> test_gated_result_function_*
  * restart selection prefers a degraded restart               -> test_selection_*
  * aliases leak withheld data                                 -> test_aliases_*
  * late / states-pass firing not covered                      -> test_late_firing_*, test_firing_inside_the_states_pass_*
  * log emitted twice or not at all                            -> test_late_firing_* and test_build_time_withheld_result_is_not_logged_again_on_read
"""
import copy
import enum
import inspect
import logging
from unittest.mock import patch

import numpy as np
import pytest

from backbone.hssm import fitting as fitting_module
from backbone.hssm.config import DEFAULT_COLD_START_CONFIG, ColdStartConfig
from backbone.hssm.dual_model import DualTimescaleHSSM
from backbone.hssm.fitting import HSSMResult, fit_dual_with_random_restarts, fit_hssm
from backbone.hssm.gating import ColdStartError, fit_hssm_gated, fit_hssm_gated_result
from backbone.hssm.model import FitState, FittingConvergenceError, HSSMError, InsightsWithheldError
from data.synthetic_dual import generate_synthetic_dual_timescale_data

LOGGER_NAME = "backbone.hssm.fitting"
FAST = dict(n_initializations=2, max_iter=4, random_seed=42, allow_fast_test_fit=True)


# ------------------------------------------------------------------ helpers
class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


class _capture_warnings:
    """Collects WARNING+ records on the fitting logger while the block runs."""

    def __enter__(self):
        self.handler, self.lg = _ListHandler(), logging.getLogger(LOGGER_NAME)
        self.old_level = self.lg.level
        self.lg.addHandler(self.handler)
        self.lg.setLevel(logging.DEBUG)
        return self

    def __exit__(self, *exc):
        self.lg.removeHandler(self.handler)
        self.lg.setLevel(self.old_level)

    @property
    def warnings(self):
        return [r for r in self.handler.records if r.levelno >= logging.WARNING]


def _break_one_regime(real_init):
    def init_then_break(self, X_):                 # the WP2 forcing method: regime 1 gets a negative observation variance
        real_init(self, X_)
        self.R_obs[1] = -5.0
    return init_then_break


def same_state(a, b):
    """Exact (bit-level) equality of two objects made of arrays, floats, ints, strings, enums, lists, tuples, dicts and plain objects."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (a != a and b != b)
    if isinstance(a, enum.Enum) or isinstance(b, enum.Enum):
        return a is b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    if hasattr(a, "__dict__") and hasattr(b, "__dict__") and not isinstance(a, type) and type(a) is type(b):
        return same_state(vars(a), vars(b))
    if isinstance(a, np.random.Generator) and isinstance(b, np.random.Generator):
        return a.bit_generator.state == b.bit_generator.state
    return type(a) is type(b) and a == b


def _old_fit_hssm_gated(X, n_regimes=3, n_features=1, n_present_sessions=None, config=DEFAULT_COLD_START_CONFIG, n_init=10, max_duration=45,
                        base_seed=0, timestamps=None, min_present_sessions=None):
    """VERBATIM copy of fit_hssm_gated as it was at commit 0661388 (before WP4), kept as the reference for the bit-identity test."""
    if n_features <= 0:
        raise ValueError("Feature count must be strictly greater than zero (F > 0)")

    if min_present_sessions is not None:
        config = ColdStartConfig(min_present_sessions=min_present_sessions)

    from backbone.hssm.fitting import fit_hssm

    res = fit_hssm(
        matrix=X,
        candidate_ks=(n_regimes,),
        n_initializations=n_init,
        random_seed=base_seed,
        allow_fast_test_fit=(n_init < 10),
        timestamps=timestamps,
        config=config,
        n_present_sessions=n_present_sessions,
        max_duration=max_duration,
    )
    return res.model, res.convergence_metadata.get("run_log", [])


@pytest.fixture(scope="module")
def dual_data():
    return generate_synthetic_dual_timescale_data(n_regimes=2, n_features=3, latent_dim=2, n_segments=4, mean_dwell=15.0, seed=505)


@pytest.fixture(scope="module")
def baseline_X():
    rs = np.random.RandomState(0)
    return np.vstack([rs.normal(0, 0.5, (30, 2)), rs.normal(3, 0.5, (30, 2))])


@pytest.fixture(scope="module")
def clean_dual(dual_data):
    return fit_hssm(matrix=dual_data.X, candidate_ks=(2,), timestamps=dual_data.timestamps, user_id="u", dataset_hash="h",
                    model_type="dual_timescale", latent_dim=2, **FAST)


@pytest.fixture(scope="module")
def forced_dual(dual_data):
    with patch.object(DualTimescaleHSSM, "_init_params", _break_one_regime(DualTimescaleHSSM._init_params)):
        return fit_hssm(matrix=dual_data.X, candidate_ks=(2,), timestamps=dual_data.timestamps, user_id="u", dataset_hash="h",
                        model_type="dual_timescale", latent_dim=2, **FAST)


def _as_dual_with_few_iterations(real_fit_hssm):
    """fit_hssm_gated / fit_hssm_gated_result do not take max_iter; this wrapper makes the dual fit short for the tests."""
    def wrapper(**kwargs):
        kwargs = dict(kwargs, model_type="dual_timescale", latent_dim=2, max_iter=4)
        return real_fit_hssm(**kwargs)
    return wrapper


# ------------------------------------------------------------------ the comparator itself (negative control)
def test_state_comparator_detects_differences(baseline_X):
    m1, _ = _old_fit_hssm_gated(baseline_X, n_regimes=2, n_features=2, n_init=10, base_seed=1)
    m2, _ = _old_fit_hssm_gated(baseline_X, n_regimes=2, n_features=2, n_init=10, base_seed=2)
    assert same_state(m1, m1) and same_state(m1.__dict__, dict(vars(m1)))
    assert not same_state(m1, m2)                                      # a different seed gives different parameters
    assert not same_state(np.array([1.0, 2.0]), np.array([1.0, 2.0 + 1e-15]))
    assert same_state(np.array([np.nan, 1.0]), np.array([np.nan, 1.0]))


# ------------------------------------------------------------------ item 1a: gated entry points
def test_gated_healthy_baseline_call_is_bit_identical_to_the_old_function(baseline_X):
    kw = dict(n_regimes=2, n_features=2, n_init=10, base_seed=1)
    old_model, old_log = _old_fit_hssm_gated(baseline_X, **kw)
    new_model, new_log = fit_hssm_gated(baseline_X, **kw)
    assert type(new_model) is type(old_model)
    assert same_state(old_model, new_model) and same_state(old_log, new_log) and len(new_log) == 10
    result = fit_hssm_gated_result(baseline_X, **kw)                    # the full result carries the same model and run log
    assert isinstance(result, HSSMResult) and not result.insights_withheld
    assert same_state(old_model, result.model) and same_state(old_log, result.convergence_metadata["run_log"])


def test_gated_signature_and_return_type_are_unchanged():
    sig = inspect.signature(fit_hssm_gated)
    assert list(sig.parameters) == ["X", "n_regimes", "n_features", "n_present_sessions", "config", "n_init", "max_duration", "base_seed",
                                    "timestamps", "min_present_sessions"]
    assert [p.default for p in sig.parameters.values()][1:2] == [3] and sig.parameters["n_init"].default == 10
    extra = [n for n in inspect.signature(fit_hssm_gated_result).parameters if n not in sig.parameters]
    assert extra == ["model_type", "latent_dim"]


def test_gated_result_function_returns_a_withheld_result_without_raising_and_honours_model_type(dual_data):
    with patch.object(DualTimescaleHSSM, "_init_params", _break_one_regime(DualTimescaleHSSM._init_params)), \
         patch.object(fitting_module, "fit_hssm", _as_dual_with_few_iterations(fitting_module.fit_hssm)):
        degraded = fit_hssm_gated_result(dual_data.X, n_regimes=2, n_features=3, n_init=2, base_seed=42, timestamps=dual_data.timestamps,
                                         model_type="dual_timescale")
    assert isinstance(degraded.model, DualTimescaleHSSM) and degraded.insights_withheld and degraded.publishable_outputs() is None
    assert "fit_degraded" in degraded.withheld_reason
    # negative control: without forcing, the same call is healthy
    with patch.object(fitting_module, "fit_hssm", _as_dual_with_few_iterations(fitting_module.fit_hssm)):
        healthy = fit_hssm_gated_result(dual_data.X, n_regimes=2, n_features=3, n_init=2, base_seed=42, timestamps=dual_data.timestamps,
                                        model_type="dual_timescale")
    assert not healthy.insights_withheld and healthy.publishable_outputs() is not None


class _Stop(Exception):
    pass


def test_gated_result_function_hands_model_type_and_latent_dim_to_fit_hssm(baseline_X):
    """The other tests force the model type through a wrapper (so that max_iter can be small); this one looks at what is actually passed."""
    seen = []

    def spy(**kwargs):
        seen.append(kwargs)
        raise _Stop

    with patch.object(fitting_module, "fit_hssm", spy):
        for call_kwargs, expected in [({}, ("baseline", 2)), (dict(model_type="dual_timescale"), ("dual_timescale", 2)),
                                      (dict(model_type="dual_timescale", latent_dim=3), ("dual_timescale", 3))]:
            with pytest.raises(_Stop):
                fit_hssm_gated_result(baseline_X, n_regimes=2, n_features=2, n_init=10, base_seed=7, **call_kwargs)
            assert (seen[-1]["model_type"], seen[-1]["latent_dim"]) == expected
        with pytest.raises(_Stop):                                       # the legacy function never changes the model type
            fit_hssm_gated(baseline_X, n_regimes=2, n_features=2, n_init=10, base_seed=7)
        assert (seen[-1]["model_type"], seen[-1]["latent_dim"]) == ("baseline", 2)
    assert all(k["candidate_ks"] == (2,) and k["n_initializations"] == 10 and k["random_seed"] == 7 and k["allow_fast_test_fit"] is False for k in seen)


def test_gated_result_function_still_gates_cold_start_and_feature_count(baseline_X):
    with pytest.raises(ColdStartError):
        fit_hssm_gated_result(baseline_X[:10], n_regimes=2, n_features=2, n_init=10)
    with pytest.raises(ValueError):
        fit_hssm_gated_result(baseline_X, n_regimes=2, n_features=0)


def test_gated_call_raises_on_a_degraded_dual_fit_and_not_on_a_healthy_one(dual_data):
    kw = dict(n_regimes=2, n_features=3, n_init=2, base_seed=42, timestamps=dual_data.timestamps)
    with patch.object(fitting_module, "fit_hssm", _as_dual_with_few_iterations(fitting_module.fit_hssm)):
        model, run_log = fit_hssm_gated(dual_data.X, **kw)               # negative control: healthy dual fit passes through
        assert isinstance(model, DualTimescaleHSSM) and len(run_log) == 2 and not model.fit_degraded_
        with patch.object(DualTimescaleHSSM, "_init_params", _break_one_regime(DualTimescaleHSSM._init_params)):
            with pytest.raises(InsightsWithheldError) as info:
                fit_hssm_gated(dual_data.X, **kw)
    err = info.value
    assert isinstance(err, HSSMError) and isinstance(err, RuntimeError)
    assert "fit_degraded" in err.reason and err.reason in str(err)
    assert isinstance(err.result, HSSMResult) and err.result.withheld_reason == err.reason
    assert err.result.p_t_estimate is not None                           # raw fields stay readable on the carried result


# ------------------------------------------------------------------ item 1b: restart selection
def _run_selection(script, X):
    """script: list of (log_likelihood, degraded) per restart, in order. fit() is replaced by one that sets those values."""
    created = []

    def fake_fit(self, X_, n_iter=50, tol=1e-4, verbose=False, timestamps=None):
        i = len(created)
        created.append(self)
        ll, degraded = script[i]
        self.fit_state_, self.log_likelihood_, self.n_iter_ = FitState.CONVERGED, ll, 3
        self.fit_degraded_, self.fit_cholesky_fallbacks_ = degraded, (2 if degraded else 0)
        return self

    with patch.object(DualTimescaleHSSM, "fit", fake_fit):
        best, run_log = fit_dual_with_random_restarts(X, n_regimes=2, n_features=3, n_init=len(script), bypass_init_gate=True)
    return best, created, run_log


def test_selection_prefers_a_healthy_restart_even_when_the_degraded_one_has_the_higher_loglik(dual_data):
    best, created, run_log = _run_selection([(-100.0, False), (-50.0, True), (-120.0, False), (-10.0, True)], dual_data.X)
    assert best is created[0] and best.log_likelihood_ == -100.0 and not best.fit_degraded_
    assert len(run_log) == 4 and [r["log_likelihood"] for r in run_log] == [-100.0, -50.0, -120.0, -10.0]      # run log unchanged
    # negative control: with no degraded restart the maximum is chosen
    best2, created2, _ = _run_selection([(-100.0, False), (-90.0, False), (-120.0, False)], dual_data.X)
    assert best2 is created2[1]


def test_selection_among_healthy_restarts_is_the_maximum_and_the_first_maximum_wins_ties(dual_data):
    best, created, _ = _run_selection([(-90.0, False), (-90.0, False), (-95.0, False)], dual_data.X)
    assert best is created[0]
    best, created, _ = _run_selection([(-95.0, False), (-90.0, False), (-90.0, False)], dual_data.X)
    assert best is created[1]


def test_selection_when_every_restart_is_degraded_picks_the_highest_loglik_and_the_result_is_withheld(dual_data):
    best, created, _ = _run_selection([(-100.0, True), (-50.0, True), (-120.0, True)], dual_data.X)
    assert best is created[1] and best.fit_degraded_
    # end to end: the HSSMResult built from such a model is withheld
    r = HSSMResult(model=best, fit_state=best.fit_state_, fit_degraded=True, fit_cholesky_fallbacks=2)
    assert r.insights_withheld and r.publishable_outputs() is None


def test_selection_end_to_end_with_one_degraded_restart_gives_a_result_that_is_not_withheld(dual_data):
    real_init, calls = DualTimescaleHSSM._init_params, {"n": 0}

    def break_only_the_first(self, X_):
        real_init(self, X_)
        calls["n"] += 1
        if calls["n"] == 1:
            self.R_obs[1] = -5.0

    with patch.object(DualTimescaleHSSM, "_init_params", break_only_the_first):
        result = fit_hssm(matrix=dual_data.X, candidate_ks=(2,), timestamps=dual_data.timestamps, model_type="dual_timescale", latent_dim=2, **FAST)
    assert len(result.run_log) == 2                                       # both restarts ran; the first one was degraded, the chosen one is not
    assert not result.model.fit_degraded_ and not result.insights_withheld and result.publishable_outputs() is not None


def test_selection_still_raises_when_no_restart_is_usable(dual_data):
    def always_failing(self, X_, n_iter=50, tol=1e-4, verbose=False, timestamps=None):
        self.fit_state_, self.log_likelihood_, self.n_iter_ = FitState.NUMERICAL_FAILURE, None, 0
        self.fit_degraded_ = False
        return self

    with patch.object(DualTimescaleHSSM, "fit", always_failing):
        with pytest.raises(FittingConvergenceError):
            fit_dual_with_random_restarts(dual_data.X, n_regimes=2, n_features=3, n_init=2, bypass_init_gate=True)


# ------------------------------------------------------------------ item 1d: aliases
def test_aliases_return_the_stored_arrays_on_a_healthy_fit(clean_dual):
    r = clean_dual
    assert not r.insights_withheld
    assert r.p_t is r.p_t_estimate and r.regime_posterior is r.p_t_posterior and r.m_t is r.m_t_posterior_mean
    assert r.p_t is not None and r.regime_posterior is not None and r.m_t is not None


def test_aliases_return_none_when_withheld_and_raw_fields_stay_readable(forced_dual):
    r = forced_dual
    assert r.insights_withheld
    assert r.p_t is None
    assert r.regime_posterior is None
    assert r.m_t is None
    assert r.p_t_estimate is not None and r.p_t_posterior is not None and r.m_t_posterior_mean is not None and r.m_t_uncertainty is not None
    assert r.publishable_outputs() is None


def test_each_alias_is_gated_separately():
    base = dict(model=None, fit_state=FitState.CONVERGED, p_t_estimate=np.array([0, 1]), p_t_posterior=np.eye(2), m_t_posterior_mean=np.ones((2, 2)))
    ok = HSSMResult(**base)
    assert ok.p_t is ok.p_t_estimate and ok.regime_posterior is ok.p_t_posterior and ok.m_t is ok.m_t_posterior_mean
    for flag in (dict(fit_degraded=True, fit_cholesky_fallbacks=1), dict(states_pass_fallbacks=1), dict(fit_state=FitState.NUMERICAL_FAILURE)):
        r = HSSMResult(**{**base, **flag})
        assert (r.p_t, r.regime_posterior, r.m_t) == (None, None, None), flag


# ------------------------------------------------------------------ item 1c: firings after fit()
def test_a_healthy_fit_records_the_count_at_build_and_has_no_late_or_states_pass_firings(clean_dual):
    r = clean_dual
    assert r.states_pass_fallbacks == 0 and r.fallback_count_at_build == r.model.cholesky_fallback_count_ == 0
    assert r.late_fallbacks == 0 and r.withheld_reason is None


def test_estimate_states_without_firing_changes_nothing(clean_dual, dual_data):
    r = copy.deepcopy(clean_dual)
    r.model.estimate_states(dual_data.X)
    assert r.late_fallbacks == 0 and not r.insights_withheld and r.publishable_outputs() is not None


def test_late_firing_in_a_later_estimate_states_withholds_the_result_and_logs_once(clean_dual, dual_data):
    r = copy.deepcopy(clean_dual)
    assert not r.insights_withheld
    r.model.R_obs[1] = -5.0                                              # break one regime AFTER the result was built
    with _capture_warnings() as cap:
        r.model.estimate_states(dual_data.X)                             # the fallback fires inside this later call
        assert r.model.cholesky_fallback_count_ > 0 and r.fit_degraded is False and r.states_pass_fallbacks == 0
        assert r.late_fallbacks == r.model.cholesky_fallback_count_ - r.fallback_count_at_build > 0
        assert r.insights_withheld and "late_fallbacks" in r.withheld_reason
        assert cap.warnings == []                                        # reading the flag is not a read of the outputs: no log yet
        assert r.publishable_outputs() is None
        assert r.p_t is None and r.regime_posterior is None and r.m_t is None
        assert r.publishable_outputs() is None
    assert len(cap.warnings) == 1 and "withheld" in cap.warnings[0].getMessage() and "late_fallbacks" in cap.warnings[0].getMessage()
    assert "random_seed=42" in cap.warnings[0].getMessage()
    # raw fields stay readable
    assert r.p_t_estimate is not None and r.m_t_posterior_mean is not None


def test_firing_inside_the_states_pass_of_fit_hssm_is_withheld_at_build_time(dual_data):
    real_estimate = DualTimescaleHSSM.estimate_states

    def break_then_estimate(self, X_):
        self.R_obs[1] = -5.0                                             # breaks regime 1 AFTER fit(), before the outputs are computed
        return real_estimate(self, X_)

    with _capture_warnings() as cap, patch.object(DualTimescaleHSSM, "estimate_states", break_then_estimate):
        r = fit_hssm(matrix=dual_data.X, candidate_ks=(2,), timestamps=dual_data.timestamps, model_type="dual_timescale", latent_dim=2, **FAST)
    assert r.fit_degraded is False and r.fit_cholesky_fallbacks == 0                  # fit() itself was healthy
    assert r.states_pass_fallbacks > 0 and r.fallback_count_at_build == r.model.cholesky_fallback_count_ == r.states_pass_fallbacks
    assert r.late_fallbacks == 0
    assert r.insights_withheld and "states_pass" in r.withheld_reason and r.publishable_outputs() is None
    assert r.p_t is None and r.m_t is None
    assert len(cap.warnings) == 1                                         # logged once at build time, not again on read


def test_build_time_withheld_result_is_not_logged_again_on_read(forced_dual):
    r = copy.deepcopy(forced_dual)
    with _capture_warnings() as cap:
        assert r.publishable_outputs() is None and r.p_t is None and r.regime_posterior is None and r.m_t is None
    assert cap.warnings == []                                             # the build-time WARNING already happened (once-flag set)


def test_a_result_built_withheld_directly_logs_once_on_first_read_only():
    r = HSSMResult(model=None, fit_state=FitState.CONVERGED, fit_degraded=True, fit_cholesky_fallbacks=3)
    with _capture_warnings() as cap:
        assert r.publishable_outputs() is None and r.p_t is None and r.m_t is None
    assert len(cap.warnings) == 1
    # negative control: a healthy result never logs
    ok = HSSMResult(model=None, fit_state=FitState.CONVERGED)
    with _capture_warnings() as cap:
        ok.publishable_outputs(), ok.p_t, ok.m_t, ok.regime_posterior
    assert cap.warnings == []


def test_late_fallbacks_is_zero_without_a_recorded_count_or_without_a_counter():
    class _Counter:
        cholesky_fallback_count_ = 5

    assert HSSMResult(model=_Counter(), fit_state=FitState.CONVERGED).late_fallbacks == 0          # count at build not recorded
    assert HSSMResult(model=None, fit_state=FitState.CONVERGED, fallback_count_at_build=0).late_fallbacks == 0   # no model
    assert HSSMResult(model=object(), fit_state=FitState.CONVERGED, fallback_count_at_build=0).late_fallbacks == 0  # baseline-like: no counter
    assert HSSMResult(model=_Counter(), fit_state=FitState.CONVERGED, fallback_count_at_build=2).late_fallbacks == 3
    assert HSSMResult(model=_Counter(), fit_state=FitState.CONVERGED, fallback_count_at_build=5).late_fallbacks == 0


def test_rule_truth_table_with_the_two_new_conditions():
    for deg, nf, states, late_at_build, withheld in [
        (False, False, 0, 5, False), (True, False, 0, 5, True), (False, True, 0, 5, True), (False, False, 1, 5, True), (False, False, 0, 4, True),
    ]:
        class _M:
            cholesky_fallback_count_ = 5

        r = HSSMResult(model=_M(), fit_state=FitState.NUMERICAL_FAILURE if nf else FitState.CONVERGED, fit_degraded=deg,
                       fit_cholesky_fallbacks=1 if deg else 0, states_pass_fallbacks=states, fallback_count_at_build=late_at_build)
        assert r.insights_withheld is withheld and (r.publishable_outputs() is None) is withheld


# ------------------------------------------------------------------ the new error class
def test_insights_withheld_error_is_exported_and_carries_reason_and_result():
    import backbone.hssm as pkg
    assert pkg.InsightsWithheldError is InsightsWithheldError and pkg.fit_hssm_gated_result is fit_hssm_gated_result
    e = InsightsWithheldError("because", result="R")
    assert e.reason == "because" and e.result == "R" and "because" in str(e)
    assert issubclass(InsightsWithheldError, HSSMError) and issubclass(InsightsWithheldError, RuntimeError)
