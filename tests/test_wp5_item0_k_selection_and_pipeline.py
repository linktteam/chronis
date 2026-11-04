"""WP5 item 0: (a) K selection skips degraded candidates, (b) the multi-user attractor validation skips users whose insights are withheld.

Criteria: tests/PREDECLARED_CRITERIA_WP5.md section 1 (item 0). What each test would catch if it were wrong (mutation set:
integration_report/wp5/scripts/mutation_wp5.py, set `item0`):
  * a degraded K is chosen although a healthy one exists          -> test_select_k_by_bic_*, test_fit_hssm_dual_k_selection_*
  * all-degraded case raises / picks the wrong K                  -> the all-degraded tests
  * healthy behaviour changed (tie-break, minimum)                -> the healthy and tie tests
  * a withheld user is not skipped / is not counted / not logged  -> test_validate_planted_recovery_*
  * too broad a catch (other errors swallowed)                    -> test_other_errors_still_propagate
  * nothing validated but reported as a pass                      -> test_validate_planted_recovery_all_users_skipped
"""
import inspect
import logging
from unittest.mock import patch

import numpy as np
import pytest

from backbone.attractors import validation as validation_module
from backbone.attractors.config import CalibrationConfig
from backbone.hssm import fitting as fitting_module
from backbone.hssm.fitting import fit_hssm, select_k_by_bic
from backbone.hssm.gating import ColdStartError
from backbone.hssm.model import FittingConvergenceError, InsightsWithheldError
from data.synthetic_dual import generate_synthetic_dual_timescale_data


# ------------------------------------------------------------------ select_k_by_bic (stub models, no fitting)
class _StubModel:
    def __init__(self, bic, degraded=None):
        self._bic = bic
        self.converged_ = True
        self.log_likelihood_ = -10.0
        self.convergence_rate_ = 1.0
        if degraded is not None:                    # the baseline model has no such attribute at all
            self.fit_degraded_ = degraded

    def bic(self, n_observations=None):
        return self._bic


def _select(script, X):
    """script: {k: (bic, degraded or None) or 'fail'}; fit_with_random_restarts is replaced by a function returning the stubs."""
    def fake(X_, n_regimes, **kwargs):
        spec = script[n_regimes]
        if spec == "fail":
            raise FittingConvergenceError("no restart converged")
        return _StubModel(*spec), [{"k": n_regimes}]

    with patch.object(fitting_module, "fit_with_random_restarts", fake):
        model, report = select_k_by_bic(X, n_features=X.shape[1], k_candidates=tuple(script), bypass_init_gate=True)
    return model, report


@pytest.fixture(scope="module")
def X40():
    return np.random.RandomState(1).randn(40, 3)


def test_select_k_by_bic_prefers_a_healthy_candidate_over_a_degraded_one_with_lower_bic(X40):
    model, report = _select({2: (100.0, False), 3: (50.0, True), 4: (120.0, False)}, X40)
    assert model._bic == 100.0 and report["selected_k"] == 2
    assert report["bic_by_k"] == {2: 100.0, 3: 50.0, 4: 120.0}           # the report still lists every candidate
    # negative control: the same BICs without the degradation flag choose K=3
    model, report = _select({2: (100.0, False), 3: (50.0, False), 4: (120.0, False)}, X40)
    assert report["selected_k"] == 3


def test_select_k_by_bic_all_degraded_chooses_the_lowest_bic(X40):
    model, report = _select({2: (100.0, True), 3: (50.0, True), 4: (120.0, True)}, X40)
    assert report["selected_k"] == 3 and model.fit_degraded_ is True


def test_select_k_by_bic_models_without_the_flag_behave_as_before_including_first_minimum_on_ties(X40):
    model, report = _select({2: (100.0, None), 3: (50.0, None), 4: (50.0, None)}, X40)
    assert report["selected_k"] == 3
    assert not hasattr(model, "fit_degraded_")


def test_select_k_by_bic_failed_candidates_are_still_skipped_and_all_failing_still_raises(X40):
    model, report = _select({2: "fail", 3: (50.0, True), 4: (60.0, False)}, X40)
    assert report["selected_k"] == 4
    with pytest.raises(FittingConvergenceError):
        _select({2: "fail", 3: "fail"}, X40)


# ------------------------------------------------------------------ fit_hssm dual K loop (real fits, log-likelihood and flag overridden after the fit)
@pytest.fixture(scope="module")
def dual_data():
    return generate_synthetic_dual_timescale_data(n_regimes=2, n_features=3, latent_dim=2, n_segments=4, mean_dwell=15.0, seed=505)


def _fit_two_ks(dual_data, k2, k3):
    """Real dual fits for K=2 and K=3; after each fit, `log_likelihood_` and `fit_degraded_` are overridden: k2 / k3 = (loglik, degraded)."""
    real = fitting_module.fit_dual_with_random_restarts
    overrides = {2: k2, 3: k3}

    def wrapper(X, n_regimes, **kwargs):
        model, run_log = real(X, n_regimes=n_regimes, **kwargs)
        model.log_likelihood_, model.fit_degraded_ = overrides[n_regimes]
        return model, run_log

    with patch.object(fitting_module, "fit_dual_with_random_restarts", wrapper):
        return fit_hssm(matrix=dual_data.X, candidate_ks=(2, 3), n_initializations=2, max_iter=4, random_seed=42, allow_fast_test_fit=True,
                        timestamps=dual_data.timestamps, model_type="dual_timescale", latent_dim=2)


def test_fit_hssm_dual_k_selection_skips_a_degraded_k_with_lower_bic(fast_dual_counts_as_converged, dual_data):
    r = _fit_two_ks(dual_data, k2=(1e6, True), k3=(-300.0, False))      # K=2 would win by BIC, but it is degraded
    assert r.K == 3 and r.model.K == 3
    assert r.fit_degraded is False and not r.insights_withheld
    assert r.bic_by_k[2] < r.bic_by_k[3]                                  # the BIC of the skipped K is still reported
    # negative control: the same numbers with K=2 healthy choose K=2
    r2 = _fit_two_ks(dual_data, k2=(1e6, False), k3=(-300.0, False))
    assert r2.K == 2 and not r2.insights_withheld


def test_fit_hssm_dual_k_selection_all_degraded_chooses_the_lowest_bic_and_is_withheld(dual_data):
    r = _fit_two_ks(dual_data, k2=(1e6, True), k3=(-300.0, True))
    assert r.K == 2 and r.fit_degraded is True and r.insights_withheld and r.publishable_outputs() is None


def test_fit_hssm_dual_k_selection_healthy_candidates_choose_the_lowest_bic(dual_data):
    r = _fit_two_ks(dual_data, k2=(-500.0, False), k3=(-300.0, False))
    assert r.K == 3
    r = _fit_two_ks(dual_data, k2=(-300.0, False), k3=(-500.0, False))
    assert r.K == 2


# ------------------------------------------------------------------ validate_planted_recovery
class _Fixture:
    """Replaces the expensive parts of validate_planted_recovery by stubs, and lets fit_user_pipeline fail for chosen users."""

    def __init__(self, withheld_users=(), other_error_users=()):
        self.withheld_users, self.other_error_users = set(withheld_users), set(other_error_users)
        self.calls = []

    def fit_user_pipeline(self, n_sessions, n_regimes, target_dims, seed):
        u = (seed - 1) // 7
        self.calls.append(u)
        if u in self.withheld_users:
            raise InsightsWithheldError(f"fit_degraded: reason for user {u}", result=None)
        if u in self.other_error_users:
            raise ColdStartError("not enough sessions")
        return object(), [], None

    def patches(self):
        return [
            patch.object(validation_module, "fit_user_pipeline", self.fit_user_pipeline),
            patch.object(validation_module, "grid_search_NT", lambda *a, **k: {"selected_N_revisit": 1, "selected_N_dwell": 1, "selected_T": 1}),
            patch.object(validation_module, "sample_from_fitted_hssm", lambda model, n, seed_attractor, attractor_regime, seed: {"m_t": seed_attractor, "regime_labels": None}),
            patch.object(validation_module, "compute_attractor_stats", lambda m_t, labels, regime: m_t),
            patch.object(validation_module, "is_attractor", lambda stats, nd, T: bool(stats)),
        ]

    def run(self, n_users=3, caplog=None):
        from contextlib import ExitStack
        config = CalibrationConfig(n_timesteps=10, n_trials=2, target_precision=0.8)
        with ExitStack() as stack:
            for p in self.patches():
                stack.enter_context(p)
            return validation_module.validate_planted_recovery(n_users=n_users, config=config, n_test_trials=4)


def _warnings(caplog):
    return [r for r in caplog.records if r.name == validation_module.__name__ and r.levelno == logging.WARNING]


def test_validate_planted_recovery_with_no_withheld_user_reports_zero_skipped(caplog):
    with caplog.at_level(logging.DEBUG, logger=validation_module.__name__):
        res = _Fixture().run(3)
    assert res["n_skipped_users"] == 0 and res["skipped_users"] == [] and set(res["recovery"]) == {"user_0", "user_1", "user_2"}
    assert res["all_users_pass"] is True and _warnings(caplog) == []


def test_validate_planted_recovery_skips_a_withheld_user_logs_one_warning_and_counts_it(caplog):
    fx = _Fixture(withheld_users={1})
    with caplog.at_level(logging.DEBUG, logger=validation_module.__name__):
        res = fx.run(3)
    assert fx.calls == [0, 1, 2]                                           # the loop continued after the withheld user
    assert res["n_skipped_users"] == 1 and res["skipped_users"] == ["user_1"]
    assert set(res["recovery"]) == {"user_0", "user_2"} and set(res["calibrations"]) == {"user_0", "user_2"}
    assert res["all_users_pass"] is True                                   # the evaluated users decide
    w = _warnings(caplog)
    assert len(w) == 1 and "user_1" in w[0].getMessage() and "reason for user 1" in w[0].getMessage()


def test_validate_planted_recovery_all_users_skipped_is_not_reported_as_a_pass(caplog):
    with caplog.at_level(logging.DEBUG, logger=validation_module.__name__):
        res = _Fixture(withheld_users={0, 1, 2}).run(3)
    assert res["n_skipped_users"] == 3 and res["recovery"] == {} and res["calibrations"] == {}
    assert res["all_users_pass"] is False                                  # all([]) is True; nothing was validated
    assert len(_warnings(caplog)) == 3


def test_other_errors_still_propagate():
    with pytest.raises(ColdStartError):
        _Fixture(other_error_users={1}).run(3)


def test_fit_user_pipeline_signature_is_unchanged():
    assert list(inspect.signature(validation_module.fit_user_pipeline).parameters) == ["n_sessions", "n_features_raw", "n_regimes", "target_dims", "seed"]
