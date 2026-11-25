"""WP8 item 0e (metadata honesty). Three claims in the model/result metadata were constants that did not describe what happened:
  1. `DualTimescaleHSSM.model_capability["is_production_grade_full_hssm"]` was True for every dual model, fitted or not;
  2. the dual path of `fit_hssm` stored `optimizer_success=True` regardless of the duration optimiser (the baseline path already used `not duration_fallback_used_`);
  3. `duration_model_identity` said "explicit_truncated_gaussian" in both models, while the code fits a truncated, renormalised discrete log-normal.
Each test below has a negative control (the opposite case) and is killed by at least one mutant in integration_report/wp8/scripts/mutation_item0e.py.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import lognorm

from backbone.hssm.dual_model import DualTimescaleHSSM
from backbone.hssm.model import FitState, GaussianHSMM
from tests.test_wp7r_item5_dual_honesty import X_LONG, _fit, _series


def _dual(tol=None):
    m = DualTimescaleHSSM(n_regimes=2, n_features=3, latent_dim=2, max_duration=6, seed=3)
    if tol is not None:
        m.fit(X_LONG, n_iter=40, tol=tol)
    return m


@pytest.fixture(scope="module")
def converged():
    m = _dual(tol=1.0)                                  # loose tolerance: a genuinely CONVERGED, non-degraded fit (checked in the first test)
    assert m.fit_state_ == FitState.CONVERGED and not m.fit_degraded_ and not any(m.duration_at_bound_.values())
    return m


@pytest.fixture(scope="module")
def not_converged():
    m = _dual(tol=1e-4)
    assert m.fit_state_ == FitState.MAX_ITER_REACHED
    return m


def _cap(m):
    return m.model_capability["is_production_grade_full_hssm"]


# ------------------------------------------------------------------------------------------------------------------- 1. capability flag
def test_unfitted_dual_model_is_not_production_grade():
    m = _dual()
    assert _cap(m) is False and m.is_certified_fit_ is False


def test_an_unfitted_model_is_not_certified_even_if_its_state_says_converged():
    m = _dual()
    m.fit_state_ = FitState.CONVERGED                      # e.g. a state restored without the fitted parameters: the flag must not rest on the state label alone
    assert m._is_fitted is False and _cap(m) is False


def test_converged_clean_fit_is_production_grade_and_the_flag_is_a_real_bool(converged):
    assert _cap(converged) is True and converged.is_certified_fit_ is True


def test_fit_that_hit_the_iteration_limit_is_not_production_grade(not_converged):
    assert _cap(not_converged) is False


@pytest.mark.parametrize("state", [s for s in FitState if s != FitState.CONVERGED], ids=lambda s: s.value)
def test_every_fit_state_other_than_converged_is_not_production_grade(converged, state):
    m = _dual(tol=1.0)
    m.fit_state_ = state
    assert _cap(m) is False
    m.fit_state_ = FitState.CONVERGED
    assert _cap(m) is True                                 # negative control: restoring the state restores the flag (nothing else was touched)


def test_degraded_fit_is_not_production_grade(converged):
    m = _dual(tol=1.0)
    m.fit_degraded_ = True
    assert _cap(m) is False
    m.fit_degraded_ = False
    assert _cap(m) is True


@pytest.mark.parametrize("wall", ["mu_low", "mu_high", "sigma_high"])
def test_duration_at_a_wall_of_the_box_is_not_production_grade_for_either_regime(wall):
    for k in (0, 1):
        m = _dual(tol=1.0)
        mu_lo, mu_hi, sig_hi = m._duration_box()
        keep_mu, keep_sg = m.dur_mu.copy(), m.dur_sigma.copy()
        if wall == "mu_low":
            m.dur_mu[k] = mu_lo
        elif wall == "mu_high":
            m.dur_mu[k] = mu_hi
        else:
            m.dur_sigma[k] = sig_hi
        assert m.duration_at_bound_[k] is True and _cap(m) is False, (wall, k)
        m.dur_mu, m.dur_sigma = keep_mu, keep_sg             # negative control: the untouched parameters give the flag back
        assert m.duration_at_bound_ == {0: False, 1: False} and _cap(m) is True, (wall, k)


def test_flag_is_read_when_asked_not_frozen_at_construction(converged):
    m = _dual()
    before = _cap(m)
    m.fit(X_LONG, n_iter=40, tol=1.0)
    assert before is False and _cap(m) is True


# ------------------------------------------------------------------------------------------------------------------- 2. optimizer_success
def test_dual_optimizer_success_is_false_when_the_duration_optimiser_fell_back_and_true_otherwise():
    ok = _fit(_series([20] * 5, seed=1))                    # public path, seed with no duration fallback
    bad = _fit(_series([20] * 5, seed=2))                   # same design, a seed where the duration optimiser fell back
    assert ok.model.duration_fallback_used_ is False and ok.optimizer_success is True
    assert bad.model.duration_fallback_used_ is True and bad.optimizer_success is False
    assert ok.duration_optimizer_success is True and bad.duration_optimizer_success is False


# ------------------------------------------------------------------------------------------------------------------- 3. duration identity
def _make(cls):
    kw = dict(n_regimes=2, n_features=3, max_duration=6, seed=0)
    return cls(**({**kw, "latent_dim": 2} if cls is DualTimescaleHSSM else kw))


@pytest.mark.parametrize("cls", [GaussianHSMM, DualTimescaleHSSM])
def test_duration_identity_names_the_law_that_is_fitted(cls):
    assert _make(cls).duration_model_identity == "explicit_truncated_lognormal"


@pytest.mark.parametrize("cls", [GaussianHSMM, DualTimescaleHSSM])
def test_the_pmf_is_a_truncated_renormalised_lognormal_not_a_gaussian(cls):
    m = _make(cls)
    m.dur_mu, m.dur_sigma = np.array([0.8, 1.4]), np.array([0.5, 0.7])
    d = np.arange(1, 7)
    for k in range(2):
        want = lognorm.pdf(d, s=m.dur_sigma[k], scale=np.exp(m.dur_mu[k]))
        want = want / want.sum()
        assert np.allclose(np.exp(m._duration_logpmf()[k]), want, atol=1e-12)
        gauss = np.exp(-0.5 * ((d - np.exp(m.dur_mu[k])) / m.dur_sigma[k]) ** 2)
        gauss = gauss / gauss.sum()
        assert not np.allclose(want, gauss, atol=1e-3)       # negative control: a Gaussian pmf on the same parameters would be a different law


def test_schema_doc_row_names_the_lognormal_and_the_old_name_is_gone():
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[1] / "docs" / "backbone" / "HSSM_RESULT_SCHEMA.md").read_text()
    assert "explicit_truncated_lognormal" in doc and "explicit_truncated_gaussian" not in doc
