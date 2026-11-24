"""WP5 item 4: the optional `shared_prior` argument of `fit_hssm` and `fit_dual_with_random_restarts`.

Only integrated because criteria 1 to 6 of tests/PREDECLARED_CRITERIA_WP5.md passed (tests/WP5_STUDY_RESULTS.md). The default (`shared_prior=None`) must behave exactly as at 16c3431
(bit-identity script, wp5_evidence/ab_item4_*); these tests check the new branch, its validation (fail closed) and that the default path does not load the prototype.
Mutation set `integ` (integration_report/wp5/scripts/mutation_wp5.py).
"""
import copy
import inspect
import json
import pathlib
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

from backbone.hssm import fitting as fitting_module
from backbone.hssm.config import SHARED_PRIOR_DEFAULT_N0, ColdStartConfig
from backbone.hssm.dual_model import DualTimescaleHSSM
from backbone.hssm.fitting import HSSMResult, fit_dual_with_random_restarts, fit_hssm
from backbone.hssm.gating import ColdStartError
from backbone.hssm.model import FittingConvergenceError, FitState
from backbone.hssm.shrinkage import SharedPrior, ShrinkageDualHSSM
from data.synthetic_population import generate_population_user

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRIOR = SharedPrior.from_dict(json.loads((ROOT / "tests" / "oracle" / "wp5_shared_prior.json").read_text()))
LOW_GATE = ColdStartConfig(min_present_sessions=10)
X = generate_population_user(21, 3, 0).X_train                      # about 45 steps, F = 4
K, F, L, DMAX = 2, 4, 2, 25


def prior_fit(**kw):
    args = dict(candidate_ks=(2,), max_iter=6, config=LOW_GATE, model_type="dual_timescale", max_duration=DMAX, random_seed=7, latent_dim=2, shared_prior=PRIOR)
    args.update(kw)
    return fit_hssm(X, **args)


def default_fit(**kw):
    args = dict(candidate_ks=(2,), n_initializations=2, max_iter=4, allow_fast_test_fit=True, config=LOW_GATE, model_type="dual_timescale", max_duration=DMAX, random_seed=7)
    args.update(kw)
    return fit_hssm(X, **args)


def params_equal(a, b):
    from backbone.hssm.shrinkage import PARAM_NAMES
    return all(np.array_equal(getattr(a, n), getattr(b, n)) for n in PARAM_NAMES)


# ------------------------------------------------------------------ signatures and defaults
def test_new_arguments_are_last_optional_and_default_to_none():
    for fn, tail in ((fit_hssm, ["shared_prior", "shared_prior_n0", "max_duration_retry"]), (fit_dual_with_random_restarts, ["shared_prior", "shared_prior_n0"])):
        params = list(inspect.signature(fn).parameters.values())
        # WP7-R item 5 added `max_duration_retry` after the WP5 arguments of fit_hssm (new arguments stay last, optional, default None): the WP5 pair keeps its place
        assert [p.name for p in params[-len(tail):]] == tail and all(p.default is None for p in params[-len(tail):]), fn.__name__
    assert [p for p in inspect.signature(fit_hssm).parameters][:6] == ["matrix", "candidate_ks", "n_initializations", "max_iter", "random_seed", "allow_fast_test_fit"]
    assert SHARED_PRIOR_DEFAULT_N0 == 5.0                                                              # the value frozen by the development study (tests/N0_FROZEN_WP5.md)
    assert "Frozen n0 = 5" in (ROOT / "tests" / "N0_FROZEN_WP5.md").read_text()


def test_the_default_result_records_no_prior_and_its_metadata_is_unchanged():
    r = default_fit()
    assert r.used_shared_prior is False and r.shared_prior_n0 is None and "shared_prior" not in r.convergence_metadata
    assert type(r.model) is DualTimescaleHSSM and r.model_class == "DualTimescaleHSSM"
    # WP6 item 3 part 2 added the keys convergence_criterion, window_gain_at_cap and duration_box (additive, docs/backbone/HSSM_RESULT_SCHEMA.md); every WP5 key is still there
    assert set(r.convergence_metadata) == {"bic_by_k", "loglik_by_k", "convergence_rate_by_k", "selected_loglik", "fit_state", "converged", "n_iter", "run_log",
                                           "convergence_criterion", "window_gain_at_cap", "duration_box"}
    model, log = fit_dual_with_random_restarts(X, n_regimes=2, n_features=F, latent_dim=L, n_init=2, n_iter=3, max_duration=DMAX, bypass_init_gate=True)
    assert type(model) is DualTimescaleHSSM and len(log) == 2 and "shared_prior" not in log[0]


def test_baseline_results_carry_the_default_fields():
    r = fit_hssm(X, candidate_ks=(2,), n_initializations=2, max_iter=15, allow_fast_test_fit=True, config=LOW_GATE)
    assert r.used_shared_prior is False and r.shared_prior_n0 is None


# ------------------------------------------------------------------ the prior branch
def test_prior_fit_is_the_shrinkage_engine_started_at_the_prior_with_the_frozen_strength():
    r = prior_fit()
    assert r.used_shared_prior is True and r.shared_prior_n0 == 5.0 and r.K == 2 and r.selected_k == 2
    assert type(r.model) is ShrinkageDualHSSM and r.model_class == "ShrinkageDualHSSM" and r.model.used_shared_prior and r.model.n0 == 5.0
    direct = ShrinkageDualHSSM(n_regimes=K, n_features=F, latent_dim=L, max_duration=DMAX, seed=2 * 1000 + 7, shared_prior=PRIOR, n0=5.0).fit(X, n_iter=6)
    assert params_equal(r.model, direct) and r.model.log_likelihood_ == direct.log_likelihood_
    meta = r.convergence_metadata["shared_prior"]
    assert meta["n0"] == 5.0 and meta["n_users"] == PRIOR.n_users == 19 and meta["k_fixed_by_prior"] is True and meta["n_fits_run"] == 1 and meta["prior_uncertainty_in_m_t_std"] is False
    assert len(r.run_log) == 1 and r.run_log[0]["shared_prior"] is True and r.run_log[0]["shared_prior_n0"] == 5.0
    assert r.p_t_posterior.shape == (len(X), K) and r.m_t_posterior_mean.shape == (len(X), L) and r.m_t_uncertainty.shape == (len(X), L)
    assert r.fit_state in (FitState.CONVERGED, FitState.MAX_ITER_REACHED, FitState.REJECTED_UPDATE)


def test_the_strength_argument_is_used_and_recorded():
    weak, strong = prior_fit(shared_prior_n0=5.0), prior_fit(shared_prior_n0=400.0)
    assert strong.shared_prior_n0 == 400.0 and strong.model.n0 == 400.0 and strong.run_log[0]["shared_prior_n0"] == 400.0
    assert not params_equal(weak.model, strong.model)
    d_weak = float(np.max(np.abs(weak.model.d_obs - PRIOR.d_obs)))
    d_strong = float(np.max(np.abs(strong.model.d_obs - PRIOR.d_obs)))
    assert d_strong < d_weak                                                                          # more strength, closer to the shared parameters
    zero = prior_fit(shared_prior_n0=0.0)                                                              # warm start only
    assert zero.used_shared_prior is True and zero.shared_prior_n0 == 0.0


def test_with_a_prior_k_is_the_priors_and_other_candidates_are_not_fitted():
    r = prior_fit(candidate_ks=(2, 3, 4))
    assert r.K == 2 and set(r.bic_by_k) == {2} and len(r.run_log) == 1


def test_the_prior_branch_is_a_single_deterministic_fit_without_the_restart_minimum():
    model, log = fit_dual_with_random_restarts(X, n_regimes=K, n_features=F, latent_dim=L, n_init=1, n_iter=3, max_duration=DMAX, shared_prior=PRIOR)   # no bypass_init_gate needed
    assert len(log) == 1 and isinstance(model, ShrinkageDualHSSM)
    again, _ = fit_dual_with_random_restarts(X, n_regimes=K, n_features=F, latent_dim=L, n_init=7, n_iter=3, max_duration=DMAX, shared_prior=PRIOR)
    assert params_equal(model, again)                                                                  # n_init does not matter


# ------------------------------------------------------------------ fail closed
def test_invalid_requests_raise_and_return_nothing():
    with pytest.raises(ValueError, match="without a shared_prior"):
        default_fit(shared_prior_n0=5.0)
    with pytest.raises(ValueError, match="without a shared_prior"):
        fit_dual_with_random_restarts(X, n_regimes=K, n_features=F, latent_dim=L, n_init=1, n_iter=2, bypass_init_gate=True, shared_prior_n0=5.0)
    with pytest.raises(ValueError, match="without a shared_prior"):
        fit_hssm(X, candidate_ks=(2,), n_initializations=2, max_iter=15, allow_fast_test_fit=True, config=LOW_GATE, shared_prior_n0=5.0)        # baseline model: not silently ignored
    with pytest.raises(ValueError, match="dual-timescale"):
        prior_fit(model_type="baseline")
    with pytest.raises(ValueError, match="not among candidate_ks"):
        prior_fit(candidate_ks=(3, 4))
    with pytest.raises(ValueError, match="the fit asks for"):
        fit_hssm(X[:, :3], candidate_ks=(2,), max_iter=3, config=LOW_GATE, model_type="dual_timescale", max_duration=DMAX, shared_prior=PRIOR)     # F = 3 against a prior with F = 4
    with pytest.raises(ValueError, match="the fit asks for"):
        prior_fit(latent_dim=3)
    with pytest.raises(ValueError):
        prior_fit(shared_prior_n0=-1.0)
    leaky = copy.deepcopy(PRIOR)
    leaky.X_train = X                                                                                  # a prior that carries data is refused
    with pytest.raises(ValueError, match="refused"):
        prior_fit(shared_prior=leaky)


def test_an_unfinished_prior_fit_raises_instead_of_falling_back():
    def failing_fit(self, X_, n_iter=50, tol=1e-4, verbose=False, timestamps=None):
        self.fit_state_ = FitState.NUMERICAL_FAILURE
        self.log_likelihood_ = None
        return self
    with patch.object(ShrinkageDualHSSM, "fit", failing_fit):
        with pytest.raises(FittingConvergenceError, match="did not finish"):
            prior_fit()


def test_the_cold_start_gate_is_not_changed_by_a_prior():
    with pytest.raises(ColdStartError):
        fit_hssm(X[:20], candidate_ks=(2,), max_iter=3, model_type="dual_timescale", max_duration=DMAX, shared_prior=PRIOR)            # default gate: 30 present sessions
    with pytest.raises(ColdStartError):
        fit_hssm(X, candidate_ks=(2,), max_iter=3, model_type="dual_timescale", max_duration=DMAX, shared_prior=PRIOR, n_present_sessions=12)


def test_the_hold_back_applies_to_prior_fits():
    real_fit = ShrinkageDualHSSM.fit

    def as_converged_fit(self, *a, **k):
        # WP6 item 3 (P0-07): these prior fits are deliberately short and end at the iteration limit or with a guard rejection, which now withholds
        # them. This test is about the numerical-health rule, so the short fit is treated as converged here (the P0-07 rule has its own tests).
        out = real_fit(self, *a, **k)
        self.fit_state_ = FitState.CONVERGED
        return out

    def degraded_fit(self, *a, **k):
        out = as_converged_fit(self, *a, **k)
        self.fit_degraded_, self.fit_cholesky_fallbacks_ = True, 2
        return out
    with patch.object(ShrinkageDualHSSM, "fit", degraded_fit):
        r = prior_fit()
    assert r.fit_degraded and r.insights_withheld and r.publishable_outputs() is None and r.used_shared_prior
    with patch.object(ShrinkageDualHSSM, "fit", as_converged_fit):
        healthy = prior_fit()
    assert not healthy.insights_withheld and healthy.publishable_outputs() is not None


# ------------------------------------------------------------------ the default path does not load the prototype; the schema lists the fields
def _modules_after(code):
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr[-600:]
    return out.stdout.split()


def test_a_default_fit_never_imports_the_prototype_and_a_prior_fit_does():
    base = ("import sys, numpy as np; from backbone.hssm.fitting import fit_hssm; from backbone.hssm.config import ColdStartConfig; "
            "X = np.random.default_rng(1).normal(size=(40, 4)); ")
    default = base + ("fit_hssm(X, candidate_ks=(2,), n_initializations=1, max_iter=2, allow_fast_test_fit=True, config=ColdStartConfig(10), model_type='dual_timescale', max_duration=25); "
                      "print('backbone.hssm.shrinkage' in sys.modules)")
    assert _modules_after(default) == ["False"]
    withprior = base + ("import json; from backbone.hssm.shrinkage import SharedPrior; "
                        "p = SharedPrior.from_dict(json.load(open('tests/oracle/wp5_shared_prior.json'))); print('backbone.hssm.shrinkage' in sys.modules)")
    assert _modules_after(withprior) == ["True"]


def test_the_schema_document_lists_the_new_fields_and_they_are_dataclass_fields():
    doc = (ROOT / "docs" / "backbone" / "HSSM_RESULT_SCHEMA.md").read_text()
    fields = {f.name for f in HSSMResult.__dataclass_fields__.values()}
    for name in ("used_shared_prior", "shared_prior_n0"):
        assert name in fields and f"| `{name}` |" in doc, name                                                  # the table row of section 2
    assert "shared_prior" in doc and "SHARED_PRIOR_DEFAULT_N0" in doc
