"""WP10B B1 to B5: the dual-study library (`integration_report/wp10b/scripts/dual_lib.py`): the train/test matrix, the frozen-parameter one-step scores of the three predictors, the reason categories, the EM-creep
extrapolation and the record of a tiny real fit. Loaded by path under a unique module name."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import multivariate_normal

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "integration_report" / "wp10b" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SC))
sys.path.insert(0, str(ROOT / "integration_report" / "wp9" / "scripts"))
_spec = importlib.util.spec_from_file_location("wp10b_dual_lib_under_test", SC / "dual_lib.py")
L = importlib.util.module_from_spec(_spec)
sys.modules["wp10b_dual_lib_under_test"] = L
_spec.loader.exec_module(L)


# ---- data ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_the_train_block_equals_the_pipeline_window_and_the_test_block_uses_the_train_scaler():
    import wp9_fit_lib as F
    d = L.user_data(11901, "easy")
    st = F.stream_for(dict(user=11901, stratum="easy"))
    w = st.window(0, 120)
    assert np.array_equal(d["X"][:120], w["matrix"], equal_nan=True)
    assert np.array_equal(d["scaler_mean"], w["scaler_mean"]) and np.array_equal(d["scaler_std"], w["scaler_std"])
    raw_test = st.X_obs[120:150]
    z = (raw_test - w["scaler_mean"]) / w["scaler_std"]
    ok = ~np.isnan(raw_test).any(axis=1)
    assert np.allclose(d["X"][120:150][ok], z[ok])
    assert d["X"].shape == (150, 7) and len(d["labels"]) == 150


def test_an_all_missing_day_is_a_nan_row_and_is_flagged():
    d = L.user_data(11902, "hard")
    assert np.array_equal(np.isnan(d["X"]).all(axis=1), d["day_missing"])


# ---- frozen-parameter scores ----------------------------------------------------------------------------------------------------------------------------------------------
def tiny_series(T=70, F=3, seed=0):
    rng = np.random.default_rng(seed)
    lab = np.repeat([0, 1, 0, 1, 0], T // 5)
    return np.array([[(-1.5 if l == 0 else 1.5) + 0.0 * j for j in range(F)] for l in lab]) + rng.standard_normal((len(lab), F))


def test_baseline_increments_sum_to_the_models_own_log_likelihood():
    from backbone.hssm.model import GaussianHSMM
    X = tiny_series()
    m = GaussianHSMM(n_regimes=2, n_features=3, max_duration=12, seed=1)
    m.fit(X, n_iter=8, verbose=False)
    m._forward_backward(X)
    lp = L.baseline_increments(m, X)
    assert lp.shape == (len(X),) and np.all(np.isfinite(lp))
    assert math.isclose(float(lp.sum()), float(m.log_likelihood_), rel_tol=1e-9, abs_tol=1e-8)


def test_baseline_increments_with_a_missing_day_are_finite_and_still_sum_to_the_log_likelihood():
    from backbone.hssm.model import GaussianHSMM
    X = tiny_series(seed=2)
    X[30] = np.nan
    m = GaussianHSMM(n_regimes=2, n_features=3, max_duration=12, seed=3)
    m.fit(X, n_iter=8, verbose=False)
    m._forward_backward(X)
    lp = L.baseline_increments(m, X)
    assert np.all(np.isfinite(lp)) and math.isclose(float(lp.sum()), float(m.log_likelihood_), rel_tol=1e-9, abs_tol=1e-8)


def test_dual_increments_equal_the_n18_harness_and_sum_to_the_forward_log_likelihood():
    from backbone.hssm.dual_model import DualTimescaleHSSM
    from tests.utils.heldout_prediction import predictive_increments
    X = tiny_series(T=50, F=3, seed=4)
    m = DualTimescaleHSSM(n_regimes=2, n_features=3, latent_dim=2, max_duration=8, seed=5)
    m.fit(X, n_iter=3)
    lp = L.dual_increments(m, X)
    ref, ll = predictive_increments(m, X)
    assert np.array_equal(lp, ref) and math.isclose(float(lp.sum()), ll, rel_tol=1e-9, abs_tol=1e-8)


def test_gaussian_increments_use_train_moments_only_and_leave_missing_rows_nan():
    rng = np.random.default_rng(6)
    Xtr = rng.standard_normal((60, 3)); Xtr[10] = np.nan
    Xte = rng.standard_normal((8, 3)); Xte[2] = np.nan
    lp = L.gaussian_increments(Xtr, Xte)
    tr = Xtr[~np.isnan(Xtr).all(axis=1)]
    ref = multivariate_normal(mean=tr.mean(axis=0), cov=np.cov(tr, rowvar=False, ddof=1)).logpdf(Xte[[0, 1, 3, 4, 5, 6, 7]])
    assert np.allclose(lp[[0, 1, 3, 4, 5, 6, 7]], ref) and math.isnan(lp[2])


def test_test_mean_averages_the_observed_test_days_only():
    lp = np.concatenate([np.zeros(5), [1.0, 2.0, 100.0, 3.0]])
    dm = np.array([False] * 5 + [False, False, True, False])
    m, n, fin = L.test_mean(lp, 5, dm)
    assert (m, n, fin) == (2.0, 3, True)
    lp[6] = np.inf
    assert L.test_mean(lp, 5, dm)[2] is False


# ---- diagnosis --------------------------------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("reasons,cats", [
    (["fit_state_not_converged: fit_state is MAX_ITER_REACHED, the EM fit did not reach convergence"], ["iteration_cap"]),
    (["fit_state_not_converged: fit_state is REJECTED_UPDATE, the EM fit did not reach convergence"], ["rejected_update"]),
    (["fit_state_not_converged: fit_state is FAILED, x"], ["other_non_convergence"]),
    (["collapsed_regime: a regime holds 0.5 effective days (minimum 3)", "degenerate_duration: regime(s) [1] have duration parameters at a wall"], ["collapsed_regime", "degenerate_duration"]),
    (["fit_degraded: the -1e6 Cholesky fallback fired 2 time(s) during fit", "states_pass: x", "late_fallbacks: y", "fit_state is NUMERICAL_FAILURE"], ["numerical"]),
    (["something else"], ["other"]),
    ([], []),
])
def test_reason_categories(reasons, cats):
    assert L.reason_categories(reasons) == cats


def test_creep_diagnostics_recover_the_geometric_rate_and_the_extrapolated_count():
    r, d0 = 0.99, 0.5
    inc = d0 * r ** np.arange(120)
    hist = np.concatenate([[-1000.0], -1000.0 + np.cumsum(inc)])
    c = L.creep_diagnostics(list(hist))
    assert math.isclose(c["rate"], r, rel_tol=1e-9) and c["n_iter"] == 121
    last = inc[-1]
    assert c["extra_iterations"] == int(math.ceil(math.log(1e-4 / last) / math.log(r)))
    assert math.isclose(c["gain_last10"], float(inc[-10:].sum()), rel_tol=1e-9)


def test_creep_diagnostics_when_already_converged_or_not_contracting():
    inc = 1e-5 * 0.9 ** np.arange(80)
    assert L.creep_diagnostics(list(np.cumsum(np.concatenate([[0.0], inc]))))["extra_iterations"] == 0
    growing = np.cumsum(np.concatenate([[0.0], 0.01 * 1.01 ** np.arange(80)]))
    assert L.creep_diagnostics(list(growing))["extra_iterations"] is None
    assert L.creep_diagnostics([1.0, 2.0, 3.0])["rate"] is None


def test_the_record_of_a_tiny_real_fit_is_consistent():
    X = tiny_series(T=60, F=4, seed=7)
    res = L.fit_dual(X, 7, "S0", dict(max_iter=3, max_duration=10))
    rec = L.dual_record(res, res.model)
    assert rec["publishable"] == (res.withheld_reason is None) and rec["publishable"] == (not rec["reasons"])
    assert rec["n_iter"] == len(rec["ll_history"]) and rec["Dmax"] == 10 and rec["fit_state"] == str(res.fit_state.value)
    assert all(c in ("iteration_cap", "rejected_update", "other_non_convergence", "collapsed_regime", "degenerate_duration", "numerical", "other") for c in rec["categories"])
    assert len(rec["dur_mu"]) == 3 and len(rec["pmf_rising"]) == 3


def test_variants_override_only_what_they_name():
    assert L.VARIANTS["D60"] == dict(max_duration=60) and L.VARIANTS["I600"] == dict(max_iter=600) and L.VARIANTS["S0"] == {}
    assert L.S0["max_iter"] == 200 and L.S0["max_duration"] == 45 and L.S0["n_initializations"] == 1 and L.S0["latent_dim"] == 2

def test_publishable_follows_the_withhold_reason_not_the_em_state():
    """A converged fit can still be withheld (a collapsed regime): `publishable` must read the reasons, not the fit state."""
    import copy
    from backbone.hssm.fitting import FitState
    X = tiny_series(T=60, F=4, seed=7)
    res = L.fit_dual(X, 7, "S0", dict(max_iter=3, max_duration=10))
    r2 = copy.copy(res)
    r2.fit_state = FitState.CONVERGED
    r2.fit_collapsed = True
    r2.collapse_report = dict(min_effective_days=0.5, min_effective_days_threshold=3.0)
    rec = L.dual_record(r2, res.model)
    assert rec["fit_state"] == "CONVERGED" and rec["publishable"] is False and "collapsed_regime" in rec["categories"]
