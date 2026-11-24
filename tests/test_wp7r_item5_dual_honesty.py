"""WP7-R item 5: dual honesty rules R1 (collapsed regime), R2 (degenerate duration) and R3 (opt-in Dmax retry). Design: tests/PREDECLARED_CRITERIA_WP7R.md section 5,
written before the code. Every rule has a boundary test with a negative control; the end-to-end cases run the real `fit_hssm(model_type="dual_timescale")`.
Mutation check: integration_report/wp7r/scripts/mutation_item5.py (every test is killed by at least one mutant).
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pytest

from backbone.hssm import fitting
from backbone.hssm.config import COLLAPSE_MIN_EFFECTIVE_DAYS, DUAL_DURATION_BOUND_TOL, DUAL_DURATION_MU_MAX_DMAX_FACTOR, DUAL_DURATION_MU_MIN, DUAL_DURATION_SIGMA_MAX
from backbone.hssm.dual_model import DualTimescaleHSSM
from backbone.hssm.fitting import HSSMResult, dual_collapse_report, dual_duration_degeneracy, fit_hssm
from backbone.hssm.model import FitState
from tests.utils.item3_stats import reason_prefixes


def _model(mu, sigma, dmax):
    m = DualTimescaleHSSM(n_regimes=len(mu), n_features=3, latent_dim=2, max_duration=dmax, seed=0)
    m.dur_mu = np.array(mu, dtype=float)
    m.dur_sigma = np.array(sigma, dtype=float)
    return m


# ------------------------------------------------------------------------------------------------------------- R1: collapse measure
def test_collapse_threshold_is_the_baseline_constant_and_strict():
    assert COLLAPSE_MIN_EFFECTIVE_DAYS == 4.0
    exactly = np.array([[4.0, 0.0]] + [[0.0, 1.0]] * 56)                         # column sums 4.0 and 56.0
    r = dual_collapse_report(exactly)
    assert r["min_effective_days"] == 4.0 and r["collapsed"] is False             # strictly below: exactly 4.0 is NOT collapsed
    below = np.array([[3.999, 0.0]] + [[0.0, 1.0]] * 56)
    assert dual_collapse_report(below)["collapsed"] is True
    assert r["min_effective_days_threshold"] == 4.0 and r["source"] == "filtered_regime_posterior"


def test_collapse_sums_the_posterior_over_rows_per_regime_and_takes_the_minimum():
    post = np.zeros((30, 3))
    post[:20, 0] = 1.0
    post[20:27, 1] = 1.0
    post[27:, 2] = 1.0                                                            # 20, 7, 3 days
    r = dual_collapse_report(post)
    assert r["effective_days"] == [20.0, 7.0, 3.0] and r["min_effective_days"] == 3.0 and r["collapsed"] is True
    post[27:, 2] = 0.0
    post[20:, 1] = 1.0                                                            # 20, 10, 0
    assert dual_collapse_report(post)["min_effective_days"] == 0.0
    healthy = np.array([[0.6, 0.4]] * 50)                                         # 30 and 20 days: negative control
    assert dual_collapse_report(healthy)["collapsed"] is False and dual_collapse_report(healthy)["min_effective_days"] == pytest.approx(20.0)


# ------------------------------------------------------------------------------------------------------------- R2: degenerate duration
def test_a_healthy_duration_law_is_not_degenerate():
    d = dual_duration_degeneracy(_model([np.log(8.0), np.log(12.0)], [0.3, 0.4], 25))
    assert d["degenerate"] is False and d["degenerate_regimes"] == [] and d["pmf_rising_at_end"] == [False, False] and d["at_bound"] == {0: False, 1: False}
    assert d["Dmax"] == 25


def test_pmf_still_rising_at_the_last_lattice_step_is_degenerate_and_falling_is_not():
    # lognormal mode = exp(mu - sigma^2). Mode beyond Dmax: rising at the end. Mode well inside: falling. Boundary: mode 24 (falling) versus 26 (rising) at Dmax 25.
    far = dual_duration_degeneracy(_model([np.log(60.0) + 0.01, np.log(8.0)], [0.1, 0.3], 25))
    assert far["pmf_rising_at_end"] == [True, False] and far["degenerate_regimes"] == [0] and far["degenerate"] is True
    inside = dual_duration_degeneracy(_model([np.log(24.0) + 0.01, np.log(8.0)], [0.1, 0.3], 25))      # mode 24: pdf(25) < pdf(24)
    assert inside["pmf_rising_at_end"] == [False, False] and inside["degenerate"] is False
    beyond = dual_duration_degeneracy(_model([np.log(26.0) + 0.01, np.log(8.0)], [0.1, 0.3], 25))      # mode 26: pdf(25) > pdf(24)
    assert beyond["pmf_rising_at_end"] == [True, False]
    wide = dual_duration_degeneracy(_model([np.log(60.0) + 0.01, np.log(8.0)], [0.1, 0.3], 80))        # the same law on a lattice that contains the mode
    assert wide["degenerate"] is False


def test_a_one_step_lattice_has_no_last_step_to_compare_and_is_not_flagged_for_rising():
    d = dual_duration_degeneracy(_model([np.log(1.0), np.log(1.0)], [0.3, 0.3], 1))
    assert d["pmf_rising_at_end"] == [False, False] and d["Dmax"] == 1                      # no IndexError, no rising flag


def test_duration_parameters_at_a_wall_of_the_box_are_degenerate_just_inside_is_not():
    tol = DUAL_DURATION_BOUND_TOL
    assert DUAL_DURATION_SIGMA_MAX == 3.0 and DUAL_DURATION_MU_MIN == -5.0
    wall_sigma = dual_duration_degeneracy(_model([np.log(5.0), np.log(5.0)], [DUAL_DURATION_SIGMA_MAX, 0.3], 25))
    assert wall_sigma["at_bound"] == {0: True, 1: False} and wall_sigma["pmf_rising_at_end"] == [False, False] and wall_sigma["degenerate_regimes"] == [0]
    inside_sigma = dual_duration_degeneracy(_model([np.log(5.0), np.log(5.0)], [DUAL_DURATION_SIGMA_MAX - 2 * tol, 0.3], 25))
    assert inside_sigma["at_bound"] == {0: False, 1: False} and inside_sigma["degenerate"] is False
    wall_mu_lo = dual_duration_degeneracy(_model([DUAL_DURATION_MU_MIN, np.log(5.0)], [0.3, 0.3], 25))
    assert wall_mu_lo["at_bound"] == {0: True, 1: False} and wall_mu_lo["degenerate_regimes"] == [0]
    mu_hi = float(np.log(DUAL_DURATION_MU_MAX_DMAX_FACTOR * 25))
    wall_mu_hi = dual_duration_degeneracy(_model([np.log(5.0), mu_hi], [0.3, 0.3], 25))
    assert wall_mu_hi["at_bound"] == {0: False, 1: True} and 1 in wall_mu_hi["degenerate_regimes"]


# ------------------------------------------------------------------------------------------------------------- the withheld reasons
def _result(state=FitState.CONVERGED, **kw):
    return HSSMResult(model=None, fit_state=state, **kw)


def test_a_converged_clean_dual_result_is_not_withheld_and_each_rule_withholds_on_its_own():
    assert _result().withheld_reason is None and _result().insights_withheld is False
    col = _result(fit_collapsed=True, collapse_report={"min_effective_days": 1.0, "min_effective_days_threshold": 4.0})
    assert reason_prefixes(col.withheld_reason) == ["collapsed_regime"] and col.insights_withheld
    deg = _result(fit_duration_degenerate=True, duration_degenerate_report={"degenerate_regimes": [1], "Dmax": 25})
    assert reason_prefixes(deg.withheld_reason) == ["degenerate_duration"] and "[1]" in deg.withheld_reason and "Dmax 25" in deg.withheld_reason and deg.insights_withheld
    both = _result(FitState.MAX_ITER_REACHED, fit_collapsed=True, fit_duration_degenerate=True)
    assert reason_prefixes(both.withheld_reason) == ["fit_state_not_converged", "collapsed_regime", "degenerate_duration"]


# ------------------------------------------------------------------------------------------------------------- end to end, real fits
def _series(seg_lens, F=3, seed=0, sep=2.5, noise=0.5):
    rng = np.random.default_rng(seed)
    means = np.array([[sep] * F, [-sep] * F])
    path = np.concatenate([[i % 2] * n for i, n in enumerate(seg_lens)])
    X = means[path] + rng.normal(0, noise, (len(path), F))
    return (X - X.mean(0)) / X.std(0)


def _fit(X, **kw):
    base = dict(candidate_ks=(2,), n_initializations=1, max_iter=40, random_seed=3, allow_fast_test_fit=True, model_type="dual_timescale", latent_dim=2,
                user_id="t", dataset_hash="t")
    base.update(kw)
    logging.disable(logging.CRITICAL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return fit_hssm(matrix=X, **base)
        finally:
            logging.disable(logging.NOTSET)


X_LONG = _series([20] * 5, seed=2)                 # five segments of 20 rows: with Dmax 6 the duration law cannot fit the lattice


@pytest.fixture(scope="module")
def degenerate_fit():
    return _fit(X_LONG, max_duration=6)


@pytest.fixture(scope="module")
def retried_fit():
    return _fit(X_LONG, max_duration=6, max_duration_retry=30)


def test_default_path_a_degenerate_dual_fit_is_flagged_and_withheld_and_no_retry_is_recorded(degenerate_fit):
    r = degenerate_fit
    assert r.fit_duration_degenerate is True and r.duration_degenerate_report["degenerate"] is True and r.duration_degenerate_report["Dmax"] == 6
    assert r.duration_degenerate_report["pmf_rising_at_end"][1] is True and 1 in r.duration_degenerate_report["degenerate_regimes"]
    assert r.insights_withheld and "degenerate_duration" in reason_prefixes(r.withheld_reason) and r.publishable_outputs() is None
    assert "dmax_retry" not in r.convergence_metadata and r.Dmax == 6                    # default: the option is absent from the metadata and Dmax is as asked
    # R1 wiring: the dual result now carries the collapse report, computed from the filtered posterior the result carries
    assert r.collapse_report["source"] == "filtered_regime_posterior"
    assert r.collapse_report["effective_days"] == pytest.approx(list(np.sum(r.p_t_posterior, axis=0)))


def test_retry_replaces_a_degenerate_fit_when_the_larger_dmax_is_not_degenerate(retried_fit, degenerate_fit):
    r = retried_fit
    info = r.convergence_metadata["dmax_retry"]
    assert info["requested"] == 30 and info["from"] == 6 and info["first_fit_degenerate"] is True and info["adopted"] is True
    assert r.Dmax == 30 and r.fit_duration_degenerate is False and r.duration_degenerate_report["Dmax"] == 30 and r.model.Dmax == 30
    assert r.convergence_metadata["bic_by_k"][2] == r.BIC                               # the retry's BIC is the one reported
    assert any(e.get("dmax_retry") for e in r.convergence_metadata["run_log"]) and not all(e.get("dmax_retry") for e in r.convergence_metadata["run_log"])
    assert degenerate_fit.Dmax == 6                                                      # the default call on the same data was not affected by the option elsewhere


def test_retry_is_not_adopted_when_it_is_still_degenerate():
    r = _fit(X_LONG, max_duration=6, max_duration_retry=8)
    info = r.convergence_metadata["dmax_retry"]
    assert info["adopted"] is False and info["requested"] == 8 and r.Dmax == 6 and r.model.Dmax == 6
    assert r.fit_duration_degenerate is True and r.insights_withheld                      # the first fit is kept, and it is withheld


def test_retry_option_is_inert_when_the_fit_is_not_degenerate():
    X = _series([20] * 5, seed=2)
    plain = _fit(X, max_duration=40, max_iter=25)
    with_option = _fit(X, max_duration=40, max_iter=25, max_duration_retry=80)
    assert plain.fit_duration_degenerate is False and "dmax_retry" not in with_option.convergence_metadata
    for name in ("m_t_posterior_mean", "m_t_uncertainty", "p_t_posterior", "p_t_estimate"):
        assert np.array_equal(getattr(plain, name), getattr(with_option, name)), name       # bit-identical
    assert plain.BIC == with_option.BIC and plain.Dmax == with_option.Dmax == 40 and plain.duration_parameters == with_option.duration_parameters


def test_a_collapsed_dual_fit_is_flagged_and_withheld():
    X = _series([30, 3, 30, 3, 30], seed=5)                                              # two 3-row blips: with K = 3 one regime is left with the blips
    r = _fit(X, candidate_ks=(3,), max_duration=30, max_iter=30)
    assert r.fit_collapsed is True and r.collapse_report["collapsed"] is True and r.collapse_report["min_effective_days"] < COLLAPSE_MIN_EFFECTIVE_DAYS
    assert r.collapse_report["min_effective_days"] == pytest.approx(min(np.sum(r.p_t_posterior, axis=0)))
    assert r.insights_withheld and "collapsed_regime" in reason_prefixes(r.withheld_reason)


def test_a_dual_fit_with_every_regime_above_the_threshold_is_not_collapsed(degenerate_fit):
    assert degenerate_fit.fit_collapsed is False and degenerate_fit.collapse_report["min_effective_days"] >= COLLAPSE_MIN_EFFECTIVE_DAYS
    assert "collapsed_regime" not in reason_prefixes(degenerate_fit.withheld_reason)      # negative control for R1 on a fit that R2 withholds


# ------------------------------------------------------------------------------------------------------------- validation of the new option
@pytest.mark.parametrize("kwargs, why", [
    (dict(model_type="baseline", max_duration_retry=60), "baseline model"),
    (dict(max_duration_retry=45), "equal to max_duration (the default 45)"),
    (dict(max_duration_retry=30, max_duration=45), "smaller than max_duration"),
    (dict(max_duration_retry=60.0), "a float"),
    (dict(max_duration_retry=True), "a bool"),
    (dict(max_duration_retry=60, shared_prior=object()), "with a shared prior"),
])
def test_max_duration_retry_is_validated_before_any_fit(kwargs, why):
    X = _series([20, 20, 20], seed=1)
    base = dict(matrix=X, candidate_ks=(2,), n_initializations=1, max_iter=2, allow_fast_test_fit=True, model_type="dual_timescale")
    base.update(kwargs)
    with pytest.raises(ValueError, match="max_duration_retry"):
        fit_hssm(**base)


def test_the_new_helpers_and_fields_are_exported_where_the_doc_says():
    assert hasattr(fitting, "dual_collapse_report") and hasattr(fitting, "dual_duration_degeneracy")
    r = _result()
    assert r.fit_duration_degenerate is False and r.duration_degenerate_report == {} and r.fit_collapsed is False        # safe defaults for every other path


# ------------------------------------------------------------------------------------------------------------- the contract document says what the code does
def test_contract_doc_section_9_9_matches_the_code(retried_fit):
    from tests.test_wp7r_item0_contract_doc import DOC, _section
    sec = _section("9.9 Dual honesty rules")
    assert "`max_duration_retry`" in sec and 'convergence_metadata["dmax_retry"]' in sec and "bit-identical to the same call without the option" in sec
    assert "`COLLAPSE_MIN_EFFECTIVE_DAYS` = 4.0" in sec and "default `Dmax` is NOT changed" in sec and "(WP7-R item 5c)" in sec
    # the keys the doc lists for the retry record are the keys the code writes
    info = retried_fit.convergence_metadata["dmax_retry"]
    assert set(info) == {"requested", "from", "first_fit_degenerate", "adopted", "retry_fit_state"}
    row = next(line for line in DOC.splitlines() if line.startswith("| `dmax_retry`"))
    for key in info:
        assert f"`{key}`" in row, key
    # every key of the dual collapse and degeneracy reports is named in the doc
    for key in ("effective_days", "min_effective_days", "min_effective_days_threshold", "collapsed", "source"):
        assert f"`{key}`" in DOC, key
    for key in ("at_bound", "pmf_rising_at_end", "degenerate_regimes", "Dmax"):
        assert key in DOC, key
