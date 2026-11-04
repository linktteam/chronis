"""WP6 item 3, part 2: the dual model's stopping rule at the iteration limit (C2) and its duration box (D1).

Pre-declared in tests/PREDECLARED_CRITERIA_WP6.md sections 5.1, 5.2, 5.4 and 5.5 (before any dual engine code). The default dual fit must stay bit-identical for every fit
whose Nelder-Mead duration results never leave the box and that ended `CONVERGED` by the old rule (identity gate, wp6_evidence/ab_item3b_*); these tests check the new
branches. Mutation set `item3b` (integration_report/wp6/scripts/mutation_wp6.py).
"""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.stats import lognorm

from backbone.hssm import dual_model as dual_module
from backbone.hssm.config import (
    DUAL_CAP_CERT_GAIN_PER_ROW,
    DUAL_CAP_WINDOW,
    DUAL_DURATION_BOUND_TOL,
    DUAL_DURATION_MU_MAX_DMAX_FACTOR,
    DUAL_DURATION_MU_MIN,
    DUAL_DURATION_SIGMA_MAX,
    ColdStartConfig,
)
from backbone.hssm.dual_model import DualTimescaleHSSM
from backbone.hssm.fitting import fit_hssm
from backbone.hssm.model import FitState
from data.synthetic_population import generate_population_user

K, F, L, DMAX, T = 2, 3, 2, 60, 100
MU_LO, MU_HI, SIG_HI = DUAL_DURATION_MU_MIN, float(np.log(DUAL_DURATION_MU_MAX_DMAX_FACTOR * DMAX)), DUAL_DURATION_SIGMA_MAX


# ------------------------------------------------------------------------------------------------ the declared constants
def test_the_declared_constants_have_their_declared_values():
    assert DUAL_CAP_WINDOW == 10 and DUAL_CAP_CERT_GAIN_PER_ROW == 2.0e-4
    assert DUAL_DURATION_MU_MIN == -5.0 and DUAL_DURATION_MU_MAX_DMAX_FACTOR == 10.0 and DUAL_DURATION_SIGMA_MAX == 3.0 and DUAL_DURATION_BOUND_TOL == 1e-3


# ------------------------------------------------------------------------------------------------ the stopping rule at the iteration limit
def scripted_fit(ll_values, n_rows=T, n_iter=None, tol=1e-4):
    """A real DualTimescaleHSSM whose E-step reports the scripted log-likelihoods and whose M-steps do nothing: the loop, the guard, the
    convergence rule and the certification at the limit are the real code, the numbers are ours."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_rows, F))
    m = DualTimescaleHSSM(n_regimes=K, n_features=F, latent_dim=L, max_duration=DMAX, seed=3)
    it = iter(ll_values)
    fwd = lambda self, X_: SimpleNamespace(log_likelihood=float(next(it)))
    with patch.object(DualTimescaleHSSM, "_forward_pass", fwd), \
         patch.object(DualTimescaleHSSM, "_discrete_backward", lambda self, f: None), \
         patch.object(DualTimescaleHSSM, "_switching_smoother", lambda self, f, b: None), \
         patch.object(DualTimescaleHSSM, "_update_discrete_params", lambda self, b: None), \
         patch.object(DualTimescaleHSSM, "_m_step_continuous", lambda self, X_, b, s: None):
        m.fit(X, n_iter=n_iter if n_iter is not None else len(ll_values), tol=tol)
    return m


def rising(n, total_gain_over_window, base=-1000.0, early_step=1.0):
    """n log-likelihoods: steps of `early_step` until the last DUAL_CAP_WINDOW iterations, which gain `total_gain_over_window` in equal steps."""
    head = [base + early_step * i for i in range(n - DUAL_CAP_WINDOW)]
    start = head[-1] + early_step
    # hist[-1] - hist[-1-W] must be exactly total_gain_over_window: hist[-1-W] is head[-1], the last W values rise in W equal steps
    tail = [head[-1] + total_gain_over_window * (j + 1) / DUAL_CAP_WINDOW for j in range(DUAL_CAP_WINDOW)]
    del start
    return head + tail


def test_a_fit_at_the_limit_with_a_gain_just_below_the_threshold_is_certified_converged():
    thr = DUAL_CAP_CERT_GAIN_PER_ROW * T
    m = scripted_fit(rising(40, 0.98 * thr))
    assert m.fit_state_ == FitState.CONVERGED and m.converged_
    assert m.convergence_criterion_ == "window_gain_at_cap"
    assert m.window_gain_at_cap_ == pytest.approx(0.98 * thr, rel=1e-9)
    assert m.n_iter_ == 40


def test_a_fit_at_the_limit_with_a_gain_just_above_the_threshold_stays_max_iter_reached():
    thr = DUAL_CAP_CERT_GAIN_PER_ROW * T
    m = scripted_fit(rising(40, 1.02 * thr))
    assert m.fit_state_ == FitState.MAX_ITER_REACHED and not m.converged_
    assert m.convergence_criterion_ is None
    assert m.window_gain_at_cap_ == pytest.approx(1.02 * thr, rel=1e-9)      # the gain is reported, also when it does not certify
    assert m.n_iter_ == 40                                                   # the number of iterations run, no longer 0


def test_the_threshold_scales_with_the_number_of_rows():
    gain = 0.03                                                              # 2.0e-4 * 100 = 0.02: not certified at T = 100, certified at T = 300, not at T = 50
    assert scripted_fit(rising(40, gain), n_rows=100).fit_state_ == FitState.MAX_ITER_REACHED
    assert scripted_fit(rising(40, gain), n_rows=300).fit_state_ == FitState.CONVERGED
    assert scripted_fit(rising(40, 0.015), n_rows=100).fit_state_ == FitState.CONVERGED
    assert scripted_fit(rising(40, 0.015), n_rows=50).fit_state_ == FitState.MAX_ITER_REACHED


def test_the_window_is_the_last_ten_iterations():
    thr = DUAL_CAP_CERT_GAIN_PER_ROW * T
    lls = rising(40, 0.5 * thr)
    lls[-DUAL_CAP_WINDOW - 1] -= 5.0 * thr                                   # a big gain just OUTSIDE the window (it 29 -> 30) does not matter ...
    lls[-DUAL_CAP_WINDOW:] = [v - 5.0 * thr for v in lls[-DUAL_CAP_WINDOW:]]  # ... keep the in-window gain at 0.5 thr: shift the whole window together
    m = scripted_fit(lls)
    assert m.window_gain_at_cap_ == pytest.approx(0.5 * thr, rel=1e-9) and m.fit_state_ == FitState.CONVERGED
    lls2 = rising(40, 0.5 * thr)
    lls2[-DUAL_CAP_WINDOW - 2] -= 5.0 * thr                                  # a big gain at (it 28 -> 29) ... still outside: certified
    assert scripted_fit(lls2).fit_state_ == FitState.CONVERGED
    lls3 = rising(40, 0.5 * thr)
    lls3[-1] += 5.0 * thr                                                    # a big gain INSIDE the window (last step) is seen
    assert scripted_fit(lls3).fit_state_ == FitState.MAX_ITER_REACHED


def test_a_history_not_longer_than_the_window_is_never_certified():
    m = scripted_fit([-1000.0 + 0.001 * i for i in range(DUAL_CAP_WINDOW)])  # exactly W values: no hist[-1-W]
    assert m.fit_state_ == FitState.MAX_ITER_REACHED and m.window_gain_at_cap_ is None and m.n_iter_ == DUAL_CAP_WINDOW
    m = scripted_fit([-1000.0 + 0.001 * i for i in range(DUAL_CAP_WINDOW + 1)])
    assert m.fit_state_ == FitState.CONVERGED and m.convergence_criterion_ == "window_gain_at_cap" and m.n_iter_ == DUAL_CAP_WINDOW + 1


def test_the_old_rule_is_untouched_and_the_window_rule_is_not_used_when_it_fires():
    lls = [-1000.0 + 5.0 * i for i in range(13)] + [-940.0 + 0.00001, -940.0 + 0.00002]                      # a flat step (< tol) at index 13
    m = scripted_fit(lls, n_iter=60)
    assert m.fit_state_ == FitState.CONVERGED and m.convergence_criterion_ == "abs_tol" and m.window_gain_at_cap_ is None
    assert m.n_iter_ == 13                                                   # unchanged: the iteration at which the rule fired


def test_a_fit_that_ended_by_the_guard_or_a_failure_is_not_certified():
    lls = [-1000.0 + 0.001 * i for i in range(30)] + [-1000.0 + 0.001 * 29 - 0.5]                              # a drop of 0.5 > 1e-3 at the end: rejected update
    with pytest.warns(UserWarning):
        m = scripted_fit(lls, n_iter=60)
    assert m.fit_state_ == FitState.REJECTED_UPDATE and m.convergence_criterion_ is None and m.window_gain_at_cap_ is None


def test_a_negative_window_gain_is_certified_like_a_small_one():
    lls = [-1000.0 + float(i) for i in range(20)]
    for step in [0.0003] * 4 + [-0.0004] * 6:                                # every step above the 1e-4 tolerance, no drop above the 1e-3 guard, net gain -0.0012
        lls.append(lls[-1] + step)
    m = scripted_fit(lls)
    assert m.fit_state_ == FitState.CONVERGED and m.convergence_criterion_ == "window_gain_at_cap"
    assert m.window_gain_at_cap_ == pytest.approx(-0.0012, abs=1e-9) and m.n_iter_ == 30


def test_the_bookkeeping_is_reset_by_the_next_fit():
    thr = DUAL_CAP_CERT_GAIN_PER_ROW * T
    m = scripted_fit(rising(40, 0.5 * thr))
    assert m.convergence_criterion_ == "window_gain_at_cap"
    m.duration_box_applied_.append({"iteration": 0, "regime": 0})
    vals = iter(rising(40, 5.0 * thr))
    X = np.random.default_rng(0).normal(size=(T, F))
    with patch.object(DualTimescaleHSSM, "_forward_pass", lambda self, X_: SimpleNamespace(log_likelihood=float(next(vals)))), \
         patch.object(DualTimescaleHSSM, "_discrete_backward", lambda self, f: None), \
         patch.object(DualTimescaleHSSM, "_switching_smoother", lambda self, f, b: None), \
         patch.object(DualTimescaleHSSM, "_update_discrete_params", lambda self, b: None), \
         patch.object(DualTimescaleHSSM, "_m_step_continuous", lambda self, X_, b, s: None):
        m.fit(X, n_iter=40)
    assert m.fit_state_ == FitState.MAX_ITER_REACHED and m.convergence_criterion_ is None and m.duration_box_applied_ == []


# ------------------------------------------------------------------------------------------------ the duration box
def lognormal_mass(mu, sigma, total=40.0):
    d = np.arange(1, DMAX + 1)
    pdf = lognorm.pdf(d, s=sigma, scale=np.exp(mu))
    return total * pdf / pdf.sum()


def bwd_with(masses):
    """A minimal backward-pass result: entry mass per regime over the duration lattice; the transition counts and the initial posterior are harmless."""
    entry = np.zeros((1, len(masses), DMAX))
    for k, v in enumerate(masses):
        entry[0, k] = v
    return SimpleNamespace(xi_counts=np.ones((len(masses), len(masses))), entry_posterior=entry, regime_posterior_smoothed=np.full((3, len(masses)), 1.0 / len(masses)))


def fresh_model(mu=(2.2, 2.4), sigma=(0.5, 0.55)):
    m = DualTimescaleHSSM(n_regimes=K, n_features=F, latent_dim=L, max_duration=DMAX, seed=1)
    m._init_params(np.random.default_rng(0).normal(size=(T, F)))
    m.dur_mu = np.array(mu, dtype=float)
    m.dur_sigma = np.array(sigma, dtype=float)
    return m


def old_update(mu0, sigma0, N_j_d):
    """The update exactly as it was before WP6 item 3 part 2 (same objective, same optimiser, same start)."""
    d_grid = np.arange(1, DMAX + 1)

    def f(params):
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        if sigma <= 1e-6 or sigma > 10 or not np.isfinite(mu):
            return 1e12
        pdf = lognorm.pdf(d_grid, s=sigma, scale=np.exp(mu))
        pdf = np.clip(pdf, 1e-300, None)
        pdf = pdf / pdf.sum()
        return -float(np.sum(N_j_d * np.log(pdf)))

    res = minimize(f, [mu0, np.log(max(sigma0, 1e-3))], method="Nelder-Mead")
    return float(res.x[0]), float(np.exp(res.x[1])), bool(res.success)


def test_a_healthy_entry_mass_is_not_touched_by_the_box_and_the_update_is_bit_identical():
    mass = [lognormal_mass(np.log(8.0), 0.4), lognormal_mass(np.log(15.0), 0.3)]
    m = fresh_model()
    mu0, sg0 = m.dur_mu.copy(), m.dur_sigma.copy()
    m._update_discrete_params(bwd_with(mass))
    for k in range(K):
        mu_old, sg_old, ok = old_update(mu0[k], sg0[k], mass[k])
        assert ok and MU_LO <= mu_old <= MU_HI and sg_old <= SIG_HI                  # the construction: the old result is inside the box
        assert m.dur_mu[k] == mu_old and m.dur_sigma[k] == sg_old                    # EXACT equality: the same floating-point result
    assert m.duration_box_applied_ == [] and m.duration_optimizer_success_ == {0: True, 1: True} and not m.duration_fallback_used_


D_GRID = np.arange(1, DMAX + 1)


def flat_mass():
    return np.full(DMAX, 40.0 / DMAX)                        # a uniform duration: the best log-normal needs mu = sigma^2 with sigma large (old result: mu 102, sigma at its wall 10)


def all_at_day_one():
    mass = np.zeros(DMAX)
    mass[0] = 40.0                                           # every segment one day long: the old update runs mu below the lower limit
    return mass


def one_over_d():
    return 40.0 * D_GRID ** -1.0 / (D_GRID ** -1.0).sum()   # a heavy tail: the old update keeps mu in the box but sigma goes to its wall 10


def two_spikes():
    mass = np.zeros(DMAX)
    mass[0], mass[-1] = 20.0, 20.0                           # a spike at 1 and a spike at Dmax
    return mass


OUT_OF_BOX_MASSES = [flat_mass, all_at_day_one, one_over_d, two_spikes]
IDS = ["flat", "all_at_day_one", "one_over_d", "two_spikes"]


@pytest.mark.parametrize("make_mass", OUT_OF_BOX_MASSES, ids=IDS)
def test_an_entry_mass_that_drives_the_old_update_out_of_the_box_is_boxed(make_mass):
    mass = make_mass()
    m = fresh_model()
    mu_old, sg_old, ok = old_update(m.dur_mu[0], m.dur_sigma[0], mass)
    assert ok and (mu_old < MU_LO or mu_old > MU_HI or sg_old > SIG_HI), (mu_old, sg_old)       # negative control: the old update does leave the box
    other = lognormal_mass(np.log(10.0), 0.4)
    m._update_discrete_params(bwd_with([mass, other]))
    assert MU_LO <= m.dur_mu[0] <= MU_HI and 0 < m.dur_sigma[0] <= SIG_HI and np.isfinite(m.dur_mu[0])
    assert m.duration_optimizer_success_[0] is True
    assert [a["regime"] for a in m.duration_box_applied_] == [0]                    # only the regime that left the box, and it is recorded
    rec = m.duration_box_applied_[0]
    assert rec["iteration"] == 0 and rec["from_mu"] == 2.2 and rec["from_sigma"] == 0.5
    assert (rec["to_mu"], rec["to_sigma"]) == (m.dur_mu[0], m.dur_sigma[0])
    mu_o, sg_o, _ = old_update(2.4, 0.55, other)                                      # the other regime is exactly as the old update left it
    assert m.dur_mu[1] == mu_o and m.dur_sigma[1] == sg_o


def test_the_boxed_update_is_not_worse_than_staying_at_the_previous_iterate():
    mass = flat_mass()
    d = np.arange(1, DMAX + 1)

    def q(mu, sigma):
        pdf = lognorm.pdf(d, s=sigma, scale=np.exp(mu))
        pdf = np.clip(pdf, 1e-300, None)
        return float(np.sum(mass * np.log(pdf / pdf.sum())))

    m = fresh_model()
    before = q(m.dur_mu[0], m.dur_sigma[0])
    m._update_discrete_params(bwd_with([mass, lognormal_mass(np.log(10.0), 0.4)]))
    assert q(m.dur_mu[0], m.dur_sigma[0]) > before


def test_the_redo_starts_at_the_previous_iterate_clipped_into_the_box_and_is_penalised_outside():
    mass = flat_mass()
    m = fresh_model(mu=(2.2, 2.4), sigma=(0.5, 0.55))
    m.dur_mu[0], m.dur_sigma[0] = 50.0, 8.0                                          # a previous iterate OUTSIDE the box (cannot happen in a fit, but the start must be clipped)
    real_minimize = dual_module.minimize
    calls = []

    def spy(fun, x0, **kw):
        calls.append((fun, list(x0)))
        return real_minimize(fun, x0, **kw)

    with patch.object(dual_module, "minimize", spy):
        m._update_discrete_params(bwd_with([mass, lognormal_mass(np.log(10.0), 0.4)]))
    boxed_calls = [c for c in calls if c[0].__name__ == "boxed"]
    assert len(boxed_calls) == 1
    fun, x0 = boxed_calls[0]
    assert x0[0] == pytest.approx(MU_HI) and x0[1] == pytest.approx(np.log(SIG_HI))
    for params in ([MU_HI + 0.1, np.log(1.0)], [MU_LO - 0.1, np.log(1.0)], [2.0, np.log(SIG_HI) + 0.1], [np.nan, 0.0], [2.0, np.inf]):
        assert fun(params) == 1e12                                                    # the penalty outside the box, also for non-finite parameters
    assert fun([2.0, np.log(1.0)]) < 1e12 and fun([MU_HI, np.log(SIG_HI) - 1e-9]) < 1e12 and fun([MU_LO, np.log(1.0)]) < 1e12      # inside, the mu walls themselves are allowed


def test_a_failed_redo_takes_the_old_fallback_path():
    mass = flat_mass()
    m = fresh_model()
    real_minimize = dual_module.minimize

    def failing_boxed(fun, x0, **kw):
        res = real_minimize(fun, x0, **kw)
        if fun.__name__ == "boxed":
            res.success = False
        return res

    with patch.object(dual_module, "minimize", failing_boxed):
        m._update_discrete_params(bwd_with([mass, lognormal_mass(np.log(10.0), 0.4)]))
    assert (m.dur_mu[0], m.dur_sigma[0]) == (2.2, 0.5) and m.duration_optimizer_success_[0] is False and m.duration_fallback_used_
    assert m.duration_box_applied_ == []                                              # nothing is recorded for a redo that was not used


def test_an_unsuccessful_first_result_keeps_the_old_fallback_even_when_it_lies_outside_the_box():
    mass = flat_mass()
    m = fresh_model()
    real_minimize = dual_module.minimize

    def unsuccessful(fun, x0, **kw):
        res = real_minimize(fun, x0, **kw)
        res.success = False
        return res

    with patch.object(dual_module, "minimize", unsuccessful):
        m._update_discrete_params(bwd_with([mass, lognormal_mass(np.log(10.0), 0.4)]))
    assert (m.dur_mu[0], m.dur_sigma[0]) == (2.2, 0.5) and m.duration_optimizer_success_[0] is False and m.duration_box_applied_ == []     # clarification 5.5 (a)


def test_the_box_limits_are_the_declared_ones_at_a_wall():
    m = fresh_model()
    assert m._duration_box() == (-5.0, pytest.approx(np.log(10.0 * DMAX)), 3.0)
    big = DualTimescaleHSSM(n_regimes=K, n_features=F, latent_dim=L, max_duration=200, seed=1)
    assert big._duration_box()[1] == pytest.approx(np.log(2000.0))                    # the upper limit of mu follows Dmax


# ------------------------------------------------------------------------------------------------ the at-bound flag
def test_duration_at_bound_flags_a_final_value_within_the_tolerance_of_a_wall():
    m = fresh_model()
    assert m.duration_at_bound_ == {0: False, 1: False}
    m.dur_mu, m.dur_sigma = np.array([MU_HI - 0.5 * DUAL_DURATION_BOUND_TOL, MU_LO + 0.5 * DUAL_DURATION_BOUND_TOL]), np.array([1.0, 1.0])
    assert m.duration_at_bound_ == {0: True, 1: True}
    m.dur_mu, m.dur_sigma = np.array([2.0, 2.0]), np.array([SIG_HI - 0.5 * DUAL_DURATION_BOUND_TOL, 1.0])
    assert m.duration_at_bound_ == {0: True, 1: False}
    m.dur_mu, m.dur_sigma = np.array([MU_HI - 2.0 * DUAL_DURATION_BOUND_TOL, MU_LO + 2.0 * DUAL_DURATION_BOUND_TOL]), np.array([SIG_HI - 2.0 * DUAL_DURATION_BOUND_TOL, 1.0])
    assert m.duration_at_bound_ == {0: False, 1: False}                               # just outside the tolerance
    m.dur_mu = None
    assert m.duration_at_bound_ == {}


def test_a_boxed_update_that_ends_on_a_wall_is_flagged():
    mass = one_over_d()
    m = fresh_model()
    m._update_discrete_params(bwd_with([mass, lognormal_mass(np.log(10.0), 0.4)]))
    assert m.duration_box_applied_ and any(a["regime"] == 0 for a in m.duration_box_applied_)
    mu, sg = m.dur_mu[0], m.dur_sigma[0]
    assert abs(sg - SIG_HI) <= DUAL_DURATION_BOUND_TOL and MU_LO < mu < MU_HI         # the construction: the 1/d mass ends on the sigma wall (sigma 2.99999998)
    assert m.duration_at_bound_ == {0: True, 1: False}


# ------------------------------------------------------------------------------------------------ the result
@pytest.fixture(scope="module")
def dual_result():
    X = generate_population_user(21, 3, 0).X_train
    return fit_hssm(X, candidate_ks=(2,), n_initializations=2, max_iter=3, allow_fast_test_fit=True, config=ColdStartConfig(min_present_sessions=10),
                    model_type="dual_timescale", max_duration=25, random_seed=7)


def test_the_dual_result_carries_the_new_fields(dual_result):
    r, m = dual_result, dual_result.model
    assert r.duration_at_bound == m.duration_at_bound_ and set(r.duration_at_bound) == {0, 1}
    md = r.convergence_metadata
    assert md["convergence_criterion"] == m.convergence_criterion_ and md["window_gain_at_cap"] == m.window_gain_at_cap_
    assert md["duration_box"]["n_applications"] == len(m.duration_box_applied_) == len(md["duration_box"]["applied"])
    assert md["n_iter"] == m.n_iter_ == 3                                             # 3 iterations at the limit, no longer 0
    assert r.fit_state == FitState.MAX_ITER_REACHED and r.insights_withheld            # fewer than W + 1 iterations: never certified, still withheld (P0-07)


def test_the_baseline_result_has_an_empty_at_bound_field():
    X = generate_population_user(21, 3, 0).X_train
    r = fit_hssm(X, candidate_ks=(2,), n_initializations=2, max_iter=15, allow_fast_test_fit=True, config=ColdStartConfig(min_present_sessions=10))
    assert r.duration_at_bound == {} and "duration_box" not in r.convergence_metadata


def test_the_result_reports_a_box_application_and_an_at_bound_regime():
    """Wiring check: a box application recorded by the model, and a duration at a wall, reach the result (the model's own behaviour is tested above)."""
    X = generate_population_user(21, 3, 0).X_train
    real_fit_impl = DualTimescaleHSSM._fit_impl

    def fit_and_pretend(self, *a, **kw):
        out = real_fit_impl(self, *a, **kw)
        self.duration_box_applied_.append({"iteration": 1, "regime": 0, "from_mu": 2.0, "from_sigma": 0.5, "to_mu": 5.0, "to_sigma": 1.0})
        self.dur_mu = np.array(self.dur_mu, dtype=float)
        self.dur_mu[0] = float(np.log(DUAL_DURATION_MU_MAX_DMAX_FACTOR * 25))        # the upper limit of mu for Dmax = 25
        return out

    with patch.object(DualTimescaleHSSM, "_fit_impl", fit_and_pretend):
        r = fit_hssm(X, candidate_ks=(2,), n_initializations=2, max_iter=3, allow_fast_test_fit=True, config=ColdStartConfig(min_present_sessions=10),
                     model_type="dual_timescale", max_duration=25, random_seed=7)
    box = r.convergence_metadata["duration_box"]
    assert box["n_applications"] == len(r.model.duration_box_applied_) >= 1 and box["applied"] == r.model.duration_box_applied_     # the fit may itself have applied the box (it does, at iteration 0, on this record)
    assert {"iteration": 1, "regime": 0, "from_mu": 2.0, "from_sigma": 0.5, "to_mu": 5.0, "to_sigma": 1.0} in box["applied"]        # the pretended application arrived
    assert r.duration_at_bound[0] is True and set(r.duration_at_bound) == {0, 1}
