"""WP10B B2: the two opt-in switches of the dual model, both OFF by default: the weak duration prior (`DURATION_PRIOR_SEGMENTS`) and the wall-clock cap (`MAX_FIT_SECONDS`).
Each switch is pinned by (i) what it does when it is on and (ii) that it changes nothing when it is off."""
from __future__ import annotations

import contextlib
import dataclasses
import math

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.stats import lognorm

from backbone.hssm.dual_model import BackwardPassResult, DualTimescaleHSSM
from backbone.hssm.model import FitState


@contextlib.contextmanager
def flags(**kw):
    old = {k: getattr(DualTimescaleHSSM, k) for k in kw}
    try:
        for k, v in kw.items():
            setattr(DualTimescaleHSSM, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(DualTimescaleHSSM, k, v)


def series(T=60, F=3, seed=0):
    rng = np.random.default_rng(seed)
    lab = np.repeat([0, 1, 0, 1, 0, 1], T // 6)
    return np.array([[(-1.5 if l == 0 else 1.5)] * F for l in lab]) + rng.standard_normal((len(lab), F))


def model(K=2, F=3, Dmax=30, seed=3):
    m = DualTimescaleHSSM(n_regimes=K, n_features=F, latent_dim=2, max_duration=Dmax, seed=seed)
    m.pi = np.full(K, 1.0 / K)
    m.A_slow = np.ones((K, K)) - np.eye(K)
    m.A_slow /= m.A_slow.sum(axis=1, keepdims=True)
    m.dur_mu = np.full(K, np.log(8.0))
    m.dur_sigma = np.full(K, 0.6)
    m.duration_optimizer_success_ = {}
    m.duration_fallback_used_ = False
    m.transition_fallback_rows_ = set()
    return m


def bwd_with_durations(K, T, Dmax, d_by_regime):
    """A BackwardPassResult whose only used fields are set: entry_posterior (mass 1 at the given durations per regime), xi_counts, the smoothed posterior at t = 0."""
    ep = np.zeros((T, K, Dmax))
    for k, ds in d_by_regime.items():
        for i, d in enumerate(ds):
            ep[i, k, d - 1] += 1.0
    xi = np.ones((K, K))
    gam = np.full((T, K), 1.0 / K)
    z = np.zeros((1,))
    return BackwardPassResult(beta=z, regime_posterior_smoothed=gam, gamma=z, entry_posterior=ep, xi_counts=xi, xi_t=z, future_entry_val=z)


def mle(N, Dmax, x0):
    d = np.arange(1, Dmax + 1)

    def f(p):
        mu, ls = p
        s = np.exp(ls)
        if s <= 1e-6 or s > 10:
            return 1e12
        pdf = np.clip(lognorm.pdf(d, s=s, scale=np.exp(mu)), 1e-300, None)
        return -float(np.sum(N * np.log(pdf / pdf.sum())))
    r = minimize(f, x0, method="Nelder-Mead")
    return float(r.x[0]), float(np.exp(r.x[1]))


# ---- the duration prior -------------------------------------------------------------------------------------------------------------------------------------------------------
def test_the_defaults_switch_both_features_off():
    assert DualTimescaleHSSM.DURATION_PRIOR_SEGMENTS == 0.0 and DualTimescaleHSSM.MAX_FIT_SECONDS is None


def test_prior_pmf_is_a_normalised_lognormal_on_the_lattice():
    m = model(Dmax=30)
    d = np.arange(1, 31)
    with flags(DURATION_PRIOR_MEDIAN=10.0, DURATION_PRIOR_SIGMA=1.0):
        p = m._duration_prior_pmf(d)
    ref = lognorm.pdf(d, s=1.0, scale=10.0)
    assert np.allclose(p, ref / ref.sum(), rtol=1e-12) and math.isclose(p.sum(), 1.0, rel_tol=1e-12)
    assert d[np.argmax(p)] < 10 and p[0] > p[-1]


def test_off_the_update_is_the_plain_weighted_fit_and_on_it_equals_the_fit_of_the_augmented_counts():
    Dmax = 60
    b = bwd_with_durations(2, 12, Dmax, {0: [22, 25, 20, 24, 21], 1: [5, 6, 7, 5]})
    m0 = model(Dmax=Dmax)
    m0._update_discrete_params(b)
    N0 = b.entry_posterior[:, 0, :].sum(axis=0)
    mu_off, sg_off = mle(N0, Dmax, [np.log(8.0), np.log(0.6)])
    assert math.isclose(m0.dur_mu[0], mu_off, rel_tol=1e-6) and math.isclose(m0.dur_sigma[0], sg_off, rel_tol=1e-6)
    m1 = model(Dmax=Dmax)
    with flags(DURATION_PRIOR_SEGMENTS=2.0):
        m1._update_discrete_params(b)
        pmf = m1._duration_prior_pmf(np.arange(1, Dmax + 1))
    mu_on, sg_on = mle(N0 + 2.0 * pmf, Dmax, [np.log(8.0), np.log(0.6)])
    assert math.isclose(m1.dur_mu[0], mu_on, rel_tol=1e-6) and math.isclose(m1.dur_sigma[0], sg_on, rel_tol=1e-6)
    assert m1.dur_mu[0] < m0.dur_mu[0] - 0.05          # the pseudo-segments pull a regime of long durations towards the prior median


def test_a_regime_without_expected_segments_is_not_changed_by_the_prior():
    """The empty-regime guard comes BEFORE the prior: pseudo-segments never invent a regime that has no data."""
    Dmax = 30
    b = bwd_with_durations(2, 12, Dmax, {1: [5, 6]})
    m = model(Dmax=Dmax)
    mu0, sg0 = float(m.dur_mu[0]), float(m.dur_sigma[0])
    with flags(DURATION_PRIOR_SEGMENTS=5.0):
        m._update_discrete_params(b)
    assert m.dur_mu[0] == mu0 and m.dur_sigma[0] == sg0 and m.duration_optimizer_success_[0] is False and m.duration_fallback_used_ is True


def test_the_prior_leaves_the_transition_and_initial_updates_untouched():
    Dmax = 30
    b = bwd_with_durations(2, 12, Dmax, {0: [10, 9], 1: [7, 8]})
    a = model(Dmax=Dmax); a._update_discrete_params(b)
    c = model(Dmax=Dmax)
    with flags(DURATION_PRIOR_SEGMENTS=3.0):
        c._update_discrete_params(b)
    assert np.array_equal(a.A_slow, c.A_slow) and np.array_equal(a.pi, c.pi)


def test_a_full_fit_with_the_switch_at_zero_is_bit_identical_to_a_fit_without_touching_it():
    X = series()
    a = DualTimescaleHSSM(n_regimes=2, n_features=3, latent_dim=2, max_duration=20, seed=5).fit(X, n_iter=6)
    with flags(DURATION_PRIOR_SEGMENTS=0.0, MAX_FIT_SECONDS=None):
        b = DualTimescaleHSSM(n_regimes=2, n_features=3, latent_dim=2, max_duration=20, seed=5).fit(X, n_iter=6)
    for n in ("pi", "A_slow", "dur_mu", "dur_sigma", "A_fast", "b_fast", "Q_fast", "C_obs", "d_obs", "R_obs"):
        assert np.array_equal(getattr(a, n), getattr(b, n)), n
    assert a.log_likelihood_history_ == b.log_likelihood_history_ and a.fit_state_ == b.fit_state_ and a.wall_clock_stop_ is False and b.wall_clock_stop_ is False


def test_a_full_fit_with_the_prior_on_differs_from_the_default_fit():
    X = series()
    a = DualTimescaleHSSM(n_regimes=2, n_features=3, latent_dim=2, max_duration=20, seed=5).fit(X, n_iter=6)
    with flags(DURATION_PRIOR_SEGMENTS=2.0):
        b = DualTimescaleHSSM(n_regimes=2, n_features=3, latent_dim=2, max_duration=20, seed=5).fit(X, n_iter=6)
    assert not np.array_equal(a.dur_mu, b.dur_mu)


# ---- the wall-clock cap ----------------------------------------------------------------------------------------------------------------------------------------------------------
class FakeClock:
    def __init__(self, step=1.0):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


def fit_with_clock(cap, n_iter=30, step=1.0):
    m = DualTimescaleHSSM(n_regimes=2, n_features=3, latent_dim=2, max_duration=20, seed=5)
    clk = FakeClock(step)
    m._now = clk
    with flags(MAX_FIT_SECONDS=cap):
        m.fit(series(), n_iter=n_iter)
    return m


def test_the_cap_stops_the_run_after_the_iteration_in_which_it_is_exceeded():
    m = fit_with_clock(cap=3.5)                      # the clock reads 1 at the start, then 2, 3, 4, 5 after iterations 0..3: elapsed 1, 2, 3, 4; the first elapsed above 3.5 is after iteration 3
    assert m.wall_clock_stop_ is True
    assert m.n_iter_ == 4 and len(m.log_likelihood_history_) == 4
    assert m.fit_state_ == FitState.MAX_ITER_REACHED     # fewer than DUAL_CAP_WINDOW + 1 iterations: the window rule cannot certify


def test_without_a_cap_the_same_fit_runs_its_whole_budget_or_converges():
    m = fit_with_clock(cap=None, n_iter=12)
    assert m.wall_clock_stop_ is False and len(m.log_likelihood_history_) >= 8


def test_a_cap_never_exceeded_changes_nothing():
    a = fit_with_clock(cap=None, n_iter=8)
    b = fit_with_clock(cap=1e9, n_iter=8, step=1.0)
    assert b.wall_clock_stop_ is False and a.log_likelihood_history_ == b.log_likelihood_history_ and np.array_equal(a.dur_mu, b.dur_mu)


def test_a_stopped_run_with_a_flat_tail_is_certified_by_the_window_rule_like_an_exhausted_budget():
    """After more than DUAL_CAP_WINDOW iterations the stop is judged exactly as the iteration limit is (window gain)."""
    m = fit_with_clock(cap=14.5, n_iter=60)
    assert m.wall_clock_stop_ is True and m.n_iter_ == len(m.log_likelihood_history_) and m.n_iter_ > 11
    assert m.window_gain_at_cap_ is not None
    assert m.fit_state_ in (FitState.CONVERGED, FitState.MAX_ITER_REACHED)
    assert (m.fit_state_ == FitState.CONVERGED) == (m.convergence_criterion_ == "window_gain_at_cap")


def test_the_flag_is_reset_by_the_next_fit():
    m = fit_with_clock(cap=3.5)
    assert m.wall_clock_stop_ is True
    m.fit(series(), n_iter=3)
    assert m.wall_clock_stop_ is False


def test_the_cap_is_a_strict_inequality_on_the_elapsed_time():
    """cap 4.0: the elapsed time after iteration 3 is exactly 4.0, which does not exceed the cap; the run stops after iteration 4."""
    m = fit_with_clock(cap=4.0)
    assert m.wall_clock_stop_ is True and m.n_iter_ == 5


def test_the_clock_is_never_read_when_the_cap_is_off():
    m = DualTimescaleHSSM(n_regimes=2, n_features=3, latent_dim=2, max_duration=20, seed=5)
    m._now = lambda: pytest.fail("the clock was read although MAX_FIT_SECONDS is None")
    m.fit(series(), n_iter=3)
    assert m.wall_clock_stop_ is False
