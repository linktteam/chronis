"""WP6 item 3, part 1: baseline duration repair, collapse report and selection, multi-start summary, P0-07 withholding.
Pre-declaration: tests/PREDECLARED_CRITERIA_WP6.md section 4; diagnosis: tests/WP6_CONVERGENCE_DIAGNOSIS.md. Every group has a negative control, and the
mutation script (integration_report/wp6/scripts/mutation_wp6.py, set item3) must catch the mutants listed there."""
import pathlib
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.stats import lognorm

from backbone.hssm import fitting
from backbone.hssm.config import COLLAPSE_MIN_EFFECTIVE_DAYS, RESTART_EPSILON_NATS, RESTART_SAME_ARI
from backbone.hssm.fitting import HSSMResult, adjusted_rand_index, collapse_report, fit_dual_with_random_restarts, fit_with_random_restarts, select_k_by_bic
from backbone.hssm.model import FitState, FittingConvergenceError, GaussianHSMM, _neg_duration_loglik

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _x50():
    return np.random.default_rng(42).normal(size=(50, 3))


def _fit(seed, X=None, n_iter=50, tol=1e-4):
    m = GaussianHSMM(n_regimes=3, n_features=3, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(_x50() if X is None else X, n_iter=n_iter, tol=tol)
    return m


def _monotone(h, tol=1e-9):
    return all(b >= a - tol for a, b in zip(h, h[1:]))


# ------------------------------------------------------------------------------------------------ 4.1 the repair
def test_a_natural_guard_event_is_repaired_and_the_fit_goes_on():
    """Seed 0 on this data ends REJECTED_UPDATE in the WP5 engine (iteration 18, drop 0.0596). Now: repaired, the fit continues, the history is monotone."""
    m = _fit(0)
    assert len(m.guard_repairs_) == 1 and m.guard_repairs_[0]["iteration"] == 18 and 0.05 < m.guard_repairs_[0]["drop"] < 0.07
    assert m.fit_state_ == FitState.CONVERGED and m.n_iter_ > 18 and m.duration_safeguard_active_
    assert _monotone(m.log_likelihood_history_) and m.is_log_likelihood_monotonic(tol=1e-9)
    assert m.log_likelihood_ == m.log_likelihood_history_[-1]


def test_a_fit_without_a_guard_event_is_untouched():
    m = _fit(2, tol=1e-3)                                                          # converges without any rejection in the WP5 engine
    assert m.guard_repairs_ == [] and not m.duration_safeguard_active_ and m.fit_state_ == FitState.CONVERGED


def _old_fit(seed, X, n_iter=50, tol=1e-4):
    """The EM loop of the WP5 engine (871a3ab) with the same helper methods: the reference for 'no event, no change'."""
    m = GaussianHSMM(n_regimes=3, n_features=3, seed=seed)
    m.log_likelihood_history_ = []
    m._init_params(X)
    prev_params, prev_ll = None, None
    state = None
    for it in range(n_iter):
        g, e, x = m._forward_backward(X)
        ll = m.log_likelihood_
        m.log_likelihood_history_.append(ll)
        if it > 0 and abs(ll - prev_ll) < tol:
            state = FitState.CONVERGED
            break
        if it > 0 and ll < prev_ll - 1e-4:
            state = FitState.REJECTED_UPDATE
            break
        prev_params = None
        prev_ll = ll
        m._m_step(X, g, e, x)
    else:
        state = FitState.MAX_ITER_REACHED
    return m, state


@pytest.mark.parametrize("seed,tol", [(2, 1e-3), (4, 1e-4), (5, 1e-4)])
def test_without_an_event_the_fit_equals_the_old_loop_bit_for_bit(seed, tol):
    X = _x50()
    new = _fit(seed, X, tol=tol)
    old, state = _old_fit(seed, X, tol=tol)
    assert new.guard_repairs_ == [] and state == new.fit_state_ == FitState.CONVERGED
    assert new.log_likelihood_history_ == old.log_likelihood_history_
    for name in ("pi", "A", "mu", "var", "dur_mu", "dur_sigma"):
        np.testing.assert_array_equal(getattr(new, name), getattr(old, name))


def _inject_drop(monkeypatch, at_call, drop):
    """Make the `at_call`-th E-step report `drop` nats less than the log-likelihood of the E-step before it (a controlled guard event)."""
    real = GaussianHSMM._forward_backward
    state = {"n": 0, "last": None}

    def fb(self, X):
        out = real(self, X)
        state["n"] += 1
        if state["n"] == at_call:
            self.log_likelihood_ = state["last"] - drop
        state["last"] = self.log_likelihood_
        return out

    monkeypatch.setattr(GaussianHSMM, "_forward_backward", fb)


def test_an_injected_drop_above_the_tolerance_is_repaired_and_one_at_noise_level_is_not_an_event(monkeypatch):
    with monkeypatch.context() as mp:
        _inject_drop(mp, at_call=4, drop=5e-4)
        m = _fit(4)
    assert [r["iteration"] for r in m.guard_repairs_] == [3] and abs(m.guard_repairs_[0]["drop"] - 5e-4) < 1e-9
    assert _monotone(m.log_likelihood_history_) and m.fit_state_ == FitState.CONVERGED
    with monkeypatch.context() as mp:                                                # negative control: a drop below the tolerance is not an event
        _inject_drop(mp, at_call=4, drop=5e-5)
        c = _fit(4)
    assert c.guard_repairs_ == [] and c.fit_state_ == FitState.CONVERGED and c.n_iter_ == 3


def test_the_repair_redoes_the_previous_m_step_from_the_previous_parameters(monkeypatch):
    seen = []
    real = GaussianHSMM._m_step

    def spy(self, X, g, e, x):
        seen.append({n: np.array(getattr(self, n), copy=True) for n in ("pi", "A", "mu", "var", "dur_mu", "dur_sigma")} | {"safe": self.duration_safeguard_active_, "g": g})
        return real(self, X, g, e, x)

    monkeypatch.setattr(GaussianHSMM, "_m_step", spy)
    with monkeypatch.context() as mp:
        _inject_drop(mp, at_call=4, drop=5e-4)
        m = _fit(4)
    # M-steps: calls 0..2 after E-steps 1..3, call 3 is the repair (E-step 4 reported the drop), call 4 is the next iteration's M-step
    assert not seen[2]["safe"] and seen[3]["safe"] and seen[3]["g"] is seen[2]["g"]            # same statistics, safeguard on
    for n in ("pi", "A", "mu", "var", "dur_mu", "dur_sigma"):
        np.testing.assert_array_equal(seen[3][n], seen[2][n])                                  # started from the previous parameters, not the rejected ones
    assert m.guard_repairs_


def test_the_safeguard_stays_on_for_the_rest_of_the_fit(monkeypatch):
    calls = {"n": 0}
    real = GaussianHSMM._safeguarded_duration_update

    def count(self, *a, **k):
        calls["n"] += 1
        return real(self, *a, **k)

    monkeypatch.setattr(GaussianHSMM, "_safeguarded_duration_update", count)
    m = _fit(0)
    n_after = m.n_iter_ - m.guard_repairs_[0]["iteration"]                                      # M-steps after the repair iteration
    assert calls["n"] >= 3 * (1 + n_after)                                                      # the repair itself plus every later M-step, K = 3 regimes each
    calls["n"] = 0
    _fit(2, tol=1e-3)
    assert calls["n"] == 0                                                                       # negative control: no event, never called


def test_a_repaired_update_that_still_drops_ends_rejected_with_the_previous_parameters(monkeypatch):
    real = GaussianHSMM._safeguarded_duration_update

    def bad(self, k, w, d, mu0, s0):
        real(self, k, w, d, mu0, s0)
        self.dur_mu[k] += 3.0

    monkeypatch.setattr(GaussianHSMM, "_safeguarded_duration_update", bad)
    m = _fit(0)
    assert m.fit_state_ == FitState.REJECTED_UPDATE and len(m.guard_repairs_) == 1
    h = m.log_likelihood_history_
    assert h[-1] == m.log_likelihood_ and h[-2] < h[-3] - 1e-4 and np.isclose(h[-1], h[-3])     # same shape of evidence as before WP6


# ------------------------------------------------------------------------------------------------ the safeguarded duration update
def _random_weights(rng, D):
    kind = rng.integers(0, 4)
    d = np.arange(1, D + 1)
    if kind == 0:
        w = np.exp(-0.5 * ((np.log(d) - rng.uniform(0.5, 3)) / rng.uniform(0.2, 1.2)) ** 2)
    elif kind == 1:                                   # bimodal: very short visits and long ones
        w = np.zeros(D)
        w[0] = rng.uniform(0.5, 3)
        w[rng.integers(D // 2, D)] = rng.uniform(1, 4)
    elif kind == 2:
        w = rng.uniform(0, 1, D)
    else:
        w = np.zeros(D)
        w[-3:] = rng.uniform(0.2, 2, 3)
    return w * rng.uniform(2, 12) / max(w.sum(), 1e-9)


def test_the_safeguarded_duration_update_never_lowers_the_duration_block_q_and_the_old_update_can():
    D = 20
    d_vals = np.arange(1, D + 1)
    log_d = np.log(d_vals)
    rng = np.random.default_rng(6)
    lowered_by_old = 0
    m = GaussianHSMM(n_regimes=2, n_features=2, max_duration=D, seed=0)
    for _ in range(60):
        w = _random_weights(rng, D)
        prev = (rng.uniform(-2, 5), rng.uniform(0.05, 4.5))
        init_mu = float((w * log_d).sum() / w.sum())
        init_sigma = float(np.sqrt(max((w * (log_d - init_mu) ** 2).sum() / w.sum(), 1e-3)))
        m.dur_mu = np.array([prev[0], 0.0]); m.dur_sigma = np.array([prev[1], 1.0])
        m.duration_optimizer_success_ = {0: True, 1: True}; m.duration_optimizer_messages_ = {0: "", 1: ""}
        q_prev = -_neg_duration_loglik([prev[0], prev[1]], w, d_vals)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m._safeguarded_duration_update(0, w, d_vals, init_mu, init_sigma)
        q_new = -_neg_duration_loglik([m.dur_mu[0], m.dur_sigma[0]], w, d_vals)
        assert q_new >= q_prev - 1e-12
        old = minimize(lambda p: _neg_duration_loglik(p, w, d_vals), [init_mu, init_sigma], method="L-BFGS-B", bounds=[(None, None), (1e-3, 5.0)],
                       options={"maxiter": 5, "ftol": 1e-2})
        q_old = -_neg_duration_loglik(old.x if old.success else [init_mu, init_sigma], w, d_vals)
        lowered_by_old += q_old < q_prev - 1e-9
    assert lowered_by_old >= 5                         # negative control: the old rule does lower Q in a good share of these draws


def test_the_safeguarded_update_reports_which_candidate_won_and_a_fallback_only_when_it_is_the_winner():
    D = 15
    d_vals = np.arange(1, D + 1)
    w = np.exp(-0.5 * ((np.log(d_vals) - 1.8) / 0.4) ** 2) * 6
    m = GaussianHSMM(n_regimes=1, n_features=1, max_duration=D, seed=0)
    m.dur_mu, m.dur_sigma = np.array([1.8]), np.array([0.4])
    m.duration_optimizer_success_ = {0: True}; m.duration_optimizer_messages_ = {0: ""}; m.duration_fallback_used_ = False
    m._safeguarded_duration_update(0, w, d_vals, 1.8, 0.4)
    assert m.duration_optimizer_success_[0] is True and m.duration_fallback_used_ is False and m.duration_optimizer_messages_[0].startswith("safeguarded update: ")


def test_the_previous_iterate_is_kept_when_every_optimiser_candidate_is_worse(monkeypatch):
    """The previous iterate is itself a candidate: if every optimiser run returns something worse, the update keeps it (the block's Q cannot fall)."""
    D = 15
    d_vals = np.arange(1, D + 1)
    w = np.exp(-0.5 * ((np.log(d_vals) - 1.8) / 0.4) ** 2) * 6
    m = GaussianHSMM(n_regimes=1, n_features=1, max_duration=D, seed=0)
    m.dur_mu, m.dur_sigma = np.array([1.8]), np.array([0.4])                        # the exact optimum of this weight vector is close to here
    m.duration_optimizer_success_ = {0: True}; m.duration_optimizer_messages_ = {0: ""}; m.duration_fallback_used_ = False

    def terrible(fun, x0, **kw):
        return SimpleNamespace(x=np.array([x0[0] + 3.0, x0[1] + 1.5]), success=True, message="stub")
    monkeypatch.setattr("backbone.hssm.model.minimize", terrible)
    m._safeguarded_duration_update(0, w, d_vals, 3.0, 0.5)                           # moment-estimate start is also worse than the previous iterate
    assert m.dur_mu[0] == 1.8 and m.dur_sigma[0] == 0.4 and "previous_iterate" in m.duration_optimizer_messages_[0]


# ------------------------------------------------------------------------------------------------ 4.3 collapse
def _stub_posterior(monkeypatch, model, col_sums, T=40):
    post = np.zeros((T, len(col_sums)))
    for k, c in enumerate(col_sums):
        post[:, k] = c / T
    post = post / post.sum(axis=1, keepdims=True) if False else post
    monkeypatch.setattr(model, "_forward_backward", lambda grid: (post, None, None), raising=False)


def _model_with_var(var):
    m = GaussianHSMM(n_regimes=3, n_features=2, seed=0)
    m.var = np.array(var, dtype=float)
    m.log_likelihood_ = -123.0
    return m


def test_collapse_threshold_is_four_effective_days_strictly_below(monkeypatch):
    assert COLLAPSE_MIN_EFFECTIVE_DAYS == 4.0
    X = np.zeros((40, 2))
    m = _model_with_var(np.ones((3, 2)))
    _stub_posterior(monkeypatch, m, [20.0, 4.0, 3.99])
    rep, labels = collapse_report(m, X)
    assert rep["collapsed"] is True and abs(rep["min_effective_days"] - 3.99) < 1e-12 and rep["effective_days"] == pytest.approx([20.0, 4.0, 3.99])
    _stub_posterior(monkeypatch, m, [20.0, 4.0, 4.0])
    assert collapse_report(m, X)[0]["collapsed"] is False                                   # exactly 4.0 is not collapsed
    assert m.log_likelihood_ == -123.0                                                        # the model's log-likelihood is not changed


def test_a_variance_at_the_floor_is_counted_but_does_not_make_a_fit_collapsed(monkeypatch):
    X = np.zeros((40, 2))
    m = _model_with_var([[1e-3, 1.0005e-3], [1.002e-3, 1.0], [1.0, 1.0]])
    _stub_posterior(monkeypatch, m, [20.0, 10.0, 10.0])
    rep, _ = collapse_report(m, X)
    assert rep["n_variance_at_floor"] == 2 and rep["collapsed"] is False


def test_effective_days_use_the_smoothed_posterior_summed_over_rows_not_columns(monkeypatch):
    X = np.zeros((40, 2))
    m = _model_with_var(np.ones((3, 2)))
    post = np.zeros((40, 3)); post[:, 0] = 1.0                                               # regime 0 holds every day, regimes 1 and 2 none
    monkeypatch.setattr(m, "_forward_backward", lambda grid: (post, None, None), raising=False)
    rep, labels = collapse_report(m, X)
    assert rep["effective_days"] == [40.0, 0.0, 0.0] and rep["collapsed"] is True and set(labels) == {0}


def test_collapse_report_on_real_fits_matches_the_posterior():
    rng = np.random.default_rng(8)
    X = rng.normal(0, 0.5, (60, 3)) + np.repeat([-3.0, 0.0, 3.0], 20)[:, None]               # three clear regimes of 20 days
    m = GaussianHSMM(n_regimes=3, n_features=3, max_duration=25, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, n_iter=100)
    rep, labels = collapse_report(m, X)
    assert abs(sum(rep["effective_days"]) - 60.0) < 1e-6 and len(labels) == 60 and rep["collapsed"] is False and rep["min_effective_days"] > 15
    assert sorted(np.bincount(labels).tolist()) == [20, 20, 20]
    m2 = _fit(2, tol=1e-3)                                                                    # noise data with K = 3: one regime holds about 2 days
    rep2, _ = collapse_report(m2, _x50())
    assert rep2["collapsed"] is True and rep2["min_effective_days"] < 4 and abs(sum(rep2["effective_days"]) - 50.0) < 1e-6


# ------------------------------------------------------------------------------------------------ restart selection and summary
class _Plan:
    """Replace GaussianHSMM.fit and the collapse report so that restart i ends in a designated state with designated log-likelihood, collapse flag and labels."""

    def __init__(self, monkeypatch, rows, mu_descending=False):
        self.rows, self.n, self.mu_descending = rows, 0, mu_descending
        plan = self

        def fit(model, X, n_iter=100, tol=1e-4, verbose=False, timestamps=None):
            return plan._fit(model)
        monkeypatch.setattr(GaussianHSMM, "fit", fit)
        monkeypatch.setattr(fitting, "collapse_report", self._report)

    def _fit(self, model):
        state, ll, collapsed, labels = self.rows[self.n]
        self.n += 1
        K, F = model.K, model.F
        model._plan = (state, ll, collapsed, np.array(labels))
        model.fit_state_, model.log_likelihood_, model.n_iter_ = state, ll, 5
        model.log_likelihood_history_ = [ll - 1.0, ll]
        model.pi = np.full(K, 1.0 / K); model.A = np.full((K, K), 1.0 / K); model.mu = (np.arange(K * F, 0, -1, dtype=float) if self.mu_descending else np.arange(K * F, dtype=float)).reshape(K, F)
        model.var = np.ones((K, F)); model.dur_mu = np.zeros(K); model.dur_sigma = np.ones(K)
        model.duration_optimizer_success_ = {k: True for k in range(K)}; model.duration_optimizer_messages_ = {k: "" for k in range(K)}
        model._is_fitted = True
        return model

    @staticmethod
    def _report(model, X, timestamps=None):
        state, ll, collapsed, labels = model._plan
        eff = [2.0, 30.0, 18.0] if collapsed else [12.0, 20.0, 18.0]
        return {"effective_days": eff, "min_effective_days": min(eff), "n_variance_at_floor": 0, "min_effective_days_threshold": 4.0, "collapsed": collapsed}, labels


C, R, M = FitState.CONVERGED, FitState.REJECTED_UPDATE, FitState.MAX_ITER_REACHED
LAB = [0, 0, 0, 1, 1, 1, 2, 2, 2]
LAB_PERMUTED = [2, 2, 2, 0, 0, 0, 1, 1, 1]
LAB_OTHER = [0, 1, 2, 0, 1, 2, 0, 1, 2]


def _run(monkeypatch, rows, mu_descending=False):
    _Plan(monkeypatch, rows, mu_descending)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_with_random_restarts(np.zeros((9, 3)), 3, 3, n_init=len(rows), n_iter=5, bypass_init_gate=True)


def test_selection_prefers_the_best_converged_restart_that_is_not_collapsed(monkeypatch):
    model, log = _run(monkeypatch, [(C, -100.0, True, LAB), (C, -110.0, False, LAB), (C, -120.0, False, LAB_OTHER), (R, -90.0, False, LAB)])
    assert model.log_likelihood_ == -110.0 and model.collapse_report_["collapsed"] is False
    assert [r["collapsed"] for r in log] == [True, False, False, False] and all("guard_repairs" in r for r in log)


def test_selection_is_unchanged_when_the_best_restart_is_healthy_and_takes_the_first_maximum_on_ties(monkeypatch):
    model, _ = _run(monkeypatch, [(C, -120.0, False, LAB), (C, -100.0, False, LAB_OTHER), (C, -100.0, False, LAB), (C, -90.0, True, LAB)])
    assert model.log_likelihood_ == -100.0 and np.array_equal(model.mu, np.arange(9.0).reshape(3, 3)[np.argsort(np.linalg.norm(np.arange(9.0).reshape(3, 3), axis=1))])
    assert model.restart_summary_["ari_to_selected"][0] == pytest.approx(adjusted_rand_index(LAB_OTHER, LAB))          # the first of the two tied maxima was selected


def test_when_every_converged_restart_is_collapsed_the_best_is_returned_and_flagged(monkeypatch):
    model, _ = _run(monkeypatch, [(C, -100.0, True, LAB), (C, -90.0, True, LAB_OTHER), (R, -50.0, False, LAB)])
    assert model.log_likelihood_ == -90.0 and model.collapse_report_["collapsed"] is True
    assert model.restart_summary_["ari_to_selected"] == [] and model.restart_summary_["ari_median"] is None and model.restart_summary_["gap_to_next_distinct"] is None


def test_no_converged_restart_still_raises(monkeypatch):
    _Plan(monkeypatch, [(R, -1.0, False, LAB), (M, -2.0, False, LAB)])
    with pytest.raises(FittingConvergenceError):
        fit_with_random_restarts(np.zeros((9, 3)), 3, 3, n_init=2, n_iter=5, bypass_init_gate=True)


def test_the_effective_days_in_the_report_follow_the_canonical_regime_order(monkeypatch):
    model, _ = _run(monkeypatch, [(C, -100.0, False, LAB)])
    # the stub's regime means are 0..8 in blocks, so the canonical order (ascending norm) is the identity and the list is unchanged
    assert model.collapse_report_["effective_days"] == [12.0, 20.0, 18.0]
    assert model._label_order_applied == [0, 1, 2]
    rev, _ = _run(monkeypatch, [(C, -100.0, False, LAB)], mu_descending=True)                  # regime norms descending: canonical order is reversed
    assert rev._label_order_applied == [2, 1, 0] and rev.collapse_report_["effective_days"] == [18.0, 20.0, 12.0]


def test_restart_summary_values(monkeypatch):
    rows = [(C, -100.0, False, LAB),             # selected (best healthy)
            (C, -100.5, False, LAB_PERMUTED),    # same solution under another labelling, within 1 nat
            (C, -103.0, False, LAB_OTHER),       # a different solution
            (C, -90.0, True, LAB),               # collapsed: counted, never compared
            (R, -80.0, False, LAB),
            (M, -85.0, False, LAB)]
    model, _ = _run(monkeypatch, rows)
    s = model.restart_summary_
    ari_other = adjusted_rand_index(LAB, LAB_OTHER)
    assert ari_other < RESTART_SAME_ARI
    assert (s["n_restarts"], s["n_converged"], s["n_rejected_update"], s["n_max_iter"], s["n_collapsed_converged"]) == (6, 4, 1, 1, 1)
    assert s["n_within_epsilon"] == 2 and RESTART_EPSILON_NATS == 1.0                       # the selected one and the one at -100.5
    assert s["ari_to_selected"] == pytest.approx([1.0, ari_other]) and s["ari_min"] == pytest.approx(ari_other) and s["ari_median"] == pytest.approx((1.0 + ari_other) / 2)
    assert s["n_other_same_solution"] == 1 and s["gap_to_next_distinct"] == pytest.approx(3.0)


def test_adjusted_rand_index_known_values_and_against_scikit_learn():
    assert adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(-0.5)
    assert adjusted_rand_index([0, 0, 0], [5, 5, 5]) == 1.0 and adjusted_rand_index([3], [4]) == 1.0
    with pytest.raises(ValueError):
        adjusted_rand_index([0, 1], [0, 1, 2])
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    for _ in range(30):
        a, b = rng.integers(0, rng.integers(2, 5), 40), rng.integers(0, rng.integers(2, 5), 40)
        assert adjusted_rand_index(a, b) == pytest.approx(sk.adjusted_rand_score(a, b), abs=1e-12)


# ------------------------------------------------------------------------------------------------ 4.9 K-level demotion
class _KModel:
    def __init__(self, bic, collapsed, ll=-100.0):
        self._bic, self.log_likelihood_, self.convergence_rate_ = bic, ll, 1.0
        self.collapse_report_ = {"collapsed": collapsed}
        self.converged_ = True

    def bic(self, n_observations=None):
        return self._bic


def _select(monkeypatch, models):
    def fake(X, n_regimes, n_features, **k):
        return models[n_regimes], []
    monkeypatch.setattr(fitting, "fit_with_random_restarts", fake)
    return select_k_by_bic(np.zeros((40, 3)), n_features=3, k_candidates=tuple(models), bypass_init_gate=True)


def test_a_collapsed_candidate_k_is_demoted_even_with_the_lowest_bic(monkeypatch):
    m, rep = _select(monkeypatch, {2: _KModel(300.0, False), 3: _KModel(250.0, True), 4: _KModel(280.0, False)})
    assert rep["selected_k"] == 4 and m is not None                                          # K = 3 has the lowest BIC but is collapsed: next best healthy K
    m, rep = _select(monkeypatch, {2: _KModel(300.0, False), 3: _KModel(250.0, False)})
    assert rep["selected_k"] == 3                                                              # negative control: not collapsed, lowest BIC wins


def test_when_every_candidate_k_is_collapsed_the_lowest_bic_wins(monkeypatch):
    m, rep = _select(monkeypatch, {2: _KModel(300.0, True), 3: _KModel(250.0, True)})
    assert rep["selected_k"] == 3 and m.collapse_report_["collapsed"] is True


# ------------------------------------------------------------------------------------------------ 4.5 P0-07
def _result(state, **kw):
    return HSSMResult(model=None, fit_state=state, **kw)


@pytest.mark.parametrize("state", [FitState.MAX_ITER_REACHED, FitState.REJECTED_UPDATE, FitState.FAILED, FitState.RUNNING])
def test_a_fit_that_did_not_converge_is_withheld(state):
    r = _result(state)
    assert r.insights_withheld and r.publishable_outputs() is None and r.withheld_reason.startswith("fit_state_not_converged") and state.value in r.withheld_reason


def test_a_converged_fit_is_not_withheld_and_numerical_failure_keeps_its_own_reason():
    r = _result(FitState.CONVERGED)
    assert r.withheld_reason is None and not r.insights_withheld
    n = _result(FitState.NUMERICAL_FAILURE)
    assert n.withheld_reason == "fit_state is NUMERICAL_FAILURE"


def test_model_selection_contested_is_judged_by_the_fits_own_state():
    assert _result(FitState.MODEL_SELECTION_CONTESTED).withheld_reason is None                                              # no metadata: counts as converged
    assert _result(FitState.MODEL_SELECTION_CONTESTED, convergence_metadata={"fit_state": FitState.CONVERGED}).withheld_reason is None
    w = _result(FitState.MODEL_SELECTION_CONTESTED, convergence_metadata={"fit_state": FitState.MAX_ITER_REACHED})
    assert w.insights_withheld and "MAX_ITER_REACHED" in w.withheld_reason


def test_a_collapsed_selected_fit_is_withheld_with_its_reason_and_reasons_accumulate():
    r = _result(FitState.CONVERGED, fit_collapsed=True, collapse_report={"min_effective_days": 2.5, "min_effective_days_threshold": 4.0})
    assert r.withheld_reason.startswith("collapsed_regime") and "2.50" in r.withheld_reason
    both = _result(FitState.MAX_ITER_REACHED, fit_collapsed=True, fit_degraded=True, fit_cholesky_fallbacks=3)
    assert all(t in both.withheld_reason for t in ("fit_state_not_converged", "collapsed_regime", "fit_degraded"))


def test_dual_selection_prefers_a_converged_restart_to_a_higher_likelihood_one_that_is_not(monkeypatch):
    from backbone.hssm.dual_model import DualTimescaleHSSM
    plan = iter([(M, -100.0), (C, -110.0), (M, -95.0)])

    def fit(self, X, n_iter=50, tol=1e-4, verbose=False, timestamps=None):
        self.fit_state_, self.log_likelihood_ = next(plan)
        self.n_iter_ = 3
        self.fit_degraded_ = False
        return self

    monkeypatch.setattr(DualTimescaleHSSM, "fit", fit)
    model, log = fit_dual_with_random_restarts(np.zeros((20, 3)), 2, 3, latent_dim=1, n_init=3, n_iter=3, bypass_init_gate=True)
    assert model.log_likelihood_ == -110.0 and model.fit_state_ == FitState.CONVERGED
    plan = iter([(M, -100.0), (R, -110.0), (M, -95.0)])                                                                    # negative control: none converged
    monkeypatch.setattr(DualTimescaleHSSM, "fit", fit)
    model, _ = fit_dual_with_random_restarts(np.zeros((20, 3)), 2, 3, latent_dim=1, n_init=3, n_iter=3, bypass_init_gate=True)
    assert model.log_likelihood_ == -95.0


# ------------------------------------------------------------------------------------------------ end to end and documents
def test_fit_hssm_baseline_carries_the_new_metadata_and_stays_publishable():
    X = np.random.default_rng(3).normal(size=(60, 3))
    X[:20] += 3.0
    X[40:] -= 3.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = fitting.fit_hssm(matrix=X, candidate_ks=(2,), n_initializations=3, max_iter=100, allow_fast_test_fit=True, max_duration=25, random_seed=5)
    assert r.fit_state == FitState.CONVERGED and not r.fit_collapsed and r.insights_withheld is False
    assert r.restart_summary["n_restarts"] == 3 and r.collapse_report["min_effective_days"] > 4 and isinstance(r.guard_repairs, int)
    assert set(r.convergence_metadata) >= {"restart_summary", "collapse", "guard_repairs"}


def test_the_contract_and_schema_document_the_new_rules():
    contract = (ROOT / "docs" / "INTEGRATION_CONTRACT_HSSM.md").read_text()
    schema = (ROOT / "docs" / "backbone" / "HSSM_RESULT_SCHEMA.md").read_text()
    for text in (contract, schema):
        for needle in ("fit_state_not_converged", "collapsed_regime", "guard_repairs", "restart_summary", "fit_collapsed"):
            assert needle in text, needle
    assert "COLLAPSE_MIN_EFFECTIVE_DAYS" in contract and "4 effective days" in contract
