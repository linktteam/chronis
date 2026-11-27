"""WP10B B3: the dual predictive certificate (`integration_report/wp10b/scripts/dual_cert_lib.py`) on stub models with known increments: eligibility, the rolling origins, the gain margin, the stability gate, missing days, the baseline comparison, fail-closed paths."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "integration_report" / "wp10b" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SC))
_spec = importlib.util.spec_from_file_location("wp10b_dual_cert_under_test", SC / "dual_cert_lib.py")
D = importlib.util.module_from_spec(_spec)
sys.modules["wp10b_dual_cert_under_test"] = D
_spec.loader.exec_module(D)
from backbone.hssm import cold_start_cert as C  # noqa: E402


def data(T=100, F=3, seed=0, nan_rows=()):
    X = np.random.default_rng(seed).standard_normal((T, F))
    for r in nan_rows:
        X[r] = np.nan
    return X


class Stub:
    """predictive_increments(X) = the single-Gaussian-like increments of the stub's own `inc` function (length len(X)); means fixed."""
    def __init__(self, inc, means=None, degraded=False):
        self.inc, self.means, self.degraded = inc, np.zeros((2, 3)) if means is None else means, degraded

    def predictive_increments(self, X):
        return np.asarray(self.inc(np.asarray(X)), dtype=float)

    def regime_observation_means(self):
        return self.means


def const(c):
    return lambda X: np.full(len(X), float(c))


def gauss_plus(delta, X_all):
    """increments that exceed the Gaussian fitted on the rows before the origin by exactly `delta`: needs the origin, so computed lazily per call from the length of X."""
    def inc(X):
        e = len(X)
        out = np.zeros(e)
        for o, ee in C.origin_blocks(len(X_all))[0]:
            if ee == e:
                obs = ~np.isnan(X_all[:o]).all(axis=1)
                g = C.gaussian_heldout(X_all[:o][obs], X_all[o:e][~np.isnan(X_all[o:e]).all(axis=1)])
                tmp = np.zeros(e - o)
                tmp[~np.isnan(X_all[o:e]).all(axis=1)] = g + delta
                out[o:e] = tmp
        return out
    return inc


EL = dict(em_state="MAX_ITER_REACHED", other_reasons=[])


def run(X, delta=1.0, means_last=None, el=EL, baseline=None, **kw):
    full = Stub(const(0.0))
    refit = lambda Xt: Stub(gauss_plus(delta, X), means=means_last)         # noqa: E731
    return D.certify_dual(X, full, refit, baseline, eligibility=el, **kw)


def test_a_clear_and_stable_gain_is_certified_and_uses_the_wp9_constants():
    r = run(data())
    assert r["state"] == C.STATE_CERTIFIED and r["reasons"] == [] and r["z"] == 3.0 and r["tau"] == 0.25 and r["mean_gain"] == pytest.approx(1.0) and r["blocks"] == [[50, 65], [65, 80], [80, 100]]
    assert r["n_increments"] == 50 and r["baseline"] is None


def test_a_loss_against_the_single_gaussian_is_not_certified():
    r = run(data(), delta=-0.5)
    assert r["state"] == C.STATE_NOT_CERTIFIED and r["reasons"] == [C.R_GAIN]


def test_the_margin_is_three_standard_errors():
    X = data()
    noise = np.random.default_rng(1).standard_normal(len(X)) * 1.0

    def noisy(delta):
        inc0 = gauss_plus(delta, X)
        return lambda Z: inc0(Z) + noise[:len(Z)]
    blocks = C.origin_blocks(len(X))[0]

    def gains(delta):
        return np.concatenate([noisy(delta)(X[:e])[o:e] - C.gaussian_heldout(X[:o], X[o:e]) for o, e in blocks])
    g0 = gains(0.0)
    delta = -g0.mean() + 2.0 * C.hac_se(g0)                           # a shift moves the mean and leaves the standard error alone: the mean now sits at two standard errors
    d = gains(delta)
    mean, se = d.mean(), C.hac_se(d)
    assert mean - 1.0 * se > 0 > mean - 3.0 * se                      # between the one-se and the three-se level
    r = D.certify_dual(X, Stub(const(0.0)), lambda Xt: Stub(noisy(delta)), eligibility=EL)
    assert r["state"] == C.STATE_NOT_CERTIFIED and r["reasons"] == [C.R_GAIN] and r["mean_gain"] == pytest.approx(mean) and r["se"] == pytest.approx(se)
    assert D.certify_dual(X, Stub(const(0.0)), lambda Xt: Stub(noisy(delta)), eligibility=EL, z=1.0)["state"] == C.STATE_CERTIFIED


def test_unstable_regime_means_refuse_even_a_clear_gain():
    r = run(data(), means_last=np.full((2, 3), 5.0))
    assert r["state"] == C.STATE_NOT_CERTIFIED and r["reasons"] == [C.R_UNSTABLE] and r["stability"] > 0.25


def test_eligibility_converged_is_not_applicable_unless_a_control_asks_for_any_state():
    X = data()
    na = run(X, el=dict(em_state="CONVERGED", other_reasons=[]))
    assert na["state"] == C.STATE_NOT_APPLICABLE and na["reasons"] == [C.R_CONVERGED]
    assert run(X, el=dict(em_state="CONVERGED", other_reasons=["x"], apply_to_any_state=True))["reasons"] == [C.R_OTHER]          # a control skips the EM-state test, never the other withhold reasons
    assert run(X, el=dict(em_state="CONVERGED", other_reasons=[], apply_to_any_state=True))["state"] == C.STATE_CERTIFIED
    assert run(X, el=dict(em_state="REJECTED_UPDATE", other_reasons=[]))["state"] == C.STATE_CERTIFIED
    r = run(X, el=dict(em_state="NUMERICAL_FAILURE", other_reasons=[]))
    assert r["state"] == C.STATE_NOT_CERTIFIED and r["reasons"] == [C.R_STATE]


def test_any_other_withhold_reason_refuses_without_refitting():
    calls = []
    X = data()
    r = D.certify_dual(X, Stub(const(0.0)), lambda Xt: calls.append(1) or Stub(const(0.0)), eligibility=dict(em_state="MAX_ITER_REACHED", other_reasons=["collapsed_regime: x"]))
    assert r["state"] == C.STATE_NOT_CERTIFIED and r["reasons"] == [C.R_OTHER] and r["other_reasons"] == ["collapsed_regime: x"] and calls == []


def test_too_few_rows_and_infinite_values_are_refused():
    assert run(data(T=20))["reasons"] == [C.R_TOO_FEW_ROWS]
    X = data(); X[3, 1] = np.inf
    assert run(X)["reasons"] == [C.R_NONFINITE]


def test_a_failed_or_degraded_refit_is_a_refusal():
    X = data()
    def boom(Xt):
        raise RuntimeError("no")
    assert D.certify_dual(X, Stub(const(0.0)), boom, eligibility=EL)["reasons"] == [C.R_REFIT]
    r = D.certify_dual(X, Stub(const(0.0)), lambda Xt: Stub(gauss_plus(1.0, X), degraded=True), eligibility=EL)
    assert r["reasons"] == [C.R_DEGRADED_REFIT]
    assert D.certify_dual(X, Stub(const(0.0)), lambda Xt: Stub(gauss_plus(1.0, X)), lambda Xt: boom(Xt), eligibility=EL)["reasons"] == [C.R_REFIT]


def test_an_exception_in_scoring_is_a_refusal_not_a_crash():
    X = data()
    class Bad(Stub):
        def predictive_increments(self, X):
            raise ValueError("nope")
    r = D.certify_dual(X, Stub(const(0.0)), lambda Xt: Bad(const(0.0)), eligibility=EL)
    assert r["state"] == C.STATE_NOT_CERTIFIED and r["reasons"] == [C.R_ERROR] and "ValueError" in r["error"]


def test_all_missing_days_are_left_out_of_every_score():
    X = data(nan_rows=(55, 70, 90, 3))
    r = run(X)
    assert r["state"] == C.STATE_CERTIFIED and r["n_increments"] == 50 - 3             # rows 55, 70, 90 fall in the held-out blocks; row 3 only in the training rows
    assert r["mean_gain"] == pytest.approx(1.0)


def test_the_baseline_comparison_is_reported_next_to_the_gaussian_gain():
    X = data()
    ahead = lambda Xt: Stub(lambda Z: gauss_plus(1.0, X)(Z) - 0.3)               # noqa: E731  baseline is 0.3 below the dual
    r = D.certify_dual(X, Stub(const(0.0)), lambda Xt: Stub(gauss_plus(1.0, X)), ahead, eligibility=EL)
    b = r["baseline"]
    assert r["state"] == C.STATE_CERTIFIED and b["n_increments"] == 50 and b["mean_gain"] == pytest.approx(0.3) and b["beats_baseline"] is True and b["lower"] == pytest.approx(0.3)
    behind = lambda Xt: Stub(lambda Z: gauss_plus(1.0, X)(Z) + 0.3)              # noqa: E731
    rb = D.certify_dual(X, Stub(const(0.0)), lambda Xt: Stub(gauss_plus(1.0, X)), behind, eligibility=EL)
    assert rb["state"] == C.STATE_CERTIFIED and rb["baseline"]["mean_gain"] == pytest.approx(-0.3) and rb["baseline"]["beats_baseline"] is False


def test_the_baseline_does_not_decide_the_certificate():
    X = data()
    behind = lambda Xt: Stub(lambda Z: gauss_plus(1.0, X)(Z) + 5.0)             # noqa: E731
    assert D.certify_dual(X, Stub(const(0.0)), lambda Xt: Stub(gauss_plus(1.0, X)), behind, eligibility=EL)["state"] == C.STATE_CERTIFIED


def test_a_non_finite_baseline_difference_is_reported_as_not_beating_it():
    X = data()
    nan_b = lambda Xt: Stub(const(float("nan")))                                 # noqa: E731
    b = D.certify_dual(X, Stub(const(0.0)), lambda Xt: Stub(gauss_plus(1.0, X)), nan_b, eligibility=EL)["baseline"]
    assert b["beats_baseline"] is False and b["mean_gain"] is None


def test_the_refit_sees_only_the_rows_before_each_origin():
    X = data()
    seen = []
    def refit(Xt):
        seen.append(len(Xt))
        return Stub(gauss_plus(1.0, X))
    D.certify_dual(X, Stub(const(0.0)), refit, eligibility=EL)
    assert seen == [50, 65, 80]


def test_the_observed_row_mask():
    X = data(T=5); X[2] = np.nan; X[3, 0] = np.nan
    assert D.observed(X).tolist() == [True, True, False, True, True]


def test_the_baseline_refit_sees_only_the_rows_before_each_origin():
    X = data()
    seen = []
    def bref(Xt):
        seen.append(len(Xt))
        return Stub(gauss_plus(1.0, X))
    D.certify_dual(X, Stub(const(0.0)), lambda Xt: Stub(gauss_plus(1.0, X)), bref, eligibility=EL)
    assert seen == [50, 65, 80]


def test_the_baseline_lower_bound_uses_the_same_margin_and_the_beats_flag_follows_it():
    X = data()
    noise = np.random.default_rng(3).standard_normal(len(X))
    blocks = C.origin_blocks(len(X))[0]
    base = lambda Xt: Stub(lambda Z: gauss_plus(1.0, X)(Z) - noise[:len(Z)])          # noqa: E731  the baseline is the dual minus noise
    diffs = np.concatenate([noise[o:e] for o, e in blocks])
    shift = 2.0 * C.hac_se(diffs) - diffs.mean()
    shifted = lambda Xt: Stub(lambda Z: gauss_plus(1.0, X)(Z) - noise[:len(Z)] - shift)        # noqa: E731  the mean difference now sits at two standard errors
    r = D.certify_dual(X, Stub(const(0.0)), lambda Xt: Stub(gauss_plus(1.0, X)), shifted, eligibility=EL)
    b = r["baseline"]
    mean, se = (diffs + shift).mean(), C.hac_se(diffs)
    assert b["mean_gain"] == pytest.approx(mean) and b["se"] == pytest.approx(se) and b["lower"] == pytest.approx(mean - 3.0 * se) and mean > 0 > b["lower"] and b["beats_baseline"] is False
    assert D.certify_dual(X, Stub(const(0.0)), lambda Xt: Stub(gauss_plus(1.0, X)), shifted, eligibility=EL, z=1.0)["baseline"]["beats_baseline"] is True
    assert base is not None


def test_stability_is_measured_in_units_of_the_feature_spread():
    means = np.zeros((2, 3)); means[0, :] = 0.5                       # a shift of half a unit in every feature of one regime (root mean square 0.35 in units of spread 1)
    narrow = run(data(), means_last=means)                              # features with spread about 1: 0.5 is far above the 0.25 limit
    assert narrow["state"] == C.STATE_NOT_CERTIFIED and narrow["reasons"] == [C.R_UNSTABLE]
    wide = run(data() * 10.0, means_last=means)                         # the same shift in features with spread about 10 is 0.05
    assert wide["state"] == C.STATE_CERTIFIED and wide["stability"] < 0.25
