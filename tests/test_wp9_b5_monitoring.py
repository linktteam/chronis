"""WP9 B5: predictability monitor and confidence display gate (design: tests/PREDECLARED_CRITERIA_WP9.md section 24). Mutation check: integration_report/wp9/scripts/mutation_b5.py."""
from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from backbone.hssm import alerts
from backbone.hssm import config as C
from backbone.hssm import monitoring as M
from backbone.hssm.model import GaussianHSMM


def series(n=120, drop_at=None, drop=2.0, sd=1.0, seed=0, level=-3.0):
    s = np.random.default_rng(seed).normal(level, sd, n)
    if drop_at is not None:
        s[drop_at:] -= drop
    return s


# ------------------------------------------------------------------------------------------------------------------- the monitor
def test_the_declared_parameters_are_in_the_config():
    assert (C.MONITOR_WINDOW_DAYS, C.MONITOR_REFERENCE_DAYS, C.MONITOR_DROP_Z, C.MONITOR_MIN_DROP_SD, C.MONITOR_COOLDOWN_DAYS) == (14, 28, 3.5, 0.75, 14)
    assert C.N17_PASSED is False and C.CONFIDENCE_DISPLAY_THRESHOLD == 0.85
    assert M.predictability_monitor(series())["parameters"] == dict(window=14, reference_days=28, drop_z=3.5, min_drop_sd=0.75, cooldown_days=14)


def test_a_stable_series_raises_nothing_and_a_clear_drop_raises_one_event_near_the_change():
    assert M.predictability_monitor(series(200, seed=1))["events"] == []
    r = M.predictability_monitor(series(120, drop_at=70, drop=2.0, seed=2))
    assert r["status"] == "ok" and len(r["events"]) == 1
    e = r["events"][0]
    assert 70 <= e["day"] <= 95 and e["drop_sd"] >= 0.75 and e["z"] >= 3.5 and e["reference_mean"] > e["recent_mean"]


def test_a_rise_is_not_a_drop():
    assert M.predictability_monitor(series(120, drop_at=70, drop=-2.0, seed=3))["events"] == []


def test_the_practical_size_and_the_statistical_size_are_both_needed():
    # a sure but tiny drop: 0.2 against a reference sd of 0.01 is 20 sd, so make the reference noisier than the drop is large
    base = np.random.default_rng(0).normal(0, 1.0, 120)
    tiny = base + np.r_[np.zeros(70), np.full(50, -0.6)]                                                       # 0.6 sd: statistically clear on a long window, practically small (< 0.75 sd)
    assert M.predictability_monitor(tiny, drop_z=0.0)["events"] == [] or all(e["drop_sd"] >= 0.75 for e in M.predictability_monitor(tiny, drop_z=0.0)["events"])
    assert M.predictability_monitor(tiny, drop_z=0.0, min_drop_sd=0.0)["events"] != []
    noisy = series(120, drop_at=70, drop=0.8, sd=6.0, seed=5)                                                   # a drop of 0.13 sd buried in noise
    assert M.predictability_monitor(noisy)["events"] == []
    big_but_short = np.r_[np.random.default_rng(3).normal(0, 1, 100), [-1.5, 1.0, -2.5, 0.5, -3.0, 1.0, -2.0, 0.5, -2.5, 1.0, -1.0, 0.0, -2.0, 0.5]]
    assert [e for e in M.predictability_monitor(big_but_short, drop_z=50.0)["events"]] == []                   # the statistical size is a hard condition too


def test_a_constant_reference_gives_no_event_and_autocorrelation_makes_the_test_stricter():
    assert M.predictability_monitor(np.r_[np.full(42, -3.0), np.full(14, -9.0)])["events"] == []                # the only evaluation has a constant reference (sd 0): nothing to measure a drop against
    rng = np.random.default_rng(7)
    e = rng.normal(size=300)
    ar = np.zeros(300)
    for t in range(1, 300):
        ar[t] = 0.7 * ar[t - 1] + np.sqrt(1 - 0.49) * e[t]
    iid = rng.normal(size=300)
    shifted = lambda x: np.r_[x[:70], x[70:100] - 2.0]
    n_ar = len(M.predictability_monitor(shifted(ar), cooldown_days=1000)["events"])
    n_iid = len(M.predictability_monitor(shifted(iid), cooldown_days=1000)["events"])
    assert n_iid >= n_ar


def test_the_z_level_is_inclusive_and_the_event_day_is_the_original_index_after_leading_gaps():
    s = series(120, drop_at=70, drop=2.0, seed=2)
    ev = M.predictability_monitor(s)["events"][0]
    assert M.predictability_monitor(s, drop_z=ev["z"])["events"][0]["day"] == ev["day"]                      # z equal to the level is an event
    assert M.predictability_monitor(s, drop_z=ev["z"] + 1e-9)["events"] == [] or M.predictability_monitor(s, drop_z=ev["z"] + 1e-9)["events"][0]["day"] > ev["day"]
    gap = np.r_[np.full(10, np.nan), s]                                                                       # ten leading days without a score: the same events, ten days later
    assert [e["day"] for e in M.predictability_monitor(gap)["events"]] == [e["day"] + 10 for e in M.predictability_monitor(s)["events"]]


def test_one_event_per_episode_and_a_new_one_after_the_cooldown():
    s = np.r_[np.full(50, -3.0), np.full(30, -6.0), np.full(40, -3.0), np.full(30, -6.0)] + np.random.default_rng(1).normal(0, 0.3, 150)
    days = [e["day"] for e in M.predictability_monitor(s)["events"]]
    assert len(days) >= 2 and days[0] < 70 and days[-1] >= 120 and all(b - a >= 14 for a, b in zip(days, days[1:]))
    assert len(M.predictability_monitor(s, cooldown_days=1000)["events"]) == 1


def test_not_enough_scores_gives_no_event_and_says_so():
    r = M.predictability_monitor(series(41, drop_at=20))
    assert r["status"] == "not_enough_scores" and r["events"] == [] and r["n_scores"] == 41
    assert M.predictability_monitor(series(42, drop_at=20))["status"] == "ok"


def test_non_finite_days_are_dropped_and_the_event_day_is_the_original_index():
    s = series(130, drop_at=80, drop=2.5, seed=4)
    s[[5, 6, 90]] = np.nan
    s[10] = np.inf
    r = M.predictability_monitor(s)
    assert r["n_dropped"] == 4 and r["n_scores"] == 126 and r["events"] and 80 <= r["events"][0]["day"] < 130
    assert M.predictability_monitor(np.full(100, np.nan))["status"] == "not_enough_scores"


@pytest.mark.parametrize("bad", [dict(window=0), dict(window=2), dict(reference_days=2), dict(window=14.5), dict(cooldown_days=0), dict(drop_z=-1.0), dict(min_drop_sd=float("nan")), dict(drop_z=True), dict(window=True)])
def test_invalid_parameters_are_reported_not_raised(bad):
    assert M.predictability_monitor(series(), **bad)["status"] == "invalid_input"


@pytest.mark.parametrize("garbage", [None, "abc", [[1, 2], [3, 4]], object(), [], 5])
def test_garbage_input_never_raises(garbage):
    r = M.predictability_monitor(garbage)
    assert r["status"] in ("invalid_input", "not_enough_scores") and r["events"] == []


def test_the_alert_carries_numbers_and_a_hashed_user_id_only(caplog):
    got = []
    alerts.clear_handlers(); alerts.register_handler(got.append)
    try:
        with caplog.at_level(logging.WARNING, logger="chronis.hssm.alerts"):
            sent = M.alert_on_drop("person-17", M.predictability_monitor(series(120, drop_at=70, drop=2.5, seed=2)))
    finally:
        alerts.clear_handlers()
    assert len(sent) == 1 and got == sent and sent[0]["event"] == "predictability_drop" and sent[0]["user_id_hash"] == alerts.user_id_hash("person-17")
    assert "person-17" not in json.dumps(sent) and "person-17" not in caplog.text
    assert M.alert_on_drop("u", M.predictability_monitor(series(200, seed=1))) == []


# ------------------------------------------------------------------------------------------------------------------- the gate
GRID = [0.0, 0.5, 0.849999, 0.85, 0.9, 0.99, 1.0, 1, float("nan"), -0.1, 1.5, None, "0.9", True]


@pytest.mark.parametrize("c", GRID)
def test_while_n17_is_not_passed_the_gate_always_says_early_estimate(c):
    for evidence in ("calibration on 960 fits", "", None):
        out = M.confidence_display_gate(c, evidence)
        assert out["display"] == "early_estimate" and out["percentage"] is None and "n17_not_passed" in out["reasons"]


def test_the_gate_shows_a_percentage_only_when_everything_holds():
    ev = "WP9 calibration, baseline fits, 960 fits"
    ok = M.confidence_display_gate(0.91, ev, n17_passed=True)
    assert ok["display"] == "percentage" and ok["percentage"] == 91 and ok["reasons"] == []
    assert M.confidence_display_gate(0.85, ev, n17_passed=True)["display"] == "percentage"                                   # the threshold is inclusive
    assert M.confidence_display_gate(0.849999, ev, n17_passed=True) == dict(display="early_estimate", percentage=None, reasons=["confidence_below_threshold"])
    assert M.confidence_display_gate(0.9, "", n17_passed=True)["reasons"] == ["no_calibration_evidence_for_this_class_of_fit"]
    assert M.confidence_display_gate(0.9, None, n17_passed=True)["display"] == "early_estimate"
    assert M.confidence_display_gate(0.9, "   ", n17_passed=True)["display"] == "early_estimate"
    for bad in (float("nan"), float("inf"), -0.2, 1.01, None, "0.9", True):
        assert M.confidence_display_gate(bad, ev, n17_passed=True)["display"] == "early_estimate"


def test_the_percentage_is_never_rounded_up():
    assert M.confidence_display_gate(0.8599, "e", n17_passed=True)["percentage"] == 85
    assert M.confidence_display_gate(0.999, "e", n17_passed=True)["percentage"] == 99
    assert M.confidence_display_gate(1.0, "e", n17_passed=True)["percentage"] == 100


def test_only_a_true_n17_flag_opens_the_gate_and_the_threshold_cannot_go_below_the_owner_floor():
    for not_true in (False, 1, "yes", None):
        assert M.confidence_display_gate(0.99, "e", n17_passed=not_true)["display"] == "early_estimate"
    low = M.confidence_display_gate(0.5, "e", n17_passed=True, threshold=0.4)
    assert low["display"] == "early_estimate" and "threshold_below_the_owner_floor" in low["reasons"]
    assert M.confidence_display_gate(0.95, "e", n17_passed=True, threshold=0.9)["display"] == "percentage"
    assert M.confidence_display_gate(0.88, "e", n17_passed=True, threshold=0.9)["display"] == "early_estimate"


# ------------------------------------------------------------------------------------------------------------------- the held-out scores
def _fitted(seed=0):
    rng = np.random.default_rng(seed)
    X = np.concatenate([rng.normal(m, 0.5, (15, 2)) for m in (0.0, 3.0, 0.0, 3.0, 0.0)])
    m = GaussianHSMM(K=2, n_features=2, Dmax=20, seed=seed)
    m.fit(X, n_iter=30)
    return m, X


def test_the_increments_sum_to_the_log_likelihood_and_the_model_is_untouched():
    m, X = _fitted()
    before = (m.A.copy(), m.pi.copy(), m.mu.copy(), m.var.copy())
    m._forward_backward(X)
    ll = m.log_likelihood_
    inc = M.baseline_predictive_increments(m, X)
    assert inc.shape == (len(X),) and np.isfinite(inc).all() and inc.sum() == pytest.approx(ll, rel=1e-9, abs=1e-9)
    assert m.log_likelihood_ == ll and all(np.array_equal(a, b) for a, b in zip(before, (m.A, m.pi, m.mu, m.var)))
    # a prefix gives the same increments (causality: the score of a day does not use later days)
    assert np.allclose(M.baseline_predictive_increments(m, X[:40]), inc[:40])


def test_the_initial_distribution_of_the_model_is_used_for_the_first_day():
    import copy
    m, X = _fitted(2)
    m = copy.deepcopy(m)
    m.pi = np.array([0.3, 0.7])                                                                              # the fitted pi is one-hot; a spread one makes the first day depend on it
    m._forward_backward(X[:1])
    inc = M.baseline_predictive_increments(m, X[:1])
    assert inc[0] == pytest.approx(m.log_likelihood_, abs=1e-9)
    m.pi = np.array([0.7, 0.3])
    m._forward_backward(X[:1])
    assert M.baseline_predictive_increments(m, X[:1])[0] == pytest.approx(m.log_likelihood_, abs=1e-9)


def test_a_day_the_model_does_not_expect_scores_lower_and_a_missing_day_scores_zero():
    m, X = _fitted(1)
    base = M.baseline_predictive_increments(m, X)
    far = X.copy(); far[50:] += 8.0
    assert M.baseline_predictive_increments(m, far)[50:].mean() < base[50:].mean() - 5.0
    miss = X.copy(); miss[30] = np.nan
    assert M.baseline_predictive_increments(m, miss)[30] == pytest.approx(0.0, abs=1e-9)
