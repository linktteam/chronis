"""WP5: the N18-v2 decision rules (tests/utils/n18v2.py) on synthetic numbers. No fitting of study data, no reserved seed.
A wrong rule here would silently flip a verdict, so every rule has a boundary case on both sides (mutation set `n18v2` in integration_report/wp5/scripts/mutation_wp5.py).
"""
import math

import numpy as np
import pytest

from tests.utils import n18v2 as N
from tests.utils.heldout_prediction import predictive_increments

SEEDS = list(range(1, 25))
LENGTHS = N.LENGTHS


def diffs(mean, delta, n=24):
    """n paired differences alternating mean +/- delta: se = delta / sqrt(n - 1) exactly (ddof 1, n even)."""
    return np.array([mean + (delta if i % 2 == 0 else -delta) for i in range(n)])


def se_of(delta, n=24):
    return delta / math.sqrt(n - 1)


def arm_scores(base, d):
    return {s: base + float(x) for s, x in zip(SEEDS, d)}


def build(c1=(1.0, 0.1), c2p=(0.0, 0.1), c2s=(0.0, 0.1), c3=(0.0, 0.1), c1_at=None, c2p_at=None, c2s_at=None, c3_at=None, atyp_lengths=LENGTHS):
    """eval_scores and atyp_scores dictionaries. Each tuple is (mean, delta) of arm - opponent; *_at maps length -> tuple to override one length."""
    ev = {"gauss": {}, "personal": {}, "shared": {}, "shrinkage": {}}
    at = {"personal": {}, "shrinkage": {}}
    for L_ in LENGTHS:
        g = {s: -3.0 for s in SEEDS}
        p = {s: -2.0 + 0.01 * s for s in SEEDS}
        sh = {s: -1.5 - 0.01 * s for s in SEEDS}
        ev["gauss"][L_], ev["personal"][L_], ev["shared"][L_] = g, p, sh
        m1, d1 = (c1_at or {}).get(L_, c1)
        m2p, d2p = (c2p_at or {}).get(L_, c2p)
        m2s, d2s = (c2s_at or {}).get(L_, c2s)
        # the arm is built so that arm - gauss = d1 exactly, then the other two differences are imposed by overriding the opponents
        arm = {s: g[s] + float(x) for s, x in zip(SEEDS, diffs(m1, d1))}
        ev["shrinkage"][L_] = arm
        ev["personal"][L_] = {s: arm[s] - float(x) for s, x in zip(SEEDS, diffs(m2p, d2p))}
        ev["shared"][L_] = {s: arm[s] - float(x) for s, x in zip(SEEDS, diffs(m2s, d2s))}
        if L_ in atyp_lengths:
            m3, d3 = (c3_at or {}).get(L_, c3)
            ap = {s: -2.5 for s in SEEDS[:12]}
            ash = {s: ap[s] + float(x) for s, x in zip(SEEDS[:12], diffs(m3, d3, n=12))}
            at["personal"][L_], at["shrinkage"][L_] = ap, ash
    return ev, at


# ------------------------------------------------------------------ paired_stats
def test_paired_stats_numbers_and_the_three_margins():
    a = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
    b = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    st = N.paired_stats(a, b)
    se = np.std([1, 2, 3, 4], ddof=1) / 2.0
    assert st["n"] == 4 and st["n_dropped"] == 0 and st["n_favour"] == 4
    assert st["mean"] == pytest.approx(2.5) and st["se"] == pytest.approx(se)
    assert st["mean_minus_3se"] == pytest.approx(2.5 - 3 * se) and st["mean_plus_1se"] == pytest.approx(2.5 + se) and st["mean_plus_3se"] == pytest.approx(2.5 + 3 * se)
    assert N.paired_stats(b, a)["mean"] == pytest.approx(-2.5) and N.paired_stats(b, a)["n_favour"] == 0


def test_paired_stats_counts_only_strictly_positive_differences_as_favouring():
    st = N.paired_stats({1: 1.0, 2: 1.0, 3: 2.0}, {1: 1.0, 2: 0.0, 3: 0.0})                     # differences 0, 1, 2
    assert st["n_favour"] == 2 and st["n"] == 3


def test_paired_stats_drops_and_counts_non_finite_pairs_and_ignores_unshared_seeds():
    a = {1: 1.0, 2: float("nan"), 3: 3.0, 4: 4.0, 9: 100.0}
    b = {1: 0.0, 2: 0.0, 3: float("inf"), 4: 0.0, 8: -100.0}
    st = N.paired_stats(a, b)
    assert st["n"] == 2 and st["n_dropped"] == 2 and st["mean"] == pytest.approx(2.5)               # seeds 1 and 4 remain; 2 and 3 are non-finite; 8 and 9 are unshared
    few = N.paired_stats({1: 1.0}, {1: 0.0})
    assert few["n"] == 1 and math.isnan(few["mean"]) and math.isnan(few["mean_minus_3se"]) and few["n_favour"] == 0


# ------------------------------------------------------------------ choose_n0
def test_choose_n0_maximises_the_minimum_over_lengths():
    lowers = {5.0: {30: -1.0, 60: 0.5, 120: 0.9}, 20.0: {30: -0.2, 60: 0.1, 120: 0.2}, 80.0: {30: 0.3, 60: 0.2, 120: -0.4}, 320.0: {30: -0.5, 60: -0.5, 120: -0.5}}
    assert N.choose_n0(lowers) == 20.0                                                          # minima: -1.0, -0.2, -0.4, -0.5; max of the max (0.9) would pick 5


def test_choose_n0_ties_go_to_the_smaller_n0_and_a_larger_n0_must_be_strictly_better():
    lowers = {5.0: {30: 0.1, 60: 0.3}, 20.0: {30: 0.1, 60: 0.9}, 80.0: {30: 0.1000001, 60: 0.2}}
    assert N.choose_n0(lowers) == 80.0
    lowers[80.0][30] = 0.1
    assert N.choose_n0(lowers) == 5.0                                                           # all minima equal 0.1: the smallest n0
    lowers = {320.0: {30: 0.1, 60: 0.2}, 5.0: {30: 0.1, 60: 0.3}}                                # dictionary order must not matter
    assert N.choose_n0(lowers) == 5.0


def test_choose_n0_skips_non_finite_candidates_and_raises_when_none_is_usable():
    assert N.choose_n0({5.0: {30: float("nan"), 60: 5.0}, 20.0: {30: -1.0, 60: -1.0}}) == 20.0
    assert N.choose_n0({5.0: {30: 5.0, 60: float("nan")}, 20.0: {30: -1.0, 60: -1.0}}) == 20.0           # NaN last: min() would hide it
    assert N.choose_n0({5.0: {30: float("inf"), 60: float("inf")}, 20.0: {30: 0.1, 60: 0.1}}) == 20.0     # +inf is not a score
    with pytest.raises(ValueError):
        N.choose_n0({5.0: {30: float("nan"), 60: 1.0}, 20.0: {30: float("-inf"), 60: 1.0}})


# ------------------------------------------------------------------ criteria 1 to 3
def test_all_criteria_pass_for_a_clearly_better_arm():
    ev, at = build()
    r = N.evaluate_criteria(ev, at, "shrinkage")
    assert r["c1"] and r["c2"] and r["c3"] and r["all_pass"]
    assert r["vs_gauss"][30]["mean"] == pytest.approx(1.0) and r["vs_personal"][60]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert N.variant_verdict(r)["discriminates"] is False


def test_criterion_1_needs_mean_minus_three_se_above_zero_at_every_length():
    delta = 0.2
    just_ok = 3 * se_of(delta) + 0.001
    ev, at = build(c1=(just_ok, delta))
    assert N.evaluate_criteria(ev, at)["c1"] is True
    ev, at = build(c1=(just_ok - 0.002, delta))
    r = N.evaluate_criteria(ev, at)
    assert r["c1"] is False and r["all_pass"] is False
    for bad_length in LENGTHS:                                                                  # one failing length is enough
        ev, at = build(c1_at={bad_length: (just_ok - 0.002, delta)}, c1=(1.0, 0.1))
        r = N.evaluate_criteria(ev, at)
        assert r["c1"] is False and r["c2"] is True and r["c3"] is True, bad_length


def test_criterion_2_tolerates_mean_plus_one_se_at_or_above_zero_against_personal_and_against_shared():
    delta = 0.3
    ok = -se_of(delta) + 0.001                                                                  # mean < 0 but mean + 1 se > 0: still a pass
    ev, at = build(c2p=(ok, delta), c2s=(ok, delta))
    assert N.evaluate_criteria(ev, at)["c2"] is True and ok < 0
    bad = -se_of(delta) - 0.001
    ev, at = build(c2p=(bad, delta))
    r = N.evaluate_criteria(ev, at)
    assert r["c2"] is False and r["all_pass"] is False
    ev, at = build(c2s=(bad, delta))
    assert N.evaluate_criteria(ev, at)["c2"] is False                                           # shared alone is enough to fail
    for bad_length in LENGTHS:
        ev, at = build(c2p_at={bad_length: (bad, delta)})
        assert N.evaluate_criteria(ev, at)["c2"] is False, bad_length


def test_criterion_3_uses_mean_plus_three_se_at_120_only():
    delta = 0.3
    edge = -3 * se_of(delta, 12)
    ev, at = build(c3=(edge + 0.001, delta))
    assert N.evaluate_criteria(ev, at)["c3"] is True
    ev, at = build(c3=(edge - 0.001, delta))
    r = N.evaluate_criteria(ev, at)
    assert r["c3"] is False and r["c1"] is True and r["c2"] is True and r["all_pass"] is False
    ev, at = build(c3_at={30: (-5.0, delta), 60: (-5.0, delta)}, c3=(0.5, delta))
    assert N.evaluate_criteria(ev, at)["c3"] is True                                            # 30 and 60 steps are reported only
    ev, at = build(c3_at={120: (-5.0, delta)}, c3=(0.5, delta))
    assert N.evaluate_criteria(ev, at)["c3"] is False
    ev, at = build(atyp_lengths=(30, 60))                                                       # no atypical result at 120: not a pass
    assert N.evaluate_criteria(ev, at)["c3"] is False


def test_missing_or_non_finite_scores_are_not_a_pass():
    ev, at = build()
    ev["shrinkage"][60] = {s: float("nan") for s in SEEDS}
    r = N.evaluate_criteria(ev, at)
    assert r["c1"] is False and r["c2"] is False


def test_variant_verdict_discriminates_when_at_least_one_criterion_fails():
    for c1, c2, c3, disc in [(True, True, True, False), (False, True, True, True), (True, False, True, True), (True, True, False, True), (False, False, False, True)]:
        v = N.variant_verdict(dict(arm="v", c1=c1, c2=c2, c3=c3))
        assert v["discriminates"] is disc and v["arm"] == "v"


# ------------------------------------------------------------------ scoring one model on one user
def test_score_model_is_the_held_out_part_of_the_forward_filter():
    from backbone.hssm.dual_model import DualTimescaleHSSM
    from data.synthetic_population import generate_population_user
    u = generate_population_user(31, 2, 1)
    m = DualTimescaleHSSM(N.K, N.F, N.L, N.MAX_DURATION, seed=1).fit(u.X_train, n_iter=3)
    lp, ll = predictive_increments(m, u.X_full)
    r = N.score_model(m, u)
    T = len(u.X_train)
    assert r["mean"] == pytest.approx(float(np.mean(lp[T:]))) and r["n_test"] == len(u.X_test) and r["finite"] and r["scoring_fallbacks"] == 0
    assert r["mean"] != pytest.approx(float(np.mean(lp[:T])))                                    # not the training part
    info = N.fit_info(m)
    assert info["fit_degraded"] is False and info["n_em_iterations"] == len(m.log_likelihood_history_) and info["fit_state"] in ("MAX_ITER_REACHED", "CONVERGED", "REJECTED_UPDATE")


def test_gauss_scores_match_the_n18_baseline_formula_and_a_singular_covariance_is_nan_not_an_error():
    from scipy.stats import multivariate_normal
    from data.synthetic_population import generate_population_user
    u = generate_population_user(32, 2, 1)
    r = N.gauss_scores(u)
    ref = multivariate_normal(mean=u.X_train.mean(axis=0), cov=np.cov(u.X_train, rowvar=False, ddof=1)).logpdf(u.X_test)
    assert r["mean"] == pytest.approx(float(np.mean(ref))) and r["finite"] and r["n_test"] == len(u.X_test)
    flat = type("U", (), {"X_train": np.ones((10, 4)), "X_test": np.zeros((3, 4))})()
    bad = N.gauss_scores(flat)
    assert math.isnan(bad["mean"]) and bad["finite"] is False and bad["n_test"] == 3
