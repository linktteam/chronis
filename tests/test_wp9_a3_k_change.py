"""WP9 A3: K-change detection and overlap candidates (design: tests/PREDECLARED_CRITERIA_WP9.md Part 7)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from backbone.hssm import config as C
from backbone.hssm import k_change as KC
from backbone.hssm import lineage as L
from backbone.hssm.lineage import ALIGNED, CONTINUITY_UNVERIFIED, K_CHANGED, AlignmentConfig, make_summary, may_claim_same_as_before, step
from backbone.hssm.ordering import LadderConfig
from data.synthetic_identity import REGISTRY, IdentitySpec, draw_user

CFG = AlignmentConfig(min_separation=1.0, ladder=LadderConfig(n_bootstrap=40, min_gap=0.5))


# ----------------------------------------------------------------------------------------------------------------- pure functions
def test_the_declared_constants():
    assert C.K_CHANGE_RECENT_DAYS == 45
    assert C.K_CHANGE_DELTA_UP >= 0 and C.K_CHANGE_DELTA_DOWN >= 0
    assert KC.DOMINANT_PARENT_SHARE == 0.5 and KC.MERGE_MIN_SHARE == 0.25


def test_within_variances_are_per_regime_and_per_feature_and_ignore_nan():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1.0, (60, 2)), rng.normal(5, 3.0, (60, 2))])
    lab = np.repeat([0, 1], 60)
    X[3, 0] = np.nan
    v = KC.anchor_within_variances(X, lab, 2)
    assert v.shape == (2, 2) and np.all(v > 0)
    assert v[0, 0] == pytest.approx(np.nanvar(X[:60, 0], ddof=1)) and v[1, 1] == pytest.approx(X[60:, 1].var(ddof=1))
    assert v[1, 0] > 4 * v[0, 0]


def test_a_regime_with_fewer_than_three_rows_takes_the_pooled_variance():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 1.0, (50, 2)), rng.normal(4, 2.0, (2, 2))])
    lab = np.array([0] * 50 + [1] * 2)
    v = KC.anchor_within_variances(X, lab, 2)
    assert v[1] == pytest.approx(v[0]) or np.all(v[1] > 0)           # never zero, never NaN
    pooled_num = ((X[:50] - X[:50].mean(0)) ** 2).sum(0) + ((X[50:] - X[50:].mean(0)) ** 2).sum(0)
    assert v[1] == pytest.approx(pooled_num / (52 - 2))


def test_a_constant_feature_gets_a_positive_variance():
    X = np.column_stack([np.repeat([0.0, 3.0], 20), np.full(40, 7.0)])
    v = KC.anchor_within_variances(X, np.repeat([0, 1], 20), 2)
    assert np.all(v > 0) and np.all(np.isfinite(v))


def test_rows_go_to_the_regime_of_highest_density_and_nan_features_are_skipped():
    means = np.array([[0.0, 0.0], [10.0, 10.0]]); var = np.ones((2, 2))
    X = np.array([[0.2, -0.1], [9.5, 10.4], [np.nan, 9.0], [np.nan, np.nan], [6.0, 5.5]])
    assert KC.assign_rows(X, means, var).tolist() == [0, 1, 1, -1, 1]


def test_a_wider_regime_takes_a_row_between_the_means_by_density_not_by_distance():
    means = np.array([[0.0], [4.0]]); var = np.array([[0.25], [9.0]])
    # at x = 1.5 the nearest mean is regime 0, but regime 1 is far wider
    assert KC.assign_rows(np.array([[3.0]]), means, var).tolist() == [1]
    assert KC.assign_rows(np.array([[0.1]]), means, var).tolist() == [0]


def test_a_narrow_regime_keeps_a_row_near_its_mean_against_a_much_wider_one():
    means = np.array([[0.0], [4.0]]); var = np.array([[0.25], [100.0]])
    # at x = 0.9: the squared distance to regime 1 is far smaller in z terms, but the normalising log variance of the wide regime makes the narrow regime the more likely
    assert KC.assign_rows(np.array([[0.9]]), means, var).tolist() == [0]


def test_overlap_shares_are_row_shares_per_new_regime():
    parent = np.array([0, 0, 0, 1, 1, 2, -1, -1])
    new = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = KC.overlap_shares(parent, new, 3, 2)
    assert s.shape == (2, 3)
    assert s[0].tolist() == pytest.approx([0.75, 0.25, 0.0])               # unassigned rows are left out
    assert s[1].tolist() == pytest.approx([0.0, 0.5, 0.5])


def test_an_empty_new_regime_has_zero_shares():
    s = KC.overlap_shares(np.array([0, 1]), np.array([0, 0]), 2, 3)
    assert s[1].sum() == 0 and s[2].sum() == 0


def test_split_candidate_needs_two_children_of_one_dominant_parent():
    sh = np.array([[0.9, 0.1, 0.0], [0.8, 0.1, 0.1], [0.0, 0.05, 0.95], [0.1, 0.9, 0.0]])
    c = KC.overlap_candidates(sh, 3, 4)
    assert c == (dict(kind="split", old_raw_regime=0, new_raw_regimes=[0, 1], method="overlap"),)


def test_one_child_is_not_a_split():
    sh = np.array([[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.0, 0.1, 0.9], [0.0, 0.05, 0.95]])
    c = KC.overlap_candidates(sh, 3, 4)
    assert [x["old_raw_regime"] for x in c] == [2]                              # only old regime 2 has two children; regime 0 and 1 have one each


def test_the_dominant_parent_level_is_inclusive_at_a_half():
    sh = np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])         # K_new = 3 > K_old = 2? use shape (3, 2)
    sh = np.array([[0.5, 0.5], [0.5, 0.5], [0.1, 0.9]])
    c = KC.overlap_candidates(sh, 2, 3)
    # rows 0 and 1 have argmax 0 (ties go to the lower index) and a share of exactly 0.5: both children of old 0
    assert c and c[0]["old_raw_regime"] == 0 and c[0]["new_raw_regimes"] == [0, 1]
    sh2 = np.array([[0.49, 0.51], [0.49, 0.51], [0.1, 0.9]])
    c2 = KC.overlap_candidates(sh2, 2, 3)
    assert all(x["old_raw_regime"] != 0 for x in c2)


def test_merge_candidate_needs_two_old_regimes_with_a_quarter_each():
    sh = np.array([[0.55, 0.45, 0.0], [0.0, 0.1, 0.9]])
    c = KC.overlap_candidates(sh, 3, 2)
    assert c == (dict(kind="merge", old_raw_regimes=[0, 1], new_raw_regime=0, method="overlap"),)


def test_the_merge_level_is_inclusive_at_a_quarter():
    assert KC.overlap_candidates(np.array([[0.75, 0.25, 0.0], [0.0, 0.0, 1.0]]), 3, 2)
    assert not KC.overlap_candidates(np.array([[0.76, 0.24, 0.0], [0.0, 0.0, 1.0]]), 3, 2)


def test_the_merge_pair_is_the_two_largest_shares():
    sh = np.array([[0.2, 0.4, 0.4], [1.0, 0.0, 0.0]])
    assert KC.overlap_candidates(sh, 3, 2)[0]["old_raw_regimes"] == [1, 2]


def test_no_candidates_when_the_number_of_regimes_is_the_same():
    assert KC.overlap_candidates(np.eye(3), 3, 3) == ()


# ----------------------------------------------------------------------------------------------------------------- the decision
def test_the_best_bic_with_a_margin_above_delta_is_suspected_up():
    d = KC.decide({2: 300.0, 3: 310.0, 4: 295.0}, 3, delta_up=10.0, delta_down=10.0)
    assert d["suspected"] and d["direction"] == "up" and d["k_best"] == 4 and d["margin"] == pytest.approx(15.0)


def test_the_margin_level_is_inclusive():
    assert KC.decide({2: 330.0, 3: 310.0, 4: 300.0}, 3, 10.0, 10.0)["suspected"]
    d = KC.decide({2: 330.0, 3: 310.0, 4: 300.1}, 3, 10.0, 10.0)
    assert not d["suspected"] and d["reason"] == "margin_below_delta" and d["direction"] == "up"


def test_the_two_directions_have_their_own_delta():
    b = {2: 300.0, 3: 312.0, 4: 330.0}
    assert KC.decide(b, 3, delta_up=10.0, delta_down=20.0)["suspected"] is False
    assert KC.decide(b, 3, delta_up=20.0, delta_down=10.0)["suspected"] is True
    assert KC.decide(b, 3, delta_up=20.0, delta_down=10.0)["direction"] == "down"
    b2 = {2: 330.0, 3: 312.0, 4: 300.0}
    assert KC.decide(b2, 3, delta_up=10.0, delta_down=99.0)["suspected"] is True
    assert KC.decide(b2, 3, delta_up=99.0, delta_down=10.0)["suspected"] is False


def test_a_tie_goes_to_the_anchor_k_and_the_anchor_best_is_not_suspected():
    d = KC.decide({2: 310.0, 3: 300.0, 4: 300.0}, 3, 0.0, 0.0)
    assert not d["suspected"] and d["reason"] == "k_anchor_is_best" and d["k_best"] == 3


def test_a_zero_delta_still_needs_the_other_k_to_be_strictly_better_or_the_tie_goes_to_the_anchor():
    assert not KC.decide({3: 300.0, 4: 300.0}, 3, 0.0, 0.0)["suspected"]
    assert KC.decide({3: 300.0, 4: 299.9}, 3, 0.0, 0.0)["suspected"]


def test_no_reference_fit_no_signal():
    d = KC.decide({2: 300.0, 4: 250.0}, 3, 0.0, 0.0)
    assert not d["suspected"] and d["reason"] == "reference_fit_unavailable"
    assert not KC.decide({2: 300.0, 3: None, 4: 250.0}, 3, 0.0, 0.0)["suspected"]


def test_non_finite_bic_values_are_unavailable():
    d0 = KC.decide({3: float("nan"), 4: 100.0}, 3, 0.0, 0.0)
    assert not d0["suspected"] and d0["reason"] == "reference_fit_unavailable" and d0["margin"] is None
    d = KC.decide({3: 300.0, 4: float("nan"), 2: float("inf")}, 3, 0.0, 0.0)
    assert not d["suspected"] and d["k_best"] == 3


def test_a_non_adjacent_best_k_is_never_a_signal():
    d = KC.decide({3: 300.0, 5: 100.0}, 3, 0.0, 0.0)
    assert not d["suspected"] and d["reason"] == "not_adjacent"


# ----------------------------------------------------------------------------------------------------------------- the assessment (fake fits)
def _fake_fit(table, labels_by_k=None, accepted=None):
    calls = []

    def fit(recent, K):
        calls.append((len(recent), K))
        if table.get(K) is None:
            return None
        return dict(bic=table[K], labels=(labels_by_k or {}).get(K, np.zeros(len(recent), dtype=int)), raw_means=np.zeros((K, recent.shape[1])), accepted=True if accepted is None else accepted.get(K, True))
    return fit, calls


def _anchor3():
    means = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]); var = np.ones((3, 2))
    return means, var


def test_the_recent_window_is_the_last_days_and_three_ks_are_fitted():
    means, var = _anchor3()
    fit, calls = _fake_fit({2: 320.0, 3: 310.0, 4: 330.0})
    raw = np.zeros((120, 2))
    r = KC.assess_k_change(means, var, raw, fit_fn=fit, recent_days=45)
    assert calls == [(45, 2), (45, 3), (45, 4)] and not r["suspected"] and r["recent_days"] == 45
    assert r["bic_by_k"] == {"2": 320.0, "3": 310.0, "4": 330.0}


def test_a_window_shorter_than_the_recent_window_is_used_whole():
    means, var = _anchor3()
    fit, calls = _fake_fit({2: 320.0, 3: 310.0, 4: 330.0})
    KC.assess_k_change(means, var, np.zeros((30, 2)), fit_fn=fit, recent_days=45)
    assert calls[0][0] == 30


def test_a_suspected_split_carries_the_overlap_candidate():
    means, var = _anchor3()
    rng = np.random.default_rng(0)
    # the recent window: regime 0 (rows 0..29) is two clusters near mean 0 -> both children of old regime 0; rows 30..44 sit on regime 1
    recent = np.vstack([rng.normal([0, 0], 0.3, (30, 2)), rng.normal([10, 0], 0.3, (15, 2))])
    lab4 = np.array([0] * 15 + [1] * 15 + [2] * 15)
    fit, _ = _fake_fit({2: 400.0, 3: 380.0, 4: 340.0}, labels_by_k={4: lab4})
    r = KC.assess_k_change(means, var, recent, fit_fn=fit, delta_up=10.0, delta_down=10.0)
    assert r["suspected"] and r["direction"] == "up" and r["k_best"] == 4
    assert r["candidates"] and r["candidates"][0]["kind"] == "split" and r["candidates"][0]["old_raw_regime"] == 0 and r["candidates"][0]["new_raw_regimes"] == [0, 1]
    assert np.asarray(r["overlap_shares"]).shape == (4, 3)


def test_a_suspected_merge_carries_the_overlap_candidate():
    means, var = _anchor3()
    rng = np.random.default_rng(1)
    recent = np.vstack([rng.normal([0, 0], 0.3, (15, 2)), rng.normal([10, 0], 0.3, (15, 2)), rng.normal([0, 10], 0.3, (15, 2))])
    lab2 = np.array([0] * 30 + [1] * 15)                                         # old regimes 0 and 1 lumped into one new regime
    fit, _ = _fake_fit({2: 300.0, 3: 330.0, 4: 340.0}, labels_by_k={2: lab2})
    r = KC.assess_k_change(means, var, recent, fit_fn=fit)
    assert r["suspected"] and r["direction"] == "down"
    assert r["candidates"] == (dict(kind="merge", old_raw_regimes=[0, 1], new_raw_regime=0, method="overlap"),)


def test_a_fit_that_is_not_accepted_gives_no_claim_from_that_k():
    means, var = _anchor3()
    fit, _ = _fake_fit({2: 320.0, 3: 310.0, 4: 200.0}, accepted={4: False})
    r = KC.assess_k_change(means, var, np.zeros((45, 2)), fit_fn=fit)
    assert not r["suspected"] and r["bic_by_k"]["4"] is None


def test_an_unaccepted_reference_fit_gives_no_signal_at_all():
    means, var = _anchor3()
    fit, _ = _fake_fit({2: 320.0, 3: 310.0, 4: 200.0}, accepted={3: False})
    r = KC.assess_k_change(means, var, np.zeros((45, 2)), fit_fn=fit)
    assert not r["suspected"] and r["reason"] == "reference_fit_unavailable"


def test_a_fit_that_raises_counts_as_unavailable_and_the_assessment_never_raises():
    means, var = _anchor3()

    def boom(recent, K):
        if K == 4:
            raise RuntimeError("no convergence")
        return dict(bic=300.0, labels=np.zeros(len(recent), dtype=int), raw_means=np.zeros((K, 2)), accepted=True)
    r = KC.assess_k_change(means, var, np.zeros((45, 2)), fit_fn=boom)
    assert not r["suspected"] and r["bic_by_k"]["4"] is None
    bad = KC.assess_k_change(means, var, "not an array", fit_fn=boom)
    assert bad["suspected"] is False and bad["reason"] == "assessment_error"


def test_k_below_two_is_never_fitted():
    means = np.array([[0.0, 0.0], [9.0, 9.0]]); var = np.ones((2, 2))
    fit, calls = _fake_fit({2: 300.0, 3: 280.0})
    KC.assess_k_change(means, var, np.zeros((45, 2)), fit_fn=fit, delta_up=1.0)
    assert [k for _, k in calls] == [2, 3]


def test_the_record_is_json_serialisable_and_carries_its_thresholds():
    means, var = _anchor3()
    fit, _ = _fake_fit({2: 320.0, 3: 310.0, 4: 330.0})
    r = KC.assess_k_change(means, var, np.zeros((45, 2)), fit_fn=fit, delta_up=7.0, delta_down=9.0)
    json.dumps(r)
    assert r["thresholds"] == dict(delta_up=7.0, delta_down=9.0) and r["schema_version"] == KC.SCHEMA_VERSION


def test_assess_against_anchor_without_state_or_window():
    assert KC.assess_against_anchor(None, np.zeros((45, 2)))["reason"] == "no_anchor"
    m = _means(3)
    s = _summary(m, seed=1)
    state = L.LineageState("t", 0, L.Anchor(s, (0, 1, 2), 0, 0, False), 1)
    s_nowin = L.FitSummary(s.engine, s.raw_means, s.scaler_mean, s.scaler_std, s.labels, s.separation_min, None)
    st2 = L.LineageState("t", 0, L.Anchor(s_nowin, (0, 1, 2), 0, 0, False), 1)
    r = KC.assess_against_anchor(st2, np.zeros((45, 3)))
    assert r["reason"] == "anchor_has_no_window" and not r["suspected"]
    fit, calls = _fake_fit({2: 400.0, 3: 380.0, 4: 340.0})
    r2 = KC.assess_against_anchor(state, s.raw_window[-60:], fit_fn=fit, delta_up=10.0)
    assert r2["suspected"] and calls[0][0] == 45 and r2["k_best"] == 4


def test_assess_against_anchor_hands_the_anchor_means_variances_and_k_to_the_assessment(monkeypatch):
    m = _means(3)
    s = _summary(m, seed=1)
    state = L.LineageState("t", 0, L.Anchor(s, (0, 1, 2), 0, 0, False), 1)
    seen = {}

    def spy(means, variances, raw, **kw):
        seen.update(means=means, variances=variances, raw=raw, **kw)
        return dict(suspected=False)
    monkeypatch.setattr(KC, "assess_k_change", spy)
    win = s.raw_window[-80:]
    KC.assess_against_anchor(state, win, delta_up=7.0, delta_down=9.0, recent_days=33, fit_fn=lambda r, k: None)
    assert np.array_equal(seen["means"], s.raw_means) and np.array_equal(seen["raw"], win)
    assert np.array_equal(seen["variances"], KC.anchor_within_variances(s.raw_window, s.labels, s.K))
    assert seen["k_anchor"] == 3 and seen["recent_days"] == 33 and seen["delta_up"] == 7.0 and seen["delta_down"] == 9.0


# ----------------------------------------------------------------------------------------------------------------- the lineage
def _means(seed, K=3, sep=2.5):
    return draw_user(seed, IdentitySpec(n_regimes=K, sep_lo=sep, sep_hi=sep, offset_sd=0.3))["means"]


def _summary(means, seed=0, T=240, noise=0.35, **kw):
    from data.synthetic_identity import SCALE
    rng = np.random.default_rng(seed)
    K, F = means.shape
    labels = rng.permutation(np.repeat(np.arange(K), T // K)[:T])
    window = means[labels] + noise * SCALE * rng.standard_normal((len(labels), F))
    return make_summary("baseline", means, window.mean(axis=0), window.std(axis=0), labels, window, **kw)


SUSPECT = dict(schema_version=1, suspected=True, direction="up", k_best=4, margin=22.0, reason=None, bic_by_k={}, candidates=(dict(kind="split", old_raw_regime=1, new_raw_regimes=[0, 1], method="overlap"),),
               recent_days=45, thresholds=dict(delta_up=10.0, delta_down=10.0))


def test_the_new_fields_default_to_nothing_known():
    assert L.FitSummary.__dataclass_fields__["k_change"].default is None
    assert L.LineageRecord.__dataclass_fields__["k_change"].default is None
    m = _means(3)
    s = _summary(m, seed=1)
    bare = L.FitSummary(s.engine, s.raw_means, s.scaler_mean, s.scaler_std, s.labels, s.separation_min, s.raw_window, s.exp_durations)
    assert bare.k_change is None and not L.k_change_suspected(bare)


def test_without_a_k_change_record_two_good_fits_are_aligned_as_in_wp8():
    m = _means(3)
    a, n = _summary(m, seed=1), _summary(m, seed=2)
    _, st = step(None, a, CFG, REGISTRY, lineage_id="t")
    r, _ = step(st, n, CFG, REGISTRY)
    assert r.continuity_state == ALIGNED and r.reasons == () and r.k_change is None and may_claim_same_as_before(r)


def test_a_record_that_says_not_suspected_changes_nothing():
    m = _means(3)
    a, n = _summary(m, seed=1), _summary(m, seed=2, k_change=dict(SUSPECT, suspected=False, candidates=()))
    _, st = step(None, a, CFG, REGISTRY, lineage_id="t")
    r, _ = step(st, n, CFG, REGISTRY)
    assert r.continuity_state == ALIGNED and r.reasons == () and may_claim_same_as_before(r)
    assert r.k_change is not None and r.k_change["suspected"] is False


def test_a_suspected_change_with_the_same_k_is_never_served_as_same_as_before():
    m = _means(3)
    a, n = _summary(m, seed=1), _summary(m, seed=2, k_change=SUSPECT)
    _, st = step(None, a, CFG, REGISTRY, lineage_id="t")
    r, st1 = step(st, n, CFG, REGISTRY)
    assert r.continuity_state == CONTINUITY_UNVERIFIED and r.reasons == ("k_change_suspected",)
    assert not may_claim_same_as_before(r) and r.message.startswith(L.MESSAGE_UNCONFIRMED) and r.proposed_new_names is not None
    assert st1.anchor is st.anchor and st1.n_fits == 2                       # a stable anchor does not move


def test_the_suspicion_must_be_exactly_true():
    m = _means(3)
    a = _summary(m, seed=1)
    _, st = step(None, a, CFG, REGISTRY, lineage_id="t")
    for odd in ("yes", 1, [True], None):
        r, _ = step(st, _summary(m, seed=2, k_change=dict(SUSPECT, suspected=odd)), CFG, REGISTRY)
        assert r.continuity_state == ALIGNED, odd


def test_a_malformed_record_is_nothing_known():
    m = _means(3)
    a = _summary(m, seed=1)
    _, st = step(None, a, CFG, REGISTRY, lineage_id="t")
    n = _summary(m, seed=2)
    n = L.FitSummary(n.engine, n.raw_means, n.scaler_mean, n.scaler_std, n.labels, n.separation_min, n.raw_window, n.exp_durations, None, False, None, "garbage")
    r, _ = step(st, n, CFG, REGISTRY)
    assert r.continuity_state == ALIGNED


def test_a_different_fitted_k_stays_k_changed_and_carries_the_record():
    m3 = _means(3)
    m4 = np.vstack([m3, m3[1] + 2.0])
    a, n = _summary(m3, seed=1), _summary(m4, seed=2, k_change=SUSPECT)
    _, st = step(None, a, CFG, REGISTRY, lineage_id="t")
    r, _ = step(st, n, CFG, REGISTRY)
    assert r.continuity_state == K_CHANGED and r.reasons == ("k_changed",) and not may_claim_same_as_before(r)
    assert r.k_change is not None and r.k_change["suspected"] is True and r.split_merge_candidates is not None   # the old nearest-mean candidates are still there


def test_the_record_gives_the_old_regimes_in_stable_ids():
    m = _means(3)
    a = _summary(m, seed=1)
    _, st = step(None, a, CFG, REGISTRY, lineage_id="t")
    r2s = st.anchor.raw_to_stable
    kc = dict(SUSPECT, candidates=(dict(kind="split", old_raw_regime=1, new_raw_regimes=[0, 1], method="overlap"), dict(kind="merge", old_raw_regimes=[0, 2], new_raw_regime=0, method="overlap")))
    r, _ = step(st, _summary(m, seed=2, k_change=kc), CFG, REGISTRY)
    c = r.k_change["candidates"]
    assert c[0]["old_raw_regime"] == 1 and c[0]["old_stable_regime"] == r2s[1]
    assert c[1]["old_raw_regimes"] == [0, 2] and c[1]["old_stable_regimes"] == sorted([r2s[0], r2s[2]])


def test_old_regimes_are_translated_with_a_non_identity_naming():
    m = _means(3)
    a = _summary(m, seed=1)
    state = L.LineageState("t", 0, L.Anchor(a, (2, 0, 1), 0, 1, True), 1)                       # raw regime 0 is stable 2, raw 1 is stable 0, raw 2 is stable 1
    kc = dict(SUSPECT, candidates=(dict(kind="split", old_raw_regime=1, new_raw_regimes=[0, 1], method="overlap"), dict(kind="merge", old_raw_regimes=[0, 2], new_raw_regime=0, method="overlap")))
    r, _ = step(state, _summary(m, seed=2, k_change=kc), CFG, REGISTRY)
    c = r.k_change["candidates"]
    assert c[0]["old_stable_regime"] == 0 and c[0]["old_raw_regime"] == 1
    assert c[1]["old_stable_regimes"] == [1, 2] and c[1]["old_raw_regimes"] == [0, 2]


def test_the_record_dict_is_json_serialisable():
    m = _means(3)
    a, n = _summary(m, seed=1), _summary(m, seed=2, k_change=SUSPECT)
    _, st = step(None, a, CFG, REGISTRY, lineage_id="t")
    r, _ = step(st, n, CFG, REGISTRY)
    d = r.as_dict()
    json.dumps(d)
    assert d["k_change"]["suspected"] is True and d["continuity_state"] == CONTINUITY_UNVERIFIED


def test_a_suspected_change_adds_to_the_other_reasons_and_does_not_hide_them():
    m = _means(3)
    a, n = _summary(m, seed=1), _summary(m, seed=2, fit_unstable=True, k_change=SUSPECT)
    _, st = step(None, a, CFG, REGISTRY, lineage_id="t")
    r, _ = step(st, n, CFG, REGISTRY)
    assert "fit_unstable" in r.reasons and "k_change_suspected" in r.reasons


def test_the_user_can_still_choose_new_names_after_a_suspected_change():
    m = _means(3)
    a, n = _summary(m, seed=1), _summary(m, seed=2, k_change=SUSPECT)
    _, st = step(None, a, CFG, REGISTRY, lineage_id="t")
    r, st1 = step(st, n, CFG, REGISTRY)
    r2, st2 = L.apply_user_choice(st1, r, "NEW_NAMES")
    assert st2.epoch == st1.epoch + 1


def test_the_first_fit_of_a_lineage_ignores_a_k_change_record():
    m = _means(3)
    rec0, _ = step(None, _summary(m, seed=1, k_change=SUSPECT), CFG, REGISTRY, lineage_id="t")
    assert rec0.continuity_state == L.NEW_LINEAGE


def test_summarize_fit_passes_the_record_on():
    import inspect
    assert "k_change" in inspect.signature(L.summarize_fit).parameters and "k_change" in inspect.signature(L.make_summary).parameters


# ----------------------------------------------------------------------------------------------------------------- the windowed fit (live)
def _blobs(seed, centres, n_each, sd=0.4):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(c, sd, (n_each, len(c))) for c in centres])
    return X[rng.permutation(len(X))]


def test_fit_recent_window_returns_bic_labels_and_raw_means_in_raw_units():
    X = _blobs(0, [[100.0, 5.0, 40.0], [130.0, 9.0, 55.0]], 25) * 1.0
    r = KC.fit_recent_window(X, 2, seed=3, n_init=3, max_iter=40, allow_fast_test_fit=True)
    assert r is not None and np.isfinite(r["bic"]) and len(r["labels"]) == 50 and r["raw_means"].shape == (2, 3)
    assert r["accepted"] is True
    order = np.argsort(r["raw_means"][:, 0])
    assert r["raw_means"][order, 0] == pytest.approx([100.0, 130.0], abs=3.0)


def test_fit_recent_window_prefers_the_true_k_by_bic_on_well_separated_data():
    X = _blobs(1, [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 6.0, 0.0]], 20, sd=0.4)
    b = {K: KC.fit_recent_window(X, K, seed=5, n_init=3, max_iter=40, allow_fast_test_fit=True) for K in (2, 3, 4)}
    assert all(v is not None for v in b.values())
    assert b[3]["bic"] < b[2]["bic"] and b[3]["bic"] < b[4]["bic"]


def test_fit_recent_window_reports_a_withheld_fit_as_not_accepted_and_the_bic_of_the_requested_k(monkeypatch):
    from types import SimpleNamespace
    from backbone.hssm import fitting

    def fake(**kw):
        n = len(kw["matrix"])
        return SimpleNamespace(p_t_posterior=np.tile([1.0, 0.0], (n, 1)), bic_by_k={2: 100.0, 3: 50.0}, model=SimpleNamespace(mu=np.zeros((2, kw["matrix"].shape[1]))), insights_withheld=True)
    monkeypatch.setattr(fitting, "fit_hssm", fake)
    r = KC.fit_recent_window(_blobs(0, [[0.0, 0.0], [5.0, 5.0]], 10), 2, seed=0)
    assert r["accepted"] is False and r["bic"] == 100.0
    monkeypatch.setattr(fitting, "fit_hssm", lambda **kw: SimpleNamespace(p_t_posterior=np.tile([1.0, 0.0], (len(kw["matrix"]), 1)), bic_by_k={2: 100.0}, model=SimpleNamespace(mu=np.zeros((2, 2))), insights_withheld=False))
    assert KC.fit_recent_window(_blobs(0, [[0.0, 0.0], [5.0, 5.0]], 10), 2, seed=0)["accepted"] is True


def test_production_fit_needs_the_full_ten_restarts():
    X = _blobs(0, [[0.0, 0.0, 0.0], [6.0, 6.0, 6.0]], 25)
    assert KC.fit_recent_window(X, 2, seed=3, n_init=3, max_iter=40) is None            # fewer than 10 restarts is refused unless the test switch is on


def test_fit_recent_window_returns_none_when_the_fit_cannot_run():
    assert KC.fit_recent_window(np.zeros((3, 3)), 5, seed=0, n_init=2, max_iter=5, allow_fast_test_fit=True) is None


def test_fit_recent_window_handles_missing_values_without_raising():
    X = _blobs(2, [[0.0, 0.0, 0.0], [6.0, 6.0, 6.0]], 25)
    X[3, 1] = np.nan; X[10, :] = np.nan
    r = KC.fit_recent_window(X, 2, seed=1, n_init=3, max_iter=40, allow_fast_test_fit=True)
    assert r is None or np.isfinite(r["bic"])
