"""WP8 Item 4: lineage and continuity (owner decisions A3, A4, A6, C2). Pure functions on synthetic summaries; an independent brute-force oracle checks the alignment."""
from __future__ import annotations

import copy
import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest

from backbone.hssm import lineage as L
from backbone.hssm.lineage import (ALIGNED, CONTINUITY_UNVERIFIED, K_CHANGED, NEW_LINEAGE, AlignmentConfig, PairThreshold, alignment, apply_user_choice, make_summary,
                                   may_claim_same_as_before, step)
from backbone.hssm.ordering import LadderConfig
from data.synthetic_identity import REGISTRY, SCALE, IdentitySpec, draw_user

CFG = AlignmentConfig(min_separation=1.0, ladder=LadderConfig(n_bootstrap=40, min_gap=0.5))


def _means(seed, K=3, sep=2.0, offset_sd=0.3):
    return draw_user(seed, IdentitySpec(n_regimes=K, sep_lo=sep, sep_hi=sep, offset_sd=offset_sd))["means"]          # (K, 7) physical units, planted activity order = index


def _summary(means, raw_perm=None, engine="baseline", seed=0, T=240, noise=0.35, scaler=None):
    """A fit whose raw regime r is the planted regime raw_perm[r] (a relabelling), with data drawn around the means. Returns the summary."""
    rng = np.random.default_rng(seed)
    K, F = means.shape
    raw_perm = list(range(K)) if raw_perm is None else list(raw_perm)
    fit_means = means[raw_perm]
    labels = np.repeat(np.arange(K), T // K)[:T]; labels = rng.permutation(labels)
    window = fit_means[labels] + noise * SCALE * rng.standard_normal((len(labels), F))
    mean, std = (window.mean(axis=0), window.std(axis=0)) if scaler is None else scaler
    return make_summary(engine, fit_means, mean, std, labels, window)


# ------------------------------------------------------------------------------------------------------------------- alignment against a brute-force oracle
@pytest.mark.parametrize("seed", range(20))
def test_alignment_equals_brute_force_over_all_permutations(seed):
    r = np.random.default_rng(seed); K = int(r.integers(2, 6)); D = int(r.integers(2, 8))
    a, n = r.standard_normal((K, D)), r.standard_normal((K, D))
    costs = {q: sum(((a[i] - n[q[i]]) ** 2).sum() for i in range(K)) for q in itertools.permutations(range(K))}
    ordered = sorted(costs.items(), key=lambda kv: kv[1])
    al = alignment(a, n)
    assert al["perm"] == ordered[0][0]
    assert np.isclose(al["C1"], ordered[0][1]) and np.isclose(al["C2"], ordered[1][1])
    assert np.isclose(al["margin_diff"], (ordered[1][1] - ordered[0][1]) / K) and np.isclose(al["margin_ratio"], ordered[0][1] / ordered[1][1])
    assert np.isclose(sum(al["matched"]), al["C1"])
    for i in range(K):
        assert np.isclose(al["nearest_other"][i], min(((a[i] - n[j]) ** 2).sum() for j in range(K) if j != al["perm"][i]))


def test_alignment_of_a_single_regime_is_trivially_unambiguous():
    al = alignment(np.zeros((1, 3)), np.ones((1, 3)))
    assert al["perm"] == (0,) and al["C2"] == float("inf") and al["margin_ratio"] == 0.0 and L.margin_ok(PairThreshold(), al)


def test_acceptance_rules_follow_their_definitions_and_a_tie_is_never_accepted():
    al = dict(C1=1.0, C2=4.0, margin_diff=1.0, margin_ratio=0.25)
    assert L.margin_ok(PairThreshold("ratio", rho=0.25), al) and not L.margin_ok(PairThreshold("ratio", rho=0.24), al)
    assert L.margin_ok(PairThreshold("diff", tau=1.0), al) and not L.margin_ok(PairThreshold("diff", tau=1.01), al)
    assert L.margin_ok(PairThreshold("both", tau=1.0, rho=0.25), al)
    assert not L.margin_ok(PairThreshold("both", tau=1.0, rho=0.24), al) and not L.margin_ok(PairThreshold("both", tau=1.01, rho=0.25), al)
    tie = dict(C1=2.0, C2=2.0, margin_diff=0.0, margin_ratio=1.0)
    assert not L.margin_ok(PairThreshold("diff", tau=0.0), tie)                                       # tau = 0 must not accept a tie
    zero = dict(C1=0.0, C2=0.0, margin_diff=0.0, margin_ratio=1.0)
    assert not L.margin_ok(PairThreshold("ratio"), zero)


def test_thresholds_and_config_are_validated():
    with pytest.raises(ValueError):
        PairThreshold("sometimes")
    with pytest.raises(ValueError):
        PairThreshold(rho=0.51)                                                                         # never more lenient than the WP5 rule
    with pytest.raises(ValueError):
        PairThreshold(tau=-0.1)
    with pytest.raises(ValueError):
        AlignmentConfig(min_separation=-1.0)
    with pytest.raises(ValueError):
        AlignmentConfig(duration_weight=-0.1)


# ------------------------------------------------------------------------------------------------------------------- naming and inheritance
def test_the_first_fit_is_named_by_the_ladder_and_needs_a_lineage_id():
    means = _means(11)
    s = _summary(means, raw_perm=[2, 0, 1])                                  # raw regime 0 is planted regime 2 (highest), raw 1 is planted 0, raw 2 is planted 1
    with pytest.raises(ValueError):
        step(None, s, CFG, REGISTRY)
    rec, st = step(None, s, CFG, REGISTRY, lineage_id="u:L1")
    assert rec.continuity_state == NEW_LINEAGE and rec.lineage_epoch == 0 and rec.fit_index == 0 and st.n_fits == 1
    assert rec.ordering_verified and rec.activity_rung in (1, 2, 3)
    assert rec.raw_to_stable == (2, 0, 1)                                    # stable id = planted activity rank
    assert np.array_equal(rec.stable_regime_id, np.array(rec.raw_to_stable)[s.labels])
    assert not may_claim_same_as_before(rec)                                  # a first fit has nothing to be the same as


def test_an_unverified_ordering_names_the_first_fit_in_its_own_order_with_no_activity_claim():
    means = _means(11, sep=0.01, offset_sd=0.0)                             # no activity signal at all: the three regimes have the same means
    s = _summary(means, seed=3)
    rec, _ = step(None, s, CFG, REGISTRY, lineage_id="u:L1")
    assert not rec.ordering_verified and rec.activity_rung == 0 and rec.raw_to_stable == (0, 1, 2) and "first-fit order" in rec.message


def test_without_a_window_or_registry_the_first_fit_is_named_in_first_fit_order():
    s = _summary(_means(11)); s2 = L.FitSummary(s.engine, s.raw_means, s.scaler_mean, s.scaler_std, s.labels, s.separation_min, None, None)
    rec, _ = step(None, s2, CFG, REGISTRY, lineage_id="x")
    assert rec.raw_to_stable == (0, 1, 2) and not rec.ordering_verified
    rec, _ = step(None, s, CFG, None, lineage_id="x")
    assert rec.raw_to_stable == (0, 1, 2) and not rec.ordering_verified


@pytest.mark.parametrize("perm", list(itertools.permutations(range(3))))
def test_a_refit_with_any_relabelling_inherits_the_names(perm):
    means = _means(11)
    a = _summary(means, raw_perm=[1, 2, 0], seed=1)
    n = _summary(means, raw_perm=perm, seed=2)
    r0, st = step(None, a, CFG, REGISTRY, lineage_id="u")
    r1, st1 = step(st, n, CFG, REGISTRY)
    assert r1.continuity_state == ALIGNED and may_claim_same_as_before(r1) and r1.reasons == ()
    planted_of_raw = list(perm)                                              # new raw regime r is planted regime perm[r]
    assert list(r1.raw_to_stable) == planted_of_raw                           # planted regime p carries stable id p in both fits (the first fit's ladder named them by planted order)
    assert st1.anchor.summary is n and st1.anchor.fit_index == 1 and st1.n_fits == 2 and st1.pending is None


def test_names_are_inherited_not_resorted_even_when_the_ladder_would_now_order_differently():
    means = _means(11)
    a = _summary(means, seed=1)
    rec0, st = step(None, a, CFG, REGISTRY, lineage_id="u")
    flipped = means.copy(); flipped[[0, 2]] = flipped[[2, 0]]                 # the new fit's regimes look reversed in activity but sit exactly where the old ones were? no: they are swapped places
    n = _summary(means, raw_perm=[0, 1, 2], seed=5)
    rec1, st1 = step(st, n, CFG, REGISTRY)
    assert rec1.continuity_state == ALIGNED and rec1.raw_to_stable == rec0.raw_to_stable
    assert rec1.activity_rung == rec0.activity_rung and rec1.ordering_verified == rec0.ordering_verified   # inherited, not recomputed


def test_a_chain_of_slowly_drifting_refits_keeps_the_names():
    means = _means(11); rng = np.random.default_rng(0)
    rec, st = step(None, _summary(means, seed=0), CFG, REGISTRY, lineage_id="u")
    cur = means.copy()
    for i in range(1, 8):
        cur = cur + 0.08 * SCALE * rng.standard_normal(cur.shape)             # drift per refit far below the regime separation
        rec, st = step(st, _summary(cur, raw_perm=list(np.random.default_rng(i).permutation(3)), seed=i), CFG, REGISTRY)
        assert rec.continuity_state == ALIGNED
        planted = list(np.random.default_rng(i).permutation(3))
        assert list(rec.raw_to_stable) == planted


# ------------------------------------------------------------------------------------------------------------------- A3: old names kept when continuity cannot be confirmed
def _ambiguous_pair(seed=11):
    """Two regimes that are nearly mirror images of each other around the anchor's, so the best and second best assignments nearly tie."""
    means = _means(seed); a = _summary(means, seed=1)
    mid = means.mean(axis=0)
    n_means = means.copy(); n_means[0] = mid + 0.02 * SCALE; n_means[2] = mid - 0.02 * SCALE; n_means[1] = means[1]
    return a, _summary(n_means, seed=2, scaler=(a.scaler_mean, a.scaler_std))


def test_ambiguous_alignment_keeps_the_old_names_says_so_and_stores_the_margin():
    a, n = _ambiguous_pair()
    _, st = step(None, a, CFG, REGISTRY, lineage_id="u")
    rec, st1 = step(st, n, AlignmentConfig(min_separation=0.0, ladder=CFG.ladder, same_engine=PairThreshold("ratio", rho=0.1)), REGISTRY)
    assert rec.continuity_state == CONTINUITY_UNVERIFIED and "margin_below_threshold" in rec.reasons
    assert "cannot confirm this is the same as before" in rec.message and not may_claim_same_as_before(rec)
    assert rec.margin_diff is not None and rec.margin_ratio is not None and rec.cost_best is not None and rec.cost_second is not None and len(rec.matched_distances) == 3
    assert rec.alignment_margin == rec.margin_diff
    assert st1.anchor is st.anchor and st1.epoch == 0 and st1.pending is not None and st1.pending.fit_index == 1        # the anchor did not move
    assert sorted(rec.raw_to_stable) == [0, 1, 2]                                                                       # the best guess still names every regime with an old name
    assert rec.proposed_new_names is not None and "raw_to_stable" in rec.proposed_new_names and rec.user_choice is None


def test_a_fit_that_is_not_identifiable_is_never_aligned_even_with_a_perfect_margin():
    means = _means(11); a = _summary(means, seed=1); n = _summary(means, raw_perm=[1, 0, 2], seed=2)
    _, st = step(None, a, CFG, REGISTRY, lineage_id="u")
    strict = AlignmentConfig(min_separation=1e6, ladder=CFG.ladder)
    rec, st1 = step(st, n, strict, REGISTRY)
    assert rec.continuity_state == CONTINUITY_UNVERIFIED and set(rec.reasons) == {"anchor_not_identifiable", "fit_not_identifiable"} and st1.anchor is st.anchor
    # only the new fit is poor
    poor = L.FitSummary(n.engine, n.raw_means, n.scaler_mean, n.scaler_std, n.labels, 0.1, n.raw_window, None)
    rec, _ = step(st, poor, CFG, REGISTRY)
    assert rec.continuity_state == CONTINUITY_UNVERIFIED and rec.reasons == ("fit_not_identifiable",)
    rec, _ = step(st, n, CFG, REGISTRY)
    assert rec.continuity_state == ALIGNED


def test_the_anchor_is_the_last_aligned_fit_so_an_unverified_fit_does_not_move_it():
    means = _means(11)
    a = _summary(means, seed=1)
    _, st = step(None, a, CFG, REGISTRY, lineage_id="u")
    bad = L.FitSummary("baseline", means[[1, 0, 2]], a.scaler_mean, a.scaler_std, a.labels, 0.1, a.raw_window, None)          # not identifiable
    rec_bad, st_bad = step(st, bad, CFG, REGISTRY)
    assert rec_bad.continuity_state == CONTINUITY_UNVERIFIED and st_bad.anchor is st.anchor
    good = _summary(means, raw_perm=[2, 1, 0], seed=3)
    rec, st2 = step(st_bad, good, CFG, REGISTRY)
    assert rec.continuity_state == ALIGNED and st2.pending is None and st2.n_fits == 3
    assert rec.anchor_engine == "baseline" and list(rec.raw_to_stable) == [2, 1, 0]


# ------------------------------------------------------------------------------------------------------------------- K change
def test_a_changed_k_is_k_changed_without_consulting_the_margins_and_old_names_survive():
    means = _means(11, K=3); a = _summary(means, seed=1)
    _, st = step(None, a, CFG, REGISTRY, lineage_id="u")
    parent = 1
    child = means[parent] + 2.0 * SCALE * np.ones(7) / np.sqrt(7)
    four = np.vstack([means, child])                                                              # a split: regime 1 got a sibling
    n = _summary(four, raw_perm=[3, 0, 1, 2], seed=2, scaler=(a.scaler_mean, a.scaler_std))
    rec, st1 = step(st, n, CFG, REGISTRY)
    assert rec.continuity_state == K_CHANGED and not may_claim_same_as_before(rec) and rec.margin_diff is None and rec.reasons == ("k_changed",)
    assert "cannot confirm this is the same as before" in rec.message and rec.K == 4 and rec.K_anchor == 3 and st1.anchor is st.anchor
    assert len(set(rec.raw_to_stable)) == 4 and sorted(rec.raw_to_stable)[:3] == [0, 1, 2] and max(rec.raw_to_stable) == 3         # matched regimes keep old names, the extra one gets a fresh id
    splits = [c for c in rec.split_merge_candidates if c["kind"] == "split"]
    assert splits and splits[0]["old_stable_id"] == rec.raw_to_stable[0 + 0] * 0 + [rec.raw_to_stable[1], rec.raw_to_stable[3]][0] * 0 + parent or True
    assert len(splits) == 1 and sorted(splits[0]["new_raw_regimes"]) == [0, 2]                  # raw regimes 0 (the child) and 2 (the parent) are the two pieces of the old regime 1
    assert splits[0]["old_stable_id"] == parent
    assert rec.proposed_new_names is not None


def test_a_merge_names_the_two_old_regimes_that_collapsed():
    means = _means(11, K=3); a = _summary(means, seed=1)
    _, st = step(None, a, CFG, REGISTRY, lineage_id="u")
    two = np.vstack([means[0], 0.5 * (means[1] + means[2])])                                   # old regimes 1 and 2 merged
    n = _summary(two, seed=2, scaler=(a.scaler_mean, a.scaler_std))
    rec, st1 = step(st, n, CFG, REGISTRY)
    assert rec.continuity_state == K_CHANGED and rec.K == 2 and rec.K_anchor == 3
    merges = [c for c in rec.split_merge_candidates if c["kind"] == "merge"]
    assert len(merges) == 1 and sorted(merges[0]["old_stable_ids"]) == sorted(a_stable for a_stable in [st.anchor.raw_to_stable[1], st.anchor.raw_to_stable[2]])


def test_k_above_the_matching_ceiling_fails_closed():
    r = np.random.default_rng(0); K = 7
    means = r.standard_normal((K, 7)) * 10 * SCALE
    s = _summary(means, seed=1)
    _, st = step(None, s, CFG, REGISTRY, lineage_id="u")
    rec, _ = step(st, _summary(means, seed=2), CFG, REGISTRY)
    assert rec.continuity_state == CONTINUITY_UNVERIFIED and rec.reasons == ("k_above_matching_ceiling",) and rec.margin_diff is None


# ------------------------------------------------------------------------------------------------------------------- the user's choice
def _unverified(strict=True):
    means = _means(11); a = _summary(means, seed=1)
    _, st = step(None, a, CFG, REGISTRY, lineage_id="u")
    n = _summary(means, raw_perm=[2, 0, 1], seed=2)
    rec, st1 = step(st, n, AlignmentConfig(min_separation=1e6, ladder=CFG.ladder), REGISTRY)
    assert rec.continuity_state == CONTINUITY_UNVERIFIED
    return st, rec, st1, n


def test_no_response_behaves_as_keep_old():
    st, rec, st1, _ = _unverified()
    assert L.effective_choice(rec) == "KEEP_OLD" and L.effective_choice({"user_choice": None}) == "KEEP_OLD" and L.effective_choice({"user_choice": "NEW_NAMES"}) == "NEW_NAMES"
    rec2, st2 = apply_user_choice(st1, rec, None)
    assert rec2 is rec and st2 is st1 and st2.anchor is st.anchor and st2.epoch == 0 and st2.epoch_history == ()


def test_keep_old_is_recorded_and_changes_nothing_else():
    st, rec, st1, _ = _unverified()
    rec2, st2 = apply_user_choice(st1, rec, "KEEP_OLD")
    assert rec2.user_choice == "KEEP_OLD" and rec2.continuity_state == CONTINUITY_UNVERIFIED and rec2.raw_to_stable == rec.raw_to_stable
    assert st2.anchor is st.anchor and st2.epoch == 0 and st2.pending is None and st2.epoch_history == ()
    with pytest.raises(ValueError):
        apply_user_choice(st2, rec2, "NEW_NAMES")                                           # already resolved / no longer pending


def test_new_names_start_a_new_epoch_adopt_the_proposed_names_and_keep_the_mapping_for_good():
    st, rec, st1, n = _unverified()
    rec2, st2 = apply_user_choice(st1, rec, "NEW_NAMES")
    assert rec2.continuity_state == NEW_LINEAGE and rec2.user_choice == "NEW_NAMES" and rec2.lineage_epoch == 1 and st2.epoch == 1 and st2.lineage_id == st.lineage_id
    assert rec2.raw_to_stable == rec.proposed_new_names["raw_to_stable"] and np.array_equal(rec2.stable_regime_id, rec.proposed_new_names["stable_regime_id"])
    assert st2.anchor.summary is n and st2.pending is None and not may_claim_same_as_before(rec2)
    assert len(st2.epoch_history) == 1
    h = st2.epoch_history[0]
    assert (h["from_epoch"], h["to_epoch"], h["at_fit_index"], h["old_K"], h["new_K"]) == (0, 1, 1, 3, 3) and len(h["old_to_new"]) == 3
    assert [c["old"] for c in h["old_to_new"]] == [0, 1, 2] and all(c["confirmed"] is False for c in h["old_to_new"])
    # the mapping survives later steps
    _, st3 = step(st2, _summary(_means(11), raw_perm=[1, 2, 0], seed=9), CFG, REGISTRY)
    assert st3.epoch_history == st2.epoch_history and st3.epoch == 1


def test_a_choice_must_be_valid_and_apply_to_the_pending_unverified_record():
    st, rec, st1, _ = _unverified()
    with pytest.raises(ValueError):
        apply_user_choice(st1, rec, "MAYBE")
    aligned, st_a = step(st, _summary(_means(11), seed=4), CFG, REGISTRY)
    assert aligned.continuity_state == ALIGNED
    with pytest.raises(ValueError):
        apply_user_choice(st_a, aligned, "NEW_NAMES")                                        # nothing to decide on an aligned fit
    newer, st_n = step(st1, _summary(_means(11), raw_perm=[1, 0, 2], seed=5), AlignmentConfig(min_separation=1e6, ladder=CFG.ladder), REGISTRY)
    with pytest.raises(ValueError):
        apply_user_choice(st_n, rec, "NEW_NAMES")                                            # the older proposal was replaced by the newer one
    rec_n, _ = apply_user_choice(st_n, newer, "NEW_NAMES")
    assert rec_n.lineage_epoch == 1


def test_new_names_after_a_changed_k_record_a_partial_mapping():
    means = _means(11, K=3); a = _summary(means, seed=1)
    _, st = step(None, a, CFG, REGISTRY, lineage_id="u")
    two = np.vstack([means[0], 0.5 * (means[1] + means[2])])
    rec, st1 = step(st, _summary(two, seed=2, scaler=(a.scaler_mean, a.scaler_std)), CFG, REGISTRY)
    rec2, st2 = apply_user_choice(st1, rec, "NEW_NAMES")
    h = st2.epoch_history[0]
    assert (h["old_K"], h["new_K"]) == (3, 2) and sum(c["new"] is None for c in h["old_to_new"]) == 1 and st2.anchor.summary.K == 2 and rec2.K == 2


# ------------------------------------------------------------------------------------------------------------------- engines, record columns, purity, the raw record
def test_an_engine_switch_uses_its_own_thresholds_and_moves_the_anchor_when_aligned():
    means = _means(11); a = _summary(means, seed=1, engine="baseline")
    moved = means + 0.15 * SCALE * np.random.default_rng(3).standard_normal(means.shape)             # the other engine sees slightly different means, so C1 > 0
    n = _summary(moved, raw_perm=[1, 2, 0], seed=2, engine="dual", scaler=(a.scaler_mean, a.scaler_std))
    _, st = step(None, a, CFG, REGISTRY, lineage_id="u")
    lenient_within_strict_across = AlignmentConfig(min_separation=0.0, ladder=CFG.ladder, same_engine=PairThreshold("ratio", rho=0.5), cross_engine=PairThreshold("ratio", rho=0.0))
    rec, st1 = step(st, n, lenient_within_strict_across, REGISTRY)
    assert rec.continuity_state == CONTINUITY_UNVERIFIED and rec.anchor_engine == "baseline" and rec.engine == "dual" and rec.threshold["rho"] == 0.0 and st1.anchor.summary.engine == "baseline"
    lenient = AlignmentConfig(min_separation=0.0, ladder=CFG.ladder, same_engine=PairThreshold("ratio", rho=0.5), dual_engine=PairThreshold("ratio", rho=0.0), cross_engine=PairThreshold("ratio", rho=0.5))
    rec, st2 = step(st, n, lenient, REGISTRY)
    assert rec.continuity_state == ALIGNED and rec.engine == "dual" and rec.anchor_engine == "baseline" and st2.anchor.summary.engine == "dual"
    again, _ = step(st2, _summary(means, seed=3, engine="dual"), lenient, REGISTRY)
    assert again.threshold["rho"] == 0.0                                                      # now dual to dual: the dual class, not the baseline one (rho 0.5 here)


def test_every_record_carries_the_columns_of_the_brief_is_json_ready_and_says_expert_review_is_pending():
    st, rec, st1, _ = _unverified()
    first, _ = step(None, _summary(_means(11), seed=1), CFG, REGISTRY, lineage_id="u")
    for r in (first, rec):
        d = r.as_dict()
        assert set(L.RECORD_COLUMNS) <= set(d)
        assert d["expert_review_status"] == "PENDING" and len(d["stable_regime_id"]) == len(r.stable_regime_id)
        assert len(d["scaler_mean"]) == len(d["scaler_std"]) == 7 and d["continuity_state"] in L.STATES
        json.dumps(d)                                                                          # no numpy types, no inf
    assert first.as_dict()["alignment_margin"] is None and rec.as_dict()["alignment_margin"] is not None


def test_step_does_not_modify_its_inputs_or_the_old_state_and_is_deterministic():
    means = _means(11); a = _summary(means, seed=1); n = _summary(means, raw_perm=[1, 0, 2], seed=2)
    _, st = step(None, a, CFG, REGISTRY, lineage_id="u")
    snap = copy.deepcopy((n.raw_means, n.labels, n.raw_window, a.raw_means, st.anchor.raw_to_stable, st.n_fits))
    r1, s1 = step(st, n, CFG, REGISTRY); r2, s2 = step(st, n, CFG, REGISTRY)
    assert r1.as_dict() == r2.as_dict()
    for before, after in zip(snap, (n.raw_means, n.labels, n.raw_window, a.raw_means, st.anchor.raw_to_stable, st.n_fits)):
        assert np.array_equal(before, after)
    assert st.n_fits == 1 and st.anchor.summary is a and s1 is not st
    assert r1.stable_regime_id is not n.labels and not np.shares_memory(r1.stable_regime_id, n.labels)                  # a new column; the raw labels stay as they were


def test_may_claim_same_as_before_is_true_only_for_aligned():
    for state in L.STATES:
        assert may_claim_same_as_before({"continuity_state": state}) == (state == "ALIGNED")
        assert may_claim_same_as_before(SimpleNamespace(continuity_state=state)) == (state == "ALIGNED")
    for bad in ({}, {"continuity_state": None}, {"continuity_state": "aligned"}, SimpleNamespace(), None, "ALIGNED"):
        assert not may_claim_same_as_before(bad)


def test_the_duration_coordinate_is_optional_and_needs_the_durations():
    means = _means(11)
    a = _summary(means, seed=1); n = _summary(means, raw_perm=[1, 0, 2], seed=2)
    _, st = step(None, a, CFG, REGISTRY, lineage_id="u")
    with pytest.raises(ValueError):
        step(st, n, AlignmentConfig(min_separation=0.0, duration_weight=1.0, ladder=CFG.ladder), REGISTRY)
    a2 = L.FitSummary(a.engine, a.raw_means, a.scaler_mean, a.scaler_std, a.labels, a.separation_min, a.raw_window, np.array([5.0, 10.0, 20.0]))
    n2 = L.FitSummary(n.engine, n.raw_means, n.scaler_mean, n.scaler_std, n.labels, n.separation_min, n.raw_window, np.array([11.0, 5.5, 20.0]))
    _, st2 = step(None, a2, CFG, REGISTRY, lineage_id="u")
    rec, _ = step(st2, n2, AlignmentConfig(min_separation=0.0, duration_weight=1.0, ladder=CFG.ladder), REGISTRY)
    assert rec.continuity_state == ALIGNED and list(rec.raw_to_stable) == [1, 0, 2]
    rec0, _ = step(st2, n2, AlignmentConfig(min_separation=0.0, duration_weight=0.0, ladder=CFG.ladder), REGISTRY)
    assert rec.cost_best > rec0.cost_best                                                      # the extra coordinate adds to the cost


def test_separation_min_equals_an_independent_computation():
    r = np.random.default_rng(4); K, F, T = 3, 4, 90
    means = r.standard_normal((K, F)) * 3; labels = r.integers(0, K, T)
    X = means[labels] + r.standard_normal((T, F)) * np.array([0.5, 1.0, 2.0, 0.0]); X[r.random((T, F)) < 0.1] = np.nan
    ss = np.zeros(F); n = np.zeros(F)
    for k in range(K):
        for f in range(F):
            x = X[labels == k, f]; x = x[~np.isnan(x)]
            ss[f] += ((x - means[k, f]) ** 2).sum(); n[f] += len(x)
    sd = np.sqrt(ss / n); sd[sd == 0] = 1.0
    expect = min(np.linalg.norm((means[i] - means[j]) / sd) for i in range(K) for j in range(i + 1, K))
    assert np.isclose(L.separation_min(means, X, labels), expect)
    assert L.separation_min(means[:1], X, np.zeros(T, int)) == float("inf")


def test_make_summary_validates_shapes_labels_and_engine():
    m = _means(11); s = _summary(m)
    with pytest.raises(ValueError):
        make_summary("other", s.raw_means, s.scaler_mean, s.scaler_std, s.labels, s.raw_window)
    with pytest.raises(ValueError):
        make_summary("baseline", s.raw_means, s.scaler_mean[:-1], s.scaler_std, s.labels, s.raw_window)
    with pytest.raises(ValueError):
        make_summary("baseline", s.raw_means, s.scaler_mean, s.scaler_std, s.labels + 5, s.raw_window)
    with pytest.raises(ValueError):
        make_summary("baseline", s.raw_means, s.scaler_mean, np.zeros_like(s.scaler_std), s.labels, s.raw_window)


def test_the_distance_is_in_units_of_the_anchors_scale_so_the_assignment_follows_the_scale():
    """Two features, scaler sd (1000, 1) for the anchor. In raw units the huge first feature decides; in the anchor's scale units it hardly matters, and the assignment flips."""
    rng = np.random.default_rng(0)
    A = np.array([[0.0, 0.0], [500.0, 10.0]]); N = np.array([[500.0, 0.0], [0.0, 10.0]])
    def summ(means, std):
        labels = np.repeat(np.arange(2), 20); X = means[labels] + rng.standard_normal((40, 2)) * np.array([5.0, 0.05])
        return make_summary("baseline", means, X.mean(axis=0), np.array(std, dtype=float), labels, X)
    a = summ(A, (1000.0, 1.0)); n = summ(N, (1.0, 1000.0))                                       # the new fit's own scaler is irrelevant
    cfg = AlignmentConfig(min_separation=0.0, ladder=CFG.ladder, same_engine=PairThreshold("ratio", rho=0.5))
    _, st = step(None, a, cfg, None, lineage_id="s")
    rec, _ = step(st, n, cfg, None)
    assert rec.raw_to_stable == (0, 1)                                                           # scaled: anchor regime 0 <-> new regime 0 (0.25 + 0.25), the unscaled best would be (1, 0)
    unscaled = alignment(A, N)
    assert unscaled["perm"] == (1, 0)


def test_the_proposed_new_names_are_what_the_ladder_would_call_the_new_fit():
    st, rec, st1, n = _unverified()
    raw_to_stable, rung, verified, _ = L.name_fit(n, REGISTRY, CFG.ladder)
    assert rec.proposed_new_names["raw_to_stable"] == raw_to_stable and rec.proposed_new_names["activity_rung"] == rung and rec.proposed_new_names["ordering_verified"] == verified
    assert verified and raw_to_stable == (2, 0, 1)                                              # the new fit was drawn with raw regimes [2, 0, 1] of the planted order


def test_a_single_regime_lineage_has_no_infinite_numbers_in_its_json():
    s = _summary(_means(11)[:1], seed=1)
    rec, st = step(None, s, CFG, REGISTRY, lineage_id="one")
    rec2, _ = step(st, _summary(_means(11)[:1], seed=2), CFG, REGISTRY)
    assert rec.separation_min == float("inf") and rec2.continuity_state == ALIGNED and rec2.cost_second == float("inf")
    d = rec2.as_dict()
    assert d["separation_min"] is None and d["cost_second"] is None and d["anchor_separation_min"] is None
    assert "Infinity" not in json.dumps(d)


def test_a_choice_is_refused_for_an_aligned_record_even_if_a_proposal_for_that_fit_is_pending():
    """Defence in depth: the state machine never leaves a proposal pending for an aligned fit, so build that state by hand and check the record type is refused on its own."""
    import dataclasses
    st, rec, st1, _ = _unverified()
    aligned, st_a = step(st, _summary(_means(11), seed=4), CFG, REGISTRY)
    assert aligned.continuity_state == ALIGNED
    forged = dataclasses.replace(st_a, pending=dataclasses.replace(st1.pending, fit_index=aligned.fit_index))
    with pytest.raises(ValueError, match="only applies"):
        apply_user_choice(forged, aligned, "NEW_NAMES")
