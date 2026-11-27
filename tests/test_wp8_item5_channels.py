"""WP8 Item 5 (owner decision A6): two comparison channels. Channel 1 is physical (regime means in real units, readable through the feature registry); channel 2 is an OPTIONAL
contextual profile per regime that the caller supplies (synthetic descriptors only here; no topic or scene extraction exists). Mutation check: integration_report/wp8/scripts/mutation_item5.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backbone.hssm import lineage as L
from backbone.hssm.lineage import (ALIGNED, CONTINUITY_UNVERIFIED, K_CHANGED, AlignmentConfig, ChannelConfig, PairThreshold, make_summary, may_claim_same_as_before, step)
from backbone.hssm.ordering import LadderConfig
from data.synthetic_identity import REGISTRY

ROOT = Path(__file__).resolve().parents[1]
CTX = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])                     # three clearly different contexts (topic or scene proportions)
M3 = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 6.0, 0.0]])                      # three clearly different physical regimes


def _s(means, ctx=None, seed=0, T=90, engine="baseline"):
    rng = np.random.default_rng(seed); K, F = means.shape; T = K * (T // K)
    labels = rng.permutation(np.repeat(np.arange(K), T // K))
    X = means[labels] + 0.1 * rng.standard_normal((T, F))
    return make_summary(engine, means, np.zeros(F), np.ones(F), labels, X, context_profiles=ctx)


def _cfg(w1=1.0, w2=1.0, rho=0.5, ctx_rho=0.5):
    return AlignmentConfig(min_separation=0.0, ladder=LadderConfig(n_bootstrap=20), same_engine=PairThreshold("ratio", rho=rho), cross_engine=PairThreshold("ratio", rho=rho),
                           channels=ChannelConfig(w1, w2, PairThreshold("ratio", rho=ctx_rho)))


def _pair(a_means, a_ctx, n_means, n_ctx, cfg):
    _, st = step(None, _s(a_means, a_ctx, seed=1), cfg, None, lineage_id="u")
    return step(st, _s(n_means, n_ctx, seed=2), cfg, None)


# ------------------------------------------------------------------------------------------------------------------- the distance and the interface
def test_hellinger_matches_the_formula_is_symmetric_zero_for_equal_and_one_for_disjoint():
    r = np.random.default_rng(0)
    P = r.dirichlet(np.ones(5), size=4); Q = r.dirichlet(np.ones(5), size=3)
    D = L.hellinger2_matrix(P, Q)
    for i in range(4):
        for j in range(3):
            assert np.isclose(D[i, j], 1.0 - np.sqrt(P[i] * Q[j]).sum())                       # 0.5 sum (sqrt p - sqrt q)^2 = 1 - sum sqrt(p q)
    assert np.allclose(L.hellinger2_matrix(P, Q), L.hellinger2_matrix(Q, P).T) and np.allclose(np.diag(L.hellinger2_matrix(P, P)), 0.0)
    assert np.isclose(L.hellinger2_matrix(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))[0, 0], 1.0)


def test_context_profiles_are_validated():
    ok = L.validate_context_profiles(CTX, 3)
    assert ok.shape == (3, 3)
    for bad in (CTX[:2], CTX * 2.0, -CTX, np.where(np.eye(3) > 0, np.nan, CTX), CTX[:, :0], CTX.ravel(), np.array([[1.5, -0.5, 0.0], [0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])):     # the last sums to 1 but is negative
        with pytest.raises(ValueError):
            L.validate_context_profiles(bad, 3)
    with pytest.raises(ValueError):
        make_summary("baseline", M3, np.zeros(3), np.ones(3), np.zeros(30, int), np.zeros((30, 3)), context_profiles=CTX * 2)


def test_daily_descriptors_are_aggregated_per_regime_from_the_decoded_days():
    r = np.random.default_rng(3); T, C, K = 60, 4, 3
    lab = np.tile(np.arange(K), T // K)
    for D in (r.dirichlet(np.ones(C), size=T), r.integers(0, 9, size=(T, C)).astype(float) + 1.0):              # daily proportions, or daily counts that do not sum to 1
        D = D.copy(); D[5] = np.nan                                                                              # a day without descriptor is skipped
        out = L.context_profiles_from_days(D, lab, K)
        for k in range(K):
            rows = D[(lab == k) & ~np.isnan(D).any(axis=1)]
            assert np.allclose(out[k], rows.mean(axis=0) / rows.mean(axis=0).sum()) and np.isclose(out[k].sum(), 1.0)
    with pytest.raises(ValueError, match="no day with a descriptor"):
        L.context_profiles_from_days(D[lab != 2], lab[lab != 2], 3)                                             # regime 2 has no day
    with pytest.raises(ValueError):
        L.context_profiles_from_days(D, lab + 5, K)


def test_channel_one_is_readable_in_physical_units_through_the_registry():
    means = np.arange(14, dtype=float).reshape(2, 7)
    s = make_summary("baseline", means, np.zeros(7), np.ones(7), np.zeros(14, int), np.random.default_rng(0).standard_normal((14, 7)))
    table = L.regime_table(s, REGISTRY)
    assert len(table) == 2 and list(table[0]) == [f.name for f in REGISTRY]
    assert table[1]["steps"] == dict(value=7.0, unit="steps/day", source="movement") and table[1]["hrv"]["unit"] == "ms" and table[1]["pitch"]["source"] == "voice"
    with pytest.raises(ValueError):
        L.regime_table(s, REGISTRY[:3])


def test_channel_weights_are_validated():
    for kw in (dict(w1=-1.0), dict(w2=-0.1), dict(w1=0.0, w2=0.0)):
        with pytest.raises(ValueError):
            ChannelConfig(**kw)
    assert ChannelConfig().w2 == 0.0                                                            # the contextual channel is off by default


# ------------------------------------------------------------------------------------------------------------------- off by default, falls back when the context is missing
def test_with_the_default_weight_the_context_is_ignored_bit_for_bit():
    perm = [2, 0, 1]
    with_ctx, _ = _pair(M3, CTX, M3[perm], CTX[::-1], AlignmentConfig(min_separation=0.0, ladder=LadderConfig(n_bootstrap=20)))     # misleading context, w2 = 0
    without, _ = _pair(M3, None, M3[perm], None, AlignmentConfig(min_separation=0.0, ladder=LadderConfig(n_bootstrap=20)))
    assert with_ctx.as_dict() == without.as_dict() and with_ctx.channels_used == ("physical",) and with_ctx.channel_detail is None


def test_a_missing_profile_falls_back_to_channel_one_and_says_so():
    rec, _ = _pair(M3, CTX, M3[[1, 0, 2]], None, _cfg())
    assert rec.continuity_state == ALIGNED and rec.channels_used == ("physical",) and "missing" in rec.channel_detail["context"]
    rec, _ = _pair(M3, None, M3[[1, 0, 2]], CTX, _cfg())
    assert rec.channels_used == ("physical",)


def test_the_two_fits_must_use_the_same_context_categories():
    # WP10B F3 (owner amendment, matching stage): the mismatch used to raise a ValueError out of `step`; it is now a crashed assessment: CONTINUITY_UNVERIFIED with `assessment_crashed`, never ALIGNED (tests/test_wp10b_f3_fail_closed.py)
    r, _ = _pair(M3, CTX, M3, np.array([[0.5, 0.5], [0.2, 0.8], [0.9, 0.1]]), _cfg())
    assert r.continuity_state == CONTINUITY_UNVERIFIED and "assessment_crashed" in r.reasons and not may_claim_same_as_before(r)


# ------------------------------------------------------------------------------------------------------------------- one channel decisive, the other not (both directions)
SAME = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [6.0, 6.0, 6.0]])                              # regimes 0 and 1 cannot be told apart by their physical means
FLAT = np.full((3, 3), 1.0 / 3.0)                                                                   # a context that says nothing


def test_context_decides_when_the_physical_channel_cannot():
    perm = [1, 0, 2]
    alone, _ = _pair(SAME, CTX, SAME[perm], CTX[perm], _cfg(w1=1.0, w2=0.0))
    assert alone.continuity_state == CONTINUITY_UNVERIFIED and "margin_below_threshold" in alone.reasons                # channel 1 alone: a tie
    both, st = _pair(SAME, CTX, SAME[perm], CTX[perm], _cfg())
    assert both.continuity_state == ALIGNED and may_claim_same_as_before(both) and list(both.raw_to_stable) == [1, 0, 2] and both.channels_used == ("physical", "context")
    assert both.channel_detail["context"]["confident"] and not both.channel_detail["physical"]["confident"] and not both.channel_detail["disagree"]
    ignores_physical, _ = _pair(M3, FLAT, M3[perm], FLAT, _cfg(w1=0.0, w2=1.0))
    assert ignores_physical.continuity_state == CONTINUITY_UNVERIFIED                                                    # context alone cannot, here


def test_the_physical_channel_decides_when_the_context_says_nothing():
    perm = [2, 0, 1]
    rec, _ = _pair(M3, FLAT, M3[perm], FLAT, _cfg())
    assert rec.continuity_state == ALIGNED and list(rec.raw_to_stable) == [2, 0, 1] and not rec.channel_detail["context"]["confident"]
    ignores_physical, _ = _pair(M3, FLAT, M3[perm], FLAT, _cfg(w1=0.0, w2=1.0))
    assert ignores_physical.continuity_state == CONTINUITY_UNVERIFIED
    ignores_context, _ = _pair(M3, FLAT, M3[perm], FLAT, _cfg(w1=1.0, w2=0.0))
    assert ignores_context.continuity_state == ALIGNED


# ------------------------------------------------------------------------------------------------------------------- the adversarial case: the channels disagree
def test_two_confident_channels_that_disagree_give_continuity_unverified_and_keep_the_old_names():
    perm = [1, 0, 2]
    rec, st = _pair(M3, CTX, M3[perm], CTX, _cfg())                    # physical: raw regime 0 looks like anchor 1; context: raw regime 0 has anchor 0's profile
    d = rec.channel_detail
    assert d["physical"]["confident"] and d["context"]["confident"] and d["physical"]["perm"] != d["context"]["perm"] and d["disagree"]
    assert rec.continuity_state == CONTINUITY_UNVERIFIED and "channels_disagree" in rec.reasons and not may_claim_same_as_before(rec)
    assert "cannot confirm this is the same as before" in rec.message and st.anchor.fit_index == 0 and rec.proposed_new_names is not None
    agree, _ = _pair(M3, CTX, M3[perm], CTX[perm], _cfg())
    assert agree.continuity_state == ALIGNED and not agree.channel_detail["disagree"]


@pytest.mark.parametrize("w1,w2", [(1.0, 0.01), (0.01, 1.0), (1.0, 1.0), (5.0, 1.0)])
def test_disagreement_between_confident_channels_is_flagged_whatever_the_weights(w1, w2):
    rec, _ = _pair(M3, CTX, M3[[1, 0, 2]], CTX, _cfg(w1=w1, w2=w2))
    assert rec.continuity_state == CONTINUITY_UNVERIFIED and "channels_disagree" in rec.reasons


def test_a_channel_that_is_not_confident_cannot_veto_the_other():
    # physical: confident (identity). context: the new profiles of regimes 0 and 1 sit almost midway between the anchor's two profiles and lean only slightly towards the swap,
    # so the context alone prefers the swap but its margin is too small to be confident.
    l0, l1, l2 = np.array([0.5, 0.3, 0.2]), np.array([0.3, 0.5, 0.2]), np.array([0.1, 0.1, 0.8])
    mid, delta = 0.5 * (l0 + l1), 0.5 * (l1 - l0)
    anchor_ctx = np.vstack([l0, l1, l2]); new_ctx = np.vstack([mid + 0.1 * delta, mid - 0.1 * delta, l2])
    rec, _ = _pair(M3, anchor_ctx, M3, new_ctx, _cfg(w1=1.0, w2=0.2))
    d = rec.channel_detail
    assert d["physical"]["confident"] and not d["context"]["confident"] and d["physical"]["perm"] != d["context"]["perm"] and not d["disagree"]
    assert rec.continuity_state == ALIGNED and list(rec.raw_to_stable) == [0, 1, 2]


def test_the_weights_decide_the_combined_assignment_when_only_one_channel_is_confident():
    a_means = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0]])
    n_means = np.array([[0.55, 0.0], [0.45, 0.0], [10.0, 10.0]])                                  # physically the new regimes 0 and 1 lean towards a swap, but only weakly
    rec_phys, _ = _pair(a_means, CTX, n_means, CTX, _cfg(w1=1.0, w2=0.0005))
    rec_ctx, _ = _pair(a_means, CTX, n_means, CTX, _cfg(w1=0.001, w2=1.0))
    assert not rec_phys.channel_detail["physical"]["confident"] and rec_phys.channel_detail["context"]["confident"]
    assert list(rec_phys.raw_to_stable) == [1, 0, 2] and list(rec_ctx.raw_to_stable) == [0, 1, 2]     # physical-dominated follows the weak physical lean, context-dominated the context


def test_a_changed_k_stays_k_changed_with_a_context_channel():
    four = np.vstack([M3, [6.0, 6.0, 0.0]]); ctx4 = np.vstack([CTX * 0.9 + 0.1 / 3, [0.3, 0.3, 0.4]]); ctx4 = ctx4 / ctx4.sum(axis=1, keepdims=True)
    rec, st = _pair(M3, CTX, four, ctx4, _cfg())
    assert rec.continuity_state == K_CHANGED and rec.channels_used == ("physical",) and not may_claim_same_as_before(rec)


def test_the_record_with_two_channels_is_json_ready_and_expert_review_stays_pending():
    rec, _ = _pair(M3, CTX, M3[[1, 0, 2]], CTX[[1, 0, 2]], _cfg())
    d = rec.as_dict()
    json.dumps(d)
    assert d["channels_used"] == ["physical", "context"] and d["channel_detail"]["weights"] == [1.0, 1.0] and d["expert_review_status"] == "PENDING"
    assert set(d["channel_detail"]) == {"weights", "scales", "physical", "context", "disagree"}


def test_the_combined_cost_uses_the_declared_scales():
    rec, _ = _pair(M3, CTX, M3[[1, 0, 2]], CTX[[1, 0, 2]], _cfg(w1=2.0, w2=3.0))
    s1 = np.mean([((M3[i] - M3[j]) ** 2).sum() for i in range(3) for j in range(3) if i != j])
    s2 = np.mean([0.5 * ((np.sqrt(CTX[i]) - np.sqrt(CTX[j])) ** 2).sum() for i in range(3) for j in range(3) if i != j])
    assert np.allclose(rec.channel_detail["scales"], (s1, s2))
    assert np.isclose(rec.cost_best, 0.0, atol=1e-12)                                          # the new fit is the old one renamed: every matched distance is 0


def test_the_contract_states_that_contextual_descriptors_are_sensitive_and_must_be_consented_and_aggregated():
    doc = (ROOT / "docs" / "INTEGRATION_CONTRACT_HSSM.md").read_text()
    sec = doc[doc.index("### 9.10"):]
    for phrase in ("contextual descriptors are sensitive", "consented", "aggregated", "non-identifying", "no topic or scene extraction"):
        assert phrase in sec.lower(), phrase


def test_each_channel_is_divided_by_the_anchors_own_inter_regime_distance_before_the_weights_apply():
    """Physical: weakly prefers the swap of regimes 0 and 1 (gap 2.4 in raw squared units, 0.2 after dividing by s1 = 12). Context: strongly prefers the identity (gap 0.30 raw, 1.18 after
    dividing by s2 = 0.258). With w1 = 1, w2 = 0.4 the scaled combination follows the context; without the scale of channel 1 the physical gap would win, without the scale of channel 2 the physical gap would win too."""
    a = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]]); n = np.array([[1.7, 0.0], [1.3, 0.0], [0.0, 3.0]])
    ctx = np.vstack([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
    rec, _ = _pair(a, ctx, n, ctx, _cfg(w1=1.0, w2=0.4))
    assert np.isclose(rec.channel_detail["scales"][0], 12.0) and np.isclose(rec.channel_detail["scales"][1], 0.258, atol=1e-3)
    assert not rec.channel_detail["physical"]["confident"] and rec.channel_detail["context"]["confident"]
    assert list(rec.raw_to_stable) == [0, 1, 2]


def test_the_contextual_channel_is_judged_by_its_own_threshold():
    l0, l1, l2 = np.array([0.5, 0.3, 0.2]), np.array([0.3, 0.5, 0.2]), np.array([0.1, 0.1, 0.8])
    mid, delta = 0.5 * (l0 + l1), 0.5 * (l1 - l0)
    anchor_ctx = np.vstack([l0, l1, l2]); new_ctx = np.vstack([mid + 0.3 * delta, mid - 0.3 * delta, l2])             # leans to the swap with a margin ratio of about 0.29
    loose, _ = _pair(M3, anchor_ctx, M3, new_ctx, _cfg(w1=1.0, w2=0.05, rho=0.5, ctx_rho=0.5))
    strict, _ = _pair(M3, anchor_ctx, M3, new_ctx, _cfg(w1=1.0, w2=0.05, rho=0.5, ctx_rho=0.05))
    assert loose.channel_detail["context"]["confident"] and loose.channel_detail["disagree"] and loose.continuity_state == CONTINUITY_UNVERIFIED
    assert not strict.channel_detail["context"]["confident"] and not strict.channel_detail["disagree"] and strict.continuity_state == ALIGNED


def test_the_anchors_profiles_follow_its_stable_names_not_its_raw_order():
    import dataclasses
    cfg = _cfg()
    a = _s(SAME, CTX, seed=1)
    anchor_names = (1, 2, 0)                                                         # anchor raw regime 0 is called 1, raw 1 is called 2, raw 2 is called 0
    st = L.LineageState("u", 0, L.Anchor(a, anchor_names, 0, 0, False), 1)
    raw_perm = [2, 0, 1]                                                             # new raw regime r is anchor raw regime raw_perm[r]
    n = _s(SAME[raw_perm], CTX[raw_perm], seed=2)
    rec, _ = step(st, n, cfg, None)
    assert rec.continuity_state == ALIGNED and list(rec.raw_to_stable) == [anchor_names[raw_perm[r]] for r in range(3)]


def test_the_channel_information_survives_the_choice_of_new_names():
    rec, st = _pair(M3, CTX, M3[[1, 0, 2]], CTX, _cfg())
    assert rec.continuity_state == CONTINUITY_UNVERIFIED
    _, st0 = step(None, _s(M3, CTX, seed=1), _cfg(), None, lineage_id="u")
    rec, st1 = step(st0, _s(M3[[1, 0, 2]], CTX, seed=2), _cfg(), None)
    chosen, _ = L.apply_user_choice(st1, rec, "NEW_NAMES")
    assert chosen.channels_used == ("physical", "context") and chosen.channel_detail == rec.channel_detail and chosen.channel_detail["disagree"]


def test_a_context_channel_cannot_rescue_a_class_of_pair_without_a_calibrated_threshold():
    """After the Item 4 freeze the dual-to-dual class has no calibrated threshold: two channels that agree, with wide margins, still do not give ALIGNED (fail closed)."""
    from dataclasses import replace
    cfg = AlignmentConfig(min_separation=0.0, ladder=LadderConfig(n_bootstrap=20), channels=ChannelConfig(1.0, 1.0, PairThreshold("ratio", rho=0.5)))        # the frozen dual class
    perm = [2, 0, 1]
    _, st = step(None, _s(M3, CTX, seed=1, engine="dual"), cfg, None, lineage_id="u")
    new = _s(M3[perm], CTX[perm], seed=2, engine="dual")
    rec, _ = step(st, new, cfg, None)
    assert rec.continuity_state == CONTINUITY_UNVERIFIED and "threshold_not_calibrated" in rec.reasons and "channels_disagree" not in rec.reasons
    assert not may_claim_same_as_before(rec) and rec.channels_used == ("physical", "context")
    control, _ = step(st, new, replace(cfg, dual_engine=PairThreshold("ratio", rho=0.5)), None)
    assert control.continuity_state == ALIGNED and control.reasons == ()
