"""WP9 A1: multi-start consensus (FIT_UNSTABLE). Design: tests/PREDECLARED_CRITERIA_WP9.md section 6. Mutation check: integration_report/wp9/scripts/mutation_a1.py (>= 15 mutants)."""
from __future__ import annotations

import contextlib
import dataclasses
import io
import logging
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from backbone.hssm import fit_stability as FS
from backbone.hssm import lineage as L
from backbone.hssm.config import FIT_CONSENSUS_M, FIT_CONSENSUS_TAU
from backbone.hssm.fitting import _restart_summary, adjusted_rand_index, fit_hssm
from backbone.hssm.lineage import ALIGNED, CONTINUITY_UNVERIFIED, K_CHANGED, NEW_LINEAGE, AlignmentConfig, apply_user_choice, make_summary, may_claim_same_as_before, step
from backbone.hssm.model import FitState
from backbone.hssm.ordering import LadderConfig
from data.synthetic_identity import REGISTRY, IdentitySpec, draw_user, generate_identity_stream


def rec(ll, labels, collapsed=False):
    return dict(ll=ll, labels=np.asarray(labels), collapsed=collapsed)


A = [0] * 20 + [1] * 20 + [2] * 20
B = [0] * 20 + [1] * 19 + [2] * 21                      # one day moved: ARI about 0.97 (below 0.99)
C = [2] * 20 + [0] * 20 + [1] * 20                      # the same partition as A under another labelling: ARI 1.0
ARI = adjusted_rand_index


def fake_ari(table):
    """An ARI that returns table[(i, j)] by the integer tags of the label vectors (labels = [tag])."""
    return lambda a, b: table[(int(a[0]), int(b[0]))]


# ------------------------------------------------------------------------------------------------------------------- the rule on records
def test_constants_are_the_declared_ones():
    assert FIT_CONSENSUS_M == 3 and FIT_CONSENSUS_TAU == 0.99
    assert ARI(A, C) == 1.0 and 0.9 < ARI(A, B) < 0.99


def test_three_identical_partitions_under_any_labelling_are_stable():
    out = FS.assess_consensus([rec(-10, A), rec(-11, C), rec(-12, A)], ARI)
    assert out["fit_unstable"] is False and out["fit_unstable_reason"] is None and out["consensus_min_ari"] == pytest.approx(1.0)
    assert set(out) == set(FS.KEYS) and out["n_healthy_restarts"] == 3 and out["consensus_m"] == 3 and out["consensus_tau"] == 0.99


def test_a_disagreeing_restart_among_the_best_three_makes_the_fit_unstable():
    out = FS.assess_consensus([rec(-10, A), rec(-11, A), rec(-12, B)], ARI)
    assert out["fit_unstable"] is True and out["fit_unstable_reason"] == FS.REASON_DISAGREE and out["consensus_min_ari"] == pytest.approx(ARI(A, B))


def test_only_the_best_three_by_log_likelihood_count_not_the_first_three_in_restart_order():
    recs = [rec(-50, B), rec(-10, A), rec(-11, A), rec(-12, A), rec(-60, B)]                       # the two disagreeing restarts are the two worst
    assert FS.assess_consensus(recs, ARI)["fit_unstable"] is False
    recs = [rec(-10, A), rec(-11, A), rec(-50, A), rec(-12, B)]                                    # the fourth restart is the third best
    assert FS.assess_consensus(recs, ARI)["fit_unstable"] is True


def test_collapsed_restarts_are_ignored_even_when_they_have_the_best_likelihood():
    recs = [rec(-1, B, collapsed=True), rec(-10, A), rec(-11, A), rec(-12, A)]
    out = FS.assess_consensus(recs, ARI)
    assert out["fit_unstable"] is False and out["n_healthy_restarts"] == 3


def test_fewer_than_three_healthy_restarts_is_unstable_with_its_own_reason():
    out = FS.assess_consensus([rec(-10, A), rec(-11, A), rec(-5, A, collapsed=True)], ARI)
    assert out["fit_unstable"] is True and out["fit_unstable_reason"] == FS.REASON_TOO_FEW and out["consensus_min_ari"] is None and out["n_healthy_restarts"] == 2
    assert FS.assess_consensus([], ARI)["fit_unstable"] is True


def test_the_smallest_pairwise_ari_decides_not_the_agreement_with_the_best_restart():
    table = {(0, 1): 1.0, (0, 2): 1.0, (1, 2): 0.5, (1, 0): 1.0, (2, 0): 1.0, (2, 1): 0.5}        # the second and third best agree with the best but not with each other
    recs = [rec(-10, [0]), rec(-11, [1]), rec(-12, [2])]
    out = FS.assess_consensus(recs, fake_ari(table))
    assert out["fit_unstable"] is True and out["consensus_min_ari"] == 0.5


def test_the_level_is_inclusive_at_tau_and_exclusive_just_below():
    recs = [rec(-10, [0]), rec(-11, [1]), rec(-12, [2])]
    ok = {(i, j): 0.99 for i in range(3) for j in range(3)}
    assert FS.assess_consensus(recs, fake_ari(ok))["fit_unstable"] is False
    below = {(i, j): 0.99 for i in range(3) for j in range(3)}
    below[(0, 2)] = below[(2, 0)] = 0.9899999
    assert FS.assess_consensus(recs, fake_ari(below))["fit_unstable"] is True


def test_ties_in_log_likelihood_go_to_the_lower_restart_index():
    # four restarts with the same log-likelihood: the first three (by index) are compared, the fourth (a different partition) is not
    recs = [rec(-10, A), rec(-10, A), rec(-10, A), rec(-10, B)]
    assert FS.assess_consensus(recs, ARI)["fit_unstable"] is False
    recs = [rec(-10, B), rec(-10, A), rec(-10, A), rec(-10, A)]
    assert FS.assess_consensus(recs, ARI)["fit_unstable"] is True


def test_parameters_are_validated_and_custom_values_are_used():
    with pytest.raises(ValueError):
        FS.assess_consensus([], ARI, m=1)
    with pytest.raises(ValueError):
        FS.assess_consensus([], ARI, tau=0.0)
    with pytest.raises(ValueError):
        FS.assess_consensus([], ARI, tau=1.01)
    recs = [rec(-10, A), rec(-11, A), rec(-12, B)]
    assert FS.assess_consensus(recs, ARI, m=2)["fit_unstable"] is False                           # only the best two compared
    assert FS.assess_consensus(recs, ARI, m=3, tau=0.9)["fit_unstable"] is False                  # ARI(A, B) is above 0.9
    assert FS.assess_consensus(recs, ARI, m=3, tau=0.9)["consensus_tau"] == 0.9


def test_not_assessed_keys_are_the_fail_closed_ones():
    out = FS.not_assessed()
    assert set(out) == set(FS.KEYS) and out["fit_unstable"] is None and out["fit_unstable_reason"] == FS.REASON_NOT_ASSESSED


# ------------------------------------------------------------------------------------------------------------------- the additive restart summary
def test_restart_summary_gains_only_the_new_keys_and_keeps_the_old_ones():
    sel = rec(-10, A)
    recs = [sel, rec(-11, A), rec(-12, A), rec(-15, B)]
    s = _restart_summary([{"fit_state": FitState.CONVERGED}] * 4, recs, sel)
    old = {"n_restarts", "n_converged", "n_rejected_update", "n_max_iter", "n_collapsed_converged", "n_within_epsilon", "epsilon_nats", "ari_to_selected", "ari_median", "ari_min",
           "n_other_same_solution", "same_solution_ari", "gap_to_next_distinct"}
    assert set(s) == old | set(FS.KEYS)
    assert s["fit_unstable"] is False and s["n_restarts"] == 4 and s["ari_to_selected"][2] == pytest.approx(ARI(A, B))


def test_a_collapsed_selected_fit_is_not_assessed():
    sel = rec(-10, A, collapsed=True)
    s = _restart_summary([{"fit_state": FitState.CONVERGED}] * 2, [sel, rec(-11, A, collapsed=True)], sel)
    assert s["fit_unstable"] is None and s["fit_unstable_reason"] == FS.REASON_NOT_ASSESSED


# ------------------------------------------------------------------------------------------------------------------- the lineage
CFG = AlignmentConfig(min_separation=1.0, ladder=LadderConfig(n_bootstrap=40, min_gap=0.5))


def _means(seed, K=3, sep=2.5):
    return draw_user(seed, IdentitySpec(n_regimes=K, sep_lo=sep, sep_hi=sep, offset_sd=0.3))["means"]


def _summary(means, seed=0, T=240, noise=0.35, **kw):
    from data.synthetic_identity import SCALE
    rng = np.random.default_rng(seed)
    K, F = means.shape
    labels = rng.permutation(np.repeat(np.arange(K), T // K)[:T])
    window = means[labels] + noise * SCALE * rng.standard_normal((len(labels), F))
    return make_summary("baseline", means, window.mean(axis=0), window.std(axis=0), labels, window, **kw)


def test_a_summary_without_a_flag_behaves_as_in_wp8_and_two_good_fits_are_aligned():
    m = _means(3)
    a, n = _summary(m, seed=1), _summary(m, seed=2)
    assert a.fit_unstable is False and a.fit_unstable_reason is None
    rec0, st = step(None, a, CFG, REGISTRY, lineage_id="t")
    rec1, _ = step(st, n, CFG, REGISTRY)
    assert rec0.continuity_state == NEW_LINEAGE and rec1.continuity_state == ALIGNED and rec1.reasons == () and not rec1.fit_unstable and not rec1.anchor_fit_unstable


def test_the_new_fields_default_to_nothing_known_against_the_fit():
    f = L.FitSummary.__dataclass_fields__
    assert f["fit_unstable"].default is False and f["fit_unstable_reason"].default is None
    r = L.LineageRecord.__dataclass_fields__
    assert r["fit_unstable"].default is False and r["anchor_fit_unstable"].default is False
    m = _means(3)
    s = _summary(m, seed=1)
    bare = L.FitSummary(s.engine, s.raw_means, s.scaler_mean, s.scaler_std, s.labels, s.separation_min, s.raw_window, s.exp_durations)          # as a WP8 caller builds it
    assert bare.fit_unstable is False and bare.fit_unstable_reason is None


def test_an_unstable_new_fit_is_never_aligned_and_the_anchor_stays():
    m = _means(3)
    a, n = _summary(m, seed=1), _summary(m, seed=2, fit_unstable=True, fit_unstable_reason=FS.REASON_DISAGREE)
    _, st = step(None, a, CFG, REGISTRY, lineage_id="t")
    r, st1 = step(st, n, CFG, REGISTRY)
    assert r.continuity_state == CONTINUITY_UNVERIFIED and "fit_unstable" in r.reasons and r.fit_unstable and not may_claim_same_as_before(r)
    assert st1.anchor is st.anchor and st1.n_fits == 2 and r.message.startswith(L.MESSAGE_UNCONFIRMED) and r.proposed_new_names is not None


def test_an_unstable_anchor_blocks_serving_later_fits():
    m = _means(3)
    a, n = _summary(m, seed=1, fit_unstable=True), _summary(m, seed=2)
    rec0, st = step(None, a, CFG, REGISTRY, lineage_id="t")
    assert rec0.continuity_state == NEW_LINEAGE and rec0.fit_unstable                           # the first fit is named as before and carries the flag
    r, _ = step(st, n, CFG, REGISTRY)
    assert r.continuity_state == CONTINUITY_UNVERIFIED and "anchor_fit_unstable" in r.reasons and "fit_unstable" not in r.reasons and r.anchor_fit_unstable and not r.fit_unstable


def test_a_flagged_fit_is_not_served_even_with_a_perfect_gate_and_margin():
    m = _means(5, sep=3.0)
    a, n = _summary(m, seed=1, noise=0.15), _summary(m, seed=2, noise=0.15, fit_unstable=True)
    _, st = step(None, a, CFG, REGISTRY, lineage_id="t")
    ok, _ = step(st, dataclasses.replace(n, fit_unstable=False), CFG, REGISTRY)
    bad, _ = step(st, n, CFG, REGISTRY)
    assert ok.continuity_state == ALIGNED and ok.margin_ratio == bad.margin_ratio and bad.continuity_state == CONTINUITY_UNVERIFIED and bad.reasons == ("fit_unstable",)


def test_the_user_can_still_choose_new_names_after_a_flagged_fit():
    m = _means(3)
    _, st = step(None, _summary(m, seed=1), CFG, REGISTRY, lineage_id="t")
    r, st1 = step(st, _summary(m, seed=2, fit_unstable=True), CFG, REGISTRY)
    r2, st2 = apply_user_choice(st1, r, "NEW_NAMES")
    assert r2.continuity_state == NEW_LINEAGE and st2.epoch == st.epoch + 1


def test_a_changed_number_of_regimes_is_still_k_changed_and_carries_the_flag():
    a = _summary(_means(3, K=3), seed=1)
    n = _summary(_means(3, K=4), seed=2, fit_unstable=True)
    _, st = step(None, a, CFG, REGISTRY, lineage_id="t")
    r, _ = step(st, n, CFG, REGISTRY)
    assert r.continuity_state == K_CHANGED and r.fit_unstable


def test_the_record_carries_the_flags_additively():
    m = _means(3)
    rec0, st = step(None, _summary(m, seed=1), CFG, REGISTRY, lineage_id="t")
    d = rec0.as_dict()
    assert d["fit_unstable"] is False and d["anchor_fit_unstable"] is False and set(L.RECORD_COLUMNS) <= set(d)


def test_summarize_fit_reads_the_flag_and_fails_closed_without_an_assessment():
    # a stand-in result: only the attributes summarize_fit uses
    base = generate_identity_stream(8801, n_days=60).window(0, 60)
    from backbone.hssm.model import GaussianHSMM
    logging.disable(logging.CRITICAL)
    model = GaussianHSMM(n_regimes=3, n_features=base["matrix"].shape[1], max_duration=12, seed=1)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        model.fit(base["matrix"], n_iter=8, verbose=False)
    post = model._forward_backward(base["matrix"])[0]

    def result(rs, baseline=True):
        return SimpleNamespace(model=model, p_t_posterior=post, is_baseline_model=baseline, restart_summary=rs)
    args = (base["raw"], base["scaler_mean"], base["scaler_std"])
    s = L.summarize_fit(result({"fit_unstable": False, "fit_unstable_reason": None}), *args)
    assert s.fit_unstable is False and s.fit_unstable_reason is None
    s = L.summarize_fit(result({"fit_unstable": True, "fit_unstable_reason": FS.REASON_DISAGREE}), *args)
    assert s.fit_unstable is True and s.fit_unstable_reason == FS.REASON_DISAGREE
    for rs in ({}, None, {"fit_unstable": None}):
        s = L.summarize_fit(result(rs), *args)
        assert s.fit_unstable is True and s.fit_unstable_reason == FS.REASON_NOT_ASSESSED          # no assessment: no "same as before" claim
    assert L.fit_stability_flag(SimpleNamespace(is_baseline_model=False)) == (False, None)         # the dual engine is never ALIGNED anyway (A6)


# ------------------------------------------------------------------------------------------------------------------- live fits
def _live(seed, n_init, window=(0, 70), max_iter=25):
    logging.disable(logging.CRITICAL)
    w = generate_identity_stream(seed, n_days=150, spec=IdentitySpec(sep_lo=2.0, sep_hi=2.5)).window(*window)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        r = fit_hssm(matrix=w["matrix"], candidate_ks=(3,), n_initializations=n_init, max_iter=max_iter, random_seed=seed, user_id="t", dataset_hash="h", allow_fast_test_fit=n_init < 10)
    return r, w


def test_a_live_fit_carries_the_keys_and_the_flag_equals_the_offline_computation_from_its_restarts():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration_report" / "wp9" / "scripts"))
    import wp9_fit_lib as W
    r, w = _live(8802, 4, max_iter=15)
    rs = r.restart_summary
    assert set(FS.KEYS) <= set(rs) and rs["consensus_m"] == 3
    recs = W.fit_restarts(w["matrix"], 3, 8802, 4, max_iter=15)
    healthy = [dict(ll=x["ll"], labels=W.str_to_labels(x["labels"]), collapsed=x["collapsed"]) for x in recs if x["state"] == "CONVERGED"]
    off = FS.assess_consensus(healthy, adjusted_rand_index)
    assert (rs["fit_unstable"], rs["n_healthy_restarts"]) == (off["fit_unstable"], off["n_healthy_restarts"])
    if off["consensus_min_ari"] is not None:
        assert rs["consensus_min_ari"] == pytest.approx(off["consensus_min_ari"])


def test_a_fast_test_fit_with_two_restarts_is_unstable_by_rule_and_withholding_is_unchanged():
    r, _ = _live(8803, 2, max_iter=15)
    assert r.restart_summary["fit_unstable"] is True and r.restart_summary["fit_unstable_reason"] in (FS.REASON_TOO_FEW, FS.REASON_NOT_ASSESSED)
    assert isinstance(r.insights_withheld, bool)                                                    # the flag is a lineage property: it does not decide publishing here
