"""WP8 Item 8 harness: the scoring functions of integration_report/wp8/scripts/lineage_study_lib.py are tested before they score anything (a scoring bug would invalidate every result).
End-to-end cases use fits built from the generator's own truth (known relabellings), so the right answer is known. Mutation check: integration_report/wp8/scripts/mutation_item8_scoring.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backbone.hssm.lineage import ALIGNED, CONTINUITY_UNVERIFIED, K_CHANGED, NEW_LINEAGE, AlignmentConfig
from backbone.hssm.ordering import LadderConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration_report" / "wp8" / "scripts"))
_spec = importlib.util.spec_from_file_location("lineage_study_lib", ROOT / "integration_report" / "wp8" / "scripts" / "lineage_study_lib.py")
S = importlib.util.module_from_spec(_spec)
sys.modules["lineage_study_lib"] = S
_spec.loader.exec_module(S)
from identity_lib import stream_for                                                 # noqa: E402

CFG = AlignmentConfig(min_separation=1.0, ladder=LadderConfig(n_bootstrap=40, min_gap=0.5))


# ------------------------------------------------------------------------------------------------------------------------------------- statistics
def test_wilson_interval_known_values_and_edges():
    lo, hi = S.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=2e-3) and hi == pytest.approx(0.7634, abs=2e-3)
    lo, hi = S.wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-9) and hi == pytest.approx(0.2775, abs=2e-3)
    lo, hi = S.wilson(10, 10)
    assert lo == pytest.approx(0.7225, abs=2e-3) and hi == pytest.approx(1.0, abs=1e-9)
    lo, hi = S.wilson(95, 100)
    assert lo < 0.95 < hi
    assert all(np.isnan(x) for x in S.wilson(0, 0))
    wide, narrow = S.wilson(5, 10), S.wilson(500, 1000)
    assert wide[1] - wide[0] > narrow[1] - narrow[0]


def test_wilson_interval_stays_inside_zero_and_one_and_prints_no_negative_zero():
    for n in (1, 5, 20, 36, 240, 1000):
        for k in (0, n):
            lo, hi = S.wilson(k, n)
            assert 0.0 <= lo <= hi <= 1.0
    assert S.wilson(0, 36)[0] == 0.0 and S.wilson(36, 36)[1] == 1.0
    assert "(95% CI 0.0-" in S.rate(0, 36) and "-0.0" not in S.rate(0, 36)                          # the printed lower bound is 0.0, not -0.0


def test_rate_text_shows_count_percent_and_interval_and_handles_zero():
    t = S.rate(19, 20)
    assert t.startswith("19/20 = 95.0%") and "95% CI" in t
    assert S.rate(0, 0) == "0/0"


def test_day_agreement():
    assert S.day_agreement([0, 1, 2, 1], [0, 1, 2, 1]) == 1.0
    assert S.day_agreement([0, 1, 2, 1], [0, 1, 0, 0]) == 0.5
    assert S.day_agreement([0, 0], [1, 1]) == 0.0
    with pytest.raises(ValueError):
        S.day_agreement([0, 1], [0, 1, 2])
    with pytest.raises(ValueError):
        S.day_agreement([], [])


# ------------------------------------------------------------------------------------------------------------------------------------- one alignment
def _info(mp, K=3, recoverable=True, engine="baseline"):
    return S.FitInfo(engine, K, list(mp), recoverable, True, (0, 90), 1, np.zeros(10, dtype=int))


def test_alignment_correct_follows_the_stable_ids_not_the_raw_order():
    a = _info([0, 1, 2])
    n = _info([1, 0, 2])                          # new raw regime 0 is planted 1, raw 1 is planted 0
    assert S.alignment_correct(a, n, (0, 1, 2), (1, 0, 2)) == (True, True)      # raw 0 named 1, raw 1 named 0: both follow the planted regime
    assert S.alignment_correct(a, n, (0, 1, 2), (0, 1, 2)) == (False, True)     # names follow the raw order: wrong
    assert S.alignment_correct(a, n, (0, 1, 2), (1, 0, 2))[0] is True
    # the anchor's own raw order is not the stable order: anchor raw 0 is called 2, raw 2 is called 0
    a2 = _info([0, 1, 2])
    assert S.alignment_correct(a2, _info([0, 1, 2]), (2, 1, 0), (2, 1, 0)) == (True, True)
    assert S.alignment_correct(a2, _info([0, 1, 2]), (2, 1, 0), (0, 1, 2)) == (False, True)


def test_a_fit_that_is_not_recoverable_is_never_correct_and_a_changed_k_has_no_answer():
    a, n = _info([0, 1, 2]), _info([0, 1, 2], recoverable=False)
    assert S.alignment_correct(a, n, (0, 1, 2), (0, 1, 2)) == (False, False)
    assert S.alignment_correct(n, a, (0, 1, 2), (0, 1, 2)) == (False, False)
    assert S.alignment_correct(a, _info([0, 1], K=2), (0, 1, 2), (0, 1)) == (None, True)
    merged = _info([0, 0, 2])                     # two fit regimes share a planted regime
    assert S.alignment_correct(a, merged, (0, 1, 2), (0, 1, 2))[0] is False


def _lr(state, i=1, reasons=(), r2s=(0, 1, 2)):
    return SimpleNamespace(fit_index=i, continuity_state=state, reasons=tuple(reasons), margin_diff=1.0, margin_ratio=0.1, separation_min=3.0, raw_to_stable=tuple(r2s))


def test_score_step_rows_for_each_state():
    infos = [_info([0, 1, 2])]
    before = SimpleNamespace(anchor=SimpleNamespace(fit_index=0, raw_to_stable=(0, 1, 2)))
    info = _info([0, 1, 2])
    first = S.score_step(_lr(NEW_LINEAGE, 0), None, [], info)
    assert first["correct"] is None and first["served"] is False and first["state"] == NEW_LINEAGE
    ok = S.score_step(_lr(ALIGNED), before, infos, info)
    assert ok["served"] is True and ok["correct"] is True and ok["both_recoverable"] is True
    bad = S.score_step(_lr(ALIGNED, r2s=(1, 0, 2)), before, infos, info)
    assert bad["served"] is True and bad["correct"] is False
    flagged = S.score_step(_lr(CONTINUITY_UNVERIFIED, r2s=(1, 0, 2), reasons=("margin_below_threshold",)), before, infos, info)
    assert flagged["served"] is False and flagged["correct"] is False and flagged["reasons"] == ["margin_below_threshold"]
    kc = S.score_step(_lr(K_CHANGED), before, infos, _info([0, 1], K=2))
    assert kc["correct"] is None and kc["served"] is False and "anchor_engine" not in kc          # no alignment exists for a changed K
    assert kc["K_anchor"] == 3 and kc["K"] == 2 and ok["K_anchor"] == 3 and "K_anchor" not in first


def test_the_anchor_used_for_scoring_is_the_one_the_state_names_not_the_first_fit():
    infos = [_info([0, 1, 2]), _info([1, 0, 2])]                          # fit 0 and fit 1 name the planted regimes differently
    before = SimpleNamespace(anchor=SimpleNamespace(fit_index=1, raw_to_stable=(0, 1, 2)))
    row = S.score_step(_lr(ALIGNED, 2), before, infos, _info([1, 0, 2]))
    assert row["correct"] is True                                        # right against fit 1 (the anchor), wrong against fit 0
    before0 = SimpleNamespace(anchor=SimpleNamespace(fit_index=0, raw_to_stable=(0, 1, 2)))
    assert S.score_step(_lr(ALIGNED, 2), before0, infos, _info([1, 0, 2]))["correct"] is False


# ------------------------------------------------------------------------------------------------------------------------------------- tally
def _row(state, correct, both=True):
    return dict(state=state, served=state == ALIGNED, correct=correct, both_recoverable=both)


def test_tally_counts():
    rows = [_row(NEW_LINEAGE, None), _row(ALIGNED, True), _row(ALIGNED, True), _row(ALIGNED, False, both=False), _row(CONTINUITY_UNVERIFIED, False), _row(CONTINUITY_UNVERIFIED, False), _row(CONTINUITY_UNVERIFIED, True),
            _row(K_CHANGED, None)]
    t = S.tally(rows)
    assert t == dict(alignments=6, served=3, served_correct=2, wrong_unflagged=1, flagged=3, flagged_would_be_wrong=2, flagged_would_be_right=1, cond_alignments=5, cond_correct=3, steps=8, k_changed=1)


# ------------------------------------------------------------------------------------------------------------------------------------- replay end to end
def _fake(user, start, end, perm, drift=0.0, accepted=True, fit_state="CONVERGED", merge_first_two=False):
    """A 'fit' that decodes the planted labels exactly but names the regimes by `perm` (fit regime r = planted regime perm[r]); regime means are the observed means per decoded regime."""
    job = dict(user=user, engine="baseline", start=start, end=end, fitseed=user, K=3, drift=drift, kchange=None, group="baseline-G")
    w = stream_for(job).window(start, end)
    inv = {p: r for r, p in enumerate(perm)}
    fit_labels = np.array([inv[int(t)] for t in w["labels"]])
    if merge_first_two:
        fit_labels = np.where(fit_labels == 1, 0, fit_labels)
    raw = w["raw"]
    means = np.array([np.nanmean(raw[fit_labels == r], axis=0) if (fit_labels == r).any() else np.nanmean(raw, axis=0) for r in range(3)])
    return dict(job=job, ok=True, K=3, raw_means=means.tolist(), scaler_mean=w["scaler_mean"].tolist(), scaler_std=w["scaler_std"].tolist(), exp_dur=[12.0, 12.0, 12.0],
                labels_fit=fit_labels.tolist(), labels_true=w["labels"].tolist(), fit_state=fit_state, accepted=accepted)


USER = 8204        # a user with a large planted separation (the Item 3 development table: min separation 8.4)


def test_replay_serves_and_scores_relabelled_fits_of_a_well_separated_user_as_correct():
    recs = [_fake(USER, 0, 90, (0, 1, 2)), _fake(USER, 0, 120, (2, 0, 1)), _fake(USER, 0, 150, (1, 2, 0))]
    rows, lrs, state = S.replay(recs, CFG)
    assert [r["state"] for r in rows] == [NEW_LINEAGE, ALIGNED, ALIGNED]
    assert [r["correct"] for r in rows] == [None, True, True]
    t = S.tally(rows)
    assert t["served"] == 2 and t["wrong_unflagged"] == 0 and state.n_fits == 3
    # the per-day stable ids of the first 90 days do not move although the raw labels were permuted
    ids = S.stable_ids_first_days(lrs, 90)
    assert S.day_agreement(ids[0], ids[1]) == 1.0 and S.day_agreement(ids[0], ids[2]) == 1.0
    assert S.day_agreement(recs[0]["labels_fit"], recs[1]["labels_fit"][:90]) < 1.0      # while the raw labels disagree


def test_replay_scores_a_merged_fit_as_not_recoverable_and_counts_a_served_one_as_wrong():
    recs = [_fake(USER, 0, 90, (0, 1, 2)), _fake(USER, 0, 120, (0, 1, 2), merge_first_two=True)]
    rows, _, _ = S.replay(recs, AlignmentConfig(min_separation=0.0, ladder=CFG.ladder))
    assert rows[1]["recoverable"] is False and rows[1]["correct"] is False
    if rows[1]["served"]:
        assert S.tally(rows)["wrong_unflagged"] == 1


def test_replay_skips_withheld_fits_unless_asked_and_always_skips_failed_ones():
    recs = [_fake(USER, 0, 90, (0, 1, 2)), _fake(USER, 0, 105, (0, 1, 2), accepted=False), _fake(USER, 0, 120, (0, 1, 2), fit_state="FAILED"), _fake(USER, 0, 135, (0, 1, 2))]
    assert [r["window"][1] for r in S.replay(recs, CFG)[0]] == [90, 135]
    assert [r["window"][1] for r in S.replay(recs, CFG, accepted_only=False)[0]] == [90, 105, 135]
    bad = dict(recs[0]); bad["ok"] = False
    assert not S.usable(bad) and S.usable(recs[0])


def test_replay_patch_hook_runs_inside_the_context_and_restores_after():
    import contextlib
    from backbone.hssm import lineage as L
    seen = {}

    @contextlib.contextmanager
    def patch():
        old = L.margin_ok
        L.margin_ok = lambda th, al: False                  # a patch that flags everything
        seen["in"] = True
        try:
            yield
        finally:
            L.margin_ok = old
    recs = [_fake(USER, 0, 90, (0, 1, 2)), _fake(USER, 0, 120, (0, 1, 2))]
    rows, _, _ = S.replay(recs, CFG, patch=patch)
    assert seen["in"] and rows[1]["state"] == CONTINUITY_UNVERIFIED
    assert L.margin_ok.__name__ == "margin_ok"
    assert S.replay(recs, CFG)[0][1]["state"] == ALIGNED


def test_the_summary_is_built_from_the_recorded_raw_means_scaler_and_labels():
    rec = _fake(USER, 0, 90, (2, 0, 1))
    s = S.summary_for(rec)
    assert np.allclose(s.raw_means, np.array(rec["raw_means"])) and np.allclose(s.scaler_std, np.array(rec["scaler_std"])) and np.allclose(s.scaler_mean, np.array(rec["scaler_mean"]))
    assert np.array_equal(s.labels, np.array(rec["labels_fit"])) and not np.array_equal(s.labels, np.array(rec["labels_true"]))
    assert np.allclose(s.exp_durations, [12.0, 12.0, 12.0])
    assert s.engine == "baseline" and s.K == 3 and s.separation_min > 3.0


# ------------------------------------------------------------------------------------------------------------------------------------- refit cadence helpers
def test_schedules_come_from_the_production_policy_and_always_end_on_the_last_window():
    assert S.schedule_days(7, 90, 150) == [90, 97, 104, 111, 118, 125, 132, 139, 146, 150]
    assert S.schedule_days(14, 90, 150) == [90, 104, 118, 132, 146, 150]
    assert S.schedule_days(1, 90, 150) == list(range(90, 151))
    assert S.schedule_days(7, 90, 153)[-1] == 153 and S.schedule_days(28, 90, 150) == [90, 118, 146, 150]
    with pytest.raises(ValueError):
        S.schedule_days(0, 90, 150)                                   # outside the policy's bounds
    with pytest.raises(ValueError):
        S.schedule_days(29, 90, 150)


def test_fits_for_days_keeps_the_order_and_refuses_a_missing_fit():
    recs = [dict(job=dict(end=e), x=e) for e in (100, 90, 97)]
    assert [r["x"] for r in S.fits_for_days(recs, [90, 97, 100])] == [90, 97, 100]
    with pytest.raises(KeyError, match="no fit for window ends \\[91\\]"):
        S.fits_for_days(recs, [90, 91])


def test_candidates_are_translated_to_planted_regimes_through_the_anchors_stable_names():
    anchor = _info([2, 0, 1])                                           # anchor raw regimes 0, 1, 2 are planted 2, 0, 1
    r2s = (1, 2, 0)                                                     # ... and carry the stable ids 1, 2, 0
    got = S.candidates_as_planted([dict(kind="split", old_stable_id=0, new_raw_regimes=[0, 3]), dict(kind="merge", old_stable_ids=[1, 2], new_raw_regime=0)], anchor, r2s)
    assert got == [dict(kind="split", planted=[1]), dict(kind="merge", planted=[0, 2])]
    assert S.candidates_as_planted([dict(kind="split", old_stable_id=7, new_raw_regimes=[0, 1])], anchor, r2s) == [dict(kind="split", planted=[-1])]
    assert S.candidates_as_planted((), anchor, r2s) == []


def test_a_k_changed_row_carries_its_candidates_in_planted_terms():
    infos = [_info([0, 1, 2])]
    before = SimpleNamespace(anchor=SimpleNamespace(fit_index=0, raw_to_stable=(0, 1, 2)))
    lr = _lr(K_CHANGED)
    lr.split_merge_candidates = (dict(kind="merge", old_stable_ids=[0, 2], new_raw_regime=0),)
    row = S.score_step(lr, before, infos, _info([0, 1], K=2))
    assert row["candidates"] == [dict(kind="merge", planted=[0, 2])]
    assert "candidates" not in S.score_step(_lr(ALIGNED), before, infos, _info([0, 1, 2]))


# ------------------------------------------------------------------------------------------------------------------------------------- verdict wording
def test_verdict_for_a_rate_that_must_be_at_least_a_threshold():
    assert S.verdict(0, 0, 0.9, ">=").startswith("UNKNOWN")
    assert S.verdict(80, 100, 0.95, ">=") == "NOT MET"
    assert S.verdict(100, 100, 0.95, ">=") == "MET"                                       # interval 96.3..100 is above 0.95
    assert S.verdict(19, 20, 0.95, ">=").startswith("MET on the point estimate, NOT DEMONSTRATED")      # 95% observed, interval reaches 76..99
    assert S.verdict(94, 100, 0.95, ">=") == "NOT MET"                                    # just below on the point estimate
    assert S.verdict(20, 20, 1.0, ">=").startswith("MET on the point estimate, NOT DEMONSTRATED")        # 100% can never be demonstrated by an interval


def test_verdict_for_a_rate_that_must_be_at_most_a_threshold():
    assert S.verdict(0, 0, 0.01, "<=").startswith("UNKNOWN")
    assert S.verdict(5, 100, 0.01, "<=") == "NOT MET"
    assert S.verdict(0, 2000, 0.01, "<=") == "MET"                                        # upper bound 0.19%
    assert S.verdict(0, 100, 0.01, "<=").startswith("MET on the point estimate, NOT DEMONSTRATED")      # upper bound 3.7%
    assert S.verdict(1, 100, 0.01, "<=").startswith("MET on the point estimate")           # exactly at the threshold
    assert S.verdict(2, 100, 0.01, "<=") == "NOT MET"


def test_verdict_refuses_an_unknown_comparison():
    with pytest.raises(ValueError):
        S.verdict(1, 2, 0.5, "==")
