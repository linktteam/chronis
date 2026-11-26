"""WP9 A2: the frozen selection rule (section 8) and the scoring helpers of `integration_report/wp9/scripts/a12_analyze.py`. Mutation check: mutation_a12_scoring.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "integration_report" / "wp9" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import a12_analyze as A                                                                 # noqa: E402

CANDS = [(1.0, 0.0), (2.0, 0.0), (2.0, 0.9), (4.0, 0.0)]


def T(served, refits, wrong=0, alignments=100):
    return dict(served=served, refits=refits, wrong_unflagged=wrong, alignments=alignments)


def curve(**per):
    """per: 'h/s' -> dict(cal=T, val=T, pooled=T). Candidates not named get a flat, wrong-heavy default."""
    out = {}
    for h, s in CANDS:
        d = per.get(f"{h}/{s}", dict(cal=T(1, 10, 5), val=T(1, 10, 5), pooled=T(2, 20, 10)))
        out[(True, h, s)] = d
        out[(False, h, s)] = dict(cal=T(99, 100, 0), val=T(99, 100, 0), pooled=T(99, 100, 0))      # the flag-off rows must never be used
    return out


def test_the_declared_levels():
    assert (A.TARGET_IDENTICAL_LINEAGE, A.TARGET_COVERAGE, A.TARGET_WRONG_UNFLAGGED) == (0.95, 0.80, 0.01)


def test_the_calibration_half_is_the_even_user_numbers():
    assert A.half_of(9302) == "cal" and A.half_of(9301) == "val" and A.half_of(0) == "cal"


# ----------------------------------------------------------------------------------------------------------------- the selection rule
def test_step_one_takes_the_highest_calibration_coverage_among_zero_wrong_candidates():
    c = curve(**{"1.0/0.0": dict(cal=T(50, 100, 0), val=T(50, 100, 0), pooled=T(100, 200, 0)),
                 "2.0/0.0": dict(cal=T(70, 100, 0), val=T(60, 100, 0), pooled=T(130, 200, 0)),
                 "4.0/0.0": dict(cal=T(95, 100, 3), val=T(90, 100, 0), pooled=T(185, 200, 3))})                # the highest coverage has wrong-and-unflagged: out
    ch, how = A.choose(c, CANDS)
    assert ch == (2.0, 0.0) and how.startswith("step 1-2")


def test_ties_go_to_the_larger_h_then_the_larger_s():
    same = dict(cal=T(70, 100, 0), val=T(70, 100, 0), pooled=T(140, 200, 0))
    assert A.choose(curve(**{"1.0/0.0": same, "2.0/0.0": same}), CANDS)[0] == (2.0, 0.0)
    assert A.choose(curve(**{"2.0/0.0": same, "2.0/0.9": same}), CANDS)[0] == (2.0, 0.9)
    assert A.choose(curve(**{"1.0/0.0": same, "4.0/0.0": same, "2.0/0.9": same}), CANDS)[0] == (4.0, 0.0)


def test_the_validation_level_is_inclusive_at_one_percent():
    ok = dict(cal=T(70, 100, 0), val=T(70, 100, 1, alignments=100), pooled=T(140, 200, 1))                   # 1/100 = 1.0%
    assert A.choose(curve(**{"2.0/0.0": ok}), CANDS) == ((2.0, 0.0), "step 1-2: highest calibration coverage among candidates with zero wrong-and-unflagged on the calibration half; validation half wrong-and-unflagged <= 1%")
    bad = dict(cal=T(70, 100, 0), val=T(70, 100, 1, alignments=99), pooled=T(140, 200, 1))                    # 1/99 > 1%
    ch, how = A.choose(curve(**{"2.0/0.0": bad}), CANDS)
    assert not how.startswith("step 1-2")


def test_a_validation_half_without_alignments_does_not_accept_the_candidate():
    c = curve(**{"2.0/0.0": dict(cal=T(70, 100, 0), val=T(0, 0, 0, alignments=0), pooled=T(70, 100, 0))})
    ch, how = A.choose(c, CANDS)
    assert how.startswith("step 3") and ch == (2.0, 0.0)


def test_when_no_candidate_has_zero_wrong_anywhere_the_fewest_wrong_is_taken():
    c = curve(**{"1.0/0.0": dict(cal=T(50, 100, 0), val=T(40, 100, 5), pooled=T(90, 200, 5)),            # fails validation
                 "2.0/0.0": dict(cal=T(60, 100, 2), val=T(60, 100, 0), pooled=T(120, 200, 2)),           # wrong on the calibration half
                 "4.0/0.0": dict(cal=T(30, 100, 1), val=T(30, 100, 0), pooled=T(60, 200, 1))})
    ch, how = A.choose(c, CANDS)
    # no candidate has zero wrong on the pooled set: the fewest decides (the default candidates have 10 wrong); 4.0/0.0 has 1, 2.0/0.0 has 2
    assert ch == (4.0, 0.0) and how.startswith("no candidate")


def test_step_three_picks_the_best_pooled_coverage_among_zero_wrong_pooled():
    c = curve(**{"1.0/0.0": dict(cal=T(60, 100, 0), val=T(40, 100, 5), pooled=T(90, 200, 0)),                 # the best calibration coverage, but it fails on the validation half
                 "2.0/0.0": dict(cal=T(50, 100, 0), val=T(60, 100, 5), pooled=T(120, 200, 0))})            # the best pooled coverage (a made-up curve, both have zero pooled wrong)
    ch, how = A.choose(c, CANDS)
    assert ch == (2.0, 0.0) and how.startswith("step 3")


def test_the_fewest_wrong_then_the_highest_pooled_coverage():
    c = curve(**{"1.0/0.0": dict(cal=T(1, 10, 2), val=T(1, 10, 2), pooled=T(100, 200, 2)),
                 "2.0/0.0": dict(cal=T(1, 10, 2), val=T(1, 10, 2), pooled=T(150, 200, 2)),
                 "4.0/0.0": dict(cal=T(1, 10, 3), val=T(1, 10, 3), pooled=T(190, 200, 3))})
    ch, how = A.choose(c, CANDS)
    assert ch == (2.0, 0.0) and how.startswith("no candidate")


def test_the_flag_off_rows_are_never_used():
    c = curve()
    ch, how = A.choose(c, CANDS)
    assert how.startswith("no candidate") and curve()[(False, 1.0, 0.0)]["cal"]["wrong_unflagged"] == 0 and ch != (None, None)


def test_coverage_is_per_half_not_a_count():
    # more served in absolute terms but a lower share of refits: the share decides
    c = curve(**{"1.0/0.0": dict(cal=T(60, 100, 0), val=T(60, 100, 0), pooled=T(120, 200, 0)),
                 "2.0/0.0": dict(cal=T(80, 200, 0), val=T(80, 200, 0), pooled=T(160, 400, 0))})
    assert A.choose(c, CANDS)[0] == (1.0, 0.0)


# ----------------------------------------------------------------------------------------------------------------- the helpers
def srow(correct, served=True, rec=True):
    return dict(correct=correct, served=served, both_recoverable=rec, state="ALIGNED" if served else "CONTINUITY_UNVERIFIED")


def test_tally_rows_counts_refits_as_steps_minus_chains_and_can_select_users():
    res = {(1, 0): ([dict(srow(None), state="NEW_LINEAGE", served=False), srow(True), srow(False)], []),
           (2, 0): ([dict(srow(None), state="NEW_LINEAGE", served=False), srow(True, served=False)], []),
           (3, 0): ([], [])}                                                                                   # a chain without any accepted fit is not a chain
    t = A.tally_rows(res)
    assert t["steps"] == 5 and t["refits"] == 3 and t["alignments"] == 3 and t["served"] == 2 and t["wrong_unflagged"] == 1
    t1 = A.tally_rows(res, {1})
    assert t1["steps"] == 3 and t1["refits"] == 2 and t1["wrong_unflagged"] == 1
    assert A.tally_rows(res, {2})["refits"] == 1


def test_fmt_prints_the_coverage_and_the_wrong_and_unflagged_counts():
    t = dict(served=3, refits=4, wrong_unflagged=1, alignments=8)
    assert A.fmt(t) == "coverage 3/4 = 75.0%, wrong-and-unflagged 1/8"
    assert "nan" in A.fmt(dict(served=0, refits=0, wrong_unflagged=0, alignments=0))


def test_verdict_line_names_the_verdict():
    assert "NOT MET" in A.verdict_line("x", 1, 10, 0.80, ">=")
    assert "MET" in A.verdict_line("x", 100, 100, 0.80, ">=") and "NOT MET" not in A.verdict_line("x", 100, 100, 0.80, ">=")
    assert "NOT MET" in A.verdict_line("x", 5, 100, 0.01, "<=")


def test_stratum_users_groups_by_the_stratum_of_the_first_fit():
    chains = {(1, 1): [dict(job=dict(stratum="easy"))], (2, 2): [dict(job=dict(stratum="hard"))], (3, 3): [dict(job=dict(stratum="easy"))]}
    assert {k: sorted(v) for k, v in A.stratum_users(chains).items()} == {"easy": [1, 3], "hard": [2]}


def test_the_split_of_the_chains_is_by_fit_seed():
    recs = [dict(job=dict(user=1, fitseed=1, end=90)), dict(job=dict(user=1, fitseed=1001, end=90)), dict(job=dict(user=2, fitseed=2, end=90))]
    a, b = A.split_chains(recs)
    assert set(k[0] for k in a) == {1, 2} and set(k[0] for k in b) == {1}


def test_the_fit_level_agreement_counts_unflagged_pairs_with_the_same_partition(monkeypatch):
    LAB = "0" * 10 + "1" * 10 + "2" * 10
    LAB2 = "0" * 10 + "1" * 10 + "2" * 5 + "1" * 5

    def rec(user, seed, end, labels):
        rs = [dict(i=i, seed=i, state="CONVERGED", n_iter=5, ll=-10.0 - i, collapsed=False, min_eff=9.0, labels=labels, z_means=[[0.0]] * 3, max_post=0.9, eff=[10.0] * 3, exp_dur=[5.0] * 3, dmax=12)
              for i in range(10)]
        return dict(job=dict(user=user, fitseed=seed, end=end, K=3), ok=True, restarts=rs)
    a = {(1, 1): [rec(1, 1, 90, LAB), rec(1, 1, 104, LAB), rec(1, 1, 100, LAB)]}
    b = {(1, 1001): [rec(1, 1001, 90, LAB), rec(1, 1001, 104, LAB2), rec(1, 1001, 100, LAB)]}
    out = A.fit_level_agreement(a, b, 10)
    assert out["pairs"] == 2 and out["both_unflagged"] == 2 and out["same_partition"] == 1 and out["flagged_either"] == 0      # window 100 is in both but not on the 14-day schedule; 104 differs
    flagged = {(1, 1001): [rec(1, 1001, 90, LAB), dict(rec(1, 1001, 104, LAB), restarts=[dict(r, labels=(LAB if r["i"] < 2 else LAB2)) for r in rec(1, 1001, 104, LAB)["restarts"]])]}
    out2 = A.fit_level_agreement(a, flagged, 10)
    assert out2["flagged_either"] == 1 and out2["both_unflagged"] == 1 and out2["same_partition"] == 1
    assert A.fit_level_agreement(a, b, 10, users={2})["pairs"] == 0
