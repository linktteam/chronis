"""WP10A A3: the declared metrics, the safety test and the frozen selection rule of `integration_report/wp10a/scripts/a3_analyze.py`. Mutation check: mutation_wp10a_a3.py."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import importlib.util

SCRIPTS = Path(__file__).resolve().parents[1] / "integration_report" / "wp10a" / "scripts"


def _load(name, path):
    """Load a script by PATH under a unique module name: the WP9 package has scripts of the same file name (a3_analyze.py), and `import a3_analyze` would return whichever was cached first."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


X = _load("wp10a_a3_analyze", SCRIPTS / "a3_analyze.py")


def row(end, state="ALIGNED"):
    return dict(end=end, state=state)


def lr(ids):
    return SimpleNamespace(stable_regime_id=np.asarray(ids))


SAME, OTHER = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0], [2, 1, 0, 2, 1, 0, 2, 1, 0, 2]


# ----------------------------------------------------------------------------------------------------------------- declared constants
def test_the_declared_levels_and_grid():
    assert (X.TARGET_NAMES, X.TARGET_COVERAGE, X.TARGET_WRONG, X.SAFETY_BOUND) == (0.98, 0.80, 0.01, 0.01)
    assert X.H_CANDS == (2.0, 2.25, 2.5) and X.SHARP_ON == 0.95 and X.FIRST_T == 90
    assert X.B_T == [90, 104, 118, 132, 146, 150] and X.G_T == [90, 97, 104, 111, 118, 125, 132, 139, 146, 150]
    assert X.cand_list() == [(True, 2.0), (True, 2.25), (True, 2.5), (False, 2.0), (False, 2.25), (False, 2.5)] and X.cand_list(True) == [(True, 2.0), (True, 2.25), (True, 2.5)]
    assert X.sharp_of(True) == 0.95 and X.sharp_of(False) == 0.0


def test_the_calibration_half_is_the_even_user_numbers():
    assert X.half_of(10002) == "cal" and X.half_of(10001) == "val"


# ----------------------------------------------------------------------------------------------------------------- the safety test (rule of three)
def test_the_numbers_of_alignments_the_bound_needs():
    assert X.alignments_needed(0) == 299 and X.alignments_needed(1) == 473


def test_exact_upper_bound_values():
    assert X.exact_upper(0, 299) <= 0.01 < X.exact_upper(0, 298)
    assert X.exact_upper(0, 100) == pytest.approx(1 - 0.05 ** 0.01)
    assert X.exact_upper(1, 473) <= 0.01 < X.exact_upper(1, 472)
    assert X.exact_upper(5, 100) == pytest.approx(0.10225, abs=1e-4)
    assert X.exact_upper(10, 10) == 1.0 and np.isnan(X.exact_upper(0, 0))


def test_the_safety_test_needs_zero_wrong_and_enough_alignments():
    assert X.safety_demonstrated(0, 299) and X.safety_demonstrated(0, 324)
    assert not X.safety_demonstrated(0, 298)
    assert not X.safety_demonstrated(1, 324) and not X.safety_demonstrated(1, 472)       # one wrong cannot be demonstrated below 473
    assert not X.safety_demonstrated(0, 0)


# ----------------------------------------------------------------------------------------------------------------- pair cases, (a), (b) and the old criterion
def test_pair_cases_skip_the_first_window_and_count_windows_where_one_seed_has_a_fit():
    ra, la = [row(90, "NEW"), row(104), row(118, "CONTINUITY_UNVERIFIED")], [lr(SAME)] * 3
    rb, lb = [row(90, "NEW"), row(118)], [lr(SAME)] * 2                                 # seed B has no accepted fit at 104
    cs = X.pair_cases(ra, la, rb, lb)
    assert [c["end"] for c in cs] == [104, 118]
    assert cs[0]["a"] is True and cs[0]["b"] is False and cs[0]["one_missing"] is True and cs[0]["agree"] is None
    assert cs[1]["a"] is False and cs[1]["b"] is True and cs[1]["one_missing"] is False


def test_names_agree_uses_the_99_percent_day_level():
    ids99 = SAME[:]; ids99[0] = 9                                                       # 9 of 10 days equal = 90%: disagree
    big = [0, 1, 2] * 34                                                                # 102 days
    big99 = big[:]; big99[0] = 7                                                        # 101 of 102 = 99.02%: agree
    big98 = big[:]; big98[0] = 7; big98[1] = 8                                          # 100 of 102 = 98.04%: disagree
    for ids, want in ((SAME, True), (ids99, False), (OTHER, False)):
        cs = X.pair_cases([row(90), row(104)], [lr(SAME)] * 2, [row(90), row(104)], [lr(SAME), lr(ids)])
        assert cs[0]["agree"] is want
    assert X.pair_cases([row(90), row(104)], [lr(big)] * 2, [row(90), row(104)], [lr(big), lr(big99)])[0]["agree"] is True
    assert X.pair_cases([row(90), row(104)], [lr(big)] * 2, [row(90), row(104)], [lr(big), lr(big98)])[0]["agree"] is False


def test_names_agree_at_exactly_99_percent_of_days():
    base = list(range(100)); one_off = base[:]; one_off[0] = -1                           # 99 of 100 days equal = 99.0%
    two_off = base[:]; two_off[0] = -1; two_off[1] = -2                                   # 98%
    assert X.pair_cases([row(90), row(104)], [lr(base)] * 2, [row(90), row(104)], [lr(base), lr(one_off)])[0]["agree"] is True
    assert X.pair_cases([row(90), row(104)], [lr(base)] * 2, [row(90), row(104)], [lr(base), lr(two_off)])[0]["agree"] is False


def test_names_metric_counts_only_windows_served_by_both_seeds():
    cases = [dict(a=True, b=True, agree=True), dict(a=True, b=True, agree=False), dict(a=True, b=False, agree=None), dict(a=False, b=False, agree=None), dict(a=True, b=True, agree=True)]
    assert X.names_metric(cases) == (2, 3)
    assert X.names_metric([]) == (0, 0)


def test_serve_agreement_counts_both_serve_both_withhold_and_one_only():
    cases = [dict(a=True, b=True, agree=True), dict(a=False, b=False, agree=None), dict(a=False, b=False, agree=None), dict(a=True, b=False, agree=None), dict(a=False, b=True, agree=None)]
    s = X.serve_agreement(cases)
    assert (s["cases"], s["both_serve"], s["both_withhold"], s["one_serves"], s["agree"]) == (5, 1, 2, 2, 3)
    assert s["agree_where_either_serves"] == (1, 3)
    assert X.serve_agreement([])["agree"] == 0


def test_the_old_strict_criterion_needs_both_seeds_and_the_same_names_among_windows_served_by_at_least_one():
    cases = [dict(a=True, b=True, agree=True), dict(a=True, b=True, agree=False), dict(a=True, b=False, agree=None), dict(a=False, b=True, agree=None), dict(a=False, b=False, agree=None)]
    assert X.strict_identical(cases) == (1, 4)


# ----------------------------------------------------------------------------------------------------------------- (c) days to first served lineage
def test_first_served_is_the_first_aligned_step_after_the_first_fit():
    assert X.first_served([row(90, "NEW"), row(97, "CONTINUITY_UNVERIFIED"), row(104), row(111)]) == 104
    assert X.first_served([row(90, "ALIGNED"), row(97, "CONTINUITY_UNVERIFIED")]) is None          # the first fit itself is never "served"
    assert X.first_served([]) is None


def test_days_summary_counts_censoring_as_61():
    d = X.days_summary([97, 104, 118, None])                                             # 7, 14, 28 days and never
    assert (d["n"], d["served_by_14"], d["served_by_28"], d["served_by_60"], d["never"]) == (4, 2, 3, 3, 1)
    assert d["median_among_served"] == 14
    assert (d["q25_censored"], d["median_censored"], d["q75_censored"]) == (7, 14, 28)
    d2 = X.days_summary([None, None])
    assert d2["median_among_served"] is None and d2["median_censored"] == 61 and d2["served_by_60"] == 0
    assert X.days_summary([150])["served_by_60"] == 1 and X.days_summary([150])["median_among_served"] == 60


# ----------------------------------------------------------------------------------------------------------------- the frozen selection rule
CANDS = X.cand_list()


def T(served, refits, wrong=0, alignments=100):
    return dict(served=served, refits=refits, wrong_unflagged=wrong, alignments=alignments)


def curve(**per):
    out = {}
    for c in CANDS:
        out[c] = per.get(f"{int(c[0])}/{c[1]}", dict(cal=T(1, 10, 5), val=T(1, 10, 5), pooled=T(2, 20, 10)))
    return out


def test_step_one_takes_the_highest_calibration_coverage_among_zero_wrong_candidates():
    c = curve(**{"1/2.0": dict(cal=T(50, 100, 0), val=T(50, 100, 0), pooled=T(100, 200, 0)), "1/2.25": dict(cal=T(60, 100, 0), val=T(40, 100, 0), pooled=T(100, 200, 0)),
                 "1/2.5": dict(cal=T(70, 100, 3), val=T(70, 100, 0), pooled=T(140, 200, 3))})
    assert X.choose(c, CANDS)[0] == (True, 2.25)                                          # 2.5 has the highest coverage but a wrong alignment on the calibration half


def test_ties_go_to_the_larger_h_then_flag_on():
    flat = dict(cal=T(50, 100, 0), val=T(50, 100, 0), pooled=T(100, 200, 0))
    assert X.choose(curve(**{"1/2.0": flat, "1/2.5": flat}), CANDS)[0] == (True, 2.5)
    assert X.choose(curve(**{"1/2.5": flat, "0/2.5": flat}), CANDS)[0] == (True, 2.5)
    assert X.choose(curve(**{"0/2.25": flat, "0/2.5": flat}), CANDS)[0] == (False, 2.5)


def test_step_two_rejects_a_candidate_whose_validation_half_exceeds_one_percent():
    c = curve(**{"1/2.0": dict(cal=T(90, 100, 0), val=T(90, 100, 2, alignments=100), pooled=T(180, 200, 2)),
                 "1/2.25": dict(cal=T(40, 100, 0), val=T(40, 100, 0), pooled=T(80, 200, 0))})
    cand, how = X.choose(c, CANDS)
    assert cand == (True, 2.25) and how.startswith("step 3")                              # falls to the pooled pick among the candidates with zero wrong on the pooled set


def test_validation_exactly_one_percent_is_accepted():
    c = curve(**{"1/2.0": dict(cal=T(90, 100, 0), val=T(90, 100, 1, alignments=100), pooled=T(180, 200, 1))})
    cand, how = X.choose(c, CANDS)
    assert cand == (True, 2.0) and how.startswith("step 1-2")


def test_no_zero_candidate_takes_the_fewest_wrong():
    c = curve(**{"1/2.0": dict(cal=T(50, 100, 4), val=T(50, 100, 4), pooled=T(100, 200, 8)), "1/2.25": dict(cal=T(50, 100, 3), val=T(50, 100, 3), pooled=T(100, 200, 6)),
                 "1/2.5": dict(cal=T(50, 100, 4), val=T(50, 100, 4), pooled=T(100, 200, 8))})
    # every other candidate has 10 wrong on the pooled set by the default
    cand, how = X.choose(c, CANDS)
    assert cand == (True, 2.25) and how.startswith("no candidate")


def test_the_flag_on_pick_can_differ_from_the_six_candidate_pick():
    good_off = dict(cal=T(95, 100, 0), val=T(95, 100, 0), pooled=T(190, 200, 0))
    good_on = dict(cal=T(60, 100, 0), val=T(60, 100, 0), pooled=T(120, 200, 0))
    c = curve(**{"0/2.25": good_off, "1/2.5": good_on})
    assert X.choose(c, X.cand_list(flag_only=True))[0] == (True, 2.5)
    assert X.choose(c, CANDS)[0] == (False, 2.25)


# ----------------------------------------------------------------------------------------------------------------- survivors of the first mutation run (added tests)
def test_the_strict_criterion_counts_a_case_identical_only_when_both_seeds_serve():
    assert X.strict_identical([dict(a=True, b=False, agree=True)]) == (0, 1)               # an inconsistent record must not count as identical
    assert X.strict_identical([dict(a=False, b=True, agree=True)]) == (0, 1)


def test_the_median_among_served_is_the_lower_middle_for_an_even_count():
    d = X.days_summary([97, 104, 118, 125])                                                # 7, 14, 28, 35 days
    assert d["median_among_served"] == 14


def test_quantiles_of_the_censored_distribution():
    d = X.days_summary([97, 104, 118, 125, None])                                          # 7, 14, 28, 35, 61
    assert (d["q25_censored"], d["median_censored"], d["q75_censored"]) == (14, 28, 35)


def test_the_fallback_breaks_a_tie_in_wrong_alignments_by_the_higher_coverage():
    c = curve(**{"1/2.0": dict(cal=T(50, 100, 3), val=T(50, 100, 3), pooled=T(60, 100, 6)), "1/2.5": dict(cal=T(50, 100, 3), val=T(50, 100, 3), pooled=T(90, 100, 6))})
    assert X.choose(c, [(True, 2.0), (True, 2.5)])[0] == (True, 2.5)


def test_step_three_takes_the_highest_pooled_coverage_among_zero_wrong_candidates():
    bad_val = dict(cal=T(90, 100, 0), val=T(90, 100, 5), pooled=T(180, 200, 5))            # step 2 rejects the best calibration candidate
    c = curve(**{"1/2.0": bad_val, "1/2.25": dict(cal=T(40, 100, 0), val=T(40, 100, 0), pooled=T(80, 200, 0)),
                 "1/2.5": dict(cal=T(30, 100, 0), val=T(30, 100, 0), pooled=T(150, 200, 0))})
    cand, how = X.choose(c, CANDS)
    assert cand == (True, 2.5) and how.startswith("step 3")
