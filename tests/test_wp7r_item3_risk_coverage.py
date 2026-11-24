"""WP7-R item 3: the withheld / risk-coverage statistics (tests/utils/item3_stats.py). Criteria: tests/PREDECLARED_CRITERIA_WP7R.md section 3.

Hand-computed expected values; every rule has a negative control (a change that must NOT move the statistic). Mutation check:
integration_report/wp7r/scripts/mutation_item3_stats.py (every test is killed by at least one mutant).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tests.utils import item3_stats as S
from tests.utils.n18v2 import paired_stats

D8 = [1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0]          # mean 2, ddof-1 sd = sqrt(8/7), se = sd / sqrt(8) = 0.37796447, mean - 3 se = 0.86610658


def _case(d, base=-5.0, publishable=None):
    seeds = list(range(100, 100 + len(d)))
    ref = {s: base for s in seeds}
    scores = {s: base + x for s, x in zip(seeds, d)}
    pub = {s: True for s in seeds} if publishable is None else dict(zip(seeds, publishable))
    return scores, ref, pub, seeds


# ---------------------------------------------------------------------------------------------------------------- reasons and tally
def test_reason_prefixes_parse_every_prefix_the_code_emits():
    r = ("fit_state_not_converged: fit_state is MAX_ITER_REACHED, the EM fit did not reach convergence; fit_degraded: the -1e6 Cholesky fallback fired 2 time(s) "
         "during fit; states_pass: the fallback fired 1 time(s) in the estimate_states pass")
    assert S.reason_prefixes(r) == ["fit_state_not_converged", "fit_degraded", "states_pass"]
    assert S.reason_prefixes(None) == [] and S.reason_prefixes("") == []
    assert S.reason_prefixes("fit_state is NUMERICAL_FAILURE") == ["numerical_failure"]
    assert S.reason_prefixes("collapsed_regime: a regime holds 1.00 effective days (minimum 4.0)") == ["collapsed_regime"]
    # a colon inside the text of a reason is not a prefix
    assert S.reason_prefixes("fit_state_not_converged: fit_state is X, note: something") == ["fit_state_not_converged"]


def test_withheld_tally_counts_states_and_prefixes():
    recs = [dict(withheld=True, fit_state="MAX_ITER_REACHED", withheld_reason="fit_state_not_converged: x"),
            dict(withheld=True, fit_state="REJECTED_UPDATE", withheld_reason="fit_state_not_converged: y; fit_degraded: z"),
            dict(withheld=True, fit_state="MAX_ITER_REACHED", withheld_reason="fit_state_not_converged: x"),
            dict(withheld=False, fit_state="CONVERGED", withheld_reason=None)]
    t = S.withheld_tally(recs)
    assert (t["n"], t["n_withheld"], t["n_publishable"]) == (4, 3, 1) and t["withheld_rate"] == 0.75
    assert t["by_fit_state"] == {"MAX_ITER_REACHED": 2, "REJECTED_UPDATE": 1}      # the converged fit is not in the withheld tally
    assert t["by_reason_prefix"] == {"fit_state_not_converged": 3, "fit_degraded": 1}
    assert t["fit_states_all"] == {"MAX_ITER_REACHED": 2, "REJECTED_UPDATE": 1, "CONVERGED": 1}
    none = S.withheld_tally([dict(withheld=False, fit_state="CONVERGED", withheld_reason=None)] * 3)
    assert none["n_withheld"] == 0 and none["withheld_rate"] == 0.0 and none["by_reason_prefix"] == {}
    assert math.isnan(S.withheld_tally([])["withheld_rate"])


# ---------------------------------------------------------------------------------------------------------------- publishable-subset statistics
def test_subset_stats_matches_the_hand_computed_values():
    scores, ref, pub, _ = _case(D8)
    st = S.subset_stats(scores, ref, pub)
    assert st["n_used"] == 8 and st["n_publishable"] == 8 and st["coverage"] == 1.0 and st["judged"] is True
    assert st["mean"] == pytest.approx(2.0) and st["se"] == pytest.approx(0.37796447, abs=1e-7) and st["mean_minus_3se"] == pytest.approx(0.86610658, abs=1e-7)
    assert st["n_favour"] == 8 and st["mean_arm"] == pytest.approx(-3.0) and st["mean_ref"] == pytest.approx(-5.0)
    # the same numbers as the WP5 paired statistic on the same pairs
    w5 = paired_stats(scores, ref)
    assert (st["mean"], st["se"], st["mean_minus_3se"]) == pytest.approx((w5["mean"], w5["se"], w5["mean_minus_3se"]))


def test_subset_stats_ignores_withheld_seeds_whatever_their_score():
    scores, ref, pub, seeds = _case(D8 + [-50.0, -80.0], publishable=[True] * 8 + [False, False])
    st = S.subset_stats(scores, ref, pub)
    assert st["n_seeds"] == 10 and st["n_publishable"] == 8 and st["coverage"] == pytest.approx(0.8)
    assert st["mean"] == pytest.approx(2.0) and st["mean_minus_3se"] == pytest.approx(0.86610658, abs=1e-7)
    scores[seeds[-1]] = +1e6                                                      # negative control: a withheld fit's score cannot move anything
    assert S.subset_stats(scores, ref, pub)["mean"] == pytest.approx(2.0)
    # flipping one withheld fit to publishable DOES move it
    pub[seeds[-2]] = True
    assert S.subset_stats(scores, ref, pub)["n_used"] == 9 and S.subset_stats(scores, ref, pub)["mean"] != pytest.approx(2.0)


def test_subset_stats_judged_flag_is_exactly_at_eight_and_counts_nonfinite():
    scores, ref, pub, seeds = _case(D8[:7])
    assert S.subset_stats(scores, ref, pub)["judged"] is False and S.MIN_PUBLISHABLE == 8
    scores, ref, pub, seeds = _case(D8)
    assert S.subset_stats(scores, ref, pub)["judged"] is True
    scores[seeds[0]] = float("nan")                                               # a publishable fit with a non-finite score: dropped AND counted
    st = S.subset_stats(scores, ref, pub)
    assert st["n_used"] == 7 and st["n_nonfinite_publishable"] == 1 and st["n_publishable"] == 8 and st["judged"] is False
    assert st["coverage"] == 1.0                                                  # coverage is the share of PUBLISHABLE fits, whatever their score
    scores, ref, pub, seeds = _case(D8)
    ref[seeds[1]] = float("inf")                                                  # a non-finite reference score is dropped the same way
    st = S.subset_stats(scores, ref, pub)
    assert st["n_used"] == 7 and st["n_nonfinite_publishable"] == 1
    assert math.isnan(S.subset_stats({1: 1.0}, {1: 0.0}, {1: True})["mean"])        # fewer than two pairs: no statistic, no crash


# ---------------------------------------------------------------------------------------------------------------- risk-coverage (withheld = no prediction)
def test_fallback_stats_counts_a_withheld_fit_as_a_zero_difference():
    scores, ref, pub, seeds = _case(D8 + [-50.0] * 4, publishable=[True] * 8 + [False] * 4)
    fb = S.fallback_stats(scores, ref, pub)
    d = np.array(D8 + [0.0] * 4)
    assert fb["n_seeds"] == 12 and fb["n_served"] == 8 and fb["coverage"] == pytest.approx(8 / 12)
    assert fb["mean"] == pytest.approx(16 / 12) and fb["mean"] == pytest.approx(d.mean())
    assert fb["se"] == pytest.approx(d.std(ddof=1) / math.sqrt(12)) and fb["mean_minus_3se"] == pytest.approx(d.mean() - 3 * d.std(ddof=1) / math.sqrt(12))
    assert fb["mean_minus_3se"] == pytest.approx(0.2051, abs=1e-3)
    scores[seeds[-1]] = 1e6                                                       # negative control: the score of a withheld fit is irrelevant
    assert S.fallback_stats(scores, ref, pub)["mean"] == pytest.approx(16 / 12)


def test_fallback_stats_edge_cases_all_withheld_and_nonfinite_publishable():
    scores, ref, pub, seeds = _case(D8, publishable=[False] * 8)
    fb = S.fallback_stats(scores, ref, pub)
    assert fb["mean"] == 0.0 and fb["se"] == 0.0 and fb["mean_minus_3se"] == 0.0 and fb["coverage"] == 0.0 and fb["n_served"] == 0
    scores, ref, pub, seeds = _case(D8)
    scores[seeds[0]] = float("nan")                                               # publishable but no finite score: no prediction, difference 0, not served
    fb = S.fallback_stats(scores, ref, pub)
    assert fb["n_served"] == 7 and fb["mean"] == pytest.approx((16 - 1.0) / 8)
    # when every fit is publishable the risk-coverage mean equals the subset mean
    scores, ref, pub, _ = _case(D8)
    assert S.fallback_stats(scores, ref, pub)["mean"] == pytest.approx(S.subset_stats(scores, ref, pub)["mean"])
    # a seed missing from the reference is not counted at all
    scores, ref, pub, seeds = _case(D8)
    del ref[seeds[0]]
    assert S.fallback_stats(scores, ref, pub)["n_seeds"] == 7
    scores, ref, pub, seeds = _case(D8)
    ref[seeds[0]] = float("nan")                                                  # and neither is a seed whose reference score is not finite
    fb = S.fallback_stats(scores, ref, pub)
    assert fb["n_seeds"] == 7 and fb["n_served"] == 7 and math.isfinite(fb["mean"])


# ---------------------------------------------------------------------------------------------------------------- the verdict word
def _st(judged, lower):
    return dict(judged=judged, mean_minus_3se=lower)


def test_survives_words():
    ok = {30: _st(True, 0.2), 60: _st(True, 0.5), 120: _st(True, 0.9)}
    assert S.survives(ok) == "survives"
    assert S.survives({**ok, 60: _st(True, -0.1)}) == "does not survive"
    assert S.survives({**ok, 60: _st(True, 0.0)}) == "does not survive"               # strictly greater than zero is required
    assert S.survives({**ok, 60: _st(True, float("nan"))}) == "does not survive"
    assert S.survives({30: _st(False, 5.0), 60: _st(True, 0.5), 120: _st(True, 0.9)}) == "partly judged"
    assert S.survives({30: _st(False, -9.0), 60: _st(True, 0.5), 120: _st(True, 0.9)}) == "partly judged"   # an unjudged length cannot veto
    assert S.survives({30: _st(False, 5.0), 60: _st(True, -0.5), 120: _st(True, 0.9)}) == "does not survive"  # a judged failure always counts
    assert S.survives({30: _st(False, 5.0), 60: _st(False, 5.0)}) == "cannot be judged"
    assert S.survives({}) == "cannot be judged"
