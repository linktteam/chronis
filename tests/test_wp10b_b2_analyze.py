"""WP10B B2: the analysis rules of `integration_report/wp10b/scripts/b2_analyze.py` (Part 4 levels), on fabricated paired records."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "integration_report" / "wp10b" / "scripts"
_spec = importlib.util.spec_from_file_location("wp10b_b2_analyze_under_test", SC / "b2_analyze.py")
A = importlib.util.module_from_spec(_spec)
sys.modules["wp10b_b2_analyze_under_test"] = A
_spec.loader.exec_module(A)


def rec(user, pub, test=0.0, cats=(), state=None, secs=60.0, finite=True, ok=True, clock=False, variant="S0"):
    r = dict(job=dict(user=user, variant=variant), ok=ok)
    if ok:
        r["dual"] = dict(publishable=pub, fit_state=state or ("CONVERGED" if pub else "MAX_ITER_REACHED"), categories=[] if pub else list(cats), test_mean=test, test_finite=finite, fit_seconds=secs, wall_clock_stop=clock)
    return r


def table(spec, variant="S0"):
    return {u: rec(u, **dict(kw, variant=variant)) for u, kw in spec.items()}


def test_exact_sign_test():
    assert A.sign_test_p(5, 0) == pytest.approx(1 / 32) and A.sign_test_p(4, 0) == pytest.approx(1 / 16) and A.sign_test_p(3, 1) == pytest.approx(5 / 16)
    assert A.sign_test_p(0, 0) == 1.0 and A.sign_test_p(0, 3) == 1.0 and A.sign_test_p(6, 2) == pytest.approx(37 / 256)


def test_paired_stats():
    s = A.paired_stats([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4 and s["mean"] == 2.5 and math.isclose(s["se"], np.std([1, 2, 3, 4], ddof=1) / 2) and math.isclose(s["lower"], 2.5 - 1.645 * s["se"])
    assert math.isnan(A.paired_stats([1.0])["mean"])


def test_classification_words():
    assert A.classify(True, True) == "HELPED" and A.classify(True, False) == "HELPED BUT HARMFUL" and A.classify(False, True) == "NO EFFECT DEMONSTRATED" and A.classify(False, False) == "HARMED"


def base(n=12, pub=0):
    return {u: dict(pub=(u - 1 < pub), test=0.0, cats=("iteration_cap",)) for u in range(1, n + 1)}


def test_gain_needs_g_minus_l_at_least_one_and_p_below_005():
    ref = table(base())
    for g, want in ((5, True), (4, False)):
        spec = {u: dict(pub=(u <= g), test=0.0, cats=("iteration_cap",)) for u in range(1, 13)}
        assert A.compare_variant(ref, table(spec, "V"))["gain_demonstrated"] is want
    ref2 = table({u: dict(pub=(u <= 2), test=0.0, cats=("iteration_cap",)) for u in range(1, 13)})
    spec = {u: dict(pub=(3 <= u <= 8), test=0.0, cats=("iteration_cap",)) for u in range(1, 13)}          # lost 2, gained 6: p = 0.1445
    c = A.compare_variant(ref2, table(spec, "V"))
    assert (c["gain"], c["loss"]) == (6, 2) and c["gain_demonstrated"] is False and c["sign_p"] == pytest.approx(37 / 256)


def test_a_variant_fit_that_failed_to_run_counts_as_not_publishable_and_a_loss_if_s0_was_publishable():
    ref = table({1: dict(pub=True), 2: dict(pub=False, cats=("iteration_cap",))})
    new = table({1: dict(pub=True), 2: dict(pub=False, cats=("iteration_cap",))}, "V")
    new[1] = rec(1, False, ok=False, variant="V")
    c = A.compare_variant(ref, new)
    assert c["failed_to_run"] == 1 and c["loss"] == 1 and c["pub_new"] == 0 and c["n_paired"] == 1


def test_heldout_pairs_skip_nonfinite_and_use_new_minus_reference():
    ref = table({1: dict(pub=False, test=1.0), 2: dict(pub=False, test=2.0), 3: dict(pub=False, test=3.0)})
    new = table({1: dict(pub=False, test=1.5), 2: dict(pub=False, test=float("nan"), finite=False), 3: dict(pub=False, test=3.5)}, "V")
    h = A.compare_variant(ref, new)["heldout"]
    assert h["n"] == 2 and h["mean"] == pytest.approx(0.5)


def test_not_harmed_boundary_is_inclusive_and_nan_is_not_harmless():
    n = 25
    for shift, want in ((-0.09, True), (-0.15, False)):
        ref = table({u: dict(pub=False, test=0.0) for u in range(1, n + 1)})
        new = table({u: dict(pub=False, test=shift + (0.01 if u % 2 else -0.01)) for u in range(1, n + 1)}, "V")
        assert A.compare_variant(ref, new)["not_harmed"] is want
    ref = table({1: dict(pub=False), 2: dict(pub=False)})
    new = table({1: dict(pub=False, finite=False, test=float("nan")), 2: dict(pub=False)}, "V")
    assert A.compare_variant(ref, new)["not_harmed"] is False
    exact = A.compare_variant(table({u: dict(pub=False, test=0.0) for u in range(1, 6)}), table({u: dict(pub=False, test=-0.1) for u in range(1, 6)}, "V"))
    assert exact["heldout"]["lower"] == -0.1 and exact["not_harmed"] is True                  # the level is inclusive


def test_categories_count_only_non_publishable_fits_and_the_clock_and_time_ratio_are_reported():
    ref = table({1: dict(pub=False, secs=60.0), 2: dict(pub=False, secs=80.0), 3: dict(pub=False, secs=100.0)})
    new = table({1: dict(pub=True, secs=180.0), 2: dict(pub=False, cats=("iteration_cap", "degenerate_duration"), secs=240.0, clock=True), 3: dict(pub=False, cats=("rejected_update",), secs=600.0)}, "V")
    c = A.compare_variant(ref, new)
    assert c["cats"] == {"iteration_cap": 1, "degenerate_duration": 1, "rejected_update": 1} and c["still_iteration_cap"] == 1 and c["clock_stops"] == 1 and c["time_ratio"] == pytest.approx(3.0)


def test_the_converged_reference_fits_that_stop_being_converged_are_counted():
    ref = table({1: dict(pub=True), 2: dict(pub=True), 3: dict(pub=True), 4: dict(pub=False)})
    new = table({1: dict(pub=True), 2: dict(pub=False, state="MAX_ITER_REACHED", cats=("iteration_cap",)), 3: dict(pub=True), 4: dict(pub=False)}, "V")
    assert A.compare_variant(ref, new)["ref_converged_changed"] == 1


def test_squarem_trigger_needs_both_conditions_at_their_boundaries():
    assert A.squarem_trigger(dict(still_iteration_cap=10, time_ratio=3.0))["build"] is True
    assert A.squarem_trigger(dict(still_iteration_cap=9, time_ratio=5.0))["build"] is False
    assert A.squarem_trigger(dict(still_iteration_cap=20, time_ratio=2.99))["build"] is False
    assert A.squarem_trigger(dict(still_iteration_cap=20, time_ratio=float("nan")))["build"] is False


def test_the_stratum_split_follows_the_user_blocks():
    assert [A.strata_of(u) for u in (11201, 11212, 11213, 11224, 11225, 11236)] == ["hard", "hard", "medium", "medium", "easy", "easy"]
    ref = table({11201: dict(pub=False), 11213: dict(pub=False), 11225: dict(pub=False)})
    new = table({11201: dict(pub=False), 11213: dict(pub=True), 11225: dict(pub=True)}, "V")
    s = A.per_stratum(ref, new, A.strata_of)
    assert (s["hard"]["pub_new"], s["medium"]["pub_new"], s["easy"]["pub_new"]) == (0, 1, 1) and s["easy"]["gain"] == 1
