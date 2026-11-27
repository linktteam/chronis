"""WP10B B1: the diagnosis analysis (`integration_report/wp10b/scripts/b1_analyze.py`): counts by reason category and stratum, the exclusive combinations, the duration-degeneracy split and the EM-creep budget table.
Synthetic records; no fit. Loaded by path under a unique module name."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "integration_report" / "wp10b" / "scripts"
sys.path.insert(0, str(SC))
_spec = importlib.util.spec_from_file_location("wp10b_b1_under_test", SC / "b1_analyze.py")
B = importlib.util.module_from_spec(_spec)
sys.modules["wp10b_b1_under_test"] = B
_spec.loader.exec_module(B)


def dual(cats=(), state="MAX_ITER_REACHED", extra=None, degenerate=(), at_bound=(), rising=(), box=False, test=-1.0):
    pub = not cats
    return dict(fit_state="CONVERGED" if pub else state, publishable=pub, categories=list(cats), creep=dict(extra_iterations=extra, rate=0.99 if extra is not None else None), degenerate_regimes=list(degenerate),
                at_bound={k: (k in at_bound) for k in range(3)}, pmf_rising=[k in rising for k in range(3)], box_refit=dict(first_state="X", first_n_iter=5) if box else None, fit_seconds=60.0, optimizer_success=True,
                test_mean=test, test_finite=True)


def rec(user, stratum, d, ok=True, gauss=-2.0, base=-1.5):
    r = dict(job=dict(user=user, stratum=stratum, variant="S0"), ok=ok, dual=d, gauss=dict(test_mean=gauss, test_finite=True), baseline=dict(test_mean=base, test_finite=True))
    return r


def test_split_degeneracy_separates_wall_rising_and_both():
    d = dual(["degenerate_duration"], degenerate=[0, 1, 2], at_bound=[0, 2], rising=[1, 2])
    s = B.split_degeneracy(d)
    assert s == dict(degenerate=3, wall=2, rising=2, both=1, wall_only=1, rising_only=1)


def test_wall_or_rising_flags_of_a_non_degenerate_regime_are_not_counted():
    d = dual([], degenerate=[], at_bound=[0], rising=[1])
    assert B.split_degeneracy(d)["degenerate"] == 0 and B.split_degeneracy(d)["wall"] == 0


def test_summarize_counts_by_stratum_category_and_exclusive_combination():
    recs = [rec(1, "hard", dual(["iteration_cap"], extra=150)), rec(2, "hard", dual(["iteration_cap", "degenerate_duration"], extra=900, degenerate=[1], rising=[1])), rec(3, "hard", dual()),
            rec(4, "easy", dual()), rec(5, "easy", dual(["rejected_update"], state="REJECTED_UPDATE", extra=None)), rec(6, "easy", dual(), ok=False)]
    s = B.summarize(recs)
    h, e = s["hard"], s["easy"]
    assert (h["fits"], h["failed"], h["publishable"], h["non_publishable"]) == (3, 0, 1, 2)
    assert h["cat_counts"]["iteration_cap"] == 2 and h["cat_counts"]["degenerate_duration"] == 1 and h["cat_counts"]["rejected_update"] == 0
    assert h["exclusive"] == {"iteration_cap": 1, "degenerate_duration+iteration_cap": 1}
    assert h["degenerate_regimes"] == 1 and h["deg_rising_only"] == 1 and h["fits_with_degenerate"] == 1
    assert (e["fits"], e["failed"], e["publishable"], e["non_publishable"]) == (3, 1, 1, 1) and e["cat_counts"]["rejected_update"] == 1
    assert "medium" not in s


def test_the_creep_budget_table_counts_fits_extrapolated_within_each_extra_budget():
    recs = [rec(i, "hard", dual(["iteration_cap"], extra=x)) for i, x in enumerate([50, 200, 201, 400, 799, 800, 1600, 1601, None], 1)]
    h = B.summarize(recs)["hard"]
    assert h["n_cap"] == 9 and h["cap_extra_known"] == 8
    assert h["cap_within"] == {200: 2, 400: 4, 800: 6, 1600: 7}
    assert h["cap_extra_median"] == float(__import__("numpy").median([50, 200, 201, 400, 799, 800, 1600, 1601]))


def test_a_fit_that_is_rejected_counts_in_the_not_finished_by_em_group():
    recs = [rec(1, "hard", dual(["rejected_update"], state="REJECTED_UPDATE", extra=10)), rec(2, "hard", dual(["collapsed_regime"], state="CONVERGED"))]
    h = B.summarize(recs)["hard"]
    assert h["n_cap"] == 1 and h["cat_counts"]["collapsed_regime"] == 1


def test_heldout_gaps_are_paired_per_user_and_skip_nonfinite_pairs():
    recs = [rec(1, "easy", dual(test=-1.0), gauss=-2.0, base=-1.5), rec(2, "easy", dual(test=-1.0), gauss=-3.0, base=-1.0), rec(3, "easy", dual(test=-1.0), gauss=-1.0, base=-4.0)]
    recs[2]["baseline"]["test_finite"] = False
    g = B.summarize(recs)["easy"]["dual_minus"]
    assert g["gauss"]["n"] == 3 and math.isclose(g["gauss"]["mean"], (1.0 + 2.0 + 0.0) / 3)
    assert g["baseline"]["n"] == 2 and math.isclose(g["baseline"]["mean"], (0.5 + 0.0) / 2)


def test_wilson_and_fmt():
    lo, hi = B.wilson(0, 10)
    assert lo == 0.0 and math.isclose(hi, 0.27753, abs_tol=1e-4)
    assert B.fmt(5, 10).startswith("5/10 = 50% [24, 76]")
    assert B.fmt(0, 0) == "0/0"
