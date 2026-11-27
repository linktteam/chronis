"""WP10B C2: the hard-stratum ceiling analysis (`integration_report/wp10b/scripts/c2_analyze.py`; rules: tests/PREDECLARED_CRITERIA_WP10B.md section 5 and Part 2). Synthetic windows and row sets; no fit.
The script is loaded by path under a unique module name."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("wp10b_c2_under_test", ROOT / "integration_report" / "wp10b" / "scripts" / "c2_analyze.py")
C = importlib.util.module_from_spec(_spec)
sys.modules["wp10b_c2_under_test"] = C
_spec.loader.exec_module(C)

from backbone.hssm.lineage import separation_min  # noqa: E402


def window(sep=3.0, n=(30, 30, 30), F=4, seed=0, nan_feature_in_regime=None):
    """Three regimes with unit within-regime sd whose means are `sep` apart along feature 0 (so the smallest pair distance is `sep`)."""
    rng = np.random.default_rng(seed)
    means = np.zeros((3, F)); means[:, 0] = [0.0, sep, 2 * sep]
    lab = np.array(sum(([k] * m for k, m in enumerate(n)), []), dtype=int)
    X = means[lab] + rng.standard_normal((len(lab), F))
    if nan_feature_in_regime is not None:
        X[lab == nan_feature_in_regime, 1] = np.nan
    return X, lab, means[lab]


# ---- the oracle quantity ----------------------------------------------------------------------------------------------------------------------------------------------------
def test_the_oracle_gate_quantity_is_separation_min_on_the_true_labels_and_sample_means():
    X, lab, _ = window()
    o = C.oracle_row(X, lab)
    means = np.array([X[lab == k].mean(axis=0) for k in range(3)])
    assert o["evaluable"] and o["sep_emp"] == separation_min(means, X, lab)


def test_a_well_separated_window_passes_the_gate_and_an_unseparated_one_does_not():
    assert C.passes(C.oracle_row(*window(sep=4.0)[:2])["sep_emp"])
    assert not C.passes(C.oracle_row(*window(sep=0.3)[:2])["sep_emp"])


def test_the_generator_means_arm_uses_the_per_day_true_means_averaged_by_regime():
    X, lab, md = window(sep=3.0)
    o = C.oracle_row(X, lab, md)
    gm = np.array([md[lab == k].mean(axis=0) for k in range(3)])
    assert o["sep_gen"] == separation_min(gm, X, lab)


@pytest.mark.parametrize("n,ev", [((30, 30, 2), False), ((2, 30, 30), False), ((30, 30, 3), True), ((3, 3, 3), True), ((30, 30, 0), False)])
def test_a_regime_with_fewer_than_three_days_makes_the_window_non_evaluable(n, ev):
    X, lab, _ = window(n=n)
    o = C.oracle_row(X, lab)
    assert o["evaluable"] is ev
    assert o["counts"] == list(n)
    if not ev:
        assert math.isnan(o["sep_emp"])


def test_a_regime_without_any_observed_value_of_a_feature_is_non_evaluable():
    X, lab, _ = window(nan_feature_in_regime=1)
    assert not C.oracle_row(X, lab)["evaluable"]


def test_missing_entries_inside_a_regime_are_ignored_by_the_sample_mean():
    X, lab, _ = window()
    X[::7, 1] = np.nan
    o = C.oracle_row(X, lab)
    assert o["evaluable"] and math.isfinite(o["sep_emp"])


# ---- production arm ---------------------------------------------------------------------------------------------------------------------------------------------------------
def prep(means, labels, raw, max_post=0.99, unstable=False):
    return dict(r8=dict(raw_means=means.tolist(), labels_fit=labels.tolist(), max_post=max_post), cons=dict(fit_unstable=unstable), raw=raw)


def test_production_row_without_a_usable_fit_is_unusable():
    r = C.production_row(None)
    assert r["usable"] is False and math.isnan(r["sep"])


def test_production_separation_uses_the_fitted_means_and_labels_not_the_true_ones():
    X, lab, _ = window(sep=3.0)
    means = np.array([X[lab == k].mean(axis=0) for k in range(3)])
    shuffled = lab.copy(); shuffled[:45] = 1                                                       # a wrong labelling
    r = C.production_row(prep(means, shuffled, X))
    assert r["usable"] and r["sep"] == separation_min(means, X, shuffled) and r["sep"] != separation_min(means, X, lab)


@pytest.mark.parametrize("max_post,unstable,flag", [(0.99, False, False), (0.94, False, True), (0.95, False, False), (0.99, True, True)])
def test_the_production_flag_is_the_consensus_flag_or_a_sharpness_below_the_floor(max_post, unstable, flag):
    X, lab, _ = window()
    means = np.array([X[lab == k].mean(axis=0) for k in range(3)])
    assert C.production_row(prep(means, lab, X, max_post=max_post, unstable=unstable))["flag"] is flag


# ---- aggregation ------------------------------------------------------------------------------------------------------------------------------------------------------------
def row(user, stratum, ev=True, o=3.0, g=3.0, usable=True, p=3.0, flag=False, end=90):
    return dict(user=user, stratum=stratum, end=end, oracle=dict(evaluable=ev, counts=[9, 9, 9], sep_emp=o if ev else float("nan"), sep_gen=g if ev else float("nan")),
                prod=dict(usable=usable, sep=p if usable else float("nan"), flag=flag if usable else None), failed=False)


def test_user_table_counts_only_evaluable_windows_and_production_needs_a_usable_fit():
    rows = [row(1, "hard", o=3.0, p=3.0), row(1, "hard", o=2.0, p=2.6), row(1, "hard", ev=False, o=9.0, p=9.0), row(1, "hard", o=3.0, usable=False), row(1, "hard", o=3.0, p=3.0, flag=True)]
    u = C.user_table(rows)
    assert u == [dict(user=1, n=4, o=3, g=4, p=3, q=2)]


def test_gate_level_is_inclusive():
    assert C.passes(2.5) and not C.passes(2.4999999) and not C.passes(float("nan")) and not C.passes(float("inf") * -1)


def test_summarize_separates_strata_and_leaves_non_evaluable_windows_out_of_every_fraction():
    rows = [row(1, "hard", o=3.0, p=1.0), row(1, "hard", o=1.0, p=1.0), row(1, "hard", ev=False, o=0, p=9.0), row(2, "medium", o=3.0, p=3.0), row(2, "medium", o=3.0, p=3.0)]
    s = C.summarize(rows)
    assert s["hard"]["refits"] == 3 and s["hard"]["evaluable"] == 2 and s["hard"]["non_evaluable"] == 1
    assert s["hard"]["pO"][0] == 0.5 and s["hard"]["pP_sep"][0] == 0.0                              # the non-evaluable window where production would pass is not counted
    assert s["hard"]["gap_sep"][0] == 0.5
    assert s["medium"]["pO"][0] == 1.0 and s["medium"]["pP_sep"][0] == 1.0 and "easy" not in s


def test_the_gate_share_excludes_flagged_fits_and_the_separation_share_does_not():
    rows = [row(1, "easy", o=3.0, p=3.0, flag=True), row(1, "easy", o=3.0, p=3.0, flag=False)]
    s = C.summarize(rows)["easy"]
    assert s["pP_sep"][0] == 1.0 and s["pP_gate"][0] == 0.5


def test_users_ever_counts_users_not_windows():
    rows = [row(1, "hard", o=3.0), row(1, "hard", o=3.0), row(2, "hard", o=1.0), row(3, "hard", o=3.0, p=1.0)]
    s = C.summarize(rows)["hard"]
    assert s["users"] == 3 and s["users_oracle_ever"] == 2 and s["users_production_gate_ever"] == 2


# ---- the bootstrap resamples USERS ------------------------------------------------------------------------------------------------------------------------------------------------
def test_user_bootstrap_resamples_users_not_rows():
    users = [dict(user=1, n=10, o=10), dict(user=2, n=10, o=0)]
    point, lo, hi = C.boot_users(users, lambda u: C.ratio(u, "o"))
    assert point == 0.5 and lo == 0.0 and hi == 1.0                                                  # two users: a resample can contain two copies of either one; resampling the 20 rows would give about 0.3..0.7


def test_user_bootstrap_is_deterministic_and_covers_the_point_estimate():
    users = [dict(user=i, n=10, o=i % 4 * 3) for i in range(20)]
    a, b = C.boot_users(users, lambda u: C.ratio(u, "o")), C.boot_users(users, lambda u: C.ratio(u, "o"))
    assert a == b and a[1] <= a[0] <= a[2]


def test_the_paired_gap_bootstrap_uses_the_same_resampled_users_for_both_arms():
    users = [dict(user=i, n=10, o=10, p=10) for i in range(10)]                                      # production equals the oracle in every user: the paired gap is exactly 0 in every resample
    point, lo, hi = C.boot_users(users, lambda u: C.ratio(u, "o") - C.ratio(u, "p"))
    assert (point, lo, hi) == (0.0, 0.0, 0.0)


# ---- the interpretation rule ------------------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("pO,hi,gap,glo,expect", [
    (0.30, 0.45, 0.10, -0.1, "NOT SERVABLE BY DESIGN (demonstrated"),
    (0.49, 0.60, 0.10, -0.1, "NOT SERVABLE BY DESIGN (on the point estimate"),
    (0.499999, 0.5, 0.0, 0.0, "NOT SERVABLE BY DESIGN (on the point estimate"),
    (0.499999, 0.499999, 0.0, 0.0, "NOT SERVABLE BY DESIGN (demonstrated"),
    (0.50, 0.70, 0.20, 0.01, "NEEDS BETTER FITTING"),
    (0.50, 0.70, 0.19, 0.10, "MIXED"),
    (0.80, 0.90, 0.30, 0.0, "MIXED"),
    (0.80, 0.90, 0.30, -0.05, "MIXED"),
    (0.80, 0.90, 0.05, 0.01, "MIXED"),
])
def test_verdict_hard(pO, hi, gap, glo, expect):
    assert C.verdict_hard(pO, hi, gap, glo).startswith(expect)


def test_the_levels_are_the_declared_ones():
    assert (C.H, C.MIN_DAYS, C.NOT_SERVABLE_LEVEL, C.GAP_LEVEL, C.BOOT, C.BOOT_SEED, C.SHARP) == (2.5, 3, 0.50, 0.20, 2000, 11999, 0.95)


def test_wilson_interval():
    lo, hi = C.wilson(0, 10)
    assert lo == 0.0 and math.isclose(hi, 0.27753, abs_tol=1e-4)
    lo, hi = C.wilson(5, 10)
    assert math.isclose(lo, 0.2366, abs_tol=1e-3) and math.isclose(hi, 0.7634, abs_tol=1e-3)
    assert math.isnan(C.wilson(0, 0)[0])


def test_quartile_and_interval_formatting_put_the_point_first():
    assert C.fmtq((1.0, 2.0, 3.0)) == "2.00 [1.00, 3.00]"
    assert C.fmt3((0.5, 0.25, 0.75)) == "0.500 [0.250, 0.750]"


def test_the_row_level_bootstrap_would_be_much_narrower_than_the_user_level_one():
    """Two users, one passing everywhere and one nowhere: the user-level interval spans 0 to 1 (a resample can hold two copies of the same user); a bootstrap over the 20 pooled rows would give about 0.3 to 0.7."""
    users = [dict(user=1, n=10, o=10, p=0), dict(user=2, n=10, o=0, p=0)]
    point, lo, hi = C.boot_users(users, lambda u: C.ratio(u, "o"))
    assert hi - lo == 1.0


def test_boot_users_is_seeded_and_the_default_seed_is_the_declared_one():
    users = [dict(user=i, x=float(i) ** 1.3 + 0.37 * (i % 3)) for i in range(15)]
    stat = lambda u: float(np.mean([v["x"] for v in u]))                                                # a continuous statistic: two unseeded runs would not agree
    assert C.boot_users(users, stat, B=300) == C.boot_users(users, stat, B=300, seed=C.BOOT_SEED)
    assert C.boot_users(users, stat, B=300, seed=1) == C.boot_users(users, stat, B=300, seed=1)
    assert C.boot_users(users, stat, B=300, seed=1) != C.boot_users(users, stat, B=300, seed=2)


def test_summarize_pairs_the_gap_inside_each_resample():
    """Production passes in exactly the windows where the oracle passes, user by user: the paired gap is 0 in every resample (interval 0..0) although the share itself varies between resamples."""
    rows = []
    for u in range(10):
        for w in range(4):
            ok = w < (u % 5)
            rows.append(row(100 + u, "hard", o=3.0 if ok else 1.0, p=3.0 if ok else 1.0, end=90 + 7 * w))
    s = C.summarize(rows)["hard"]
    assert s["gap_sep"] == (0.0, 0.0, 0.0)
    assert s["pO"][2] - s["pO"][1] > 0.05 and s["pO"][0] == s["pP_sep"][0]


def test_refit_rows_marks_a_failed_fit_as_non_evaluable_and_unusable():
    rows = C.refit_rows([dict(job=dict(user=1, stratum="hard", end=90, start=0), ok=False)])
    assert rows[0]["failed"] is True and rows[0]["oracle"]["evaluable"] is False and rows[0]["prod"]["usable"] is False


def test_refit_rows_end_to_end_on_a_small_real_fit():
    """A real (small) fit of a spare-block user: the oracle uses the generator's true labels of the window and the production arm the selected restart; both agree with an independent computation."""
    sys.path.insert(0, str(ROOT / "integration_report" / "wp9" / "scripts"))
    import wp9_fit_lib as F
    from backbone.hssm.lineage import separation_min as sep_min
    job = dict(user=11901, start=0, end=90, fitseed=11901, K=3, n_init=3, max_iter=30, stratum="easy", drift=0.0, kchange=None)
    rec = F.fit_job(job)
    assert rec["ok"]
    r = C.refit_rows([rec])[0]
    st = F.stream_for(job)
    w = st.window(0, 90)
    assert r["oracle"]["evaluable"] and r["oracle"]["sep_emp"] == C.oracle_row(w["raw"], w["labels"], st.truth["means_t"][0:90])["sep_emp"]
    assert F.labels_to_str(w["labels"]) == rec["labels_true"]
    sel = F.select_restart(rec["restarts"])
    if sel is not None and not sel["collapsed"]:
        assert r["prod"]["usable"]
        z = np.asarray(sel["z_means"]); raw_means = np.asarray(rec["scaler_mean"]) + np.asarray(rec["scaler_std"]) * z
        assert r["prod"]["sep"] == sep_min(raw_means, w["raw"], F.str_to_labels(sel["labels"]))
