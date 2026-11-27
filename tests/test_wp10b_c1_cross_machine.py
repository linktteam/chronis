"""WP10B C1(3)(b): the comparison functions of `integration_report/wp10b/scripts/c1_cross_machine.py` (what the product sees across two numpy CPU-dispatch configurations)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "integration_report" / "wp10b" / "scripts"
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("wp10b_c1_cross_machine_under_test", SC / "c1_cross_machine.py")
M = importlib.util.module_from_spec(_spec)
sys.modules["wp10b_c1_cross_machine_under_test"] = M
_spec.loader.exec_module(M)


def test_label_agreement_is_invariant_to_renaming_and_reports_exact_equality():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert M.label_agreement(a, a) == (1.0, True)
    b = np.array([2, 2, 0, 0, 1, 1])
    assert M.label_agreement(a, b) == (1.0, False)
    c = np.array([0, 0, 1, 1, 2, 1])
    ag, eq = M.label_agreement(a, c)
    assert eq is False and abs(ag - 5 / 6) < 1e-12


def test_label_agreement_handles_different_numbers_of_labels_and_shapes():
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 0, 1, 2])
    ag, eq = M.label_agreement(a, b)
    assert eq is False and abs(ag - 0.75) < 1e-12
    assert np.isnan(M.label_agreement(a, np.array([0, 1]))[0])


def test_max_abs_diff_rules_for_nan_and_shape():
    nan2 = np.full((3, 2), np.nan)
    assert M.max_abs_diff(nan2, nan2.copy()) == 0.0
    x = np.array([1.0, np.nan, 3.0])
    assert M.max_abs_diff(x, np.array([1.0, np.nan, 3.5])) == 0.5
    assert M.max_abs_diff(x, np.array([1.0, 2.0, 3.0])) == float("inf")
    assert M.max_abs_diff(np.zeros(3), np.zeros(4)) == float("inf")
    assert M.max_abs_diff(np.array([0.0, -2.0]), np.array([0.0, 1.0])) == 3.0


def test_duration_diff_requires_the_same_regimes_and_returns_the_largest_difference():
    a = {0: [1.0, 2.0], 1: [3.0, 4.0]}
    b = {0: [1.0, 2.25], 1: [3.0, 3.0]}
    assert M.duration_diff({k: np.asarray(v) for k, v in a.items()}, {k: np.asarray(v) for k, v in b.items()}) == 1.0
    assert M.duration_diff({0: np.zeros(2)}, {1: np.zeros(2)}) == float("inf")


def rec(seed=1, K=2, shift=0.0, unstable=False, withheld=None, state="CONVERGED"):
    rng = np.random.default_rng(seed)
    T, F = 40, 3
    lab = np.repeat([0, 1, 0, 1], 10)
    X = np.array([[(-3.0 if l == 0 else 3.0)] * F for l in lab]) + rng.standard_normal((T, F)) * 0.3
    r = dict(seed=seed, fit_state=state, K=K, Dmax=45, withheld_reason=withheld, fit_collapsed=False, BIC=100.0 + shift, bic_by_k={2: 100.0 + shift}, m_t_posterior_mean=np.array(None, dtype=object),
             m_t_uncertainty=np.array(None, dtype=object), p_t_estimate=lab.copy(), p_t_posterior=np.eye(2)[lab], regime_means=np.array([[-3.0] * F, [3.0] * F]) + shift, duration_parameters={0: [2.0, 0.3], 1: [2.0, 0.3]},
             restart_summary=dict(fit_unstable=unstable), labels=lab.copy(), raw_means=np.array([[-3.0] * F, [3.0] * F]) + shift, exp_dur=np.array([10.0, 10.0]), separation_min=20.0, unstable=unstable,
             unstable_reason="x" if unstable else None, raw_window=X, scaler_mean=np.zeros(F), scaler_std=np.ones(F), engine="baseline")
    return r


def test_identical_records_are_bit_identical_and_aligned():
    c = M.compare_all([rec(1), rec(2)], [rec(1), rec(2)])
    assert c["bit_identical"] == 2 and c["same_discrete"] == 2 and c["labels_equal"] == 2 and c["max_bic_abs"] == 0 and c["max_abs"]["m_t_posterior_mean"] == 0.0
    assert c["served_ids_equal"] == 2 and c["aligned"] == 2 and c["may_claim"] == 2 and c["lineage_ok"] == 2


def test_a_rounding_difference_is_not_bit_identical_but_keeps_every_discrete_output():
    ref, new = rec(1), rec(1, shift=1e-12)
    c = M.compare_all([ref], [new])
    assert c["bit_identical"] == 0 and c["same_discrete"] == 1 and c["labels_equal"] == 1 and 0 < c["max_bic_abs"] < 1e-9 and c["aligned"] == 1


@pytest.mark.parametrize("field,value", [("fit_state", "MAX_ITER_REACHED"), ("K", 3), ("withheld_reason", "x"), ("fit_collapsed", True), ("Dmax", 40), ("unstable", True)])
def test_every_discrete_field_difference_is_reported(field, value):
    new = rec(1)
    new[field] = value
    c = M.compare_all([rec(1)], [new])
    assert c["same_discrete"] == 0


def test_a_different_label_sequence_is_reported_and_the_lineage_may_not_claim_continuity():
    ref, new = rec(1), rec(1)
    new["labels"] = np.zeros_like(new["labels"])
    new["raw_means"] = np.array([[0.0] * 3, [0.2] * 3])
    new["separation_min"] = 0.1
    c = M.compare_all([ref], [new])
    assert c["labels_equal"] == 0 and c["served_ids_equal"] == 0 and c["may_claim"] == 0 and c["min_label_agree"] < 1.0


def test_a_flagged_fit_is_never_aligned():
    ref, new = rec(1), rec(1, unstable=True)
    c = M.compare_all([ref], [new])
    assert c["aligned"] == 0 and c["may_claim"] == 0 and c["same_discrete"] == 0


def test_the_render_names_the_counts_and_the_differing_users():
    c = M.compare_all([rec(1), rec(2)], [rec(1), rec(2, shift=1e-9)])
    txt = M.render("t", c)
    assert "bit-identical fitted outputs" in txt and ": 1 of 2" in txt and "user 2" in txt and "user 1:" not in txt


def test_seed_order_must_agree():
    with pytest.raises(AssertionError):
        M.compare_all([rec(1), rec(2)], [rec(2), rec(1)])


def test_a_duration_parameter_difference_alone_breaks_bit_identity():
    new = rec(1)
    new["duration_parameters"] = {0: [2.0, 0.3000001], 1: [2.0, 0.3]}
    c = M.compare_all([rec(1)], [new])
    assert c["bit_identical"] == 0 and c["max_duration_abs"] > 0 and c["same_discrete"] == 1


def test_the_summary_takes_the_worst_user_not_the_best():
    new = rec(2)
    new["labels"] = new["labels"].copy()
    new["labels"][:5] = 1 - new["labels"][:5]
    c = M.compare_all([rec(1), rec(2)], [rec(1), new])
    assert c["labels_equal"] == 1 and c["min_label_agree"] < 1.0 and c["bit_identical"] == 2


def test_the_same_fit_fed_twice_separates_the_gate_from_the_machine():
    """A flagged fit is not ALIGNED even with itself: the counts must show that the non-alignment is the gate's, not a cross-configuration effect."""
    c = M.compare_all([rec(1, unstable=True), rec(2)], [rec(1, unstable=True), rec(2)])
    assert c["self_aligned"] == 1 and c["aligned"] == 1 and c["aligned_other_not_self"] == 0 and c["aligned_self_not_other"] == 0
    new = rec(2)
    new["labels"] = np.zeros_like(new["labels"])
    new["raw_means"] = np.array([[0.0] * 3, [0.2] * 3])
    new["separation_min"] = 0.1
    c2 = M.compare_all([rec(2)], [new])
    assert c2["self_aligned"] == 1 and c2["aligned"] == 0 and c2["aligned_self_not_other"] == 1


def test_numeric_restart_summary_differences_are_measured_and_booleans_are_not():
    ref, new = rec(1), rec(1)
    ref["restart_summary"] = dict(gap_to_next_distinct=1.0, fit_unstable=False, n=3)
    new["restart_summary"] = dict(gap_to_next_distinct=1.25, fit_unstable=True, n=3)
    c = M.compare_all([ref], [new])
    assert c["restart_summary_keys_differing"] == ["fit_unstable", "gap_to_next_distinct"] and c["restart_summary_float_abs"] == {"gap_to_next_distinct": 0.25}
