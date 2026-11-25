"""WP8 Item 8, criterion 10: the comparison of the baseline identity battery (`integration_report/wp8/scripts/item8_baseline_battery.py`, function `compare_records`).
It must call two batteries identical only when every fitted output is identical and the BIC differs by exactly K ln(T) (the owner-decided BIC convention B1) or not at all, and it must
refuse unexplained status changes. Mutation check: integration_report/wp8/scripts/mutation_item8_battery.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("item8_baseline_battery", ROOT / "integration_report" / "wp8" / "scripts" / "item8_baseline_battery.py")
B = importlib.util.module_from_spec(_spec)
sys.modules["item8_baseline_battery"] = B
_spec.loader.exec_module(B)

T = 121
K = 2


def rec(seed=5301, bic=1715.5, **kw):
    rng = np.random.default_rng(seed)
    r = dict(seed=seed, m_t_posterior_mean=rng.random(T), m_t_uncertainty=rng.random(T), p_t_estimate=rng.random(T), p_t_posterior=rng.random((T, K)), regime_means=rng.random((K, 7)),
             duration_parameters={"a": [1.0, 2.0]}, BIC=bic, fit_state="CONVERGED", Dmax=25, insights_withheld=False, withheld_reason=None, T=T, fit_collapsed=False, fit_duration_degenerate=False)
    r.update(kw)
    return r


def shifted(r, shift=K * np.log(T), **kw):
    n = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in r.items()}
    n["BIC"] = r["BIC"] - shift
    n.update(kw)
    return n


def test_identical_fits_with_the_declared_bic_shift_pass_and_are_counted():
    ref = [rec(5301), rec(5302)]
    c = B.compare_records(ref, [shifted(r) for r in ref])
    assert c["ok"] and c["n"] == 2 and c["n_fit_identical"] == 2 and c["n_bic_shift_exact"] == 2 and c["n_strict"] == 0
    assert c["classes"]["previously healthy, unchanged"] == [5301, 5302]


def test_an_unchanged_bic_also_passes_and_counts_as_strictly_identical():
    ref = [rec()]
    c = B.compare_records(ref, [shifted(ref[0], shift=0.0)])
    assert c["ok"] and c["n_strict"] == 1


@pytest.mark.parametrize("field", ["m_t_posterior_mean", "m_t_uncertainty", "p_t_estimate", "p_t_posterior", "regime_means"])
def test_one_changed_value_in_any_fitted_array_fails(field):
    r = rec(); n = shifted(r)
    n[field] = np.array(n[field]); n[field].flat[3] += 1e-12
    c = B.compare_records([r], [n])
    assert not c["ok"] and c["n_fit_identical"] == 0


@pytest.mark.parametrize("change", [dict(duration_parameters={"a": [1.0, 2.0000001]}), dict(fit_state="MAX_ITER_REACHED"), dict(Dmax=26)])
def test_changed_duration_parameters_fit_state_or_dmax_fail(change):
    r = rec(); n = shifted(r, **change)
    assert not B.compare_records([r], [n])["ok"]


@pytest.mark.parametrize("wrong_shift", [0.0 + 0.01, K * np.log(T) + 0.01, K * np.log(T) - 1e-6, 1 * np.log(T), K])
def test_a_bic_shift_other_than_k_ln_t_fails(wrong_shift):
    r = rec(); n = shifted(r, shift=wrong_shift)
    c = B.compare_records([r], [n])
    assert not c["ok"] and c["n_bic_shift_exact"] == 0 and c["n_fit_identical"] == 1


def test_a_non_finite_bic_fails():
    r = rec(); n = shifted(r, BIC=float("nan"))
    assert not B.compare_records([r], [n])["ok"]


def test_status_changes_are_classified_and_unexplained_ones_fail():
    r = rec()
    by_r1 = shifted(r, insights_withheld=True, fit_collapsed=True)
    c = B.compare_records([r], [by_r1])
    assert c["ok"] and c["classes"]["previously publishable, newly withheld by R1"] == [5301]
    by_r2 = shifted(r, insights_withheld=True, fit_duration_degenerate=True)
    assert B.compare_records([r], [by_r2])["classes"]["previously publishable, newly withheld by R2"] == [5301]
    both = shifted(r, insights_withheld=True, fit_collapsed=True, fit_duration_degenerate=True)
    assert B.compare_records([r], [both])["classes"]["previously publishable, newly withheld by R1 and R2"] == [5301]
    unexplained = shifted(r, insights_withheld=True)
    c = B.compare_records([r], [unexplained])
    assert not c["ok"] and c["classes"]["NOT EXPLAINED"] == [5301]
    was_withheld = rec(insights_withheld=True)
    assert B.compare_records([was_withheld], [shifted(was_withheld)])["classes"]["already withheld (unchanged status)"] == [5301]
    newly_pub = B.compare_records([was_withheld], [shifted(was_withheld, insights_withheld=False)])
    assert not newly_pub["ok"] and newly_pub["classes"]["NOT EXPLAINED"] == [5301]
    flagged_but_published = B.compare_records([r], [shifted(r, fit_collapsed=True)])
    assert not flagged_but_published["ok"] and flagged_but_published["classes"]["NOT EXPLAINED"] == [5301]


def test_batteries_with_different_seeds_are_refused():
    with pytest.raises(AssertionError):
        B.compare_records([rec(5301)], [shifted(rec(5302))])


def test_the_flag_share_counts_new_flags():
    r = rec()
    c = B.compare_records([r], [shifted(r, insights_withheld=True, fit_collapsed=True)])
    assert c["n_flagged"] == 1 and B.compare_records([r], [shifted(r)])["n_flagged"] == 0
    assert B.compare_records([r], [shifted(r, insights_withheld=True, fit_duration_degenerate=True)])["n_flagged"] == 1
