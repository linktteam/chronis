"""WP10B A0: the bit-identity compare script (`integration_report/wp9/scripts/a1_battery.py`) prints PASS against a reference that already has the six WP9 restart_summary keys (it used to print FAIL: its last
check demanded that exactly those six keys be ADDED). Synthetic records only; no fit is run. The script is loaded by path under a unique module name."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("wp10b_a1_battery_under_test", ROOT / "integration_report" / "wp9" / "scripts" / "a1_battery.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

OLD = {"n_restarts": 3, "best_ll": -10.5}


def rec(seed=5301, summary=None, **kw):
    r = dict(seed=seed, m_t_posterior_mean=np.arange(4.0), m_t_uncertainty=np.ones(4), p_t_estimate=np.full(4, 0.5), p_t_posterior=np.full((4, 2), 0.5), regime_means=np.eye(2),
             duration_parameters={0: [1.0, 2.0]}, BIC=12.5, fit_state="CONVERGED", Dmax=45, K=2, insights_withheld=False, withheld_reason=None, fit_collapsed=False,
             restart_summary=dict(OLD if summary is None else summary), bic_by_k={2: 12.5}, seconds=1.0)
    r.update(kw)
    return r


def with_keys(r, keys):
    r = copy.deepcopy(r)
    r["restart_summary"].update({k: 0 for k in keys})
    return r


def test_identical_trees_with_the_six_wp9_keys_in_both_pass():
    six = {k: 0 for k in B.NEW_KEYS}
    ref = [rec(summary={**OLD, **six})]
    c = B.compare_records(ref, copy.deepcopy(ref))
    assert c["ok"] and c["added_only"] == 1 and c["old_keys"] == 1 and c["fit"] == 1


def test_a_wp8_reference_still_accepts_exactly_the_six_new_keys():
    ref = [rec()]
    assert B.compare_records(ref, [with_keys(ref[0], B.NEW_KEYS)])["ok"]


def test_a_partial_set_of_the_six_keys_is_still_refused_when_not_allowed():
    ref = [rec()]
    some = sorted(B.NEW_KEYS)[:2]
    assert not B.compare_records(ref, [with_keys(ref[0], some)])["ok"]


def test_an_unlisted_new_key_is_refused_and_a_listed_one_is_accepted():
    ref = [rec()]
    new = [with_keys(ref[0], ["brand_new_key"])]
    assert not B.compare_records(ref, new)["ok"]
    assert B.compare_records(ref, new, allowed_new=("brand_new_key",))["ok"]


def test_a_removed_old_key_is_refused_even_if_allowed_keys_are_added():
    ref = [rec()]
    new = copy.deepcopy(ref)
    del new[0]["restart_summary"]["best_ll"]
    new[0]["restart_summary"]["brand_new_key"] = 1
    assert not B.compare_records(ref, new, allowed_new=("brand_new_key",))["ok"]


@pytest.mark.parametrize("field,value", [("BIC", 12.5 + 1e-12), ("K", 3), ("withheld_reason", "x"), ("fit_state", "MAX_ITER_REACHED")])
def test_any_changed_output_fails(field, value):
    ref = [rec()]
    new = [rec(**{field: value})]
    assert not B.compare_records(ref, new)["ok"]


def test_a_changed_old_restart_summary_value_fails():
    ref = [rec()]
    new = [rec(summary={**OLD, "best_ll": -10.5000001})]
    assert not B.compare_records(ref, new)["ok"]


def test_a_one_bit_change_in_a_fitted_array_fails():
    ref = [rec()]
    new = copy.deepcopy(ref)
    arr = np.array(new[0]["m_t_posterior_mean"], copy=True)
    arr[2] = np.nextafter(arr[2], 1e9)                                                 # one ulp of one element: any tolerance would accept it
    new[0]["m_t_posterior_mean"] = arr
    assert not B.compare_records(ref, new)["ok"]
