"""WP10B B2: the dual-engine bit-identity comparison (`integration_report/wp10b/scripts/dual_battery.py`): PASS only if every field of every fit is bit-identical."""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "integration_report" / "wp10b" / "scripts"
_spec = importlib.util.spec_from_file_location("wp10b_dual_battery_under_test", SC / "dual_battery.py")
B = importlib.util.module_from_spec(_spec)
sys.modules["wp10b_dual_battery_under_test"] = B
_spec.loader.exec_module(B)


def rec(seed=1):
    r = dict(seed=seed, variant="pop", ll_history=[-10.0, -9.5, -9.4], fit_state="MAX_ITER_REACHED", n_iter=3, criterion=None, withheld_reason="x", Dmax=25, box_refit=None, window_gain_at_cap=None, seconds=1.0)
    for i, n in enumerate(B.PARAMS):
        r[n] = np.arange(4.0).reshape(2, 2) * (i + 1)
    return r


def test_identical_records_pass_and_the_wall_time_is_ignored():
    a, b = [rec(1), rec(2)], copy.deepcopy([rec(1), rec(2)])
    b[0]["seconds"] = 99.0
    ok, rows = B.compare_records(a, b)
    assert ok and all(not d for *_, d in rows)


def test_a_one_ulp_difference_in_any_parameter_fails_and_names_it():
    for n in B.PARAMS:
        a, b = [rec()], [rec()]
        x = b[0][n]
        x[1, 1] = np.nextafter(x[1, 1], np.inf)
        ok, rows = B.compare_records(a, b)
        assert not ok and rows[0][3] == [n]


def test_any_other_field_difference_fails():
    for k, v in (("ll_history", [-10.0, -9.5, -9.3999999]), ("fit_state", "CONVERGED"), ("n_iter", 4), ("criterion", "abs_tol"), ("withheld_reason", None), ("Dmax", 26),
                 ("box_refit", {"first_fit_n_iter": 3.0}), ("window_gain_at_cap", 0.1)):
        b = [rec()]
        b[0][k] = v
        ok, rows = B.compare_records([rec()], b)
        assert not ok and rows[0][3] == [k], k


def test_a_different_seed_order_is_refused():
    import pytest
    with pytest.raises(AssertionError):
        B.compare_records([rec(1), rec(2)], [rec(2), rec(1)])


def test_the_job_lists_cover_the_declared_users():
    assert [j[2] for j in B.jobs_for("T", "pop")] == list(range(11911, 11923)) and [j[3] for j in B.jobs_for("T", "pop")] == ["none"] * 6 + ["prior"] * 6
    assert [(j[2], j[3]) for j in B.jobs_for("T", "ident")] == B.IDENT


def test_a_difference_in_the_first_record_is_not_hidden_by_identical_later_records():
    b = [rec(1), rec(2), rec(3)]
    b[0]["n_iter"] = 9
    ok, rows = B.compare_records([rec(1), rec(2), rec(3)], b)
    assert not ok and [r[3] for r in rows] == [["n_iter"], [], []]
