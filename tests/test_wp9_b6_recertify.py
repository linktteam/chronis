"""WP9 B6: re-certification of withheld fits (design: tests/PREDECLARED_CRITERIA_WP9.md section 25). Mutation check: integration_report/wp9/scripts/mutation_b6.py."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from backbone.hssm import recertify as R
from backbone.hssm.refit_policy import RefitPolicy

SETTINGS = dict(candidate_ks=(2, 3), n_initializations=10, max_iter=60, random_seed=3, user_id="u")


class FakeFit:
    """A stand-in for fit_hssm: records the call and returns a result whose publish decision is set by `withheld_for(n_rows)`."""

    def __init__(self, withheld_for):
        self.calls, self.withheld_for = [], withheld_for

    def __call__(self, matrix, **kw):
        self.calls.append((np.asarray(matrix).shape, kw))
        reasons = self.withheld_for(len(matrix))
        return SimpleNamespace(insights_withheld=bool(reasons), _withheld_reasons=lambda: list(reasons), n=len(matrix))


def X(n, f=3, seed=0):
    return np.random.default_rng(seed).normal(size=(n, f))


def meta(**kw):
    return dict(dict(settings=dict(SETTINGS), earlier_state="STILL_WITHHELD", last_check_day=None), **kw)


# ------------------------------------------------------------------------------------------------------------------- the state follows the result's own decision
def test_a_result_that_is_not_withheld_is_publishable_and_carries_the_result():
    fit = FakeFit(lambda n: [])
    out = R.recertify(X(30), X(30, seed=1), metadata=meta(), fit_fn=fit)
    assert out["state"] == "PUBLISHABLE" and out["reasons"] == [] and out["rules_relaxed"] is False and out["n_rows"] == 60 and out["result"].n == 60
    assert fit.calls[0][0] == (60, 3) and fit.calls[0][1] == SETTINGS


def test_a_withheld_result_stays_withheld_with_its_reasons_and_no_result_is_returned():
    fit = FakeFit(lambda n: ["collapsed_regime: ...", "degenerate_duration: ..."])
    out = R.recertify(X(30), X(30), metadata=meta(), fit_fn=fit)
    assert out["state"] == "STILL_WITHHELD" and out["reasons"] == ["collapsed_regime: ...", "degenerate_duration: ..."] and out["result"] is None and out["rules_relaxed"] is False


def test_more_days_can_cure_a_withheld_fit_and_the_two_steps_use_the_stored_plus_new_rows():
    fit = FakeFit(lambda n: ["not enough"] if n < 60 else [])
    first = R.recertify(X(30), X(10), metadata=meta(), fit_fn=fit)
    assert first["state"] == "STILL_WITHHELD"
    second = R.recertify(np.vstack([X(30), X(10)]), X(20, seed=5), metadata=meta(), fit_fn=fit)
    assert second["state"] == "PUBLISHABLE" and second["n_rows"] == 60


def test_the_publish_decision_is_read_from_the_result_not_computed_here():
    res = SimpleNamespace(insights_withheld=True, _withheld_reasons=lambda: [])                  # inconsistent on purpose: the property decides
    out = R.recertify(X(30), X(5), metadata=meta(), fit_fn=lambda matrix, **kw: res)
    assert out["state"] == "STILL_WITHHELD" and out["result"] is None
    res2 = SimpleNamespace(insights_withheld=False, _withheld_reasons=lambda: ["something"])
    assert R.recertify(X(30), X(5), metadata=meta(), fit_fn=lambda matrix, **kw: res2)["state"] == "PUBLISHABLE"


# ------------------------------------------------------------------------------------------------------------------- nothing is relaxed
def test_there_is_no_parameter_that_loosens_a_rule():
    names = set(inspect.signature(R.recertify).parameters)
    assert names == {"X", "new_rows", "metadata", "timestamps", "new_timestamps", "today", "policy", "fit_fn"}
    assert not any(w in n for n in names for w in ("threshold", "relax", "force", "override", "min_", "tol", "gate"))


@pytest.mark.parametrize("bad", [dict(allow_fast_test_fit=True), dict(config=object()), dict(n_initializations=9), dict(n_initializations=1), dict(min_present_sessions=1), dict(tol=1.0), dict(force=True)])
def test_settings_that_would_relax_a_rule_or_are_unknown_are_refused_and_nothing_is_fitted(bad):
    fit = FakeFit(lambda n: [])
    out = R.recertify(X(30), X(30), metadata=meta(settings=dict(SETTINGS, **bad)), fit_fn=fit)
    assert out["state"] == "NOT_ELIGIBLE" and out["result"] is None and fit.calls == [] and out["rules_relaxed"] is False


def test_the_minimum_number_of_restarts_is_the_production_one_and_ten_is_accepted():
    assert R.MIN_RESTARTS == 10
    assert R.settings_refusal(dict(n_initializations=10)) is None and R.settings_refusal({}) is None
    assert R.settings_refusal(dict(n_initializations=9)) == "n_initializations_below_the_production_minimum"
    assert R.settings_refusal(dict(n_initializations=True)) == "n_initializations_below_the_production_minimum"
    assert R.settings_refusal(dict(n_initializations=10.0)) == "n_initializations_below_the_production_minimum"
    assert R.settings_refusal(dict(allow_fast_test_fit=False)).startswith("unknown_setting")        # even a harmless-looking flag is not passed on: the production default is used


def test_an_earlier_published_state_is_never_touched():
    fit = FakeFit(lambda n: ["x"])
    out = R.recertify(X(30), X(30), metadata=meta(earlier_state="PUBLISHABLE"), fit_fn=fit)
    assert out["state"] == "NOT_ELIGIBLE" and out["reasons"] == ["already_publishable"] and fit.calls == []


def test_a_fit_that_fails_or_a_cold_start_refusal_is_never_publishable():
    from backbone.hssm.gating import ColdStartError

    def boom(matrix, **kw):
        raise RuntimeError("did not converge")

    def cold(matrix, **kw):
        raise ColdStartError("fewer than 30")
    a = R.recertify(X(30), X(2), metadata=meta(), fit_fn=boom)
    b = R.recertify(X(10), X(2), metadata=meta(), fit_fn=cold)
    assert a["state"] == "ERROR" and a["reasons"] == ["fit_failed:RuntimeError"] and a["result"] is None
    assert b["state"] == "NOT_ELIGIBLE" and b["reasons"] == ["cold_start_gate"] and b["result"] is None


def test_garbage_input_fails_closed_without_raising():
    for args in ((None, None), ("a", [1]), (X(30), None)):
        out = R.recertify(*args, metadata=meta(), fit_fn=FakeFit(lambda n: []))
        assert out["state"] in ("ERROR", "NOT_DUE", "NOT_ELIGIBLE") and out["result"] is None
    assert R.recertify(X(30), X(5), metadata=None, fit_fn=FakeFit(lambda n: []))["state"] == "PUBLISHABLE"           # missing metadata = no settings = the production defaults of fit_hssm


# ------------------------------------------------------------------------------------------------------------------- shapes, timestamps and the schedule
def test_no_new_days_and_a_shape_mismatch_are_not_fitted():
    fit = FakeFit(lambda n: [])
    assert R.recertify(X(30), np.zeros((0, 3)), metadata=meta(), fit_fn=fit)["state"] == "NOT_DUE"
    assert R.recertify(X(30), [], metadata=meta(), fit_fn=fit)["reasons"] == ["no_new_days"]
    assert R.recertify(X(30), X(5, f=4), metadata=meta(), fit_fn=fit)["reasons"] == ["shape_mismatch"] and fit.calls == []


def test_timestamps_are_joined_and_must_come_in_pairs():
    fit = FakeFit(lambda n: [])
    out = R.recertify(X(30), X(5), metadata=meta(), timestamps=np.arange(30.0), new_timestamps=np.arange(30.0, 35.0), fit_fn=fit)
    assert out["state"] == "PUBLISHABLE" and np.array_equal(fit.calls[0][1]["timestamps"], np.arange(35.0))
    assert R.recertify(X(30), X(5), metadata=meta(), timestamps=np.arange(30.0), fit_fn=fit)["reasons"] == ["timestamps_for_only_part_of_the_rows"]


def test_the_schedule_is_the_refit_policy():
    assert R.recertify_due(None, 100) is True
    assert R.recertify_due(100, 106) is False and R.recertify_due(100, 107) is True and R.recertify_due(100, 100) is False
    assert R.recertify_due(100, 103, RefitPolicy(3)) is True and R.recertify_due(100, 102, RefitPolicy(3)) is False
    fit = FakeFit(lambda n: [])
    assert R.recertify(X(30), X(5), metadata=meta(last_check_day=100), today=103, fit_fn=fit)["reasons"] == ["not_due_yet"] and fit.calls == []
    assert R.recertify(X(30), X(5), metadata=meta(last_check_day=100), today=107, fit_fn=fit)["state"] == "PUBLISHABLE"
    with pytest.raises(ValueError):
        R.recertify_due(100, 99)


def test_the_sweep_runs_only_the_due_items_in_order_and_advances_the_check_day():
    calls = []

    def fit(matrix, **kw):
        calls.append(len(matrix))
        return SimpleNamespace(insights_withheld=len(matrix) < 50, _withheld_reasons=lambda: ["short"] if len(matrix) < 50 else [])
    items = [dict(key="a", X=X(30), new_rows=X(10), metadata=meta(last_check_day=100)),             # due: 107 - 100 >= 7
             dict(key="b", X=X(30), new_rows=X(30), metadata=meta(last_check_day=105)),             # not due
             dict(key="c", X=X(30), new_rows=X(30), metadata=meta(last_check_day=None)),            # never checked: due
             dict(key="d", X=X(30), new_rows=X(30), metadata=meta(last_check_day=200))]             # a check day in the future: an error record, the others go on
    out = R.recertification_sweep(items, 107, fit_fn=fit)
    assert [o["key"] for o in out] == ["a", "b", "c", "d"] and calls == [40, 60]
    assert [o["state"] for o in out] == ["STILL_WITHHELD", "NOT_DUE", "PUBLISHABLE", "ERROR"]
    assert out[0]["next_metadata"]["last_check_day"] == 107 and out[0]["next_metadata"]["earlier_state"] == "STILL_WITHHELD"
    assert out[1]["next_metadata"]["last_check_day"] == 105 and out[2]["next_metadata"]["earlier_state"] == "PUBLISHABLE"
    assert all(o["rules_relaxed"] is False for o in out)


def test_a_published_item_is_not_refitted_by_a_later_sweep():
    out1 = R.recertification_sweep([dict(key="c", X=X(30), new_rows=X(30), metadata=meta())], 107, fit_fn=FakeFit(lambda n: []))
    fit = FakeFit(lambda n: ["x"])
    out2 = R.recertification_sweep([dict(key="c", X=X(60), new_rows=X(7), metadata=out1[0]["next_metadata"])], 114, fit_fn=fit)
    assert out2[0]["state"] == "NOT_ELIGIBLE" and out2[0]["reasons"] == ["already_publishable"] and fit.calls == []


# ------------------------------------------------------------------------------------------------------------------- the real publish rule
def test_the_default_fit_is_the_production_fit_hssm_and_its_own_decision_is_returned():
    from data.synthetic_identity import IdentitySpec, generate_identity_stream
    s = generate_identity_stream(9, 60, IdentitySpec(n_regimes=2, sep_lo=2.0, sep_hi=2.5))
    w = s.window(0, 60)
    Z = w["matrix"]
    out = R.recertify(Z[:35], Z[35:], metadata=dict(settings=dict(candidate_ks=(2,), n_initializations=10, max_iter=40, random_seed=1)))
    assert out["state"] in ("PUBLISHABLE", "STILL_WITHHELD") and out["rules_relaxed"] is False
    if out["state"] == "PUBLISHABLE":
        assert out["result"].insights_withheld is False and out["reasons"] == []
    else:
        assert out["result"] is None and out["reasons"]
