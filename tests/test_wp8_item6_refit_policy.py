"""WP8 Item 6 (owner decision A5): the refit cadence is a configuration (default weekly, bounds 1..28 days) and `should_refit` is pure.
Mutation check: integration_report/wp8/scripts/mutation_item6.py. The cadence study (daily, weekly, every 14 days) is integration_report/wp8/scripts/item6_study.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backbone.hssm import refit_policy as R
from backbone.hssm.config import MAX_TIMESTAMP_SPAN_DAYS
from backbone.hssm.refit_policy import RefitPolicy, next_refit_day, refit_schedule, should_refit

ROOT = Path(__file__).resolve().parents[1]


def test_the_default_is_weekly_and_the_bounds_are_one_to_twenty_eight_days():
    assert RefitPolicy().refit_interval_days == 7 and R.DEFAULT_REFIT_INTERVAL_DAYS == 7
    assert (R.MIN_REFIT_INTERVAL_DAYS, R.MAX_REFIT_INTERVAL_DAYS) == (1, 28)
    assert RefitPolicy(1).refit_interval_days == 1 and RefitPolicy(28).refit_interval_days == 28 and RefitPolicy(np.int64(14)).refit_interval_days == 14


@pytest.mark.parametrize("bad", [0, -1, 29, 100, 7.5, 7.0, True, False, "7", None, [7]])
def test_an_interval_outside_the_bounds_or_not_a_whole_number_of_days_is_refused(bad):
    with pytest.raises(ValueError):
        RefitPolicy(bad)


@pytest.mark.parametrize("interval", range(1, 29))
def test_should_refit_follows_the_interval_exactly_for_every_allowed_interval(interval):
    p = RefitPolicy(interval)
    for gap in range(0, 60):
        assert should_refit(100, 100 + gap, p) == (gap >= interval), (interval, gap)
    assert next_refit_day(100, p) == 100 + interval


def test_a_user_without_an_earlier_refit_is_due():
    assert should_refit(None, 0) and should_refit(None, 150, RefitPolicy(28))


def test_should_refit_is_pure_and_the_policy_is_immutable():
    p = RefitPolicy(10)
    results = {should_refit(5, 20, p) for _ in range(5)}
    assert results == {True}
    with pytest.raises(Exception):
        p.refit_interval_days = 3                                                       # frozen dataclass
    assert p.refit_interval_days == 10
    import inspect
    src = inspect.getsource(R.should_refit)
    assert "time." not in src and "datetime" not in src and "global " not in src          # no clock, no state


def test_invalid_days_fail_closed():
    for last, today in ((10, 9), (150, 0)):
        with pytest.raises(ValueError, match="before"):
            should_refit(last, today)
    for last, today in ((1.5, 10), (1, 10.5), ("1", 10), (True, 10), (1, True), (1, None)):
        with pytest.raises(ValueError):
            should_refit(last, today)
    with pytest.raises(ValueError, match="calendar days"):
        should_refit(0, MAX_TIMESTAMP_SPAN_DAYS + 1)
    assert should_refit(0, MAX_TIMESTAMP_SPAN_DAYS)                                     # the guard is strict: exactly the limit is accepted
    assert should_refit(np.int32(3), np.int64(40))


@pytest.mark.parametrize("interval", range(1, 29))
def test_the_schedule_is_an_arithmetic_progression_whatever_the_interval(interval):
    days = refit_schedule(90, 150, RefitPolicy(interval))
    assert days == list(range(90, 151, interval))
    assert all(should_refit(a, b, RefitPolicy(interval)) for a, b in zip(days[:-1], days[1:]))


def test_changing_the_interval_needs_a_config_change_only():
    daily = RefitPolicy.from_config({"refit_interval_days": 1})
    weekly = RefitPolicy.from_config({})
    fortnight = RefitPolicy.from_config({"refit_interval_days": 14})
    assert [len(refit_schedule(90, 150, p)) for p in (daily, weekly, fortnight)] == [61, 9, 5]
    assert weekly == RefitPolicy() and refit_schedule(90, 150, fortnight) == [90, 104, 118, 132, 146]
    for bad in ({"interval": 3}, {"refit_interval_days": 3, "extra": 1}, {"refit_interval_days": 0}):
        with pytest.raises(ValueError):
            RefitPolicy.from_config(bad)
    import ast, inspect
    consts = [n.value for n in ast.walk(ast.parse(inspect.getsource(R))) if isinstance(n, ast.Constant) and n.value == 7 and not isinstance(n.value, bool)]
    assert len(consts) == 1                                                             # 7 appears once, as the default; no other code depends on it


def test_the_contract_documents_the_default_and_the_bounds():
    doc = (ROOT / "docs" / "INTEGRATION_CONTRACT_HSSM.md").read_text()
    sec = doc[doc.index("### 9.10"):]
    for phrase in ("refit_interval_days", "default 7", "1 to 28", "should_refit"):
        assert phrase in sec, phrase
