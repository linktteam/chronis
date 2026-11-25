"""WP8 Item 6 (owner decision A5): how often a user's model is refitted, as a configuration, not code.

`RefitPolicy(refit_interval_days=7)`: weekly by default, changeable to any whole number of days from 1 to 28 with a config change only. `should_refit(last_refit_day, today, policy)` is a pure
function of its arguments (no clock, no state). Days are whole calendar days (the unit of the HSSM contract, docs/INTEGRATION_CONTRACT_HSSM.md section 2).
This module only says WHEN a refit is due. Whether a user is eligible for a fit at all (the cold-start gate, the 4-day regime rule) is decided elsewhere; the identity of the regimes after a
refit is decided by `backbone/hssm/lineage.py`.

Bounds, declared with their reasons (tests/PREDECLARED_CRITERIA_WP8.md 4.6):
  * lower bound 1 day: the data arrive per calendar day, so a refit interval below one day would refit on the same data;
  * upper bound 28 days: the longer the gap between two fits, the further the regimes can drift between them and the less an alignment of the new fit to the previous one can be
    trusted (drift of the regime means is the main way an alignment goes wrong in the WP8 studies); four weeks is also the longest interval that keeps at least one refit per calendar
    month. The Item 8 cadence study measures daily, weekly and every-14-days; intervals above 14 days are bounded by declaration, not by a measurement.
EXPERT_REVIEW(identity-distance): the cadence changes how far apart two aligned fits are, and therefore how large the distances in the alignment are; an expert reviewing the matching
distance should read this module's bounds together with the alignment thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import MAX_TIMESTAMP_SPAN_DAYS

MIN_REFIT_INTERVAL_DAYS = 1
MAX_REFIT_INTERVAL_DAYS = 28
DEFAULT_REFIT_INTERVAL_DAYS = 7


def _whole_day(x, name: str) -> int:
    if isinstance(x, bool) or not hasattr(x, "__index__"):
        raise ValueError(f"{name} must be a whole number of calendar days, got {x!r}")
    return int(x)


@dataclass(frozen=True)
class RefitPolicy:
    refit_interval_days: int = DEFAULT_REFIT_INTERVAL_DAYS

    def __post_init__(self) -> None:
        d = _whole_day(self.refit_interval_days, "refit_interval_days")
        if not (MIN_REFIT_INTERVAL_DAYS <= d <= MAX_REFIT_INTERVAL_DAYS):
            raise ValueError(f"refit_interval_days must lie in [{MIN_REFIT_INTERVAL_DAYS}, {MAX_REFIT_INTERVAL_DAYS}], got {d}")
        object.__setattr__(self, "refit_interval_days", d)

    @classmethod
    def from_config(cls, config: dict | None = None) -> "RefitPolicy":
        """Builds the policy from a plain mapping (for example loaded from a config file). Unknown keys are refused so that a misspelt key cannot silently leave the default in place."""
        cfg = dict(config or {})
        unknown = set(cfg) - {"refit_interval_days"}
        if unknown:
            raise ValueError(f"unknown refit policy keys: {sorted(unknown)}")
        return cls(**cfg)


def should_refit(last_refit_day, today, policy: RefitPolicy = RefitPolicy()) -> bool:
    """True when a refit is due: no earlier refit (`last_refit_day` is None), or at least `policy.refit_interval_days` whole days have passed since it. Pure. Raises ValueError for
    days that are not whole numbers, for a `today` before `last_refit_day`, and for a gap above `MAX_TIMESTAMP_SPAN_DAYS` (the contract's guard against seconds passed as days)."""
    today = _whole_day(today, "today")
    if last_refit_day is None:
        return True
    last = _whole_day(last_refit_day, "last_refit_day")
    if today < last:
        raise ValueError(f"today ({today}) is before the last refit day ({last})")
    if today - last > MAX_TIMESTAMP_SPAN_DAYS:
        raise ValueError(f"{today - last} days since the last refit exceeds {MAX_TIMESTAMP_SPAN_DAYS}: are the days really calendar days?")
    return (today - last) >= policy.refit_interval_days


def next_refit_day(last_refit_day, policy: RefitPolicy = RefitPolicy()) -> int:
    return _whole_day(last_refit_day, "last_refit_day") + policy.refit_interval_days


def refit_schedule(first_day, end_day, policy: RefitPolicy = RefitPolicy()) -> list[int]:
    """The days in [first_day, end_day] on which a refit is due when `should_refit` is asked every day and a refit is made whenever it says yes (the first fit is made on `first_day`)."""
    first, end = _whole_day(first_day, "first_day"), _whole_day(end_day, "end_day")
    out, last = [], None
    for day in range(first, end + 1):
        if should_refit(last, day, policy):
            out.append(day); last = day
    return out
