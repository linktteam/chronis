"""Mutation check for tests/test_wp8_item6_refit_policy.py (WP8 item 6). usage: PYTHONPATH=. python3 integration_report/wp8/scripts/mutation_item6.py [out.txt]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mutation_runner import run

RP, DOC = "backbone/hssm/refit_policy.py", "docs/INTEGRATION_CONTRACT_HSSM.md"
MUTANTS = [
    ("R01 default interval 8", RP, "DEFAULT_REFIT_INTERVAL_DAYS = 7", "DEFAULT_REFIT_INTERVAL_DAYS = 8"),
    ("R02 lower bound 0", RP, "MIN_REFIT_INTERVAL_DAYS = 1", "MIN_REFIT_INTERVAL_DAYS = 0"),
    ("R03 upper bound 29", RP, "MAX_REFIT_INTERVAL_DAYS = 28", "MAX_REFIT_INTERVAL_DAYS = 29"),
    ("R04 upper bound 27", RP, "MAX_REFIT_INTERVAL_DAYS = 28", "MAX_REFIT_INTERVAL_DAYS = 27"),
    ("R05 bounds not checked", RP, "if not (MIN_REFIT_INTERVAL_DAYS <= d <= MAX_REFIT_INTERVAL_DAYS):", "if False:"),
    ("R06 a refit is due only after more than the interval", RP, "return (today - last) >= policy.refit_interval_days", "return (today - last) > policy.refit_interval_days"),
    ("R07 a refit is due one day early", RP, "return (today - last) >= policy.refit_interval_days", "return (today - last) >= policy.refit_interval_days - 1"),
    ("R08 no earlier refit means not due", RP, "    if last_refit_day is None:\n        return True", "    if last_refit_day is None:\n        return False"),
    ("R09 time going backwards accepted", RP, "    if today < last:\n        raise", "    if False:\n        raise"),
    ("R10 the seconds-as-days guard removed", RP, "    if today - last > MAX_TIMESTAMP_SPAN_DAYS:", "    if False:"),
    ("R11 the seconds-as-days guard is off by one", RP, "if today - last > MAX_TIMESTAMP_SPAN_DAYS:", "if today - last >= MAX_TIMESTAMP_SPAN_DAYS:"),
    ("R12 booleans accepted as days", RP, "if isinstance(x, bool) or not hasattr(x, \"__index__\"):", "if not hasattr(x, \"__index__\"):"),
    ("R13 fractional days accepted", RP, "if isinstance(x, bool) or not hasattr(x, \"__index__\"):", "if isinstance(x, bool):"),
    ("R14 unknown config keys accepted", RP, "        if unknown:\n", "        if False:\n"),
    ("R15 config ignored", RP, "        return cls(**cfg)", "        return cls()"),
    ("R16 schedule starts one day late", RP, "for day in range(first, end + 1):", "for day in range(first + 1, end + 1):"),
    ("R17 the last day is excluded from the schedule", RP, "for day in range(first, end + 1):", "for day in range(first, end):"),
    ("R18 next_refit_day off by one", RP, "return _whole_day(last_refit_day, \"last_refit_day\") + policy.refit_interval_days", "return _whole_day(last_refit_day, \"last_refit_day\") + policy.refit_interval_days + 1"),
    ("R19 a clock enters should_refit", RP, "    today = _whole_day(today, \"today\")\n    if last_refit_day is None:", "    import time\n    today = _whole_day(int(time.time() // 86400) if today is None else today, \"today\")\n    if last_refit_day is None:"),
    ("R20 the policy is mutable", RP, "@dataclass(frozen=True)\nclass RefitPolicy:", "@dataclass\nclass RefitPolicy:"),
    ("R21 contract loses the bounds", DOC, "allowed range **1 to 28** whole calendar days", "allowed range whole calendar days"),
]

if __name__ == "__main__":
    sys.exit(run(MUTANTS, ["tests/test_wp8_item6_refit_policy.py"], sys.argv[1] if len(sys.argv) > 1 else None, workers=1))
