"""Mutation check for tests/test_wp9_b6_recertify.py (WP9 B6). usage: PYTHONPATH=. python3 integration_report/wp9/scripts/mutation_b6.py [out.txt]
The live-fit test is excluded (the mechanics are tested with a stand-in for fit_hssm)."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp8" / "scripts"))
import mutation_runner as R

_base = shutil.ignore_patterns(".git", "__pycache__", "wp7r_evidence", "wp8_evidence", "wp9_evidence", ".pytest_cache")
R.IGNORE = lambda d, names: set(_base(d, names))

RC = "backbone/hssm/recertify.py"
MUTANTS = [
    ("R01 any number of restarts", RC, "if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < MIN_RESTARTS:", "if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:"),
    ("R02 the minimum is 5 restarts", RC, "MIN_RESTARTS = HSSMFitConfig().n_init", "MIN_RESTARTS = 5"),
    ("R03 a fast-test fit is allowed", RC, '"max_duration_retry", "certify_prior_anchored"})', '"max_duration_retry", "certify_prior_anchored", "allow_fast_test_fit"})'),
    ("R04 a cold-start config is allowed", RC, '"max_duration_retry", "certify_prior_anchored"})', '"max_duration_retry", "certify_prior_anchored", "config"})'),
    ("R05 unknown settings are passed on", RC, "        if k not in ALLOWED_SETTINGS:", "        if False:"),
    ("R06 a published fit is refitted", RC, 'if meta.get("earlier_state") == STATE_PUBLISHABLE:', "if False:"),
    ("R07 the publish decision is inverted", RC, "withheld = bool(result.insights_withheld)", "withheld = not bool(result.insights_withheld)"),
    ("R08 a withheld result is returned", RC, "            return _record(STATE_STILL_WITHHELD, reasons, n_rows=int(len(full)))", "            return dict(_record(STATE_STILL_WITHHELD, reasons, n_rows=int(len(full))), result=result)"),
    ("R09 the record says the rules were relaxed", RC, "rules_relaxed=False, schema_version=SCHEMA_VERSION", "rules_relaxed=True, schema_version=SCHEMA_VERSION"),
    ("R10 no new days are fitted anyway", RC, "        if N.size == 0:", "        if False:"),
    ("R11 the schedule is ignored", RC, "if today is not None and not recertify_due(meta.get(\"last_check_day\"), today, policy):", "if False:"),
    ("R12 always due", RC, "    return bool(should_refit(last_check_day, today, policy))", "    return True"),
    ("R13 a failed fit is publishable", RC, 'return _record(STATE_ERROR, [f"fit_failed:{type(exc).__name__}"]', 'return _record(STATE_PUBLISHABLE, [f"fit_failed:{type(exc).__name__}"]'),
    ("R14 a cold-start refusal is publishable", RC, 'return _record(STATE_NOT_ELIGIBLE, ["cold_start_gate"]', 'return _record(STATE_PUBLISHABLE, ["cold_start_gate"]'),
    ("R15 the timestamps are joined in the wrong order", RC, "np.concatenate([np.asarray(timestamps, dtype=float), np.asarray(new_timestamps, dtype=float)])", "np.concatenate([np.asarray(new_timestamps, dtype=float), np.asarray(timestamps, dtype=float)])"),
    ("R16 the new rows come first", RC, "full = np.vstack([X, N])", "full = np.vstack([N, X])"),
    ("R17 the sweep fits items that are not due", RC, "        if not due:", "        if False:"),
    ("R18 the sweep does not advance the check day", RC, "last_check_day=int(today)", 'last_check_day=meta.get("last_check_day")'),
    ("R19 the sweep does not record the new state", RC, 'earlier_state=rec["state"] if rec["state"] in (STATE_PUBLISHABLE, STATE_STILL_WITHHELD) else meta.get("earlier_state"))', 'earlier_state=meta.get("earlier_state"))'),
    ("R20 the reasons are dropped", RC, 'reasons = [str(r) for r in result._withheld_reasons()] if hasattr(result, "_withheld_reasons") else ([] if not withheld else ["withheld"])', "reasons = []"),
    ("R21 an unexpected error is publishable", RC, 'return _record(STATE_ERROR, [f"recertify_error:{type(exc).__name__}"]', 'return _record(STATE_PUBLISHABLE, [f"recertify_error:{type(exc).__name__}"]'),
    ("R22 a boolean count of restarts is accepted", RC, "if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < MIN_RESTARTS:", "if not isinstance(n, (int, np.integer)) or n < MIN_RESTARTS:"),
    ("R23 a shape mismatch is fitted", RC, "if X.ndim != 2 or N.ndim != 2 or X.shape[1] != N.shape[1]:", "if False:"),
    ("R24 timestamps for part of the rows are accepted", RC, "if timestamps is None or new_timestamps is None:", "if False:"),
    ("R25 the schedule error stops the sweep", RC, '        except Exception as exc:                                              # noqa: BLE001\n            out.append(dict(_record(STATE_ERROR, [f"schedule_error', '        except KeyError as exc:                                              # noqa: BLE001\n            out.append(dict(_record(STATE_ERROR, [f"schedule_error'),
]

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(R.run(MUTANTS, ["tests/test_wp9_b6_recertify.py", "-x", "-q", "-k", "not production_fit_hssm"], out, workers=1))
