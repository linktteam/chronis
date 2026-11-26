"""Mutation check for tests/test_wp9_b5_monitoring.py (WP9 B5). usage: PYTHONPATH=. python3 integration_report/wp9/scripts/mutation_b5.py [out.txt]"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp8" / "scripts"))
import mutation_runner as R

_base = shutil.ignore_patterns(".git", "__pycache__", "wp7r_evidence", "wp8_evidence", "wp9_evidence", ".pytest_cache")
R.IGNORE = lambda d, names: set(_base(d, names))

M, CFG = "backbone/hssm/monitoring.py", "backbone/hssm/config.py"
MUTANTS = [
    ("M01 the recent window is 13 days", CFG, "MONITOR_WINDOW_DAYS = 14", "MONITOR_WINDOW_DAYS = 13"),
    ("M02 the z level is 3.0", CFG, "MONITOR_DROP_Z = 3.5", "MONITOR_DROP_Z = 3.0"),
    ("M03 the size level is 0.5 sd", CFG, "MONITOR_MIN_DROP_SD = 0.75", "MONITOR_MIN_DROP_SD = 0.5"),
    ("M04 the cooldown is 1 day", CFG, "MONITOR_COOLDOWN_DAYS = 14", "MONITOR_COOLDOWN_DAYS = 1"),
    ("M05 the reference is 27 days", CFG, "MONITOR_REFERENCE_DAYS = 28", "MONITOR_REFERENCE_DAYS = 27"),
    ("M06 a rise counts as a drop", M, "drop = float(ref.mean() - recent.mean())", "drop = abs(float(ref.mean() - recent.mean()))"),
    ("M07 either condition is enough", M, "if drop >= min_drop_sd * s0 and z >= drop_z:", "if drop >= min_drop_sd * s0 or z >= drop_z:"),
    ("M08 the z level is strict", M, "if drop >= min_drop_sd * s0 and z >= drop_z:", "if drop >= min_drop_sd * s0 and z > drop_z:"),
    ("M09 no autocorrelation inflation", M, "r = min(max(_lag1(ref), 0.0), 0.8)", "r = 0.0"),
    ("M10 the inflation is inverted", M, "g = (1.0 + r) / (1.0 - r)", "g = (1.0 - r) / (1.0 + r)"),
    ("M11 the reference mean has no error", M, "se = s0 * math.sqrt(g / window + g / reference_days)", "se = s0 * math.sqrt(g / window)"),
    ("M12 non-finite days are kept", M, "keep = np.isfinite(y)", "keep = np.ones(len(y), dtype=bool)"),
    ("M13 the event day is the compressed index", M, "day = int(idx[i - 1])", "day = i - 1"),
    ("M14 the cooldown is ignored", M, "if last_event_day is not None and day - last_event_day < cooldown_days:", "if False:"),
    ("M15 a constant reference is evaluated", M, "if not (math.isfinite(s0) and s0 > 0.0):", "if not math.isfinite(s0):"),
    ("M16 the gate opens for any truthy n17 flag", M, "if n17_passed is not True:", "if not n17_passed:"),
    ("M17 the gate does not need calibration evidence", M, "    if not (isinstance(calibration_evidence, str) and calibration_evidence.strip()):", "    if False:"),
    ("M18 the threshold is inclusive of less", M, "    elif float(c) < threshold:", "    elif float(c) < threshold - 0.05:"),
    ("M19 the percentage is rounded", M, "percentage=int(math.floor(float(c) * 100.0 + 1e-9))", "percentage=int(round(float(c) * 100.0))"),
    ("M20 the owner floor is not enforced", M, "math.isfinite(float(threshold)) and threshold >= CONFIDENCE_DISPLAY_THRESHOLD)", "math.isfinite(float(threshold)) and threshold >= 0.0)"),
    ("M21 N17 is declared passed", CFG, "N17_PASSED = False", "N17_PASSED = True"),
    ("M22 the confidence threshold is 0.8", CFG, "CONFIDENCE_DISPLAY_THRESHOLD = 0.85", "CONFIDENCE_DISPLAY_THRESHOLD = 0.8"),
    ("M23 a boolean confidence is a number", M, "if isinstance(c, bool) or not isinstance(c, (int, float, np.integer, np.floating))", "if not isinstance(c, (int, float, np.integer, np.floating))"),
    ("M24 the first increment ignores the initial distribution", M, "alpha[0] = (log_pi[:, None] + dur) + emis[0][:, None]", "alpha[0] = dur + emis[0][:, None]"),
    ("M25 increments are not differenced", M, "return np.diff(np.concatenate([[0.0], lse]))", "return lse"),
    ("M26 the alert carries the raw id", "backbone/hssm/alerts.py", "user_id_hash=user_id_hash(user_id), **fields)", "user_id_hash=str(user_id), **fields)"),
]

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(R.run(MUTANTS, ["tests/test_wp9_b5_monitoring.py", "-x", "-q"], out, workers=1))
