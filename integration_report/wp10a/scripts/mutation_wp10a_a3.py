"""Mutation check for the A3 analysis (`a3_analyze.py`: the declared metrics (a) to (c), the safety test, the frozen selection rule) against tests/test_wp10a_a3_analyze.py (and the wiring test
tests/test_wp10a_a3_wiring.py when it exists). usage: PYTHONPATH=. python3 integration_report/wp10a/scripts/mutation_wp10a_a3.py [out.txt]"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp8" / "scripts"))
import mutation_runner as R

_base = shutil.ignore_patterns(".git", "__pycache__", "wp7r_evidence", "wp8_evidence", "wp9_evidence", "wp10a_evidence", ".pytest_cache")
R.IGNORE = lambda d, names: set(_base(d, names))

AN = "integration_report/wp10a/scripts/a3_analyze.py"
CONST = "TARGET_NAMES, TARGET_COVERAGE, TARGET_WRONG = 0.98, 0.80, 0.01"
MUT = [
    ("M01 the rule-of-three bound uses n + 1", AN, "return 1.0 - (1.0 - conf) ** (1.0 / n)", "return 1.0 - (1.0 - conf) ** (1.0 / (n + 1))"),
    ("M02 all-wrong is not capped at 1", AN, "if k >= n:\n        return 1.0", "if k > n:\n        return 1.0"),
    ("M03 the confidence is 90%", AN, "def exact_upper(k: int, n: int, conf: float = 0.95)", "def exact_upper(k: int, n: int, conf: float = 0.90)"),
    ("M04 one wrong alignment is tolerated", AN, "return n > 0 and wrong == 0 and exact_upper(0, n) <= bound", "return n > 0 and wrong <= 1 and exact_upper(0, n) <= bound"),
    ("M05 the safety bound check is twice as loose", AN, "and exact_upper(0, n) <= bound", "and exact_upper(0, n) <= 2 * bound"),
    ("M06 the safety bound is 2%", AN, "SAFETY_BOUND = 0.01", "SAFETY_BOUND = 0.02"),
    ("M07 the names level is 95%", AN, CONST, "TARGET_NAMES, TARGET_COVERAGE, TARGET_WRONG = 0.95, 0.80, 0.01"),
    ("M08 the coverage level is 75%", AN, CONST, "TARGET_NAMES, TARGET_COVERAGE, TARGET_WRONG = 0.98, 0.75, 0.01"),
    ("M09 the wrong-and-unflagged level is 2%", AN, CONST, "TARGET_NAMES, TARGET_COVERAGE, TARGET_WRONG = 0.98, 0.80, 0.02"),
    ("M10 the first window is a case", AN, "if end <= FIRST_T:\n            continue", "if end < FIRST_T:\n            continue"),
    ("M11 the names level is 95% of days", AN, "names_level: float = 0.99", "names_level: float = 0.95"),
    ("M12 the day agreement is exclusive", AN, 'S.day_agreement(x[1].stable_regime_id, y[1].stable_regime_id) >= names_level', 'S.day_agreement(x[1].stable_regime_id, y[1].stable_regime_id) > names_level'),
    ("M13 one_missing is inverted", AN, "one_missing=(x is None) != (y is None)", "one_missing=(x is None) == (y is None)"),
    ("M14 (a) counts windows served by one seed", AN, 'both = [c for c in cases if c["a"] and c["b"]]', 'both = [c for c in cases if c["a"] or c["b"]]'),
    ("M15 (a) counts disagreements as agreement", AN, 'return sum(1 for c in both if c["agree"]), len(both)', 'return sum(1 for c in both if not c["agree"]), len(both)'),
    ("M16 (b) counts a window as both-withhold when either withholds", AN, 'bw = sum(1 for c in cases if not c["a"] and not c["b"])', 'bw = sum(1 for c in cases if not c["a"] or not c["b"])'),
    ("M17 (b) counts equal serve flags as one-serves", AN, 'one = sum(1 for c in cases if c["a"] != c["b"])', 'one = sum(1 for c in cases if c["a"] == c["b"])'),
    ("M18 (b) agreement ignores both-withhold", AN, "agree=bs + bw,", "agree=bs,"),
    ("M19 the strict criterion counts only windows both serve", AN, 'st = [c for c in cases if c["a"] or c["b"]]', 'st = [c for c in cases if c["a"] and c["b"]]'),
    ("M20 the strict criterion accepts one seed", AN, 'sum(1 for c in st if c["a"] and c["b"] and c["agree"])', 'sum(1 for c in st if (c["a"] or c["b"]) and c["agree"])'),
    ("M21 (c) counts the first fit", AN, 'if r["end"] > first_t and r["state"] == "ALIGNED":', 'if r["end"] >= first_t and r["state"] == "ALIGNED":'),
    ("M22 (c) takes the first withheld step", AN, 'if r["end"] > first_t and r["state"] == "ALIGNED":', 'if r["end"] > first_t and r["state"] != "ALIGNED":'),
    ("M23 (c) censors at 60", AN, "cen = horizon - first_t + 1", "cen = horizon - first_t"),
    ("M24 (c) served by day 14 is exclusive", AN, "sum(1 for d in served if d <= 14)", "sum(1 for d in served if d < 14)"),
    ("M25 (c) served by day 28 is exclusive", AN, "sum(1 for d in served if d <= 28)", "sum(1 for d in served if d < 28)"),
    ("M26 (c) the median takes the upper middle", AN, "sorted(served)[(len(served) - 1) // 2]", "sorted(served)[len(served) // 2]"),
    ("M27 (c) the quantile rounds down", AN, "int(math.ceil(p * len(cens))) - 1", "int(math.floor(p * len(cens))) - 1"),
    ("M28 the lowest calibration coverage is chosen", AN, 'c = max(zero_cal, key=lambda c: key(c, "cal"))', 'c = min(zero_cal, key=lambda c: key(c, "cal"))'),
    ("M29 ties go to the smaller h", AN, 'return (t["served"] / max(t["refits"], 1), c[1], 1 if c[0] else 0)', 'return (t["served"] / max(t["refits"], 1), -c[1], 1 if c[0] else 0)'),
    ("M30 ties go to flag OFF", AN, 'return (t["served"] / max(t["refits"], 1), c[1], 1 if c[0] else 0)', 'return (t["served"] / max(t["refits"], 1), c[1], 0 if c[0] else 1)'),
    ("M31 the validation level is exclusive", AN, 'v["wrong_unflagged"] / v["alignments"] <= 0.01', 'v["wrong_unflagged"] / v["alignments"] < 0.01'),
    ("M32 zero wrong is judged on the pooled set in step 1", AN, 'zero_cal = [c for c in cands if curve[c]["cal"]["wrong_unflagged"] == 0]', 'zero_cal = [c for c in cands if curve[c]["pooled"]["wrong_unflagged"] == 0]'),
    ("M33 the candidate is validated on the calibration half", AN, 'v = curve[c]["val"]', 'v = curve[c]["cal"]'),
    ("M34 the halves are swapped", AN, 'return "cal" if user % 2 == 0 else "val"', 'return "cal" if user % 2 == 1 else "val"'),
    ("M35 the fallback takes the lowest coverage", AN, '-key(c, "pooled")[0]))', 'key(c, "pooled")[0]))'),
    ("M36 the flag-only list is flag OFF", AN, "[(True, h) for h in H_CANDS] if flag_only", "[(False, h) for h in H_CANDS] if flag_only"),
    ("M37 the gate grid is 2.0, 2.5, 3.0", AN, "H_CANDS = (2.0, 2.25, 2.5)", "H_CANDS = (2.0, 2.5, 3.0)"),
    ("M38 the sharpness floor is 0.9", AN, "SHARP_ON = 0.95", "SHARP_ON = 0.9"),
    ("M39 flag OFF keeps the sharpness floor", AN, "return SHARP_ON if u else 0.0", "return SHARP_ON"),
    ("M40 step 3 takes the lowest pooled coverage", AN, 'return max(zero_all, key=lambda c: key(c, "pooled"))', 'return min(zero_all, key=lambda c: key(c, "pooled"))'),
]
if __name__ == "__main__":
    tests = ["tests/test_wp10a_a3_analyze.py"] + (["tests/test_wp10a_a3_wiring.py"] if Path("tests/test_wp10a_a3_wiring.py").exists() else [])
    sys.exit(R.run(MUT, tests, sys.argv[1] if len(sys.argv) > 1 else None, workers=1))
