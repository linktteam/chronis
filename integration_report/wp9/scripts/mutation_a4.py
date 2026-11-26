"""Mutation check for tests/test_wp9_a4_analyze.py and tests/test_wp9_a4_generator.py (WP9 A4: the analysis script of the negative control and the generator option `feature_scale_change`).
usage: PYTHONPATH=. python3 integration_report/wp9/scripts/mutation_a4.py [out.txt]
A mutant edits ONE thing in a throw-away copy of the repository; the tests must FAIL (KILLED). A pattern that does not occur exactly once is NOT APPLIED and counts as not killed."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp8" / "scripts"))
import mutation_runner as R

_base = shutil.ignore_patterns(".git", "__pycache__", "wp7r_evidence", "wp8_evidence", "wp9_evidence", ".pytest_cache")
R.IGNORE = lambda d, names: set(_base(d, names))

AN, GEN = "integration_report/wp9/scripts/a4_analyze.py", "data/synthetic_identity.py"
MUTANTS = [
    ("D01 the minimum number of alignments is 30", AN, "MIN_ALIGNMENTS = 40", "MIN_ALIGNMENTS = 30"),
    ("D02 the minimum number of alignments is exclusive", AN, "return m[\"alignments\"] >= MIN_ALIGNMENTS and", "return m[\"alignments\"] > MIN_ALIGNMENTS and"),
    ("D03 the correct-among-served level of the raw method is exclusive", AN, "m[\"correct_rate\"] >= 0.95 and m[\"wu_rate\"] <= 0.01", "m[\"correct_rate\"] > 0.95 and m[\"wu_rate\"] <= 0.01"),
    ("D04 the wrong-and-unflagged level of the raw method is exclusive", AN, "m[\"correct_rate\"] >= 0.95 and m[\"wu_rate\"] <= 0.01", "m[\"correct_rate\"] >= 0.95 and m[\"wu_rate\"] < 0.01"),
    ("D05 the failure level of the control on correctness is inclusive", AN, "(m[\"served\"] > 0 and m[\"correct_rate\"] < 0.95)", "(m[\"served\"] > 0 and m[\"correct_rate\"] <= 0.95)"),
    ("D06 the failure level of the control on wrong-and-unflagged is inclusive", AN, "or m[\"wu_rate\"] > 0.01)", "or m[\"wu_rate\"] >= 0.01)"),
    ("D07 the control does not use the stored mean of the scaler", AN, "(summary.raw_means - summary.scaler_mean) / np.where(summary.scaler_std > 0, summary.scaler_std, 1.0)", "summary.raw_means / np.where(summary.scaler_std > 0, summary.scaler_std, 1.0)"),
    ("D08 the control keeps the stored scaler", AN, "(summary.raw_means - summary.scaler_mean) / np.where(summary.scaler_std > 0, summary.scaler_std, 1.0)", "summary.raw_means / np.where(summary.scaler_std > 0, 1.0, 1.0)"),
    ("D09 the shared windows are left out of a design's chain", AN, "if g == \"A4-shared\" or g == f\"A4-{design}\":", "if g == f\"A4-{design}\":"),
    ("D10 the paired counts are swapped", AN, "z_wrong_r_right=sum(wrong(b[k]) and right(a[k]) for k in both)", "z_wrong_r_right=sum(wrong(a[k]) and right(b[k]) for k in both)"),
    ("D11 a design qualifies when either condition holds", AN, "qualifies=bool(raw_works(mr) and z_fails(mz))", "qualifies=bool(raw_works(mr) or z_fails(mz))"),
    ("D12 the chosen design is the last qualifying one", AN, "chosen = next((d for d in (\"A\", \"B\", \"C\") if d in res and res[d][\"qualifies\"]), None)", "chosen = next((d for d in (\"C\", \"B\", \"A\") if d in res and res[d][\"qualifies\"]), None)"),
    ("D13 coverage is served over served", AN, "coverage=(t[\"served\"] / t[\"alignments\"] if t[\"alignments\"] else float(\"nan\"))", "coverage=(t[\"served\"] / t[\"served\"] if t[\"alignments\"] else float(\"nan\"))"),
    ("D14 the scale change starts one day late", GEN, "gain[np.arange(n_days) >= int(feature_scale_change[\"start\"]), int(f)]", "gain[np.arange(n_days) > int(feature_scale_change[\"start\"]), int(f)]"),
    ("D15 the scale change has no effect", GEN, "gain[np.arange(n_days) >= int(feature_scale_change[\"start\"]), int(f)] = float(feature_scale_change[\"factor\"])", "gain[np.arange(n_days) >= int(feature_scale_change[\"start\"]), int(f)] = 1.0"),
    ("D16 the option is ignored", GEN, "    if feature_scale_change:\n        gain = np.ones((n_days, F))", "    if False:\n        gain = np.ones((n_days, F))"),
    ("D17 the scale change moves the regime means instead of the noise", GEN, "        e = e * gain\n", "        means_t = means_t * gain\n"),
    ("D18 only the first listed feature is changed", GEN, "for f in feature_scale_change[\"features\"]:", "for f in feature_scale_change[\"features\"][:1]:"),
]

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(R.run(MUTANTS, ["tests/test_wp9_a4_analyze.py", "tests/test_wp9_a4_generator.py", "-x", "-q"], out, workers=1))
