"""Mutation check for the A1/A2 scoring (`a12_analyze.py`: the frozen selection rule, the tally, the identical-lineage and fit-level counts) against tests/test_wp9_a12_analyze.py and the scoring tests
of tests/test_wp9_replay_lib.py. usage: PYTHONPATH=. python3 integration_report/wp9/scripts/mutation_a12_scoring.py [out.txt]"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp8" / "scripts"))
import mutation_runner as R

_base = shutil.ignore_patterns(".git", "__pycache__", "wp7r_evidence", "wp8_evidence", "wp9_evidence", ".pytest_cache")
R.IGNORE = lambda d, names: set(_base(d, names))

AN = "integration_report/wp9/scripts/a12_analyze.py"
MUT = [
    ("S01 the validation level is exclusive", AN, 'v["wrong_unflagged"] / v["alignments"] <= 0.01', 'v["wrong_unflagged"] / v["alignments"] < 0.01'),
    ("S02 the lowest calibration coverage is chosen", AN, 'c = max(zero_cal, key=lambda c: key(c, "cal"))', 'c = min(zero_cal, key=lambda c: key(c, "cal"))'),
    ("S03 ties go to the smaller h", AN, 'curve[(True,) + c][half]["refits"], 1), c[0], c[1])', 'curve[(True,) + c][half]["refits"], 1), -c[0], c[1])'),
    ("S04 ties go to the smaller s", AN, 'curve[(True,) + c][half]["refits"], 1), c[0], c[1])', 'curve[(True,) + c][half]["refits"], 1), c[0], -c[1])'),
    ("S05 zero wrong is judged on the pooled set in step 1", AN, 'zero_cal = [c for c in cands if curve[(True,) + c]["cal"]["wrong_unflagged"] == 0]', 'zero_cal = [c for c in cands if curve[(True,) + c]["pooled"]["wrong_unflagged"] == 0]'),
    ("S06 the candidate is validated on the calibration half", AN, 'v = curve[(True,) + c]["val"]', 'v = curve[(True,) + c]["cal"]'),
    ("S07 a validation half without alignments is accepted", AN, 'if v["alignments"] and v["wrong_unflagged"]', 'if v["wrong_unflagged"]'),
    ("S08 step 3 maximises the calibration coverage", AN, 'chosen = max(zero_all, key=lambda c: key(c, "pooled"))', 'chosen = max(zero_all, key=lambda c: key(c, "cal"))'),
    ("S09 the fallback takes the lowest coverage", AN, '-key(c, "pooled")[0]))', 'key(c, "pooled")[0]))'),
    ("S10 the fallback takes the most wrong", AN, 'chosen = min(cands, key=lambda c: (curve[(True,) + c]["pooled"]["wrong_unflagged"],', 'chosen = min(cands, key=lambda c: (-curve[(True,) + c]["pooled"]["wrong_unflagged"],'),
    ("S11 the halves are swapped", AN, 'return "cal" if user % 2 == 0 else "val"', 'return "cal" if user % 2 == 1 else "val"'),
    ("S12 refits are all steps", AN, 't["refits"] = t["steps"] - chains', 't["refits"] = t["steps"]'),
    ("S13 empty chains count", AN, 'chains = sum(1 for (usr, _), (rs, _) in res.items() if (users is None or usr in users) and rs)', 'chains = sum(1 for (usr, _), (rs, _) in res.items() if (users is None or usr in users))'),
    ("S14 the user selection is ignored", AN, 'rows = [r for (usr, _), (rs, _) in res.items() if users is None or usr in users for r in rs]', 'rows = [r for (usr, _), (rs, _) in res.items() if True for r in rs]'),
    ("S15 the identical-lineage level is 90%", AN, "TARGET_IDENTICAL_LINEAGE, TARGET_COVERAGE, TARGET_WRONG_UNFLAGGED = 0.95, 0.80, 0.01", "TARGET_IDENTICAL_LINEAGE, TARGET_COVERAGE, TARGET_WRONG_UNFLAGGED = 0.90, 0.80, 0.01"),
    ("S16 the coverage level is 75%", AN, "TARGET_IDENTICAL_LINEAGE, TARGET_COVERAGE, TARGET_WRONG_UNFLAGGED = 0.95, 0.80, 0.01", "TARGET_IDENTICAL_LINEAGE, TARGET_COVERAGE, TARGET_WRONG_UNFLAGGED = 0.95, 0.75, 0.01"),
    ("S17 the wrong-and-unflagged level is 2%", AN, "TARGET_IDENTICAL_LINEAGE, TARGET_COVERAGE, TARGET_WRONG_UNFLAGGED = 0.95, 0.80, 0.01", "TARGET_IDENTICAL_LINEAGE, TARGET_COVERAGE, TARGET_WRONG_UNFLAGGED = 0.95, 0.80, 0.02"),
    ("S18 the coverage is over the alignments", AN, 'cov = 100 * t["served"] / t["refits"] if t["refits"] else float("nan")', 'cov = 100 * t["served"] / t["alignments"] if t["alignments"] else float("nan")'),
    ("S19 the fit-level agreement level is 0.5", AN, 'F.partition_ari(sa["labels"], sb["labels"]) >= 0.99)', 'F.partition_ari(sa["labels"], sb["labels"]) >= 0.5)'),
    ("S20 a window off the 14-day schedule is compared", AN, "if e not in B_T or e not in db:", "if e not in db:"),
    ("S21 a pair counts as flagged only when both are", AN, "flagged_either += bool(fa or fb)", "flagged_either += bool(fa and fb)"),
    ("S22 the chains are split the wrong way", AN, 'r["job"]["fitseed"] == r["job"]["user"]])', 'r["job"]["fitseed"] != r["job"]["user"]])'),
    ("S23 a case is counted only when both seeds serve", AN, "            if sa or sb:\n                strict[1] += 1", "            if sa and sb:\n                strict[1] += 1"),
    ("S24 the strict identity needs one seed only", AN, "strict[0] += bool(sa and sb and agree)", "strict[0] += bool((sa or sb) and agree)"),
    ("S25 a flagged anchor is not counted as a violation", AN, "viol += bool(lr.fit_unstable or lr.anchor_fit_unstable)", "viol += bool(lr.fit_unstable)"),
    ("S26 the day agreement level is exclusive", AN, "S.day_agreement(a[1].stable_regime_id, b[1].stable_regime_id) >= 0.99", "S.day_agreement(a[1].stable_regime_id, b[1].stable_regime_id) > 0.99"),
]
KEEP = "not accepted_and_labels_agree_with_fit_hssm and not cached_name_fit"

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(R.run(MUT, ["tests/test_wp9_a12_analyze.py", "tests/test_wp9_replay_lib.py", "-k", KEEP], out, workers=1))
