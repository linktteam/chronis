"""Mutation check for the A4 analysis and run scripts (`a4_analyze.py`: the declared criteria, the pairing, the selection, the control; `a4_run.py`: the design parameters) against
tests/test_wp10a_a4_analyze.py. usage: PYTHONPATH=. python3 integration_report/wp10a/scripts/mutation_wp10a_a4.py [out.txt]"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp8" / "scripts"))
import mutation_runner as R

_base = shutil.ignore_patterns(".git", "__pycache__", "wp7r_evidence", "wp8_evidence", "wp9_evidence", "wp10a_evidence", ".pytest_cache")
R.IGNORE = lambda d, names: set(_base(d, names))

AN = "integration_report/wp10a/scripts/a4_analyze.py"
RN = "integration_report/wp10a/scripts/a4_run.py"
MUT = [
    ("M01 the gate is 2.0", AN, "H = 2.5\nSHARP = 0.95", "H = 2.0\nSHARP = 0.95"),
    ("M02 no sharpness floor", AN, "SHARP = 0.95\nMIN_ALIGNMENTS", "SHARP = 0.0\nMIN_ALIGNMENTS"),
    ("M03 30 alignments are enough", AN, "MIN_ALIGNMENTS = 40", "MIN_ALIGNMENTS = 30"),
    ("M04 the correct level is 90%", AN, "CORRECT_LEVEL = 0.95", "CORRECT_LEVEL = 0.90"),
    ("M05 the wrong-and-unflagged level is 2%", AN, "WU_LEVEL = 0.01", "WU_LEVEL = 0.02"),
    ("M06 zr >= 2 is enough", AN, "ZR_MIN = 3", "ZR_MIN = 2"),
    ("M07 the order of ties is reversed", AN, 'ORDER = ("D", "E", "F")', 'ORDER = ("F", "E", "D")'),
    ("M08 a zero sd is not guarded", AN, "np.where(summary.scaler_std > 0, summary.scaler_std, 1.0)", "np.where(summary.scaler_std >= 0, summary.scaler_std, 1.0)"),
    ("M09 the control is not restored", AN, "    finally:\n        L._vectors = old", "    finally:\n        pass"),
    ("M10 the control does not centre", AN, "(summary.raw_means - summary.scaler_mean) / np.where", "(summary.raw_means) / np.where"),
    ("M11 R works with 39 alignments", AN, 'm["alignments"] >= MIN_ALIGNMENTS and', 'm["alignments"] > MIN_ALIGNMENTS and'),
    ("M12 R works needs more than 95% correct", AN, 'm["correct_rate"] >= CORRECT_LEVEL and m["wu_rate"] <= WU_LEVEL', 'm["correct_rate"] > CORRECT_LEVEL and m["wu_rate"] <= WU_LEVEL'),
    ("M13 R works needs under 1% wrong", AN, 'm["wu_rate"] <= WU_LEVEL\n', 'm["wu_rate"] < WU_LEVEL\n'),
    ("M14 old criterion: 95% is a failure", AN, '(m["served"] > 0 and m["correct_rate"] < CORRECT_LEVEL)', '(m["served"] > 0 and m["correct_rate"] <= CORRECT_LEVEL)'),
    ("M15 old criterion: 1% is a failure", AN, 'or m["wu_rate"] > WU_LEVEL)', 'or m["wu_rate"] >= WU_LEVEL)'),
    ("M16 zr ignores whether R was right", AN, "zr=sum(wrong(b[k]) and right(a[k]) for k in both)", "zr=sum(wrong(b[k]) for k in both)"),
    ("M17 rz ignores whether Z was right", AN, "rz=sum(wrong(a[k]) and right(b[k]) for k in both)", "rz=sum(wrong(a[k]) for k in both)"),
    ("M18 pairing takes the union", AN, "set(a) & set(b)", "set(a) | set(b)"),
    ("M19 a withheld correct answer counts as right", AN, 'right = lambda r: bool(r["served"]) and bool(r["correct"])', 'right = lambda r: bool(r["correct"])'),
    ("M20 a withheld wrong answer counts as wrong", AN, 'wrong = lambda r: bool(r["served"]) and not r["correct"]', 'wrong = lambda r: not r["correct"]'),
    ("M21 zr = 3 is not enough", AN, 'p["zr"] >= ZR_MIN and', 'p["zr"] > ZR_MIN and'),
    ("M22 zr = rz counts as failure", AN, 'and p["zr"] > p["rz"]', 'and p["zr"] >= p["rz"]'),
    ("M23 zr or the margin is enough", AN, 'p["zr"] >= ZR_MIN and p["zr"] > p["rz"]', 'p["zr"] >= ZR_MIN or p["zr"] > p["rz"]'),
    ("M24 qualifies needs R or the control", AN, "r_works(mr) and control_fails(p)", "r_works(mr) or control_fails(p)"),
    ("M25 best = smallest zr", AN, 'return max(q, key=lambda d: (results[d]["paired"]["zr"], -ORDER.index(d)))', 'return min(q, key=lambda d: (results[d]["paired"]["zr"], -ORDER.index(d)))'),
    ("M26 ties go to the last design", AN, '(results[d]["paired"]["zr"], -ORDER.index(d))', '(results[d]["paired"]["zr"], ORDER.index(d))'),
    ("M27 best ignores qualification", AN, 'if d in results and results[d]["qualifies"]]', 'if d in results]'),
    ("M28 chains_d drops the shared windows", AN, 'if g == "A4D-shared" or g == f"A4D-{variant}":', 'if g == f"A4D-{variant}":'),
    ("M29 chains_d in reverse window order", AN, 'sorted(v, key=lambda r: r["job"]["end"]) for k, v in out.items()', 'sorted(v, key=lambda r: -r["job"]["end"]) for k, v in out.items()'),
    ("M30 chains_e puts the short window first", AN, '= [anchors[r["job"]["user"]], r]', '= [r, anchors[r["job"]["user"]]]'),
    ("M31 chains_f takes the other seed", AN, "a, _ = A.split_chains(recs)", "_, a = A.split_chains(recs)"),
    ("M32 R is replayed without the flag", AN, "n=10, use_flag=True, sharpness=SHARP, lineage_id", "n=10, use_flag=False, sharpness=SHARP, lineage_id"),
    ("M33 the patch is applied to R instead of Z", AN, "ctx = patch_zspace() if zspace else contextlib.nullcontext()", "ctx = patch_zspace() if not zspace else contextlib.nullcontext()"),
    ("M34 the pair key drops the window", AN, 'row["pair_key"] = (key, row["end"])', 'row["pair_key"] = (key,)'),
    ("M35 withheld counts the usable fits", AN, "if R.prepared(r, 10) is None)", "if R.prepared(r, 10) is not None)"),
    ("M36 the wrong-and-unflagged rate is per served", AN, 'wu_rate=(t["wrong_unflagged"] / t["alignments"] if t["alignments"] else float("nan"))', 'wu_rate=(t["wrong_unflagged"] / t["served"] if t["served"] else float("nan"))'),
    ("M37 the evaluation reads the wrong design", AN, 'res["design"] = design', 'res["design"] = "D"'),
    ("M38 the evaluation of D has no comparison", AN, '        res["D_none"] = study("D_none (the same users without the planted change)", chains_d(recs, "none"))', '        pass'),
    ("R01 design D triples nothing", RN, "factor=3.0", "factor=2.0"),
    ("R02 design D changes regime 1", RN, "dict(regime=2, features", "dict(regime=1, features"),
    ("R03 design D starts at day 75", RN, "factor=3.0, start=60", "factor=3.0, start=75"),
    ("R04 design D changes three features", RN, "features=[0, 1, 2, 3]", "features=[0, 1, 2]"),
    ("R05 the development range starts at 10300", RN, '"dev": range(10301, 10317)', '"dev": range(10300, 10316)'),
    ("R06 the evaluation range overlaps development", RN, '"eval": range(10321, 10337)', '"eval": range(10311, 10327)'),
    ("R07 design E lengths are 20 and 40", RN, "E_T, E_L = (105, 120, 135, 150), (20, 30)", "E_T, E_L = (105, 120, 135, 150), (20, 40)"),
    ("R08 design E ends differ", RN, "E_T, E_L = (105, 120, 135, 150), (20, 30)", "E_T, E_L = (100, 120, 135, 150), (20, 30)"),
    ("R09 the shared windows include 75", RN, "            if e <= 60:\n                out.append(_job(u, e - 45, e, \"A4D-shared\"))", "            if e <= 75:\n                out.append(_job(u, e - 45, e, \"A4D-shared\"))"),
    ("R10 the anchor is [0, 100)", RN, '_job(u, 0, 90, "A4E-anchor")', '_job(u, 0, 100, "A4E-anchor")'),
    ("R11 the fit seed is user + 1", RN, "fitseed=user, K=3", "fitseed=user + 1, K=3"),
    ("R12 the evaluation does not need the freeze file", RN, 'if not fz.exists():', 'if False:'),
    ("R13 design F is fitted", RN, 'if chosen == "F":', 'if chosen == "Q":'),
]
if __name__ == "__main__":
    sys.exit(R.run(MUT, ["tests/test_wp10a_a4_analyze.py"], sys.argv[1] if len(sys.argv) > 1 else None, workers=1))
