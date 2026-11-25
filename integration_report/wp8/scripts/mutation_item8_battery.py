"""Mutation check for tests/test_wp8_item8_battery_compare.py. usage: PYTHONPATH=. python3 integration_report/wp8/scripts/mutation_item8_battery.py [out.txt]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mutation_runner import run

BB = "integration_report/wp8/scripts/item8_baseline_battery.py"
MUTANTS = [
    ("B01 the BIC shift tolerance is loose", BB, "<= 1e-8)", "<= 1.0)"),
    ("B02 the expected BIC shift is K, not K ln T", BB, 'abs(shift - K * np.log(max(int(r["T"]), 1)))', "abs(shift - K)"),
    ("B03 the expected BIC shift uses log2", BB, 'K * np.log(max(int(r["T"]), 1))', 'K * np.log2(max(int(r["T"]), 1))'),
    ("B04 regime means are not compared", BB, 'for k in NUMERIC + ("regime_means",))', "for k in NUMERIC)"),
    ("B05 the posterior arrays are compared with allclose", BB, "np.array_equal(r[k], n[k])", "np.allclose(r[k], n[k], atol=1e-6)"),
    ("B06 duration parameters are not compared", BB, 'and r["duration_parameters"] == n["duration_parameters"] \\', "\\"),
    ("B07 the fit state is not compared", BB, 'and r["fit_state"] == n["fit_state"] and r["Dmax"] == n["Dmax"]', 'and r["Dmax"] == n["Dmax"]'),
    ("B08 Dmax is not compared", BB, 'and r["fit_state"] == n["fit_state"] and r["Dmax"] == n["Dmax"]', 'and r["fit_state"] == n["fit_state"]'),
    ("B09 a wrong BIC shift is accepted", BB, "ok = n_fit_id == len(ref) and n_bic_ok == len(ref) and not classes", "ok = n_fit_id == len(ref) and not classes"),
    ("B10 changed fitted outputs are accepted", BB, "ok = n_fit_id == len(ref) and n_bic_ok == len(ref) and not classes", "ok = n_bic_ok == len(ref) and not classes"),
    ("B11 unexplained status changes are accepted", BB, 'ok = n_fit_id == len(ref) and n_bic_ok == len(ref) and not classes["NOT EXPLAINED"]', "ok = n_fit_id == len(ref) and n_bic_ok == len(ref)"),
    ("B13 newly withheld by R1 is called unexplained", BB, 'elif (not new_pub) and r1:\n            c = "previously publishable, newly withheld by R1"', 'elif (not new_pub) and r1:\n            c = "NOT EXPLAINED"'),
    ("B14 newly withheld by R2 is called R1", BB, 'elif (not new_pub) and r2:\n            c = "previously publishable, newly withheld by R2"', 'elif (not new_pub) and r2:\n            c = "previously publishable, newly withheld by R1"'),
    ("B15 withheld by both is called R1 only", BB, 'c = "previously publishable, newly withheld by R1 and R2"', 'c = "previously publishable, newly withheld by R1"'),
    ("B16 a previously withheld fit that is now published is explained", BB, 'c = "already withheld (unchanged status)" if not new_pub else "NOT EXPLAINED"', 'c = "already withheld (unchanged status)"'),
    ("B17 a flagged but still published fit is explained", BB, "elif new_pub and not (r1 or r2):", "elif new_pub:"),
    ("B18 the strict count ignores the BIC", BB, "n_strict += bool(fit_same and bic_same)", "n_strict += bool(fit_same)"),
    ("B19 the flag share ignores R2", BB, 'n_flag = sum(1 for r in new if r["fit_collapsed"] or r["fit_duration_degenerate"])', 'n_flag = sum(1 for r in new if r["fit_collapsed"])'),
    ("B20 different seeds are accepted", BB, 'assert [r["seed"] for r in ref] == [r["seed"] for r in new]\n    classes', "classes"),
]

if __name__ == "__main__":
    sys.exit(run(MUTANTS, ["tests/test_wp8_item8_battery_compare.py"], sys.argv[1] if len(sys.argv) > 1 else None, workers=1))
