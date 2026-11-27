"""WP10B A0: mutation control of the fixed last check of the bit-identity compare script (`integration_report/wp9/scripts/a1_battery.py::compare_records`) against `tests/test_wp10b_battery_compare.py`.
Caught only if the NAMED test fails by its own assertion (runner: `mutation_runner10b.py`); the unmutated control runs first in the same invocation.
usage: PYTHONPATH=. python3.12 integration_report/wp10b/scripts/mutation_wp10b_a0.py [out.txt]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutation_runner10b as R

S = "integration_report/wp9/scripts/a1_battery.py"
T = "tests/test_wp10b_battery_compare.py::"
MUT = [
    ("B01 any added key is accepted", S, "added == NEW_KEYS or added <= allowed_new", "added == NEW_KEYS or True", [T + "test_an_unlisted_new_key_is_refused_and_a_listed_one_is_accepted"]),
    ("B02 a subset of the six WP9 keys is accepted", S, "added == NEW_KEYS or added <= allowed_new", "added <= NEW_KEYS or added <= allowed_new", [T + "test_a_partial_set_of_the_six_keys_is_still_refused_when_not_allowed"]),
    ("B03 no added key at all is refused (the old WP10A failure)", S, "old_keys_same and (added == NEW_KEYS or added <= allowed_new)", "old_keys_same and bool(added) and (added == NEW_KEYS or added <= allowed_new)", [T + "test_identical_trees_with_the_six_wp9_keys_in_both_pass"]),
    ("B04 the allowed keys are never accepted", S, "added <= allowed_new)", "added <= set())", [T + "test_an_unlisted_new_key_is_refused_and_a_listed_one_is_accepted"]),
    ("B05 the added-keys condition is dropped from ok", S, " and n_added_only == N", "", [T + "test_an_unlisted_new_key_is_refused_and_a_listed_one_is_accepted"]),
    ("B06 fitted arrays compared with a tolerance", S, "np.array_equal(r[k], n[k]) for k in NUMERIC", "np.allclose(r[k], n[k], rtol=1e-9, atol=0) for k in NUMERIC", [T + "test_a_one_bit_change_in_a_fitted_array_fails"]),
    ("B07 BIC compared with a tolerance", S, 'bic_same = r["BIC"] == n["BIC"]', 'bic_same = abs(r["BIC"] - n["BIC"]) < 1e-9', [T + "test_any_changed_output_fails"]),
    ("B08 old restart_summary values are not compared", S, '(r["restart_summary"][k] == n["restart_summary"][k])', "True", [T + "test_a_changed_old_restart_summary_value_fails"]),
    ("B09 the status check is dropped from ok", S, "n_bic == N and n_status == N and ", "n_bic == N and ", [T + "test_any_changed_output_fails"]),
    ("B10 K is not compared", S, 'and r["Dmax"] == n["Dmax"] and r["K"] == n["K"]', 'and r["Dmax"] == n["Dmax"]', [T + "test_any_changed_output_fails"]),
]
if __name__ == "__main__":
    sys.exit(R.run(MUT, sys.argv[1] if len(sys.argv) > 1 else None, workers=2))
