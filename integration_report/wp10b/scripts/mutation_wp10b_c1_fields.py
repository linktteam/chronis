"""WP10B C1(1): mutation control of `c1_optimizer_fields.py` against `tests/test_wp10b_c1_fields.py` (named test must fail by its own assertion; unmutated control first, same run; runner `mutation_runner10b.py`).
usage: PYTHONPATH=. python3.12 integration_report/wp10b/scripts/mutation_wp10b_c1_fields.py [out.txt]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutation_runner10b as R

S = "integration_report/wp10b/scripts/c1_optimizer_fields.py"
T = "tests/test_wp10b_c1_fields.py::"
MUT = [
    ("C01 flag fields are not recognised as flags", S, "    if sub in FLAGS:\n        return \"flag\"", "    if False:\n        return \"flag\"", [T + "test_a_flipped_flag_is_reported_as_a_flag"]),
    ("C02 array float values are not recognised as floats", S, 'return (k.endswith(".f") or ".head." in k) and', 'return (k.endswith(".f")) and', [T + "test_a_float_difference_is_a_model_attribute_not_a_flag_and_not_a_discrete_outcome"]),
    ("C03 a difference needs three distinct values", S, "differs=len(distinct) > 1", "differs=len(distinct) > 2", [T + "test_a_float_difference_is_a_model_attribute_not_a_flag_and_not_a_discrete_outcome"]),
    ("C04 array hashes are classified as discrete", S, 'if sub.endswith((".sha", ".shape.0"', 'if sub.endswith((".shaX", ".shape.0"', [T + "test_a_float_difference_is_a_model_attribute_not_a_flag_and_not_a_discrete_outcome"]),
    ("C05 only seed 1 is analysed", S, "for seed in (1, 2):", "for seed in (1,):", [T + "test_the_two_labels_are_analysed_separately"]),
    ("C06 the relative difference is the absolute one", S, "d = abs(xa - xb) / max(abs(xa), abs(xb), 1e-300)", "d = abs(xa - xb)", [T + "test_a_float_difference_is_a_model_attribute_not_a_flag_and_not_a_discrete_outcome"]),
    ("C07 discrete differences are never listed", S, "discrete_subfields_that_differ=[r[\"key\"] for r in discrete_rows if r[\"differs\"]]", "discrete_subfields_that_differ=[]", [T + "test_a_differing_iteration_counter_is_reported_as_a_discrete_outcome"]),
    ("C08 flag values are taken from the first kernel only", S, "for x in kernels}) for f in FLAGS}", "for x in kernels[:1]}) for f in FLAGS}", [T + "test_a_flipped_flag_is_reported_as_a_flag"]),
]
if __name__ == "__main__":
    sys.exit(R.run(MUT, sys.argv[1] if len(sys.argv) > 1 else None, workers=2))
