"""WP10B C1(3)(b): mutation control of the cross-machine comparison (`c1_cross_machine.py`) against `tests/test_wp10b_c1_cross_machine.py` (named test must fail by its own assertion; control first; runner `mutation_runner10b.py`).
usage: PYTHONPATH=. python3.12 integration_report/wp10b/scripts/mutation_wp10b_c1_cross.py [out.txt]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutation_runner10b as R

F = "integration_report/wp10b/scripts/c1_cross_machine.py"
T = "tests/test_wp10b_c1_cross_machine.py::"
MUT = [
    ("X1 labels are compared without renaming", F, "    for perm in itertools.permutations(range(K)):", "    for perm in [tuple(range(K))]:", [T + "test_label_agreement_is_invariant_to_renaming_and_reports_exact_equality"]),
    ("X2 exact equality always reported", F, "return best, bool(np.array_equal(a, b))", "return best, True", [T + "test_label_agreement_is_invariant_to_renaming_and_reports_exact_equality"]),
    ("X3 a different NaN pattern is not infinite", F, "    if x.shape != y.shape or not np.array_equal(np.isnan(x), np.isnan(y)):", "    if x.shape != y.shape:", [T + "test_max_abs_diff_rules_for_nan_and_shape"]),
    ("X4 the mean difference replaces the largest", F, "return float(np.abs(x[ok] - y[ok]).max()) if ok.any() else 0.0", "return float(np.abs(x[ok] - y[ok]).mean()) if ok.any() else 0.0", [T + "test_max_abs_diff_rules_for_nan_and_shape"]),
    ("X5 a different regime set counts as no difference", F, '    if set(ref) != set(new):\n        return float("inf")\n', '    if set(ref) != set(new):\n        return 0.0\n', [T + "test_duration_diff_requires_the_same_regimes_and_returns_the_largest_difference"]),
    ("X6 the stability flag is not a discrete output", F, '("fit_state", "K", "Dmax", "withheld", "collapsed", "unstable")', '("fit_state", "K", "Dmax", "withheld", "collapsed")', [T + "test_every_discrete_field_difference_is_reported"]),
    ("X7 the withheld reason is not a discrete output", F, '("fit_state", "K", "Dmax", "withheld", "collapsed", "unstable")', '("fit_state", "K", "Dmax", "collapsed", "unstable")', [T + "test_every_discrete_field_difference_is_reported"]),
    ("X8 duration parameters are not part of bit identity", F, ' and r["duration_parameters"] == n["duration_parameters"]\n    row["max_abs"]', '\n    row["max_abs"]', [T + "test_a_duration_parameter_difference_alone_breaks_bit_identity"]),
    ("X9 the BIC difference is signed", F, 'row["bic_abs"] = abs(float(r["BIC"]) - float(n["BIC"]))', 'row["bic_abs"] = float(r["BIC"]) - float(n["BIC"])', [T + "test_a_rounding_difference_is_not_bit_identical_but_keeps_every_discrete_output"]),
    ("X10 every lineage step that ran counts as aligned", F, 'aligned=sum(l.get("next_state") == "ALIGNED" for l in lin)', 'aligned=sum(bool(l.get("ok")) for l in lin)', [T + "test_a_flagged_fit_is_never_aligned"]),
    ("X11 the lowest agreement is replaced by the highest", F, "min_label_agree=min(r[\"label_agree\"] for r in rows)", "min_label_agree=max(r[\"label_agree\"] for r in rows)", [T + "test_the_summary_takes_the_worst_user_not_the_best"]),
    ("X12 the may-claim count ignores the claim", F, 'may_claim=sum(bool(l.get("may_claim")) for l in lin)', 'may_claim=sum(bool(l.get("ok")) for l in lin)', [T + "test_a_different_label_sequence_is_reported_and_the_lineage_may_not_claim_continuity"]),
    ("X13 the self-alignment is the other configuration's", F, 'self_aligned=sum(l.get("self_state") == "ALIGNED" for l in lin)', 'self_aligned=sum(l.get("next_state") == "ALIGNED" for l in lin)', [T + "test_the_same_fit_fed_twice_separates_the_gate_from_the_machine"]),
    ("X14 booleans count as numeric differences", F, "and not isinstance(r[\"restart_summary\"].get(k), bool)}", "}", [T + "test_numeric_restart_summary_differences_are_measured_and_booleans_are_not"]),
]
if __name__ == "__main__":
    sys.exit(R.run(MUT, sys.argv[1] if len(sys.argv) > 1 else None, workers=1))
