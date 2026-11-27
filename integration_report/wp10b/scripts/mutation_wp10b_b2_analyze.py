"""WP10B B2: mutation control of the B2 analysis rules (`b2_analyze.py`) and of the class-flag variants of `dual_lib.py` (named test must fail by its own assertion; control first; runner `mutation_runner10b.py`).
usage: PYTHONPATH=. python3.12 integration_report/wp10b/scripts/mutation_wp10b_b2_analyze.py [out.txt]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutation_runner10b as R

A = "integration_report/wp10b/scripts/b2_analyze.py"
L = "integration_report/wp10b/scripts/dual_lib.py"
T = "tests/test_wp10b_b2_analyze.py::"
TL = "tests/test_wp10b_dual_lib.py::"
MUT = [
    ("Y1 the sign test excludes the observed count", A, "for k in range(g, n + 1)", "for k in range(g + 1, n + 1)", [T + "test_exact_sign_test"]),
    ("Y2 an empty sign test gives p = 0", A, "    if n <= 0:\n        return 1.0", "    if n <= 0:\n        return 0.0", [T + "test_exact_sign_test"]),
    ("Y3 the standard error uses ddof 0", A, "float(d.std(ddof=1) / math.sqrt(n))", "float(d.std(ddof=0) / math.sqrt(n))", [T + "test_paired_stats"]),
    ("Y4 the lower bound uses 1.96", A, "Z95 = 1.645", "Z95 = 1.96", [T + "test_paired_stats"]),
    ("Y5 a gain with harm is called HELPED", A, '    if gain_demonstrated:\n        return "HELPED BUT HARMFUL"', '    if gain_demonstrated:\n        return "HELPED"', [T + "test_classification_words"]),
    ("Y6 the sign level is 0.5", A, "SIGN_ALPHA = 0.05", "SIGN_ALPHA = 0.5", [T + "test_gain_needs_g_minus_l_at_least_one_and_p_below_005"]),
    ("Y7 losses are not counted", A, "    l = sum(1 for u in pub_ref if pub_ref[u] and not pub_new[u])", "    l = 0", [T + "test_a_variant_fit_that_failed_to_run_counts_as_not_publishable_and_a_loss_if_s0_was_publishable"]),
    ("Y8 a failed variant fit counts as publishable", A, 'if new[u]["ok"] else False for u in users', 'if new[u]["ok"] else True for u in users', [T + "test_a_variant_fit_that_failed_to_run_counts_as_not_publishable_and_a_loss_if_s0_was_publishable"]),
    ("Y9 the harm margin is -0.20", A, "HARM_MARGIN = -0.10", "HARM_MARGIN = -0.20", [T + "test_not_harmed_boundary_is_inclusive_and_nan_is_not_harmless"]),
    ("Y10 the harm level is exclusive", A, "bool(ps[\"lower\"] >= HARM_MARGIN)", "bool(ps[\"lower\"] > HARM_MARGIN)", [T + "test_not_harmed_boundary_is_inclusive_and_nan_is_not_harmless"]),
    ("Y11 the held-out difference is reference minus variant", A, 'd = [new[u]["dual"]["test_mean"] - ref[u]["dual"]["test_mean"] for u in pair', 'd = [ref[u]["dual"]["test_mean"] - new[u]["dual"]["test_mean"] for u in pair', [T + "test_heldout_pairs_skip_nonfinite_and_use_new_minus_reference"]),
    ("Y12 non-finite variant scores are not skipped", A, 'if new[u]["dual"].get("test_finite") and ref[u]["dual"].get("test_finite")]', 'if ref[u]["dual"].get("test_finite")]', [T + "test_heldout_pairs_skip_nonfinite_and_use_new_minus_reference"]),
    ("Y13 a non-finite lower bound counts as harmless", A, "if math.isfinite(ps[\"lower\"]) else False", "if math.isfinite(ps[\"lower\"]) else True", [T + "test_not_harmed_boundary_is_inclusive_and_nan_is_not_harmless"]),
    ("Y14 the time ratio uses means", A, "float(np.median(t_new) / np.median(t_ref))", "float(np.mean(t_new) / np.mean(t_ref))", [T + "test_categories_count_only_non_publishable_fits_and_the_clock_and_time_ratio_are_reported"]),
    ("Y15 the clock stops are not counted", A, 'clock_stops=sum(1 for u in pair if new[u]["dual"].get("wall_clock_stop"))', "clock_stops=0", [T + "test_categories_count_only_non_publishable_fits_and_the_clock_and_time_ratio_are_reported"]),
    ("Y16 a changed converged fit is counted when it stays converged", A, '(new[u]["dual"]["fit_state"] != "CONVERGED")', '(new[u]["dual"]["fit_state"] == "CONVERGED")', [T + "test_the_converged_reference_fits_that_stop_being_converged_are_counted"]),
    ("Y17 the SQUAREM count level is exclusive", A, "capped >= SQUAREM_MIN_STILL_CAPPED", "capped > SQUAREM_MIN_STILL_CAPPED", [T + "test_squarem_trigger_needs_both_conditions_at_their_boundaries"]),
    ("Y18 the SQUAREM time level is 2", A, "SQUAREM_MIN_TIME_RATIO = 3.0", "SQUAREM_MIN_TIME_RATIO = 2.0", [T + "test_squarem_trigger_needs_both_conditions_at_their_boundaries"]),
    ("Y19 the hard stratum ends one user early", A, 'return "hard" if u <= 11212 else ("medium" if u <= 11224 else "easy")\n\n\ndef main', 'return "hard" if u <= 11211 else ("medium" if u <= 11224 else "easy")\n\n\ndef main', [T + "test_the_stratum_split_follows_the_user_blocks"]),
    ("Y20 the ALL variant leaves the prior out", L, 'dict(max_duration=60, max_iter=600, _flags=dict(MAX_FIT_SECONDS=900.0, DURATION_PRIOR_SEGMENTS=1.0))', 'dict(max_duration=60, max_iter=600, _flags=dict(MAX_FIT_SECONDS=900.0))', [TL + "test_variants_override_only_what_they_name", TL + "test_class_flags_of_a_variant_are_set_for_the_fit_and_restored_after_it"]),
    ("Y21 the class flags are not restored", L, "    finally:\n        for k, v in old.items():\n            setattr(DualTimescaleHSSM, k, v)\n", "    finally:\n        pass\n", [TL + "test_class_flags_of_a_variant_are_set_for_the_fit_and_restored_after_it"]),
    ("Y22 the _flags key is passed to fit_hssm", L, '    flags = dict(kw.pop("_flags", {}))', '    flags = dict(kw.get("_flags", {}))', [TL + "test_class_flags_of_a_variant_are_set_for_the_fit_and_restored_after_it"]),
]
if __name__ == "__main__":
    sys.exit(R.run(MUT, sys.argv[1] if len(sys.argv) > 1 else None, workers=1))
