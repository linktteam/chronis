"""WP10B B2: mutation control of the two opt-in switches of the dual model (`backbone/hssm/dual_model.py`) and of the dual bit-identity comparison (`dual_battery.py`), against `tests/test_wp10b_b2.py` and
`tests/test_wp10b_dual_battery.py` (a mutant is caught only if a NAMED test fails by its own assertion; the unmutated control passes first, in the same run; runner `mutation_runner10b.py`).
usage: PYTHONPATH=. python3.12 integration_report/wp10b/scripts/mutation_wp10b_b2.py [out.txt]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutation_runner10b as R

DM = "backbone/hssm/dual_model.py"
DB = "integration_report/wp10b/scripts/dual_battery.py"
T = "tests/test_wp10b_b2.py::"
TB = "tests/test_wp10b_dual_battery.py::"
ON = T + "test_off_the_update_is_the_plain_weighted_fit_and_on_it_equals_the_fit_of_the_augmented_counts"
MUT = [
    ("P1 the guard for an empty regime counts the prior", DM, "if N_j_d.sum() <= 1e-10:", "if N_j_d.sum() + float(self.DURATION_PRIOR_SEGMENTS) <= 1e-10:", [T + "test_a_regime_without_expected_segments_is_not_changed_by_the_prior"]),
    ("P2 the prior weight is doubled", DM, "N_j_d = N_j_d + float(self.DURATION_PRIOR_SEGMENTS) * self._duration_prior_pmf(d_grid)", "N_j_d = N_j_d + 2.0 * float(self.DURATION_PRIOR_SEGMENTS) * self._duration_prior_pmf(d_grid)", [ON]),
    ("P3 the prior pmf is not normalised", DM, "        pdf = np.clip(pdf, 1e-300, None)\n        return pdf / pdf.sum()\n\n    def _now", "        pdf = np.clip(pdf, 1e-300, None)\n        return pdf\n\n    def _now", [T + "test_prior_pmf_is_a_normalised_lognormal_on_the_lattice"]),
    ("P4 the prior median is read from the sigma", DM, "scale=float(self.DURATION_PRIOR_MEDIAN))", "scale=float(self.DURATION_PRIOR_SIGMA))", [T + "test_prior_pmf_is_a_normalised_lognormal_on_the_lattice"]),
    ("P5 the prior sigma is read from the median", DM, "s=float(self.DURATION_PRIOR_SIGMA), scale", "s=float(self.DURATION_PRIOR_MEDIAN), scale", [T + "test_prior_pmf_is_a_normalised_lognormal_on_the_lattice"]),
    ("P6 the prior pmf is on a shifted lattice", DM, "lognorm.pdf(d_grid, s=float(self.DURATION_PRIOR_SIGMA)", "lognorm.pdf(d_grid + 1, s=float(self.DURATION_PRIOR_SIGMA)", [T + "test_prior_pmf_is_a_normalised_lognormal_on_the_lattice"]),
    ("P7 the prior is on by default", DM, "    DURATION_PRIOR_SEGMENTS: float = 0.0\n", "    DURATION_PRIOR_SEGMENTS: float = 1.0\n", [T + "test_the_defaults_switch_both_features_off", T + "test_a_full_fit_with_the_switch_at_zero_is_bit_identical_to_a_fit_without_touching_it"]),
    ("P8 the prior also changes the transition update", DM, "        self.A_slow = A_new\n\n        # duration parameters", "        self.A_slow = A_new * (1.0 + 0.001 * float(self.DURATION_PRIOR_SEGMENTS))\n\n        # duration parameters", [T + "test_the_prior_leaves_the_transition_and_initial_updates_untouched"]),
    ("C1 the cap is on by default", DM, "    MAX_FIT_SECONDS: Optional[float] = None\n", "    MAX_FIT_SECONDS: Optional[float] = 1e-9\n", [T + "test_the_defaults_switch_both_features_off", T + "test_a_full_fit_with_the_switch_at_zero_is_bit_identical_to_a_fit_without_touching_it"]),
    ("C2 the cap is not strict", DM, "> float(self.MAX_FIT_SECONDS)", ">= float(self.MAX_FIT_SECONDS)", [T + "test_the_cap_is_a_strict_inequality_on_the_elapsed_time"]),
    ("C3 the elapsed time is the clock reading", DM, "self._now() - t_start > float(self.MAX_FIT_SECONDS)", "self._now() > float(self.MAX_FIT_SECONDS)", [T + "test_the_cap_stops_the_run_after_the_iteration_in_which_it_is_exceeded"]),
    ("C4 the stop flag is not set", DM, "                self.wall_clock_stop_ = True\n                break", "                break", [T + "test_the_cap_stops_the_run_after_the_iteration_in_which_it_is_exceeded"]),
    ("C5 the stop returns before the end-of-budget rule", DM, "                self.wall_clock_stop_ = True\n                break", "                self.wall_clock_stop_ = True\n                return self", [T + "test_the_cap_stops_the_run_after_the_iteration_in_which_it_is_exceeded"]),
    ("C6 the flag is not reset at the start of a fit", DM, "        self._em_iteration = 0\n        self.wall_clock_stop_ = False\n        t_start", "        self._em_iteration = 0\n        t_start", [T + "test_the_flag_is_reset_by_the_next_fit"]),
    ("C7 the clock is read even when the cap is off (state leaks into the start time)", DM, "t_start = self._now() if self.MAX_FIT_SECONDS is not None else None", "t_start = self._now()", [T + "test_the_clock_is_never_read_when_the_cap_is_off"]),
    ("B1 the comparison is approximate", DB, "            if not np.array_equal(r[k], n[k]):", "            if not np.allclose(r[k], n[k], rtol=1e-9, atol=1e-12):", [TB + "test_a_one_ulp_difference_in_any_parameter_fails_and_names_it"]),
    ("B2 the log-likelihood history is not compared", DB, 'for k in ("ll_history", "fit_state"', 'for k in ("fit_state"', [TB + "test_any_other_field_difference_fails"]),
    ("B3 only the last record decides", DB, "        ok = ok and not diffs", "        ok = not diffs", [TB + "test_a_difference_in_the_first_record_is_not_hidden_by_identical_later_records"]),
    ("B4 the wall time is compared", DB, '"box_refit", "window_gain_at_cap"):', '"box_refit", "window_gain_at_cap", "seconds"):', [TB + "test_identical_records_pass_and_the_wall_time_is_ignored"]),
]
if __name__ == "__main__":
    sys.exit(R.run(MUT, sys.argv[1] if len(sys.argv) > 1 else None, workers=2))
