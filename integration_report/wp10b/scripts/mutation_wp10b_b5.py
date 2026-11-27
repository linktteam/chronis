"""WP10B B5: mutation control of the feasibility arithmetic (`b5_feasibility.py`) against `tests/test_wp10b_b5_feasibility.py` (criteria Part 10, section 16).
Caught only if the NAMED test fails by its own assertion (runner: `mutation_runner10b.py`); the unmutated control runs first in the same invocation.
usage: PYTHONPATH=. python3.12 integration_report/wp10b/scripts/mutation_wp10b_b5.py [out.txt]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutation_runner10b as R

S = "integration_report/wp10b/scripts/b5_feasibility.py"
T = "tests/test_wp10b_b5_feasibility.py::"
MUT = [
    ("H01 ten alignments per chain instead of nine", S, "ALIGNMENTS_PER_CHAIN = 9 ", "ALIGNMENTS_PER_CHAIN = 10 ", [T + "test_chains_needed_counts_nine_alignments_per_chain_and_rounds_up"]),
    ("H02 chains rounded down", S, "math.ceil(n / (per_chain * rate))", "math.floor(n / (per_chain * rate))", [T + "test_chains_needed_counts_nine_alignments_per_chain_and_rounds_up"]),
    ("H03 a zero rate is not treated as impossible", S, "math.inf if rate <= 0 else", "math.inf if rate < 0 else", [T + "test_chains_needed_counts_nine_alignments_per_chain_and_rounds_up"]),
    ("H04 the bound uses the level instead of its complement", S, "return 1.0 - (1.0 - level) ** (1.0 / n)", "return 1.0 - level ** (1.0 / n)", [T + "test_the_required_n_is_299_for_a_one_percent_bound"]),
    ("H05 the required bound is 5% instead of 1%", S, "BOUND, LEVEL = 0.01, 0.95", "BOUND, LEVEL = 0.05, 0.95", [T + "test_the_required_n_is_299_for_a_one_percent_bound"]),
    ("H06 the cost ignores the workers", S, "return fits * seconds_per_fit / workers / 3600.0", "return fits * seconds_per_fit / 3600.0", [T + "test_cost_hours_divides_by_the_workers"]),
    ("H07 the cost budget is exclusive", S, 'out["hours_favourable"] <= budget', 'out["hours_favourable"] < budget', [T + "test_the_cost_budget_is_inclusive"]),
    ("H08 the budget is 60 hours", S, "BUDGET_HOURS = 6.0", "BUDGET_HOURS = 60.0", [T + "test_the_cost_budget_is_inclusive", T + "test_all_three_conditions_must_hold_to_run_b5"]),
    ("H09 the favourable count is the larger one", S, "best = min(total_all, total_pub)", "best = max(total_all, total_pub)", [T + "test_the_favourable_count_is_the_smaller_one", T + "test_the_s0_rates_are_infeasible_and_the_numbers_are_pinned"]),
    ("H10 the run condition needs only one of cost and passing alignments", S, 'out["a_cost_ok"] and out["c_passing_ok"]', 'out["a_cost_ok"] or out["c_passing_ok"]', [T + "test_all_three_conditions_must_hold_to_run_b5"]),
    ("H11 one pair type with enough passing alignments is enough", S, "all(k >= MIN_PASSING_DEV for k, _ in passing.values())", "any(k >= MIN_PASSING_DEV for k, _ in passing.values())", [T + "test_all_three_conditions_must_hold_to_run_b5"]),
    ("H12 the passing threshold is 4", S, "MIN_PASSING_DEV = 20 ", "MIN_PASSING_DEV = 4 ", [T + "test_all_three_conditions_must_hold_to_run_b5"]),
    ("H13 a dual stability signal is not required", S, "and bool(dual_signal)", "and True", [T + "test_all_three_conditions_must_hold_to_run_b5"]),
    ("H14 the variant filter is ignored", S, 'if variant is not None and r["job"].get("variant") != variant:', "if False:", [T + "test_publishable_shares_and_median_seconds_by_stratum_and_variant"]),
    ("H15 a failed fit is counted", S, 'if not r.get("ok"):', "if False:", [T + "test_publishable_shares_and_median_seconds_by_stratum_and_variant"]),
    ("H16 the mean instead of the median fit time", S, "statistics.median(secs)", "statistics.mean(secs)", [T + "test_publishable_shares_and_median_seconds_by_stratum_and_variant"]),
    ("H17 the dual signal test is inverted", S, 'facts["dual_stability_flag"] != (False, None)', 'facts["dual_stability_flag"] == (False, None)', [T + "test_main_prints_the_verdict_for_both_configurations"]),
    ("H18 fits per chain is 9", S, "FITS_PER_CHAIN = 10", "FITS_PER_CHAIN = 9", [T + "test_chains_needed_counts_nine_alignments_per_chain_and_rounds_up", T + "test_the_s0_rates_are_infeasible_and_the_numbers_are_pinned"]),
    ("H19 the budget is shared by one stratum instead of three", S, "fits / FITS_PER_CHAIN / len(STRATA)", "fits / FITS_PER_CHAIN", [T + "test_achievable_alignments_per_stratum_in_the_budget_are_whole_chains_shared_by_the_strata"]),
    ("H20 partial chains are counted", S, "int(math.floor(fits / FITS_PER_CHAIN / len(STRATA))) * ALIGNMENTS_PER_CHAIN", "int(round(fits / FITS_PER_CHAIN / len(STRATA) * ALIGNMENTS_PER_CHAIN))", [T + "test_achievable_alignments_per_stratum_in_the_budget_are_whole_chains_shared_by_the_strata"]),
    ("H21 the achievable n ignores the workers", S, "fits = budget_hours * 3600.0 * workers / seconds_per_fit", "fits = budget_hours * 3600.0 / seconds_per_fit", [T + "test_achievable_alignments_per_stratum_in_the_budget_are_whole_chains_shared_by_the_strata"]),
]
if __name__ == "__main__":
    sys.exit(R.run(MUT, sys.argv[1] if len(sys.argv) > 1 else None, workers=2))
