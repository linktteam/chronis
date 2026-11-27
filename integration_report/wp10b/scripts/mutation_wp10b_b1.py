"""WP10B B1: mutation control of the dual-study library (`dual_lib.py`) and the diagnosis analysis (`b1_analyze.py`) against `tests/test_wp10b_dual_lib.py` and `tests/test_wp10b_b1.py` (named test must fail by its own
assertion; unmutated control first, same run; runner `mutation_runner10b.py`). usage: PYTHONPATH=. python3.12 integration_report/wp10b/scripts/mutation_wp10b_b1.py [out.txt]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutation_runner10b as R

LIB = "integration_report/wp10b/scripts/dual_lib.py"
AN = "integration_report/wp10b/scripts/b1_analyze.py"
TL = "tests/test_wp10b_dual_lib.py::"
TB = "tests/test_wp10b_b1.py::"
MUT = [
    ("L01 the scaler uses the whole record", LIB, "mean = np.nanmean(raw[:t_train], axis=0)\n    std = np.nanstd(raw[:t_train], axis=0)", "mean = np.nanmean(raw, axis=0)\n    std = np.nanstd(raw, axis=0)", [TL + "test_the_train_block_equals_the_pipeline_window_and_the_test_block_uses_the_train_scaler"]),
    ("L02 a missing single feature is not filled", LIB, "z = np.where(np.isnan(z), 0.0, z)\n    z[day_missing] = np.nan", "z[day_missing] = np.nan", [TL + "test_the_train_block_equals_the_pipeline_window_and_the_test_block_uses_the_train_scaler"]),
    ("L03 the baseline forward omits the duration continuation", LIB, "cont[:, : D - 1] = alpha[t - 1, :, 1:D]", "cont[:, : D - 1] = alpha[t - 1, :, 0:D - 1]", [TL + "test_baseline_increments_sum_to_the_models_own_log_likelihood"]),
    ("L04 the baseline increment is not differenced", LIB, "    lse = logsumexp(alpha.reshape(T, -1), axis=1)\n    return np.diff(np.concatenate([[0.0], lse]))\n\n\ndef gaussian_increments", "    lse = logsumexp(alpha.reshape(T, -1), axis=1)\n    return lse\n\n\ndef gaussian_increments", [TL + "test_baseline_increments_sum_to_the_models_own_log_likelihood"]),
    ("L05 the baseline ignores the initial distribution", LIB, "alpha[0, k, :] = log_pi[k] + dur_logpmf[k] + emis[0][k]", "alpha[0, k, :] = dur_logpmf[k] + emis[0][k]", [TL + "test_baseline_increments_sum_to_the_models_own_log_likelihood"]),
    ("L06 the dual increment is not differenced", LIB, "    lse = logsumexp(fwd.alpha.reshape(len(X), -1), axis=1)\n    return np.diff(np.concatenate([[0.0], lse]))", "    lse = logsumexp(fwd.alpha.reshape(len(X), -1), axis=1)\n    return lse", [TL + "test_dual_increments_equal_the_n18_harness_and_sum_to_the_forward_log_likelihood"]),
    ("L07 the Gaussian uses the test moments", LIB, "tr = X_train[~np.isnan(X_train).all(axis=1)]", "tr = X_test[~np.isnan(X_test).all(axis=1)]", [TL + "test_gaussian_increments_use_train_moments_only_and_leave_missing_rows_nan"]),
    ("L08 the Gaussian covariance uses ddof 0", LIB, "cov = np.atleast_2d(np.cov(tr, rowvar=False, ddof=1))", "cov = np.atleast_2d(np.cov(tr, rowvar=False, ddof=0))", [TL + "test_gaussian_increments_use_train_moments_only_and_leave_missing_rows_nan"]),
    ("L09 the test mean includes missing days", LIB, "keep = ~np.asarray(day_missing)[t_train:len(lp_full)]", "keep = np.ones(len(lp) , dtype=bool)", [TL + "test_test_mean_averages_the_observed_test_days_only"]),
    ("L10 the test mean ignores non-finite values", LIB, "bool(len(x) > 0 and np.all(np.isfinite(x)))", "bool(len(x) > 0)", [TL + "test_test_mean_averages_the_observed_test_days_only"]),
    ("L11 MAX_ITER is classed as rejected", LIB, 'cats.append("iteration_cap" if "MAX_ITER_REACHED" in s else ("rejected_update" if "REJECTED_UPDATE" in s else "other_non_convergence"))', 'cats.append("rejected_update" if "MAX_ITER_REACHED" in s else ("iteration_cap" if "REJECTED_UPDATE" in s else "other_non_convergence"))', [TL + "test_reason_categories"]),
    ("L12 numerical reasons are classed as other", LIB, 's.startswith(("fit_degraded", "fit_state is NUMERICAL_FAILURE", "states_pass", "late_fallbacks"))', 's.startswith(("fit_degraded",))', [TL + "test_reason_categories"]),
    ("L13 the creep rate is the mean ratio", LIB, "r = float(np.median([b / a for a, b in pos]))", "r = float(np.mean([b / a for a, b in pos]) * 1.01)", [TL + "test_creep_diagnostics_recover_the_geometric_rate_and_the_extrapolated_count"]),
    ("L14 the extrapolation uses the wrong tolerance", LIB, "math.ceil(math.log(tol / d[-1]) / math.log(r))", "math.ceil(math.log(tol / d[-1]) / math.log(r)) + 1", [TL + "test_creep_diagnostics_recover_the_geometric_rate_and_the_extrapolated_count"]),
    ("L15 a rate of at least one still extrapolates", LIB, "if 0 < r < 1 and d[-1] > 0:", "if 0 < r and d[-1] > 0:", [TL + "test_creep_diagnostics_when_already_converged_or_not_contracting"]),
    ("L16 the gain over the last ten iterations uses nine", LIB, "gain_last10=float(h[-1] - h[-11])", "gain_last10=float(h[-1] - h[-10])", [TL + "test_creep_diagnostics_recover_the_geometric_rate_and_the_extrapolated_count"]),
    ("L17 publishable is not tied to the withhold reason", LIB, "publishable=bool(res.withheld_reason is None)", "publishable=bool(res.fit_state.value == 'CONVERGED')", [TL + "test_the_record_of_a_tiny_real_fit_is_consistent", TL + "test_publishable_follows_the_withhold_reason_not_the_em_state"]),
    ("L18 the D60 variant keeps Dmax 45", LIB, '"D60": dict(max_duration=60)', '"D60": dict(max_duration=45)', [TL + "test_variants_override_only_what_they_name"]),
    ("A01 wall regimes that are not degenerate are counted", AN, "wall = {k for k, v in d.get(\"at_bound\", {}).items() if v}\n    rising = {k for k, v in enumerate(d.get(\"pmf_rising\", [])) if v}\n    return dict(degenerate=len(deg), wall=len(wall & deg)", "wall = {k for k, v in d.get(\"at_bound\", {}).items() if v}\n    rising = {k for k, v in enumerate(d.get(\"pmf_rising\", [])) if v}\n    return dict(degenerate=len(deg), wall=len(wall)", [TB + "test_wall_or_rising_flags_of_a_non_degenerate_regime_are_not_counted"]),
    ("A02 both is the union", AN, "both=len(wall & rising & deg)", "both=len((wall | rising) & deg)", [TB + "test_split_degeneracy_separates_wall_rising_and_both"]),
    ("A03 publishable fits are counted as non-publishable", AN, "non = [d for d in D if not d[\"publishable\"]]", "non = [d for d in D]", [TB + "test_summarize_counts_by_stratum_category_and_exclusive_combination"]),
    ("A04 failed fits are included in the dual list", AN, "ok = [r for r in R if r[\"ok\"]]", "ok = [r for r in R]", [TB + "test_summarize_counts_by_stratum_category_and_exclusive_combination"]),
    ("A05 the exclusive key is unsorted", AN, "excl = collections.Counter(tuple(sorted(d[\"categories\"])) for d in non)", "excl = collections.Counter(tuple(d[\"categories\"]) for d in non)", [TB + "test_summarize_counts_by_stratum_category_and_exclusive_combination"]),
    ("A06 the budget is exclusive", AN, "sum(1 for e in extra if e is not None and e <= b)", "sum(1 for e in extra if e is not None and e < b)", [TB + "test_the_creep_budget_table_counts_fits_extrapolated_within_each_extra_budget"]),
    ("A07 rejected fits are not in the not-finished group", AN, 'if "iteration_cap" in d["categories"] or "rejected_update" in d["categories"]]', 'if "iteration_cap" in d["categories"]]', [TB + "test_a_fit_that_is_rejected_counts_in_the_not_finished_by_em_group"]),
    ("A08 the held-out gap is the other way round", AN, 'v = [r["dual"]["test_mean"] - r[nm]["test_mean"] for r in ok', 'v = [r[nm]["test_mean"] - r["dual"]["test_mean"] for r in ok', [TB + "test_heldout_gaps_are_paired_per_user_and_skip_nonfinite_pairs"]),
    ("A09 non-finite comparators are not skipped", AN, 'if r.get(nm, {}).get("test_finite") and r["dual"].get("test_finite")]', 'if r["dual"].get("test_finite")]', [TB + "test_heldout_gaps_are_paired_per_user_and_skip_nonfinite_pairs"]),
    ("A10 the Wilson interval uses z = 1.645", AN, "def wilson(k: int, n: int, z: float = 1.959964)", "def wilson(k: int, n: int, z: float = 1.645)", [TB + "test_wilson_and_fmt"]),
]
if __name__ == "__main__":
    sys.exit(R.run(MUT, sys.argv[1] if len(sys.argv) > 1 else None, workers=2))
