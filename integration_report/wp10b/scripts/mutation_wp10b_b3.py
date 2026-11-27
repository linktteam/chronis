"""WP10B B3: mutation control of the dual predictive certificate (`dual_cert_lib.py`) against `tests/test_wp10b_dual_cert.py` (named test must fail by its own assertion; control first; runner `mutation_runner10b.py`).
usage: PYTHONPATH=. python3.12 integration_report/wp10b/scripts/mutation_wp10b_b3.py [out.txt]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutation_runner10b as R

F = "integration_report/wp10b/scripts/dual_cert_lib.py"
T = "tests/test_wp10b_dual_cert.py::"
ELIG = T + "test_eligibility_converged_is_not_applicable_unless_a_control_asks_for_any_state"
MUT = [
    ("Z1 a converged fit is not 'not applicable'", F, '        if not el.get("apply_to_any_state"):\n            if el.get("em_state") == "CONVERGED":', '        if not el.get("apply_to_any_state"):\n            if False:', [ELIG]),
    ("Z2 a control cannot skip the EM-state test", F, '        if not el.get("apply_to_any_state"):', '        if True:', [ELIG]),
    ("Z3 any EM state is eligible", F, '            if el.get("em_state") not in C.ELIGIBLE_EM_STATES:', '            if False:', [ELIG]),
    ("Z4 other withhold reasons do not refuse", F, '        if el.get("other_reasons"):', '        if False:', [T + "test_any_other_withhold_reason_refuses_without_refitting"]),
    ("Z5 all-missing days are scored", F, "        obs = observed(X)\n", "        obs = np.ones(len(X), dtype=bool)\n", [T + "test_all_missing_days_are_left_out_of_every_score"]),
    ("Z6 the dual increments are not masked", F, "[o:e][keep]\n            g = ", "[o:e]\n            g = ", [T + "test_all_missing_days_are_left_out_of_every_score"]),
    ("Z7 the training Gaussian includes missing days", F, "C.gaussian_heldout(X[:o][obs[:o]], X[o:e][keep])", "C.gaussian_heldout(X[:o], X[o:e][keep])", [T + "test_all_missing_days_are_left_out_of_every_score"]),
    ("Z8 margin default is two standard errors", F, "z: float = Z, tau: float = TAU,\n                 fractions", "z: float = 2.0, tau: float = TAU,\n                 fractions", [T + "test_a_clear_and_stable_gain_is_certified_and_uses_the_wp9_constants"]),
    ("Z9 stability limit default is 0.5", F, "z: float = Z, tau: float = TAU,", "z: float = Z, tau: float = 0.5,", [T + "test_a_clear_and_stable_gain_is_certified_and_uses_the_wp9_constants"]),
    ("Z10 other rolling origins", F, "fractions: Sequence[float] = COLD_START_CERT_FRACTIONS", "fractions: Sequence[float] = (0.5, 0.7, 0.9)", [T + "test_a_clear_and_stable_gain_is_certified_and_uses_the_wp9_constants"]),
    ("Z11 the dual refit sees the rows up to the end of the block", F, "m = refit_fn(X[:o])", "m = refit_fn(X[:e])", [T + "test_the_refit_sees_only_the_rows_before_each_origin"]),
    ("Z12 the baseline refit sees all the rows", F, "bm = baseline_refit_fn(X[:o])", "bm = baseline_refit_fn(X)", [T + "test_the_baseline_refit_sees_only_the_rows_before_each_origin"]),
    ("Z13 the baseline gain has the wrong sign", F, "parts_b.append(lp - lb)", "parts_b.append(lb - lp)", [T + "test_the_baseline_comparison_is_reported_next_to_the_gaussian_gain"]),
    ("Z14 the baseline lower bound has no margin", F, "lower = mean - float(z) * se if math.isfinite(se) else None", "lower = mean if math.isfinite(se) else None", [T + "test_the_baseline_lower_bound_uses_the_same_margin_and_the_beats_flag_follows_it"]),
    ("Z15 beating the baseline is judged on the mean", F, 'beats_baseline=bool(lower is not None and lower > 0.0))', 'beats_baseline=bool(mean > 0.0))', [T + "test_the_baseline_lower_bound_uses_the_same_margin_and_the_beats_flag_follows_it"]),
    ("Z16 the stability is measured against the full fit itself", F, "C.stability_distance(full_model.regime_observation_means(), last.regime_observation_means(), sd)", "C.stability_distance(full_model.regime_observation_means(), full_model.regime_observation_means(), sd)", [T + "test_unstable_regime_means_refuse_even_a_clear_gain"]),
    ("Z17 the stability ignores the feature spread", F, "sd = np.nanstd(X[obs], axis=0, ddof=1)", "sd = np.ones(X.shape[1])", [T + "test_stability_is_measured_in_units_of_the_feature_spread"]),
    ("Z18 a degraded refit is accepted", F, '            if getattr(m, "degraded", False):', "            if False:", [T + "test_a_failed_or_degraded_refit_is_a_refusal"]),
    ("Z19 a failed refit gives the wrong reason", F, "return dict(base, reasons=[C.R_REFIT], blocks=[list(b) for b in blocks])", "return dict(base, reasons=[C.R_ERROR], blocks=[list(b) for b in blocks])", [T + "test_a_failed_or_degraded_refit_is_a_refusal"]),
    ("Z20 an unexpected error leaves the reasons empty", F, "reasons=[C.R_ERROR], error=", "reasons=[], error=", [T + "test_an_exception_in_scoring_is_a_refusal_not_a_crash"]),
    ("Z21 infinite values are not refused", F, "        if np.isinf(X).any():", "        if False:", [T + "test_too_few_rows_and_infinite_values_are_refused"]),
    ("Z22 too few rows is not refused", F, "        if blocks is None:", "        if False:", [T + "test_too_few_rows_and_infinite_values_are_refused"]),
    ("Z23 a baseline from two increments is reported", F, "if len(db) >= 3 and np.all(np.isfinite(db)):", "if len(db) >= 300 and np.all(np.isfinite(db)):", [T + "test_the_baseline_comparison_is_reported_next_to_the_gaussian_gain"]),
    ("Z24 a non-finite baseline difference is reported as a number", F, "if len(db) >= 3 and np.all(np.isfinite(db)):", "if len(db) >= 3:", [T + "test_a_non_finite_baseline_difference_is_reported_as_not_beating_it"]),
    ("Z25 the gain is not taken over the Gaussian", F, "parts.append(lp - g)", "parts.append(lp)", [T + "test_a_clear_and_stable_gain_is_certified_and_uses_the_wp9_constants"]),
    ("Z26 an all-missing day is a row with one observed feature", F, "return ~np.isnan(np.asarray(X, dtype=float)).all(axis=1)", "return ~np.isnan(np.asarray(X, dtype=float)).any(axis=1)", [T + "test_the_observed_row_mask"]),
    ("Z28 the baseline refit ignores the user's seed", F, "return BaselineAdapter(dual_lib.fit_baseline(X_train, seed).model)", "return BaselineAdapter(dual_lib.fit_baseline(X_train, 0).model)", [T + "test_the_baseline_refit_passes_the_seed_and_the_rows_to_the_production_baseline_fit"]),
    ("Z27 the baseline decides the certificate", F, '        out.update(blocks=[list(b) for b in blocks], schema_version=1, baseline=None)\n', '        out.update(blocks=[list(b) for b in blocks], schema_version=1, baseline=None)\n        if parts_b and float(np.concatenate(parts_b).mean()) < 0.0:\n            out["state"] = C.STATE_NOT_CERTIFIED\n', [T + "test_the_baseline_does_not_decide_the_certificate"]),
]

if __name__ == "__main__":
    sys.exit(R.run(MUT, sys.argv[1] if len(sys.argv) > 1 else None, workers=2))
