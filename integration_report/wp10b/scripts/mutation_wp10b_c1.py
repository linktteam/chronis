"""WP10B C1(2): mutation control of the mild-prior construction (`c1_mild_run.py`) and rules (`c1_mild_analyze.py`) against `tests/test_wp10b_c1.py` (named test must fail by its own assertion; unmutated
control first, same run; runner `mutation_runner10b.py`). usage: PYTHONPATH=. python3.12 integration_report/wp10b/scripts/mutation_wp10b_c1.py [out.txt]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutation_runner10b as R

RUN = "integration_report/wp10b/scripts/c1_mild_run.py"
AN = "integration_report/wp10b/scripts/c1_mild_analyze.py"
T = "tests/test_wp10b_c1.py::"
MUT = [
    ("D01 M1 shifts by 7", RUN, 'out["d_obs"] = out["d_obs"] + 8.0', 'out["d_obs"] = out["d_obs"] + 7.0', [T + "test_m1_and_m2_shift_every_intercept_and_nothing_else"]),
    ("D02 M2 shifts by 10", RUN, 'out["d_obs"] = out["d_obs"] + 12.0', 'out["d_obs"] = out["d_obs"] + 10.0', [T + "test_m1_and_m2_shift_every_intercept_and_nothing_else"]),
    ("D03 M3 exchanges only the intercepts", RUN, 'for k in ("d_obs", "C_obs", "R_obs"):\n            out[k] = out[k][::-1].copy()', 'for k in ("d_obs",):\n            out[k] = out[k][::-1].copy()',
     [T + "test_m3_exchanges_the_observation_model_of_the_two_regimes_and_keeps_the_dynamics"]),
    ("D04 M3 exchanges nothing", RUN, "out[k] = out[k][::-1].copy()", "out[k] = out[k].copy()", [T + "test_m3_exchanges_the_observation_model_of_the_two_regimes_and_keeps_the_dynamics"]),
    ("D05 the norm is the L1 norm", RUN, "return float(np.linalg.norm(new - np.asarray(d[\"d_obs\"], dtype=float)))", "return float(np.abs(new - np.asarray(d[\"d_obs\"], dtype=float)).sum())", [T + "test_every_candidate_is_milder_than_h3_by_the_declared_norm"]),
    ("D06 the H3 reference shift is 8", RUN, "H3_SHIFT = 16.0", "H3_SHIFT = 8.0", [T + "test_every_candidate_is_milder_than_h3_by_the_declared_norm"]),
    ("D07 an unknown candidate falls through silently", RUN, 'raise ValueError(f"unknown candidate {cand!r}")', "pass", [T + "test_an_unknown_candidate_is_an_error"]),
    ("E01 rule (b) needs five", AN, "B_MIN, C_MIN, N_DEV = 6, 4, 36", "B_MIN, C_MIN, N_DEV = 5, 4, 36", [T + "test_validity_needs_six_below_the_gaussian_and_four_of_them_eligible"]),
    ("E02 rule (c) needs three", AN, "B_MIN, C_MIN, N_DEV = 6, 4, 36", "B_MIN, C_MIN, N_DEV = 6, 3, 36", [T + "test_validity_needs_six_below_the_gaussian_and_four_of_them_eligible"]),
    ("E03 a fit equal to the Gaussian is worse", AN, "return (not math.isfinite(s)) or s < gauss", "return (not math.isfinite(s)) or s <= gauss", [T + "test_a_fit_equal_to_the_gaussian_is_not_worse_and_a_nonfinite_fit_is"]),
    ("E04 a non-finite fit is not worse", AN, "return (not math.isfinite(s)) or s < gauss", "return math.isfinite(s) and s < gauss", [T + "test_a_fit_equal_to_the_gaussian_is_not_worse_and_a_nonfinite_fit_is"]),
    ("E05 rule (b) is strict", AN, "ok_b, ok_c = worse >= B_MIN, worse_elig >= C_MIN", "ok_b, ok_c = worse > B_MIN, worse_elig >= C_MIN", [T + "test_validity_needs_six_below_the_gaussian_and_four_of_them_eligible"]),
    ("E06 the most severe candidate is frozen", AN, "return min(ok, key=lambda c: (norms[c], ORDER.index(c))) if ok else None", "return max(ok, key=lambda c: (norms[c], ORDER.index(c))) if ok else None", [T + "test_the_mildest_verified_candidate_is_frozen"]),
    ("E07 an unverified candidate can be chosen", AN, 'ok = [c for c in ORDER if c in results and results[c]["verified"]]', 'ok = [c for c in ORDER if c in results]', [T + "test_the_mildest_verified_candidate_is_frozen"]),
    ("E08 the blind count includes arm T", AN, 'if r["arm"] != "M":\n            continue', 'if r["arm"] == "gauss":\n            continue', [T + "test_the_blind_count_counts_only_eligible_fits_below_the_gaussian_in_arm_m"]),
    ("E09 the blind count reads the certification field", AN, 'if math.isfinite(gg) and r.get("eligible") and worse_than_gauss(r, gg):', 'if math.isfinite(gg) and r.get("eligible") and not certified(r) and worse_than_gauss(r, gg):', [T + "test_the_blind_count_does_not_depend_on_the_certification_field"]),
    ("E10 the blind count ignores eligibility", AN, 'if math.isfinite(gg) and r.get("eligible") and worse_than_gauss(r, gg):', 'if math.isfinite(gg) and worse_than_gauss(r, gg):', [T + "test_the_blind_count_counts_only_eligible_fits_below_the_gaussian_in_arm_m"]),
    ("E11 the extension is allowed at twelve", AN, "return n_eligible_harmful < REQUIRED_EVAL", "return n_eligible_harmful <= REQUIRED_EVAL", [T + "test_the_extension_is_allowed_only_below_twelve"]),
    ("E12 the required number is ten", AN, "REQUIRED_EVAL = 12", "REQUIRED_EVAL = 10", [T + "test_verdict_e1", T + "test_the_extension_is_allowed_only_below_twelve"]),
    ("E13 one certified fit is tolerated", AN, "    if k >= 1:\n        return \"NOT MET\"", "    if k >= 2:\n        return \"NOT MET\"", [T + "test_verdict_e1"]),
    ("E14 MET needs more than twelve", AN, 'return "MET" if n >= REQUIRED_EVAL else', 'return "MET" if n > REQUIRED_EVAL else', [T + "test_verdict_e1"]),
    ("E15 all-certified is not capped at one", AN, "return 1.0 if k >= n else", "return 1.0 if k > n else", [T + "test_cp_upper_matches_the_closed_form_for_zero_events"]),
    ("E16 the bound is two-sided 90%", AN, "def cp_upper(k: int, n: int, alpha: float = 0.05)", "def cp_upper(k: int, n: int, alpha: float = 0.10)", [T + "test_cp_upper_matches_the_closed_form_for_zero_events", T + "test_evaluate_reports_the_counts_and_the_bound"]),
    ("E17 E1 counts all arm-M fits that are eligible, worse or not", AN, 'ew = [r for r in worse if r.get("eligible")]', 'ew = [r for r in usable if r.get("eligible")]', [T + "test_evaluate_reports_the_counts_and_the_bound"]),
    ("E18 certified means not certified", AN, 'return (rec.get("certification") or {}).get("state") == CERTIFIED', 'return (rec.get("certification") or {}).get("state") != CERTIFIED', [T + "test_evaluate_reports_the_counts_and_the_bound"]),
    ("E19 the Gaussian of the wrong pair is used", AN, 'g = gauss_from(recs)\n    M = [r for r in recs if r["arm"] == "M"]', 'g = {k: -99.0 for k in gauss_from(recs)}\n    M = [r for r in recs if r["arm"] == "M"]', [T + "test_evaluate_reports_the_counts_and_the_bound"]),
]
if __name__ == "__main__":
    sys.exit(R.run(MUT, sys.argv[1] if len(sys.argv) > 1 else None, workers=2))
