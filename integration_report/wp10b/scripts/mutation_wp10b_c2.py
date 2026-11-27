"""WP10B C2: mutation control of the hard-stratum ceiling analysis (`c2_analyze.py`) against `tests/test_wp10b_c2.py` (named test must fail by its own assertion; unmutated control first, same run;
runner `mutation_runner10b.py`). usage: PYTHONPATH=. python3.12 integration_report/wp10b/scripts/mutation_wp10b_c2.py [out.txt]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutation_runner10b as R

AN = "integration_report/wp10b/scripts/c2_analyze.py"
T = "tests/test_wp10b_c2.py::"
MUT = [
    ("G01 the gate is exclusive", AN, "return bool(math.isfinite(sep) and sep >= H)", "return bool(math.isfinite(sep) and sep > H)", [T + "test_gate_level_is_inclusive"]),
    ("G02 the gate level is 2.0", AN, "H = 2.5\n", "H = 2.0\n", [T + "test_the_levels_are_the_declared_ones", T + "test_gate_level_is_inclusive"]),
    ("G03 two days make a regime evaluable", AN, "MIN_DAYS = 3", "MIN_DAYS = 2", [T + "test_a_regime_with_fewer_than_three_days_makes_the_window_non_evaluable"]),
    ("G04 only one regime needs enough days", AN, "if (counts < MIN_DAYS).any():", "if (counts < MIN_DAYS).all():", [T + "test_a_regime_with_fewer_than_three_days_makes_the_window_non_evaluable"]),
    ("G05 a feature without observations is tolerated", AN, "        if np.isnan(sel).all(axis=0).any():\n            return out\n", "", [T + "test_a_regime_without_any_observed_value_of_a_feature_is_non_evaluable"]),
    ("G06 the oracle uses the fitted-style pooled means of the whole window", AN, "means.append(np.nanmean(sel, axis=0))", "means.append(np.nanmean(X, axis=0))", [T + "test_the_oracle_gate_quantity_is_separation_min_on_the_true_labels_and_sample_means", T + "test_a_well_separated_window_passes_the_gate_and_an_unseparated_one_does_not"]),
    ("G07 the oracle uses the wrong labels (all zero)", AN, "out[\"sep_emp\"] = float(separation_min(np.array(means), X, lab))", "out[\"sep_emp\"] = float(separation_min(np.array(means), X, np.zeros_like(lab)))", [T + "test_the_oracle_gate_quantity_is_separation_min_on_the_true_labels_and_sample_means"]),
    ("G08 the generator-means arm uses the sample means", AN, "gm = np.array([np.asarray(means_t, dtype=float)[lab == k].mean(axis=0) for k in range(K)])", "gm = np.array(means)", [T + "test_the_generator_means_arm_uses_the_per_day_true_means_averaged_by_regime"]),
    ("G09 production separation uses the true labels", AN, "np.array(r8[\"labels_fit\"], dtype=int)))", "np.zeros(len(prep[\"raw\"]), dtype=int)))", [T + "test_production_separation_uses_the_fitted_means_and_labels_not_the_true_ones"]),
    ("G10 the sharpness floor is 0.90", AN, "SHARP = 0.95", "SHARP = 0.90", [T + "test_the_production_flag_is_the_consensus_flag_or_a_sharpness_below_the_floor", T + "test_the_levels_are_the_declared_ones"]),
    ("G11 the consensus flag is ignored", AN, "flag = bool(cons[\"fit_unstable\"]) or (r8[\"max_post\"] < SHARP)", "flag = (r8[\"max_post\"] < SHARP)", [T + "test_the_production_flag_is_the_consensus_flag_or_a_sharpness_below_the_floor"]),
    ("G12 sharpness equal to the floor is flagged", AN, "(r8[\"max_post\"] < SHARP)", "(r8[\"max_post\"] <= SHARP)", [T + "test_the_production_flag_is_the_consensus_flag_or_a_sharpness_below_the_floor"]),
    ("G13 non-evaluable windows are counted in the user table", AN, "        if not r[\"oracle\"][\"evaluable\"]:\n            continue\n        u = by.setdefault", "        u = by.setdefault", [T + "test_user_table_counts_only_evaluable_windows_and_production_needs_a_usable_fit", T + "test_summarize_separates_strata_and_leaves_non_evaluable_windows_out_of_every_fraction"]),
    ("G14 an unusable production fit counts as passing", AN, "u[\"p\"] += bool(r[\"prod\"][\"usable\"] and passes(r[\"prod\"][\"sep\"]))", "u[\"p\"] += bool(passes(r[\"prod\"][\"sep\"]) or not r[\"prod\"][\"usable\"])", [T + "test_user_table_counts_only_evaluable_windows_and_production_needs_a_usable_fit"]),
    ("G15 the gate share ignores the flag", AN, "u[\"q\"] += bool(r[\"prod\"][\"usable\"] and passes(r[\"prod\"][\"sep\"]) and not r[\"prod\"][\"flag\"])", "u[\"q\"] += bool(r[\"prod\"][\"usable\"] and passes(r[\"prod\"][\"sep\"]))", [T + "test_the_gate_share_excludes_flagged_fits_and_the_separation_share_does_not"]),
    ("G16 the bootstrap resamples pooled rows instead of users", AN, "    vals = []\n    for _ in range(B):\n        idx = rng.integers(0, len(users), len(users))\n        v = stat([users[i] for i in idx])",
     "    vals = []\n    pool = [(float(k < u[\"o\"]), float(k < u.get(\"p\", 0))) for u in users for k in range(u[\"n\"])]\n    for _ in range(B):\n        idx = rng.integers(0, len(pool), len(pool))\n        v = stat([dict(user=0, n=len(idx), o=sum(pool[i][0] for i in idx), p=sum(pool[i][1] for i in idx))])",
     [T + "test_user_bootstrap_resamples_users_not_rows", T + "test_the_row_level_bootstrap_would_be_much_narrower_than_the_user_level_one"]),
    ("G17 the bootstrap is not seeded", AN, "rng = np.random.default_rng(seed)\n    vals = []", "rng = np.random.default_rng()\n    vals = []", [T + "test_boot_users_is_seeded_and_the_default_seed_is_the_declared_one"]),
    ("G18 the paired gap uses independent resamples", AN, "gap = boot_users(users, lambda u: ratio(u, \"o\") - ratio(u, \"p\"))", "gap = (ratio(users, \"o\") - ratio(users, \"p\"), -0.5, 0.5)", [T + "test_summarize_pairs_the_gap_inside_each_resample"]),
    ("G19 the not-servable level is 0.60", AN, "NOT_SERVABLE_LEVEL, GAP_LEVEL = 0.50, 0.20", "NOT_SERVABLE_LEVEL, GAP_LEVEL = 0.60, 0.20", [T + "test_verdict_hard", T + "test_the_levels_are_the_declared_ones"]),
    ("G20 the gap level is 0.10", AN, "NOT_SERVABLE_LEVEL, GAP_LEVEL = 0.50, 0.20", "NOT_SERVABLE_LEVEL, GAP_LEVEL = 0.50, 0.10", [T + "test_verdict_hard", T + "test_the_levels_are_the_declared_ones"]),
    ("G21 the lower end of the gap interval is not required", AN, "if gap >= GAP_LEVEL and gap_lo > 0:", "if gap >= GAP_LEVEL:", [T + "test_verdict_hard"]),
    ("G22 the lower end may be zero", AN, "if gap >= GAP_LEVEL and gap_lo > 0:", "if gap >= GAP_LEVEL and gap_lo >= 0:", [T + "test_verdict_hard"]),
    ("G23 'demonstrated' needs only the point estimate", AN, "if pO < NOT_SERVABLE_LEVEL:\n        return \"NOT SERVABLE BY DESIGN (demonstrated: the upper end of the interval is below 0.50)\" if pO_hi < NOT_SERVABLE_LEVEL else", "if pO < NOT_SERVABLE_LEVEL:\n        return \"NOT SERVABLE BY DESIGN (demonstrated: the upper end of the interval is below 0.50)\" if True else", [T + "test_verdict_hard"]),
    ("G24 the not-servable rule is inclusive", AN, "if pO < NOT_SERVABLE_LEVEL:", "if pO <= NOT_SERVABLE_LEVEL:", [T + "test_verdict_hard"]),
    ("G25 the Wilson interval uses z = 1.645", AN, "def wilson(k: int, n: int, z: float = 1.959964)", "def wilson(k: int, n: int, z: float = 1.645)", [T + "test_wilson_interval"]),
    ("G26 the median and the quartiles are swapped in the report format", AN, "return f\"{t[1]:.2f} [{t[0]:.2f}, {t[2]:.2f}]\"", "return f\"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]\"", [T + "test_quartile_and_interval_formatting_put_the_point_first"]),
    ("G27 a failed fit is treated as evaluable and passing", AN, "oracle=dict(evaluable=False, counts=None, sep_emp=float(\"nan\"), sep_gen=float(\"nan\")), prod=production_row(None), failed=True", "oracle=dict(evaluable=True, counts=None, sep_emp=9.0, sep_gen=9.0), prod=production_row(None), failed=True", [T + "test_refit_rows_marks_a_failed_fit_as_non_evaluable_and_unusable"]),
]
if __name__ == "__main__":
    sys.exit(R.run(MUT, sys.argv[1] if len(sys.argv) > 1 else None, workers=2))
