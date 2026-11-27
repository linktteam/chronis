"""WP10B B3 / B4: mutation control of the analysis (`b3_analyze.py`) and of the runner's pure parts (`b3_run.py`) against `tests/test_wp10b_b3_analyze.py` and `tests/test_wp10b_b3_run.py` (named test must fail by its own
assertion; control first; runner `mutation_runner10b.py`). usage: PYTHONPATH=. python3.12 integration_report/wp10b/scripts/mutation_wp10b_b3_analyze.py [out.txt]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutation_runner10b as R

F = "integration_report/wp10b/scripts/b3_analyze.py"
G = "integration_report/wp10b/scripts/b3_run.py"
T = "tests/test_wp10b_b3_analyze.py::"
U = "tests/test_wp10b_b3_run.py::"
MUT = [
    ("Y1 harmful means a loss of the strict kind", F, 'harmful=bool(mean < 0), good=bool(mean > 0)', 'harmful=bool(mean < -0.1), good=bool(mean > 0)', [T + "test_truth_is_the_point_estimate_and_the_strict_version_uses_1_645_se"]),
    ("Y2 good means a gain of the strict kind", F, 'good=bool(mean > 0), strict_good', 'good=bool(mean > 0.1), strict_good', [T + "test_truth_is_the_point_estimate_and_the_strict_version_uses_1_645_se"]),
    ("Y3 the strict margin is 1 se", F, "Z_STRICT = 1.645", "Z_STRICT = 1.0", [T + "test_truth_is_the_point_estimate_and_the_strict_version_uses_1_645_se"]),
    ("Y4 the strict harmful side has the wrong sign", F, "strict_harmful=bool(ok and mean + Z_STRICT * se < 0)", "strict_harmful=bool(ok and mean - Z_STRICT * se < 0)", [T + "test_truth_is_the_point_estimate_and_the_strict_version_uses_1_645_se"]),
    ("Y5 two days are enough for a truth", F, "if len(d) < 3 or not np.all(np.isfinite(d)):\n        return None\n    mean, se = float(d.mean()), C.hac_se(d)", "if len(d) < 2 or not np.all(np.isfinite(d)):\n        return None\n    mean, se = float(d.mean()), C.hac_se(d)", [T + "test_truth_is_none_for_too_few_or_non_finite_days"]),
    ("Y6 the exact bound is the Wald bound", F, "return 1.0 if c >= n else float(beta.ppf(conf, c + 1, n - c))", "return 1.0 if c >= n else float(min(1.0, c / n + 1.645 * math.sqrt(max(c, 1) / n * (1 - c / n) / n)))", [T + "test_exact_upper_bound"]),
    ("Y7 the bound is two-sided 95%", F, "def exact_upper(c: int, n: int, conf: float = 0.95)", "def exact_upper(c: int, n: int, conf: float = 0.975)", [T + "test_exact_upper_bound"]),
    ("Y8 the Wilson interval has no continuity term", F, "centre = (p + z * z / (2 * n)) / den", "centre = p / den", [T + "test_wilson_interval"]),
    ("Y9 a violation with many fits is still met", F, "    if c >= 1:\n        return \"NOT MET\"", "    if c >= 3:\n        return \"NOT MET\"", [T + "test_refusal_verdict_words"]),
    ("Y10 thirteen harmful fits are enough", F, "N_HARM_MIN = 14 ", "N_HARM_MIN = 13 ", [T + "test_refusal_verdict_words"]),
    ("Y11 the useful-share level is 40%", F, "SHARE_LEVEL = 0.50 ", "SHARE_LEVEL = 0.40 ", [T + "test_share_verdict_words"]),
    ("Y12 the share verdict ignores the sample size", F, "    if n < N_GOOD_MIN:\n        return \"NOT DEMONSTRATED\"\n    lo, _ = wilson(k, n)", "    lo, _ = wilson(k, n)", [T + "test_share_verdict_words"]),
    ("Y13 MET is judged on the point estimate", F, "    if lo >= level:\n        return \"MET\"", "    if k / n >= level:\n        return \"MET\"", [T + "test_share_verdict_words"]),
    ("Y14 ineligible fits are counted as asked", F, 'rows = [(r, truth(r)) for r in recs if r.get("ok") and r.get("eligible", True)]\n    h = ', 'rows = [(r, truth(r)) for r in recs if r.get("ok")]\n    h = ', [T + "test_refusal_block_counts_only_harmful_asked_fits_and_the_certified_among_them"]),
    ("Y15 certified is counted over every fit, not the harmful ones", F, "c = sum(is_certified(r) for r, _ in h)\n    hs", "c = sum(is_certified(r) for r, _ in rows)\n    hs", [T + "test_refusal_block_counts_only_harmful_asked_fits_and_the_certified_among_them"]),
    ("Y16 the good share counts harmful fits", F, "g = [(r, t) for r, t in rows if t and t[\"good\"]]", "g = [(r, t) for r, t in rows if t]", [T + "test_share_block_counts_good_asked_fits"]),
    ("Y17 the blind count includes ineligible fits", F, 'if r.get("ok") and r.get("eligible") and (truth(r) or {}).get("harmful"))', 'if r.get("ok") and (truth(r) or {}).get("harmful"))', [T + "test_the_extension_rule_reads_harm_without_the_certificate"]),
    ("Y18 the blind count reads the certificate", F, 'if r.get("ok") and r.get("eligible") and (truth(r) or {}).get("harmful"))', 'if r.get("ok") and r.get("eligible") and (truth(r) or {}).get("harmful") and not is_certified(r))', [T + "test_the_extension_rule_reads_harm_without_the_certificate"]),
    ("Y19 the extension is wanted at fourteen", F, "return blind_harmful_count(recs) < N_HARM_MIN", "return blind_harmful_count(recs) <= N_HARM_MIN", [T + "test_the_extension_rule_reads_harm_without_the_certificate"]),
    ("Y20 the paired margin uses two standard errors", F, "Z_B4 = 3.0", "Z_B4 = 2.0", [T + "test_paired_users_margin_is_across_users_at_three_standard_errors"]),
    ("Y21 the paired standard error uses ddof 0", F, "se = float(v.std(ddof=1) / math.sqrt(len(v)))", "se = float(v.std(ddof=0) / math.sqrt(len(v)))", [T + "test_paired_users_margin_is_across_users_at_three_standard_errors"]),
    ("Y22 a failed fit counts as published", F, 'return bool(rec.get("ok") and (d.get("publishable") or is_certified(rec)))', 'return bool(d.get("publishable") or is_certified(rec))', [T + "test_published_is_converged_publishable_or_certified"]),
    ("Y23 only certified fits count as published", F, 'return bool(rec.get("ok") and (d.get("publishable") or is_certified(rec)))', 'return bool(rec.get("ok") and is_certified(rec))', [T + "test_published_is_converged_publishable_or_certified"]),
    ("Y24 the publish rule ignores the baseline margin", F, "rule_met=bool(beats and verdict in (\"MET\", \"MET on the point estimate\"))", "rule_met=bool(verdict in (\"MET\", \"MET on the point estimate\"))", [T + "test_b4_rule_needs_both_the_baseline_margin_and_the_published_share"]),
    ("Y25 the publish rule ignores the published share", F, "rule_met=bool(beats and verdict in (\"MET\", \"MET on the point estimate\"))", "rule_met=bool(beats)", [T + "test_b4_rule_needs_both_the_baseline_margin_and_the_published_share"]),
    ("Y26 the strict rule accepts the point estimate", F, "rule_met_strict=bool(beats and verdict == \"MET\")", "rule_met_strict=bool(beats and verdict in (\"MET\", \"MET on the point estimate\"))", [T + "test_the_point_estimate_verdict_of_the_published_share"]),
    ("Y27 the published level is 50%", F, "PUBLISHED_LEVEL = 0.60 ", "PUBLISHED_LEVEL = 0.50 ", [T + "test_the_point_estimate_verdict_of_the_published_share"]),
    ("Y28 hard users are in the B4 set", F, 'em = [r for r in recs if r["job"]["stratum"] in ("easy", "medium")]', 'em = [r for r in recs]', [T + "test_b4_uses_easy_and_medium_users_only"]),
    ("Y29 the baseline margin counts every certified fit", F, 'both = [r for r in cert if ((r["certificate"].get("baseline") or {}).get("beats_baseline"))]', 'both = list(cert)', [T + "test_stratum_table_and_baseline_margin"]),
    ("Y30 the breaker fits are not marked as asked", F, '            r["eligible"] = True                                                      # the certificate was forced on every breaker fit', '            pass', [T + "test_analyse_marks_breaker_fits_as_asked_and_reads_the_real_engine_as_a_share"]),
    ("Y31 the ranges overlap (eval users start at the ext range)", G, '"ext": list(range(11457, 11481))', '"ext": list(range(11456, 11480))', [U + "test_the_ranges_do_not_overlap_and_stay_inside_the_reserved_band"]),
    ("Y32 a stratum block of 12 is 10", G, "return STRATA[(user - 11421) // 12]", "return STRATA[(user - 11421) // 10]", [U + "test_strata_follow_the_declared_blocks"]),
    ("Y33 the guard accepts a file without the B3 part", G, 'if a.range != "smoke" and (not CRITERIA.exists() or CRITERIA_MARK not in CRITERIA.read_text()):', 'if a.range != "smoke" and (not CRITERIA.exists()):', [U + "test_a_scored_range_is_refused_before_the_dated_criteria_exist"]),
    ("Y34 the driver reruns finished jobs", G, "    todo = [j for j in jobs if job_key(j) not in done]", "    todo = list(jobs)", [U + "test_the_driver_is_resumable_and_writes_one_line_per_job"]),
    ("Y35 the breaker refit is the real engine", G, "cls = HP.ENGINE_CLASSES[engine]", "cls = HP.ENGINE_CLASSES['real']", [U + "test_a_wrong_engine_job_is_fitted_refitted_and_refused"]),
    ("Y36 the certificate is not forced on the breakers", G, 'el = dict(apply_to_any_state=True, em_state=rec["fit_state"], other_reasons=[])', 'el = dict(apply_to_any_state=False, em_state=rec["fit_state"], other_reasons=[])', [U + "test_a_wrong_engine_job_is_fitted_refitted_and_refused"]),
]

if __name__ == "__main__":
    sys.exit(R.run(MUT, sys.argv[1] if len(sys.argv) > 1 else None, workers=2))
