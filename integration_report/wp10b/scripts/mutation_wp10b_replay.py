"""WP10B replay script: mutation control of `a3_safety_replay.py` against `tests/test_wp10b_replay.py` (set declared in tests/PREDECLARED_CRITERIA_WP10B.md section 14.5; named test must fail by its own assertion; control first).
usage: PYTHONPATH=. python3.12 integration_report/wp10b/scripts/mutation_wp10b_replay.py [out.txt]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shutil

import mutation_runner10b as R

# the replay tests read the STORED WP10A fits (wp10a_evidence/a3_eval_fits.jsonl, 16 MB): the isolated copy of the repository must contain them (the runner's default ignore list drops that directory)
R._BASE_IGNORE = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "wp7r_evidence", "wp8_evidence", "wp9_evidence", "wp10b_evidence")

F = "integration_report/wp10b/scripts/a3_safety_replay.py"
T = "tests/test_wp10b_replay.py::"
MUT = [
    ("R01 a wrong alignment is treated as safe", F, "    if w > 0:\n        out.append(f\"{tag}SAFETY FAILURE", "    if w > 10**9:\n        out.append(f\"{tag}SAFETY FAILURE", [T + "test_the_verdict_words"]),
    ("R02 the bound threshold is inverted", F, "'at most' if bound <= SAFETY_BOUND else 'ABOVE'", "'at most' if bound > SAFETY_BOUND else 'ABOVE'", [T + "test_the_verdict_words"]),
    ("R03 the exact bound formula is wrong", F, "return 1.0 - (1.0 - conf) ** (1.0 / n)", "return 1.0 - (1.0 - conf) ** (1.0 / (n + 1))", [T + "test_the_exact_bound"]),
    ("R04 the restart states are not compared", F, '        if a["restarts"] != b["restarts"]:\n            diffs.append("restart_states")', '        if False:\n            diffs.append("restart_states")', [T + "test_a_fit_is_identical_to_itself_and_every_discrete_field_is_compared"]),
    ("R05 the selected restart is not compared", F, '        if a["selected"] != b["selected"]:\n            diffs.append("selected_restart")', '        if False:\n            diffs.append("selected_restart")', [T + "test_a_fit_is_identical_to_itself_and_every_discrete_field_is_compared"]),
    ("R06 a failed fit against a good one is not a difference", F, '    if a.get("ok") != b.get("ok"):\n        diffs.append("ok")', '    if False:\n        diffs.append("ok")', [T + "test_a_planted_failed_fit_is_a_missing_step_and_a_difference"]),
    ("R07 the iteration count is not part of the discrete view", F, "[(r.get(\"state\"), bool(r.get(\"collapsed\")), r.get(\"n_iter\")) for r in rs]", "[(r.get(\"state\"), bool(r.get(\"collapsed\"))) for r in rs]", [T + "test_a_fit_is_identical_to_itself_and_every_discrete_field_is_compared"]),
    ("R08 the users are not round-robin over the strata", F, "    return [(first + i, st) for i in range(per_stratum) for st, first in STRATUM_USERS.items()]", "    return [(first + i, st) for st, first in STRATUM_USERS.items() for i in range(per_stratum)]", [T + "test_the_user_list_is_36_users_round_robin_over_the_strata"]),
    ("R09 the users-per-stratum bound is removed", F, "    if not 1 <= per_stratum <= 20:", "    if False:", [T + "test_users_per_stratum_is_bounded"]),
    ("R10 the operating-point guard is removed", F, "    if point != OPERATING_POINT:", "    if False:", [T + "test_the_operating_point_guard"]),
    ("R11 the first user of a stratum is shifted", F, 'STRATUM_USERS = {"hard": 10151, "medium": 10171, "easy": 10191}', 'STRATUM_USERS = {"hard": 10152, "medium": 10171, "easy": 10191}', [T + "test_the_user_list_is_36_users_round_robin_over_the_strata"]),
    ("R12 the totals do not add the wrong count", F, 't["wrong_unflagged"] += d["wrong_unflagged"]; ', "", [T + "test_the_totals_add_up"]),
    ("R13 the job carries the seed-B fit seed", F, "return dict(user=user, start=0, end=t, fitseed=user, K=3,", "return dict(user=user, start=0, end=t, fitseed=user + 1000, K=3,", [T + "test_the_job_is_the_wp10a_a3_job_so_the_stored_fits_are_found"]),
    ("R14 a missing step is not a state difference", F, " + abs(len(rows) - len(srows))", "", [T + "test_a_planted_failed_fit_is_a_missing_step_and_a_difference"]),
    ("R15 a resumed run refits finished fits", F, "        rec = done_fits.get(key)\n        if rec is None:", "        rec = None\n        if rec is None:", [T + "test_the_run_writes_one_line_per_fit_and_per_user_and_resumes_without_refitting"]),
    ("R16 the log-likelihood difference is absolute", F, "worst = max(worst, abs(lx - ly) / max(abs(lx), abs(ly), 1e-300))", "worst = max(worst, abs(lx - ly))", [T + "test_the_log_likelihood_difference_is_relative_and_never_a_verdict"]),
    ("R17 the replay reuses the cached new fit for the stored replay", F, "    R._prep_cache.clear()\n", "", [T + "test_the_replay_of_the_stored_fits_does_not_reuse_the_new_fit_from_the_cache"]),
    ("R18 the state comparison ignores correctness and the served flag", F, '(a["state"], a["served"], a["correct"]) != (b["state"], b["served"], b["correct"])', '(a["state"],) != (b["state"],)', [T + "test_a_planted_wrong_alignment_is_counted_as_wrong_and_unflagged_and_as_a_state_difference"]),
    ("R19 a partial run is not marked", F, '    tag = "" if complete else "PARTIAL (not every user is done): "', '    tag = ""', [T + "test_the_verdict_words"]),
    ("R20 a finished fit is not written to the fits file", F, "                fh.write(json.dumps(rec) + \"\\n\"); fh.flush(); os.fsync(fh.fileno())\n        recs.append(rec)", "                pass\n        recs.append(rec)", [T + "test_the_run_writes_one_line_per_fit_and_per_user_and_resumes_without_refitting"]),
]

if __name__ == "__main__":
    sys.exit(R.run(MUT, sys.argv[1] if len(sys.argv) > 1 else None, workers=2))
