"""Mutation check for A7: the analysis (`a7_analyze.py`), the run helpers and driver (`a7_run.py`) and the independent publish check / summary of the WP9 runner (`wp9/scripts/b6_run.py`) against
tests/test_wp10a_a7_analyze.py. (The 25 mutants of `recertify.py` itself are `wp9/scripts/mutation_b6.py` and are rerun separately.) usage: PYTHONPATH=. python3 integration_report/wp10a/scripts/mutation_wp10a_a7.py [out.txt]"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp8" / "scripts"))
import mutation_runner as R

_base = shutil.ignore_patterns(".git", "__pycache__", "wp7r_evidence", "wp8_evidence", "wp9_evidence", "wp10a_evidence", ".pytest_cache")
R.IGNORE = lambda d, names: set(_base(d, names))

AN = "integration_report/wp10a/scripts/a7_analyze.py"
RN = "integration_report/wp10a/scripts/a7_run.py"
B6 = "integration_report/wp9/scripts/b6_run.py"
MUT = [
    ("AM01 the target is 11", AN, "TARGET, MIN_WITHHELD = 12, 8", "TARGET, MIN_WITHHELD = 11, 8"),
    ("AM02 the non-vacuity level is 7", AN, "TARGET, MIN_WITHHELD = 12, 8", "TARGET, MIN_WITHHELD = 12, 7"),
    ("AM03 the target needs one more", AN, "        if k >= target:", "        if k > target:"),
    ("AM04 the user that reaches the target is also extra", AN, "return used, recs[i + 1:]", "return used, recs[i:]"),
    ("AM05 the users are not sorted by seed", AN, 'recs = sorted(recs, key=lambda r: r["user"])', "recs = list(recs)"),
    ("AM06 a day-30 publication needs no independent agreement", AN, 'if not d30["withheld"] and not d30["independent_publishable"]:', "if False:"),
    ("AM07 a relaxed record is only a violation when True", AN, 'if e.get("rules_relaxed") is not False:', 'if e.get("rules_relaxed") is True:'),
    ("AM08 a published state without a result is accepted", AN, 'if not e.get("result_returned"):', "if False:"),
    ("AM09 a missing result_withheld is accepted", AN, 'if e.get("result_withheld") is not False:', 'if e.get("result_withheld") is True:'),
    ("AM10 a result with a withheld state is accepted", AN, 'elif e.get("result_returned"):', "elif False:"),
    ("AM11 the probes are also required of published users", AN, '    if d30["withheld"]:\n        probes =', '    if True:\n        probes ='),
    ("AM12 a probe is accepted whatever its state", AN, 'elif p["state"] != "NOT_ELIGIBLE" or p["result_returned"] or p["rules_relaxed"] is not False:', 'elif p["result_returned"] or p["rules_relaxed"] is not False:'),
    ("AM13 a probe may return a result", AN, 'elif p["state"] != "NOT_ELIGIBLE" or p["result_returned"] or p["rules_relaxed"] is not False:', 'elif p["state"] != "NOT_ELIGIBLE" or p["rules_relaxed"] is not False:'),
    ("AM14 the fresh fit must disagree", AN, 'if "f1_withheld" in e and bool(e["f1_withheld"]) != (e["state"] != "PUBLISHABLE"):', 'if "f1_withheld" in e and bool(e["f1_withheld"]) == (e["state"] != "PUBLISHABLE"):'),
    ("AM15 a missing alteration passes the control", AN, 'if c["altered"].get(n) is not False:', 'if c["altered"].get(n):'),
    ("AM16 the unaltered snapshot may be refused", AN, 'bad = [] if c["unaltered"] else ["the unaltered snapshot is refused"]', "bad = []"),
    ("AM17 day 90 needs both states", AN, 'return s60 == "PUBLISHABLE" or r["recert"].get("90", {}).get("state") in ("PUBLISHABLE", "PUBLISHABLE_EARLIER")', 'return s60 == "PUBLISHABLE" and r["recert"].get("90", {}).get("state") in ("PUBLISHABLE", "PUBLISHABLE_EARLIER")'),
    ("AM18 H1 needs more than eight", AN, "met=n_w >= MIN_WITHHELD", "met=n_w > MIN_WITHHELD"),
    ("AM19 H2 allows one violation", AN, "met=n_viol == 0", "met=n_viol <= 1"),
    ("AM20 H3 is met when nothing was checked", AN, "met=not f1 and n_f1 > 0", "met=not f1"),
    ("AM21 H4 is met without a control", AN, "met=n_ctl > 0 and not ctl_bad", "met=not ctl_bad"),
    ("AM22 the day-90 rate counts day 60", AN, 'p60, p90 = sum(1 for r in w if published_by(r, "60")), sum(1 for r in w if published_by(r, "90"))', 'p60, p90 = sum(1 for r in w if published_by(r, "60")), sum(1 for r in w if published_by(r, "60"))'),
    ("AM23 the reasons of the still withheld are those that published", AN, 'still = [x for r in w if not published_by(r, "90") for x in', 'still = [x for r in w if published_by(r, "90") for x in'),
    ("AM24 failed users are kept", AN, 'return [r for r in recs if r.get("ok")], sum(1 for r in recs if not r.get("ok"))', 'return recs, sum(1 for r in recs if not r.get("ok"))'),
    ("AM25 the non-vacuity verdict text is inverted", AN, '("MET" if lv["H1"]["met"] else "NOT MET: too few to read")', '("NOT MET: too few to read" if lv["H1"]["met"] else "MET")'),
    ("AM26 the first probe name changes", AN, 'PROBES = ("n_initializations_3",', 'PROBES = ("n_init_3",'),
    ("RM01 the chunk is 8", RN, "CHUNK, TARGET = 6, 12", "CHUNK, TARGET = 8, 12"),
    ("RM02 the run target is 10", RN, "CHUNK, TARGET = 6, 12", "CHUNK, TARGET = 6, 10"),
    ("RM03 the W1 seeds start at 10600", RN, "list(range(10601, 10649))", "list(range(10600, 10649))"),
    ("RM04 W1 forces K = 4", RN, 'candidates=(5,), sep=', "candidates=(4,), sep="),
    ("RM05 the W1 strata are swapped", RN, "sep=lambda u: (0.5, 1.0) if u % 2 == 0 else (1.0, 2.0)", "sep=lambda u: (0.5, 1.0) if u % 2 == 1 else (1.0, 2.0)"),
    ("RM06 the W2 separation is 0.3 to 0.7", RN, "sep=lambda u: (0.3, 0.6)", "sep=lambda u: (0.3, 0.7)"),
    ("RM07 a failed user counts as withheld", RN, 'return sum(1 for r in recs if r.get("ok") and r["day30"]["withheld"])', 'return sum(1 for r in recs if r["day30"]["withheld"])'),
    ("RM08 the stopping rule needs one more", RN, "return withheld_count(recs) >= target", "return withheld_count(recs) > target"),
    ("RM09 the collapsed alteration sets nothing", RN, "        s.fit_collapsed = True", "        s.fit_collapsed = s.fit_collapsed"),
    ("RM10 the fallback alteration sets zero", RN, "        s.states_pass_fallbacks = 1", "        s.states_pass_fallbacks = 0"),
    ("RM11 the non-converged alteration keeps the state", RN, "        s.fit_state = FitState.MAX_ITER_REACHED", "        s.fit_state = s.fit_state"),
    ("RM12 the snapshot shares the result's objects", RN, "{k: copy.copy(getattr(result, k, None)) for k in CHECKED_ATTRS}", "{k: getattr(result, k, None) for k in CHECKED_ATTRS}"),
    ("RM13 the n_initializations probe is the production value", RN, "dict(n_initializations=3)", "dict(n_initializations=10)"),
    ("RM14 the probes are not recorded", RN, '                    if rec["probes"] is None:', "                    if False:"),
    ("RM15 a published user is recertified", RN, '                if cur == "STILL_WITHHELD":', "                if True:"),
    ("RM16 the control is not taken from a recertified result", RN, "                        if snap is None:\n                            snap = snapshot(out[\"result\"])", "                        pass"),
    ("RM17 the cold-start gate counts 29 days", RN, "while int((~np.isnan(st.X_obs[:end0]).all(axis=1)).sum()) < 30:", "while int((~np.isnan(st.X_obs[:end0]).all(axis=1)).sum()) < 29:"),
    ("RM18 the forced candidates are not passed", RN, '            if ARMS[arm]["candidates"]:', "            if False:"),
    ("RM19 the fit seed is zero", RN, 'kw = dict(B6.SETTINGS, random_seed=user, user_id=f"a7-{user}"', 'kw = dict(B6.SETTINGS, random_seed=0, user_id=f"a7-{user}"'),
    ("RM20 the stopping rule is checked after the chunk is queued", RN, "            if should_stop(recs):\n                print(", "            if False:\n                print("),
    ("RM21 finished users are run again", RN, "todo = [s for s in ARMS[a.arm][\"seeds\"] if s not in done]", "todo = [s for s in ARMS[a.arm][\"seeds\"]]"),
    ("RM22 the fresh fit uses the new rows only", RN, "f1 = fit_hssm(matrix=np.vstack([stored, new]), timestamps=np.concatenate([ts, new_ts]), **kw)", "f1 = fit_hssm(matrix=new, timestamps=new_ts, **kw)"),
    ("RM23 the stored rows are not extended", RN, "                stored, ts = np.vstack([stored, new]), np.concatenate([ts, new_ts])\n            rec[\"snapshot_control\"]", "                pass\n            rec[\"snapshot_control\"]"),
    ("IM01 the check ignores a collapsed regime", B6, "not r.fit_collapsed and not r.fit_duration_degenerate", "True and not r.fit_duration_degenerate"),
    ("IM02 the check ignores a degenerate duration", B6, "not r.fit_duration_degenerate and not r.fit_degraded", "True and not r.fit_degraded"),
    ("IM03 the check ignores a degraded fit", B6, "not r.fit_degraded and r.states_pass_fallbacks == 0", "True and r.states_pass_fallbacks == 0"),
    ("IM04 the check ignores the state-pass fallbacks", B6, "r.states_pass_fallbacks == 0 and r.late_fallbacks == 0", "True and r.late_fallbacks == 0"),
    ("IM05 the check ignores the late fallbacks", B6, "and r.late_fallbacks == 0", "and True"),
    ("IM06 the check accepts a numerical failure", B6, "and underlying != FitState.NUMERICAL_FAILURE", "and True"),
    ("IM07 the check ignores the underlying state of a contested fit", B6, "    if underlying == FitState.MODEL_SELECTION_CONTESTED:", "    if False:"),
    ("IM08 a contested fit without a recorded state is not converged", B6, '.get("fit_state", FitState.CONVERGED)', '.get("fit_state", FitState.MAX_ITER_REACHED)'),
    ("IM09 the summary inverts the withheld flag", B6, "return dict(withheld=bool(r.insights_withheld),", "return dict(withheld=not bool(r.insights_withheld),"),
]
if __name__ == "__main__":
    sys.exit(R.run(MUT, ["tests/test_wp10a_a7_analyze.py", "-q", "-x"], sys.argv[1] if len(sys.argv) > 1 else None, workers=1))
