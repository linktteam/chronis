"""WP10B replay of the A3 safety demonstration on a subset (`integration_report/wp10b/scripts/a3_safety_replay.py`; criteria section 14). No new fit is made here: the end-to-end tests feed the STORED WP10A fits in as if they were new."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "integration_report" / "wp10b" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import a3_safety_replay as RP  # noqa: E402

OP = RP.OPERATING_POINT


@pytest.fixture(scope="module")
def stored():
    return RP.load_stored(RP.STORED_DEFAULT)


def chain_of(stored, user):
    return [stored[(user, t)] for t in RP.G_T]


def swap01(s):
    return s.translate(str.maketrans("01", "10"))


# ----------------------------------------------------------------------------------------------------------- users, jobs, operating point
def test_the_user_list_is_36_users_round_robin_over_the_strata():
    us = RP.users_for(12)
    assert len(us) == 36 and len({u for u, _ in us}) == 36
    assert [st for _, st in us[:6]] == ["hard", "medium", "easy", "hard", "medium", "easy"]
    assert [u for u, _ in us[:3]] == [10151, 10171, 10191] and us[-3:] == [(10162, "hard"), (10182, "medium"), (10202, "easy")]
    assert len(RP.users_for(1)) == 3 and len(RP.users_for(20)) == 60


@pytest.mark.parametrize("n", [0, -1, 21])
def test_users_per_stratum_is_bounded(n):
    with pytest.raises(ValueError):
        RP.users_for(n)


def test_the_job_is_the_wp10a_a3_job_so_the_stored_fits_are_found(stored):
    spec = importlib.util.spec_from_file_location("wp10a_a3_run_for_test", ROOT / "integration_report" / "wp10a" / "scripts" / "a3_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for user, st in RP.users_for(2):
        for t in RP.G_T:
            assert RP.job_for(user, st, t) == mod._job(user, t, user, st)
            assert (user, t) in stored and stored[(user, t)]["job"] == RP.job_for(user, st, t)
    assert RP.G_T == mod.G_T and RP.G_T == [90, 97, 104, 111, 118, 125, 132, 139, 146, 150]
    assert all(RP.job_for(10151, "hard", t)["fitseed"] == 10151 for t in RP.G_T)


def test_the_operating_point_guard():
    assert RP.check_operating_point((ROOT / "tests" / "DESIGN_FREEZE_WP10A_A3.txt").read_text()) == (2.5, True, 0.95)
    for bad in ('{"h": 2.0, "u": true, "s": 0.95}', '{"h": 2.5, "u": false, "s": 0.95}', '{"h": 2.5, "u": true, "s": 0.9}'):
        with pytest.raises(SystemExit):
            RP.check_operating_point(bad + "\nrest")


# ----------------------------------------------------------------------------------------------------------- the bound
def test_the_exact_bound():
    assert RP.exact_upper(0, 299) <= 0.01 < RP.exact_upper(0, 298)
    assert abs(RP.exact_upper(0, 324) - (1 - 0.05 ** (1 / 324))) < 1e-15 and 0.0091 < RP.exact_upper(0, 324) < 0.0093
    assert RP.exact_upper(1, 473) <= 0.01 < RP.exact_upper(1, 472)
    assert RP.exact_upper(0, 0) != RP.exact_upper(0, 0)                                            # NaN
    assert RP.exact_upper(5, 5) == 1.0


# ----------------------------------------------------------------------------------------------------------- the comparison of fits
def test_a_fit_is_identical_to_itself_and_every_discrete_field_is_compared(stored):
    base = stored[(10191, 150)]
    c = RP.compare_fit(base, base)
    assert c["compared"] and c["identical"] is True and c["diffs"] == [] and c["max_rel_ll"] == 0.0
    for field, change in (("state", lambda r: r.__setitem__("state", "MAX_ITER_REACHED")), ("collapsed", lambda r: r.__setitem__("collapsed", not r["collapsed"])), ("n_iter", lambda r: r.__setitem__("n_iter", r["n_iter"] + 1))):
        new = copy.deepcopy(base)
        change(new["restarts"][3])
        c = RP.compare_fit(new, base)
        assert c["identical"] is False and c["diffs"] == ["restart_states"], field
    new = copy.deepcopy(base)
    sel_i = RP.discrete_view(base)["selected"][0]
    new["restarts"][sel_i]["labels"] = swap01(new["restarts"][sel_i]["labels"])
    c = RP.compare_fit(new, base)
    assert c["identical"] is False and "selected_restart" in c["diffs"]
    failed = dict(job=base["job"], ok=False, error="x")
    assert RP.compare_fit(failed, base)["diffs"] == ["ok"] and RP.compare_fit(base, failed)["diffs"] == ["ok"]
    assert RP.compare_fit(failed, failed)["identical"] is True


def test_a_fit_without_a_stored_counterpart_is_not_compared(stored):
    assert RP.compare_fit(stored[(10191, 150)], None) == dict(compared=False, identical=None, diffs=[], max_rel_ll=None)


def test_the_log_likelihood_difference_is_relative_and_never_a_verdict(stored):
    base = copy.deepcopy(stored[(10191, 150)])
    new = copy.deepcopy(base)
    for r in new["restarts"]:
        r["ll"] = r["ll"] * (1 + 3e-7)
    c = RP.compare_fit(new, base)
    assert c["identical"] is True and 2.9e-7 < c["max_rel_ll"] < 3.1e-7                              # relative, and a float difference does not make a discrete difference
    big, small = copy.deepcopy(base), copy.deepcopy(base)
    for r in big["restarts"]:
        r["ll"] = 1e6
    for r in small["restarts"]:
        r["ll"] = 1e6 + 1.0
    assert 0.9e-6 < RP.max_rel_ll(big, small) < 1.1e-6                                              # 1 in 1e6 is 1e-6 relative, not 1.0
    assert RP.max_rel_ll(base, dict(base, restarts=base["restarts"][:5])) == 0.0


# ----------------------------------------------------------------------------------------------------------- the verdict words and the totals
def totals(**kw):
    t = dict(users=36, users_expected=36, alignments=324, served=167, wrong_unflagged=0, compared=360, identical=360, state_diffs=0, max_rel_ll=0.0, refits=324)
    t.update(kw)
    return t


def test_the_verdict_words():
    ok = RP.verdict_lines(totals(), True)
    assert ok[0].startswith("SAFETY: no wrong-and-unflagged alignment in 324") and "at most 1%" in ok[0] and "0.92%" in ok[0] and ok[1].startswith("REPRODUCTION: the discrete outcomes of all 360 fits")
    small = RP.verdict_lines(totals(alignments=250), True)
    assert small[0].startswith("SAFETY: no wrong") and "ABOVE 1%" in small[0] and "FAILURE" not in small[0]                   # a large bound is reported, not a failure
    bad = RP.verdict_lines(totals(wrong_unflagged=1), True)
    assert bad[0].startswith("SAFETY FAILURE: 1 wrong-and-unflagged alignment(s) in 324")
    diff = RP.verdict_lines(totals(identical=359), True)
    assert diff[1].startswith("REPRODUCTION FAILURE: 1 of 360 fits differ")
    sdiff = RP.verdict_lines(totals(state_diffs=2), True)
    assert sdiff[1].startswith("REPRODUCTION FAILURE") and "2 window(s)" in sdiff[1]
    part = RP.verdict_lines(totals(users=10), False)
    assert all(x.startswith("PARTIAL") for x in part)
    nostore = RP.verdict_lines(totals(compared=0, identical=0), True)
    assert "not compared" in nostore[1] and "FAILURE" not in nostore[1]


def test_the_totals_add_up():
    lines = [dict(alignments=9, served=5, wrong_unflagged=0, refits=9, compared=10, identical=10, state_diffs=0, max_rel_ll=1e-9),
             dict(alignments=8, served=3, wrong_unflagged=2, refits=9, compared=10, identical=9, state_diffs=1, max_rel_ll=5e-7)]
    t = RP.sum_users(lines, 36)
    assert (t["users"], t["alignments"], t["served"], t["wrong_unflagged"], t["refits"], t["compared"], t["identical"], t["state_diffs"]) == (2, 17, 8, 2, 18, 20, 19, 1)
    assert t["max_rel_ll"] == 5e-7 and t["users_expected"] == 36


# ----------------------------------------------------------------------------------------------------------- end to end (stored fits fed in as new)
def test_the_stored_fits_reproduce_the_reference_and_themselves(stored):
    tot = []
    for user, st in RP.users_for(1):
        d = RP.replay_user(chain_of(stored, user), stored, *OP)
        assert d["stratum"] == st and d["fits"] == 10 and d["fits_ok"] == 10 and d["compared"] == 10 and d["identical"] == 10 and d["state_diffs"] == 0 and d["diffs"] == [] and d["max_rel_ll"] == 0.0
        assert d["wrong_unflagged"] == 0 and d["alignments"] <= 9 and d["states"][0] == "NEW_LINEAGE"
        tot.append(d)
    assert [d["served"] for d in tot] == [0, 1, 9] and sum(d["alignments"] for d in tot) == 27                                       # users 10151 / 10171 / 10191 from the stored fits (RUN)
    assert tot[0]["states"][1:] == ["CONTINUITY_UNVERIFIED"] * 9 and tot[2]["states"][1:] == ["ALIGNED"] * 9 and tot[1]["states"][4] == "ALIGNED"


def test_a_planted_wrong_alignment_is_counted_as_wrong_and_unflagged_and_as_a_state_difference(stored):
    user = 10191
    planted = copy.deepcopy(chain_of(stored, user))
    for r in planted:
        if r["job"]["end"] >= 118:
            r["labels_true"] = swap01(r["labels_true"])                                             # the planted truth no longer matches what the lineage served
    d = RP.replay_user(planted, stored, *OP)
    assert d["wrong_unflagged"] == 1 and d["served"] == 9 and d["state_diffs"] == 1                  # also seen against the stored replay (a different correctness)
    assert RP.verdict_lines(RP.sum_users([d], 1), True)[0].startswith("SAFETY FAILURE")


def test_a_planted_failed_fit_is_a_missing_step_and_a_difference(stored):
    user = 10191
    planted = copy.deepcopy(chain_of(stored, user))
    planted[4] = dict(job=planted[4]["job"], ok=False, error="planted")
    d = RP.replay_user(planted, stored, *OP)
    assert d["alignments"] == 8 and d["identical"] == 9 and d["diffs"] == [(118, ["ok"])] and d["state_diffs"] == 1
    assert RP.verdict_lines(RP.sum_users([d], 1), True)[1].startswith("REPRODUCTION FAILURE")


def test_the_replay_of_the_stored_fits_does_not_reuse_the_new_fit_from_the_cache(stored):
    """`wp9_replay_lib.prepared` caches by job, not by content: a replay of the new fit followed by a replay of the stored fit of the same job would otherwise return the same thing twice."""
    user = 10191
    planted = copy.deepcopy(chain_of(stored, user))
    for r in planted:
        r["labels_true"] = swap01(r["labels_true"]) if r["job"]["end"] >= 118 else r["labels_true"]
    first = RP.replay_user(planted, stored, *OP)
    second = RP.replay_user(chain_of(stored, user), stored, *OP)
    assert first["wrong_unflagged"] == 1 and second["wrong_unflagged"] == 0 and second["state_diffs"] == 0


# ----------------------------------------------------------------------------------------------------------- the run: resume and partial results
class SerialPool:
    def __init__(self, n):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def imap_unordered(self, f, it):
        for x in it:
            yield f(x)


def test_the_run_writes_one_line_per_fit_and_per_user_and_resumes_without_refitting(stored, tmp_path, monkeypatch, capsys):
    import wp9_fit_lib as L
    calls = []

    def fake_fit(job):
        calls.append((job["user"], job["end"]))
        return stored[(job["user"], job["end"])]
    monkeypatch.setattr(L, "fit_job", fake_fit)
    monkeypatch.setattr(RP.mp, "Pool", SerialPool)
    args = type("A", (), dict(out=str(tmp_path / "o"), users_per_stratum=1, workers=1, stored=str(RP.STORED_DEFAULT)))()
    RP.cmd_run(args)
    assert len(calls) == 30 and len(set(calls)) == 30
    fits = RP.read_jsonl(tmp_path / "o" / "fits.jsonl")
    users = RP.read_jsonl(tmp_path / "o" / "users.jsonl")
    assert len(fits) == 30 and len(users) == 3 and {d["user"] for d in users} == {10151, 10171, 10191}
    assert (tmp_path / "o" / "env.json").exists() and "machine" in json.loads((tmp_path / "o" / "env.json").read_text())
    out = capsys.readouterr().out
    assert out.count("running: users") == 3 and "SAFETY: no wrong-and-unflagged alignment in" in out and "REPRODUCTION: the discrete outcomes of all 30 fits" in out
    calls.clear()
    RP.cmd_run(args)                                                                                 # a finished run starts nothing
    assert calls == [] and len(RP.read_jsonl(tmp_path / "o" / "fits.jsonl")) == 30
    # a run that was stopped after some fits of a user: those fits are not repeated
    o2 = tmp_path / "o2"
    o2.mkdir()
    with open(o2 / "fits.jsonl", "w") as fh:
        for t in RP.G_T[:4]:
            fh.write(json.dumps(stored[(10151, t)]) + "\n")
    calls.clear()
    args2 = type("A", (), dict(out=str(o2), users_per_stratum=1, workers=1, stored=str(RP.STORED_DEFAULT)))()
    RP.cmd_run(args2)
    assert len(calls) == 26 and all(not (u == 10151 and t in RP.G_T[:4]) for u, t in calls)


def test_the_summary_says_partial_until_every_user_is_done(stored, tmp_path, capsys):
    o = tmp_path / "o"
    o.mkdir()
    d = RP.replay_user(chain_of(stored, 10191), stored, *OP)
    (o / "users.jsonl").write_text(json.dumps(d) + "\n")
    RP.cmd_summary(type("A", (), dict(out=str(o), users_per_stratum=12))())
    out = capsys.readouterr().out
    assert "users 1/36" in out and "PARTIAL" in out
