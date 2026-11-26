"""WP10A A7: the run script (`a7_run.py`: stopping rule, snapshot and alterations, settings probes, the per-user flow with a stand-in for `fit_hssm`) and the analysis (`a7_analyze.py`: the stopping rule in seed
order, the four hard levels recomputed from stored fields) of the re-certification arm that is withheld by design (tests/PREDECLARED_CRITERIA_WP10A.md Part 5). No real fit is run here. Mutation check:
mutation_wp10a_a7.py."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "integration_report" / "wp10a" / "scripts"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RUN = _load("wp10a_a7_run_t", SCRIPTS / "a7_run.py")
AN = _load("wp10a_a7_analyze_t", SCRIPTS / "a7_analyze.py")


# ------------------------------------------------------------------------------------------------------------------- stand-ins
def result(withheld, K=5, **over):
    from backbone.hssm.model import FitState
    r = SimpleNamespace(insights_withheld=withheld, _withheld_reasons=lambda: (["collapsed_regime: a regime holds 1.00 effective days (minimum 3)"] if withheld else []), fit_state=FitState.CONVERGED, K=K,
                        convergence_metadata=None, fit_collapsed=withheld, fit_duration_degenerate=False, fit_degraded=False, states_pass_fallbacks=0, late_fallbacks=0)
    for k, v in over.items():
        setattr(r, k, v)
    return r


def good_snapshot():
    return RUN.snapshot(result(False))


def rec(user, withheld=True, s60="PUBLISHABLE", s90="PUBLISHABLE_EARLIER", control=True, stratum="hard", ok=True):
    """A stored per-user record as `a7_run.one` writes it (a handcrafted one for the analysis tests)."""
    r = dict(arm="W1", user=user, ok=ok, stratum=stratum, seconds=1.0, day30=dict(withheld=withheld, reasons=(["collapsed_regime: x"] if withheld else []), independent_publishable=not withheld, fit_state="CONVERGED", K=5, withheld_flag=withheld),
             recert={}, probes=None, snapshot_control=None)
    if withheld:
        r["probes"] = [dict(probe=n, state="NOT_ELIGIBLE", reasons=["x"], result_returned=False, rules_relaxed=False) for n in AN.PROBES]
        r["recert"]["60"] = dict(state=s60, reasons=([] if s60 == "PUBLISHABLE" else ["collapsed_regime: y"]), rules_relaxed=False, n_rows=60, result_returned=s60 == "PUBLISHABLE",
                                 f1_withheld=s60 != "PUBLISHABLE")
        if s60 == "PUBLISHABLE":
            r["recert"]["60"].update(result_withheld=False, independent_publishable=True)
            r["recert"]["90"] = dict(state="PUBLISHABLE_EARLIER")
        else:
            r["recert"]["90"] = dict(state=s90, reasons=([] if s90 == "PUBLISHABLE" else ["collapsed_regime: z"]), rules_relaxed=False, n_rows=90, result_returned=s90 == "PUBLISHABLE", f1_withheld=s90 != "PUBLISHABLE")
            if s90 == "PUBLISHABLE":
                r["recert"]["90"].update(result_withheld=False, independent_publishable=True)
    if control and (not withheld or s60 == "PUBLISHABLE" or s90 == "PUBLISHABLE"):
        r["snapshot_control"] = dict(unaltered=True, altered={n: False for n in AN.ALTERATIONS})
    return r


# ------------------------------------------------------------------------------------------------------------------- the run helpers
def test_the_arms_are_the_declared_ones():
    assert RUN.ARMS["W1"]["seeds"] == list(range(10601, 10649)) and RUN.ARMS["W2"]["seeds"] == list(range(10651, 10699))
    assert RUN.ARMS["W1"]["candidates"] == (5,) and RUN.ARMS["W2"]["candidates"] is None
    assert RUN.ARMS["W1"]["sep"](10602) == (0.5, 1.0) and RUN.ARMS["W1"]["sep"](10601) == (1.0, 2.0)
    assert RUN.ARMS["W1"]["stratum"](10602) == "hard" and RUN.ARMS["W1"]["stratum"](10601) == "medium"
    assert RUN.ARMS["W2"]["sep"](10651) == (0.3, 0.6) and RUN.ARMS["W2"]["stratum"](10651) == "very_hard"
    assert (RUN.CHUNK, RUN.TARGET) == (6, 12) and (AN.TARGET, AN.MIN_WITHHELD) == (12, 8)
    assert RUN.ALTERATIONS == AN.ALTERATIONS and tuple(n for n, _ in RUN.PROBES) == AN.PROBES


def test_chunks_cut_in_order_and_keep_the_remainder():
    assert RUN.chunks(range(14), 6) == [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13]] and RUN.chunks([], 6) == []


def test_the_stopping_rule_counts_only_ok_users_withheld_at_day_30_and_stops_at_the_target():
    recs = [rec(i, withheld=(i % 2 == 0)) for i in range(23)]                              # 12 even users: 0, 2, ..., 22
    assert RUN.withheld_count(recs) == 12 and RUN.should_stop(recs)
    assert RUN.withheld_count(recs[:22]) == 11 and not RUN.should_stop(recs[:22])
    assert RUN.should_stop(recs[:22], target=11) and not RUN.should_stop([], target=1)
    bad = [dict(rec(0), ok=False)] + recs[1:22]
    assert RUN.withheld_count(bad) == 10                                                   # the failed user does not count
    assert RUN.withheld_count([dict(rec(i), ok=False, day30=dict(withheld=True)) for i in range(5)]) == 0


def test_a_snapshot_is_a_copy_of_exactly_the_attributes_the_independent_check_reads():
    r = result(False, convergence_metadata={"fit_state": "CONVERGED"})
    s = RUN.snapshot(r)
    assert set(vars(s)) == set(RUN.CHECKED_ATTRS)
    s.convergence_metadata["fit_state"] = "X"
    assert r.convergence_metadata == {"fit_state": "CONVERGED"}                            # deep enough: the result is not touched
    assert RUN.snapshot(SimpleNamespace(fit_state=1)).fit_collapsed is None                # a missing attribute is None, not an error


@pytest.mark.parametrize("name", RUN.ALTERATIONS)
def test_each_alteration_changes_exactly_one_rule_and_is_refused_by_the_independent_check(name):
    s = good_snapshot()
    assert RUN.B6.independent_publish_check(s) is True
    a = RUN.alter(s, name)
    assert RUN.B6.independent_publish_check(a) is False
    assert vars(s) == vars(good_snapshot())                                                # the original is unchanged
    changed = {k for k in vars(s) if vars(a)[k] != vars(s)[k]}
    assert changed <= ({"fit_state", "convergence_metadata"} if name == "fit_state_not_converged" else {name})
    assert changed


def test_an_unknown_alteration_is_an_error():
    with pytest.raises(ValueError):
        RUN.alter(good_snapshot(), "nothing")


def test_the_independent_check_follows_the_contested_state_and_every_flag():
    from backbone.hssm.model import FitState
    s = good_snapshot()
    s.fit_state, s.convergence_metadata = FitState.MODEL_SELECTION_CONTESTED, {"fit_state": FitState.MAX_ITER_REACHED}
    assert RUN.B6.independent_publish_check(s) is False                                    # contested: the underlying state decides
    s.convergence_metadata = {"fit_state": FitState.CONVERGED}
    assert RUN.B6.independent_publish_check(s) is True
    for md in ({}, None):                                                                   # no underlying state recorded: the default is CONVERGED (the production reading)
        s.convergence_metadata = md
        assert RUN.B6.independent_publish_check(s) is True, md
    for attr, val in (("late_fallbacks", 1), ("fit_state", FitState.NUMERICAL_FAILURE), ("fit_state", FitState.REJECTED_UPDATE), ("fit_state", FitState.FAILED)):
        t = good_snapshot()
        setattr(t, attr, val)
        assert RUN.B6.independent_publish_check(t) is False, (attr, val)


def test_the_checker_control_passes_on_a_good_snapshot_and_fails_when_the_checker_is_blind(monkeypatch):
    c = RUN.checker_control(good_snapshot())
    assert c["unaltered"] is True and set(c["altered"]) == set(RUN.ALTERATIONS) and not any(c["altered"].values())
    monkeypatch.setattr(RUN.B6, "independent_publish_check", lambda r: True)               # a checker that accepts everything: the control must show it
    blind = RUN.checker_control(good_snapshot())
    assert all(blind["altered"].values()) and AN.checker_problems(dict(snapshot_control=blind)) != []


def test_the_settings_probes_are_refused_before_anything_is_fitted():
    meta = dict(settings=dict(candidate_ks=(5,), n_initializations=10, max_iter=100, random_seed=1), earlier_state="STILL_WITHHELD", last_check_day=30)
    stored, new = np.zeros((30, 3)), np.ones((5, 3))
    out = RUN.settings_probes(stored, new, meta, np.arange(30.0), np.arange(30.0, 35.0))
    assert [p["probe"] for p in out] == list(AN.PROBES)
    assert all(p["state"] == "NOT_ELIGIBLE" and p["result_returned"] is False and p["rules_relaxed"] is False for p in out)
    assert "n_initializations_below_the_production_minimum" in out[0]["reasons"][0] and out[1]["reasons"] == ["unknown_setting:allow_fast_test_fit"] and out[2]["reasons"] == ["unknown_setting:config"]
    assert meta["settings"] == dict(candidate_ks=(5,), n_initializations=10, max_iter=100, random_seed=1)    # the caller's settings are not changed by the probes


# ------------------------------------------------------------------------------------------------------------------- the per-user flow with a stand-in for fit_hssm
@pytest.fixture
def fake_fit(monkeypatch):
    calls = []
    state = dict(publish_at=45)                                                            # a fit is published from this many rows on

    def fit(matrix=None, timestamps=None, **kw):
        calls.append(dict(rows=len(matrix), kw=kw, ts=None if timestamps is None else len(timestamps)))
        return result(len(matrix) < state["publish_at"], K=5)
    import backbone.hssm.fitting as F
    monkeypatch.setattr(F, "fit_hssm", fit)
    return calls, state


def test_a_withheld_user_is_recertified_with_the_same_settings_and_publishes_when_the_stand_in_does(fake_fit):
    calls, state = fake_fit
    r = RUN.one(("W1", 10601))
    assert r["ok"] and r["arm"] == "W1" and r["user"] == 10601 and r["stratum"] == "medium" and r["sep"] == [1.0, 2.0] and r["end0"] >= 30
    assert r["day30"]["withheld"] is True and r["day30"]["independent_publishable"] is False
    assert [p["state"] for p in r["probes"]] == ["NOT_ELIGIBLE"] * 3
    e = r["recert"]["60"]
    assert e["state"] == "PUBLISHABLE" and e["rules_relaxed"] is False and e["result_returned"] and e["result_withheld"] is False and e["independent_publishable"] is True and e["f1_withheld"] is False
    assert r["recert"]["90"] == dict(state="PUBLISHABLE_EARLIER")
    assert r["snapshot_control"]["unaltered"] is True and not any(r["snapshot_control"]["altered"].values())
    # the fits: day 30, recertify (rows to day 60), F1 on the same matrix; same settings every time
    assert [c["rows"] for c in calls] == [r["end0"], 60, 60]
    for c in calls:
        assert c["kw"]["candidate_ks"] == (5,) and c["kw"]["n_initializations"] == 10 and c["kw"]["max_iter"] == 100 and c["kw"]["random_seed"] == 10601
        assert c["ts"] == c["rows"]
    assert AN.violations(r) == [] and AN.f1_disagreements(r) == [] and AN.checker_problems(r) == []


@pytest.mark.parametrize("user", [10601, 10602, 10603, 10604, 10605, 10606, 10607, 10608])
def test_the_first_check_is_the_shortest_prefix_with_thirty_present_days(fake_fit, user):
    st, _ = RUN.stream_for("W1", user)
    present = lambda e: int((~np.isnan(st.X_obs[:e]).all(axis=1)).sum())
    expected = next(e for e in range(30, 91) if present(e) >= 30)
    r = RUN.one(("W1", user))
    assert r["end0"] == expected and present(r["end0"]) >= 30 and (r["end0"] == 30 or present(r["end0"] - 1) < 30)


def test_a_user_that_stays_withheld_goes_through_both_checks_and_has_no_control(fake_fit):
    calls, state = fake_fit
    state["publish_at"] = 10 ** 6
    r = RUN.one(("W2", 10651))
    assert r["ok"] and r["stratum"] == "very_hard" and r["sep"] == [0.3, 0.6]
    assert r["recert"]["60"]["state"] == "STILL_WITHHELD" and r["recert"]["90"]["state"] == "STILL_WITHHELD" and r["recert"]["90"]["result_returned"] is False
    assert r["recert"]["90"]["f1_withheld"] is True and r["snapshot_control"] is None and len(r["probes"]) == 3
    assert [c["rows"] for c in calls] == [r["end0"], 60, 60, 90, 90]
    assert calls[0]["kw"]["candidate_ks"] == (2, 3, 4)                                     # the natural arm keeps the production candidates
    assert AN.violations(r) == [] and AN.checker_problems(r) is None


def test_a_user_published_at_day_30_is_not_recertified_and_its_control_comes_from_the_day_30_result(fake_fit):
    calls, state = fake_fit
    state["publish_at"] = 0
    r = RUN.one(("W1", 10602))
    assert r["day30"]["withheld"] is False and r["probes"] is None and r["recert"] == {"60": dict(state="PUBLISHABLE_EARLIER"), "90": dict(state="PUBLISHABLE_EARLIER")}
    assert len(calls) == 1 and r["snapshot_control"]["unaltered"] is True and AN.violations(r) == []


def test_a_user_published_only_at_day_90_is_recertified_twice(fake_fit):
    calls, state = fake_fit
    state["publish_at"] = 75
    r = RUN.one(("W1", 10603))
    assert r["recert"]["60"]["state"] == "STILL_WITHHELD" and r["recert"]["90"]["state"] == "PUBLISHABLE" and r["recert"]["90"]["f1_withheld"] is False
    assert r["snapshot_control"] is not None and AN.published_by(r, "60") is False and AN.published_by(r, "90") is True


def test_a_failing_fit_gives_an_unsuccessful_record_with_the_error(monkeypatch):
    import backbone.hssm.fitting as F
    monkeypatch.setattr(F, "fit_hssm", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    r = RUN.one(("W1", 10601))
    assert r["ok"] is False and r["error"].startswith("RuntimeError: boom")


def test_read_done_keeps_only_successful_lines_and_skips_garbage(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps(rec(1)) + "\n" + "not json\n" + json.dumps(dict(rec(2), ok=False)) + "\n")
    assert [r["user"] for r in RUN.read_done(str(p))] == [1] and RUN.read_done(str(tmp_path / "none.jsonl")) == []


# ------------------------------------------------------------------------------------------------------------------- the analysis
def test_the_stopping_rule_is_applied_in_seed_order_and_includes_the_user_that_reaches_the_target():
    recs = [rec(u, withheld=(u % 2 == 0)) for u in reversed(range(40))]                    # shuffled order on purpose
    used, extra = AN.used_users(recs)
    assert [r["user"] for r in used] == list(range(23)) and [r["user"] for r in extra] == list(range(23, 40))
    assert sum(r["day30"]["withheld"] for r in used) == 12
    used, extra = AN.used_users(recs, target=3)
    assert [r["user"] for r in used] == [0, 1, 2, 3, 4] and extra[0]["user"] == 5
    few = [rec(u, withheld=(u == 3)) for u in range(10)]
    used, extra = AN.used_users(few)
    assert len(used) == 10 and extra == []


def test_a_clean_withheld_user_has_no_violation_and_each_breach_is_named():
    assert AN.violations(rec(1)) == []
    r = rec(1); r["day30"]["independent_publishable"] = True; r["day30"]["withheld"] = False
    assert AN.violations(r) == []                                                           # published at day 30 and the independent check agrees
    r["day30"]["independent_publishable"] = False
    assert "day-30" in AN.violations(r)[0]
    cases = [
        ("recert.60.rules_relaxed", lambda r: r["recert"]["60"].update(rules_relaxed=True), "rules_relaxed"),
        ("recert.60.rules_relaxed missing", lambda r: r["recert"]["60"].pop("rules_relaxed"), "rules_relaxed"),
        ("PUBLISHABLE without a result", lambda r: r["recert"]["60"].update(result_returned=False), "without a result"),
        ("PUBLISHABLE with a withheld result", lambda r: r["recert"]["60"].update(result_withheld=True), "itself withheld"),
        ("PUBLISHABLE refused independently", lambda r: r["recert"]["60"].update(independent_publishable=False), "independent"),
        ("PUBLISHABLE independent flag missing", lambda r: r["recert"]["60"].pop("independent_publishable"), "independent"),
        ("PUBLISHABLE result_withheld missing", lambda r: r["recert"]["60"].pop("result_withheld"), "itself withheld"),
        ("probe missing", lambda r: r.update(probes=r["probes"][:2]), "missing"),
        ("probe not refused", lambda r: r["probes"][1].update(state="PUBLISHABLE"), "not refused"),
        ("probe returned a result", lambda r: r["probes"][0].update(result_returned=True), "not refused"),
        ("probe relaxed", lambda r: r["probes"][2].update(rules_relaxed=True), "not refused"),
        ("no probes at all", lambda r: r.update(probes=None), "missing"),
    ]
    for label, mutate, word in cases:
        r = rec(1); mutate(r)
        v = AN.violations(r)
        assert v and any(word in x for x in v), (label, v)
    r = rec(1, s60="STILL_WITHHELD", s90="STILL_WITHHELD"); r["recert"]["90"]["result_returned"] = True
    assert any("handed out" in x for x in AN.violations(r))
    r = rec(1, s60="STILL_WITHHELD", s90="STILL_WITHHELD"); r["recert"]["60"]["rules_relaxed"] = None
    assert any("rules_relaxed" in x for x in AN.violations(r))


def test_an_earlier_published_entry_is_not_a_recertify_output_and_is_not_checked():
    r = rec(1)
    r["recert"]["90"] = dict(state="PUBLISHABLE_EARLIER", rules_relaxed=True, result_returned=True)
    assert AN.violations(r) == []


def test_f1_must_agree_with_the_state_on_every_call():
    assert AN.f1_disagreements(rec(1)) == [] and AN.f1_disagreements(rec(1, s60="STILL_WITHHELD", s90="STILL_WITHHELD")) == []
    r = rec(1); r["recert"]["60"]["f1_withheld"] = True
    assert AN.f1_disagreements(r) == ["60"]
    r = rec(1, s60="STILL_WITHHELD", s90="PUBLISHABLE"); r["recert"]["60"]["f1_withheld"] = False
    assert AN.f1_disagreements(r) == ["60"]
    r = rec(1, s60="STILL_WITHHELD", s90="STILL_WITHHELD"); r["recert"]["90"]["f1_withheld"] = False
    assert AN.f1_disagreements(r) == ["90"]


def test_the_checker_control_is_judged_per_alteration():
    assert AN.checker_problems(rec(1, withheld=True, s60="STILL_WITHHELD", s90="STILL_WITHHELD")) is None
    assert AN.checker_problems(rec(1)) == []
    r = rec(1); r["snapshot_control"]["unaltered"] = False
    assert AN.checker_problems(r) == ["the unaltered snapshot is refused"]
    for n in AN.ALTERATIONS:
        r = rec(1); r["snapshot_control"]["altered"][n] = True
        assert AN.checker_problems(r) == [f"the alteration {n} is not refused"]
    r = rec(1); del r["snapshot_control"]["altered"]["fit_degraded"]
    assert AN.checker_problems(r) == ["the alteration fit_degraded is not refused"]       # a missing alteration is a failure, not a pass


def test_published_by_distinguishes_day_60_and_day_90():
    a = rec(1)                                                                              # published at 60
    b = rec(2, s60="STILL_WITHHELD", s90="PUBLISHABLE")
    c = rec(3, s60="STILL_WITHHELD", s90="STILL_WITHHELD")
    assert [AN.published_by(x, "60") for x in (a, b, c)] == [True, False, False]
    assert [AN.published_by(x, "90") for x in (a, b, c)] == [True, True, False]


def test_reason_kinds_counts_the_prefix_before_the_colon():
    assert AN.reason_kinds(["collapsed_regime: a", "collapsed_regime: b", "fit_state_not_converged"]) == {"collapsed_regime": 2, "fit_state_not_converged": 1}


def arms_of(recs_w1, recs_w2=()):
    return {"W1": (list(recs_w1), [], 0), "W2": (list(recs_w2), [], 0)}


def test_h1_needs_at_least_eight_withheld_users_over_both_arms():
    assert AN.levels(arms_of([rec(i) for i in range(4)], [rec(100 + i) for i in range(3)]))["H1"] == dict(withheld=7, level=8, met=False)
    assert AN.levels(arms_of([rec(i) for i in range(4)], [rec(100 + i) for i in range(4)]))["H1"] == dict(withheld=8, level=8, met=True)
    assert AN.levels(arms_of([rec(i, withheld=False) for i in range(20)]))["H1"]["withheld"] == 0


def test_h2_counts_the_violations_of_all_used_users_and_names_the_users():
    good = [rec(i) for i in range(8)]
    assert AN.levels(arms_of(good))["H2"]["met"] is True
    bad = rec(99); bad["recert"]["60"]["rules_relaxed"] = True
    lv = AN.levels(arms_of(good, [bad]))["H2"]
    assert lv["violations"] == 1 and lv["met"] is False and list(lv["detail"]) == [99]


def test_h3_and_h4_are_not_met_when_nothing_was_checked():
    lv = AN.levels(arms_of([rec(i, withheld=False) for i in range(5)]))
    assert lv["H3"]["calls"] == 0 and lv["H3"]["met"] is False                              # nothing re-certified: no evidence
    assert lv["H4"]["controls"] == 5 and lv["H4"]["met"] is True
    lv = AN.levels(arms_of([rec(i, s60="STILL_WITHHELD", s90="STILL_WITHHELD") for i in range(5)]))
    assert lv["H4"]["controls"] == 0 and lv["H4"]["met"] is False and lv["H3"]["calls"] == 10 and lv["H3"]["met"] is True


def test_h3_and_h4_report_the_disagreements_and_problems():
    r = rec(1); r["recert"]["60"]["f1_withheld"] = True
    s = rec(2); s["snapshot_control"]["altered"]["fit_collapsed"] = True
    lv = AN.levels(arms_of([r, s, rec(3)]))
    assert lv["H3"]["met"] is False and lv["H3"]["disagreements"] == [(1, "60")] and lv["H4"]["met"] is False and list(lv["H4"]["problems"]) == [2]


def test_the_arm_block_counts_and_the_pooled_rates():
    used = [rec(1), rec(2, s60="STILL_WITHHELD", s90="PUBLISHABLE"), rec(3, s60="STILL_WITHHELD", s90="STILL_WITHHELD"), rec(4, withheld=False, stratum="medium")]
    b = AN.arm_block("W1", used, [rec(9)], 2)
    assert (b["users"], b["extra"], b["failed"], b["withheld"], b["p60"], b["p90"]) == (4, 1, 2, 3, 1, 2)
    assert b["seeds_with_withheld"] == [1, 2, 3] and b["day30_reasons"] == {"collapsed_regime": 3} and b["still_withheld_reasons"] == {"collapsed_regime": 1}
    assert b["strata"] == {"hard": (3, 3), "medium": (1, 0)}


def test_the_report_states_every_level_and_is_written_as_json(tmp_path):
    AN.OUT.clear()
    arms = {"W1": ([rec(i, s60=("PUBLISHABLE" if i % 2 else "STILL_WITHHELD"), s90=("PUBLISHABLE" if i % 4 == 0 else "STILL_WITHHELD")) for i in range(10)], [rec(50)], 1), "W2": ([rec(100)], [], 0)}
    res = AN.report(arms)
    text = "\n".join(AN.OUT)
    for word in ("H1 non-vacuous", "H2 no relaxation", "H3 same procedure", "H4 the checker can fail", "MET", "arm W1", "arm W2", "pooled over the arms", "UNKNOWN", "Wilson"):
        assert word in text, word
    assert res["levels"]["H1"]["met"] is True and res["blocks"]["W1"]["users"] == 10 and res["blocks"]["W1"]["extra"] == 1 and res["blocks"]["W1"]["failed"] == 1
    json.dumps(res, default=str)
    assert "NOT MET: too few to read" not in text
    AN.OUT.clear()
    AN.report({"W1": ([rec(1)], [], 0), "W2": ([], [], 0)})
    assert "NOT MET: too few to read" in "\n".join(AN.OUT)


def test_the_analysis_script_runs_end_to_end_on_two_files(tmp_path):
    import subprocess
    w1, w2 = tmp_path / "w1.jsonl", tmp_path / "w2.jsonl"
    w1.write_text("\n".join(json.dumps(rec(i)) for i in range(6)) + "\n" + json.dumps(dict(rec(7), ok=False)) + "\n")
    w2.write_text("\n".join(json.dumps(rec(100 + i, s60="STILL_WITHHELD", s90="STILL_WITHHELD")) for i in range(4)) + "\n")
    p = subprocess.run([sys.executable, str(SCRIPTS / "a7_analyze.py"), str(tmp_path / "o.txt"), str(tmp_path / "o.json"), str(w1), str(w2)], capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr
    txt = (tmp_path / "o.txt").read_text()
    assert "H1 non-vacuous (>= 8 users withheld at the first check): 10 -> MET" in txt and "failed fits excluded: 1" in txt
    js = json.load(open(tmp_path / "o.json"))
    assert js["levels"]["H2"]["met"] is True and js["blocks"]["W2"]["p90"] == 0
    p = subprocess.run([sys.executable, str(SCRIPTS / "a7_analyze.py")], capture_output=True, text=True, cwd=ROOT)
    assert p.returncode != 0


# ------------------------------------------------------------------------------------------------------------------- the driver (chunks, the stopping rule, resumability) with a stand-in for `one`
def _fake_one(job):
    arm, user = job
    return rec(user, withheld=(user % 2 == 0), s60="PUBLISHABLE")


_fake_one.__name__ = _fake_one.__qualname__ = "one"
_fake_one.__module__ = RUN.__name__


def _run_main(monkeypatch, out, extra=()):
    monkeypatch.setattr(RUN, "one", _fake_one)
    monkeypatch.setattr(sys, "argv", ["a7_run.py", "--arm", "W1", "--out", str(out), "--workers", "1", *extra])
    RUN.main()


def test_the_driver_runs_chunks_of_six_until_twelve_are_withheld_and_stops(monkeypatch, tmp_path, capsys):
    out = tmp_path / "w1.jsonl"
    _run_main(monkeypatch, out)
    users = [json.loads(l)["user"] for l in open(out)]
    assert users == list(range(10601, 10625))                                              # 4 chunks of 6: withheld (even seeds) 3, 6, 9, 12 -> stop
    assert "stopping rule met: 12 withheld at day 30" in capsys.readouterr().out
    _run_main(monkeypatch, out)                                                             # a second call does nothing more
    assert [json.loads(l)["user"] for l in open(out)] == users


def test_the_driver_resumes_without_repeating_a_user_and_counts_the_earlier_ones(monkeypatch, tmp_path):
    out = tmp_path / "w1.jsonl"
    out.write_text("".join(json.dumps(_fake_one(("W1", u))) + "\n" for u in (10601, 10602, 10603, 10604, 10605, 10606, 10607)) + json.dumps(dict(_fake_one(("W1", 10608)), ok=False)) + "\n")
    _run_main(monkeypatch, out)
    users = [json.loads(l)["user"] for l in open(out) if json.loads(l)["ok"]]
    assert users[:7] == list(range(10601, 10608)) and len(users) == len(set(users))        # nothing is run twice; the failed user 10608 is run again
    assert 10608 in users and withheld_ok(users) >= 12 and len(users) <= 7 + 24


def withheld_ok(users):
    return sum(1 for u in users if u % 2 == 0)
