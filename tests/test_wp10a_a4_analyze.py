"""WP10A A4: the declared criteria, the pairing and the selection of `integration_report/wp10a/scripts/a4_analyze.py` on fabricated rows, plus a wiring test on ONE stored chain (a development user of A3). Mutation check:
mutation_wp10a_a4.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "integration_report" / "wp10a" / "scripts"
FIXTURE = ROOT / "tests" / "fixtures" / "wp10a_a4_chain_10001.jsonl"


def _load(name, path):
    """Load a script by PATH under a unique module name (the WP9 package has scripts of the same file names)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


X = _load("wp10a_a4_analyze", SCRIPTS / "a4_analyze.py")
RUN = _load("wp10a_a4_run", SCRIPTS / "a4_run.py")


def m(al, served, correct, wu=None):
    wu = served - correct if wu is None else wu
    return dict(alignments=al, served=served, served_correct=correct, wrong_unflagged=wu, correct_rate=(correct / served if served else float("nan")), wu_rate=(wu / al if al else float("nan")), coverage=0.0)


def r(key, served=True, correct=True):
    return dict(pair_key=key, served=served, correct=correct)


# ----------------------------------------------------------------------------------------------------------------- constants
def test_the_declared_levels():
    assert (X.H, X.SHARP, X.MIN_ALIGNMENTS, X.CORRECT_LEVEL, X.WU_LEVEL, X.ZR_MIN, X.ORDER) == (2.5, 0.95, 40, 0.95, 0.01, 3, ("D", "E", "F"))


def test_the_production_default_gate_is_the_declared_one():
    from backbone.hssm.lineage import AlignmentConfig
    assert AlignmentConfig().min_separation == X.H


# ----------------------------------------------------------------------------------------------------------------- R works / old criterion
def test_r_works_boundaries():
    assert X.r_works(m(40, 40, 40))                                   # exactly 40 alignments, 100% correct
    assert not X.r_works(m(39, 39, 39))                               # one alignment short
    assert X.r_works(m(100, 100, 95, wu=1))                           # 95% correct and 1% wrong-and-unflagged are inside
    assert not X.r_works(m(100, 100, 94, wu=1))                       # 94% correct is outside
    assert not X.r_works(m(100, 100, 96, wu=2))                       # 2% wrong-and-unflagged is outside
    assert not X.r_works(m(100, 0, 0, wu=0))                          # nothing served: the method does not "work"
    assert not X.r_works(m(200, 150, 140, wu=10))


def test_the_old_wp9_criterion_is_kept_beside_the_new_one():
    assert not X.z_old_fails(m(100, 100, 100))
    assert X.z_old_fails(m(100, 100, 94, wu=1))                       # correct among served < 95%
    assert X.z_old_fails(m(100, 100, 100, wu=2))                      # wrong-and-unflagged > 1% (counts are fabricated: only the rates matter)
    assert not X.z_old_fails(m(100, 100, 95, wu=1))
    assert not X.z_old_fails(m(0, 0, 0, wu=0))


# ----------------------------------------------------------------------------------------------------------------- pairing and the new criterion
def test_paired_counts_zr_and_rz_on_common_alignments_only():
    rr = [r(1, True, True), r(2, True, True), r(3, True, False), r(4, True, True), r(5, False, False), r(6, True, True), r(7, True, True), r(8, True, False)]
    rz = [r(1, True, False), r(2, True, False), r(3, True, True), r(4, True, True), r(5, True, False), r(6, True, False), r(99, True, False), r(8, True, False)]
    p = X.paired(rr, rz)
    assert p["common"] == 7                                            # key 7 only in R, key 99 only in Z
    assert p["zr"] == 3                                                # keys 1, 2, 6: Z served wrong, R served right
    assert p["rz"] == 1                                                # key 3: R served wrong, Z served right (key 8: both wrong, counted in neither)
    assert p["z_wrong"] == 5 and p["r_wrong"] == 2                     # Z wrong at 1, 2, 5, 6, 8; R wrong at 3, 8
    assert p["served_differs"] == 1                                    # key 5: R withheld, Z served


def test_a_withheld_alignment_is_neither_wrong_nor_right():
    p = X.paired([r(1, False, False), r(2, True, True)], [r(1, True, False), r(2, False, True)])
    assert p["zr"] == 0 and p["rz"] == 0 and p["z_wrong"] == 1 and p["r_wrong"] == 0


def test_rows_without_a_verdict_are_ignored():
    p = X.paired([r(1, True, None), r(2, True, True)], [r(1, True, False), r(2, True, False)])
    assert p["common"] == 1 and p["zr"] == 1


@pytest.mark.parametrize("zr,rz,expect", [(3, 2, True), (3, 3, False), (2, 0, False), (4, 5, False), (10, 0, True), (0, 0, False), (3, 0, True)])
def test_the_new_criterion(zr, rz, expect):
    assert X.control_fails(dict(zr=zr, rz=rz)) is expect


def test_qualifies_needs_both():
    good, bad = m(100, 100, 100), m(100, 100, 80, wu=20)
    assert X.qualifies(good, dict(zr=5, rz=0))
    assert not X.qualifies(good, dict(zr=2, rz=0))
    assert not X.qualifies(bad, dict(zr=5, rz=0))


# ----------------------------------------------------------------------------------------------------------------- the selection
def res(q, zr):
    return dict(qualifies=q, paired=dict(zr=zr))


def test_best_design_is_the_qualifying_one_with_the_largest_zr():
    assert X.best_design(dict(D=res(True, 5), E=res(True, 9), F=res(True, 7))) == "E"
    assert X.best_design(dict(D=res(True, 5), E=res(False, 99), F=res(True, 7))) == "F"


def test_ties_go_in_the_order_d_e_f():
    assert X.best_design(dict(D=res(True, 6), E=res(True, 6), F=res(True, 6))) == "D"
    assert X.best_design(dict(D=res(False, 6), E=res(True, 6), F=res(True, 6))) == "E"


def test_nothing_qualifies_means_closed():
    assert X.best_design(dict(D=res(False, 9), E=res(False, 9), F=res(False, 9))) is None
    assert X.best_design({}) is None
    assert X.best_design(dict(D_none=res(True, 50))) is None          # only the three designs are candidates


# ----------------------------------------------------------------------------------------------------------------- chains
def rec(user, group, end, start=0, seed=None):
    return dict(job=dict(user=user, group=group, end=end, start=start, fitseed=user if seed is None else seed))


def test_chains_d_take_the_shared_windows_and_one_variant_in_window_order():
    recs = [rec(1, "A4D-D", 90), rec(1, "A4D-shared", 60), rec(1, "A4D-none", 75), rec(1, "A4D-shared", 45), rec(1, "A4D-D", 75), rec(2, "A4D-shared", 45), rec(2, "A4D-none", 60), rec(3, "A4E-anchor", 90)]
    d, n = X.chains_d(recs, "D"), X.chains_d(recs, "none")
    assert [x["job"]["end"] for x in d[(1,)]] == [45, 60, 75, 90] and all(x["job"]["group"] != "A4D-none" for x in d[(1,)])
    assert [x["job"]["end"] for x in n[(1,)]] == [45, 60, 75] and all(x["job"]["group"] != "A4D-D" for x in n[(1,)])
    assert set(d) == {(1,), (2,)} and (3,) not in d


def test_chains_e_are_one_pair_chains_with_the_long_anchor():
    recs = [rec(1, "A4E-anchor", 90), rec(1, "A4E-L20", 105, 85), rec(1, "A4E-L30", 105, 75), rec(1, "A4E-L20", 120, 100), rec(2, "A4E-L20", 105, 85), rec(3, "A4D-D", 105)]
    c = X.chains_e(recs)
    assert set(c) == {(1, 20, 105), (1, 30, 105), (1, 20, 120)}       # user 2 has no anchor; user 3 is a D record
    for key, ch in c.items():
        assert len(ch) == 2 and ch[0]["job"]["group"] == "A4E-anchor" and ch[1]["job"]["end"] == key[2] and ch[1]["job"]["group"] == f"A4E-L{key[1]}"


def test_chains_f_are_the_seed_a_chains_only():
    recs = [rec(5, "x", 90), rec(5, "x", 97), rec(5, "x", 90, seed=1005), rec(6, "x", 90)]
    c = X.chains_f(recs)
    assert set(c) == {(5,), (6,)} and len(c[(5,)]) == 2


# ----------------------------------------------------------------------------------------------------------------- the control itself
def test_patch_zspace_replaces_and_restores_the_vector_function():
    from backbone.hssm import lineage as Lin
    orig = Lin._vectors
    s = SimpleNamespace(raw_means=np.array([[1.0, 2.0], [3.0, 8.0]]), scaler_mean=np.array([1.0, 2.0]), scaler_std=np.array([2.0, 0.0]))
    with X.patch_zspace():
        assert Lin._vectors is not orig
        v = Lin._vectors(s, None, 0.0)
        assert np.allclose(v, [[0.0, 0.0], [1.0, 6.0]])                # (x - mean) / sd, with sd == 0 replaced by 1
    assert Lin._vectors is orig
    with pytest.raises(RuntimeError):
        with X.patch_zspace():
            raise RuntimeError("x")
    assert Lin._vectors is orig                                       # restored on an exception too


def test_the_control_is_the_same_replacement_as_the_wp9_one():
    from backbone.hssm import lineage as Lin
    w9 = _load("wp9_a4_analyze_for_comparison", ROOT / "integration_report" / "wp9" / "scripts" / "a4_analyze.py")
    s = SimpleNamespace(raw_means=np.array([[1.0, 2.0, 5.0], [3.0, 8.0, 1.0]]), scaler_mean=np.array([1.0, 2.0, 3.0]), scaler_std=np.array([2.0, 0.0, 4.0]))
    with X.patch_zspace():
        a = Lin._vectors(s, None, 0.0)
    with w9.patch_zspace():
        b = Lin._vectors(s, None, 0.0)
    assert np.array_equal(a, b)


# ----------------------------------------------------------------------------------------------------------------- wiring on a stored chain
@pytest.fixture(scope="module")
def chain():
    recs = [json.loads(line) for line in open(FIXTURE)]
    return X.chains_f(recs), recs


def test_the_stored_chain_replays_under_both_methods_without_changing_the_production_function(chain):
    from backbone.hssm import lineage as Lin
    orig = Lin._vectors
    chains, recs = chain
    assert len(chains) == 1 and len(next(iter(chains.values()))) == 10
    out = X.study("F", chains)
    assert Lin._vectors is orig
    assert out["R"]["alignments"] >= 1 and out["Z"]["alignments"] >= 1
    assert out["paired"]["common"] >= 1
    assert out["fits"] == 10 and 0 <= out["withheld"] <= 10
    assert set(out) >= {"R", "Z", "paired", "r_works", "control_fails", "z_old_fails", "qualifies", "fits", "withheld", "by_stratum"}


def test_the_replay_is_deterministic(chain):
    chains, _ = chain
    a = X.replay_all(chains, False, "t1")
    b = X.replay_all(chains, False, "t2")
    assert [(x["pair_key"], x["state"], x["served"]) for x in a] == [(x["pair_key"], x["state"], x["served"]) for x in b]


# ----------------------------------------------------------------------------------------------------------------- run script and refusal
def test_job_counts_are_the_declared_ones():
    assert len(RUN.jobs_for(list(RUN.RANGES["dev"]), ["D"])) == 224
    assert len(RUN.jobs_for(list(RUN.RANGES["dev"]), ["E"])) == 144
    assert len(RUN.jobs_for(list(RUN.RANGES["eval"]), ["D"])) == 224 and len(RUN.jobs_for(list(RUN.RANGES["eval"]), ["E"])) == 144


def test_seed_ranges_are_inside_the_wp10a_reservation_and_do_not_overlap_a3():
    dev, ev = set(RUN.RANGES["dev"]), set(RUN.RANGES["eval"])
    assert min(dev) == 10301 and max(dev) == 10316 and min(ev) == 10321 and max(ev) == 10336 and not dev & ev
    assert all(10001 <= u <= 10999 for u in dev | ev)
    a3 = set(range(10001, 10049)) | set(range(10101, 10149)) | set(range(10151, 10211))
    assert not (dev | ev) & a3


def test_design_d_windows_and_the_planted_change():
    jobs = RUN.jobs_for([10301], ["D"])
    by = {(j["group"], j["end"]): j for j in jobs}
    assert {j["end"] for j in jobs if j["group"] == "A4D-shared"} == {45, 60}
    for e in (75, 90, 105, 120, 135, 150):
        assert by[("A4D-D", e)]["regime_change"] == dict(regime=2, features=[0, 1, 2, 3], factor=3.0, start=60)
        assert by[("A4D-none", e)]["regime_change"] is None
        assert by[("A4D-D", e)]["start"] == e - 45 and by[("A4D-D", e)]["end"] == e
    assert all(j["regime_change"] is None for j in jobs if j["group"] == "A4D-shared")
    assert all(j["stratum"] == "easy" and j["K"] == 3 and j["n_init"] == 10 and j["fitseed"] == j["user"] for j in jobs)


def test_design_e_windows():
    jobs = RUN.jobs_for([10301], ["E"])
    anchor = [j for j in jobs if j["group"] == "A4E-anchor"]
    assert len(anchor) == 1 and (anchor[0]["start"], anchor[0]["end"]) == (0, 90)
    short = sorted((j["end"], j["end"] - j["start"]) for j in jobs if j["group"] != "A4E-anchor")
    assert short == sorted((t, L) for t in (105, 120, 135, 150) for L in (20, 30))
    assert all(j["regime_change"] is None and j["scale_change"] is None for j in jobs)


def test_the_evaluation_is_refused_without_the_freeze_file(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "FREEZE", tmp_path / "missing.txt")
    with pytest.raises(SystemExit):
        X.evaluate(str(tmp_path / "o.json"), [str(FIXTURE)], False)


def test_run_refuses_eval_without_the_freeze_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["a4_run.py", "--range", "eval", "--out", str(tmp_path / "x.jsonl")])
    with pytest.raises(SystemExit) as e:
        RUN.main()
    assert "DESIGN_FREEZE_WP10A_A4" in str(e.value)
    assert not (tmp_path / "x.jsonl").exists()


def test_run_refuses_to_fit_design_f(tmp_path, monkeypatch):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "DESIGN_FREEZE_WP10A_A4.txt").write_text('{"design": "F"}\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["a4_run.py", "--range", "eval", "--out", str(tmp_path / "x.jsonl")])
    with pytest.raises(SystemExit) as e:
        RUN.main()
    assert "replay-only" in str(e.value)


# ----------------------------------------------------------------------------------------------------------------- more pairing, metrics and wiring (added after the first mutation run)
def test_a_withheld_alignment_that_would_have_been_right_is_not_a_right_one():
    p = X.paired([r(1, False, True)], [r(1, True, False)])            # R withheld (its answer would have been right); Z served wrong
    assert p["zr"] == 0 and p["rz"] == 0 and p["z_wrong"] == 1 and p["r_wrong"] == 0


def arow(correct, served, state="ALIGNED"):
    return dict(correct=correct, served=served, both_recoverable=True, state=state)


def test_metrics_on_fabricated_rows():
    rows = [arow(True, True)] * 6 + [arow(False, True)] * 2 + [arow(True, False), arow(False, False)] + [arow(None, False, "NEW_LINEAGE")]
    mm = X.metrics(rows)
    assert (mm["alignments"], mm["served"], mm["served_correct"], mm["wrong_unflagged"]) == (10, 8, 6, 2)
    assert mm["correct_rate"] == 6 / 8 and mm["wu_rate"] == 2 / 10 and mm["coverage"] == 8 / 10
    e = X.metrics([])
    assert e["alignments"] == 0 and np.isnan(e["correct_rate"]) and np.isnan(e["wu_rate"])


def test_replay_all_passes_the_declared_settings_and_patches_only_for_z(monkeypatch):
    from backbone.hssm import lineage as Lin
    seen = []

    def spy(chain, cfg, n=None, use_flag=True, sharpness=0.0, lineage_id="x"):
        seen.append(dict(h=cfg.min_separation, n=n, use_flag=use_flag, sharpness=sharpness, patched=Lin._vectors is not orig, chain=chain))
        return [], [], None, []

    orig = Lin._vectors
    monkeypatch.setattr(X.R, "replay9", spy)
    X.replay_all({("k",): ["rec"]}, False, "t")
    X.replay_all({("k",): ["rec"]}, True, "t")
    assert [(s["h"], s["n"], s["use_flag"], s["sharpness"], s["patched"]) for s in seen] == [(2.5, 10, True, 0.95, False), (2.5, 10, True, 0.95, True)]
    assert seen[0]["chain"] == ["rec"] and Lin._vectors is orig


def test_every_row_gets_a_distinct_pair_key(chain):
    chains, _ = chain
    rows = X.replay_all(chains, False, "k")
    assert len(rows) >= 2 and len({x["pair_key"] for x in rows}) == len(rows)
    assert all(x["pair_key"] == (next(iter(chains)), x["end"]) for x in rows)


def test_withheld_counts_the_fits_the_replay_cannot_use(monkeypatch):
    a, b, c = dict(n=1), dict(n=2), dict(n=3)
    monkeypatch.setattr(X.R, "prepared", lambda rec, n: None if rec is b else dict(ok=1))
    assert X.withheld({("x",): [a, b], ("y",): [b, c]}) == (3, 1)    # a record shared by two chains is counted once


def test_the_evaluation_reads_the_design_from_the_freeze_file_and_runs_that_design_only(tmp_path, monkeypatch):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    for design, expect in (("E", {"E"}), ("F", {"F"}), ("D", {"D", "D_none"})):
        fz = tmp_path / f"fz_{design}.txt"
        fz.write_text(json.dumps({"design": design}) + "\n")
        monkeypatch.setattr(X, "FREEZE", fz)
        X.OUT.clear()
        out = tmp_path / f"o_{design}.json"
        X.evaluate(str(out), [str(empty)], False)
        got = json.load(open(out))
        assert got["design"] == design and set(got) - {"design"} == expect
        assert any("frozen design " + design in line for line in X.OUT) and any(f"EVALUATION design {design}" in line for line in X.OUT)
