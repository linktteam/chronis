"""WP10A A3 wiring: the replay, the pair cases, the declared metrics, the development choice and the report of `integration_report/wp10a/scripts/a3_analyze.py` on REAL stored fits (development user 10001, hard
stratum: the seed-A weekly chain, 10 fits, and the seed-B 14-day chain, 6 fits; `tests/fixtures/wp10a_a4_chain_10001.jsonl` and `wp10a_a3_chain_10001_B.jsonl`), checked against an independent recomputation that does
not use the script's own helpers for pairing and counting. Mutation check: mutation_wp10a_a3.py (this file is run with the pure tests)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "integration_report" / "wp10a" / "scripts"
FIX_A, FIX_B = ROOT / "tests" / "fixtures" / "wp10a_a4_chain_10001.jsonl", ROOT / "tests" / "fixtures" / "wp10a_a3_chain_10001_B.jsonl"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


X = _load("wp10a_a3_analyze_wiring", SCRIPTS / "a3_analyze.py")
A, R, S = X.A, X.R, X.S


@pytest.fixture(scope="module")
def data():
    recs = [json.loads(l) for l in open(FIX_A)] + [json.loads(l) for l in open(FIX_B)]
    a, b = A.split_chains(recs)
    return recs, a, b


def test_the_fixtures_are_one_user_two_seeds(data):
    recs, a, b = data
    assert list(a) == [(10001, 10001)] and list(b) == [(10001, 11001)]
    assert [r["job"]["end"] for r in a[(10001, 10001)]] == X.G_T and [r["job"]["end"] for r in b[(10001, 11001)]] == X.B_T
    assert all(r["job"]["stratum"] == "hard" and r["ok"] for r in recs)


def independent_cases(a_rows, a_lrs, b_rows, b_lrs):
    """Pairing recomputed from the replay output without the script's helpers."""
    da = {r["end"]: (r, lr) for r, lr in zip(a_rows, a_lrs)}
    db = {r["end"]: (r, lr) for r, lr in zip(b_rows, b_lrs)}
    out = []
    for e in sorted(set(da) | set(db)):
        if e <= 90:
            continue
        sa = e in da and da[e][0]["state"] == "ALIGNED"
        sb = e in db and db[e][0]["state"] == "ALIGNED"
        agree = None
        if sa and sb:
            ia, ib = np.asarray(da[e][1].stable_regime_id), np.asarray(db[e][1].stable_regime_id)
            agree = bool(np.mean(ia == ib) >= 0.99)
        out.append((e, sa, sb, agree))
    return out


@pytest.mark.parametrize("u,h", [(True, 2.5), (True, 2.0), (False, 2.25), (False, 2.5)])
def test_pair_cases_equal_an_independent_pairing_of_the_replays(data, u, h):
    _, a, b = data
    ra, la = X.replay_chain(a[(10001, 10001)], u, h, X.B_T)
    rb, lb = X.replay_chain(b[(10001, 11001)], u, h, X.B_T)
    cases = X.pair_cases(ra, la, rb, lb)
    assert [(c["end"], c["a"], c["b"], c["agree"]) for c in cases] == independent_cases(ra, la, rb, lb)
    assert all(c["end"] > 90 and c["end"] in X.B_T for c in cases)


def test_the_replay_uses_the_declared_flag_and_gate(data, monkeypatch):
    _, a, _ = data
    seen = []
    orig = R.replay9
    monkeypatch.setattr(R, "replay9", lambda chain, cfg, n=None, use_flag=True, sharpness=0.0, lineage_id="x": (seen.append((cfg.min_separation, n, use_flag, sharpness, [r["job"]["end"] for r in chain])), orig(chain, cfg, n=n, use_flag=use_flag, sharpness=sharpness))[1])
    X.replay_chain(a[(10001, 10001)], True, 2.25, X.B_T)
    X.replay_chain(a[(10001, 10001)], False, 2.0, X.G_T)
    assert seen[0][:4] == (2.25, 10, True, 0.95) and seen[0][4] == [90, 104, 118, 132, 146, 150]       # only the windows of the schedule are fed
    assert seen[1][:4] == (2.0, 10, False, 0.0) and seen[1][4] == X.G_T


def test_evaluate_candidate_and_summarize_agree_with_the_replays(data):
    _, a, b = data
    strata = {"hard": {10001}, "medium": set(), "easy": set()}
    ev = X.evaluate_candidate(a, b, True, 2.5, strata)
    pu = ev["per_user"][10001]
    assert set(pu) == {"tstar_A", "cases", "tstar_B"}
    rows, _ = ev["res"][(10001, 10001)]
    assert pu["tstar_A"] == next((r["end"] for r in rows if r["end"] > 90 and r["state"] == "ALIGNED"), None)
    rb, _ = X.replay_chain(b[(10001, 11001)], True, 2.5, X.B_T)
    assert pu["tstar_B"] == next((r["end"] for r in rb if r["end"] > 90 and r["state"] == "ALIGNED"), None)
    s = X.summarize(ev, None)
    cases = pu["cases"]
    assert s["names"] == (sum(1 for c in cases if c["a"] and c["b"] and c["agree"]), sum(1 for c in cases if c["a"] and c["b"]))
    sv = s["serve"]
    assert sv["cases"] == len(cases) and sv["both_serve"] + sv["both_withhold"] + sv["one_serves"] == len(cases)
    assert s["strict"][1] == sum(1 for c in cases if c["a"] or c["b"]) and s["users"] == 1
    assert s["days_A"]["n"] == 1 and s["days_B"]["n"] == 1
    t = s["tally"]
    assert t["refits"] == len(rows) - 1 and t["alignments"] == sum(1 for r in rows if r["correct"] is not None)       # steps after the first fit of the chain


def test_the_weekly_chain_is_cut_to_the_weekly_schedule(data):
    _, a, b = data
    ev = X.evaluate_candidate(a, b, True, 2.5, {"hard": {10001}, "medium": set(), "easy": set()})
    rows, _ = ev["res"][(10001, 10001)]
    assert {r["end"] for r in rows} <= set(X.G_T)


def test_every_candidate_of_the_grid_runs_and_flag_off_is_never_more_conservative_in_withholding(data):
    _, a, b = data
    strata = {"hard": {10001}, "medium": set(), "easy": set()}
    served = {}
    for (u, h) in X.cand_list():
        ev = X.evaluate_candidate(a, b, u, h, strata)
        served[(u, h)] = X.A.tally_rows(ev["res"])["served"]
    assert len(served) == 6
    for h in X.H_CANDS:
        assert served[(False, h)] >= served[(True, h)], h                                            # the flag and the floor can only withhold more


def test_dev_writes_the_frozen_rule_output_for_the_fixture(data, tmp_path):
    recs, _, _ = data
    X.OUT.clear()
    X.dev(recs, str(tmp_path / "dev.json"))
    got = json.load(open(tmp_path / "dev.json"))
    assert got["chosen"]["u"] is True and got["chosen"]["h"] in X.H_CANDS and got["chosen"]["s"] == 0.95
    assert set(got["curve"]) == {f"u={int(u)},h={h}" for u in (False, True) for h in X.H_CANDS}
    assert set(got["readout"]) == set(got["curve"])
    for v in got["readout"].values():
        assert set(v) == {"pooled", "hard", "medium", "easy"}


@pytest.mark.parametrize("rehearsal", [True, False])
def test_evaluate_reports_the_declared_blocks_and_the_safety_bound_for_the_fixture(data, tmp_path, rehearsal):
    recs, a, b = data
    X.OUT.clear()
    gate = dict(h=2.25, s=0.95, u=True)
    X.evaluate(recs, str(tmp_path / "e.json"), gate, rehearsal)
    got = json.load(open(tmp_path / "e.json"))
    assert got["gate"] == gate and set(got["safety"]) == {"pooled", "hard", "medium", "easy"}
    ev = X.evaluate_candidate(a, b, True, 2.25, {"hard": {10001}, "medium": set(), "easy": set()})
    t = A.tally_rows(ev["res"])
    sf = got["safety"]["pooled"]
    assert (sf["n"], sf["wrong"]) == (t["alignments"], t["wrong_unflagged"]) and sf["exact_upper"] == pytest.approx(X.exact_upper(sf["wrong"], sf["n"]))
    assert sf["demonstrated"] is False                                                                 # a handful of alignments cannot demonstrate a 1% bound
    assert got["safety"]["medium"]["n"] == 0 and got["safety"]["hard"]["n"] == sf["n"]
    assert set(got["curve"]) == {f"u={int(u)},h={h}" for u in (False, True) for h in X.H_CANDS}
    assert set(got["secondary"]) == {"current default (h = 2.5, flag ON)", "flag OFF at the chosen h = 2.25", "flag OFF at h = 2.5"}
    text = "\n".join(X.OUT)
    assert "frozen operating point" in text and "1% bound demonstrated: NO" in text and "coverage target >= 80%" in text and "wrong-and-unflagged target <= 1%" in text
    assert ("EVAL users only" in text) is (not rehearsal)
