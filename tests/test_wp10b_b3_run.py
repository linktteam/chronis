"""WP10B B3 runner (`integration_report/wp10b/scripts/b3_run.py`): seed ranges and strata, job lists, the pre-declaration guard, the resumable driver, one real wrong-engine job."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "integration_report" / "wp10b" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SC))
_spec = importlib.util.spec_from_file_location("wp10b_b3_run_under_test", SC / "b3_run.py")
R = importlib.util.module_from_spec(_spec)
sys.modules["wp10b_b3_run_under_test"] = R
_spec.loader.exec_module(R)


def test_strata_follow_the_declared_blocks():
    assert [R.stratum_of(u) for u in (11421, 11432, 11433, 11444, 11445, 11456)] == ["hard", "hard", "medium", "medium", "easy", "easy"]
    assert [R.stratum_of(u) for u in (11457, 11464, 11465, 11472, 11473, 11480)] == ["hard", "hard", "medium", "medium", "easy", "easy"]
    assert [R.stratum_of(u) for u in R.S0_USERS["smoke"]] == ["hard", "medium", "easy", "hard"]
    with pytest.raises(ValueError):
        R.stratum_of(11500)


def test_the_ranges_do_not_overlap_and_stay_inside_the_reserved_band():
    ev, ex, sm = R.S0_USERS["eval"], R.S0_USERS["ext"], R.S0_USERS["smoke"]
    assert len(ev) == 36 and len(ex) == 24 and not set(ev) & set(ex) and not set(sm) & (set(ev) | set(ex))
    assert min(ev) >= 11401 and max(ex) <= 11599
    br = [s for v in R.BREAKER_ENGINES.values() for s in v]
    assert not set(br) & (set(ev) | set(ex) | set(sm))
    assert len(R.BREAKER_ENGINES["stops"]) == 30 and len(R.BREAKER_ENGINES["B2"]) == 30 and len(R.BREAKER_ENGINES["real"]) == 20
    assert not set(R.BREAKER_ENGINES["real"]) & set(R.BREAKER_ENGINES["stops"]) and all(11401 <= s <= 11599 for s in br)
    assert set(R.BREAKER_SMOKE["stops"]) == {11905}


def test_job_lists():
    jobs = R.build_jobs("s0", "eval", "S0")
    assert len(jobs) == 36 and jobs[0] == dict(user=11421, stratum="hard", variant="S0") and jobs[-1] == dict(user=11456, stratum="easy", variant="S0")
    assert R.build_jobs("s0", "ext", "ALL")[0]["variant"] == "ALL"
    bj = R.build_jobs("breaker", "eval", "S0")
    assert len(bj) == 80 and {j["engine"] for j in bj} == {"stops", "B2", "real"}
    assert len(R.build_jobs("breaker", "smoke", "S0")) == 3


def test_a_scored_range_is_refused_before_the_dated_criteria_exist(tmp_path, monkeypatch, capsys):
    crit = tmp_path / "c.md"
    crit.write_text("# nothing about B3\n")
    monkeypatch.setattr(R, "CRITERIA", crit)
    monkeypatch.setattr(R, "run", lambda *a, **k: pytest.fail("the scored run started without the dated B3 part"))
    monkeypatch.setattr(sys, "argv", ["b3_run.py", "--arm", "breaker", "--range", "eval", "--out", str(tmp_path / "o.jsonl")])
    with pytest.raises(SystemExit) as e:
        R.main()
    assert "dated B3 part" in str(e.value)
    monkeypatch.setattr(R, "CRITERIA", tmp_path / "missing.md")
    with pytest.raises(SystemExit):
        R.main()
    assert not (tmp_path / "o.jsonl").exists()


def test_the_extension_is_refused_for_the_breaker_arm(tmp_path, monkeypatch):
    crit = tmp_path / "c.md"
    crit.write_text(R.CRITERIA_MARK)
    monkeypatch.setattr(R, "CRITERIA", crit)
    monkeypatch.setattr(sys, "argv", ["b3_run.py", "--arm", "breaker", "--range", "ext", "--out", str(tmp_path / "o.jsonl")])
    with pytest.raises(SystemExit) as e:
        R.main()
    assert "no extension" in str(e.value)


def test_an_unknown_variant_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["b3_run.py", "--arm", "s0", "--range", "smoke", "--variant", "NOPE", "--out", str(tmp_path / "o.jsonl")])
    with pytest.raises(SystemExit) as e:
        R.main()
    assert "unknown variant" in str(e.value)


def fake_job(job):
    return dict(job=job, ok=True, certificate=dict(state="NOT_CERTIFIED", reasons=["x"]), seconds=0.0)


def test_the_driver_is_resumable_and_writes_one_line_per_job(tmp_path):
    out = tmp_path / "o.jsonl"
    jobs = [dict(engine="stops", seed=s) for s in (1, 2, 3)]
    R.run(jobs[:2], fake_job, str(out), workers=1)
    assert len(out.read_text().splitlines()) == 2
    R.run(jobs, fake_job, str(out), workers=1)
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(lines) == 3 and sorted(l["job"]["seed"] for l in lines) == [1, 2, 3]
    R.run(jobs, fake_job, str(out), workers=1)
    assert len(out.read_text().splitlines()) == 3


def test_paired_summary():
    s = R.paired_summary(np.array([0.1, 0.3, 0.2, 0.4, 0.5]))
    assert s["n"] == 5 and s["mean"] == pytest.approx(0.3) and s["se"] is not None and s["finite"] is True
    s = R.paired_summary(np.array([0.1, np.nan, 0.2]))
    assert s["finite"] is False and s["mean"] is None and s["se"] is None


def test_a_wrong_engine_job_is_fitted_refitted_and_refused():
    r = R.breaker_job(dict(engine="stops", seed=11905))
    assert r["ok"] is True and r["job"] == dict(engine="stops", seed=11905) and r["T_train"] > 80 and r["n_test_days"] > 20
    assert r["certificate"]["state"] == "NOT_CERTIFIED" and r["certificate"]["reasons"] == ["gain_not_demonstrated"] and r["certificate"]["blocks"] is not None
    assert len(r["test_g_diff"]) == r["n_test_days"] and float(np.mean(r["test_g_diff"])) < 0
