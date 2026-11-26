"""WP9 A3 analysis script (integration_report/wp9/scripts/a3_analyze.py) on fabricated fits of real generator streams."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "integration_report" / "wp9" / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parents[1] / "wp8" / "scripts"))
import a3_analyze as A                                        # noqa: E402
import a3_run as R                                            # noqa: E402
import wp9_fit_lib as W                                       # noqa: E402

from backbone.hssm import config as C                         # noqa: E402

USER = 206                                                    # split parent 2, child days in all three recent windows (found offline; see the test below)


def _record(job, **kw):
    return dict(job=job, ok=True, **kw)


def _job(kind, stream, end, K, user=USER):
    return dict(user=user, stream=stream, kchange=None if stream == "none" else stream, stratum="wp8", drift=0.0, fitseed=user, n_init=10, max_iter=60, kind=kind,
                start=0 if kind == "anchor" else end - 45, end=end, K=K)


def fabricate(user=USER, bic_best=None):
    """Fits of the three streams of one user as a perfect fitter would give them: the anchor = planted labels of [0, 90); in every recent window the fit at the planted K has the lowest BIC when
    `bic_best` says so (stream -> best K), otherwise K = 3 does."""
    recs = []
    for stream in ("split", "merge", "none"):
        st = W.stream_for(dict(user=user, kchange=None if stream == "none" else stream, stratum="wp8", drift=0.0))
        tr = st.truth["user"]; lab = st.labels
        w0 = st.window(0, 90)
        recs.append(_record(_job("anchor", stream, 90, 3, user), accepted=True, bic=1.0, labels_fit=W.labels_to_str(lab[:90]), raw_means=tr["means"].tolist(), scaler_mean=w0["scaler_mean"].tolist(),
                            scaler_std=w0["scaler_std"].tolist(), labels_true=W.labels_to_str(lab[:90])))
        best = (bic_best or {}).get(stream, 3)
        for t in A.REFIT_ENDS:
            w = st.window(t - 45, t); l = lab[t - 45:t]
            for K in (2, 3, 4):
                if K == 4:
                    lf = l.copy()
                elif K == 3:
                    lf = np.minimum(l, 2)
                else:
                    a, b = tr["merge_pair"]
                    c = ({0, 1, 2} - {a, b}).pop()
                    lf = np.where(np.minimum(l, 2) == c, 1, 0)
                means = np.vstack([tr["means"], tr["means"][tr["split_parent"]]])[:K]
                bic = 100.0 if K == best else 100.0 + 30.0 + 10.0 * abs(K - best)
                recs.append(_record(_job("refit", stream, t, K, user), accepted=True, bic=bic, labels_fit=W.labels_to_str(lf), raw_means=means.tolist(),
                                    scaler_mean=w["scaler_mean"].tolist(), scaler_std=w["scaler_std"].tolist(), labels_true=W.labels_to_str(l)))
    return recs


def test_the_chosen_user_really_has_child_days_in_every_recent_window():
    st = W.stream_for(dict(user=USER, kchange="split", stratum="wp8", drift=0.0))
    assert all(int((st.labels[t - 45:t] == 3).sum()) >= 8 for t in A.REFIT_ENDS) and st.truth["user"]["split_parent"] == 2


# ----------------------------------------------------------------------------------------------------------------- pure helpers
def test_pick_delta_takes_the_smallest_that_meets_the_cap():
    fa = {0: (5, 12), 3: (3, 12), 6: (2, 12), 10: (1, 12), 20: (0, 12), 40: (0, 12)}
    assert A.pick_delta(fa) == (10, True)                                         # 1/12 = 8.3% <= 10%; 2/12 = 16.7% is not
    assert A.pick_delta({d: (0, 12) for d in A.GRID}) == (0, True)


def test_pick_delta_without_a_passing_level_takes_the_largest_and_says_so():
    fa = {d: (4, 12) for d in A.GRID}
    assert A.pick_delta(fa) == (40, False)


def test_pick_delta_never_passes_on_an_empty_control_set():
    assert A.pick_delta({d: (0, 0) for d in A.GRID}) == (40, False)


def test_the_cap_is_inclusive():
    assert A.pick_delta({d: (1, 10) for d in A.GRID}) == (0, True)               # exactly 10%
    assert A.pick_delta({d: (2, 19) for d in A.GRID}) == (40, False)             # 10.5%


def test_split_candidates_are_right_only_when_they_name_the_planted_parent():
    mp = [1, 2, 0]                                                                # anchor raw regime 0 is planted 1, raw 1 is planted 2, raw 2 is planted 0
    assert A.split_correct([dict(kind="split", old_raw_regime=1)], mp, 2)
    assert not A.split_correct([dict(kind="split", old_raw_regime=0)], mp, 2)
    assert not A.split_correct([dict(kind="merge", old_raw_regimes=[0, 1])], mp, 2)
    assert not A.split_correct([], mp, 2)
    assert A.split_correct([dict(kind="split", old_raw_regime=0), dict(kind="split", old_raw_regime=1)], mp, 2)      # any listed candidate


def test_merge_candidates_need_the_planted_pair_exactly():
    mp = [1, 2, 0]
    assert A.merge_correct([dict(kind="merge", old_raw_regimes=[0, 1])], mp, (2, 1))
    assert not A.merge_correct([dict(kind="merge", old_raw_regimes=[0, 2])], mp, (2, 1))
    assert not A.merge_correct([dict(kind="split", old_raw_regime=0)], mp, (2, 1))


def test_streams_with_needs_the_right_direction():
    ups = [dict(suspected=True, direction="up"), dict(suspected=False, direction="up"), dict(suspected=True, direction="down")]
    assert A.streams_with(ups, "up") and A.streams_with(ups, "down")
    assert not A.streams_with([dict(suspected=False, direction="up")], "up")


# ----------------------------------------------------------------------------------------------------------------- on fabricated fits
@pytest.fixture(scope="module")
def planted():
    return fabricate(bic_best={"split": 4, "merge": 2, "none": 3})


@pytest.fixture(scope="module")
def built(planted):
    idx = A.index(planted)
    streams, skipped = A.build_streams(idx)
    return idx, streams, skipped


def test_streams_are_built_from_usable_anchors_only(planted):
    idx = A.index(planted)
    streams, skipped = A.build_streams(idx)
    assert len(streams) == 3 and not skipped
    bad = [dict(r, accepted=False) if r["job"]["kind"] == "anchor" and r["job"]["stream"] == "none" else r for r in planted]
    s2, sk2 = A.build_streams(A.index(bad))
    assert len(s2) == 2 and sk2 == [(USER, "none")]


def test_a_planted_split_is_detected_up_and_the_candidate_names_the_parent(built):
    _, streams, _ = built
    S_ = next(s for s in streams if s["stream"] == "split")
    a = A.assess(S_, 118, 10.0, 10.0)
    assert a["suspected"] and a["direction"] == "up" and a["k_best"] == 4 and a["margin"] == pytest.approx(40.0)
    assert A.split_correct(a["candidates"], S_["mp"], S_["truth"]["split_parent"])


def test_a_planted_merge_is_detected_down(built):
    _, streams, _ = built
    S_ = next(s for s in streams if s["stream"] == "merge")
    a = A.assess(S_, 118, 10.0, 10.0)
    assert a["suspected"] and a["direction"] == "down" and a["k_best"] == 2


def test_no_change_gives_no_signal_at_any_delta(built):
    _, streams, _ = built
    S_ = next(s for s in streams if s["stream"] == "none")
    assert all(not A.assess(S_, t, d, d)["suspected"] for t in A.REFIT_ENDS for d in (0.0, 10.0))


def test_the_delta_decides(built):
    _, streams, _ = built
    S_ = next(s for s in streams if s["stream"] == "split")
    assert A.assess(S_, 118, 40.0, 99.0)["suspected"] and not A.assess(S_, 118, 40.1, 0.0)["suspected"]


def test_counts_are_per_stream_not_per_refit(built):
    _, streams, _ = built
    c = A.counts(streams, 10.0, 10.0)
    assert c["split"] == dict(n=1, up=1, down=0) and c["merge"] == dict(n=1, up=0, down=1) and c["none"] == dict(n=1, up=0, down=0)


def test_a_fit_that_is_missing_or_not_accepted_gives_no_signal(planted):
    drop = [r for r in planted if not (r["job"]["kind"] == "refit" and r["job"]["stream"] == "split" and r["job"]["end"] == 118 and r["job"]["K"] == 4)]
    streams, _ = A.build_streams(A.index(drop))
    S_ = next(s for s in streams if s["stream"] == "split")
    assert not A.assess(S_, 118, 0.0, 0.0)["suspected"] and A.assess(S_, 134, 0.0, 0.0)["suspected"]
    bad = [dict(r, accepted=False) if (r["job"]["kind"] == "refit" and r["job"]["stream"] == "split" and r["job"]["end"] == 118 and r["job"]["K"] == 3) else r for r in planted]
    streams2, _ = A.build_streams(A.index(bad))
    S2 = next(s for s in streams2 if s["stream"] == "split")
    assert not A.assess(S2, 118, 0.0, 0.0)["suspected"]                            # the reference fit is unavailable


def test_the_lineage_never_serves_a_suspected_or_k_changed_record_as_same_as_before(built):
    _, streams, _ = built
    n_susp, n_kch, bad, old = A.lineage_checks(streams, 10.0, 10.0)
    assert n_susp >= 2 and n_kch >= 10 and bad == 0
    assert any(k == "split" and d == "up" for k, t, d, p, tr in old) and any(k == "merge" and d == "down" for k, t, d, p, tr in old)


def test_dev_report_picks_the_declared_thresholds(planted, tmp_path):
    fits = tmp_path / "f.jsonl"
    fits.write_text("\n".join(json.dumps(r) for r in planted) + "\n")
    A.report_dev(A.load(fits), tmp_path / "o.txt", tmp_path / "o.json")
    j = json.loads((tmp_path / "o.json").read_text())
    assert j["delta_up"] == 0 and j["cap_met_up"] is True and j["delta_down"] == 0 and j["cap_met_down"] is True       # the lone no-change stream never signals
    txt = (tmp_path / "o.txt").read_text()
    assert "CHOSEN" in txt and "delta_up = 0" in txt


def test_eval_report_refuses_to_call_a_run_valid_without_the_freeze_file(planted, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fits = tmp_path / "f.jsonl"
    fits.write_text("\n".join(json.dumps(r) for r in planted) + "\n")
    A.report_eval(A.load(fits), tmp_path / "o.txt", tmp_path / "o.json")
    assert "NOT valid" in (tmp_path / "o.txt").read_text()


def test_eval_report_with_a_matching_freeze_file(planted, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "DESIGN_FREEZE_WP9_A3.txt").write_text(json.dumps(dict(delta_up=C.K_CHANGE_DELTA_UP, delta_down=C.K_CHANGE_DELTA_DOWN)) + "\n")
    fits = tmp_path / "f.jsonl"
    fits.write_text("\n".join(json.dumps(r) for r in planted) + "\n")
    A.report_eval(A.load(fits), tmp_path / "o.txt", tmp_path / "o.json")
    txt = (tmp_path / "o.txt").read_text()
    assert "NOT valid" not in txt and "(a) split detection" in txt and "(d) records" in txt and "must be 0" in txt
    j = json.loads((tmp_path / "o.json").read_text())
    assert j["records"]["bad"] == 0 and j["counts"]["split"]["up"] == 1


# ----------------------------------------------------------------------------------------------------------------- the run script
def test_the_job_list_is_the_declared_protocol():
    jobs = R.jobs_for([9221])
    kinds = {}
    for j in jobs:
        kinds[(j["stream"], j["kind"])] = kinds.get((j["stream"], j["kind"]), 0) + 1
    assert kinds[("split", "anchor")] == 1 and kinds[("split", "refit")] == 9 and kinds[("split", "ksel")] == 1
    assert kinds[("none", "refit")] == 9 and ("none", "ksel") not in kinds
    anchor = next(j for j in jobs if j["kind"] == "anchor")
    assert (anchor["start"], anchor["end"], anchor["K"]) == (0, 90, 3)
    ends = sorted({j["end"] for j in jobs if j["kind"] == "refit"})
    assert ends == [118, 134, 150] and all(j["end"] - j["start"] == 45 for j in jobs if j["kind"] == "refit") and all(j["fitseed"] == j["user"] for j in jobs)
    assert all(j["stratum"] == "wp8" for j in jobs)


def test_the_seed_ranges_are_the_declared_ones_and_do_not_overlap():
    assert list(R.RANGES["dev"]) == list(range(9221, 9233)) and list(R.RANGES["eval"]) == list(range(9241, 9253))
    assert not (set(R.RANGES["dev"]) & set(R.RANGES["eval"]))


def test_the_job_key_ignores_the_restart_budget_only():
    a = R.jobs_for([9221])[0]
    assert R.job_key(a) == R.job_key(dict(a, n_init=20, max_iter=5)) and R.job_key(a) != R.job_key(dict(a, user=9222))


def test_the_frozen_thresholds_are_the_ones_in_the_freeze_file():
    root = Path(__file__).resolve().parents[1]
    frz = json.loads((root / "tests" / "DESIGN_FREEZE_WP9_A3.txt").read_text().splitlines()[0])
    assert (C.K_CHANGE_DELTA_UP, C.K_CHANGE_DELTA_DOWN) == (frz["delta_up"], frz["delta_down"]) == (0.0, 40.0)
