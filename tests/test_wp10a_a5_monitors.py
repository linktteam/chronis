"""WP10A A5: `baseline_forward_scores` (production, additive) and the study code of the three candidate monitors and of the frozen development rule (`a5_monitors.py`, `a5_analyze.py`) on fabricated series; one
small fitted model for the forward pass. Mutation check: mutation_wp10a_a5.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from backbone.hssm import monitoring as PM
from backbone.hssm.model import GaussianHSMM

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "integration_report" / "wp10a" / "scripts"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "integration_report" / "wp9" / "scripts"))
MON = _load("wp10a_a5_monitors", SCRIPTS / "a5_monitors.py")
sys.modules["a5_monitors"] = MON                                                       # a5_analyze imports it by this name
AN = _load("wp10a_a5_analyze", SCRIPTS / "a5_analyze.py")


# ----------------------------------------------------------------------------------------------------------------- the forward pass
def _fitted(seed=0):
    rng = np.random.default_rng(seed)
    X = np.concatenate([rng.normal(m, 0.5, (15, 2)) for m in (0.0, 3.0, 0.0, 3.0, 0.0)])
    m = GaussianHSMM(K=2, n_features=2, Dmax=20, seed=seed)
    m.fit(X, n_iter=30)
    return m, X


def test_forward_scores_increments_are_bit_identical_to_the_existing_function():
    m, X = _fitted()
    inc, post = PM.baseline_forward_scores(m, X)
    assert np.array_equal(inc, PM.baseline_predictive_increments(m, X))


def test_filtered_posterior_is_a_probability_and_picks_the_regime_of_each_segment():
    m, X = _fitted()
    inc, post = PM.baseline_forward_scores(m, X)
    assert post.shape == (len(X), m.K) and np.all(post >= 0) and np.allclose(post.sum(axis=1), 1.0)
    dec = post.argmax(axis=1)
    assert len(set(dec[5:12])) == 1 and len(set(dec[20:27])) == 1 and dec[8] != dec[22]             # inside the first (mean 0) and second (mean 3) segment


def test_filtered_posterior_uses_only_days_up_to_t_and_the_model_is_untouched():
    m, X = _fitted()
    before = (m.A.copy(), m.pi.copy(), m.mu.copy(), m.var.copy())
    inc, post = PM.baseline_forward_scores(m, X)
    inc2, post2 = PM.baseline_forward_scores(m, X[:40])
    assert np.array_equal(post[:40], post2) and np.array_equal(inc[:40], inc2)
    assert all(np.array_equal(a, b) for a, b in zip(before, (m.A, m.pi, m.mu, m.var)))


def test_a_missing_day_scores_zero_and_keeps_a_posterior():
    m, X = _fitted()
    miss = X.copy(); miss[30] = np.nan
    inc, post = PM.baseline_forward_scores(m, miss)
    assert inc[30] == pytest.approx(0.0, abs=1e-9) and np.isfinite(post).all() and post[30].sum() == pytest.approx(1.0)


# ----------------------------------------------------------------------------------------------------------------- transition days
def post_of(path, K=3):
    p = np.full((len(path), K), 0.05)
    p[np.arange(len(path)), path] = 0.9
    return p


def test_transition_days_are_the_change_day_and_the_day_after():
    d = MON.transition_days(post_of([0, 0, 0, 1, 1, 1, 1, 2, 2, 0]))
    assert d.tolist() == [False, False, False, True, True, False, False, True, True, True]
    assert MON.transition_days(post_of([1, 1, 1, 1])).tolist() == [False] * 4


def test_the_first_day_is_compared_with_the_row_before_it_when_given():
    p = post_of([1, 1, 1, 1])
    assert MON.transition_days(p).tolist() == [False] * 4
    assert MON.transition_days(p, prev=post_of([0])[0]).tolist() == [True, True, False, False]
    assert MON.transition_days(p, prev=post_of([1])[0]).tolist() == [False] * 4


def test_decoded_is_the_argmax_of_the_posterior():
    assert MON.decoded(np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]])).tolist() == [1, 0]


# ----------------------------------------------------------------------------------------------------------------- MA
def iid(n, seed=0, sd=1.0):
    return np.random.default_rng(seed).normal(0.0, sd, n)


def test_block_bootstrap_se_is_deterministic_positive_and_larger_for_clustered_scores():
    ref_iid = iid(28, 1)
    a, b = MON._lag1_free_se_block(ref_iid), MON._lag1_free_se_block(ref_iid)
    assert a == b and a > 0
    clustered = np.concatenate([np.full(7, 1.0), np.full(7, -1.0), np.full(7, 1.0), np.full(7, -1.0)]) + 0.05 * iid(28, 2)
    shuffled = np.random.default_rng(5).permutation(clustered)
    assert MON._lag1_free_se_block(clustered) > 1.5 * MON._lag1_free_se_block(shuffled)           # same values, blocks keep the clustering


def test_block_bootstrap_se_formula_on_a_known_case():
    ref = np.arange(28, dtype=float)
    se = MON._lag1_free_se_block(ref, draws=4000)
    # blocks of 7: block means are s + 3 for s in 0..21 (uniform on 3..24: var = (22**2 - 1) / 12 = 40.25); recent* = mean of 2, ref* = mean of 4 independent draws: var = 40.25 / 2 + 40.25 / 4
    assert se == pytest.approx(np.sqrt(40.25 / 2 + 40.25 / 4), rel=0.05)


def test_ma_detects_a_clear_drop_and_has_a_low_false_alarm_rate_on_noise():
    y = np.concatenate([iid(60, 3), iid(30, 4) - 4.0])
    ev = MON.monitor_block_bootstrap(y)
    assert ev["status"] == "ok" and ev["events"] and ev["events"][0]["day"] >= 60 and all(e["drop"] > 0 for e in ev["events"])
    alarmed = sum(bool(MON.monitor_block_bootstrap(iid(120, 100 + s))["events"]) for s in range(60))
    assert alarmed <= 3                                                                      # Part 4b: about 1.7% on iid noise (the bootstrap alone gave 16%)


def test_ma_never_alarms_before_the_production_monitor_on_the_same_series():
    for s in range(40):
        y = np.concatenate([iid(70, 200 + s), iid(50, 300 + s) - (1.0 if s % 2 else 0.0)])
        a, b = MON.monitor_block_bootstrap(y)["events"], PM.predictability_monitor(y)["events"]
        if a:
            assert b and b[0]["day"] <= a[0]["day"], s


def test_ma_takes_the_larger_of_the_two_standard_errors(monkeypatch):
    y = np.concatenate([iid(60, 3), iid(30, 4) - 4.0])
    base = MON.monitor_block_bootstrap(y)["events"]
    assert base
    monkeypatch.setattr(MON, "_lag1_free_se_block", lambda ref, window=14, **k: 1e6)           # a huge bootstrap se: the larger one wins, nothing is an event
    assert MON.monitor_block_bootstrap(y)["events"] == []
    monkeypatch.setattr(MON, "_lag1_free_se_block", lambda ref, window=14, **k: 1e-9)          # a tiny one: the production se decides, the events are the production ones
    assert [e["day"] for e in MON.monitor_block_bootstrap(y)["events"]] == [e["day"] for e in PM.predictability_monitor(y)["events"]]


def test_ma_keeps_the_original_day_index_drops_missing_days_and_needs_42_scores():
    y = np.concatenate([iid(60, 3), iid(30, 4) - 4.0])
    gap = y.copy(); gap[[5, 6, 7]] = np.nan
    ev = MON.monitor_block_bootstrap(gap)
    assert ev["n_dropped"] == 3 and ev["events"] and ev["events"][0]["day"] >= 60
    assert MON.monitor_block_bootstrap(iid(41, 0))["status"] == "not_enough_scores"
    assert MON.monitor_block_bootstrap(iid(42, 0))["status"] == "ok"


def test_ma_one_event_per_cooldown_and_a_constant_reference_gives_none():
    y = np.concatenate([iid(60, 3), iid(60, 4) - 4.0])
    days = [e["day"] for e in MON.monitor_block_bootstrap(y)["events"]]
    assert all(b - a >= 14 for a, b in zip(days, days[1:]))
    assert MON.monitor_block_bootstrap(np.ones(80))["events"] == []


def test_ma_uses_the_declared_levels():
    # a drop of 0.5 reference sd is not an event even when it is statistically large (min_drop_sd 0.75); a drop of 1.5 sd is
    base = iid(200, 7)
    small = np.concatenate([base[:100], base[100:] - 0.5])
    big = np.concatenate([base[:100], base[100:] - 1.5])
    assert MON.monitor_block_bootstrap(small)["events"] == [] or all(e["drop_sd"] >= 0.75 for e in MON.monitor_block_bootstrap(small)["events"])
    ev = MON.monitor_block_bootstrap(big)["events"]
    assert ev and all(e["drop_sd"] >= 0.75 and e["z"] >= 3.5 for e in ev)


# ----------------------------------------------------------------------------------------------------------------- MB, MC
def test_mb_removes_the_transition_days_from_the_series():
    y = iid(100, 1)
    post = post_of([0] * 40 + [1] * 30 + [0] * 30)
    out = MON.monitor_transition_excluded(y, post)
    assert out["n_dropped"] == 4 and out["n_scores"] == 96                                          # changes at 40 and 70: days 40, 41, 70, 71
    with pytest.raises(ValueError):
        MON.monitor_transition_excluded(y, post[:50])


def test_mb_equals_the_production_monitor_on_the_masked_series():
    y = np.concatenate([iid(60, 3), iid(40, 4) - 4.0])
    post = post_of([0] * 30 + [1] * 30 + [0] * 40)
    masked = y.copy(); masked[MON.transition_days(post)] = np.nan
    assert MON.monitor_transition_excluded(y, post)["events"] == PM.predictability_monitor(masked)["events"]


def test_regime_stats_use_the_decoded_regime_and_pool_a_small_regime():
    s = np.concatenate([np.full(20, 1.0) + iid(20, 1, 0.1), np.full(20, -2.0) + iid(20, 2, 0.1), np.array([5.0, 5.2, 4.8])])
    post = post_of([0] * 20 + [1] * 20 + [2] * 3)
    m, sd = MON.regime_stats(s, post)
    assert m[0] == pytest.approx(1.0, abs=0.1) and m[1] == pytest.approx(-2.0, abs=0.1) and sd[0] == pytest.approx(0.1, abs=0.05)
    assert m[2] == pytest.approx(s.mean()) and sd[2] == pytest.approx(s.std(ddof=1))                # 3 days < 5: pooled
    s4 = np.concatenate([s[:40], np.array([5.0, 5.2, 4.8, 5.1, 4.9])])
    m4, _ = MON.regime_stats(s4, post_of([0] * 20 + [1] * 20 + [2] * 5))
    assert m4[2] == pytest.approx(5.0, abs=0.1)                                                     # 5 days are enough
    assert MON.MIN_REGIME_DAYS == 5


def test_regime_stats_ignore_missing_training_days():
    s = np.concatenate([np.full(10, 1.0) + iid(10, 1, 0.1), [np.nan] * 5])
    post = post_of([0] * 15, K=2)
    m, sd = MON.regime_stats(s, post)
    assert np.isfinite(m).all() and np.isfinite(sd).all() and m[0] == pytest.approx(1.0, abs=0.1)


def test_standardise_uses_the_regime_of_each_day_and_drops_a_zero_sd_regime():
    e = MON.standardise(np.array([3.0, 5.0, 7.0]), post_of([0, 1, 0]), np.array([1.0, 4.0, 0.0]), np.array([2.0, 0.5, 1.0]))
    assert e.tolist() == [1.0, 2.0, 3.0]
    z = MON.standardise(np.array([3.0, 5.0]), post_of([0, 1]), np.array([1.0, 4.0, 0.0]), np.array([0.0, 0.5, 1.0]))
    assert np.isnan(z[0]) and z[1] == 2.0


def test_mc_removes_a_regime_effect_that_the_raw_rule_alarms_on():
    # a high-scoring regime 0 (+1) followed by a low-scoring regime 1 (-3): the raw series drops by about 13 reference sd when the regime changes; the standardised residual is flat noise
    rng = np.random.default_rng(0)
    train_path, held_path = [0] * 24 + [1] * 24, [0] * 40 + [1] * 20 + [0] * 30
    lvl = np.array([1.0, -3.0])
    tr = lvl[train_path] + rng.normal(0, 0.3, 48)
    held = lvl[held_path] + rng.normal(0, 0.3, len(held_path))
    out = MON.monitor_regime_standardised(held, post_of(held_path), tr, post_of(train_path))
    assert out["status"] == "ok" and out["events"] == []
    assert PM.predictability_monitor(held)["events"], "the raw rule is expected to alarm on the regime swings of this fabricated series"


def test_run_candidate_dispatches_and_refuses_an_unknown_name():
    y = np.concatenate([iid(60, 3), iid(40, 4) - 4.0])
    post = post_of([0] * 100)
    tr, trp = iid(40, 9), post_of([0] * 40)
    for name in ("M0", "MA", "MB", "MC"):
        assert MON.run_candidate(name, y, post, tr, trp, post_of([0])[0])["status"] == "ok"
    assert MON.run_candidate("M0", y, post, tr, trp)["events"] == PM.predictability_monitor(y)["events"]
    assert MON.run_candidate("MA", y, post, tr, trp)["events"] == MON.monitor_block_bootstrap(y)["events"]
    with pytest.raises(ValueError):
        MON.run_candidate("MX", y, post, tr, trp)


def test_the_declared_parameters():
    assert (MON.WINDOW, MON.REFERENCE, MON.COOLDOWN, MON.MIN_DROP_SD, MON.DROP_Z, MON.BLOCK, MON.DRAWS) == (14, 28, 14, 0.75, 3.5, 7, 2000)
    assert (PM.MONITOR_WINDOW_DAYS, PM.MONITOR_REFERENCE_DAYS, PM.MONITOR_COOLDOWN_DAYS) == (14, 28, 14) and (PM.MONITOR_DROP_Z, PM.MONITOR_MIN_DROP_SD) == (3.5, 0.75)
    assert MON.CANDIDATES == ("MA", "MB", "MC")


# ----------------------------------------------------------------------------------------------------------------- the analysis and the frozen rule
def test_the_levels_and_the_order():
    assert (AN.LEVEL_FA, AN.LEVEL_DET, AN.CHANGE_DAY, AN.END_DAY, AN.TRAIN_END, AN.ORDER) == (0.10, 0.70, 135, 179, 90, ("MA", "MB", "MC"))


@pytest.mark.parametrize("fa,ds,dn,expect", [(0.10, 0.70, 0.70, True), (0.101, 0.70, 0.70, False), (0.10, 0.699, 0.70, False), (0.10, 0.70, 0.699, False), (0.0, 1.0, 1.0, True)])
def test_eligibility_boundaries(fa, ds, dn, expect):
    assert AN.eligible(fa, ds, dn) is expect


def test_shortfall_adds_the_three_gaps():
    assert AN.shortfall(0.30, 0.50, 0.60) == pytest.approx(0.20 + 0.20 + 0.10)
    assert AN.shortfall(0.05, 0.80, 0.90) == 0.0
    assert AN.shortfall(0.10, 0.70, 0.70) == 0.0


def S(fa, ds, dn):
    return dict(fa=fa, det_shift=ds, det_sensor=dn)


def test_choose_takes_the_eligible_candidate_with_the_lowest_false_alarm():
    best, how = AN.choose(dict(MA=S(0.08, 0.7, 0.9), MB=S(0.02, 0.7, 0.9), MC=S(0.0, 0.5, 0.9)))
    assert best == "MB" and how.startswith("eligible")                                    # MC has the lowest false alarm but is not eligible


def test_choose_breaks_ties_by_regime_shift_detection_then_order():
    assert AN.choose(dict(MA=S(0.04, 0.72, 0.9), MB=S(0.04, 0.80, 0.9), MC=S(0.04, 0.80, 0.9)))[0] == "MB"
    assert AN.choose(dict(MA=S(0.04, 0.80, 0.9), MB=S(0.04, 0.80, 0.9), MC=S(0.04, 0.80, 0.9)))[0] == "MA"
    assert AN.choose(dict(MA=S(0.2, 0.8, 0.9), MB=S(0.04, 0.80, 0.9), MC=S(0.04, 0.80, 0.9)))[0] == "MB"


def test_choose_with_none_eligible_takes_the_smallest_total_shortfall_and_says_so():
    best, how = AN.choose(dict(MA=S(0.40, 0.5, 0.9), MB=S(0.12, 0.3, 0.9), MC=S(0.20, 0.5, 0.8)))      # shortfalls .50, .42, .30
    assert best == "MC" and how.startswith("NONE ELIGIBLE")
    assert AN.choose(dict(MA=S(0.2, 0.7, 0.7), MB=S(0.2, 0.7, 0.7), MC=S(0.2, 0.7, 0.7)))[0] == "MA"   # equal shortfalls: the order


def test_choose_ignores_m0():
    assert AN.choose(dict(M0=S(0.0, 1.0, 1.0), MA=S(0.5, 0.1, 0.1), MB=S(0.4, 0.1, 0.1), MC=S(0.6, 0.1, 0.1)))[0] == "MB"


def test_tally_none_stream_counts_alarmed_users_and_change_streams_count_events_from_day_135():
    users = [dict(status="ok", event_days=[100]), dict(status="ok", event_days=[]), dict(status="not_enough_scores", event_days=[]), dict(status="ok", event_days=[170])]
    t = AN.tally(users, "none")
    assert (t["n"], t["k"]) == (3, 2)
    u2 = [dict(status="ok", event_days=[100, 138]), dict(status="ok", event_days=[134]), dict(status="ok", event_days=[135]), dict(status="ok", event_days=[179]), dict(status="ok", event_days=[180]), dict(status="ok", event_days=[])]
    t = AN.tally(u2, "regime_shift")
    assert (t["n"], t["k"], t["pre_change_events"]) == (6, 3, 2)                                     # detected: 138, 135, 179; events before 135: 100 and 134
    assert t["latency_median"] == 3                                                                   # latencies 0, 3, 44 -> median 3


def test_series_of_slices_at_day_90_masks_missing_days_and_returns_the_row_before():
    inc = np.arange(180, dtype=float)
    post = np.tile(np.arange(180, dtype=float)[:, None], (1, 3))
    miss = np.zeros(180, bool); miss[95] = miss[10] = True
    rec = dict(streams=dict(none=dict(inc=inc.tolist(), post=post.tolist(), miss=miss.tolist())))
    sc, p, prev, tr, trp = AN.series_of(rec, "none")
    assert len(sc) == 90 and sc[0] == 90.0 and np.isnan(sc[5]) and sc[6] == 96.0
    assert p[0, 0] == 90.0 and prev[0] == 89.0 and len(tr) == 90 and np.isnan(tr[10]) and trp[-1, 0] == 89.0


def test_the_evaluation_is_refused_without_the_freeze_file_and_on_development_users(tmp_path, monkeypatch):
    f = tmp_path / "x.jsonl"
    f.write_text(json.dumps(dict(user=10401, ok=True)) + "\n")
    monkeypatch.setattr(AN, "FREEZE", tmp_path / "none.txt")
    with pytest.raises(SystemExit):
        AN.evaluate(str(f), str(tmp_path / "o.json"), False)
    fz = tmp_path / "fz.txt"
    fz.write_text('{"monitor": "MB"}\n')
    monkeypatch.setattr(AN, "FREEZE", fz)
    with pytest.raises(SystemExit):
        AN.evaluate(str(f), str(tmp_path / "o.json"), False)                                          # a development user in the evaluation file
    fz.write_text('{"monitor": "MZ"}\n')
    with pytest.raises(SystemExit):
        AN.evaluate(str(f), str(tmp_path / "o.json"), True)                                           # unknown monitor


def fake_user(user, shift_day=None, seed=0):
    rng = np.random.default_rng(seed)
    streams = {}
    for name in ("none", "regime_shift", "sensor_change"):
        inc = rng.normal(-1.0, 0.3, 180)
        if name != "none":
            inc[135:] -= 3.0
        post = np.tile(np.array([1.0, 0.0, 0.0]), (180, 1))
        streams[name] = dict(inc=inc.tolist(), post=post.tolist(), miss=[False] * 180)
    return dict(user=user, ok=True, withheld=False, streams=streams)


def test_the_study_pipeline_on_fabricated_users(tmp_path, monkeypatch):
    recs = [fake_user(10401 + i, seed=i) for i in range(6)]
    ev = AN.evaluate_set(recs, ["M0", "MA", "MB", "MC"])
    for c in ("MA", "MB", "MC"):
        assert ev[c]["none"]["k"] == 0 and ev[c]["regime_shift"]["k"] == 6 and ev[c]["sensor_change"]["k"] == 6, c
    line = AN.verdict_line("MA", ev["MA"])
    assert "false alarm 0/6" in line and "regime shift 6/6" in line and "sensor change 6/6" in line
    f = tmp_path / "dev.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    AN.OUT.clear()
    AN.dev(str(f), str(tmp_path / "d.json"))
    got = json.load(open(tmp_path / "d.json"))
    assert got["chosen"] in AN.ORDER and set(got["tallies"]) == {"M0", "MA", "MB", "MC"}
    fz = tmp_path / "fz.txt"
    fz.write_text('{"monitor": "MB"}\n')
    monkeypatch.setattr(AN, "FREEZE", fz)
    AN.OUT.clear()
    AN.evaluate(str(f), str(tmp_path / "e.json"), True)
    e = json.load(open(tmp_path / "e.json"))
    assert e["monitor"] == "MB" and set(e["tallies"]) == {"M0", "MB"} and e["all_levels_met_on_point_estimates"] is True


# ----------------------------------------------------------------------------------------------------------------- added after the first mutation run
def test_the_bootstrap_stream_is_the_declared_one():
    assert MON._lag1_free_se_block(np.arange(28, dtype=float)) == pytest.approx(5.416182176963957, rel=2e-3)      # default_rng(0), 2000 draws; seed 1 gives 5.557


def test_mc_takes_its_regime_means_from_the_training_days_and_so_sees_a_level_shift():
    rng = np.random.default_rng(0)
    train_path, held_path = [0] * 24 + [1] * 24, [0] * 40 + [1] * 50
    lvl = np.array([1.0, -3.0])
    tr = lvl[train_path] + rng.normal(0, 0.3, 48)
    held = lvl[held_path] + rng.normal(0, 0.3, 90)
    held[40:] -= 2.0                                                                                # the low regime is now lower than it was in training
    out = MON.monitor_regime_standardised(held, post_of(held_path), tr, post_of(train_path))
    assert out["events"] and out["events"][0]["day"] >= 40


def test_mc_run_candidate_uses_the_training_arguments():
    rng = np.random.default_rng(1)
    path = [0] * 90
    held = rng.normal(0, 0.3, 90)
    trp = post_of([0] * 40)
    a = MON.run_candidate("MC", held, post_of(path), rng.normal(0, 0.3, 40), trp)
    b = MON.run_candidate("MC", held, post_of(path), rng.normal(5, 0.3, 40), trp)                    # a training level of 5: every held-out residual is about -17 sd
    assert a["events"] == [] and b["status"] == "ok" and b["n_scores"] == 90


def test_regime_sd_is_the_sample_sd_and_pooled_stats_use_the_same():
    s = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    m, sd = MON.regime_stats(s, post_of([0] * 5 + [1] * 5, K=2))
    assert sd[0] == pytest.approx(np.std([0, 1, 2, 3, 4], ddof=1)) and sd[1] == pytest.approx(np.std([10, 11, 12, 13, 14], ddof=1))
    m2, sd2 = MON.regime_stats(s, post_of([0] * 9 + [1], K=2))                                       # regime 1 has one day: pooled
    assert sd2[1] == pytest.approx(s.std(ddof=1)) and m2[1] == pytest.approx(s.mean())


def test_the_autocorrelation_is_capped_at_0_8_in_the_production_standard_error(monkeypatch):
    monkeypatch.setattr(MON, "_lag1_free_se_block", lambda ref, window=14, **k: 1e-9)                  # the bootstrap se is negligible: the AR(1) se decides
    ref = np.linspace(0.0, 1.0, 28)                                                                      # a ramp: lag-1 autocorrelation about 0.95, clipped to 0.8
    s0 = ref.std(ddof=1)
    y = np.concatenate([ref, np.full(14, -0.7)])                                                         # drop of 3.9 s0: z = 4.0 with the cap 0.8 (a cap of 0.9 would give 2.8)
    ev = MON.monitor_block_bootstrap(y)["events"]
    assert len(ev) == 1 and ev[0]["z"] == pytest.approx(4.01, abs=0.02) and ev[0]["drop_sd"] == pytest.approx((ref.mean() + 0.7) / s0)
    assert MON.monitor_block_bootstrap(np.concatenate([ref, np.full(14, -0.45)]))["events"] == []      # 3.1 s0: below z = 3.5


def test_run_candidate_mb_passes_the_row_before_so_that_the_first_day_can_be_a_transition():
    y = iid(60, 1)
    post = post_of([1] * 60)
    out_with = MON.run_candidate("MB", y, post, iid(40, 2), post_of([0] * 40), post_of([0])[0])          # the last training day was regime 0, the first held-out day is regime 1: days 0 and 1 are dropped
    out_without = MON.run_candidate("MB", y, post, iid(40, 2), post_of([0] * 40))
    assert out_with["n_dropped"] == 2 and out_without["n_dropped"] == 0
