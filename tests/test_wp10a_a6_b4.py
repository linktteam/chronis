"""WP10A A6 part 1 (B4): the design of `a6_b4_run.py` (seeds, conditions, what each user is told), the generator copy against the original, the prior edits, the coverage measure, the per-user flow with a stand-in
for `fit_hssm`, and the analysis `a6_b4_analyze.py` (user-level bootstrap, bands, comparison with C0). No real fit is run here. Mutation check: mutation_wp10a_a6_b4.py."""
from __future__ import annotations

import importlib.util
import json
import math
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


RUN = _load("wp10a_a6_b4_run_t", SCRIPTS / "a6_b4_run.py")
AN = _load("wp10a_a6_b4_analyze_t", SCRIPTS / "a6_b4_analyze.py")


# ------------------------------------------------------------------------------------------------------------------- the design
def test_the_seeds_and_the_conditions_are_the_declared_ones():
    assert RUN.SEEDS == list(range(10701, 10765)) and RUN.PER_CONDITION == 16 and RUN.CONDITIONS == ("C0", "C1", "C2", "C3")
    assert (RUN.K, RUN.F, RUN.L, RUN.DMAX, RUN.N0, RUN.N_TRAIN_SEG, RUN.N_TEST_SEG) == (2, 4, 2, 25, 5.0, 6, 2)
    by = {}
    for s in RUN.SEEDS:
        j = RUN.job_for(s)
        by.setdefault((j["condition"], j["variant"]), []).append(s)
    assert {k: len(v) for k, v in by.items()} == {("C0", "favourable"): 16, ("C1", "off_centre_1sd"): 8, ("C1", "off_centre_2sd"): 8, ("C2", "atypical_3x"): 16,
                                                  ("C3", "drop_feature"): 4, ("C3", "dmax_8"): 4, ("C3", "dmax_30"): 4, ("C3", "k_too_small"): 4}
    assert by[("C0", "favourable")] == list(range(10701, 10717)) and by[("C2", "atypical_3x")] == list(range(10733, 10749))
    assert by[("C1", "off_centre_1sd")] == list(range(10717, 10725)) and by[("C1", "off_centre_2sd")] == list(range(10725, 10733))
    assert by[("C3", "drop_feature")] == list(range(10749, 10753)) and by[("C3", "k_too_small")] == list(range(10761, 10765))


def test_each_user_is_told_exactly_what_its_condition_changes():
    base = dict(prior_shift_sd=0.0, spread=1.0, true_K=2, drop_feature=None, dmax=25)
    for s in (10701, 10716):
        assert RUN.job_for(s) == dict(seed=s, condition="C0", variant="favourable", **base)
    assert RUN.job_for(10717)["prior_shift_sd"] == 1.0 and RUN.job_for(10724)["prior_shift_sd"] == 1.0 and RUN.job_for(10725)["prior_shift_sd"] == 2.0 and RUN.job_for(10732)["prior_shift_sd"] == 2.0
    assert RUN.job_for(10733)["spread"] == 3.0 and RUN.job_for(10748)["spread"] == 3.0 and RUN.job_for(10732)["spread"] == 1.0
    j = RUN.job_for(10749)
    assert j["drop_feature"] == 3 and j["dmax"] == 25 and j["true_K"] == 2
    assert RUN.job_for(10753)["dmax"] == 8 and RUN.job_for(10756)["dmax"] == 8 and RUN.job_for(10757)["dmax"] == 30 and RUN.job_for(10760)["dmax"] == 30
    j = RUN.job_for(10761)
    assert j["true_K"] == 3 and j["dmax"] == 25 and j["drop_feature"] is None
    for s in (10700, 10765, 0):
        with pytest.raises(ValueError):
            RUN.job_for(s)


# ------------------------------------------------------------------------------------------------------------------- the generator copy and the prior edits
@pytest.mark.parametrize("seed,spread,K", [(10701, 1.0, 2), (10733, 3.0, 2), (10761, 1.0, 3), (5, 1.0, 2)])
def test_the_generator_copy_gives_bit_identical_data_and_the_true_states(seed, spread, K):
    from data.synthetic_population import PopulationSpec, generate_population_user
    spec = PopulationSpec(n_regimes=K)
    ref = generate_population_user(seed, 6, 2, spec, spread_factor=spread)
    mine, m_tr, m_te = RUN.population_user_with_states(seed, 6, 2, spec, spread_factor=spread)
    assert np.array_equal(ref.X_train, mine.X_train) and np.array_equal(ref.X_test, mine.X_test)
    assert np.array_equal(ref.p_train, mine.p_train) and np.array_equal(ref.p_test, mine.p_test) and ref.spread_factor == mine.spread_factor
    assert m_tr.shape == (len(ref.X_train), 2) and m_te.shape == (len(ref.X_test), 2) and np.all(np.isfinite(m_tr)) and np.all(np.isfinite(m_te))
    assert set(np.unique(mine.p_train)) <= set(range(K))


def test_the_states_returned_are_the_ones_the_observations_were_made_from():
    from data.synthetic_population import PopulationSpec
    user, m_tr, m_te = RUN.population_user_with_states(10701, 6, 2, PopulationSpec())
    C, d = user.truth["C"], user.truth["d"]
    resid = user.X_train - (m_tr @ C.T + np.stack([d[k] for k in user.p_train]))
    assert abs(resid.mean()) < 0.1 and 0.05 < resid.std() < 1.0                                 # what is left is the observation noise only (sd sqrt(r))
    assert np.isclose(resid.std(), math.sqrt(user.truth["r"]), rtol=0.25)
    resid_te = user.X_test - (m_te @ C.T + np.stack([d[k] for k in user.p_test]))
    assert np.isclose(resid_te.std(), math.sqrt(user.truth["r"]), rtol=0.35)


def prior():
    from backbone.hssm.shrinkage import SharedPrior
    return SharedPrior.from_dict(json.load(open(ROOT / RUN.PRIOR_PATH)))


def test_the_off_centre_prior_moves_every_intercept_by_sds_of_the_population_and_nothing_else():
    p = prior()
    before = {k: np.array(v) for k, v in p.params().items()}
    for sd in (1.0, 2.0):
        q = RUN.shifted_prior(p, sd)
        assert np.allclose(np.asarray(q.d_obs) - np.asarray(p.d_obs), sd * 0.6)
        for k, v in p.params().items():
            if k != "d_obs":
                assert np.array_equal(np.asarray(q.params()[k]), v), k
    assert all(np.array_equal(before[k], v) for k, v in p.params().items())                    # the original is untouched
    assert RUN.SD_INTERCEPT == 0.6


def test_dropping_a_feature_cuts_the_feature_axis_of_the_prior_only():
    p = prior()
    q = RUN.drop_feature_prior(p, 3)
    assert q.F == 3 and np.asarray(q.d_obs).shape == (2, 3) and np.asarray(q.C_obs).shape == (2, 3, 2) and np.asarray(q.R_obs).shape == (2, 3)
    assert np.array_equal(np.asarray(q.d_obs), np.asarray(p.d_obs)[:, :3]) and np.array_equal(np.asarray(q.C_obs), np.asarray(p.C_obs)[:, :3])
    assert np.array_equal(np.asarray(q.A_fast), np.asarray(p.A_fast)) and p.F == 4
    r = RUN.drop_feature_prior(p, 1)
    assert np.array_equal(np.asarray(r.d_obs), np.asarray(p.d_obs)[:, [0, 2, 3]])


# ------------------------------------------------------------------------------------------------------------------- the coverage measure
def test_coverage_is_the_share_of_rows_within_one_and_two_sd_after_the_affine_map():
    rng = np.random.default_rng(0)
    T = 40000
    true_p = np.repeat([0, 1, 0, 1], T // 4)
    true_m = rng.normal(size=(T, 2))
    std = np.full((T, 2), 0.5)
    m_hat = 2.0 * true_m + 0.3 + 0.5 * rng.normal(size=(T, 2))                              # an affine image of the truth plus noise of exactly one posterior sd
    c = RUN.coverage_on(m_hat, std * 1.0, true_m, true_p, slice(None))
    # the map is fitted in sample: the residual of the regression is what the posterior sd has to cover
    assert c["n"] == T and np.allclose(c["c1"], RUN.NOMINAL1, atol=0.01) and np.allclose(c["c2"], RUN.NOMINAL2, atol=0.01) and np.allclose(c["z2"], 1.0, atol=0.05)
    wide = RUN.coverage_on(m_hat, std * 2.0, true_m, true_p, slice(None))
    narrow = RUN.coverage_on(m_hat, std * 0.5, true_m, true_p, slice(None))
    assert all(w > c_ > n_ for w, c_, n_ in zip(wide["c2"], c["c2"], narrow["c2"])) and np.allclose(wide["z2"], 0.25, atol=0.03)


def test_the_rows_selected_are_the_only_rows_counted_and_the_map_uses_all_rows():
    rng = np.random.default_rng(1)
    T = 600
    true_p = np.repeat([0, 1, 0, 1, 0, 1], 100)
    true_m = rng.normal(size=(T, 2))
    m_hat = true_m + rng.normal(scale=0.3, size=(T, 2))
    std = np.full((T, 2), 0.3)
    last = RUN.coverage_on(m_hat, std, true_m, true_p, slice(450, None))
    first = RUN.coverage_on(m_hat, std, true_m, true_p, slice(0, 450))
    assert last["n"] == 150 and first["n"] == 450
    from tests.test_dual_model_latent_calibration import per_regime_affine_prediction
    ref = per_regime_affine_prediction(true_m, m_hat, true_p, min_points=9)
    z = (m_hat - ref) / std
    assert np.allclose(last["c2"], np.mean(np.abs(z[450:]) <= 2.0, axis=0)) and np.allclose(last["c1"], np.mean(np.abs(z[450:]) <= 1.0, axis=0))
    assert np.allclose(first["z2"], np.mean(z[:450] ** 2, axis=0))


# ------------------------------------------------------------------------------------------------------------------- the per-user flow with a stand-in for fit_hssm
class FakeModel:
    def estimate_states(self, X):
        T = len(X)
        rng = np.random.default_rng(0)
        return rng.normal(size=(T, 2)), np.full((T, 2), 1.0), np.zeros(T, dtype=int), np.zeros((T, 2))


@pytest.fixture
def fake_fit(monkeypatch):
    calls = []
    state = dict(withheld=False, certified=False, raise_=False)

    def fit(**kw):
        calls.append(kw)
        if state["raise_"]:
            raise RuntimeError("boom")
        return SimpleNamespace(insights_withheld=state["withheld"], withheld_reason=("fit_state_not_converged: x" if state["withheld"] else None), fit_state=SimpleNamespace(value="MAX_ITER_REACHED"),
                               certification=dict(state="NOT_CERTIFIED", reasons=["gain_not_demonstrated"]), certified_prior_anchored=state["certified"], model=FakeModel())
    import backbone.hssm.fitting as F
    monkeypatch.setattr(F, "fit_hssm", fit)
    return calls, state


def test_the_production_fit_is_called_with_the_declared_arguments_and_nothing_that_relaxes_a_rule(fake_fit):
    calls, state = fake_fit
    r = RUN.one(10701)
    kw = calls[0]
    assert set(kw) == {"matrix", "candidate_ks", "random_seed", "max_duration", "model_type", "latent_dim", "shared_prior", "shared_prior_n0", "certify_prior_anchored"}
    assert kw["candidate_ks"] == (2,) and kw["random_seed"] == 10701 and kw["max_duration"] == 25 and kw["model_type"] == "dual_timescale" and kw["latent_dim"] == 2
    assert kw["shared_prior_n0"] == 5.0 and kw["certify_prior_anchored"] is True and kw["matrix"].shape[1] == 4
    assert r["ok"] and r["condition"] == "C0" and r["variant"] == "favourable" and r["withheld"] is False and r["fit_state"] == "MAX_ITER_REACHED" and r["certification_state"] == "NOT_CERTIFIED"
    assert r["certification_reasons"] == ["gain_not_demonstrated"] and r["certified"] is False and r["n_train"] == len(kw["matrix"]) and r["n_test"] > 0
    assert r["heldout"]["n"] == r["n_test"] and r["train"]["n"] == r["n_train"] and len(r["heldout"]["c2"]) == 2 and r["std_mean"] == pytest.approx(1.0) and r["cpu_s"] >= 0


def test_the_conditions_change_the_prior_the_matrix_and_the_settings_as_declared(fake_fit):
    calls, state = fake_fit
    base = prior()
    RUN.one(10717)                                                                           # off-centre by 1 sd
    assert np.allclose(np.asarray(calls[-1]["shared_prior"].d_obs) - np.asarray(base.d_obs), 0.6)
    RUN.one(10725)                                                                           # off-centre by 2 sd
    assert np.allclose(np.asarray(calls[-1]["shared_prior"].d_obs) - np.asarray(base.d_obs), 1.2)
    RUN.one(10701)
    assert np.array_equal(np.asarray(calls[-1]["shared_prior"].d_obs), np.asarray(base.d_obs))
    RUN.one(10749)                                                                           # one feature dropped from the data and from the prior
    assert calls[-1]["matrix"].shape[1] == 3 and calls[-1]["shared_prior"].F == 3
    RUN.one(10753)
    assert calls[-1]["max_duration"] == 8 and calls[-1]["matrix"].shape[1] == 4
    RUN.one(10757)
    assert calls[-1]["max_duration"] == 30
    RUN.one(10761)                                                                           # K too small: three true regimes, K = 2 fitted
    assert calls[-1]["candidate_ks"] == (2,)
    from data.synthetic_population import PopulationSpec, generate_population_user
    u3 = generate_population_user(10761, 6, 2, PopulationSpec(n_regimes=3))
    assert np.array_equal(calls[-1]["matrix"], u3.X_train)
    RUN.one(10733)                                                                           # atypical: spread 3
    u_at = generate_population_user(10733, 6, 2, PopulationSpec(), spread_factor=3.0)
    assert np.array_equal(calls[-1]["matrix"], u_at.X_train)


def test_a_certified_fit_is_recorded_as_certified(fake_fit):
    calls, state = fake_fit
    state["certified"] = True
    assert RUN.one(10704)["certified"] is True


def test_a_withheld_fit_is_recorded_as_withheld_and_a_failure_as_an_error(fake_fit):
    calls, state = fake_fit
    state["withheld"] = True
    r = RUN.one(10702)
    assert r["ok"] and r["withheld"] is True and r["withheld_reason"].startswith("fit_state_not_converged")
    state["raise_"] = True
    r = RUN.one(10703)
    assert r["ok"] is False and r["error"].startswith("RuntimeError: boom") and "heldout" not in r


def test_read_done_keeps_successful_users_only(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps(dict(seed=1, ok=True)) + "\nnot json\n" + json.dumps(dict(seed=2, ok=False)) + "\n")
    assert list(RUN.read_done(str(p))) == [1] and RUN.read_done(str(tmp_path / "none")) == {}


def _fake_one(seed):
    return dict(RUN.job_for(seed), ok=True, error=None, withheld=False, cpu_s=1.0, wall_s=1.0)


_fake_one.__name__ = _fake_one.__qualname__ = "one"
_fake_one.__module__ = RUN.__name__


def test_the_driver_runs_the_requested_users_once_and_resumes(monkeypatch, tmp_path):
    out = tmp_path / "b4.jsonl"
    monkeypatch.setattr(RUN, "one", _fake_one)
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(out), "--workers", "1", "--seeds", "10701,10717,10733", "--limit", "2"])
    RUN.main()
    assert sorted(json.loads(l)["seed"] for l in open(out)) == [10701, 10717]
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(out), "--workers", "1", "--seeds", "10701,10717,10733"])
    RUN.main()
    assert sorted(json.loads(l)["seed"] for l in open(out)) == [10701, 10717, 10733]


# ------------------------------------------------------------------------------------------------------------------- the analysis
def brec(seed, cond, variant="v", withheld=False, c1=(0.68, 0.68), c2=(0.95, 0.95), z2=(1.0, 1.0), ok=True):
    return dict(seed=seed, condition=cond, variant=variant, ok=ok, withheld=withheld, withheld_reason=("fit_state_not_converged: x" if withheld else None), certified=False,
                heldout=dict(c1=list(c1), c2=list(c2), z2=list(z2), n=30), train=dict(c1=list(c1), c2=list(c2), z2=list(z2), n=90))


def test_the_nominal_levels_and_constants():
    assert AN.NOMINAL1 == pytest.approx(0.6826894921) and AN.NOMINAL2 == pytest.approx(0.9544997361) and (AN.BAND, AN.WORSE, AN.MIN_READ, AN.DRAWS) == (0.10, 0.10, 6, 2000)
    assert AN.NOMINAL1 == RUN.NOMINAL1 and AN.NOMINAL2 == RUN.NOMINAL2


def test_load_keeps_one_record_per_seed_and_counts_the_failures(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [brec(3, "C0"), dict(brec(1, "C0"), ok=False), brec(2, "C0"), brec(1, "C0")]) + "\n")
    recs, failed = AN.load(str(p))
    assert [r["seed"] for r in recs] == [1, 2, 3] and failed == 1
    p.write_text("\n".join(json.dumps(r) for r in [brec(2, "C0", withheld=True), brec(2, "C0", withheld=False)]) + "\n")
    recs, failed = AN.load(str(p))
    assert len(recs) == 1 and recs[0]["withheld"] is False                                   # a user that was run again: the later line counts


def test_published_means_ok_and_not_withheld():
    assert AN.is_published(brec(1, "C0")) and not AN.is_published(brec(1, "C0", withheld=True)) and not AN.is_published(dict(brec(1, "C0"), ok=False))


def test_the_user_matrix_orders_one_and_two_sd_by_dimension():
    c, z = AN.user_matrix([brec(1, "C0", c1=(0.1, 0.2), c2=(0.3, 0.4), z2=(5.0, 6.0))])
    assert c.tolist() == [[0.1, 0.2, 0.3, 0.4]] and z.tolist() == [[5.0, 6.0]]
    c, _ = AN.user_matrix([brec(1, "C0", c1=(0.1, 0.2), c2=(0.3, 0.4))], key="train")
    assert c.shape == (1, 4)


def test_the_bootstrap_is_deterministic_and_centred_on_the_mean():
    m = np.random.default_rng(3).uniform(size=(30, 4))
    b1, b2 = AN.boot_mean(m, np.random.default_rng(0)), AN.boot_mean(m, np.random.default_rng(0))
    assert np.array_equal(b1, b2) and b1.shape == (2000, 4) and np.allclose(b1.mean(axis=0), m.mean(axis=0), atol=0.02)
    lo, hi = AN.interval(b1)
    assert np.all(lo < m.mean(axis=0)) and np.all(hi > m.mean(axis=0)) and np.all(hi - lo < 0.5)
    assert AN.boot_mean(m, np.random.default_rng(0), draws=10).shape == (10, 4)


def test_in_band_needs_all_four_coverages_within_ten_points():
    n1, n2 = AN.NOMINAL1, AN.NOMINAL2
    assert AN.in_band([n1, n1, n2, n2]) and AN.in_band([n1 + 0.099, n1 - 0.099, n2 - 0.099, n2 + 0.04])
    for i, (v, nom) in enumerate([(n1 + 0.101, n1), (n1 - 0.101, n1), (n2 - 0.101, n2), (n2 - 0.101, n2)]):
        c = [n1, n1, n2, n2]
        c[i] = v
        assert not AN.in_band(c), i


def test_worse_than_c0_means_more_than_ten_points_lower_in_two_sd_coverage_in_either_dimension():
    c0 = [0.6, 0.6, 0.9, 0.9]
    assert not AN.worse_than([0.1, 0.1, 0.8001, 0.8001], c0) and AN.worse_than([0.6, 0.6, 0.7999, 0.9], c0) and AN.worse_than([0.6, 0.6, 0.9, 0.7999], c0)
    assert not AN.worse_than([0.6, 0.6, 0.95, 0.95], c0) and not AN.worse_than(c0, c0)
    assert AN.misses([AN.NOMINAL1 + 0.05, AN.NOMINAL1 - 0.1, AN.NOMINAL2 - 0.2, AN.NOMINAL2]) == pytest.approx([0.05, -0.1, -0.2, 0.0])


def test_cond_stats_counts_published_and_withheld_and_reads_only_enough_published_users():
    recs = [brec(i, "C0", variant="a") for i in range(6)] + [brec(10 + i, "C0", variant="b", withheld=True) for i in range(4)] + [brec(20 + i, "C1") for i in range(5)]
    s = AN.cond_stats(recs, "C0")
    assert (s["ok_users"], s["published"], s["withheld"], s["used"], s["readable"]) == (10, 6, 4, 6, True)
    assert s["variants"] == {"a": dict(users=6, published=6, withheld=0), "b": dict(users=4, published=0, withheld=4)}
    assert np.allclose(s["mean"], [0.68, 0.68, 0.95, 0.95]) and "lo" in s and "hi" in s and s["matrix"].shape == (6, 4)
    s5 = AN.cond_stats(recs, "C1")
    assert s5["used"] == 5 and s5["readable"] is False and "mean" in s5                      # five published users: kept for the record, not read
    sa = AN.cond_stats(recs, "C0", published_only=False)
    assert sa["used"] == 10 and sa["published"] == 6
    s0 = AN.cond_stats([brec(i, "C2", withheld=True) for i in range(3)], "C2")
    assert s0["used"] == 0 and "mean" not in s0 and s0["readable"] is False


def test_the_interval_is_the_percentile_interval_of_the_user_level_bootstrap_with_seed_zero():
    vals = [0.2, 0.35, 0.5, 0.55, 0.61, 0.7, 0.9]
    recs = [brec(i, "C0", c1=(v, v), c2=(v, v)) for i, v in enumerate(vals)]
    s = AN.cond_stats(recs, "C0")
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(vals), size=(2000, len(vals)))
    means = np.asarray(vals)[idx].mean(axis=1)
    assert s["lo"][0] == pytest.approx(np.percentile(means, 2.5)) and s["hi"][3] == pytest.approx(np.percentile(means, 97.5)) and s["mean"][0] == pytest.approx(np.mean(vals))


def test_the_difference_to_c0_resamples_both_groups_and_is_none_for_a_group_of_one():
    a = AN.cond_stats([brec(i, "C0", c2=(0.9, 0.9)) for i in range(10)], "C0")
    b = AN.cond_stats([brec(i, "C1", c2=(0.7, 0.8)) for i in range(10)], "C1")
    d = AN.delta_vs_c0(b, a)
    assert d["mean"] == pytest.approx([0, 0, -0.2, -0.1]) and d["lo"][2] == pytest.approx(-0.2) and d["hi"][2] == pytest.approx(-0.2)
    one = AN.cond_stats([brec(1, "C1")], "C1")
    assert AN.delta_vs_c0(one, a) is None and AN.delta_vs_c0(a, AN.cond_stats([], "C0")) is None


def test_the_report_states_every_condition_the_bands_the_comparison_and_the_variants(tmp_path):
    recs = ([brec(i, "C0", c1=(0.55, 0.55), c2=(0.80, 0.80)) for i in range(8)] + [brec(20 + i, "C1", variant="off_centre_1sd", c1=(0.4, 0.4), c2=(0.6, 0.6)) for i in range(8)]
            + [brec(40 + i, "C2", variant="atypical_3x", withheld=True) for i in range(5)] + [brec(60 + i, "C3", variant="dmax_8") for i in range(3)] + [brec(70, "C3", variant="dmax_8", withheld=True)])
    p = tmp_path / "f.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    AN.OUT.clear()
    AN.main(str(p), str(tmp_path / "o.txt"), str(tmp_path / "o.json"))
    txt = (tmp_path / "o.txt").read_text()
    for word in ("C0 favourable control: users 8, published 8", "in band", "WORSE than C0", "nothing to read: no published fit", "too few published fits to read", "SECONDARY, all fits", "HEADLINE", "UNKNOWN",
                 "C3/dmax_8: users 4, published 3, withheld 1", "C1 off-centre prior against C0", "N17 stays an expected failure"):
        assert word in txt, word
    assert "in band (+-0.10 of both nominal levels in both dimensions): NO" in txt and "'C2': {'fit_state_not_converged': 5}" in txt and "certified (CERTIFIED_PRIOR_ANCHORED) fits: 0; withheld fits: 6" in txt
    js = json.load(open(tmp_path / "o.json"))
    assert js["users"] == 25 and js["failed"] == 0 and js["published"]["C0"]["readable"] is True and js["published"]["C2"]["used"] == 0 and "matrix" not in js["published"]["C0"]
    assert js["all"]["C2"]["used"] == 5


def test_the_script_needs_three_arguments():
    import subprocess
    assert subprocess.run([sys.executable, str(SCRIPTS / "a6_b4_analyze.py")], capture_output=True, cwd=ROOT).returncode != 0
