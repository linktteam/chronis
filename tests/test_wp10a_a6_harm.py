"""WP10A A6 part 2 (the harmful-prior control): the constants against the WP9 B1 study, the candidate priors, the exact bound, the validity rule (a), (b), (c), the choice, the evaluation levels E0, E1, E3 and the
refusals of the run script, with a stand-in for `fit_hssm` and for the scoring. No real fit is run here. Mutation check: mutation_wp10a_a6_harm.py."""
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


RUN = _load("wp10a_a6_harm_run_t", SCRIPTS / "a6_harm_run.py")
AN = _load("wp10a_a6_harm_analyze_t", SCRIPTS / "a6_harm_analyze.py")
B1 = _load("wp9_b1_run_for_a6_t", ROOT / "integration_report" / "wp9" / "scripts" / "b1_run.py")


# ------------------------------------------------------------------------------------------------------------------- the study is the WP9 B1 study
def test_the_constants_are_those_of_the_wp9_b1_study():
    assert (RUN.K, RUN.F, RUN.L, RUN.DMAX, RUN.N0) == (B1.K, B1.F, B1.L, B1.DMAX, B1.N0) == (2, 4, 2, 25, 5.0)
    assert RUN.LENGTHS == B1.LENGTHS == (30, 60, 120) and RUN.SEGMENTS == B1.SEGMENTS and RUN.MAX_ITER == B1.MAX_ITER == 200 and RUN.PRIOR_PATH == B1.I3.PRIOR_PATH
    assert RUN.RANGES["dev"] == B1.RANGES["dev"] == list(range(9631, 9643)) and RUN.RANGES["eval"] == list(range(10771, 10795))
    assert RUN.CANDIDATES == {"H1": 4.0, "H2": 8.0, "H3": 16.0} and AN.ORDER == ("H1", "H2", "H3")
    assert (AN.A_MAX, AN.B_MIN, AN.C_MIN, AN.N_DEV, AN.REQUIRED_EVAL) == (-0.25, 6, 4, 36, 12) and AN.CERTIFIED == "CERTIFIED_PRIOR_ANCHORED"
    assert RUN.ELIGIBLE_EM_STATES == ("REJECTED_UPDATE", "MAX_ITER_REACHED")


def test_a_harmful_prior_is_the_true_prior_with_every_intercept_moved_and_nothing_else():
    from backbone.hssm.shrinkage import SharedPrior
    path = ROOT / RUN.PRIOR_PATH
    true = SharedPrior.from_dict(json.load(open(path)))
    for c, shift in RUN.CANDIDATES.items():
        h = RUN.harmful_prior(path, shift)
        assert np.allclose(np.asarray(h.d_obs) - np.asarray(true.d_obs), shift), c
        for k, v in true.params().items():
            if k != "d_obs":
                assert np.array_equal(np.asarray(h.params()[k]), np.asarray(v)), (c, k)
    t = RUN.true_prior(path)
    assert all(np.array_equal(np.asarray(t.params()[k]), np.asarray(v)) for k, v in true.params().items())


def test_the_arms_of_a_range():
    assert RUN.arms_for("dev", ["H1", "H3"]) == ["H1", "H3"] and RUN.arms_for("eval") == ["H", "T"]


# ------------------------------------------------------------------------------------------------------------------- the exact bound and the scores
def test_the_exact_upper_bound():
    assert AN.cp_upper(0, 12) == pytest.approx(1 - 0.05 ** (1 / 12)) and round(AN.cp_upper(0, 12), 3) == 0.221
    assert AN.cp_upper(0, 299) == pytest.approx(1 - 0.05 ** (1 / 299)) and AN.cp_upper(0, 299) < 0.01
    assert AN.cp_upper(12, 12) == 1.0 and AN.cp_upper(15, 12) == 1.0 and math.isnan(AN.cp_upper(0, 0))
    assert AN.cp_upper(1, 12) == pytest.approx(0.3396, abs=1e-3) and AN.cp_upper(1, 12) > AN.cp_upper(0, 12)


def test_a_score_is_nan_when_it_is_not_finite_or_not_flagged_finite():
    assert AN.score_of(dict(mean=-1.5, finite=True)) == -1.5
    for bad in (dict(mean=-1.5, finite=False), dict(mean=float("nan"), finite=True), dict(mean=float("inf"), finite=True), dict(finite=True), dict(mean=None, finite=True), dict(mean=-1.5)):
        assert math.isnan(AN.score_of(bad)), bad


def test_worse_than_the_gaussian_counts_a_non_finite_score_and_not_a_tie():
    assert AN.worse_than_gauss(dict(mean=-2.0, finite=True), -1.0) and not AN.worse_than_gauss(dict(mean=-1.0, finite=True), -1.0) and not AN.worse_than_gauss(dict(mean=-0.5, finite=True), -1.0)
    assert AN.worse_than_gauss(dict(mean=float("nan"), finite=False), -1.0) and AN.worse_than_gauss(dict(finite=False), -1.0)


def jl(path, recs):
    Path(path).write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def test_the_baseline_reads_the_personal_and_gaussian_scores_of_every_seed_and_length(tmp_path):
    p = tmp_path / "b1.jsonl"
    jl(p, [dict(seed=1, length=30, arm="personal", mean=-1.0, finite=True), dict(seed=1, length=30, arm="gauss", mean=-2.0, finite=True), dict(seed=1, length=30, arm="shrink5", mean=-9.0, finite=True),
           dict(seed=2, length=60, arm="personal", mean=float("nan"), finite=False), dict(seed=2, length=60, arm="gauss", mean=-3.0, finite=True)])
    b = AN.baseline_from_b1(str(p))
    assert b[(1, 30)] == dict(personal=-1.0, gauss=-2.0) and math.isnan(b[(2, 60)]["personal"]) and b[(2, 60)]["gauss"] == -3.0 and len(b) == 2


# ------------------------------------------------------------------------------------------------------------------- the development validity rule
def fit(seed, length, mean, eligible=True, finite=True, arm="H1", cert=None, published=False):
    return dict(seed=seed, length=length, arm=arm, mean=mean, finite=finite, eligible=eligible, published_after=published, certification=dict(state=cert or "NOT_CERTIFIED"))


def base_for(n, personal=-1.0, gauss=-1.5):
    return {(s, l): dict(personal=personal, gauss=gauss) for s in range(n) for l in (30,)}


def test_rule_a_is_the_mean_difference_to_the_personal_fit_over_the_finite_pairs():
    base = base_for(36)
    fits = [fit(s, 30, -1.25) for s in range(36)]                                             # exactly -0.25 below the personal fit
    v = AN.validity(fits, base)
    assert v["mean_diff_to_personal"] == pytest.approx(-0.25) and v["rule_a"] is True and v["n_in_mean"] == 36 and v["left_out"] == 0
    assert AN.validity([fit(s, 30, -1.2499) for s in range(36)], base)["rule_a"] is False
    # a non-finite fit is left out of the mean (and counted), a non-finite personal score likewise
    f2 = [fit(s, 30, -1.25) for s in range(34)] + [fit(34, 30, float("nan"), finite=False), fit(35, 30, -5.0)]
    b2 = dict(base); b2[(35, 30)] = dict(personal=float("nan"), gauss=-1.5)
    v = AN.validity(f2, b2)
    assert v["left_out"] == 2 and v["n_in_mean"] == 34 and v["mean_diff_to_personal"] == pytest.approx(-0.25)
    assert AN.validity([], base)["rule_a"] is False and math.isnan(AN.validity([], base)["mean_diff_to_personal"])


def test_rule_b_needs_six_fits_below_the_gaussian_and_a_non_finite_one_counts():
    base = base_for(36)
    fits = [fit(s, 30, -3.0) for s in range(5)] + [fit(s, 30, -1.4) for s in range(5, 36)]
    assert AN.validity(fits, base)["worse_than_gauss"] == 5 and AN.validity(fits, base)["rule_b"] is False
    fits[5] = fit(5, 30, float("nan"), finite=False)
    v = AN.validity(fits, base)
    assert v["worse_than_gauss"] == 6 and v["rule_b"] is True
    tie = [fit(s, 30, -1.5) for s in range(36)]
    assert AN.validity(tie, base)["worse_than_gauss"] == 0                                    # equal to the Gaussian is not worse


def test_rule_c_needs_four_of_the_worse_fits_to_be_eligible_and_an_unusable_pair_is_skipped():
    base = base_for(36)
    fits = [fit(s, 30, -3.0, eligible=(s < 3)) for s in range(8)] + [fit(s, 30, -1.4) for s in range(8, 36)]
    v = AN.validity(fits, base)
    assert v["worse_than_gauss"] == 8 and v["worse_and_eligible"] == 3 and v["rule_c"] is False
    fits[3] = fit(3, 30, -3.0, eligible=True)
    assert AN.validity(fits, base)["worse_and_eligible"] == 4 and AN.validity(fits, base)["rule_c"] is True
    b2 = dict(base); b2[(0, 30)] = dict(personal=-1.0, gauss=float("nan"))
    v = AN.validity(fits, b2)
    assert v["unusable_pairs"] == 1 and v["worse_than_gauss"] == 7


def test_a_candidate_is_verified_only_when_all_three_rules_hold():
    base = base_for(36)
    ok = [fit(s, 30, -3.0) for s in range(36)]
    assert AN.validity(ok, base)["verified"] is True
    mild = [fit(s, 30, -1.1) for s in range(36)]                                              # not worse than the personal fit by 0.25, none below the Gaussian
    v = AN.validity(mild, base)
    assert v["verified"] is False and not v["rule_a"] and not v["rule_b"] and not v["rule_c"]
    half = [fit(s, 30, -3.0, eligible=False) for s in range(36)]
    assert AN.validity(half, base)["verified"] is False


def test_the_first_verified_candidate_in_the_declared_order_is_chosen():
    r = {"H1": dict(verified=False), "H2": dict(verified=True), "H3": dict(verified=True)}
    assert AN.choose(r) == "H2" and AN.choose({"H3": dict(verified=True), "H2": dict(verified=False)}) == "H3"
    assert AN.choose({"H1": dict(verified=False)}) is None and AN.choose({}) is None
    assert AN.choose({"H2": dict(verified=True), "H1": dict(verified=True)}) == "H1"


def _dev_files(tmp_path, h1_mean, h2_mean=None):
    b1 = []
    harm = []
    for s in range(9631, 9643):
        for l in (30, 60, 120):
            b1 += [dict(seed=s, length=l, arm="personal", mean=-1.0, finite=True), dict(seed=s, length=l, arm="gauss", mean=-1.5, finite=True)]
            harm.append(fit(s, l, h1_mean, arm="H1"))
            if h2_mean is not None:
                harm.append(fit(s, l, h2_mean, arm="H2"))
    jl(tmp_path / "b1.jsonl", b1); jl(tmp_path / "harm.jsonl", harm)


def test_the_development_report_freezes_the_first_verified_candidate_and_says_so(tmp_path):
    _dev_files(tmp_path, -1.1, h2_mean=-4.0)
    AN.OUT.clear()
    chosen = AN.dev(str(tmp_path / "b1.jsonl"), str(tmp_path / "harm.jsonl"), str(tmp_path / "o.json"))
    assert chosen == "H2"
    txt = "\n".join(AN.OUT)
    assert "H1: 36 fits" in txt and "H2: 36 fits" in txt and "VERIFIED HARMFUL: False" in txt and "VERIFIED HARMFUL: True" in txt and "FROZEN BY THE DECLARED RULE: H2" in txt
    js = json.load(open(tmp_path / "o.json"))
    assert js["chosen"] == "H2" and js["candidates"]["H2"]["verified"] is True and js["candidates"]["H1"]["verified"] is False


def test_when_no_candidate_is_verified_the_question_stays_unknown(tmp_path):
    _dev_files(tmp_path, -1.1)
    AN.OUT.clear()
    assert AN.dev(str(tmp_path / "b1.jsonl"), str(tmp_path / "harm.jsonl"), str(tmp_path / "o.json")) is None
    assert "NONE VERIFIED" in "\n".join(AN.OUT) and "UNKNOWN" in "\n".join(AN.OUT) and json.load(open(tmp_path / "o.json"))["chosen"] is None


# ------------------------------------------------------------------------------------------------------------------- the evaluation
def eval_file(tmp_path, n_h_elig_worse=12, certified=0, n_t=36, t_cert=2, extra_h=()):
    recs = []
    h_i = 0
    for s in range(10771, 10795):
        for l in (30, 60, 120):
            recs.append(dict(seed=s, length=l, arm="gauss", mean=-1.5, finite=True))
    seeds = [(s, l) for s in range(10771, 10795) for l in (30, 60, 120)]
    for i in range(n_h_elig_worse):
        s, l = seeds[i]
        recs.append(fit(s, l, -4.0, arm="H", eligible=True, cert=("CERTIFIED_PRIOR_ANCHORED" if i < certified else "NOT_CERTIFIED"), published=(i < certified)) | dict(fit_state="MAX_ITER_REACHED"))
    for s, l in seeds[n_h_elig_worse:n_h_elig_worse + 3]:
        recs.append(fit(s, l, -1.0, arm="H", eligible=True) | dict(fit_state="REJECTED_UPDATE"))                        # eligible but not worse than the Gaussian
    for s, l in seeds[n_h_elig_worse + 3:n_h_elig_worse + 5]:
        recs.append(fit(s, l, -4.0, arm="H", eligible=False, published=True) | dict(fit_state="CONVERGED"))             # converged, worse, published (E3)
    for s, l in seeds[n_h_elig_worse + 5:n_h_elig_worse + 6]:
        recs.append(fit(s, l, -4.0, arm="H", eligible=False) | dict(fit_state="FAILED"))
    for i, (s, l) in enumerate(seeds[:n_t]):
        recs.append(fit(s, l, -1.0, arm="T", eligible=True, cert=("CERTIFIED_PRIOR_ANCHORED" if i < t_cert else "NOT_CERTIFIED")))
    jl(tmp_path / "e.jsonl", recs)
    return recs


def run_eval(tmp_path):
    AN.OUT.clear()
    AN.evaluate(str(tmp_path / "e.jsonl"), str(tmp_path / "e.json"))
    return "\n".join(AN.OUT), json.load(open(tmp_path / "e.json"))


def test_e1_is_met_when_enough_harmful_eligible_fits_are_all_refused(tmp_path):
    eval_file(tmp_path, n_h_elig_worse=12, certified=0)
    txt, js = run_eval(tmp_path)
    assert js["eligible_and_worse"] == 12 and js["certified"] == 0 and js["valid"] is True and js["E1"] == "MET" and js["upper"] == pytest.approx(AN.cp_upper(0, 12))
    assert "-> control VALID" in txt and "-> MET" in txt and "22%" in txt
    assert js["E0"] is True and js["T_certified"] == 2 and "informative (the certificate does certify some fits)" in txt
    assert js["worse"] == 12 + 2 + 1 and js["E3_published_and_worse"] == 2 and js["converged"] == 2 and js["other"] == 1


def test_one_certified_harmful_fit_makes_e1_not_met(tmp_path):
    eval_file(tmp_path, n_h_elig_worse=14, certified=1)
    txt, js = run_eval(tmp_path)
    assert js["certified"] == 1 and js["E1"] == "NOT MET" and js["upper"] == pytest.approx(AN.cp_upper(1, 14)) and "-> NOT MET" in txt and js["E3_published_and_worse"] == 1 + 2


def test_eleven_harmful_eligible_fits_are_not_a_valid_control(tmp_path):
    eval_file(tmp_path, n_h_elig_worse=11, certified=0)
    txt, js = run_eval(tmp_path)
    assert js["valid"] is False and js["E1"].startswith("NOT VALID") and "the question stays UNKNOWN" in txt and "control NOT VALID" in txt


def test_a_certificate_that_certifies_nothing_is_not_informative(tmp_path):
    eval_file(tmp_path, n_h_elig_worse=12, certified=0, t_cert=0)
    txt, js = run_eval(tmp_path)
    assert js["E0"] is False and "NOT INFORMATIVE" in txt and js["E1"] == "MET" and "E1 is met but not informative" in txt


def test_a_harmful_fit_without_a_usable_gaussian_is_not_counted(tmp_path):
    recs = eval_file(tmp_path, n_h_elig_worse=12, certified=0)
    recs = [dict(r, mean=float("nan"), finite=False) if (r["arm"] == "gauss" and (r["seed"], r["length"]) == (10771, 30)) else r for r in recs]
    jl(tmp_path / "e.jsonl", recs)
    txt, js = run_eval(tmp_path)
    assert js["eligible_and_worse"] == 11 and js["valid"] is False                          # the pair (10771, 30) holds the first harmful fit: unusable


def test_a_non_finite_harmful_score_is_worse_than_the_gaussian(tmp_path):
    recs = eval_file(tmp_path, n_h_elig_worse=12, certified=0)
    recs = [dict(r, mean=float("nan"), finite=False) if (r["arm"] == "H" and (r["seed"], r["length"]) == (10771, 30)) else r for r in recs]
    jl(tmp_path / "e.jsonl", recs)
    txt, js = run_eval(tmp_path)
    assert js["eligible_and_worse"] == 12


# ------------------------------------------------------------------------------------------------------------------- the run script
class FakeRes:
    def __init__(self, em_state, other, certified, raise_in_score=False):
        self.fit_state = SimpleNamespace(value=em_state)
        self.certification = dict(state="CERTIFIED_PRIOR_ANCHORED" if certified else "NOT_CERTIFIED", reasons=[])
        self._other = other
        self.fit_degraded = False
        self.model = object()

    @property
    def withheld_reason(self):
        cert = (self.certification or {}).get("state") == "CERTIFIED_PRIOR_ANCHORED"
        if self.fit_state.value in ("REJECTED_UPDATE", "MAX_ITER_REACHED") and not cert:
            return "fit_state_not_converged: x"
        return self._other[0] if self._other else None

    def _withheld_reasons(self, ignore_non_convergence=False):
        assert ignore_non_convergence is True
        return list(self._other)


@pytest.fixture
def fake_env(monkeypatch):
    calls = []
    state = dict(em="MAX_ITER_REACHED", other=[], certified=False, raise_=False)

    def fake_fit(**kw):
        calls.append(kw)
        if state["raise_"]:
            raise RuntimeError("boom")
        return FakeRes(state["em"], state["other"], state["certified"])
    import backbone.hssm.fitting as F
    import tests.utils.n18v2 as N
    monkeypatch.setattr(F, "fit_hssm", fake_fit)
    monkeypatch.setattr(N, "score_model", lambda model, user: dict(mean=-2.0, n_test=10, finite=True, ll_full=-1.0, scoring_fallbacks=0))
    return calls, state


def test_the_arm_is_fitted_with_the_wp9_b1_settings_and_the_prior_of_the_arm(fake_env):
    calls, state = fake_env
    priors = dict(H1=RUN.harmful_prior(ROOT / RUN.PRIOR_PATH, 4.0))
    r = RUN.fit_arm("H1", SimpleNamespace(X_train=np.zeros((60, 4))), 9631, priors)
    kw = calls[0]
    assert kw["candidate_ks"] == (2,) and kw["n_initializations"] == 1 and kw["max_iter"] == 200 and kw["random_seed"] == 9631 - 2000 and kw["allow_fast_test_fit"] is True
    assert kw["config"].min_present_sessions == 1 and kw["max_duration"] == 25 and kw["model_type"] == "dual_timescale" and kw["latent_dim"] == 2
    assert kw["shared_prior"] is priors["H1"] and kw["shared_prior_n0"] == 5.0 and kw["certify_prior_anchored"] is True
    assert r["arm"] == "H1" and r["error"] is None and r["fit_state"] == "MAX_ITER_REACHED" and r["eligible"] is True and r["mean"] == -2.0 and r["finite"] is True
    assert r["withheld_em"] is True and r["em_reason"].startswith("fit_state_not_converged") and r["published_after"] is False and r["certification"]["state"] == "NOT_CERTIFIED"


def test_the_em_verdict_is_read_with_the_certificate_switched_off_and_published_after_with_it_on(fake_env):
    calls, state = fake_env
    state["certified"] = True
    r = RUN.fit_arm("T", SimpleNamespace(X_train=np.zeros((60, 4))), 1, dict(T=RUN.true_prior(ROOT / RUN.PRIOR_PATH)))
    assert r["withheld_em"] is True and r["published_after"] is True and r["certification"]["state"] == "CERTIFIED_PRIOR_ANCHORED"


@pytest.mark.parametrize("em,other,eligible", [("MAX_ITER_REACHED", [], True), ("REJECTED_UPDATE", [], True), ("CONVERGED", [], False), ("FAILED", [], False), ("MAX_ITER_REACHED", ["collapsed_regime: x"], False)])
def test_eligibility_is_the_certificates_own_condition(fake_env, em, other, eligible):
    calls, state = fake_env
    state["em"], state["other"] = em, other
    r = RUN.fit_arm("H1", SimpleNamespace(X_train=np.zeros((60, 4))), 1, dict(H1=RUN.true_prior(ROOT / RUN.PRIOR_PATH)))
    assert r["eligible"] is eligible and r["other_reasons"] == [x[:80] for x in other]


def test_a_failing_fit_is_recorded_as_an_error_that_is_not_published_and_not_eligible(fake_env):
    calls, state = fake_env
    state["raise_"] = True
    r = RUN.fit_arm("H1", SimpleNamespace(X_train=np.zeros((60, 4))), 1, dict(H1=None))
    assert r["error"].startswith("RuntimeError: boom") and r["published_after"] is False and r["eligible"] is False and r["finite"] is False and math.isnan(r["mean"])


def test_one_job_gives_the_gaussian_and_one_line_per_arm(fake_env):
    calls, state = fake_env
    job, out = RUN.one((9631, 30, ["H1", "H2"], str(ROOT / RUN.PRIOR_PATH), dict(RUN.CANDIDATES)))
    assert [r["arm"] for r in out] == ["gauss", "H1", "H2"] and all(r["seed"] == 9631 and r["length"] == 30 and r["n_segments"] == 2 for r in out)
    assert np.allclose(np.asarray(calls[0]["shared_prior"].d_obs) - np.asarray(RUN.true_prior(ROOT / RUN.PRIOR_PATH).d_obs), 4.0)
    assert np.allclose(np.asarray(calls[1]["shared_prior"].d_obs) - np.asarray(RUN.true_prior(ROOT / RUN.PRIOR_PATH).d_obs), 8.0)
    job, out = RUN.one((10771, 60, ["H", "T"], str(ROOT / RUN.PRIOR_PATH), dict(H=16.0)))
    assert [r["arm"] for r in out] == ["gauss", "H", "T"]
    base = np.asarray(RUN.true_prior(ROOT / RUN.PRIOR_PATH).d_obs)
    assert np.allclose(np.asarray(calls[-2]["shared_prior"].d_obs) - base, 16.0) and np.array_equal(np.asarray(calls[-1]["shared_prior"].d_obs), base)


def test_the_evaluation_is_refused_without_the_freeze_file_and_with_an_unknown_candidate(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--range", "eval", "--out", str(tmp_path / "o.jsonl")])
    with pytest.raises(SystemExit) as e:
        RUN.main()
    assert "DESIGN_FREEZE_WP10A_A6" in str(e.value)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "DESIGN_FREEZE_WP10A_A6.txt").write_text('{"candidate": "H9"}\n')
    with pytest.raises(SystemExit) as e:
        RUN.main()
    assert "unknown candidate" in str(e.value)


def test_the_development_run_needs_named_candidates(monkeypatch, tmp_path):
    for argv in (["x", "--range", "dev", "--out", str(tmp_path / "o.jsonl")], ["x", "--range", "dev", "--candidates", "H4", "--out", str(tmp_path / "o.jsonl")]):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            RUN.main()


def _fake_job(job):
    seed, length, arms, prior_path, shifts = job
    return job, [dict(seed=seed, length=length, arm=a, certification=dict(state="NOT_CERTIFIED"), wall_s=1.0) for a in ["gauss"] + list(arms)]


_fake_job.__name__ = _fake_job.__qualname__ = "one"
_fake_job.__module__ = RUN.__name__


def test_the_driver_runs_only_the_jobs_that_are_incomplete(monkeypatch, tmp_path):
    out = tmp_path / "h.jsonl"
    jl(out, [dict(seed=9631, length=l, arm=a) for l in (30, 60, 120) for a in ("gauss", "H1")] + [dict(seed=9632, length=30, arm="gauss"), dict(seed=9633, length=30, arm="H1")])      # 9631 complete, 9632/30 and 9633/30 half done
    monkeypatch.setattr(RUN, "one", _fake_job)
    monkeypatch.setattr(sys, "argv", ["x", "--range", "dev", "--candidates", "H1", "--out", str(out), "--workers", "1"])
    RUN.main()
    done = {}
    for l in open(out):
        d = json.loads(l)
        done.setdefault((d["seed"], d["length"]), []).append(d["arm"])
    assert len(done) == 36 and done[(9631, 30)] == ["gauss", "H1"]                                                                           # the complete job was not run again
    assert sorted(done[(9632, 30)]) == ["H1", "gauss", "gauss"] and all("H1" in v for v in done.values())
    assert sorted(done[(9633, 30)]) == ["H1", "H1", "gauss"]                                                                                  # an arm without its Gaussian is incomplete
