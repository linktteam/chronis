"""WP9 B1: the certification of prior-anchored fits (backbone/hssm/cold_start_cert.py; design: tests/PREDECLARED_CRITERIA_WP9.md section 16).
Pure parts and the decision rules on fakes (no EM fit); the wiring into `fit_hssm` on live fits at the end."""
import contextlib
import io
import json
import logging
import math
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import multivariate_normal

import backbone.hssm.cold_start_cert as C
from backbone.hssm.config import (COLD_START_CERT_FRACTIONS, COLD_START_CERT_MIN_BLOCK, COLD_START_CERT_MIN_ROWS, COLD_START_CERT_MIN_TRAIN, COLD_START_CERT_TAU, COLD_START_CERT_Z)

RNG = np.random.default_rng(7)


# ------------------------------------------------------------------------------------------------------------------- the declared constants
def test_the_declared_constants():
    assert COLD_START_CERT_FRACTIONS == (0.50, 0.65, 0.80) and COLD_START_CERT_MIN_ROWS == 24 and COLD_START_CERT_MIN_TRAIN == 12 and COLD_START_CERT_MIN_BLOCK == 3
    assert COLD_START_CERT_TAU == 0.25 and COLD_START_CERT_Z in (1.0, 2.0, 3.0)


# ------------------------------------------------------------------------------------------------------------------- origins
def test_origin_blocks_follow_the_declared_fractions():
    assert C.origin_blocks(100) == ([(50, 65), (65, 80), (80, 100)], None)
    assert C.origin_blocks(60) == ([(30, 39), (39, 48), (48, 60)], None)
    assert C.origin_blocks(24) == ([(12, 16), (16, 20), (20, 24)], None)


def test_too_short_series_are_refused_and_blocks_are_contiguous_and_cover_the_end():
    assert C.origin_blocks(23) == (None, C.R_TOO_FEW_ROWS) and C.origin_blocks(10) == (None, C.R_TOO_FEW_ROWS)
    for T in range(24, 140):
        blocks, why = C.origin_blocks(T)
        assert why is None and blocks[-1][1] == T and all(b[0] == a[1] for a, b in zip(blocks, blocks[1:])) and all(e - o >= 3 for o, e in blocks) and blocks[0][0] >= 12


def test_a_block_that_is_too_short_refuses():
    assert C.origin_blocks(100, fractions=(0.5, 0.65, 0.98)) == (None, C.R_TOO_FEW_ROWS)
    assert C.origin_blocks(100, fractions=(0.05, 0.5, 0.8)) == (None, C.R_TOO_FEW_ROWS)


# ------------------------------------------------------------------------------------------------------------------- HAC standard error
def test_hac_se_of_white_noise_is_close_to_the_naive_error_and_constant_series_has_none():
    d = RNG.normal(size=400)
    assert C.hac_se(d) == pytest.approx(d.std() / math.sqrt(400), rel=0.35)
    assert C.hac_se(np.full(30, 0.7)) == pytest.approx(0.0, abs=1e-12)


def test_hac_se_grows_with_autocorrelation_and_matches_a_hand_computation():
    e = RNG.normal(size=400)
    ar = np.zeros(400)
    for t in range(1, 400):
        ar[t] = 0.9 * ar[t - 1] + e[t]
    assert C.hac_se(ar) > 1.8 * ar.std() / math.sqrt(400)
    d = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0])
    n, m = 6, 2                                                                   # ceil(6 / 4) = 2
    c = d - d.mean()
    lrv = c @ c / n + 2 * (1 - 1 / 3) * (c[:-1] @ c[1:]) / n + 2 * (1 - 2 / 3) * (c[:-2] @ c[2:]) / n
    assert C.hac_se(d) == pytest.approx(math.sqrt(lrv / n))
    assert C.hac_se(d, bandwidth=1) != pytest.approx(C.hac_se(d))


def test_hac_se_refuses_short_or_non_finite_input():
    assert math.isnan(C.hac_se(np.array([1.0, 2.0]))) and math.isnan(C.hac_se(np.array([1.0, np.nan, 2.0, 3.0])))


# ------------------------------------------------------------------------------------------------------------------- single Gaussian and stability
def test_gaussian_heldout_is_the_ddof1_gaussian():
    X = RNG.normal(size=(30, 3)); Y = RNG.normal(size=(5, 3))
    ref = multivariate_normal(mean=X.mean(axis=0), cov=np.cov(X, rowvar=False, ddof=1)).logpdf(Y)
    assert np.allclose(C.gaussian_heldout(X, Y), ref)
    assert C.gaussian_heldout(X, Y[:1]).shape == (1,)


def test_gaussian_heldout_refuses_a_singular_covariance():
    X = np.column_stack([np.arange(20.0), 2 * np.arange(20.0), RNG.normal(size=20)])
    with pytest.raises(ValueError):
        C.gaussian_heldout(X, X[:3])


def test_stability_distance_is_zero_for_equal_means_symmetric_in_regime_order_and_in_sd_units():
    m = np.array([[0.0, 1.0], [3.0, 4.0]]); sd = np.array([1.0, 2.0])
    assert C.stability_distance(m, m, sd) == 0.0
    assert C.stability_distance(m, m[::-1], sd) == 0.0                              # regimes matched, not compared by index
    moved = m + np.array([[0.5, 1.0], [0.5, 1.0]])
    assert C.stability_distance(m, moved, sd) == pytest.approx(0.5)                 # (0.5 / 1, 1.0 / 2) -> rms 0.5
    assert C.stability_distance(m, moved, sd * 2) == pytest.approx(0.25)


def test_stability_distance_is_nan_on_bad_input():
    m = np.zeros((2, 2))
    assert math.isnan(C.stability_distance(m, np.zeros((3, 2)), np.ones(2))) and math.isnan(C.stability_distance(m, m, np.array([1.0, 0.0])))
    assert math.isnan(C.stability_distance(m, np.array([[0.0, np.nan], [0.0, 0.0]]), np.ones(2)))


# ------------------------------------------------------------------------------------------------------------------- the verdict
GOOD = np.full(30, 0.8) + RNG.normal(scale=0.2, size=30)


def test_decide_certifies_a_clear_gain_with_stable_parameters():
    out = C.decide(GOOD, 0.1, z=3.0)
    assert out["state"] == C.STATE_CERTIFIED and out["reasons"] == [] and out["lower"] > 0 and out["n_increments"] == 30


def test_decide_refuses_when_the_lower_bound_is_not_strictly_positive():
    d = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    assert C.decide(d, 0.0)["reasons"] == [C.R_GAIN]
    # the boundary: with se known, mean - z se == 0 does not certify
    d2 = GOOD.copy(); mean, se = d2.mean(), C.hac_se(d2)
    assert C.decide(d2, 0.0, z=mean / se - 1e-6)["state"] == C.STATE_CERTIFIED
    assert C.decide(d2, 0.0, z=mean / se + 1e-6)["reasons"] == [C.R_GAIN]


def test_a_lower_bound_of_exactly_zero_does_not_certify():
    d = np.array([1.0, -1.0] * 5)                                                   # mean exactly 0.0
    out = C.decide(d, 0.0, z=0.0)
    assert out["lower"] == 0.0 and out["state"] == C.STATE_NOT_CERTIFIED and out["reasons"] == [C.R_GAIN]


def test_a_larger_margin_certifies_less():
    d = np.full(30, 0.3) + RNG.normal(scale=0.5, size=30)
    verdicts = [C.decide(d, 0.0, z=z)["state"] == C.STATE_CERTIFIED for z in (1.0, 2.0, 3.0, 6.0)]
    assert verdicts == sorted(verdicts, reverse=True) and verdicts[0] and not verdicts[-1]


def test_decide_refuses_unstable_parameters_at_the_declared_level_inclusive():
    assert C.decide(GOOD, 0.2501)["reasons"] == [C.R_UNSTABLE]
    assert C.decide(GOOD, 0.25)["state"] == C.STATE_CERTIFIED
    assert C.decide(GOOD, 0.25, tau=0.2)["reasons"] == [C.R_UNSTABLE]


def test_decide_is_fail_closed_on_non_finite_or_missing_numbers():
    for bad in (None, np.array([]), np.array([1.0, 2.0]), np.append(GOOD, np.nan), np.append(GOOD, np.inf)):
        out = C.decide(bad, 0.0)
        assert out["state"] == C.STATE_NOT_CERTIFIED and C.R_NONFINITE in out["reasons"]
    for bad in (None, float("nan"), float("inf")):
        out = C.decide(GOOD, bad)
        assert out["state"] == C.STATE_NOT_CERTIFIED and C.R_NONFINITE in out["reasons"]


def test_decide_reports_every_failed_condition():
    d = np.array([1.0, -1.0] * 5)
    out = C.decide(d, 1.0)
    assert out["reasons"] == [C.R_GAIN, C.R_UNSTABLE] and out["state"] == C.STATE_NOT_CERTIFIED


# ------------------------------------------------------------------------------------------------------------------- the procedure on fakes
class Fake:
    """Adapter stand-in: log density increments from a fixed Gaussian, regime means given."""
    degraded = False

    def __init__(self, mean_shift=0.0, means=None, boost=0.0):
        self.mean_shift, self.boost = mean_shift, boost
        self.means = np.array([[0.0, 0.0], [3.0, 3.0]]) if means is None else means

    def predictive_increments(self, X):
        return multivariate_normal(mean=np.full(X.shape[1], self.mean_shift), cov=np.eye(X.shape[1])).logpdf(X) + self.boost

    def regime_observation_means(self):
        return self.means


ELIG = dict(used_shared_prior=True, n0=5.0, em_state="REJECTED_UPDATE", other_reasons=[])
X80 = RNG.normal(size=(80, 2))


def run(fit_fn=lambda Xt: Fake(), full=None, el=ELIG, X=X80, **kw):
    return C.certify_prior_anchored(X, full or Fake(), fit_fn, eligibility=el, **kw)


def test_a_model_that_beats_the_gaussian_and_is_stable_is_certified():
    out = run(lambda Xt: Fake(boost=1.0))
    assert out["state"] == C.STATE_CERTIFIED and out["blocks"] == [[40, 52], [52, 64], [64, 80]] and out["n_increments"] == 40 and out["stability"] == 0.0 and out["mean_gain"] > 0.9


def test_a_model_that_does_not_beat_the_gaussian_is_not_certified():
    out = run(lambda Xt: Fake(mean_shift=3.0))                                   # centred 3 away from the data: much worse than the Gaussian
    assert out["state"] == C.STATE_NOT_CERTIFIED and out["reasons"] == [C.R_GAIN] and out["mean_gain"] < 0


def test_unstable_refit_means_are_refused_even_with_a_gain():
    out = run(lambda Xt: Fake(boost=1.0, means=np.array([[2.0, 2.0], [5.0, 5.0]])))
    assert out["state"] == C.STATE_NOT_CERTIFIED and out["reasons"] == [C.R_UNSTABLE] and out["stability"] > 1.0


def test_only_the_last_origin_fit_decides_stability():
    calls = []

    def fit_fn(Xt):
        calls.append(len(Xt))
        return Fake(boost=1.0, means=np.array([[2.0, 2.0], [5.0, 5.0]]) if len(Xt) < 60 else None)
    out = run(fit_fn)
    assert calls == [40, 52, 64] and out["state"] == C.STATE_CERTIFIED


def test_refits_are_trained_on_rows_before_the_origin_and_scored_after_it():
    seen = []

    class Spy(Fake):
        def predictive_increments(self, X):
            seen.append(len(X))
            return super().predictive_increments(X) + 1.0
    run(lambda Xt: Spy())
    assert seen == [52, 64, 80]                                                  # the filter runs over rows [0, end); the score uses rows [origin, end)


def test_a_refit_that_raises_or_is_degraded_refuses():
    def boom(Xt):
        raise RuntimeError("did not converge")
    out = run(boom)
    assert out["state"] == C.STATE_NOT_CERTIFIED and out["reasons"] == [C.R_REFIT]
    class Deg(Fake):
        degraded = True
    out = run(lambda Xt: Deg(boost=1.0))
    assert out["state"] == C.STATE_NOT_CERTIFIED and out["reasons"] == [C.R_DEGRADED_REFIT]


def test_non_finite_scores_refuse():
    class Bad(Fake):
        def predictive_increments(self, X):
            out = super().predictive_increments(X) + 1.0
            out[-1] = np.nan
            return out
    out = run(lambda Xt: Bad())
    assert out["state"] == C.STATE_NOT_CERTIFIED and C.R_NONFINITE in out["reasons"]
    Xn = X80.copy(); Xn[5, 0] = np.nan
    assert run(X=Xn)["reasons"] == [C.R_NONFINITE]


def test_eligibility_is_checked_before_anything_is_fitted():
    def never(Xt):
        raise AssertionError("must not fit")
    assert run(never, el=dict(ELIG, used_shared_prior=False))["state"] == C.STATE_NOT_APPLICABLE
    assert run(never, el=dict(ELIG, n0=0.0))["reasons"] == [C.R_NOT_PRIOR_ANCHORED]
    assert run(never, el=dict(ELIG, n0=None))["reasons"] == [C.R_NOT_PRIOR_ANCHORED]
    assert run(never, el=dict(ELIG, em_state="CONVERGED")) == dict(run(never, el=dict(ELIG, em_state="CONVERGED")), state=C.STATE_NOT_APPLICABLE, reasons=[C.R_CONVERGED])
    for st in ("FAILED", "RUNNING", "NUMERICAL_FAILURE", "MODEL_SELECTION_CONTESTED", None):
        assert run(never, el=dict(ELIG, em_state=st))["reasons"] == [C.R_STATE]
    out = run(never, el=dict(ELIG, other_reasons=["collapsed_regime: a regime holds 1.0 effective days"]))
    assert out["state"] == C.STATE_NOT_CERTIFIED and out["reasons"] == [C.R_OTHER] and out["other_reasons"]
    assert run(never, X=X80[:23])["reasons"] == [C.R_TOO_FEW_ROWS]


def test_max_iter_reached_is_eligible_too():
    assert run(lambda Xt: Fake(boost=1.0), el=dict(ELIG, em_state="MAX_ITER_REACHED"))["state"] == C.STATE_CERTIFIED


def test_an_unexpected_exception_inside_the_procedure_is_a_refusal_not_a_crash():
    class Broken(Fake):
        def regime_observation_means(self):
            raise RuntimeError("boom")
    out = run(lambda Xt: Broken(boost=1.0), full=Broken())
    assert out["state"] == C.STATE_NOT_CERTIFIED and out["reasons"] == [C.R_ERROR] and "RuntimeError" in out["error"]


def test_the_margin_is_used():
    assert run(lambda Xt: Fake(boost=0.0), z=3.0)["state"] == C.STATE_NOT_CERTIFIED
    assert run(lambda Xt: Fake(boost=2.0), z=3.0)["state"] == C.STATE_CERTIFIED


# ------------------------------------------------------------------------------------------------------------------- inside fit_hssm (live fits, small iteration budgets)
PRIOR = Path(__file__).resolve().parents[1] / "wp7r_evidence" / "prior_main_wp5.json"


def _live(certify, seed=9997, max_iter=3, n_segments=4, **kw):
    from backbone.hssm.config import ColdStartConfig
    from backbone.hssm.fitting import fit_hssm
    from backbone.hssm.shrinkage import SharedPrior
    from data.synthetic_population import PopulationSpec, generate_population_user
    logging.disable(logging.CRITICAL)
    prior = SharedPrior.from_dict(json.load(open(PRIOR)))
    user = generate_population_user(seed, n_segments, 2, PopulationSpec())
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        res = fit_hssm(matrix=user.X_train, candidate_ks=(2,), n_initializations=1, max_iter=max_iter, random_seed=seed - 2000, allow_fast_test_fit=True,
                       config=ColdStartConfig(min_present_sessions=1), max_duration=25, model_type="dual_timescale", latent_dim=2, shared_prior=prior, shared_prior_n0=5.0,
                       certify_prior_anchored=certify, **kw)
    return res, user


@pytest.fixture(scope="module")
def live_default():
    return _live(False)[0]


def test_the_default_call_is_unchanged_no_certification_and_the_em_rule_decides(live_default):
    r = live_default
    assert r.certification == {} and r.certified_prior_anchored is False
    assert str(r.fit_state.value) in ("MAX_ITER_REACHED", "REJECTED_UPDATE") and r.withheld_reason is not None and r.withheld_reason.startswith("fit_state_not_converged")


def test_certification_requires_a_prior_and_a_dual_model():
    from backbone.hssm.fitting import fit_hssm
    X = RNG.normal(size=(40, 2))
    with pytest.raises(ValueError):
        fit_hssm(matrix=X, candidate_ks=(2,), n_initializations=10, max_iter=5, allow_fast_test_fit=True, model_type="dual_timescale", certify_prior_anchored=True)
    with pytest.raises(ValueError):
        fit_hssm(matrix=X, candidate_ks=(2,), n_initializations=10, max_iter=5, allow_fast_test_fit=True, certify_prior_anchored=True)


def test_a_certified_fit_drops_only_the_non_convergence_reason(monkeypatch, live_default):
    import backbone.hssm.cold_start_cert as mod
    seen = {}

    def fake(X, full, refit, eligibility=None, **kw):
        seen.update(el=eligibility, n=len(X), refit=refit, full=full)
        return dict(state=mod.STATE_CERTIFIED, reasons=[], schema_version=1)
    monkeypatch.setattr(mod, "certify_prior_anchored", fake)
    r, user = _live(True)
    assert r.certification["state"] == "CERTIFIED_PRIOR_ANCHORED" and r.certified_prior_anchored is True
    assert str(r.fit_state.value) in ("MAX_ITER_REACHED", "REJECTED_UPDATE")             # EM's own verdict is kept
    assert r.withheld_reason is None and r.insights_withheld is False and r.publishable_outputs() is not None
    assert seen["el"] == dict(used_shared_prior=True, n0=5.0, em_state=str(r.fit_state.value), other_reasons=[]) and seen["n"] == len(user.X_train)
    # everything else still withholds a certified fit
    r.fit_collapsed = True
    assert r.withheld_reason.startswith("collapsed_regime") and "fit_state_not_converged" not in r.withheld_reason and r.publishable_outputs() is None
    r.fit_collapsed = False
    r.fit_duration_degenerate = True
    assert r.withheld_reason.startswith("degenerate_duration")
    r.fit_duration_degenerate = False
    r.fit_degraded = True
    assert "fit_degraded" in r.withheld_reason
    r.fit_degraded = False
    assert r.withheld_reason is None


def test_the_certificate_is_void_unless_the_fit_is_prior_anchored_and_not_converged(monkeypatch):
    import backbone.hssm.cold_start_cert as mod
    from backbone.hssm.model import FitState
    monkeypatch.setattr(mod, "certify_prior_anchored", lambda *a, **k: dict(state=mod.STATE_CERTIFIED, reasons=[]))
    r, _ = _live(True)
    assert r.withheld_reason is None
    for st in (FitState.FAILED, FitState.RUNNING, FitState.NUMERICAL_FAILURE):
        r.fit_state = st
        assert r.certified_prior_anchored is False and r.withheld_reason is not None
    r.fit_state = FitState.MAX_ITER_REACHED
    assert r.certified_prior_anchored is True and r.withheld_reason is None                  # both eligible EM states count
    r.fit_state = FitState.REJECTED_UPDATE
    assert r.certified_prior_anchored is True and r.withheld_reason is None
    r.used_shared_prior = False
    assert r.certified_prior_anchored is False and r.withheld_reason is not None
    r.used_shared_prior = True
    r.certification = dict(state="NOT_CERTIFIED", reasons=["gain_not_demonstrated"])
    assert r.withheld_reason is not None and r.withheld_reason.startswith("fit_state_not_converged")
    r.certification = {}
    assert r.withheld_reason is not None


def test_an_exception_in_the_certification_is_a_refusal_and_does_not_break_the_fit(monkeypatch):
    import backbone.hssm.cold_start_cert as mod

    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(mod, "certify_prior_anchored", boom)
    r, _ = _live(True)
    assert r.certification["state"] == "NOT_CERTIFIED" and r.certification["reasons"] == ["certification_error"] and r.certified_prior_anchored is False
    assert r.withheld_reason is not None and r.fit_state.value in ("MAX_ITER_REACHED", "REJECTED_UPDATE")


def test_the_refit_is_the_production_fit_on_fewer_rows_and_the_scoring_is_the_n18_scoring(monkeypatch):
    import backbone.hssm.cold_start_cert as mod
    from tests.utils.heldout_prediction import predictive_increments
    got = {}
    orig = mod.make_refit_fn

    def spy(**kw):
        got.update(kw)
        return orig(**kw)
    monkeypatch.setattr(mod, "make_refit_fn", spy)
    r, user = _live(True, seed=9996)
    assert got["base_seed"] == (9996 - 2000) + 2 * 1000 and got["max_iter"] == 3 and got["n_regimes"] == 2 and got["max_duration"] == 25 and got["shared_prior_n0"] == 5.0 and got["shared_prior"] is not None
    ad = mod.make_refit_fn(**got)(user.X_train[:40])
    m = ad.model
    assert type(m).__name__ == "ShrinkageDualHSSM" and m.n0 == 5.0 and m.Dmax == 25
    from backbone.hssm.shrinkage import ShrinkageDualHSSM
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        direct = ShrinkageDualHSSM(n_regimes=2, n_features=user.X_train.shape[1], latent_dim=2, max_duration=25, seed=got["base_seed"], shared_prior=got["shared_prior"], n0=5.0)
        direct.fit(user.X_train[:40], n_iter=3)
    assert np.allclose(direct.d_obs, m.d_obs) and np.allclose(direct.C_obs, m.C_obs)           # same class, seed, prior, n0, iterations: the same fit
    lp = ad.predictive_increments(user.X_train)
    ref, _ = predictive_increments(m, user.X_train)
    assert np.allclose(lp, ref)
    assert r.certification["state"] in ("CERTIFIED_PRIOR_ANCHORED", "NOT_CERTIFIED") and r.certification["schema_version"] == 1
    if r.certification["state"] == "NOT_CERTIFIED":
        assert r.withheld_reason.startswith("fit_state_not_converged")


def test_the_pooled_increments_are_model_minus_a_gaussian_fitted_on_the_earlier_rows():
    X = RNG.normal(size=(80, 2)); X[40:] += 3.0                                    # a regime shift after the first origin: a Gaussian that saw the shift would score better
    out = run(lambda Xt: Fake(boost=0.0), X=X)
    ref = []
    for o, e in out["blocks"]:
        ref.append(Fake().predictive_increments(X[:e])[o:e] - multivariate_normal(mean=X[:o].mean(axis=0), cov=np.cov(X[:o], rowvar=False, ddof=1)).logpdf(X[o:e]))
    assert out["mean_gain"] == pytest.approx(float(np.concatenate(ref).mean())) and out["n_increments"] == 40
