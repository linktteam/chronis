"""WP8 Item 4, both engines (owner decision C2) and the raw record (A4). A real baseline fit through `fit_hssm`; dual models with hand-set parameters (a real dual fit takes a minute and is
rarely publishable, WP8 Item 3); a dual model whose regime means equal a baseline fit's, to test an engine switch. Test seeds 8791..8794 are not in any scoring range."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import logging
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from backbone.hssm import lineage as L
from backbone.hssm.dual_model import DualTimescaleHSSM
from backbone.hssm.fitting import HSSMResult, fit_hssm
from backbone.hssm.lineage import ALIGNED, AlignmentConfig, PairThreshold, annotate_result, step, summarize_fit
from backbone.hssm.ordering import LadderConfig
from backbone.hssm.shrinkage import extract_params, permute_regimes, regime_observation_means
from data.synthetic_identity import REGISTRY, generate_identity_stream

CFG = AlignmentConfig(min_separation=0.0, ladder=LadderConfig(n_bootstrap=40))
K, F, LAT, DMAX = 3, 7, 2, 12


def _fit(seed, start, end):
    logging.disable(logging.CRITICAL)
    w = generate_identity_stream(seed, n_days=150).window(start, end)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        r = fit_hssm(matrix=w["matrix"], candidate_ks=(3,), n_initializations=10, max_iter=60, random_seed=seed, user_id="t", dataset_hash="h")
    return r, w


@pytest.fixture(scope="module")
def baseline_fits():
    return _fit(8791, 0, 90), _fit(8791, 0, 120)


def _dual_with_means(z_means, seed, window):
    """A dual model whose regime observation means are `z_means` (C_obs = 0, so the mean is d_obs) and whose other parameters are valid and arbitrary."""
    rng = np.random.default_rng(seed)
    m = DualTimescaleHSSM(n_regimes=K, n_features=F, latent_dim=LAT, max_duration=DMAX, seed=seed)
    m._init_params(window)
    m.pi = np.full(K, 1.0 / K)
    A = np.ones((K, K)) - np.eye(K); m.A_slow = A / A.sum(axis=1, keepdims=True)
    m.dur_mu = rng.uniform(0.8, 2.0, K); m.dur_sigma = rng.uniform(0.3, 0.8, K)
    m.A_fast = np.array([0.5 * np.eye(LAT) for _ in range(K)]); m.b_fast = np.zeros((K, LAT)); m.Q_fast = np.array([0.3 * np.eye(LAT) for _ in range(K)])
    m.m0 = np.zeros((K, LAT)); m.P0 = np.array([np.eye(LAT) for _ in range(K)])
    m.C_obs = np.zeros((K, F, LAT)); m.d_obs = np.array(z_means, dtype=float); m.R_obs = np.full((K, F), 0.5)
    m._is_fitted = True
    return m


def _dual_result(model, window):
    _, _, _, post = model.estimate_states(window)
    return SimpleNamespace(model=model, p_t_posterior=np.asarray(post), is_baseline_model=False)


def test_a_baseline_result_is_summarised_in_raw_units_with_the_stored_scaler(baseline_fits):
    (r, w), _ = baseline_fits
    s = summarize_fit(r, w["raw"], w["scaler_mean"], w["scaler_std"])
    assert s.engine == "baseline" and s.K == 3
    assert np.array_equal(s.raw_means, w["scaler_mean"] + w["scaler_std"] * np.array(r.model.mu))
    assert np.array_equal(s.labels, np.asarray(r.p_t_posterior).argmax(axis=1))
    assert s.exp_durations.shape == (3,) and (s.exp_durations > 0).all() and np.isfinite(s.separation_min)
    assert np.array_equal(s.scaler_mean, w["scaler_mean"]) and np.array_equal(s.scaler_std, w["scaler_std"])


def test_two_refits_of_a_real_baseline_user_go_through_the_state_machine_and_the_names_follow_the_alignment(baseline_fits):
    (r1, w1), (r2, w2) = baseline_fits
    a = summarize_fit(r1, w1["raw"], w1["scaler_mean"], w1["scaler_std"]); n = summarize_fit(r2, w2["raw"], w2["scaler_mean"], w2["scaler_std"])
    rec0, st = step(None, a, CFG, REGISTRY, lineage_id="t:L1")
    rec1, st1 = step(st, n, CFG, REGISTRY)
    assert rec0.continuity_state == "NEW_LINEAGE" and rec1.continuity_state in L.STATES and rec1.fit_index == 1
    # the names are the Hungarian assignment of the raw-unit means divided by the anchor's scale (independent re-computation)
    from itertools import permutations
    scale = w1["scaler_std"]
    best = min(permutations(range(3)), key=lambda q: sum((((a.raw_means[rec0.raw_to_stable.index(i)] - n.raw_means[q[i]]) / scale) ** 2).sum() for i in range(3)))
    expected = [0] * 3
    for i, raw in enumerate(best):
        expected[raw] = i
    assert list(rec1.raw_to_stable) == expected
    assert rec1.margin_diff is not None and rec1.margin_ratio is not None and rec1.expert_review_status == "PENDING"


def test_annotating_a_result_adds_one_field_and_leaves_every_raw_output_bit_identical(baseline_fits):
    (r, w), _ = baseline_fits
    assert HSSMResult.__dataclass_fields__["lineage"].default is None and r.lineage is None            # fit_hssm never fills it
    before = dict(post=np.array(r.p_t_posterior, copy=True), est=np.array(r.p_t_estimate, copy=True), mu=np.array(r.model.mu, copy=True), var=np.array(r.model.var, copy=True),
                  A=np.array(r.model.A, copy=True), dur=np.array(r.model.dur_mu, copy=True), K=r.K, bic=r.BIC, state=r.fit_state, withheld=r.insights_withheld, meta=copy.deepcopy(r.convergence_metadata))
    s = summarize_fit(r, w["raw"], w["scaler_mean"], w["scaler_std"])
    rec, _ = step(None, s, CFG, REGISTRY, lineage_id="t:L1")
    out = annotate_result(r, rec)
    assert out is r and r.lineage == rec.as_dict() and set(L.RECORD_COLUMNS) <= set(r.lineage)
    json.dumps(r.lineage)
    for name, arr in (("post", r.p_t_posterior), ("est", r.p_t_estimate), ("mu", r.model.mu), ("var", r.model.var), ("A", r.model.A), ("dur", r.model.dur_mu)):
        assert np.array_equal(before[name], arr), name
    assert (before["K"], before["bic"], before["state"], before["withheld"]) == (r.K, r.BIC, r.fit_state, r.insights_withheld) and before["meta"] == r.convergence_metadata
    r.lineage = None                                                                                    # leave the module fixture as it was


def test_a_dual_model_is_summarised_with_the_stationary_regime_means_in_raw_units():
    w = generate_identity_stream(8792, n_days=60).window(0, 60)
    z = np.random.default_rng(1).standard_normal((K, F)) * 3
    m = _dual_with_means(z, 2, w["matrix"])
    s = summarize_fit(_dual_result(m, w["matrix"]), w["raw"], w["scaler_mean"], w["scaler_std"])
    assert s.engine == "dual" and np.allclose(s.raw_means, w["scaler_mean"] + w["scaler_std"] * z)
    assert np.allclose(s.raw_means, w["scaler_mean"] + w["scaler_std"] * regime_observation_means(extract_params(m)))
    assert s.exp_durations.shape == (K,)


@pytest.mark.parametrize("seed", [8793, 8794])
def test_dual_canonicalization_by_permute_regimes_is_undone_by_the_alignment(seed):
    """A dual model and the same model with its regimes permuted (`permute_regimes`, what a refit's arbitrary relabelling amounts to): the alignment recovers the permutation exactly."""
    w = generate_identity_stream(seed, n_days=60).window(0, 60)
    z = np.random.default_rng(seed).standard_normal((K, F)) * 4
    m = _dual_with_means(z, seed, w["matrix"])
    perm = (2, 0, 1)
    p2 = permute_regimes(extract_params(m), perm)
    m2 = copy.deepcopy(m)
    for name, val in p2.items():
        setattr(m2, name, np.array(val, copy=True))
    a = summarize_fit(_dual_result(m, w["matrix"]), w["raw"], w["scaler_mean"], w["scaler_std"])
    n = summarize_fit(_dual_result(m2, w["matrix"]), w["raw"], w["scaler_mean"], w["scaler_std"])
    assert np.allclose(n.raw_means, a.raw_means[list(perm)])                                              # new regime k = old regime perm[k]
    rec0, st = step(None, a, CFG, REGISTRY, lineage_id="d:L1")
    rec1, _ = step(st, n, CFG, REGISTRY)
    assert rec1.continuity_state == ALIGNED and rec1.engine == "dual" and rec1.anchor_engine == "dual"
    assert [rec0.raw_to_stable[perm[k]] for k in range(K)] == list(rec1.raw_to_stable)                    # the same underlying regime keeps its stable id


def test_an_engine_switch_from_a_real_baseline_fit_to_a_dual_model_aligns_across_engines(baseline_fits):
    (r, w), _ = baseline_fits
    a = summarize_fit(r, w["raw"], w["scaler_mean"], w["scaler_std"])
    z = np.array(r.model.mu)
    perm = [1, 2, 0]
    m = _dual_with_means(z[perm] + 0.02, 5, w["matrix"])                                                   # the dual engine sees the same regimes, named in another order
    n = summarize_fit(_dual_result(m, w["matrix"]), w["raw"], w["scaler_mean"], w["scaler_std"])
    cfg = AlignmentConfig(min_separation=0.0, ladder=CFG.ladder, same_engine=PairThreshold("ratio", rho=0.5), cross_engine=PairThreshold("ratio", rho=0.5))
    rec0, st = step(None, a, cfg, REGISTRY, lineage_id="x:L1")
    rec1, st1 = step(st, n, cfg, REGISTRY)
    assert rec1.continuity_state == ALIGNED and rec1.engine == "dual" and rec1.anchor_engine == "baseline" and st1.anchor.summary.engine == "dual"
    assert [rec0.raw_to_stable[perm[k]] for k in range(3)] == list(rec1.raw_to_stable)
    strict = AlignmentConfig(min_separation=0.0, ladder=CFG.ladder, same_engine=PairThreshold("ratio", rho=0.5), cross_engine=PairThreshold("ratio", rho=0.0))
    rec2, _ = step(st, n, strict, REGISTRY)
    assert rec2.continuity_state == "CONTINUITY_UNVERIFIED" and "cannot confirm this is the same as before" in rec2.message
