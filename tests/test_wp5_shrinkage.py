"""WP5 item 2: the shrinkage prototype (backbone/hssm/shrinkage.py), the population generators (data/synthetic_population.py).
Criteria: tests/PREDECLARED_CRITERIA_WP5.md sections 2, 4, 5. No data from the reserved study seeds (6001.., 7001.., 8001.., 9001..) is used here.
Mutation set `shrink` (integration_report/wp5/scripts/mutation_wp5.py) lists what each test would catch if it were wrong.
"""
import copy
import dataclasses
import json

import numpy as np
import pytest
from scipy.optimize import minimize

from backbone.hssm.dual_model import BackwardPassResult, DualTimescaleHSSM
from backbone.hssm.model import FitState
from backbone.hssm.shrinkage import (
    PARAM_NAMES, PriorBuildError, SharedPrior, ShrinkageDualHSSM, align_user, apply_frame, build_shared_prior, duration_pmf, extract_params,
    match_regimes, permute_regimes, pool, regime_observation_means, stationary_moments,
)
from data.synthetic_population import PopulationSpec, draw_perturbation, generate_atypical_user, generate_population_user
from tests.utils.heldout_prediction import continued_synthetic

K, F, L, DMAX = 2, 4, 2, 25
SPEC = PopulationSpec()


def _user(seed=11, n_train=4):
    return generate_population_user(seed, n_train, 2, SPEC)


@pytest.fixture(scope="module")
def prior():
    """A prior built directly from the population centre's parameters (no fitting): C loadings 1.5, d = -2 / +2, A = 0.75 I, Q = 0.3 I, R = 0.05."""
    C = np.zeros((K, F, L))
    for k in range(K):
        for f in range(F):
            C[k, f, f % L] = 1.5
    p = dict(pi=np.array([0.9, 0.1]), A_slow=np.array([[0.0, 1.0], [1.0, 0.0]]), dur_mu=np.full(K, np.log(15.0)), dur_sigma=np.full(K, 0.3),
             A_fast=np.stack([0.75 * np.eye(L)] * K), b_fast=np.zeros((K, L)), Q_fast=np.stack([0.3 * np.eye(L)] * K), m0=np.zeros((K, L)),
             P0=np.stack([np.eye(L)] * K), C_obs=C, d_obs=np.array([np.full(F, -2.0), np.full(F, 2.0)]), R_obs=np.full((K, F), 0.05))
    return SharedPrior.from_params(p, n_users=20, n_excluded={"unhealthy": 0, "regime_ambiguous": 1, "frame_failed": 0})


def _engine(cls=ShrinkageDualHSSM, **kw):
    return cls(n_regimes=K, n_features=F, latent_dim=L, max_duration=DMAX, seed=3, **kw)


def _params_equal(a, b):
    return all(np.array_equal(getattr(a, n), getattr(b, n)) for n in PARAM_NAMES)


# ------------------------------------------------------------------ generators
def test_zero_spread_population_user_equals_the_existing_generator_exactly():
    for seed in (11, 12):
        u = generate_population_user(seed, 4, 2, SPEC.with_zero_spread())
        ref = continued_synthetic(seed)
        assert np.array_equal(u.X_train, ref.X_train) and np.array_equal(u.X_test, ref.X_test)
        assert np.array_equal(u.p_train, ref.p_train) and np.array_equal(u.p_test, ref.p_test)


def test_population_users_differ_by_the_declared_perturbations():
    u, v = _user(11), _user(12)
    assert not np.array_equal(u.X_train, v.X_train)
    assert np.array_equal(u.X_train, _user(11).X_train)                           # deterministic
    base_d = np.array([np.full(F, -2.0), np.full(F, 2.0)])
    e_d = u.eps[: K * F].reshape(K, F)
    assert np.allclose(u.truth["d"], base_d + 0.6 * e_d)                          # intercept perturbation = sd * draw
    assert 0.4 <= u.truth["A"][0, 0] <= 0.9 and u.truth["q"] > 0 and u.truth["r"] > 0 and u.truth["mean_dwell"] > 0
    # negative control: with zero spread the parameters are the centre
    z = generate_population_user(11, 4, 2, SPEC.with_zero_spread())
    assert np.allclose(z.truth["d"], base_d) and z.truth["q"] == 0.3 and z.truth["r"] == 0.05 and z.truth["mean_dwell"] == 15.0


def test_the_users_true_parameters_follow_the_declared_formulas():
    for seed in (11, 12, 13):
        u = _user(seed)
        e = u.eps
        e_d, e_c, (e_a, e_q, e_r, e_w) = e[: K * F].reshape(K, F), e[K * F: K * F + F], e[K * F + F:]
        t = u.truth
        assert np.allclose(t["d"], np.array([np.full(F, -2.0), np.full(F, 2.0)]) + 0.6 * e_d)
        assert np.allclose([t["C"][f, f % L] for f in range(F)], 1.5 * np.exp(0.15 * e_c)) and np.count_nonzero(t["C"]) == F
        assert np.allclose(t["A"], np.clip(0.75 + 0.05 * e_a, 0.4, 0.9) * np.eye(L))
        assert np.isclose(t["q"], 0.3 * np.exp(0.25 * e_q)) and np.isclose(t["r"], 0.05 * np.exp(0.25 * e_r)) and np.isclose(t["mean_dwell"], 15.0 * np.exp(0.2 * e_w))
    assert np.isclose(_user(11).truth["mean_dwell"], 15.0 * np.exp(0.2 * _user(11).eps[-1])) and _user(11).truth["mean_dwell"] != 15.0


def test_perturbation_draws_come_from_the_documented_independent_stream():
    u = _user(11)
    e = np.random.default_rng([11, 1]).standard_normal((K, F))
    assert np.array_equal(u.eps[: K * F], e.ravel())
    assert not np.array_equal(u.eps[: K * F], np.random.default_rng(11).standard_normal((K, F)).ravel())      # not the data stream


def test_atypical_user_is_three_times_the_population_draw_and_farther_from_the_centre():
    pop, atyp = _user(13), generate_atypical_user(13, 4, 2, SPEC)
    assert np.allclose(atyp.eps, 3.0 * pop.eps) and atyp.spread_factor == 3.0
    assert np.linalg.norm(atyp.eps) > 2.9 * np.linalg.norm(pop.eps)
    centre = np.array([np.full(F, -2.0), np.full(F, 2.0)])
    assert np.linalg.norm(atyp.truth["d"] - centre) > 2.5 * np.linalg.norm(pop.truth["d"] - centre)


def test_intercept_shift_moves_every_intercept():
    shifted = generate_population_user(11, 3, 2, dataclasses.replace(SPEC, intercept_shift=2.0))
    plain = generate_population_user(11, 3, 2, SPEC)
    assert np.allclose(shifted.truth["d"], plain.truth["d"] + 2.0)


# ------------------------------------------------------------------ the prior object and privacy (criterion 5)
def test_prior_is_aggregate_only_and_roundtrips_through_json(prior):
    prior.assert_aggregate_only()
    d = json.loads(prior.to_json())
    assert set(d) <= {"K", "F", "L", "n_users", "n_excluded", "prior_uncertainty_in_m_t_std", *PARAM_NAMES}
    back = SharedPrior.from_dict(d)
    assert _params_equal(back, prior) and back.n_users == 20 and back.prior_uncertainty_in_m_t_std is False
    assert len(prior.to_json()) < 20_000                                         # a few hundred numbers, not a data set


def test_privacy_check_fails_when_the_prior_holds_data():
    u = _user(11)
    p = _engine()
    p._init_params(u.X_train)
    clean = SharedPrior.from_params(extract_params(p), n_users=3)
    clean.assert_aggregate_only()
    leaky = copy.deepcopy(clean)
    leaky.X_train = u.X_train                                                      # raw data smuggled in
    with pytest.raises(AssertionError, match="whitelist"):
        leaky.assert_aggregate_only()
    wrong_shape = copy.deepcopy(clean)
    wrong_shape.d_obs = np.vstack([clean.d_obs, clean.d_obs[:1]])                  # a per-user row smuggled into a parameter
    with pytest.raises(AssertionError, match="shape"):
        wrong_shape.assert_aggregate_only()
    bad_counts = copy.deepcopy(clean)
    bad_counts.n_excluded = {"user_7_sequence": 3}
    with pytest.raises(AssertionError):
        bad_counts.assert_aggregate_only()
    with pytest.raises(ValueError):
        SharedPrior.from_params({**clean.params(), "d_obs": np.zeros((K, F + 1))})
    nan_d = clean.params()
    nan_d["d_obs"][0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        SharedPrior.from_params(nan_d)


# ------------------------------------------------------------------ pseudo statistics (the shared model as pseudo data)
def test_stationary_moments_of_the_centre(prior):
    mu, Sigma = stationary_moments(prior.A_fast[0], prior.b_fast[0], prior.Q_fast[0], prior.P0[0])
    assert np.allclose(mu, 0.0) and np.allclose(Sigma, 0.3 / (1 - 0.75 ** 2) * np.eye(L))
    mu2, Sigma2 = stationary_moments(0.5 * np.eye(L), np.array([1.0, 2.0]), 0.2 * np.eye(L), np.eye(L))
    assert np.allclose(mu2, [2.0, 4.0]) and np.allclose(Sigma2, 0.2 / 0.75 * np.eye(L))                  # mu = (I-A)^-1 b
    mu3, Sigma3 = stationary_moments(np.eye(L), np.zeros(L), np.eye(L), 7.0 * np.eye(L))                  # unstable: documented fallback
    assert np.allclose(mu3, 0.0) and np.allclose(Sigma3, 7.0 * np.eye(L))


def test_pseudo_statistics_reproduce_the_shared_parameters(prior):
    m = _engine(shared_prior=prior, n0=10.0)
    m.dur_mu, m.dur_sigma = prior.dur_mu.copy(), prior.dur_sigma.copy()
    ps = m._prepare_pseudo()
    for k in range(K):
        Theta0 = np.vstack([prior.A_fast[k].T, prior.b_fast[k][None, :]])
        assert np.allclose(np.linalg.solve(ps["G0"][k], ps["H0_dyn"][k]), Theta0)
        assert np.allclose(ps["S0"][k] - Theta0.T @ ps["H0_dyn"][k], prior.Q_fast[k])                      # residual per pseudo-step = Q0
        assert np.allclose(ps["G0"][k][:L, :L], 0.3 / (1 - 0.75 ** 2) * np.eye(L)) and np.allclose(ps["G0"][k][L, L], 1.0)
        for f in range(F):
            th = np.concatenate([prior.C_obs[k, f], [prior.d_obs[k, f]]])
            assert np.allclose(np.linalg.solve(ps["G0"][k], ps["H0_obs"][k, f]), th)
            assert np.isclose(ps["sy0"][k, f] - th @ ps["H0_obs"][k, f], prior.R_obs[k, f])
    engine_pmf = np.exp(m._duration_logpmf())                                                             # the engine's own truncated, renormalised pmf
    assert np.allclose(duration_pmf(prior.dur_mu, prior.dur_sigma, DMAX), engine_pmf, atol=1e-12) and np.allclose(engine_pmf.sum(axis=1), 1.0)
    assert np.allclose(ps["dbar"], engine_pmf @ np.arange(1, DMAX + 1))
    assert 12.0 < ps["dbar"][0] < 18.0                                                                     # about one 15-step segment


def test_dynamics_hook_adds_exactly_n0_pseudo_steps():
    u = _user()
    base = _engine(cls=DualTimescaleHSSM)
    base.fit(u.X_train, n_iter=2)
    fwd = base._forward_pass(u.X_train)
    smoother = base._switching_smoother(fwd, base._discrete_backward(fwd))
    G, H, S, W = base._dynamics_sufficient_stats(smoother)
    p = SharedPrior.from_params(extract_params(base), n_users=5)
    m = _engine(shared_prior=p, n0=7.5)
    m._pseudo = m._prepare_pseudo()
    m.__dict__.update({n: getattr(base, n) for n in PARAM_NAMES})
    G2, H2, S2, W2 = m._dynamics_sufficient_stats(smoother)
    assert np.allclose(W2 - W, 7.5) and np.allclose(G2 - G, 7.5 * m._pseudo["G0"]) and np.allclose(H2 - H, 7.5 * m._pseudo["H0_dyn"])
    assert np.allclose(S2 - S, 7.5 * m._pseudo["S0"])
    inactive = _engine(shared_prior=p, n0=0.0)                                                            # negative control: n0 = 0 adds nothing
    for a, b in zip(inactive._dynamics_sufficient_stats(smoother), (G, H, S, W)):
        assert np.array_equal(a, b)


def test_observation_map_step_with_n0_zero_equals_the_engines_regression():
    u = _user()
    base = _engine(cls=DualTimescaleHSSM)
    base.fit(u.X_train, n_iter=2)
    X = u.X_train.copy()
    X[3, 1] = np.nan                                                                                       # a missing value must be handled like the engine does
    fwd = base._forward_pass(X)
    bwd = base._discrete_backward(fwd)
    smoother = base._switching_smoother(fwd, bwd)
    ref = copy.deepcopy(base)
    ref._m_step_continuous(X, bwd, smoother)                                                                # the engine's own M-step (obs block is the ML regression)
    p = SharedPrior.from_params(extract_params(base), n_users=5)
    m = copy.deepcopy(base)
    m.__class__ = ShrinkageDualHSSM
    m.shared_prior, m.n0, m.warm_start, m._pseudo = p, 0.0, True, None
    m._pseudo = m._prepare_pseudo()
    m._observation_map_step(X, bwd, smoother, 0.0)
    assert np.allclose(m.C_obs, ref.C_obs, atol=1e-8) and np.allclose(m.d_obs, ref.d_obs, atol=1e-8) and np.allclose(m.R_obs, ref.R_obs, atol=1e-8)
    # negative control: n0 > 0 moves the estimates toward the prior (here the prior is the first-iteration fit itself)
    m2 = copy.deepcopy(m)
    m2.C_obs, m2.d_obs, m2.R_obs = base.C_obs.copy(), base.d_obs.copy(), base.R_obs.copy()
    m2._observation_map_step(X, bwd, smoother, 1e6)
    assert np.allclose(m2.C_obs, p.C_obs, atol=1e-3) and np.allclose(m2.d_obs, p.d_obs, atol=1e-3) and np.allclose(m2.R_obs, p.R_obs, atol=1e-3)


def test_duration_and_transition_pseudo_counts_equal_the_map_of_the_augmented_counts(prior):
    """_update_discrete_params with n0 > 0 must equal the engine's own update on counts augmented by n0 / Dbar pseudo-segments."""
    T = 30
    entry = np.zeros((T, K, DMAX))
    entry[0, 0, 7] = 1.0
    entry[10, 0, 9] = 1.0
    entry[5, 1, 12] = 1.0
    gamma = np.zeros((T, K))
    gamma[:, 0] = 0.5
    gamma[:, 1] = 0.5
    zeros = lambda *s: np.zeros(s)
    bwd = BackwardPassResult(beta=zeros(T, K, DMAX), regime_posterior_smoothed=gamma, gamma=zeros(T, K, DMAX), entry_posterior=entry,
                             xi_counts=np.array([[0.0, 2.0], [1.0, 0.0]]), xi_t=zeros(T - 1, K, K), future_entry_val=zeros(T, K))
    n0 = 45.0
    m = _engine(shared_prior=prior, n0=n0)
    m._init_params(np.zeros((5, F)))
    dbar = m._pseudo["dbar"]
    c0 = n0 / dbar
    ref = _engine(cls=DualTimescaleHSSM)
    ref._init_params(np.zeros((5, F)))
    ref.dur_mu, ref.dur_sigma, ref.A_slow = prior.dur_mu.copy(), prior.dur_sigma.copy(), prior.A_slow.copy()      # the engine's own update starts from the prior's values too
    aug = dataclasses.replace(bwd, xi_counts=bwd.xi_counts + c0[:, None] * prior.A_slow, entry_posterior=entry.copy())
    aug.entry_posterior[0] += c0[:, None] * duration_pmf(prior.dur_mu, prior.dur_sigma, DMAX)
    aug.regime_posterior_smoothed = gamma.copy()
    aug.regime_posterior_smoothed[0] = gamma[0] + c0 * prior.pi
    ref._update_discrete_params(aug)
    m._update_discrete_params(bwd)
    for n in ("dur_mu", "dur_sigma", "A_slow", "pi"):
        assert np.allclose(getattr(m, n), getattr(ref, n), atol=1e-9), n
    # the pull is toward the prior and strong: 45 pseudo-steps = 3 pseudo-segments against 2 real segments in regime 0 (durations 8 and 10)
    plain = _engine(cls=DualTimescaleHSSM)
    plain._init_params(np.zeros((5, F)))
    plain.dur_mu, plain.dur_sigma = prior.dur_mu.copy(), prior.dur_sigma.copy()
    plain._update_discrete_params(bwd)
    assert abs(m.dur_mu[0] - prior.dur_mu[0]) < abs(plain.dur_mu[0] - prior.dur_mu[0])


def test_transition_pseudo_counts_with_three_regimes_pull_each_row_toward_the_shared_rows():
    """K = 2 cannot test A_slow (the off-diagonal row is always [1])); with K = 3 the pseudo-counts n0 / Dbar_k times the shared row matter."""
    K3 = 3
    base = ShrinkageDualHSSM(n_regimes=K3, n_features=F, latent_dim=L, max_duration=DMAX, seed=5)
    base._init_params(np.zeros((5, F)))
    params = extract_params(base)
    params["A_slow"] = np.array([[0.0, 0.8, 0.2], [0.3, 0.0, 0.7], [0.5, 0.5, 0.0]])
    params["pi"] = np.array([0.5, 0.3, 0.2])
    params["dur_mu"], params["dur_sigma"] = np.full(K3, np.log(15.0)), np.full(K3, 0.3)
    pr = SharedPrior.from_params(params, n_users=4)
    T = 20
    zeros = lambda *s: np.zeros(s)
    entry = zeros(T, K3, DMAX)
    entry[0, 0, 9] = entry[4, 1, 9] = entry[9, 2, 9] = 1.0
    gamma = zeros(T, K3)
    gamma[:, 0] = 1.0
    N = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])                  # data: every row goes to exactly one regime
    bwd = BackwardPassResult(beta=zeros(T, K3, DMAX), regime_posterior_smoothed=gamma, gamma=zeros(T, K3, DMAX), entry_posterior=entry, xi_counts=N,
                             xi_t=zeros(T - 1, K3, K3), future_entry_val=zeros(T, K3))
    n0 = 30.0
    m = ShrinkageDualHSSM(n_regimes=K3, n_features=F, latent_dim=L, max_duration=DMAX, seed=5, shared_prior=pr, n0=n0)
    m._init_params(np.zeros((5, F)))
    m._update_discrete_params(bwd)
    c0 = n0 / m._pseudo["dbar"]
    expected = N + c0[:, None] * pr.A_slow
    expected = expected / expected.sum(axis=1, keepdims=True)
    assert np.allclose(m.A_slow, expected, atol=1e-12) and np.allclose(m.A_slow.sum(axis=1), 1.0) and np.all(np.diag(m.A_slow) == 0.0)
    assert np.allclose(m.pi, (gamma[0] + c0 * pr.pi) / (gamma[0] + c0 * pr.pi).sum())
    # negative control: n0 = 0 gives the unpenalised counts, and the penalised rows are strictly between the data rows and the shared rows
    plain = ShrinkageDualHSSM(n_regimes=K3, n_features=F, latent_dim=L, max_duration=DMAX, seed=5)
    plain._init_params(np.zeros((5, F)))
    plain._update_discrete_params(bwd)
    assert np.allclose(plain.A_slow, N)
    assert np.all(m.A_slow[N > 0] < 1.0) and np.all(m.A_slow[N > 0] > pr.A_slow[N > 0])


# ------------------------------------------------------------------ the model: off by default, warm start, strength
def test_without_a_prior_or_with_n0_zero_and_no_warm_start_the_engine_is_unchanged(prior):
    u = _user()
    base = _engine(cls=DualTimescaleHSSM).fit(u.X_train, n_iter=4)
    for kw in ({}, dict(shared_prior=prior, n0=0.0, warm_start=False), dict(shared_prior=None, n0=50.0)):
        m = _engine(**kw).fit(u.X_train, n_iter=4)
        assert _params_equal(m, base) and m.log_likelihood_ == base.log_likelihood_ and m.fit_state_ == base.fit_state_, kw
        assert np.array_equal(m.estimate_states(u.X_train)[0], base.estimate_states(u.X_train)[0])
    assert _engine(shared_prior=prior, n0=0.0, warm_start=False).used_shared_prior is False
    assert _engine(shared_prior=prior, n0=0.0).used_shared_prior is True          # warm start counts as using the prior


def test_warm_start_initialises_at_the_prior_and_the_shared_model_has_the_prior_parameters(prior):
    m = _engine(shared_prior=prior, n0=0.0)
    m._init_params(_user().X_train)
    assert _params_equal(m, prior)
    s = _engine(shared_prior=prior, n0=5.0).become_shared_model()
    assert _params_equal(s, prior) and s.fit_state_ == FitState.CONVERGED
    cold = _engine(shared_prior=prior, n0=0.0, warm_start=False)
    cold._init_params(_user().X_train)
    assert not _params_equal(cold, prior)                                           # negative control: no warm start, random initialisation
    with pytest.raises(ValueError):
        _engine(n0=-1.0)
    with pytest.raises(ValueError):
        ShrinkageDualHSSM(n_regimes=3, n_features=F, latent_dim=L, max_duration=DMAX, shared_prior=prior)


def test_the_prior_pulls_harder_the_larger_n0_and_dominates_when_huge(prior):
    u = _user(14, 3)
    dist = {}
    for n0 in (0.0, 5.0, 50.0, 1e6):
        m = _engine(shared_prior=prior, n0=n0).fit(u.X_train, n_iter=4)
        dist[n0] = max(float(np.max(np.abs(getattr(m, n) - getattr(prior, n)))) for n in ("A_fast", "Q_fast", "C_obs", "d_obs", "R_obs", "dur_mu", "dur_sigma"))
        if n0 == 1e6:
            for n in ("A_fast", "b_fast", "Q_fast", "C_obs", "d_obs", "R_obs", "dur_mu", "dur_sigma", "A_slow"):
                assert np.allclose(getattr(m, n), getattr(prior, n), atol=2e-3), n
    assert dist[0.0] > dist[5.0] > dist[50.0] > dist[1e6]
    assert dist[0.0] > 10 * dist[1e6]
    assert not (dist[0.0] == dist[1e6])


def test_shrinkage_does_not_change_the_m_t_std_pipeline(prior):
    u = _user(15, 3)
    m = _engine(shared_prior=prior, n0=20.0).fit(u.X_train, n_iter=3)
    assert m.prior_uncertainty_in_m_t_std is False and prior.prior_uncertainty_in_m_t_std is False
    assert "estimate_states" not in ShrinkageDualHSSM.__dict__ and ShrinkageDualHSSM.estimate_states is DualTimescaleHSSM.estimate_states
    m_t, m_std, p_t, post = m.estimate_states(u.X_train)
    plain = copy.deepcopy(m)
    plain.__class__ = DualTimescaleHSSM                                              # the unchanged engine method with the same parameters gives the same arrays
    assert all(np.array_equal(a, b) for a, b in zip(plain.estimate_states(u.X_train), (m_t, m_std, p_t, post)))


# ------------------------------------------------------------------ alignment
def _random_frame(rng, cond_max=4.0):
    while True:
        M = np.eye(L) + 0.6 * rng.normal(size=(L, L))
        if abs(np.linalg.det(M)) > 0.3 and np.linalg.cond(M) < cond_max:
            return M


def _data_params(prior):
    p = prior.params()
    p["b_fast"] = np.array([[0.3, -0.2], [-0.1, 0.4]])
    p["A_fast"] = np.array([[[0.70, 0.15], [-0.10, 0.60]], [[0.65, -0.12], [0.08, 0.72]]])      # not a multiple of the identity (a frame change must matter)
    p["Q_fast"] = np.array([[[0.30, 0.05], [0.05, 0.25]], [[0.20, -0.04], [-0.04, 0.35]]])
    p["C_obs"] = p["C_obs"] + 0.1 * np.arange(K * F * L).reshape(K, F, L) / 16.0
    p["m0"] = np.array([[0.2, 0.1], [-0.3, 0.2]])
    p["dur_mu"], p["dur_sigma"] = np.log(np.array([12.0, 18.0])), np.array([0.25, 0.35])                   # regime-specific durations (a permutation must move them)
    return p


def _loglik(params, X):
    m = _engine()
    m._init_params(X)
    for n, v in params.items():
        setattr(m, n, np.array(v, copy=True))
    return m._forward_pass(X).log_likelihood


def test_frame_change_and_regime_relabelling_preserve_the_likelihood(prior):
    p = _data_params(prior)
    X = _user(16, 3).X_train
    base = _loglik(p, X)
    M = _random_frame(np.random.default_rng(0))
    # the engine adds small absolute jitter to covariances, which is not frame invariant: agreement to 1e-4 of a log-likelihood near -1000 (1e-7 relative)
    assert np.isclose(_loglik(apply_frame(p, M), X), base, rtol=0, atol=1e-4)
    assert np.isclose(_loglik(permute_regimes(p, (1, 0)), X), base, rtol=0, atol=1e-8)
    # negative control: a wrong transform (A not conjugated) changes it
    wrong = apply_frame(p, M)
    wrong["A_fast"] = p["A_fast"].copy()
    assert abs(_loglik(wrong, X) - base) > 0.1                                       # 1000 times the tolerance above


def test_align_user_recovers_the_reference_from_a_relabelled_and_reframed_copy(prior):
    ref = _data_params(prior)
    M = _random_frame(np.random.default_rng(1))
    user = permute_regimes(apply_frame(ref, np.linalg.inv(M)), (1, 0))               # user's labels swapped and latent coordinates changed
    aligned, info = align_user(user, ref)
    assert info["status"] == "ok" and info["perm"] == (1, 0)
    for n in PARAM_NAMES:
        assert np.allclose(aligned[n], ref[n], atol=1e-8), n
    # the same user is NOT equal to the reference without alignment
    assert not np.allclose(user["C_obs"], ref["C_obs"])


def test_ambiguous_regimes_and_unrelated_frames_are_reported_as_failures(prior):
    ref = _data_params(prior)
    same_means = copy.deepcopy(ref)
    same_means["d_obs"][:] = ref["d_obs"].mean()                                     # regime means indistinguishable -> matching is ambiguous
    same_means["b_fast"][:] = 0.0
    ref_flat = copy.deepcopy(same_means)
    _, info = align_user(same_means, ref_flat)
    assert info["status"] == "regime_ambiguous"
    flat_b = copy.deepcopy(ref)
    flat_b["b_fast"][:] = 0.0                                                         # regime means are then exactly d_k, unambiguous
    scrambled = copy.deepcopy(flat_b)
    scrambled["C_obs"] = np.random.default_rng(2).normal(size=ref["C_obs"].shape)     # loadings unrelated to the reference's by any common linear map
    aligned, info = align_user(scrambled, flat_b)
    assert aligned is None and info["status"] == "frame_failed" and info["rel_residual"] > 0.5
    squeezed = apply_frame(ref, np.linalg.inv(np.diag([1.0, 0.02])))                  # the user's latent axis 2 is 50 times shorter: exact fit, cond(M) = 50
    aligned, info = align_user(squeezed, ref)
    assert aligned is None and info["status"] == "frame_failed" and info["cond"] > 10 and info["rel_residual"] < 1e-8
    ok, info = align_user(copy.deepcopy(ref), ref)                                    # negative control: the reference aligns with itself
    assert info["status"] == "ok" and info["perm"] == (0, 1) and np.isclose(info["cond"], 1.0, atol=1e-6)


def test_match_regimes_and_pooling():
    ref = np.array([[0.0, 0.0], [4.0, 4.0]])
    perm, best, second, amb = match_regimes(ref, np.array([[4.1, 3.9], [0.1, -0.1]]))
    assert perm == (1, 0) and not amb and best < 0.1 and second > 30
    assert match_regimes(ref, ref)[0] == (0, 1)
    ref1 = np.array([[0.0], [1.0]])
    assert match_regimes(ref1, np.array([[0.40], [0.60]]))[3] is False                       # best / second = 0.44 <= 0.5: matched
    assert match_regimes(ref1, np.array([[0.45], [0.55]]))[3] is True                        # best / second = 0.67 > 0.5: ambiguous
    assert match_regimes(ref1, np.array([[0.5], [0.5]]))[3] is True                          # best == second: ambiguous
    a, b = {n: np.full((2, 2), 1.0) for n in PARAM_NAMES}, {n: np.full((2, 2), 3.0) for n in PARAM_NAMES}
    assert all(np.allclose(v, 2.0) for v in pool([a, b]).values())


def test_pooling_uses_the_median_for_the_two_duration_parameters_and_the_mean_for_everything_else():
    users = [{n: np.full((2, 2), v) for n in PARAM_NAMES} for v in (1.0, 2.0, 12.0)]            # one outlying user (a degenerate fit)
    pooled = pool(users)
    for n in PARAM_NAMES:
        expected = 2.0 if n in ("dur_mu", "dur_sigma") else 5.0                                  # median 2, mean 5
        assert np.allclose(pooled[n], expected), n
    assert all(np.allclose(v, 5.0) for v in pool(users, ()).values())                            # the pre-declared all-mean form is still available
    degenerate = [{**{n: np.full((2, 2), 1.0) for n in PARAM_NAMES}, "dur_mu": np.full((2, 2), m)} for m in (2.5, 2.6, 300.0)]
    assert np.allclose(pool(degenerate)["dur_mu"], 2.6)


# ------------------------------------------------------------------ building the prior (small: 10 users, 3 segments, 3 iterations, one round)
@pytest.fixture(scope="module")
def small_population():
    return [generate_population_user(100 + i, 3, 0, SPEC).X_train for i in range(10)]


def test_build_shared_prior_returns_an_aggregate_only_prior_and_a_counts_only_report(small_population):
    pr, report = build_shared_prior(small_population, K, F, L, DMAX, reference_seeds=(1, 2), n_iter=3, rounds=1, min_users=5)
    pr.assert_aggregate_only()
    assert pr.n_users == report["rounds"][0]["n_pooled"] and 5 <= pr.n_users <= 10
    assert pr.n_users + sum(report["rounds"][0]["excluded"].values()) == 10                      # every user is pooled or counted as excluded
    json.dumps(report)                                                                           # counts and numbers only
    assert set(report["rounds"][0]["excluded"]) == {"unhealthy", "regime_ambiguous", "frame_failed"}
    scrambled, _ = build_shared_prior(small_population, K, F, L, DMAX, reference_seeds=(1, 2), n_iter=3, rounds=1, min_users=5, scramble_labels_seed=0)
    assert not _params_equal(scrambled, pr)                                                      # the "alignment broken" variant changes the prior


def test_build_shared_prior_records_both_duration_poolings_and_rejects_an_unknown_one(small_population):
    pr, report = build_shared_prior(small_population, K, F, L, DMAX, reference_seeds=(1, 2), n_iter=3, rounds=1, min_users=5)
    rd = report["rounds"][0]
    assert rd["duration_pooling"] == "median" and len(rd["dur_mu_pooled_by_mean"]) == K and len(rd["dur_sigma_pooled_by_mean"]) == K
    assert isinstance(rd["n_user_regimes_with_dur_sigma_ge_5"], int)
    by_mean, rep2 = build_shared_prior(small_population, K, F, L, DMAX, reference_seeds=(1, 2), n_iter=3, rounds=1, min_users=5, duration_pooling="mean")
    assert rep2["rounds"][0]["duration_pooling"] == "mean"
    assert np.allclose(by_mean.dur_mu, rd["dur_mu_pooled_by_mean"]) and np.allclose(rep2["rounds"][0]["dur_mu_pooled_by_mean"], rd["dur_mu_pooled_by_mean"])
    assert not np.allclose(pr.dur_mu, by_mean.dur_mu)                                            # the median and the mean differ on this population
    with pytest.raises(ValueError, match="duration_pooling"):
        build_shared_prior(small_population, K, F, L, DMAX, reference_seeds=(1,), n_iter=2, rounds=1, min_users=5, duration_pooling="trimmed")


def test_build_shared_prior_counts_user_regimes_with_degenerate_durations(small_population):
    from unittest.mock import patch
    import backbone.hssm.shrinkage as module
    def fake_align(user, ref):
        q = {n: np.array(v, copy=True) for n, v in user.items()}
        q["dur_sigma"] = np.array([6.0, 0.3])                                    # one degenerate regime per user
        return q, {"status": "ok", "perm": (0, 1), "cond": 1.0, "rel_residual": 0.0}
    with patch.object(module, "align_user", fake_align):
        _, report = build_shared_prior(small_population[:6], K, F, L, DMAX, reference_seeds=(1,), n_iter=2, rounds=1, min_users=5)
    assert report["rounds"][0]["n_pooled"] == 6 and report["rounds"][0]["n_user_regimes_with_dur_sigma_ge_5"] == 6


def test_build_shared_prior_refuses_too_few_users(small_population):
    from unittest.mock import patch
    import backbone.hssm.shrinkage as module
    with patch.object(module, "ShrinkageDualHSSM", side_effect=AssertionError("no model may be built for too few users")):
        with pytest.raises(PriorBuildError, match="minimum"):
            build_shared_prior(small_population[:4], K, F, L, DMAX, reference_seeds=(1,), n_iter=2, rounds=1, min_users=5)
    with pytest.raises(PriorBuildError, match="minimum"):
        build_shared_prior(small_population[:6], K, F, L, DMAX, reference_seeds=(1,), n_iter=2, rounds=1, min_users=7)


def test_build_shared_prior_counts_unhealthy_users(small_population):
    from unittest.mock import patch
    import backbone.hssm.shrinkage as module
    calls = []
    def healthy_once(model):                                   # the reference fit is healthy, every later fit is not
        calls.append(1)
        return len(calls) == 1
    with patch.object(module, "_healthy", healthy_once):
        with pytest.raises(PriorBuildError, match="survived alignment") as exc:
            build_shared_prior(small_population[:6], K, F, L, DMAX, reference_seeds=(1,), n_iter=2, rounds=1, min_users=5)
    assert "'unhealthy': 6" in str(exc.value)


def test_build_shared_prior_stops_when_too_few_users_survive_alignment(small_population):
    from unittest.mock import patch
    import backbone.hssm.shrinkage as module
    with patch.object(module, "align_user", lambda user, ref: (None, {"status": "frame_failed"})):
        with pytest.raises(PriorBuildError, match="survived alignment") as exc:
            build_shared_prior(small_population[:6], K, F, L, DMAX, reference_seeds=(1,), n_iter=2, rounds=1, min_users=5)
    assert "frame_failed" in str(exc.value) and "'frame_failed': 6" in str(exc.value)


def test_the_default_path_does_not_import_the_prototype():
    import subprocess
    import sys
    code = ("import sys; import backbone.hssm, backbone.hssm.fitting, backbone.hssm.gating, backbone.hssm.dual_model, backbone.hssm.model; "
            "print('backbone.hssm.shrinkage' in sys.modules, 'data.synthetic_population' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.split() == ["False", "False"]
