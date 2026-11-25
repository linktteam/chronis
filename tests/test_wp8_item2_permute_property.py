"""WP8 item 2 (diagnosis, point 2): every regime-indexed parameter of each engine, and the property that permuting the regimes leaves the likelihood and the outputs unchanged.
Seeds 8701..8712 (the declared 'dual permutation property' range; each seed gives one data window, one random model and one random permutation).
Method: a parameter 'enters the likelihood' if multiplying it by a random factor near 1 changes the log-likelihood; the test then requires that `shrinkage.permute_regimes` (dual)
covers every such parameter. A deliberately incomplete permutation is the negative control. Mutation check: integration_report/wp8/scripts/mutation_item2.py."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from backbone.hssm.dual_model import DualTimescaleHSSM
from backbone.hssm.label_switching import canonicalize_labels
from backbone.hssm.model import GaussianHSMM
from backbone.hssm.shrinkage import PARAM_NAMES, extract_params, permute_regimes
from data.synthetic_identity import generate_identity_stream

SEEDS = list(range(8701, 8713))
K, F, L, DMAX, T = 3, 7, 2, 12, 60
BASELINE_PARAMS = ("pi", "A", "mu", "var", "dur_mu", "dur_sigma")


def _window(seed):
    return generate_identity_stream(seed, n_days=T).window(0, T)["matrix"]


def _random_dual(seed):
    X = _window(seed)
    rng = np.random.default_rng(seed)
    m = DualTimescaleHSSM(n_regimes=K, n_features=F, latent_dim=L, max_duration=DMAX, seed=seed)
    m._init_params(X)
    m.pi = rng.dirichlet(np.ones(K))
    A = rng.random((K, K)); np.fill_diagonal(A, 0.0); m.A_slow = A / A.sum(axis=1, keepdims=True)
    m.dur_mu = rng.uniform(0.8, 2.0, K); m.dur_sigma = rng.uniform(0.3, 0.8, K)
    m.A_fast = np.array([0.5 * np.eye(L) + 0.1 * rng.standard_normal((L, L)) for _ in range(K)])
    m.b_fast = 0.3 * rng.standard_normal((K, L))
    m.Q_fast = np.array([(0.2 + 0.3 * rng.random()) * np.eye(L) for _ in range(K)])
    m.m0 = rng.standard_normal((K, L)); m.P0 = np.array([(0.5 + rng.random()) * np.eye(L) for _ in range(K)])
    m.C_obs = rng.standard_normal((K, F, L)); m.d_obs = rng.standard_normal((K, F)); m.R_obs = 0.2 + rng.random((K, F))
    m._is_fitted = True                                   # parameters are set by hand; estimate_states only asks for this flag
    return m, X


def _dual_ll(m, X):
    return float(m._forward_pass(X).log_likelihood)


def _with_params(m, params):
    out = copy.deepcopy(m)
    for n in PARAM_NAMES:
        setattr(out, n, np.array(params[n], copy=True))
    return out


def _random_baseline(seed):
    X = _window(seed)
    rng = np.random.default_rng(seed)
    m = GaussianHSMM(n_regimes=K, n_features=F, max_duration=DMAX, seed=seed)
    m.pi = rng.dirichlet(np.ones(K))
    A = rng.random((K, K)); np.fill_diagonal(A, 0.0); m.A = A / A.sum(axis=1, keepdims=True)
    m.mu = rng.standard_normal((K, F)); m.var = 0.3 + rng.random((K, F))
    m.dur_mu = rng.uniform(0.8, 2.0, K); m.dur_sigma = rng.uniform(0.3, 0.8, K)
    return m, X


def _base_ll_post(m, X):
    post, _, _ = m._forward_backward(X)
    return float(m.log_likelihood_), np.asarray(post)


def _permute_baseline(m, perm):
    out = copy.deepcopy(m)
    idx = np.asarray(perm)
    out.pi, out.A = m.pi[idx], m.A[np.ix_(idx, idx)]
    out.mu, out.var, out.dur_mu, out.dur_sigma = m.mu[idx], m.var[idx], m.dur_mu[idx], m.dur_sigma[idx]
    return out


def _entering_arrays(m, X, ll_fn, seed):
    """Names of the arrays with leading dimension K whose perturbation changes the log-likelihood."""
    rng = np.random.default_rng(seed)
    base = ll_fn(m, X)
    found = []
    for name, val in sorted(vars(m).items()):
        if isinstance(val, np.ndarray) and val.dtype.kind == "f" and val.ndim >= 1 and val.shape[0] == m.K:
            trial = copy.deepcopy(m)
            setattr(trial, name, val * (1.0 + 0.05 * rng.random(val.shape)))
            try:
                changed = abs(ll_fn(trial, X) - base) > 1e-9
            except Exception:
                changed = True
            if changed:
                found.append(name)
    return found


# ------------------------------------------------------------------------------------------------------------------- dual
@pytest.mark.parametrize("seed", SEEDS)
def test_dual_permuting_regimes_leaves_likelihood_and_outputs_identical(seed):
    m, X = _random_dual(seed)
    perm = tuple(int(i) for i in np.random.default_rng(seed + 1).permutation(K))
    if perm == tuple(range(K)):
        perm = (1, 2, 0)
    m2 = _with_params(m, permute_regimes(extract_params(m), perm))
    assert np.isclose(_dual_ll(m, X), _dual_ll(m2, X), rtol=1e-9, atol=1e-9)
    mt, _, pt, post = m.estimate_states(X)
    mt2, _, pt2, post2 = m2.estimate_states(X)
    assert np.allclose(post2, post[:, list(perm)], atol=1e-9)             # new regime k is the old regime perm[k]
    inv = np.argsort(perm)
    assert np.array_equal(pt2, inv[pt])                                   # decoded labels are renamed, never changed
    assert np.allclose(mt, mt2, atol=1e-9)                                # the latent trajectory does not depend on the regime names


def test_dual_every_parameter_that_enters_the_likelihood_is_covered_by_permute_regimes():
    for seed in SEEDS[:4]:
        m, X = _random_dual(seed)
        entering = _entering_arrays(m, X, _dual_ll, seed)
        assert set(entering) <= set(PARAM_NAMES), sorted(set(entering) - set(PARAM_NAMES))     # nothing regime-indexed is left out
        assert set(PARAM_NAMES) <= set(entering), sorted(set(PARAM_NAMES) - set(entering))     # and every listed name does enter (no dead entry)


def test_dual_permute_regimes_changes_every_listed_parameter_for_a_generic_model():
    m, _ = _random_dual(SEEDS[0])
    p, q = extract_params(m), permute_regimes(extract_params(m), (1, 2, 0))
    for n in PARAM_NAMES:
        assert not np.array_equal(p[n], q[n]), n
        assert p[n].shape[0] == K and q[n].shape == p[n].shape
    assert all(np.array_equal(p[n], permute_regimes(p, (0, 1, 2))[n]) for n in PARAM_NAMES)           # identity permutation changes nothing


def test_negative_control_an_incomplete_permutation_breaks_the_property():
    m, X = _random_dual(SEEDS[0])
    perm = (1, 2, 0)
    params = permute_regimes(extract_params(m), perm)
    for skipped in PARAM_NAMES:
        broken = dict(params); broken[skipped] = extract_params(m)[skipped]                           # that one parameter keeps the old order
        post_ok = _with_params(m, params).estimate_states(X)[3]
        post_bad = _with_params(m, broken).estimate_states(X)[3]
        ll_bad = _dual_ll(_with_params(m, broken), X)
        differs = (not np.allclose(post_ok, post_bad, atol=1e-6)) or abs(ll_bad - _dual_ll(m, X)) > 1e-6
        assert differs, f"skipping {skipped} was not detected"


# ------------------------------------------------------------------------------------------------------------------- baseline
@pytest.mark.parametrize("seed", SEEDS)
def test_baseline_permuting_regimes_leaves_likelihood_and_posterior_identical(seed):
    m, X = _random_baseline(seed)
    perm = tuple(int(i) for i in np.random.default_rng(seed + 1).permutation(K))
    if perm == tuple(range(K)):
        perm = (1, 2, 0)
    ll, post = _base_ll_post(m, X)
    ll2, post2 = _base_ll_post(_permute_baseline(m, perm), X)
    assert np.isclose(ll, ll2, rtol=1e-9, atol=1e-9)
    assert np.allclose(post2, post[:, list(perm)], atol=1e-9)


def test_baseline_every_parameter_that_enters_the_likelihood_is_in_the_permuted_list():
    for seed in SEEDS[:4]:
        m, X = _random_baseline(seed)
        assert set(_entering_arrays(m, X, lambda mm, XX: _base_ll_post(mm, XX)[0], seed)) == set(BASELINE_PARAMS)


@pytest.mark.parametrize("seed", SEEDS[:4])
def test_baseline_canonicalize_labels_is_such_a_permutation_and_keeps_the_likelihood(seed):
    m, X = _random_baseline(seed)
    ll, post = _base_ll_post(m, X)
    m2 = canonicalize_labels(copy.deepcopy(m))
    order = m2._label_order_applied
    ll2, post2 = _base_ll_post(m2, X)
    assert np.isclose(ll, ll2, rtol=1e-9, atol=1e-9) and np.allclose(post2, post[:, order], atol=1e-9)
    assert np.allclose(np.linalg.norm(m2.mu, axis=1), np.sort(np.linalg.norm(m.mu, axis=1)))        # what it sorts by: the L2 norm of the mean vector
