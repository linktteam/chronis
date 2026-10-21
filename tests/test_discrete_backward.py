"""Negative controls for the ported discrete HSMM forward-backward (section 5.1)."""
import numpy as np

from tests.utils.model_factory import make_tiny_model, make_tiny_data
from tests.oracle.tiny_exact_oracle import run_tiny_exact_oracle

EXACT_TOL = 1e-6


def test_gamma_sums_to_one_every_timestep():
    """Hard identity: sum over (k,r) of gamma[t] must be 1 at EVERY t, for any T."""
    for (K, F, L, D, T, seed) in [(2, 1, 1, 2, 6, 1), (2, 2, 1, 3, 5, 2), (1, 1, 1, 4, 4, 3)]:
        model = make_tiny_model(K=K, F=F, L=L, Dmax=D, seed=seed)
        X = make_tiny_data(model, T, seed=seed + 400)
        fwd = model._forward_pass(X)
        bwd = model._discrete_backward(fwd)
        sums = bwd.regime_posterior_smoothed.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-8), f"K={K} T={T} seed={seed}: gamma sums {sums}"


def test_alpha_beta_are_dual():
    """sum_{k,r} alpha[t]*beta[t] must equal P(Y) for every t. This is the check that
    exposed the literal-spec ll_entry problem (NOTE (c) in dual_model._discrete_backward)."""
    from scipy.special import logsumexp
    for (K, F, L, D, T, seed) in [(2, 1, 1, 2, 5, 1), (2, 2, 1, 3, 6, 2), (3, 1, 1, 2, 6, 3)]:
        model = make_tiny_model(K=K, F=F, L=L, Dmax=D, seed=seed)
        X = make_tiny_data(model, T, seed=seed + 450)
        fwd = model._forward_pass(X)
        bwd = model._discrete_backward(fwd)
        for t in range(T):
            z = logsumexp((fwd.alpha[t] + bwd.beta[t]).ravel())
            assert abs(z - fwd.log_likelihood) < 1e-8, (
                f"K={K} T={T} seed={seed} t={t}: logsumexp(alpha*beta)={z} vs loglik={fwd.log_likelihood}")


def test_ported_forward_backward_matches_oracle_marginal_loglik():
    for (K, F, L, D, T, seed) in [(2, 1, 1, 2, 2, 7), (1, 1, 1, 3, 3, 8)]:
        model = make_tiny_model(K=K, F=F, L=L, Dmax=D, seed=seed)
        X = make_tiny_data(model, T, seed=seed + 500)
        fwd = model._forward_pass(X)
        oracle = run_tiny_exact_oracle(model, X)
        gap = abs(fwd.log_likelihood - oracle.log_likelihood)
        assert gap < EXACT_TOL, f"K={K} T={T} seed={seed}: {gap}"


def test_discrete_smoothing_matches_oracle_at_T2():
    """At T=2 the forward pass is exact, so the smoothed regime posterior and xi_t must
    match the oracle to float precision."""
    for seed in range(4):
        model = make_tiny_model(K=2, F=1, L=1, Dmax=2, seed=seed)
        X = make_tiny_data(model, T=2, seed=300 + seed)
        fwd = model._forward_pass(X)
        bwd = model._discrete_backward(fwd)
        oracle = run_tiny_exact_oracle(model, X)
        g = np.max(np.abs(bwd.regime_posterior_smoothed - oracle.smoothed_regime_posterior))
        x = np.max(np.abs(bwd.xi_t - oracle.pairwise_transition_marginal))
        assert g < EXACT_TOL, f"seed={seed}: smoothed regime posterior gap {g}"
        assert x < EXACT_TOL, f"seed={seed}: xi_t vs oracle pairwise marginal gap {x}"


def test_entry_posterior_is_a_subset_of_gamma():
    model = make_tiny_model(K=2, F=1, L=1, Dmax=3, seed=9)
    X = make_tiny_data(model, T=5, seed=909)
    fwd = model._forward_pass(X)
    bwd = model._discrete_backward(fwd)
    assert np.all(bwd.entry_posterior <= bwd.gamma + 1e-6), (
        "entry_posterior exceeds gamma somewhere: entry_mass/ll_entry bookkeeping is inconsistent")


def test_entry_mass_does_not_double_count_transition_prior():
    """NOTE (a) in dual_model._forward_pass: entry_mass[t>=1] must be dur_logpmf alone."""
    model = make_tiny_model(K=2, F=1, L=1, Dmax=2, seed=4)
    X = make_tiny_data(model, T=3, seed=404)
    fwd = model._forward_pass(X)
    dur_logpmf = model._duration_logpmf()
    assert np.allclose(fwd.entry_mass[1:], dur_logpmf[None, :, :], atol=1e-10), (
        "entry_mass[t>=1] is not exactly dur_logpmf: trans_in may have been reintroduced")
