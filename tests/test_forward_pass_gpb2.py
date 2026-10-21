"""Negative controls for the corrected GPB2 forward pass (section 6.5).

Each test targets a specific defect described for the old code:
  - the entry-branch collapse weight never used A_slow
  - continuation and entry were conflated and shared one flat emission term
plus oracle agreement.
"""
import numpy as np

from tests.utils.model_factory import make_tiny_model, make_tiny_data
from tests.oracle.tiny_exact_oracle import run_tiny_exact_oracle

EXACT_TOL = 1e-6
T3PLUS_LL_TOL = 0.5  # provisional; checked against the measured battery in RESULTS.md


def test_entry_weight_uses_transition_matrix():
    """Changing only A_slow must change ll_entry. Uses K=3: with K=2 and a zero diagonal
    the single off-diagonal entry per row is forced to 1.0, so it cannot demonstrate this."""
    K, F, L, D, T = 3, 1, 1, 2, 4
    model_a = make_tiny_model(K=K, F=F, L=L, Dmax=D, seed=11)
    model_b = make_tiny_model(K=K, F=F, L=L, Dmax=D, seed=11)
    model_b.A_slow = np.clip(model_a.A_slow + np.array([[0, 0.3, -0.1], [-0.1, 0, 0.3], [0.3, -0.1, 0]]),
                              1e-6, None)
    for i in range(K):
        model_b.A_slow[i, i] = 0.0
        model_b.A_slow[i] /= model_b.A_slow[i].sum()
    assert not np.allclose(model_a.A_slow, model_b.A_slow)

    X = make_tiny_data(model_a, T, seed=211)
    fwd_a = model_a._forward_pass(X)
    fwd_b = model_b._forward_pass(X)
    assert not np.allclose(fwd_a.ll_entry[1:], fwd_b.ll_entry[1:]), (
        "changing A_slow did not change ll_entry: entry branch is not using the transition matrix")


def test_continuation_and_entry_likelihoods_differ():
    """The two branch-specific likelihoods must be tracked separately, not forced equal."""
    any_diff = False
    for seed in range(5):
        model = make_tiny_model(K=2, F=1, L=1, Dmax=2, seed=seed)
        X = make_tiny_data(model, T=4, seed=seed + 300)
        fwd = model._forward_pass(X)
        if not np.allclose(fwd.ll_continuation[1:], fwd.ll_entry[1:], atol=1e-9):
            any_diff = True
    assert any_diff, "ll_continuation and ll_entry were identical across every seed tested"


def test_forward_pass_returns_required_interface():
    model = make_tiny_model(K=2, F=1, L=1, Dmax=2, seed=1)
    X = make_tiny_data(model, T=3, seed=1)
    fwd = model._forward_pass(X)
    assert fwd.ll_continuation.shape == (3, 2)
    assert fwd.ll_entry.shape == (3, 2)
    assert fwd.ll_branch.shape == (3, 2, 2)


def test_forward_loglik_matches_oracle_tiny_case():
    """T<=2: exact (no approximation has had a chance to matter). T>=3: bounded."""
    configs = [
        (2, 1, 1, 2, 2, 3), (1, 1, 1, 3, 3, 0), (2, 1, 1, 2, 3, 4),
        (2, 2, 1, 2, 4, 5), (2, 1, 1, 3, 4, 6),
    ]
    for (K, F, L, D, T, seed) in configs:
        model = make_tiny_model(K=K, F=F, L=L, Dmax=D, seed=seed)
        X = make_tiny_data(model, T, seed=seed + 100)
        fwd = model._forward_pass(X)
        oracle = run_tiny_exact_oracle(model, X)
        tol = EXACT_TOL if T <= 2 else T3PLUS_LL_TOL
        gap = abs(fwd.log_likelihood - oracle.log_likelihood)
        assert gap < tol, f"K={K} T={T} seed={seed}: loglik gap {gap} exceeds tolerance {tol}"
        if T <= 2:
            fgap = np.max(np.abs(fwd.regime_post_filtered - oracle.filtered_regime_posterior))
            assert fgap < EXACT_TOL, f"K={K} T={T} seed={seed}: filtered posterior gap {fgap} at T<=2"


def test_filtered_posterior_is_causal():
    """Filtered posterior at t must not change when only future data changes."""
    model = make_tiny_model(K=2, F=1, L=1, Dmax=2, seed=5)
    rng = np.random.default_rng(0)
    X1 = rng.normal(size=(4, model.F))
    X2 = X1.copy()
    X2[3] += 5.0
    fwd1 = model._forward_pass(X1)
    fwd2 = model._forward_pass(X2)
    assert np.allclose(fwd1.regime_post_filtered[:3], fwd2.regime_post_filtered[:3], atol=1e-10), (
        "filtered posterior at t<3 changed when only X[3] was modified: forward pass is not causal")
