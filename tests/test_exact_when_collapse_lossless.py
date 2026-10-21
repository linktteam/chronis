"""Exactness at T > 2 when the GPB2 collapse is lossless.

The main oracle comparisons show gaps at T >= 3 that are attributed to GPB2 moment-matching.
This test separates "approximation" from "bug": when every regime shares IDENTICAL continuous
parameters (A, b, Q, m0, P0, C, d, R), all Gaussian components being collapsed are identical,
so the collapse loses nothing and every quantity must match the brute-force oracle at any T.
Duration parameters and A_slow still differ per regime, so the discrete layer is exercised
with genuinely regime-dependent structure.

Tolerances: 1e-8 for discrete quantities and means. Covariances get 1e-6, reflecting the
documented ~1e-6 diagonal jitter in the unchanged `_kalman_update_regime` (observed gaps are
~1e-8; the tolerance is not tight to the observation on purpose).
"""
import numpy as np

from tests.utils.model_factory import make_tiny_model, make_tiny_data
from tests.oracle.tiny_exact_oracle import run_tiny_exact_oracle

DISCRETE_TOL = 1e-8
COV_TOL = 1e-6


def _shared_dynamics_model(K, F, L, D, seed):
    m = make_tiny_model(K=K, F=F, L=L, Dmax=D, seed=seed)
    for name in ["A_fast", "b_fast", "Q_fast", "m0", "P0", "C_obs", "d_obs", "R_obs"]:
        arr = getattr(m, name)
        for k in range(1, K):
            arr[k] = arr[0]
    return m


def test_everything_matches_oracle_at_T_up_to_6_when_collapse_is_lossless():
    configs = [(2, 1, 1, 2, 3, 1), (2, 1, 1, 2, 4, 2), (2, 2, 1, 3, 5, 3),
               (2, 1, 2, 2, 5, 4), (2, 1, 1, 3, 6, 5), (2, 2, 2, 2, 6, 6)]
    for (K, F, L, D, T, seed) in configs:
        m = _shared_dynamics_model(K, F, L, D, seed)
        X = make_tiny_data(m, T, seed=seed + 700)
        fwd = m._forward_pass(X)
        bwd = m._discrete_backward(fwd)
        sm = m._switching_smoother(fwd, bwd)
        o = run_tiny_exact_oracle(m, X)
        tag = f"K={K} F={F} L={L} D={D} T={T} seed={seed}"

        assert abs(fwd.log_likelihood - o.log_likelihood) < DISCRETE_TOL, f"{tag}: log-likelihood"
        assert np.max(np.abs(fwd.regime_post_filtered - o.filtered_regime_posterior)) < DISCRETE_TOL, f"{tag}: filtered"
        assert np.max(np.abs(bwd.regime_posterior_smoothed - o.smoothed_regime_posterior)) < DISCRETE_TOL, f"{tag}: smoothed"
        assert np.max(np.abs(bwd.xi_t - o.pairwise_transition_marginal)) < DISCRETE_TOL, f"{tag}: pairwise xi_t"
        assert np.max(np.abs(sm.smoothed_m - o.smoothed_m)) < DISCRETE_TOL, f"{tag}: smoothed mean"
        assert np.max(np.abs(sm.smoothed_P - o.smoothed_P)) < COV_TOL, f"{tag}: smoothed covariance"


# ---------------------------------------------------------------------------------------
# Added in the real-repo integration (gate G5): the same lossless-collapse exactness check at
# K=3 and K=4, which the packaged tests never exercised (oracle guard was K<=2). Same
# tolerances as above. Configs were fixed before the first run and are not to be pruned.
# ---------------------------------------------------------------------------------------
LOSSLESS_K3_K4_CONFIGS = [
    # (K, F, L, Dmax, T, seed)
    (3, 1, 1, 2, 4, 1), (3, 1, 1, 3, 5, 2), (3, 2, 1, 2, 6, 3), (3, 2, 2, 3, 6, 4), (3, 1, 2, 2, 6, 7),
    (4, 1, 1, 2, 5, 5), (4, 2, 1, 3, 6, 6),
]


def test_everything_matches_oracle_when_collapse_is_lossless_K3_K4():
    for (K, F, L, D, T, seed) in LOSSLESS_K3_K4_CONFIGS:
        m = _shared_dynamics_model(K, F, L, D, seed)
        X = make_tiny_data(m, T, seed=seed + 700)
        fwd = m._forward_pass(X)
        bwd = m._discrete_backward(fwd)
        sm = m._switching_smoother(fwd, bwd)
        o = run_tiny_exact_oracle(m, X)
        tag = f"K={K} F={F} L={L} D={D} T={T} seed={seed}"

        assert abs(fwd.log_likelihood - o.log_likelihood) < DISCRETE_TOL, f"{tag}: log-likelihood"
        assert np.max(np.abs(fwd.regime_post_filtered - o.filtered_regime_posterior)) < DISCRETE_TOL, f"{tag}: filtered"
        assert np.max(np.abs(bwd.regime_posterior_smoothed - o.smoothed_regime_posterior)) < DISCRETE_TOL, f"{tag}: smoothed"
        assert np.max(np.abs(bwd.xi_t - o.pairwise_transition_marginal)) < DISCRETE_TOL, f"{tag}: pairwise xi_t"
        assert np.max(np.abs(sm.smoothed_m - o.smoothed_m)) < DISCRETE_TOL, f"{tag}: smoothed mean"
        assert np.max(np.abs(sm.smoothed_P - o.smoothed_P)) < COV_TOL, f"{tag}: smoothed covariance"


def test_oracle_size_guard_allows_k4_and_rejects_runaway_sizes():
    """The relaxed guard must still refuse sizes beyond T<=6, K<=4, Dmax<=3 and any problem
    whose enumerated path count exceeds the cap (so it cannot run away)."""
    import pytest
    from tests.oracle.tiny_exact_oracle import (count_legal_paths, enumerate_legal_paths,
                                                MAX_ORACLE_PATHS)
    for size in [(5, 2, 3), (4, 3, 3), (3, 4, 2)]:
        assert count_legal_paths(*size) == sum(1 for _ in enumerate_legal_paths(*size)), size
    m = _shared_dynamics_model(5, 1, 1, 2, seed=1)  # K=5 > MAX_K
    with pytest.raises(AssertionError):
        run_tiny_exact_oracle(m, make_tiny_data(m, 3, seed=1))
    m = _shared_dynamics_model(4, 1, 1, 3, seed=1)
    with pytest.raises(AssertionError):             # T=7 > MAX_T
        run_tiny_exact_oracle(m, make_tiny_data(m, 7, seed=1))
    assert count_legal_paths(6, 4, 3) <= MAX_ORACLE_PATHS
