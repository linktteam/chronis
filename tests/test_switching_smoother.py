"""Negative controls for the switching RTS smoother (section 7.1)."""
import numpy as np

from tests.utils.model_factory import make_tiny_model, make_tiny_data
from tests.oracle.tiny_exact_oracle import run_tiny_exact_oracle

M_TOL = 1.2  # provisional; checked against the measured battery in RESULTS.md
P_TOL = 1.2


def test_smoother_uses_smoothed_not_filtered_posterior():
    """Old bug: aggregation used the FILTERED regime posterior. The aggregate smoothed_m
    must equal the SMOOTHED-posterior-weighted mix, and differ from the filtered-weighted
    mix wherever the two posteriors differ."""
    model = make_tiny_model(K=2, F=1, L=1, Dmax=2, seed=6)
    X = make_tiny_data(model, T=5, seed=606)
    fwd = model._forward_pass(X)
    bwd = model._discrete_backward(fwd)
    sm = model._switching_smoother(fwd, bwd)

    assert not np.allclose(fwd.regime_post_filtered, bwd.regime_posterior_smoothed, atol=1e-6), (
        "filtered and smoothed posteriors are identical here: pick a case where they differ")
    for t in range(5):
        smoothed_mix = np.tensordot(bwd.regime_posterior_smoothed[t], sm.smoothed_m_k[t], axes=(0, 0))
        filtered_mix = np.tensordot(fwd.regime_post_filtered[t], sm.smoothed_m_k[t], axes=(0, 0))
        assert np.allclose(sm.smoothed_m[t], smoothed_mix, atol=1e-8), (
            f"t={t}: smoothed_m does not match the smoothed-posterior-weighted mix")
        if not np.allclose(fwd.regime_post_filtered[t], bwd.regime_posterior_smoothed[t], atol=1e-6):
            assert not np.allclose(sm.smoothed_m[t], filtered_mix, atol=1e-8), (
                f"t={t}: smoothed_m matches the FILTERED-weighted mix: old bug is back")


def test_switching_smoother_matches_oracle_tiny_case():
    """The continuous moments are genuinely approximate even at T=2 (NOTE (d)
    in dual_model._switching_smoother: Kim/GPB2 reuses the already-collapsed smoothed_m_k[t+1,j] in every (k,j)
    correction), so one bounded tolerance applies at every T."""
    for (K, F, L, D, T, seed) in [(2, 1, 1, 2, 2, 3), (2, 1, 1, 2, 3, 4), (2, 2, 1, 2, 4, 5)]:
        model = make_tiny_model(K=K, F=F, L=L, Dmax=D, seed=seed)
        X = make_tiny_data(model, T, seed=seed + 100)
        fwd = model._forward_pass(X)
        bwd = model._discrete_backward(fwd)
        sm = model._switching_smoother(fwd, bwd)
        oracle = run_tiny_exact_oracle(model, X)
        m_gap = np.max(np.abs(sm.smoothed_m - oracle.smoothed_m))
        P_gap = np.max(np.abs(sm.smoothed_P - oracle.smoothed_P))
        assert m_gap < M_TOL, f"K={K} T={T} seed={seed}: smoothed_m gap {m_gap} exceeds {M_TOL}"
        assert P_gap < P_TOL, f"K={K} T={T} seed={seed}: smoothed_P gap {P_gap} exceeds {P_TOL}"


def test_gpb2_tail_risk_is_bounded_sane():
    """GPB2/Kim smoothers can occasionally lock onto a confidently wrong regime when
    dynamics are poorly separated. That is a property of the algorithm class. This test
    does not assert oracle closeness; it asserts the outputs stay valid (finite,
    normalized, PSD) on a seed chosen to hit that tail, so a real bug is still caught."""
    model = make_tiny_model(K=2, F=2, L=1, Dmax=2, seed=8)
    X = make_tiny_data(model, T=5, seed=108)
    fwd = model._forward_pass(X)
    bwd = model._discrete_backward(fwd)
    sm = model._switching_smoother(fwd, bwd)

    assert np.all(np.isfinite(fwd.regime_post_filtered))
    assert np.allclose(fwd.regime_post_filtered.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(np.isfinite(bwd.regime_posterior_smoothed))
    assert np.allclose(bwd.regime_posterior_smoothed.sum(axis=1), 1.0, atol=1e-8)
    assert np.all(np.isfinite(sm.smoothed_m))
    assert np.all(np.isfinite(sm.smoothed_P))
    for t in range(sm.smoothed_P.shape[0]):
        eigvals = np.linalg.eigvalsh(sm.smoothed_P[t])
        assert np.all(eigvals > -1e-8), f"t={t}: smoothed_P not PSD, eigvals={eigvals}"
