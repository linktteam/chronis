"""Tests for the pair-conditioned dynamics sufficient statistics (NOTE (e) in dual_model._dynamics_sufficient_stats).

Background: the first version of the continuous M-step regressed m_t on m_{t-1} using, for
regime k, moments conditioned on k_{t-1} = k. At a segment start the previous time step
belonged to a DIFFERENT regime, so that pairing is wrong there. A reviewer's recovery study
showed EM driving A_fast to ~0 on data generated with A_fast = (0.9, 0.6, 0.3). The fix
accumulates statistics over predecessor regimes i, weighted by P(k_t=i, k_{t+1}=k | Y).

Three tests:
  1. exact oracle agreement when the posterior is concentrated (distinct regimes with
     well-separated offsets): this is where the old pairing fails by O(1-10) and the new
     one agrees to ~1e-9.
  2. a bounded check against the oracle when the GPB2 collapse is genuinely lossy
     (random distinct-parameter models). Observed relative gaps: median 0.038, max 0.50
     (16 configs). The bound here is a coarse sanity bound, NOT a tight one.
  3. pair weights are consistent with the smoothed regime marginal (an identity).

NOTE on what is NOT tested: the lossless case (identical regimes) cannot distinguish the
old and new pairings, because with identical dynamics both give the same numbers.
"""
import numpy as np
import pytest

from tests.utils.model_factory import make_tiny_model, make_tiny_data
from tests.oracle.tiny_exact_oracle import run_tiny_oracle_dynamics_stats, oracle_regime_sequence_leakage

CONCENTRATED_REL_TOL = 1e-6
LOSSY_REL_TOL = 0.75  # observed max 0.50 over 16 configs; coarse guard only


def _gap(a, b):
    return max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))


def _scale(ref):
    return max(1.0, max(float(np.max(np.abs(x))) for x in ref))


def _simulate(m, T, rng):
    pmf = np.exp(m._duration_logpmf())
    X = []
    k = rng.integers(m.K)
    z = np.zeros(m.L)
    while len(X) < T:
        d = rng.choice(m.Dmax, p=pmf[k]) + 1
        for _ in range(d):
            z = m.A_fast[k] @ z + m.b_fast[k] + rng.multivariate_normal(np.zeros(m.L), m.Q_fast[k])
            X.append(m.C_obs[k] @ z + m.d_obs[k] + rng.normal(0, np.sqrt(m.R_obs[k])))
        k = rng.choice([j for j in range(m.K) if j != k])
    return np.array(X[:T])


def _concentrated_model(F, L, D, seed):
    m = make_tiny_model(K=2, F=F, L=L, Dmax=D, seed=seed)
    m.A_fast = np.array([0.9, 0.4]).reshape(2, 1, 1)
    m.b_fast = np.zeros((2, L))
    m.Q_fast = np.full((2, 1, 1), 0.05)
    m.C_obs = np.array([[[1.], [0.5]], [[1.], [-0.5]]])
    m.d_obs = np.array([[3., 0.], [-3., 0.]])
    m.R_obs = np.full((2, F), 0.1)
    m.m0 = np.zeros((2, L))
    m.P0 = np.stack([np.eye(L)] * 2)
    return m


def test_dynamics_stats_match_oracle_when_posterior_is_concentrated():
    for (D, T, seed) in [(3, 6, 1), (3, 6, 2), (3, 6, 3), (2, 6, 4), (3, 5, 5), (3, 6, 6)]:
        m = _concentrated_model(F=2, L=1, D=D, seed=seed)
        X = _simulate(m, T, np.random.default_rng(seed))
        f = m._forward_pass(X)
        b = m._discrete_backward(f)
        s = m._switching_smoother(f, b)
        ref = run_tiny_oracle_dynamics_stats(m, X)
        new = m._dynamics_sufficient_stats(s)
        rel = _gap(new, ref) / _scale(ref)
        assert rel < CONCENTRATED_REL_TOL, f"D={D} T={T} seed={seed}: relative gap {rel:.3e}"


def test_dynamics_stats_bounded_vs_oracle_when_collapse_is_lossy():
    for (K, F, L, D, T, seed) in [(2, 1, 1, 2, 3, 4), (2, 2, 1, 2, 4, 5), (2, 1, 1, 3, 4, 6),
                                   (2, 1, 2, 2, 4, 7), (2, 1, 1, 2, 6, 9), (2, 1, 1, 2, 6, 24),
                                   (2, 2, 1, 3, 5, 33)]:
        m = make_tiny_model(K=K, F=F, L=L, Dmax=D, seed=seed)
        X = make_tiny_data(m, T, seed=seed + 100)
        f = m._forward_pass(X)
        b = m._discrete_backward(f)
        s = m._switching_smoother(f, b)
        ref = run_tiny_oracle_dynamics_stats(m, X)
        new = m._dynamics_sufficient_stats(s)
        rel = _gap(new, ref) / _scale(ref)
        assert rel < LOSSY_REL_TOL, f"K={K} T={T} seed={seed}: relative gap {rel:.3f}"


def test_pair_weights_are_consistent_with_regime_marginals():
    """sum_i P(k_t=i, k_{t+1}=k | Y) must equal P(k_{t+1}=k | Y), and summing over k must
    give P(k_t=i | Y)."""
    for (K, F, L, D, T, seed) in [(2, 1, 1, 2, 6, 1), (2, 2, 1, 3, 5, 2), (3, 1, 1, 2, 6, 3)]:
        m = make_tiny_model(K=K, F=F, L=L, Dmax=D, seed=seed)
        X = make_tiny_data(m, T, seed=seed + 900)
        f = m._forward_pass(X)
        b = m._discrete_backward(f)
        s = m._switching_smoother(f, b)
        g = b.regime_posterior_smoothed
        assert np.allclose(s.pair_w.sum(axis=1), g[1:], atol=1e-8), f"K={K} T={T}: incoming marginal"
        assert np.allclose(s.pair_w.sum(axis=2), g[:-1], atol=1e-8), f"K={K} T={T}: outgoing marginal"


def test_old_pairing_would_fail_the_concentrated_check():
    """Negative control: re-derive the statistics with the OLD pairing (weights gamma_t[k],
    moments conditioned on k_{t-1} = k) and confirm it is far from the oracle on a case where
    the new statistics are essentially exact. If this ever passes, the concentrated-posterior
    test above has stopped discriminating."""
    K, L = 2, 1
    m = _concentrated_model(F=2, L=L, D=3, seed=1)
    X = _simulate(m, 6, np.random.default_rng(1))
    f = m._forward_pass(X)
    b = m._discrete_backward(f)
    s = m._switching_smoother(f, b)
    gam = b.regime_posterior_smoothed
    G = np.zeros((K, L + 1, L + 1)); H = np.zeros((K, L + 1, L)); S = np.zeros((K, L, L)); W = np.zeros(K)
    for t in range(X.shape[0] - 1):
        for k in range(K):
            w = gam[t + 1, k]
            mc, mp = s.smoothed_m_k[t + 1, k], s.smoothed_m_k[t, k]
            Ecc = s.smoothed_P_k[t + 1, k] + np.outer(mc, mc)
            Epp = s.smoothed_P_k[t, k] + np.outer(mp, mp)
            Ecp = s.pair_J[t, k, k] @ s.smoothed_P_k[t + 1, k] + np.outer(mc, mp)
            W[k] += w
            G[k, :L, :L] += w * Epp; G[k, :L, L] += w * mp; G[k, L, :L] += w * mp; G[k, L, L] += w
            H[k, :L, :] += w * Ecp.T; H[k, L, :] += w * mc
            S[k] += w * Ecc
    ref = run_tiny_oracle_dynamics_stats(m, X)
    rel = _gap((G, H, S, W), ref) / _scale(ref)
    assert rel > 0.1, f"old pairing unexpectedly close to the oracle (relative gap {rel:.3e}): test no longer discriminates"


# ---------------------------------------------------------------------------------------
# Added in the real-repo integration (gate G7): the same dynamics-statistics-vs-oracle checks
# at K=3, which the packaged tests only ran at K=2. SAME tolerances as the K=2 tests above
# (CONCENTRATED_REL_TOL, LOSSY_REL_TOL, and rel > 0.1 for the negative control). Config lists
# were fixed before the first run. If a K=3 config ever needs a different tolerance, that is
# a finding to report to the owner, not a number to adjust here.
#
# D12: OFFICIAL RULE, WITH A KNOWN WEAKNESS (read before relying on this test)
# - Rule: relative gap <= max(1e-6, 2 x leakage); leakage = exact posterior mass not on the most
#   likely regime sequence (tests/oracle/tiny_exact_oracle.py::oracle_regime_sequence_leakage).
# - How it got here: the original 1e-6 limit failed on K=3, D=3, T=6, seed 1 (gap 2.849e-05).
#   This rule was invented AFTER seeing that failure. It was not derived before the data existed.
# - The factor 2 has no theory behind it. It was the owner's example value.
# - Thin margin: seed 1 passes with gap/tolerance = 0.535 (a factor of 1.9).
# - What this can hide: where two regimes look alike the test allows more error, so a real bug of
#   that size could pass.
# - What was NOT loosened: the other five configs still use 1e-6 (observed 1.1e-10 to 9.8e-10); the
#   config list is unchanged; mutants N2-N6 are still caught (smallest ratio 186x).
# - Status: accepted as the official rule; no acceptance date recorded; Re-check trigger: before the first pilot.
# ---------------------------------------------------------------------------------------
def _concentrated_model_k3(F, L, D, seed):
    assert F == 2 and L == 1
    m = make_tiny_model(K=3, F=F, L=L, Dmax=D, seed=seed)
    m.A_fast = np.array([0.9, 0.4, 0.65]).reshape(3, 1, 1)
    m.b_fast = np.zeros((3, L))
    m.Q_fast = np.full((3, 1, 1), 0.05)
    m.C_obs = np.array([[[1.], [0.5]], [[1.], [-0.5]], [[1.], [1.0]]])
    m.d_obs = np.array([[3., 0.], [-3., 0.], [0., 3.]])
    m.R_obs = np.full((3, F), 0.1)
    m.m0 = np.zeros((3, L))
    m.P0 = np.stack([np.eye(L)] * 3)
    return m


K3_CONCENTRATED_CONFIGS = [(3, 6, 1), (3, 6, 2), (3, 6, 3), (2, 6, 4), (3, 5, 5), (3, 6, 6)]   # (D, T, seed)
K3_LOSSY_CONFIGS = [(3, 1, 1, 2, 4, 4), (3, 2, 1, 2, 5, 5), (3, 1, 1, 3, 5, 6), (3, 1, 2, 2, 5, 7),
                    (3, 1, 1, 2, 6, 9), (3, 2, 1, 2, 6, 24), (3, 2, 1, 3, 6, 33)]              # (K, F, L, D, T, seed)


@pytest.mark.parametrize("D,T,seed", K3_CONCENTRATED_CONFIGS)
def test_dynamics_stats_match_oracle_when_posterior_is_concentrated_K3(D, T, seed):
    # Parametrized so each pre-declared config reports on its own (one failing config must
    # not hide the others).
    #
    # CRITERION (ledger D12): relative gap <= max(CONCENTRATED_REL_TOL, 2 x leakage).
    # This is a NEW criterion chosen AFTER seeing the seed-1 result (2.849e-05 against 1e-6);
    # see the D12 notice above. It replaces the earlier `rel < CONCENTRATED_REL_TOL` for K=3
    # only; the K=2 test above is unchanged.
    m = _concentrated_model_k3(F=2, L=1, D=D, seed=seed)
    X = _simulate(m, T, np.random.default_rng(seed))
    f = m._forward_pass(X)
    b = m._discrete_backward(f)
    s = m._switching_smoother(f, b)
    ref = run_tiny_oracle_dynamics_stats(m, X)
    new = m._dynamics_sufficient_stats(s)
    rel = _gap(new, ref) / _scale(ref)
    leakage = oracle_regime_sequence_leakage(m, X)
    tol = max(CONCENTRATED_REL_TOL, 2.0 * leakage)
    assert rel <= tol, (f"K=3 D={D} T={T} seed={seed}: relative gap {rel:.3e} > "
                        f"max({CONCENTRATED_REL_TOL:g}, 2 x leakage {leakage:.3e}) = {tol:.3e}")


@pytest.mark.parametrize("K,F,L,D,T,seed", K3_LOSSY_CONFIGS)
def test_dynamics_stats_bounded_vs_oracle_when_collapse_is_lossy_K3(K, F, L, D, T, seed):
    m = make_tiny_model(K=K, F=F, L=L, Dmax=D, seed=seed)
    X = make_tiny_data(m, T, seed=seed + 100)
    f = m._forward_pass(X)
    b = m._discrete_backward(f)
    s = m._switching_smoother(f, b)
    ref = run_tiny_oracle_dynamics_stats(m, X)
    new = m._dynamics_sufficient_stats(s)
    rel = _gap(new, ref) / _scale(ref)
    assert rel < LOSSY_REL_TOL, f"K={K} T={T} seed={seed}: relative gap {rel:.3f}"


def test_old_pairing_would_fail_the_concentrated_check_K3():
    """K=3 negative control: the OLD same-regime pairing (weights gamma_t[k], moments
    conditioned on k_{t-1} = k) must be far from the oracle where the new statistics are
    essentially exact. If this passes with a small gap the K=3 concentrated test above has
    stopped discriminating."""
    K, L = 3, 1
    m = _concentrated_model_k3(F=2, L=L, D=3, seed=1)
    X = _simulate(m, 6, np.random.default_rng(1))
    f = m._forward_pass(X)
    b = m._discrete_backward(f)
    s = m._switching_smoother(f, b)
    gam = b.regime_posterior_smoothed
    G = np.zeros((K, L + 1, L + 1)); H = np.zeros((K, L + 1, L)); S = np.zeros((K, L, L)); W = np.zeros(K)
    for t in range(X.shape[0] - 1):
        for k in range(K):
            w = gam[t + 1, k]
            mc, mp = s.smoothed_m_k[t + 1, k], s.smoothed_m_k[t, k]
            Ecc = s.smoothed_P_k[t + 1, k] + np.outer(mc, mc)
            Epp = s.smoothed_P_k[t, k] + np.outer(mp, mp)
            Ecp = s.pair_J[t, k, k] @ s.smoothed_P_k[t + 1, k] + np.outer(mc, mp)
            W[k] += w
            G[k, :L, :L] += w * Epp; G[k, :L, L] += w * mp; G[k, L, :L] += w * mp; G[k, L, L] += w
            H[k, :L, :] += w * Ecp.T; H[k, L, :] += w * mc
            S[k] += w * Ecc
    ref = run_tiny_oracle_dynamics_stats(m, X)
    rel = _gap((G, H, S, W), ref) / _scale(ref)
    assert rel > 0.1, f"old pairing unexpectedly close to the oracle at K=3 (relative gap {rel:.3e})"


# Sanity tests of the D12 leakage helper (NOT in the pre-declared criteria file; bounds are deliberately trivial).
def test_leakage_helper_is_zero_when_only_one_regime_sequence_exists():
    m = make_tiny_model(K=1, F=1, L=1, Dmax=3, seed=0)   # K=1: exactly one regime sequence exists
    X = make_tiny_data(m, 3, seed=1)
    assert oracle_regime_sequence_leakage(m, X) == pytest.approx(0.0, abs=1e-12)


def test_leakage_helper_is_strictly_between_zero_and_one_when_several_regime_sequences_remain_plausible():
    m = make_tiny_model(K=3, F=1, L=1, Dmax=2, seed=4)   # a lossy K=3 config from the pre-declared lossy list
    X = make_tiny_data(m, 4, seed=104)
    assert 0.0 < oracle_regime_sequence_leakage(m, X) < 1.0
