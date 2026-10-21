"""Truth-start recovery test for the continuous M-step at K >= 2.

Background: a reviewer's recovery study found that EM drives A_fast to ~0 on data generated
with A_fast = (0.9, 0.6, 0.3). Started AT the true parameters, one raw M-step moved A_fast
to (0.11, 0.06, 0.01) and dropped the log-likelihood by 33 nats. Every earlier test passed
because the K=1 check is exact and the oracle has no M-step reference.

This test encodes that case. At the true parameters, with plenty of data, a single EM step
must leave the parameters near the truth: the expected complete-data statistics under the
true posterior reproduce the true parameters, so the M-step is (up to sampling noise and
the approximate E-step) a fixed point there.

CRITERIA WERE FIXED BEFORE THE FIX WAS WRITTEN and are not to be loosened to make a failing
run pass:
  - max_k |A_fast[k] - truth[k]| < 0.15     (old code: errors of 0.2-0.8 at T=900)
  - log-likelihood after the step >= log-likelihood before it - 2.0 nats
    (exact EM cannot decrease it; 2 nats of slack is for the approximate GPB2 E-step)
"""
import copy

import numpy as np

from tests.utils.planted import planted_model, simulate, TRUE_A_FAST

A_TOL = 0.15
LL_DROP_TOL = 2.0
T = 300
SEEDS = (1, 2)


def _one_raw_em_step(model, X):
    """E-step then M-step with NO backtracking guard, so a bad update cannot hide."""
    fwd = model._forward_pass(X)
    bwd = model._discrete_backward(fwd)
    sm = model._switching_smoother(fwd, bwd)
    ll_before = float(fwd.log_likelihood)
    model._update_discrete_params(bwd)
    model._m_step_continuous(X, bwd, sm)
    ll_after = float(model._forward_pass(X).log_likelihood)
    return ll_before, ll_after


def test_one_em_step_from_truth_stays_near_truth():
    for seed in SEEDS:
        truth = planted_model()
        X, _, _ = simulate(truth, T, np.random.default_rng(seed))
        model = copy.deepcopy(truth)
        ll_before, ll_after = _one_raw_em_step(model, X)

        a_hat = model.A_fast.ravel()
        a_err = np.abs(a_hat - TRUE_A_FAST)
        assert np.max(a_err) < A_TOL, (
            f"seed={seed}: one EM step from the truth moved A_fast to {np.round(a_hat, 3)} "
            f"(truth {TRUE_A_FAST}); max error {np.max(a_err):.3f} >= {A_TOL}")
        assert ll_after >= ll_before - LL_DROP_TOL, (
            f"seed={seed}: log-likelihood fell {ll_before - ll_after:.2f} nats in one EM step "
            f"from the truth ({ll_before:.1f} -> {ll_after:.1f}); tolerance {LL_DROP_TOL}")
