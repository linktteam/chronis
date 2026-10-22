"""DIAGNOSTIC, NOT PART OF THE SUITE (file name does not match test_*.py; run it by path).

Design 2 of the B1-against-oracle check (pre-declared in tests/PREDECLARED_CRITERIA_WP3.md, dated section "DESIGN 2 ... ONE revision only"). FIRST RUN
(wp3_evidence/B1_design2_first_run.log): 16 passed, 2 FAILED on the real engine: seed 0 (ratio 0.278, limit 0.25) and seed 3 (ratio 1.602). Seed 3 shows that the engine's
own smoothed regime posterior differs from the oracle's by 0.497 (max abs), far more than the filtered-vs-smoothed difference B1 would introduce, i.e. at these sizes the
discrete smoothing approximation is as large as the effect of B1 (wp3_evidence/B1_design_ratios.txt). Per the pre-declared STOP RULE no third design was tried and the claim
was not changed. It is not a suite test (the brief allows no new xfail). The only thing that catches B1 in the suite is the existing structural test
tests/test_switching_smoother.py::test_smoother_uses_smoothed_not_filtered_posterior.
"""

"""B1 against the exact oracle (WP3 item 1, outside N18), DESIGN 2: does the smoother aggregate its per-regime means with the SMOOTHED regime posterior?

Why a new check exists. B1 (smoother aggregated with the FILTERED posterior) is NOT seen by any existing exact-oracle comparison: in the lossless-collapse
configs every regime has the same dynamics, so per-regime means coincide and the weights cannot matter (differences about 1e-16, wp3_evidence/B1_design_scan.txt).
Only a structural test (test_switching_smoother.py::test_smoother_uses_smoothed_not_filtered_posterior) catches it.

Design 1 (is smoothed_m closer to the oracle than the filtered-weight mix?) failed on the real engine on 2 of 6 configs because the engine's own per-regime-mean error
can exceed the effect of the weights; it is kept, unedited, in tests/diagnostics/. Design 2 isolates the weights: both sides use the engine's OWN per-regime means and
the comparison is against the mix that uses the oracle's EXACT smoothed regime posterior.

Pre-declared in tests/PREDECLARED_CRITERIA_WP3.md (dated sections on B1). Configs: the first six seeds (0..5) satisfying the two SELECTION conditions D1, D2.
Claim: g_engine <= 0.25 * g_filtered, with
    mix_oracle[t] = sum_k oracle.smoothed_regime_posterior[t,k] * smoothed_m_k[t,k]
    g_engine      = max_t |smoothed_m - mix_oracle|
    g_filtered    = max_t |filtered_mix - mix_oracle|,  filtered_mix[t] = sum_k filtered_post[t,k] * smoothed_m_k[t,k]
Control: the B1 engine makes smoothed_m equal filtered_mix (ratio exactly 1) so the claim must fail for it.
"""
import numpy as np
import pytest

from tests.oracle.tiny_exact_oracle import run_tiny_exact_oracle
from tests.utils.model_factory import make_tiny_data, make_tiny_model
from tests.utils.synthetic_dual_fits import EngineFilteredSmoother

CONFIG_SEEDS = (0, 1, 2, 3, 4, 5)
MIN_POSTERIOR_DIFFERENCE = 0.05     # D1
MIN_REGIME_MEAN_SEPARATION = 0.1    # D2
RATIO_LIMIT = 0.25


def _case(seed, engine_class=None):
    model = make_tiny_model(K=2, F=1, L=1, Dmax=2, seed=seed)
    if engine_class is not None:
        model.__class__ = engine_class
    X = make_tiny_data(model, 5, seed=seed + 100)
    fwd = model._forward_pass(X)
    bwd = model._discrete_backward(fwd)
    sm = model._switching_smoother(fwd, bwd)
    oracle = run_tiny_exact_oracle(model, X)
    mix_oracle = np.einsum("tk,tkl->tl", oracle.smoothed_regime_posterior, sm.smoothed_m_k)
    filtered_mix = np.einsum("tk,tkl->tl", fwd.regime_post_filtered, sm.smoothed_m_k)
    g_engine = float(np.max(np.abs(sm.smoothed_m - mix_oracle)))
    g_filtered = float(np.max(np.abs(filtered_mix - mix_oracle)))
    return fwd, bwd, sm, g_engine, g_filtered


@pytest.mark.parametrize("seed", CONFIG_SEEDS)
def test_selection_conditions_hold_for_the_chosen_configs(seed):
    fwd, bwd, sm, _, _ = _case(seed)
    assert np.max(np.abs(fwd.regime_post_filtered - bwd.regime_posterior_smoothed)) >= MIN_POSTERIOR_DIFFERENCE
    assert np.max(np.abs(sm.smoothed_m_k[:, 0, :] - sm.smoothed_m_k[:, 1, :])) >= MIN_REGIME_MEAN_SEPARATION


@pytest.mark.parametrize("seed", CONFIG_SEEDS)
def test_smoother_weights_its_regime_means_with_the_exact_smoothed_posterior_not_the_filtered_one(seed):
    _, _, _, g_engine, g_filtered = _case(seed)
    assert g_filtered > 0.0
    assert g_engine <= RATIO_LIMIT * g_filtered, (
        f"seed {seed}: smoothed_m is {g_engine:.4e} from the oracle-weighted mix of its own regime means, filtered weights give {g_filtered:.4e}; "
        f"ratio {g_engine / g_filtered:.3f} (needs <= {RATIO_LIMIT})")


@pytest.mark.parametrize("seed", CONFIG_SEEDS)
def test_control_filtered_smoother_engine_violates_the_claim(seed):
    """Negative control: with B1, smoothed_m IS the filtered-weight mix, so the ratio is 1 and the claim must fail."""
    _, _, _, g_engine, g_filtered = _case(seed, engine_class=EngineFilteredSmoother)
    assert g_filtered > 0.0
    assert not (g_engine <= RATIO_LIMIT * g_filtered), f"seed {seed}: B1 engine ratio {g_engine / g_filtered:.3f}; the check cannot see B1"
