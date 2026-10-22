"""DIAGNOSTIC, NOT PART OF THE SUITE (file name does not match test_*.py; run it by path).

Design 1 of the B1-against-oracle check, as pre-declared in tests/PREDECLARED_CRITERIA_WP3.md (dated amendment "B1 against the existing
oracle-based tests"). FIRST RUN (wp3_evidence/B1_new_check_first_run.log): 16 passed, 2 FAILED on the real engine: seed 0 (engine 5.956e-02 vs
filtered-weight mix 5.943e-02, ratio 1.002) and seed 5 (3.359e-01 vs 2.704e-01, ratio 1.242). The claim "the engine's smoothed_m is closer to the exact oracle than
the filtered-weight mix of its own per-regime means" does NOT hold on 2 of the 6 pre-declared configs: the engine's per-regime-mean approximation error
(Kim/GPB2 smoother, see NOTE (d) in dual_model._switching_smoother) can exceed the effect of the weights. The claim was not changed. It is kept here so the result is
reproducible, and it is not a suite test (the brief allows no new xfail). Design 2 (a second, narrower claim declared afterwards) is in tests/test_smoother_oracle_regime_weights.py.
"""

"""B1 against the exact oracle (WP3 item 1, outside N18): does the smoother weight its per-regime means with the SMOOTHED regime posterior?

Why a new check exists. B1 (smoother aggregated with the FILTERED posterior) is NOT seen by any existing exact-oracle comparison: in the lossless-collapse
configs every regime has the same dynamics, so the per-regime means coincide and the weights cannot matter (differences about 1e-16, see
wp3_evidence/B1_design_scan.txt). Only a structural test (test_switching_smoother.py::test_smoother_uses_smoothed_not_filtered_posterior) catches it.

Design and configs: tests/PREDECLARED_CRITERIA_WP3.md, the dated amendment "B1 against the existing oracle-based tests". Configs are the first six seeds
that satisfy two SELECTION conditions only (posteriors differ, regime means differ); no oracle gap was looked at to choose them.

Claim: max_t |smoothed_m - oracle.smoothed_m| < max_t |filtered_mix - oracle.smoothed_m|, where filtered_mix uses the same per-regime means with the FILTERED weights.
Control: a B1 engine makes smoothed_m equal to filtered_mix, so the predicate must fail for it.
"""
import numpy as np
import pytest

from tests.oracle.tiny_exact_oracle import run_tiny_exact_oracle
from tests.utils.model_factory import make_tiny_data, make_tiny_model
from tests.utils.synthetic_dual_fits import EngineFilteredSmoother

CONFIG_SEEDS = (0, 1, 2, 3, 4, 5)
MIN_POSTERIOR_DIFFERENCE = 0.05     # D1
MIN_REGIME_MEAN_SEPARATION = 0.1    # D2


def _case(seed, engine_class=None):
    model = make_tiny_model(K=2, F=1, L=1, Dmax=2, seed=seed)
    if engine_class is not None:
        model.__class__ = engine_class
    X = make_tiny_data(model, 5, seed=seed + 100)
    fwd = model._forward_pass(X)
    bwd = model._discrete_backward(fwd)
    sm = model._switching_smoother(fwd, bwd)
    oracle = run_tiny_exact_oracle(model, X)
    filtered_mix = np.einsum("tk,tkl->tl", fwd.regime_post_filtered, sm.smoothed_m_k)
    gap_engine = float(np.max(np.abs(sm.smoothed_m - oracle.smoothed_m)))
    gap_filtered = float(np.max(np.abs(filtered_mix - oracle.smoothed_m)))
    return fwd, bwd, sm, gap_engine, gap_filtered


@pytest.mark.parametrize("seed", CONFIG_SEEDS)
def test_selection_conditions_hold_for_the_chosen_configs(seed):
    fwd, bwd, sm, _, _ = _case(seed)
    assert np.max(np.abs(fwd.regime_post_filtered - bwd.regime_posterior_smoothed)) >= MIN_POSTERIOR_DIFFERENCE
    assert np.max(np.abs(sm.smoothed_m_k[:, 0, :] - sm.smoothed_m_k[:, 1, :])) >= MIN_REGIME_MEAN_SEPARATION


@pytest.mark.parametrize("seed", CONFIG_SEEDS)
def test_smoothed_regime_weights_land_closer_to_the_exact_oracle_than_filtered_weights(seed):
    _, _, _, gap_engine, gap_filtered = _case(seed)
    assert gap_engine < gap_filtered, (
        f"seed {seed}: the engine's smoothed_m is {gap_engine:.4e} from the exact oracle, not closer than the filtered-weight mix "
        f"({gap_filtered:.4e}); ratio {gap_engine / gap_filtered:.3f}")


@pytest.mark.parametrize("seed", CONFIG_SEEDS)
def test_control_filtered_smoother_engine_is_not_closer_than_the_filtered_mix(seed):
    """Negative control: with B1 the engine's smoothed_m IS the filtered-weight mix, so the strict claim above must fail for it."""
    _, _, _, gap_engine, gap_filtered = _case(seed, engine_class=EngineFilteredSmoother)
    assert not (gap_engine < gap_filtered), f"seed {seed}: B1 engine ratio {gap_engine / gap_filtered:.3f}; the check cannot see B1"
