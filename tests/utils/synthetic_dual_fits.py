"""Fitted DualTimescaleHSSM engines on the synthetic dual-timescale generator, cached per process.

Used by the A2 (latent recovery / calibration) and A6 (information sets) tests so each expensive fit
runs once. Settings reproduce the two OLD tests exactly (tests/test_dual_model_verification.py, tests 39
and 41), as pre-declared in tests/PREDECLARED_CRITERIA_WP2.md section 1.1:

  cell "A": seed 202, fit(n_iter=25)   (old test 39's data and fit)
  cell "B": seed 404, fit(n_iter=20)   (old test 41's data and fit)

The generator gives every regime the SAME loading matrix, the same A_fast = 0.75 I and Q_fast = 0.3 I and
differs only in an observation offset (ledger section 4.5), while the model class fits per-regime
C_k, A_k, Q_k. Cached objects must not be mutated by tests: deep-copy first.

The broken engines below exist only to show that the new tests can fail (negative controls).
"""
import dataclasses
import functools

import numpy as np

from backbone.hssm.dual_model import DualTimescaleHSSM
from data.synthetic_dual import generate_synthetic_dual_timescale_data

CELLS = {"A": dict(seed=202, n_iter=25), "B": dict(seed=404, n_iter=20)}


class EngineFilteredSmoother(DualTimescaleHSSM):
    """B1: the smoother aggregates regimes with the FILTERED posterior instead of the smoothed one
    (the repo's mutation M3)."""

    def _switching_smoother(self, fwd, bwd):
        bwd_filtered = dataclasses.replace(bwd, regime_posterior_smoothed=fwd.regime_post_filtered)
        return super()._switching_smoother(fwd, bwd_filtered)


class EngineADrivenToZero(DualTimescaleHSSM):
    """B2: after every M-step A_fast is multiplied by 0.1, driving the fast dynamics toward 0
    (the symptom of the original pairing bug found by the reviewer)."""

    def _m_step_continuous(self, X, bwd, smoother):
        super()._m_step_continuous(X, bwd, smoother)
        self.A_fast = 0.1 * self.A_fast


ENGINES = {
    "real": DualTimescaleHSSM,
    "B1_filtered_smoother": EngineFilteredSmoother,
    "B2_A_fast_to_zero": EngineADrivenToZero,
}


@functools.lru_cache(maxsize=None)
def synthetic_data(cell: str):
    cfg = CELLS[cell]
    return generate_synthetic_dual_timescale_data(
        n_regimes=2, n_features=4, latent_dim=2, n_segments=4, mean_dwell=15.0, seed=cfg["seed"])


@functools.lru_cache(maxsize=None)
def fitted(engine: str, cell: str):
    """Returns (data, model, m_t, m_t_std, p_t, p_t_post) for one engine on one cell."""
    cfg = CELLS[cell]
    data = synthetic_data(cell)
    model = ENGINES[engine](n_regimes=2, n_features=4, latent_dim=2, max_duration=25, seed=cfg["seed"])
    model.fit(data.X, n_iter=cfg["n_iter"])
    m_t, m_t_std, p_t, p_t_post = model.estimate_states(data.X)
    return data, model, m_t, m_t_std, p_t, p_t_post
