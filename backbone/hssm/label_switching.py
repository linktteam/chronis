"""
MP-02: post-hoc label convention of the baseline engine.

Regimes of one fit are sorted by the ascending L2 norm of each regime's mean vector in the fit's z-scored feature space. This is a DETERMINISTIC LABEL CONVENTION. It is
NOT an activity ordering: after per-person z-scoring the regime closest to the person's own average has the smallest norm, which for three regimes is usually the middle one
(WP8 diagnosis: this order equals the planted low-to-high activity order in 7.7% of fits, chance 16.7%). Do not read "regime 0" of a raw fit as "lowest activity".

Names that mean something across refits are the stable ids of `backbone/hssm/lineage.py`; they are named at a lineage's first accepted fit by the score ladder of
`backbone/hssm/ordering.py` (which says when it cannot be sure) and then inherited by alignment. The raw convention below is kept unchanged so that raw outputs of previously
healthy fits stay bit-identical (WP8 owner decision A4: the raw record is untouched, the stable names are an added column).
"""

from __future__ import annotations
import numpy as np

LABEL_ORDER_CONVENTION = "mean_L2_norm_of_regime_mean_vector"   # what the sort key really is: a label convention, not an activity measure
ACTIVITY_LEVEL_CONVENTION = LABEL_ORDER_CONVENTION                # historical name, kept so that importers keep working; the name over-claims (WP8 Item 4)


def canonicalize_labels(model) -> object:
    """Reorders pi, A, mu, var, dur_mu, dur_sigma in place by ascending L2 norm of
    each regime's mean vector in the fit's z-scored feature space. This is a label
    convention, not an activity ordering (see the module docstring). Returns the
    same model object, mutated, plus a `_label_order_applied` attribute recording
    the permutation used (for audit)."""
    activity = np.linalg.norm(model.mu, axis=1)
    order = np.argsort(activity)

    model.pi = model.pi[order]
    model.A = model.A[np.ix_(order, order)]
    model.mu = model.mu[order]
    model.var = model.var[order]
    model.dur_mu = model.dur_mu[order]
    model.dur_sigma = model.dur_sigma[order]
    if hasattr(model, "duration_optimizer_success_") and model.duration_optimizer_success_:
        model.duration_optimizer_success_ = {
            new_k: model.duration_optimizer_success_.get(int(old_k), True)
            for new_k, old_k in enumerate(order)
        }
    if hasattr(model, "duration_optimizer_messages_") and model.duration_optimizer_messages_:
        model.duration_optimizer_messages_ = {
            new_k: model.duration_optimizer_messages_.get(int(old_k), "")
            for new_k, old_k in enumerate(order)
        }
    model._label_order_applied = order.tolist()
    return model


def canonicalize_regime_order(model) -> object:
    """Legacy alias matching foundation's old label switching interface."""
    return canonicalize_labels(model)


def activity_levels(model) -> np.ndarray:
    """Returns the L2 norm of each regime's mean vector (the sort key of the label
    convention, in current label order). Despite the historical name it is not an
    activity level. Exposed separately so tests/callers can assert the convention
    without recomputing it inline."""
    return np.linalg.norm(model.mu, axis=1)
