"""WP9 A1: multi-start consensus of one baseline fit (design and thresholds: tests/PREDECLARED_CRITERIA_WP9.md section 6).

Problem (WP8, criterion 3): the same data refitted with another random seed gave a different partition in 31% of the windows, so the stable names of a lineage could differ between two
runs of the same system. The cause is local optima and, for weakly separated regimes, a flat likelihood between different partitions (WP9 diagnosis, `wp9_evidence/a1_diag_analysis.txt`).
Rule C: take the healthy restarts (converged and not collapsed), order them by log-likelihood (ties: lower restart index first), take the best `m`. The fit is FIT_UNSTABLE when fewer than
`m` healthy restarts exist or when the smallest pairwise adjusted Rand index among those `m` is below `tau`.
No restart is added, nothing about the selected model changes: this is a flag (additive metadata). `lineage.step` never serves a FIT_UNSTABLE fit as ALIGNED.
The adjusted Rand index function is passed in (`fitting.adjusted_rand_index`) so that this module does not import `fitting`.
"""
from __future__ import annotations

from typing import Callable, Sequence

from .config import FIT_CONSENSUS_M, FIT_CONSENSUS_TAU, FIT_MIN_POSTERIOR_SHARPNESS

REASON_TOO_FEW = "fewer_than_m_healthy_restarts"
REASON_DISAGREE = "best_restarts_disagree"
REASON_NOT_ASSESSED = "stability_not_assessed"
REASON_NOT_SHARP = "low_posterior_sharpness"
KEYS = ("consensus_m", "consensus_tau", "n_healthy_restarts", "consensus_min_ari", "fit_unstable", "fit_unstable_reason")


def assess_consensus(records: Sequence[dict], ari: Callable, m: int = FIT_CONSENSUS_M, tau: float = FIT_CONSENSUS_TAU) -> dict:
    """`records`: the CONVERGED restarts of one fit in restart order, each with `ll`, `collapsed` and `labels` (decoded day labels). Returns the six additive keys of `KEYS`."""
    if int(m) != m or m < 2:
        raise ValueError("the consensus needs m >= 2 restarts to compare")
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0, 1]")
    healthy = [(i, r) for i, r in enumerate(records) if not r["collapsed"]]
    healthy.sort(key=lambda t: (-t[1]["ll"], t[0]))
    top = [r for _, r in healthy[:m]]
    out = dict(consensus_m=int(m), consensus_tau=float(tau), n_healthy_restarts=len(healthy), consensus_min_ari=None, fit_unstable=True, fit_unstable_reason=REASON_TOO_FEW)
    if len(top) < m:
        return out
    min_ari = min(float(ari(top[i]["labels"], top[j]["labels"])) for i in range(m) for j in range(i + 1, m))
    unstable = min_ari < tau
    out.update(consensus_min_ari=min_ari, fit_unstable=bool(unstable), fit_unstable_reason=REASON_DISAGREE if unstable else None)
    return out


def not_assessed() -> dict:
    """Keys for a fit whose stability cannot be assessed (the selected restart is collapsed: such a fit is withheld anyway). Fail closed: the lineage treats it as FIT_UNSTABLE."""
    return dict(consensus_m=int(FIT_CONSENSUS_M), consensus_tau=float(FIT_CONSENSUS_TAU), n_healthy_restarts=0, consensus_min_ari=None, fit_unstable=None, fit_unstable_reason=REASON_NOT_ASSESSED)


def posterior_sharpness(p_t_posterior) -> float | None:
    """Mean over the rows of the largest regime posterior (1.0 = every day is assigned with certainty). None when the posterior is missing, empty or not finite (nothing known)."""
    import numpy as np
    if p_t_posterior is None:
        return None
    p = np.asarray(p_t_posterior, dtype=float)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] == 0 or not np.isfinite(p).all():
        return None
    return float(p.max(axis=1).mean())


# EXPERT_REVIEW(identity-distance): applies the posterior-sharpness floor (`FIT_MIN_POSTERIOR_SHARPNESS`), the second part of the identifiability gate of WP9 A2.
def sharpness_unstable(p_t_posterior, floor: float = FIT_MIN_POSTERIOR_SHARPNESS) -> bool:
    """WP9 A2: True iff the posterior is known and its mean maximum is strictly below `floor` (`floor <= 0` switches the check off). A posterior that is missing or not finite is NOT flagged here:
    the consensus flag and the withhold rules of the fit are the ones that fail closed for a fit without a usable posterior."""
    if floor <= 0.0:
        return False
    s = posterior_sharpness(p_t_posterior)
    return s is not None and s < floor
