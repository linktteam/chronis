"""WP9 B3 (owner decision B3): is the selected number of regimes K adequate? Additive result field `k_adequacy`; it NEVER changes which K is selected and the best guess is still shown.

WP7-R found that BIC picked the planted K in 20 of 24 easy and 8 of 24 hard datasets and never said when it was probably wrong. `assess_k_adequacy` collects three signals of the fit that was
selected (design and thresholds: tests/PREDECLARED_CRITERIA_WP9.md, B3 section):
  * `regimes_overlap`: the smallest pairwise distance between regime means, in pooled within-regime standard deviations per feature (root mean square over the features of the standardized
    difference, `pairwise_separation`), is below `min_overlap_sd`. Two regimes that close are not distinguishable, so the count is not trustworthy.
  * `k_not_clearly_best`: another candidate K has a BIC within `bic_margin` points of the selected one (the evidence for the selected K over the runner-up is weak; a likelihood-gain test of K+1
    against its BIC penalty says the same thing in BIC points), or the selected K is the largest candidate (K+1 was not tried).
  * `restart_disagreement`: the multi-start consensus of the selected fit failed (`fit_stability`, FIT_UNSTABLE).
`k_unsure` is True when any signal fires; `reasons` lists them. Pure functions; `fit_hssm` attaches the result and emits a developer alert (`alerts.emit`) when `k_unsure`.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from .config import K_ADEQUACY_BIC_MARGIN, K_ADEQUACY_FLAG_LARGEST, K_ADEQUACY_MIN_SEPARATION

REASON_OVERLAP = "regimes_overlap"
REASON_NOT_BEST = "k_not_clearly_best"
REASON_RESTARTS = "restart_disagreement"
REASON_UPPER = "selected_k_is_largest_candidate"


def pairwise_separation(mu: np.ndarray, var: np.ndarray) -> tuple[float, tuple[int, int] | None]:
    """Smallest pairwise distance between regime means in pooled within-regime standard deviations per feature: d_ij = sqrt(mean_f (mu_if - mu_jf)^2 / ((var_if + var_jf) / 2)) (the root mean
    square over the F features, so that it does not grow with the number of features). Returns (inf, None) for one regime."""
    mu = np.asarray(mu, dtype=float); var = np.asarray(var, dtype=float)
    if mu.shape != var.shape or mu.ndim != 2:
        raise ValueError("mu and var must be (K, F) arrays of the same shape")
    if not np.all(var > 0):
        raise ValueError("variances must be positive")
    K = mu.shape[0]
    best, pair = math.inf, None
    for i in range(K):
        for j in range(i + 1, K):
            d = float(np.sqrt((((mu[i] - mu[j]) ** 2) / ((var[i] + var[j]) / 2.0)).mean()))
            if d < best:
                best, pair = d, (i, j)
    return best, pair


def assess_k_adequacy(selected_k: int, bic_by_k: Mapping[int, float], mu, var, fit_unstable: bool | None, min_overlap_sd: float = K_ADEQUACY_MIN_SEPARATION,
                      bic_margin: float = K_ADEQUACY_BIC_MARGIN, candidates: Sequence[int] | None = None, flag_largest: bool = K_ADEQUACY_FLAG_LARGEST) -> dict:
    """The adequacy record of one selected fit. `fit_unstable` None (not assessed) counts as a restart disagreement (fail closed)."""
    cands = sorted(int(k) for k in (candidates if candidates is not None else bic_by_k))
    finite = {int(k): float(b) for k, b in bic_by_k.items() if math.isfinite(float(b))}
    sep, pair = pairwise_separation(mu, var)
    reasons: list[str] = []
    if math.isfinite(sep) and sep < min_overlap_sd:
        reasons.append(REASON_OVERLAP)
    others = {k: b - finite[int(selected_k)] for k, b in finite.items() if k != int(selected_k)} if int(selected_k) in finite else {}
    runner_up = min(others.items(), key=lambda kv: kv[1]) if others else None
    if runner_up is not None and runner_up[1] < bic_margin:
        reasons.append(REASON_NOT_BEST)
    if flag_largest and len(cands) > 1 and int(selected_k) == cands[-1]:
        reasons.append(REASON_UPPER)
    if fit_unstable is None or fit_unstable:
        reasons.append(REASON_RESTARTS)
    return dict(k_unsure=bool(reasons), reasons=reasons, selected_k=int(selected_k), best_guess_shown=True,
                signals=dict(min_separation_sd=None if not math.isfinite(sep) else float(sep), closest_pair=None if pair is None else list(pair), runner_up_k=None if runner_up is None else int(runner_up[0]),
                             runner_up_bic_gap=None if runner_up is None else float(runner_up[1]), fit_unstable=fit_unstable, candidates=cands),
                thresholds=dict(min_overlap_sd=float(min_overlap_sd), bic_margin=float(bic_margin), flag_largest=bool(flag_largest)))
