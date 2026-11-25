"""WP8 item 3 (owner decision A2): the ordering score ladder, used ONLY to name the regimes at the first accepted fit of a lineage.

What it does: given the decoded regime label of every day and the raw (physical units) feature matrix of the fit window, it scores each regime by a weighted combination of
the features that carry an activity direction (registry `direction`, a domain prior, never learned), and orders the regimes from lowest to highest activity. Three rungs:
  rung 1  all features with a direction, weights from reliability;
  rung 2  movement and time moving (roles "movement" and "time_moving");
  rung 3  movement only (role "movement").
The weights of a rung come from DATA QUALITY only: `w_f proportional to coverage_f * SNR_f / (1 + SNR_f)`, SNR_f = between-regime variance over within-regime variance of
feature f over the decoded regimes of the window. The functions take no outcome label and no planted truth (a test checks the signature).
A rung is RELIABLE iff (i) its modal ordering appears in at least `min_modal_freq` of `n_bootstrap` block-bootstrap resamples of the window's days (blocks of `block_days` days, the
decoded labels are kept, the weights and the scores are recomputed in every resample, no refit), and (ii) the smallest gap between adjacent regime scores of the point
estimate is at least `min_gap` (score units: weighted person-level z-scores). The highest reliable rung names the regimes, provided every other reliable rung gives the same order
(agreement check); a conflict, or no reliable rung, makes the ordering UNVERIFIED: ids then follow first-fit order and nothing about activity may be claimed.
EXPERT_REVIEW(identity-distance): the ladder weights, the rung definitions and the thresholds below are defined here and are awaiting expert sign-off (owner decision A7).
The score is NOT used on refits: later fits inherit the names by alignment (lineage.py). Re-sorting on every refit caused flips in the reviewer's data.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .feature_registry import validate_registry

# EXPERT_REVIEW(identity-distance): the rung definitions (which registry roles each rung uses).
RUNG_ROLES: dict[int, tuple[str, ...] | None] = {1: None, 2: ("movement", "time_moving"), 3: ("movement",)}   # None = every feature with a direction


# EXPERT_REVIEW(identity-distance): the reliability thresholds (modal frequency, minimum adjacent gap), the bootstrap size and the block length are defined here.
@dataclass(frozen=True)
class LadderConfig:
    n_bootstrap: int = 200
    block_days: int = 7
    min_modal_freq: float = 0.90
    min_gap: float = 0.7            # fixed by the declared rule on development seeds 8201..8212; see the 'Design freeze (Item 3)' section of tests/PREDECLARED_CRITERIA_WP8.md
    seed: int = 0

    # EXPERT_REVIEW(identity-distance): validates the ladder's reliability thresholds.
    def __post_init__(self) -> None:
        if self.n_bootstrap < 10 or self.block_days < 1:
            raise ValueError("n_bootstrap must be at least 10 and block_days at least 1")
        if not (0.5 < self.min_modal_freq <= 1.0):
            raise ValueError("min_modal_freq must be in (0.5, 1]")
        if self.min_gap < 0:
            raise ValueError("min_gap must not be negative")


@dataclass
class RungResult:
    rung: int
    available: bool
    order: tuple | None = None            # raw regime indices from lowest to highest activity
    weights: dict = field(default_factory=dict)
    scores: tuple | None = None
    min_gap: float | None = None
    modal_freq: float | None = None
    modal_order: tuple | None = None      # the most frequent ordering over the bootstrap resamples
    reliable: bool = False


    # EXPERT_REVIEW(identity-distance): reports the ladder's reliability thresholds with the rung result.
    def as_dict(self) -> dict:
        return dict(rung=self.rung, available=self.available, order=list(self.order) if self.order else None, weights=dict(self.weights),
                    scores=list(self.scores) if self.scores else None, min_gap=self.min_gap, modal_freq=self.modal_freq,
                    modal_order=list(self.modal_order) if self.modal_order else None, reliable=self.reliable)


@dataclass
class OrderingResult:
    order: tuple                          # raw regime indices, position = stable id; first-fit order (0..K-1) when unverified
    activity_rung: int                    # 1, 2, 3; 0 = no rung was reliable or the rungs conflicted
    verified: bool
    reason: str
    rungs: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(order=list(self.order), activity_rung=self.activity_rung, verified=self.verified, reason=self.reason, rungs={str(r): v.as_dict() for r, v in self.rungs.items()})


# EXPERT_REVIEW(identity-distance): the per-feature coverage and signal-to-noise inputs of the ladder weights.
def _stats(raw: np.ndarray, labels: np.ndarray, K: int):
    """Per regime and feature: mean over observed days and the number of observed days; pooled within-regime variance per feature; the window mean and sd per feature."""
    T, F = raw.shape
    obs = ~np.isnan(raw)
    n = np.zeros((K, F)); means = np.full((K, F), np.nan); ss = np.zeros(F)
    for k in range(K):
        sel = labels == k
        for f in range(F):
            x = raw[sel & obs[:, f], f]
            n[k, f] = len(x)
            if len(x):
                means[k, f] = x.mean(); ss[f] += ((x - x.mean()) ** 2).sum()
    n_f = n.sum(axis=0)
    within = np.where(n_f > 0, ss / np.maximum(n_f, 1), np.nan)
    with np.errstate(all="ignore"):
        mean_f = np.nanmean(raw, axis=0); sd_f = np.nanstd(raw, axis=0)
    return n, means, within, mean_f, np.where(sd_f > 0, sd_f, np.nan), obs.mean(axis=0)


def _rung_weights_scores(raw, labels, K, registry, rung):
    """EXPERT_REVIEW(identity-distance): the ladder weights are computed here, w_f proportional to coverage_f * SNR_f / (1 + SNR_f), from data quality only."""
    n, means, within, mean_f, sd_f, coverage = _stats(raw, labels, K)
    roles = RUNG_ROLES[rung]
    w = np.zeros(len(registry)); sign = np.zeros(len(registry))
    for f, spec in enumerate(registry):
        if spec.direction == 0 or (roles is not None and spec.role not in roles):
            continue
        usable = (n[:, f] >= 1).all() and np.isfinite(sd_f[f]) and np.isfinite(within[f]) and coverage[f] > 0
        if not usable:
            continue
        n_f = n[:, f]
        grand = np.nansum(n_f * means[:, f]) / n_f.sum()
        between = np.nansum(n_f * (means[:, f] - grand) ** 2) / n_f.sum()
        snr = between / within[f] if within[f] > 0 else 1e6
        w[f] = coverage[f] * snr / (1.0 + snr)
        sign[f] = spec.direction
    if w.sum() <= 0:
        return None
    w = w / w.sum()
    z = (means - mean_f) / sd_f                                    # (K, F), nan where a regime has no observation
    score = np.nansum(np.where(w > 0, z, 0.0) * (w * sign), axis=1)
    return w, score


def _order(score):
    return tuple(int(i) for i in np.argsort(score, kind="stable"))


# EXPERT_REVIEW(identity-distance): the block-bootstrap reliability test (modal order frequency and minimum adjacent gap) of the ladder.
def _bootstrap_modal(raw, labels, K, registry, rung, cfg: LadderConfig, rng):
    T = raw.shape[0]
    n_blocks = int(np.ceil(T / cfg.block_days))
    starts = np.arange(n_blocks) * cfg.block_days
    counts: Counter = Counter()
    for _ in range(cfg.n_bootstrap):
        pick = rng.integers(0, n_blocks, n_blocks)
        idx = np.concatenate([np.arange(starts[b], min(starts[b] + cfg.block_days, T)) for b in pick])[:T]
        res = _rung_weights_scores(raw[idx], labels[idx], K, registry, rung)
        if res is None:
            counts["invalid"] += 1
        else:
            counts[_order(res[1])] += 1
    counts.pop("invalid", None)
    if not counts:
        return None, 0.0
    order, c = counts.most_common(1)[0]
    return order, c / cfg.n_bootstrap


def order_regimes(raw: np.ndarray, labels: np.ndarray, K: int, registry, config: LadderConfig = LadderConfig()) -> OrderingResult:
    """raw: (T, F) physical units, NaN where missing; labels: (T,) decoded regime per day in 0..K-1; registry: FeatureSpec per column.
    EXPERT_REVIEW(identity-distance): applies the reliability thresholds of `config` (modal frequency and minimum gap) to choose the rung."""
    raw = np.asarray(raw, dtype=float); labels = np.asarray(labels, dtype=int)
    if raw.ndim != 2 or labels.shape != (raw.shape[0],):
        raise ValueError("raw must be (T, F) and labels (T,)")
    if K < 1 or labels.min() < 0 or labels.max() >= K:
        raise ValueError(f"labels must lie in 0..{K - 1}")
    registry = validate_registry(registry, raw.shape[1])
    rng = np.random.default_rng(config.seed)
    rungs: dict[int, RungResult] = {}
    for rung in (1, 2, 3):
        res = _rung_weights_scores(raw, labels, K, registry, rung)
        if res is None:
            rungs[rung] = RungResult(rung, False)
            continue
        w, score = res
        order = _order(score)
        gaps = np.diff(np.sort(score))
        modal_order, freq = _bootstrap_modal(raw, labels, K, registry, rung, config, rng)
        reliable = bool(modal_order == order and freq >= config.min_modal_freq and gaps.min() >= config.min_gap)
        rungs[rung] = RungResult(rung, True, order, {registry[f].name: float(w[f]) for f in range(len(w)) if w[f] > 0}, tuple(float(s) for s in score), float(gaps.min()), float(freq), modal_order, reliable)
    reliable = [r for r in (1, 2, 3) if rungs[r].reliable]
    first_fit = tuple(range(K))
    if not reliable:
        return OrderingResult(first_fit, 0, False, "no rung is reliable: ids follow first-fit order, no activity claim", rungs)
    top = reliable[0]
    if any(rungs[r].order != rungs[top].order for r in reliable[1:]):
        return OrderingResult(first_fit, 0, False, "reliable rungs disagree: ids follow first-fit order, no activity claim", rungs)
    return OrderingResult(rungs[top].order, top, True, f"rung {top} is reliable and no reliable rung disagrees", rungs)
