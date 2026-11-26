"""WP10A A5: the three candidate monitors of tests/PREDECLARED_CRITERIA_WP10A.md Part 4, section 10 (declared before any A5 fit; parameters frozen, not tuned). Study code: nothing here is called by the fit path.
All keep window 14, reference 28, cooldown 14, min_drop_sd 0.75, drop_z 3.5 of the production `predictability_monitor` and its event rule; they differ in the series or in the standard error.
  MA  standard error = the LARGER of the production AR(1)-inflated one and the block-bootstrap one (Part 4b: the bootstrap alone is biased low on 28 days). Block bootstrap of (reference mean - recent
      mean): blocks of 7 days, recent = 2 blocks, reference = 4 blocks, both drawn with replacement from the blocks of the reference window, B = 2000 draws, `default_rng(0)` for every decision day.
  MB  transition-excluded series: the days on which the decoded regime (argmax of the filtered posterior) differs from that of the day before, and the day after such a day, are removed.
  MC  regime-standardised residual: (score - m_k) / s_k with m_k, s_k the mean and sd of the training-day scores of the decoded regime k (fewer than 5 such days: pooled mean and sd), k = decoded regime of the day.
Every function takes the arrays of one user and one stream and returns the production result format (`dict(status, events, ...)`); an event day is the index in the series given."""
from __future__ import annotations

import math

import numpy as np

from backbone.hssm.monitoring import _lag1, predictability_monitor

WINDOW, REFERENCE, COOLDOWN, MIN_DROP_SD, DROP_Z = 14, 28, 14, 0.75, 3.5
BLOCK, DRAWS = 7, 2000
MIN_REGIME_DAYS = 5
CANDIDATES = ("MA", "MB", "MC")


def _lag1_free_se_block(ref: np.ndarray, window: int = WINDOW, block: int = BLOCK, draws: int = DRAWS) -> float:
    """Moving-block bootstrap standard deviation of `ref* - recent*` where `recent*` is the mean of `window // block` blocks and `ref*` the mean of `len(ref) // block` blocks, all blocks
    (consecutive `block` scores, any start inside `ref`) drawn with replacement from `ref`. `default_rng(0)`: the same reference always gives the same se."""
    rng = np.random.default_rng(0)
    n_starts = len(ref) - block + 1
    csum = np.concatenate([[0.0], np.cumsum(ref)])
    block_mean = (csum[block:block + n_starts] - csum[:n_starts]) / block
    nb_recent, nb_ref = window // block, len(ref) // block
    recent_star = block_mean[rng.integers(0, n_starts, size=(draws, nb_recent))].mean(axis=1)
    ref_star = block_mean[rng.integers(0, n_starts, size=(draws, nb_ref))].mean(axis=1)
    return float(np.std(ref_star - recent_star, ddof=1))


def monitor_block_bootstrap(scores, window: int = WINDOW, reference_days: int = REFERENCE, drop_z: float = DROP_Z, min_drop_sd: float = MIN_DROP_SD, cooldown_days: int = COOLDOWN) -> dict:
    """MA. The production event rule with the standard error max(AR(1)-inflated production se, block-bootstrap se) (Part 4b). NaN or infinite scores are dropped as in the production monitor; the event
    day is the index in the original series."""
    y = np.asarray(scores, dtype=float).reshape(-1)
    keep = np.isfinite(y)
    idx, v = np.flatnonzero(keep), y[keep]
    out = dict(status="ok", events=[], n_scores=int(len(v)), n_dropped=int((~keep).sum()))
    need = window + reference_days
    if len(v) < need:
        return dict(out, status="not_enough_scores")
    last = None
    for i in range(need, len(v) + 1):
        recent, ref = v[i - window:i], v[i - window - reference_days:i - window]
        day = int(idx[i - 1])
        if last is not None and day - last < cooldown_days:
            continue
        s0 = float(ref.std(ddof=1))
        if not (math.isfinite(s0) and s0 > 0.0):
            continue
        r = min(max(_lag1(ref), 0.0), 0.8)
        g = (1.0 + r) / (1.0 - r)
        se_ar = s0 * math.sqrt(g / window + g / reference_days)                                   # the production standard error
        se = max(se_ar, _lag1_free_se_block(ref, window))
        if not (math.isfinite(se) and se > 0.0):
            continue
        drop = float(ref.mean() - recent.mean())
        z = drop / se
        if drop >= min_drop_sd * s0 and z >= drop_z:
            out["events"].append(dict(day=day, reference_mean=float(ref.mean()), recent_mean=float(recent.mean()), drop=drop, drop_sd=drop / s0, z=float(z)))
            last = day
    return out


def decoded(post: np.ndarray) -> np.ndarray:
    """Decoded regime of every day: argmax of the filtered posterior."""
    return np.argmax(np.asarray(post, dtype=float), axis=1)


def transition_days(post: np.ndarray, prev=None) -> np.ndarray:
    """Boolean per day: True on a day on which the decoded regime differs from the day before, and on the day after such a day. `prev`: the posterior row of the day BEFORE the first day (so that the
    first day can be compared with it); without it the first day is never a transition."""
    p = np.asarray(post, dtype=float)
    if prev is not None:
        p = np.vstack([np.asarray(prev, dtype=float)[None, :], p])
    k = decoded(p)
    change = np.zeros(len(k), dtype=bool)
    change[1:] = k[1:] != k[:-1]
    out = change.copy()
    out[1:] |= change[:-1]
    return out[1:] if prev is not None else out


def monitor_transition_excluded(scores, post, prev=None, **kw) -> dict:
    """MB. The production monitor on the series with the transition days removed (set to NaN: the production monitor drops them and keeps the original day index)."""
    y = np.asarray(scores, dtype=float).reshape(-1).copy()
    mask = transition_days(post, prev)
    if len(mask) != len(y):
        raise ValueError("scores and posterior must cover the same days")
    y[mask] = np.nan
    return predictability_monitor(y, **kw)


def regime_stats(train_scores, train_post, min_days: int = MIN_REGIME_DAYS):
    """(mean_k, sd_k) per regime from the training days (finite scores only); a regime with fewer than `min_days` such days takes the pooled mean and sd (ddof 1)."""
    s = np.asarray(train_scores, dtype=float).reshape(-1)
    k = decoded(train_post)
    K = np.asarray(train_post).shape[1]
    fin = np.isfinite(s)
    pooled_m, pooled_s = float(s[fin].mean()), float(s[fin].std(ddof=1))
    means, sds = [], []
    for r in range(K):
        sel = fin & (k == r)
        if sel.sum() >= min_days:
            means.append(float(s[sel].mean())); sds.append(float(s[sel].std(ddof=1)))
        else:
            means.append(pooled_m); sds.append(pooled_s)
    return np.array(means), np.array(sds)


def standardise(scores, post, means, sds) -> np.ndarray:
    """e_t = (score_t - m_{k_t}) / s_{k_t}, k_t the decoded regime of day t; a regime with a zero or non-finite sd gives NaN (the day is dropped)."""
    s = np.asarray(scores, dtype=float).reshape(-1)
    k = decoded(post)
    sd = np.asarray(sds)[k]
    with np.errstate(divide="ignore", invalid="ignore"):
        e = (s - np.asarray(means)[k]) / sd
    e[~(np.isfinite(sd) & (sd > 0))] = np.nan
    return e


def monitor_regime_standardised(scores, post, train_scores, train_post, **kw) -> dict:
    """MC. The production monitor on the regime-standardised residual of the held-out days (`scores`, `post`: the held-out days; `train_*`: the training days that fix the per-regime mean and sd)."""
    means, sds = regime_stats(train_scores, train_post)
    return predictability_monitor(standardise(scores, post, means, sds), **kw)


def run_candidate(name: str, scores, post, train_scores, train_post, prev=None) -> dict:
    """One candidate on the held-out series of one user and stream. `scores`, `post`: the held-out days (days 90..179 of the study); `train_scores`, `train_post`: the training days; `prev`: the posterior row of the last training day."""
    if name == "M0":
        return predictability_monitor(scores)
    if name == "MA":
        return monitor_block_bootstrap(scores)
    if name == "MB":
        return monitor_transition_excluded(scores, post, prev)
    if name == "MC":
        return monitor_regime_standardised(scores, post, train_scores, train_post)
    raise ValueError(f"unknown monitor {name!r}")
