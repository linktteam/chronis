"""WP9 A3: detection of a change in the number of regimes between refits, and split / merge candidates from posterior overlap.

Why. WP8 criterion 6: the K selection on the growing window saw a planted K change in 3 of 24 streams and the split / merge candidates (nearest regime mean) were right in 3 of 38 (7.9%).
The growing window is dominated by the days before the change, and the 38 K-differing refits mix real changes with BIC noise.

Signal (design: tests/PREDECLARED_CRITERIA_WP9.md Part 7). At a refit with the anchor's K_a the RECENT window (the last `recent_days` days of the new window) is fitted at K' in
{K_a - 1, K_a, K_a + 1} and compared by BIC. `suspected` when K' != K_a has the lowest BIC and beats the K_a fit on that window by at least `delta` points (a separate `delta` for each
direction). Candidates: every row of the recent window is assigned to the anchor regime of highest Gaussian density (anchor means and the within-regime variances of the anchor's own window,
raw units); the share of each regime's rows that belongs to each old regime says which old regime a new one came from (split: an old regime that is the dominant parent of two or more new
ones; merge: a new regime whose rows divide between two old ones).

Nothing here changes which K is fitted or any raw output. `lineage.step` reads the result from `FitSummary.k_change`; a suspected change with the same K gives CONTINUITY_UNVERIFIED with the
reason `k_change_suspected` ("cannot confirm this is the same as before"). Fails closed: a recent-window fit that does not finish or is not accepted means NO signal, and the signal
never serves anything as ALIGNED. Pure functions apart from `fit_recent_window`, which runs `fit_hssm`.
"""
from __future__ import annotations

import math
from typing import Callable, Mapping

import numpy as np

from .alerts import exception_class_name
from .config import K_CHANGE_DELTA_DOWN, K_CHANGE_DELTA_UP, K_CHANGE_RECENT_DAYS
from .gating import ColdStartError
from .model import FittingConvergenceError

SCHEMA_VERSION = 1
REASON_CRASHED = "assessment_crashed"      # WP10B F3: the assessment crashed (or returned a malformed record); `lineage.step` never serves such a fit as ALIGNED
LEGACY_CRASH_REASONS = ("assessment_error",)   # what records written before F3 say
EXPECTED_NO_FIT = (FittingConvergenceError, ColdStartError)   # a recent-window fit that did not finish: unavailable, no signal. Any OTHER exception of a fit is a crash.
DOMINANT_PARENT_SHARE = 0.5          # split: the old regime must hold at least this share of a new regime's rows
MERGE_MIN_SHARE = 0.25               # merge: each of the two old regimes must hold at least this share of the new regime's rows
MIN_ROWS_FOR_OWN_VARIANCE = 3


# ------------------------------------------------------------------------------------------------------------------- the anchor in raw units
def anchor_within_variances(raw_window: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
    """(K, F) within-regime variance per feature of the anchor's own window, raw units, ignoring NaN. A regime with fewer than 3 rows, a feature with fewer than 2 observed values in it,
    or a zero variance takes the pooled within-regime variance of that feature (and a feature with no spread at all takes 1.0)."""
    X = np.asarray(raw_window, dtype=float); lab = np.asarray(labels, dtype=int)
    F = X.shape[1]
    pooled = np.ones(F)
    num, den = np.zeros(F), np.zeros(F)
    for k in range(K):
        Xk = X[lab == k]
        for f in range(F):
            v = Xk[:, f][~np.isnan(Xk[:, f])]
            if len(v) >= 2:
                num[f] += float(((v - v.mean()) ** 2).sum()); den[f] += len(v) - 1
    pooled = np.where(den > 0, num / np.maximum(den, 1), 1.0)
    pooled = np.where(pooled > 0, pooled, 1.0)
    out = np.tile(pooled, (K, 1))
    for k in range(K):
        Xk = X[lab == k]
        if len(Xk) < MIN_ROWS_FOR_OWN_VARIANCE:
            continue
        for f in range(F):
            v = Xk[:, f][~np.isnan(Xk[:, f])]
            if len(v) >= 2 and v.var(ddof=1) > 0:
                out[k, f] = float(v.var(ddof=1))
    return out


def assign_rows(recent_raw: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """For every row of the recent window, the anchor regime of highest diagonal-Gaussian log density over the observed features. A row with no observed feature gets -1."""
    X = np.asarray(recent_raw, dtype=float)
    K = means.shape[0]
    out = np.full(len(X), -1, dtype=int)
    for t in range(len(X)):
        obs = ~np.isnan(X[t])
        if not obs.any():
            continue
        ll = np.array([float((-0.5 * (np.log(variances[k, obs]) + (X[t, obs] - means[k, obs]) ** 2 / variances[k, obs])).sum()) for k in range(K)])
        out[t] = int(ll.argmax())
    return out


def overlap_shares(parent_of_row: np.ndarray, new_labels: np.ndarray, K_old: int, K_new: int) -> np.ndarray:
    """(K_new, K_old): the share of the rows of each new regime that the anchor assigns to each old regime (rows without an assignment are left out; an empty regime has zeros)."""
    out = np.zeros((K_new, K_old))
    for j in range(K_new):
        rows = (np.asarray(new_labels) == j) & (parent_of_row >= 0)
        n = int(rows.sum())
        if n:
            for i in range(K_old):
                out[j, i] = float(((parent_of_row == i) & rows).sum()) / n
    return out


def overlap_candidates(shares: np.ndarray, K_old: int, K_new: int) -> tuple:
    """Split / merge candidates from the overlap table. K_new > K_old: an old regime that is the dominant parent (>= 50% of the rows) of two or more new regimes is a split candidate.
    K_new < K_old: a new regime whose rows divide between two old regimes with at least 25% each is a merge candidate (the two largest shares). Old ids are the anchor's RAW regime indices."""
    out = []
    if K_new > K_old:
        dominant = shares.argmax(axis=1)
        for i in range(K_old):
            kids = [int(j) for j in range(K_new) if shares[j].sum() > 0 and dominant[j] == i and shares[j, i] >= DOMINANT_PARENT_SHARE]
            if len(kids) >= 2:
                out.append(dict(kind="split", old_raw_regime=int(i), new_raw_regimes=kids, method="overlap"))
    elif K_new < K_old:
        for j in range(K_new):
            order = np.argsort(-shares[j])
            if shares[j].sum() > 0 and shares[j, order[0]] >= MERGE_MIN_SHARE and shares[j, order[1]] >= MERGE_MIN_SHARE:
                out.append(dict(kind="merge", old_raw_regimes=sorted(int(x) for x in order[:2]), new_raw_regime=int(j), method="overlap"))
    return tuple(out)


# ------------------------------------------------------------------------------------------------------------------- the signal
def decide(bic_by_k: Mapping[int, float | None], k_anchor: int, delta_up: float = K_CHANGE_DELTA_UP, delta_down: float = K_CHANGE_DELTA_DOWN) -> dict:
    """Direction and margin from the BIC of the recent window at K_a - 1, K_a, K_a + 1 (None = that fit is unavailable). Not suspected when the K_a fit is unavailable, when K_a has the
    lowest BIC (ties go to K_a), or when the winner's margin over K_a is below its delta."""
    got = {int(k): float(b) for k, b in bic_by_k.items() if b is not None and math.isfinite(float(b))}
    base = dict(suspected=False, direction=None, k_best=int(k_anchor), margin=None, reason=None)
    if int(k_anchor) not in got:
        return dict(base, reason="reference_fit_unavailable")
    best = min(got, key=lambda k: (got[k], k != int(k_anchor)))                     # a tie goes to K_a
    margin = got[int(k_anchor)] - got[best]
    base.update(margin=float(margin), k_best=int(best))
    if best == int(k_anchor):
        return dict(base, reason="k_anchor_is_best")
    direction = "up" if best > int(k_anchor) else "down"
    need = float(delta_up if direction == "up" else delta_down)
    if best not in (int(k_anchor) - 1, int(k_anchor) + 1):
        return dict(base, reason="not_adjacent")
    if margin >= need:
        return dict(base, suspected=True, direction=direction)
    return dict(base, direction=direction, reason="margin_below_delta")


def assess_k_change(anchor_means: np.ndarray, anchor_variances: np.ndarray, raw_window: np.ndarray, *, fit_fn: Callable, k_anchor: int | None = None, recent_days: int = K_CHANGE_RECENT_DAYS,
                    delta_up: float = K_CHANGE_DELTA_UP, delta_down: float = K_CHANGE_DELTA_DOWN) -> dict:
    """The K-change record of one refit. `fit_fn(recent_raw, K) -> dict(bic, labels, raw_means, accepted) or None` fits the recent window at K (see `fit_recent_window`). A candidate K whose fit
    is missing or not accepted is unavailable (no claim from it). Never raises. WP10B F3: a fit that did not finish (`EXPECTED_NO_FIT`) is unavailable; ANY other exception, in a fit or anywhere else in the
    assessment, makes the record a crash record (`reason = "assessment_crashed"`, `crashed = True`, `error_class` = the exception class name and nothing else), which `lineage.step` never serves as ALIGNED."""
    Ka = int(anchor_means.shape[0] if k_anchor is None else k_anchor)
    rec = dict(schema_version=SCHEMA_VERSION, suspected=False, direction=None, k_best=Ka, margin=None, reason=None, bic_by_k={}, candidates=(), recent_days=int(recent_days),
               thresholds=dict(delta_up=float(delta_up), delta_down=float(delta_down)))
    try:
        recent = np.asarray(raw_window, dtype=float)[-int(recent_days):]
        fits, bic = {}, {}
        for K in (Ka - 1, Ka, Ka + 1):
            if K < 2:
                bic[K] = None; continue
            try:
                f = fit_fn(recent, K)
            except EXPECTED_NO_FIT:                                                         # an unfinished fit is unavailable; any other exception is a crash and goes to the handler below
                f = None
            ok = f is not None and f.get("accepted", False) and f.get("bic") is not None and math.isfinite(float(f["bic"]))
            bic[K] = float(f["bic"]) if ok else None
            if ok:
                fits[K] = f
        rec["bic_by_k"] = {str(k): v for k, v in bic.items()}
        d = decide(bic, Ka, delta_up, delta_down)
        rec.update({k: d[k] for k in ("suspected", "direction", "k_best", "margin", "reason")})
        if d["suspected"]:
            f = fits[d["k_best"]]
            parent = assign_rows(recent, np.asarray(anchor_means, dtype=float), np.asarray(anchor_variances, dtype=float))
            shares = overlap_shares(parent, np.asarray(f["labels"], dtype=int), Ka, d["k_best"])
            rec["candidates"] = overlap_candidates(shares, Ka, d["k_best"])
            rec["overlap_shares"] = shares.round(4).tolist()
    except Exception as exc:                                                                # noqa: BLE001  fail closed: a crash record, never a claim of "no change"
        rec.update(suspected=False, direction=None, reason=REASON_CRASHED, crashed=True, error_class=exception_class_name(exc))
    return rec


def assess_against_anchor(state, raw_window: np.ndarray, *, seed: int = 0, n_init: int = 10, max_iter: int = 60, recent_days: int = K_CHANGE_RECENT_DAYS,
                          delta_up: float = K_CHANGE_DELTA_UP, delta_down: float = K_CHANGE_DELTA_DOWN, fit_fn: Callable | None = None) -> dict:
    """The K-change record of a refit window against the lineage state's anchor (`lineage.LineageState`), for `lineage.summarize_fit(..., k_change=record)`. `raw_window` is the NEW window in
    physical units (days x features, NaN = missing). With no state, or an anchor that carries no window, there is nothing to compare with and the record is `suspected = False` with a reason.
    `fit_fn` defaults to `fit_recent_window` (seeded with `seed`)."""
    rec = dict(schema_version=SCHEMA_VERSION, suspected=False, direction=None, k_best=None, margin=None, reason=None, bic_by_k={}, candidates=(), recent_days=int(recent_days),
               thresholds=dict(delta_up=float(delta_up), delta_down=float(delta_down)))
    anchor = None if state is None else getattr(state, "anchor", None)
    if anchor is None:
        return dict(rec, reason="no_anchor")
    s = anchor.summary
    if s.raw_window is None:
        return dict(rec, k_best=int(s.K), reason="anchor_has_no_window")
    try:
        variances = anchor_within_variances(s.raw_window, s.labels, s.K)
    except Exception as exc:                                                                # noqa: BLE001
        return dict(rec, k_best=int(s.K), reason=REASON_CRASHED, crashed=True, error_class=exception_class_name(exc))
    fit = fit_fn if fit_fn is not None else (lambda recent, K: fit_recent_window(recent, K, seed=seed, n_init=n_init, max_iter=max_iter))
    return assess_k_change(s.raw_means, variances, raw_window, fit_fn=fit, k_anchor=s.K, recent_days=recent_days, delta_up=delta_up, delta_down=delta_down)


# ------------------------------------------------------------------------------------------------------------------- the fit of the recent window (production path)
def fit_recent_window(recent_raw: np.ndarray, K: int, *, seed: int = 0, n_init: int = 10, max_iter: int = 60, allow_fast_test_fit: bool = False) -> dict | None:
    """Fit the recent raw window (days x features, NaN = missing) at K regimes through the public `fit_hssm` (per-window z-scoring as the pipeline does, a missing single feature filled with 0, a
    missing day left as a NaN row). Returns dict(bic, labels, raw_means, accepted) or None when the fit did not finish (`EXPECTED_NO_FIT`); any other exception propagates (WP10B F3). `accepted` = not withheld by the publish rule.
    Production uses the default 10 restarts; fewer need `allow_fast_test_fit=True` (unit tests only)."""
    import contextlib, io, logging, warnings
    from .fitting import fit_hssm
    raw = np.asarray(recent_raw, dtype=float)
    mean = np.nanmean(raw, axis=0); std = np.nanstd(raw, axis=0)
    std = np.where(std > 0, std, 1.0)
    z = (raw - mean) / std
    day_missing = np.isnan(raw).all(axis=1)
    z = np.where(np.isnan(z), 0.0, z); z[day_missing] = np.nan
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            res = fit_hssm(matrix=z, candidate_ks=(int(K),), n_initializations=n_init, max_iter=max_iter, random_seed=int(seed), user_id="k-change", dataset_hash="k-change",
                           allow_fast_test_fit=bool(allow_fast_test_fit))
    except EXPECTED_NO_FIT:                                                                 # WP10B F3: only "did not finish" is no signal; any other exception is a crash and propagates to `assess_k_change`
        return None
    labels = np.asarray(res.p_t_posterior).argmax(axis=1)
    return dict(bic=float(res.bic_by_k[int(K)]), labels=labels, raw_means=mean + std * np.asarray(res.model.mu, dtype=float), accepted=not res.insights_withheld)
