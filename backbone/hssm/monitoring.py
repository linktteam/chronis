"""WP9 B5 (owner decision C1): monitoring primitives. Pure functions; no UI, no clock, no state. Design and levels: tests/PREDECLARED_CRITERIA_WP9.md section 24.

`predictability_monitor`: a series of per-day held-out one-step scores (log predictive density of the day given the days before it under a frozen model, nats per day; or the gain of that
over a reference model) is compared with its own trailing window; a drop of the recent mean below the reference mean by both a practical amount (`min_drop_sd` reference standard deviations) and a statistical one
(`drop_z` standard errors of the difference, inflated for the autocorrelation of the scores) is an EVENT. The user-facing consequence (owner decision C1) is that the display stays
"early estimate" and a developer is alerted (`alert_on_drop`); nothing here changes a fit.
`confidence_display_gate`: a percentage is shown only when N17 has passed, calibration evidence exists for the class of fit and the calibrated confidence is at least 0.85; in every other
case the display is "early_estimate". While `config.N17_PASSED` is False the answer is always "early_estimate".
`baseline_predictive_increments`: the held-out one-step scores of a fitted baseline model (frozen parameters), the input the monitor needs.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from . import alerts
from .config import (CONFIDENCE_DISPLAY_THRESHOLD, MONITOR_COOLDOWN_DAYS, MONITOR_DROP_Z, MONITOR_MIN_DROP_SD, MONITOR_REFERENCE_DAYS, MONITOR_WINDOW_DAYS, N17_PASSED)

EARLY_ESTIMATE = "early_estimate"
PERCENTAGE = "percentage"
STATUS_OK = "ok"
STATUS_NOT_ENOUGH = "not_enough_scores"
STATUS_INVALID = "invalid_input"
EVENT_KIND = "predictability_drop"


def predictability_monitor(scores, window: int = MONITOR_WINDOW_DAYS, reference_days: int = MONITOR_REFERENCE_DAYS, drop_z: float = MONITOR_DROP_Z, min_drop_sd: float = MONITOR_MIN_DROP_SD,
                           cooldown_days: int = MONITOR_COOLDOWN_DAYS) -> dict:
    """Events of a score series. `scores`: one number per day in order (NaN or infinite = no score that day; those days are dropped, `n_dropped` says how many; the day index of an event
    is the index in the ORIGINAL series). At each day with `window + reference_days` scores behind it: recent = the last `window` scores, reference = the `reference_days` scores before them.
    s0 = sd of the reference (ddof 1), r = its lag-1 autocorrelation clipped to [0, 0.8], g = (1 + r) / (1 - r) (the variance inflation of the mean of an AR(1) series),
    se = s0 * sqrt(g / window + g / reference_days), drop = mean(reference) - mean(recent). EVENT when drop >= min_drop_sd * s0 and drop / se >= drop_z; one event per `cooldown_days`.
    A constant reference (s0 = 0) gives no event. Returns dict(status, events, n_scores, n_dropped, parameters); an event is dict(day, reference_mean, recent_mean, drop, drop_sd, z). Never raises."""
    params = dict(window=window, reference_days=reference_days, drop_z=drop_z, min_drop_sd=min_drop_sd, cooldown_days=cooldown_days)
    out = dict(status=STATUS_OK, events=[], n_scores=0, n_dropped=0, parameters=params)
    try:
        if not (_posint(window) and _posint(reference_days) and _posint(cooldown_days) and window >= 3 and reference_days >= 3):
            return dict(out, status=STATUS_INVALID)
        if not (_finite(drop_z) and _finite(min_drop_sd) and drop_z >= 0 and min_drop_sd >= 0):
            return dict(out, status=STATUS_INVALID)
        y = np.asarray(scores, dtype=float).reshape(-1)
    except Exception:                                                                                                  # noqa: BLE001
        return dict(out, status=STATUS_INVALID)
    keep = np.isfinite(y)
    idx, v = np.flatnonzero(keep), y[keep]
    out.update(n_scores=int(len(v)), n_dropped=int((~keep).sum()))
    need = window + reference_days
    if len(v) < need:
        return dict(out, status=STATUS_NOT_ENOUGH)
    last_event_day = None
    for i in range(need, len(v) + 1):                                                                                  # i = number of scores seen; the recent window ends at score i - 1
        recent, ref = v[i - window:i], v[i - window - reference_days:i - window]
        day = int(idx[i - 1])
        if last_event_day is not None and day - last_event_day < cooldown_days:
            continue
        s0 = float(ref.std(ddof=1))
        if not (math.isfinite(s0) and s0 > 0.0):
            continue
        r = min(max(_lag1(ref), 0.0), 0.8)
        g = (1.0 + r) / (1.0 - r)
        se = s0 * math.sqrt(g / window + g / reference_days)
        drop = float(ref.mean() - recent.mean())
        z = drop / se
        if drop >= min_drop_sd * s0 and z >= drop_z:
            out["events"].append(dict(day=day, reference_mean=float(ref.mean()), recent_mean=float(recent.mean()), drop=drop, drop_sd=drop / s0, z=float(z)))
            last_event_day = day
    return out


def _lag1(v: np.ndarray) -> float:
    c = v - v.mean()
    d = float(c @ c)
    return float(c[:-1] @ c[1:]) / d if d > 0 else 0.0


def alert_on_drop(user_id, result: dict) -> list[dict]:
    """One developer alert per event of a `predictability_monitor` result (hashed user id, numbers only). Returns the emitted events."""
    return [alerts.emit(EVENT_KIND, user_id, day=e["day"], drop=round(e["drop"], 4), drop_sd=round(e["drop_sd"], 3), z=round(e["z"], 3), reference_mean=round(e["reference_mean"], 4), recent_mean=round(e["recent_mean"], 4))
            for e in result.get("events", [])]


def confidence_display_gate(calibrated_confidence, calibration_evidence, n17_passed: bool = N17_PASSED, threshold: float = CONFIDENCE_DISPLAY_THRESHOLD) -> dict:
    """What may be shown for a confidence. dict(display, percentage, reasons). `display` is "early_estimate" unless N17 has passed, `calibration_evidence` is a non-empty description of the
    calibration evidence for this class of fit, and `calibrated_confidence` is a finite number in [0, 1] that is at least `threshold`; then it is "percentage" and `percentage` is the
    integer part of 100 * confidence (never rounded up). The threshold can only be raised: a value below the owner's 0.85 is refused (the call returns early_estimate with a reason)."""
    reasons = []
    if n17_passed is not True:
        reasons.append("n17_not_passed")
    if not (isinstance(calibration_evidence, str) and calibration_evidence.strip()):
        reasons.append("no_calibration_evidence_for_this_class_of_fit")
    c = calibrated_confidence
    if isinstance(c, bool) or not isinstance(c, (int, float, np.integer, np.floating)) or not math.isfinite(float(c)) or not 0.0 <= float(c) <= 1.0:
        reasons.append("confidence_not_a_number_in_0_1")
    elif float(c) < threshold:
        reasons.append("confidence_below_threshold")
    if not (isinstance(threshold, (int, float)) and not isinstance(threshold, bool) and math.isfinite(float(threshold)) and threshold >= CONFIDENCE_DISPLAY_THRESHOLD):
        reasons.append("threshold_below_the_owner_floor")
    if reasons:
        return dict(display=EARLY_ESTIMATE, percentage=None, reasons=reasons)
    return dict(display=PERCENTAGE, percentage=int(math.floor(float(c) * 100.0 + 1e-9)), reasons=[])


def baseline_predictive_increments(model, X: np.ndarray) -> np.ndarray:
    """Per-day log p(y_t | y_<t) of a FITTED baseline model (`GaussianHSMM`) with frozen parameters, for the rows of `X` (a missing day is an uninformative row: its increment is 0).
    Forward pass only; `model` is not modified.
    increments[t] = logsumexp(alpha[t]) - logsumexp(alpha[t-1]); their sum is the log-likelihood of X (tested against `model.log_likelihood_`)."""
    from scipy.special import logsumexp
    X = np.asarray(X, dtype=float)
    T, K, D = X.shape[0], model.K, model.Dmax
    NEG = -np.inf
    emis = model._emission_loglik(X)
    dur = model._duration_logpmf()
    logA = np.log(np.clip(model.A, 1e-300, None))
    log_pi = np.log(np.clip(model.pi, 1e-300, None))
    alpha = np.full((T, K, D), NEG)
    alpha[0] = (log_pi[:, None] + dur) + emis[0][:, None]
    lse = np.full(T, np.nan)
    lse[0] = logsumexp(alpha[0])
    for t in range(1, T):
        cont = np.full((K, D), NEG)
        cont[:, : D - 1] = alpha[t - 1, :, 1:D]
        trans_in = logsumexp(alpha[t - 1, :, 0][:, None] + logA, axis=0)
        alpha[t] = np.logaddexp(cont, trans_in[:, None] + dur) + emis[t][:, None]
        lse[t] = logsumexp(alpha[t])
    return np.diff(np.concatenate([[0.0], lse]))


def baseline_forward_scores(model, X: np.ndarray):
    """WP10A A5 (additive; `baseline_predictive_increments` is unchanged): the same forward pass, returning BOTH the per-day held-out increments and the FILTERED regime posterior
    p(k_t | y_<=t) of every day (shape T x K, rows sum to 1), for a fitted baseline model with frozen parameters. The increments are computed by the same operations in the same order as
    `baseline_predictive_increments` and are bit-identical to it (tested). `model` is not modified. Returns (increments, regime_posterior)."""
    from scipy.special import logsumexp
    X = np.asarray(X, dtype=float)
    T, K, D = X.shape[0], model.K, model.Dmax
    NEG = -np.inf
    emis = model._emission_loglik(X)
    dur = model._duration_logpmf()
    logA = np.log(np.clip(model.A, 1e-300, None))
    log_pi = np.log(np.clip(model.pi, 1e-300, None))
    alpha = np.full((T, K, D), NEG)
    alpha[0] = (log_pi[:, None] + dur) + emis[0][:, None]
    lse = np.full(T, np.nan)
    lse[0] = logsumexp(alpha[0])
    for t in range(1, T):
        cont = np.full((K, D), NEG)
        cont[:, : D - 1] = alpha[t - 1, :, 1:D]
        trans_in = logsumexp(alpha[t - 1, :, 0][:, None] + logA, axis=0)
        alpha[t] = np.logaddexp(cont, trans_in[:, None] + dur) + emis[t][:, None]
        lse[t] = logsumexp(alpha[t])
    per_regime = logsumexp(alpha, axis=2)                                                                              # (T, K): log p(k_t = k, y_<=t)
    post = np.exp(per_regime - lse[:, None])
    return np.diff(np.concatenate([[0.0], lse])), post


def _posint(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool) and x >= 1


def _finite(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) and math.isfinite(float(x))
