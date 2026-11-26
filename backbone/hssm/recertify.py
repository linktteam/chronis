"""WP9 B6 (owner decision B5): periodic re-certification of WITHHELD fits. Design and levels: tests/PREDECLARED_CRITERIA_WP9.md section 25.

A fit that the publish rule withholds (not converged, collapsed regime, degenerate duration, ...) may pass once more days have arrived. `recertify` refits on the stored days plus the new days with
the production `fit_hssm`, with the SAME settings as the earlier fit, and returns the new state. Nothing here decides what is publishable: the state is PUBLISHABLE iff the new result's own
`insights_withheld` is False. There is no parameter that loosens a rule: the settings of the earlier fit are checked against an allow-list and refused when they would relax a rule (a fast-test
fit, fewer than 10 restarts, a lowered cold-start gate), so a caller cannot use this function to publish what `fit_hssm` would refuse. An earlier PUBLISHABLE state is never touched.
`recertify_due` is the schedule hook (the refit policy: weekly by default). `recertification_sweep` runs the due items; it is pure orchestration (no clock, no state, no I/O).
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from .config import HSSMFitConfig
from .refit_policy import RefitPolicy, should_refit

SCHEMA_VERSION = 1
STATE_PUBLISHABLE = "PUBLISHABLE"
STATE_STILL_WITHHELD = "STILL_WITHHELD"
STATE_NOT_DUE = "NOT_DUE"
STATE_NOT_ELIGIBLE = "NOT_ELIGIBLE"
STATE_ERROR = "ERROR"

# the settings of the earlier fit that may be passed on to `fit_hssm`; everything else is refused (a setting this module does not know could loosen a rule)
ALLOWED_SETTINGS = frozenset({"candidate_ks", "n_initializations", "max_iter", "random_seed", "max_duration", "user_id", "dataset_hash", "model_type", "latent_dim", "shared_prior", "shared_prior_n0",
                              "max_duration_retry", "certify_prior_anchored"})
MIN_RESTARTS = HSSMFitConfig().n_init                       # the production hard minimum (10)


def _record(state: str, reasons=(), **extra) -> dict:
    return dict(state=state, reasons=list(reasons), result=None, rules_relaxed=False, schema_version=SCHEMA_VERSION, **extra)


def settings_refusal(settings: dict) -> str | None:
    """None when the settings can be passed on unchanged; otherwise the reason they are refused (they would relax a rule, or are unknown)."""
    for k in settings:
        if k not in ALLOWED_SETTINGS:
            return f"unknown_setting:{k}"
    n = settings.get("n_initializations", MIN_RESTARTS)
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < MIN_RESTARTS:
        return "n_initializations_below_the_production_minimum"
    return None


def recertify_due(last_check_day, today, policy: RefitPolicy = RefitPolicy()) -> bool:
    """True when a re-check is due: `refit_policy.should_refit` (never checked, or at least the policy's interval has passed). The same policy as the refit; pure."""
    return bool(should_refit(last_check_day, today, policy))


def recertify(X, new_rows, *, metadata: dict, timestamps=None, new_timestamps=None, today=None, policy: RefitPolicy = RefitPolicy(), fit_fn: Callable | None = None) -> dict:
    """Re-run the publish rules on a withheld fit with more days. `X`: the stored matrix (rows = days, as passed to the earlier fit); `new_rows`: the days that arrived since (may be empty);
    `metadata`: dict(settings=<the keyword arguments of the earlier `fit_hssm` call except matrix and timestamps>, earlier_state=<"STILL_WITHHELD" or the earlier withheld state>,
    last_check_day=<whole day of the last check or None>). `today` with `metadata["last_check_day"]` applies the schedule (`recertify_due`). `fit_fn` is for tests; the default is `fit_hssm`.
    Returns dict(state, reasons, result, rules_relaxed, schema_version, n_rows): `result` is the new `HSSMResult` ONLY when the state is PUBLISHABLE. Never raises."""
    try:
        meta = dict(metadata or {})
        settings = dict(meta.get("settings") or {})
        if meta.get("earlier_state") == STATE_PUBLISHABLE:
            return _record(STATE_NOT_ELIGIBLE, ["already_publishable"])
        bad = settings_refusal(settings)
        if bad is not None:
            return _record(STATE_NOT_ELIGIBLE, [bad])
        X = np.asarray(X, dtype=float)
        N = np.asarray(new_rows, dtype=float)
        if N.size == 0:
            return _record(STATE_NOT_DUE, ["no_new_days"], n_rows=int(len(X)))
        if X.ndim != 2 or N.ndim != 2 or X.shape[1] != N.shape[1]:
            return _record(STATE_NOT_ELIGIBLE, ["shape_mismatch"])
        if today is not None and not recertify_due(meta.get("last_check_day"), today, policy):
            return _record(STATE_NOT_DUE, ["not_due_yet"], n_rows=int(len(X)))
        full = np.vstack([X, N])
        kw = dict(settings)
        if timestamps is not None or new_timestamps is not None:
            if timestamps is None or new_timestamps is None:
                return _record(STATE_NOT_ELIGIBLE, ["timestamps_for_only_part_of_the_rows"])
            kw["timestamps"] = np.concatenate([np.asarray(timestamps, dtype=float), np.asarray(new_timestamps, dtype=float)])
        if fit_fn is None:
            from .fitting import fit_hssm as fit_fn                          # noqa: PLC0415
        try:
            result = fit_fn(matrix=full, **kw)
        except Exception as exc:                                              # noqa: BLE001
            from .gating import ColdStartError
            if isinstance(exc, ColdStartError):
                return _record(STATE_NOT_ELIGIBLE, ["cold_start_gate"], n_rows=int(len(full)))
            return _record(STATE_ERROR, [f"fit_failed:{type(exc).__name__}"], n_rows=int(len(full)), error=str(exc)[:200])
        withheld = bool(result.insights_withheld)                            # the result's own decision; this module has no rule of its own
        reasons = [str(r) for r in result._withheld_reasons()] if hasattr(result, "_withheld_reasons") else ([] if not withheld else ["withheld"])
        if withheld:
            return _record(STATE_STILL_WITHHELD, reasons, n_rows=int(len(full)))
        out = _record(STATE_PUBLISHABLE, [], n_rows=int(len(full)))
        out["result"] = result
        return out
    except Exception as exc:                                                  # noqa: BLE001  fail closed: nothing is published on an error
        return _record(STATE_ERROR, [f"recertify_error:{type(exc).__name__}"], error=str(exc)[:200])


def recertification_sweep(items: Iterable[dict], today, policy: RefitPolicy = RefitPolicy(), fit_fn: Callable | None = None) -> list[dict]:
    """Runs `recertify` for every item that is due. An item is dict(key=<any label>, X, new_rows, metadata, timestamps=None, new_timestamps=None). Returns one record per item, in order, with the
    item's `key` and, for a due item, the new `last_check_day` (= today) in `next_metadata`; an item that is not due is returned as NOT_DUE and is not fitted. One failing item never stops the others."""
    out = []
    for it in items:
        meta = dict(it.get("metadata") or {})
        try:
            due = recertify_due(meta.get("last_check_day"), today, policy)
        except Exception as exc:                                              # noqa: BLE001
            out.append(dict(_record(STATE_ERROR, [f"schedule_error:{type(exc).__name__}"]), key=it.get("key"), next_metadata=meta))
            continue
        if not due:
            out.append(dict(_record(STATE_NOT_DUE, ["not_due_yet"]), key=it.get("key"), next_metadata=meta))
            continue
        rec = recertify(it["X"], it["new_rows"], metadata=meta, timestamps=it.get("timestamps"), new_timestamps=it.get("new_timestamps"), today=None, policy=policy, fit_fn=fit_fn)
        nxt = dict(meta, last_check_day=int(today), earlier_state=rec["state"] if rec["state"] in (STATE_PUBLISHABLE, STATE_STILL_WITHHELD) else meta.get("earlier_state"))
        out.append(dict(rec, key=it.get("key"), next_metadata=nxt))
    return out
