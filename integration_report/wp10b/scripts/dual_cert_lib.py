"""WP10B B3: the predictive certificate for DUAL fits that are not prior-anchored (study library; the production module `backbone/hssm/cold_start_cert.py` is unchanged and still only certifies prior-anchored fits).
It is the WP9 B1 procedure with three declared differences (tests/PREDECLARED_CRITERIA_WP10B.md, the B3 part): (1) eligibility does not require a shared prior (a dual fit whose EM state is REJECTED_UPDATE or MAX_ITER_REACHED with NO other withhold reason);
(2) days with every feature missing are left out of every score (the pipeline matrix has NaN rows; the production certificate would fail closed on them); (3) the one-step gain over the BASELINE engine refitted on the same rows is reported next to the gain over the single Gaussian.
Unchanged from WP9 B1: rolling origins at 50 / 65 / 80% of the rows, the refit procedure = the production fit on fewer rows, margin z = 3.0 HAC standard errors of the mean gain over the single Gaussian, regime-mean stability tau = 0.25, fail closed.
`certify_dual` is pure: it takes adapters with `predictive_increments(X)`, `regime_observation_means()` and `degraded`."""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from backbone.hssm import cold_start_cert as C
from backbone.hssm.config import COLD_START_CERT_FRACTIONS, COLD_START_CERT_TAU, COLD_START_CERT_Z

Z = COLD_START_CERT_Z
TAU = COLD_START_CERT_TAU


def observed(X: np.ndarray) -> np.ndarray:
    """Rows with at least one observed feature (an all-missing day is a NaN row and is never scored)."""
    return ~np.isnan(np.asarray(X, dtype=float)).all(axis=1)


def certify_dual(X: np.ndarray, full_model, refit_fn: Callable, baseline_refit_fn: Callable | None = None, *, eligibility: dict | None = None, z: float = Z, tau: float = TAU,
                 fractions: Sequence[float] = COLD_START_CERT_FRACTIONS) -> dict:
    """Certify one dual fit. `eligibility`: em_state, other_reasons (the withhold reasons other than non-convergence) and, for a CONTROL only, apply_to_any_state=True (the EM-state test is skipped so that the certificate can be asked about a
    fit that EM calls converged, as the wrong-engine controls are). Never raises: any failure is a NOT_CERTIFIED verdict with a reason."""
    el = eligibility or {}
    base = dict(state=C.STATE_NOT_CERTIFIED, reasons=[], z=float(z), tau=float(tau), n_increments=None, mean_gain=None, se=None, lower=None, stability=None, blocks=None, baseline=None, schema_version=1)
    try:
        if not el.get("apply_to_any_state"):
            if el.get("em_state") == "CONVERGED":
                return dict(base, state=C.STATE_NOT_APPLICABLE, reasons=[C.R_CONVERGED])
            if el.get("em_state") not in C.ELIGIBLE_EM_STATES:
                return dict(base, reasons=[C.R_STATE])
        if el.get("other_reasons"):
            return dict(base, reasons=[C.R_OTHER], other_reasons=list(el["other_reasons"]))
        X = np.asarray(X, dtype=float)
        blocks, why = C.origin_blocks(len(X), fractions)
        if blocks is None:
            return dict(base, reasons=[why])
        obs = observed(X)
        if np.isinf(X).any():
            return dict(base, reasons=[C.R_NONFINITE])
        parts, parts_b, last = [], [], None
        for o, e in blocks:
            try:
                m = refit_fn(X[:o])
                bm = baseline_refit_fn(X[:o]) if baseline_refit_fn is not None else None
            except Exception:                                                                                      # noqa: BLE001  an unfinished refit is a refusal
                return dict(base, reasons=[C.R_REFIT], blocks=[list(b) for b in blocks])
            if getattr(m, "degraded", False):
                return dict(base, reasons=[C.R_DEGRADED_REFIT], blocks=[list(b) for b in blocks])
            keep = obs[o:e]
            lp = np.asarray(m.predictive_increments(X[:e]), dtype=float)[o:e][keep]
            g = C.gaussian_heldout(X[:o][obs[:o]], X[o:e][keep])
            parts.append(lp - g)
            if bm is not None:
                lb = np.asarray(bm.predictive_increments(X[:e]), dtype=float)[o:e][keep]
                parts_b.append(lp - lb)
            last = m
        d = np.concatenate(parts)
        sd = np.nanstd(X[obs], axis=0, ddof=1)
        stab = C.stability_distance(full_model.regime_observation_means(), last.regime_observation_means(), sd)
        out = C.decide(d, stab, z=z, tau=tau)
        out.update(blocks=[list(b) for b in blocks], schema_version=1, baseline=None)
        if parts_b:
            db = np.concatenate(parts_b)
            if len(db) >= 3 and np.all(np.isfinite(db)):
                mean, se = float(db.mean()), C.hac_se(db)
                lower = mean - float(z) * se if math.isfinite(se) else None
                out["baseline"] = dict(n_increments=int(len(db)), mean_gain=mean, se=C._f(se), lower=lower, beats_baseline=bool(lower is not None and lower > 0.0))
            else:
                out["baseline"] = dict(n_increments=int(len(db)), mean_gain=None, se=None, lower=None, beats_baseline=False)
        return out
    except Exception as exc:                                                                                       # noqa: BLE001  fail closed
        return dict(base, reasons=[C.R_ERROR], error=f"{type(exc).__name__}: {str(exc)[:120]}")


class BaselineAdapter:
    """Adapter of a fitted baseline `GaussianHSMM` to the interface of `certify_dual` (one-step predictive increments with frozen parameters)."""

    def __init__(self, model):
        self.model = model
        self.degraded = False

    def predictive_increments(self, X: np.ndarray) -> np.ndarray:
        from dual_lib import baseline_increments
        return baseline_increments(self.model, np.asarray(X, dtype=float))

    def regime_observation_means(self) -> np.ndarray:
        return np.asarray(self.model.mu, dtype=float)


def make_baseline_refit(n_regimes: int, base_seed: int) -> Callable:
    """The production baseline fit (10 restarts, 60 iterations) on the rows before the origin."""
    def refit(X_train: np.ndarray) -> BaselineAdapter:
        from dual_lib import fit_baseline
        return BaselineAdapter(fit_baseline(X_train, base_seed).model)
    return refit
