"""WP9 B1 (owner decision B2): a separate, fail-closed certification for PRIOR-ANCHORED fits that checks predictive validity, not EM tolerance.

Why. WP7-R item 3: `shrink5` fits (shared prior, n0 = 5) stop after 2 to 3 EM iterations (`REJECTED_UPDATE`) or hit the iteration cap, so the EM-convergence publish rule withholds
them, although the withheld fits predicted better than the converged ones. The honest question for such a fit is not "did EM finish" but "does it predict the user's next days better
than a single Gaussian, out of sample, and are its parameters settled". This module answers that, never relaxes any other rule, and changes nothing unless asked for
(`fit_hssm(..., certify_prior_anchored=True)`).

Procedure (design: tests/PREDECLARED_CRITERIA_WP9.md section 16):
  1. Eligibility: the fit started at a shared prior with n0 > 0; its EM state is REJECTED_UPDATE or MAX_ITER_REACHED; EVERY OTHER withhold reason is absent (collapsed regime, degenerate
     duration, degraded fit, Cholesky fallbacks ...); at least `CERT_MIN_ROWS` rows. A converged fit is NOT_APPLICABLE: the EM rule decides it.
  2. Rolling origins at 50 / 65 / 80% of the rows: the model is refitted with the SAME procedure on the rows before the origin and scored one step ahead (frozen parameters) on the next
     block, against a single Gaussian fitted on the same earlier rows. The per-step differences are pooled in time order.
  3. Gain: mean(d) - z * se > 0, se = Bartlett-kernel HAC standard error of the mean (bandwidth ceil(n / 4): regimes persist, a short bandwidth would understate the error).
  4. Parameter stability: the regime observation means of the full fit and of the last-origin fit differ by a root mean square of at most `tau` single-Gaussian standard deviations.
  5. Fail closed: any exception, non-finite number, unfinished or degraded refit, too few rows: NOT_CERTIFIED with the reason.
The result is a dict stored in `HSSMResult.certification`; `fit_state` keeps what EM said. `HSSMResult.withheld_reason` drops ONLY the `fit_state_not_converged` reason, ONLY when the state
is CERTIFIED_PRIOR_ANCHORED. Everything here is pure except `make_refit_fn` / `adapt_dual` (they touch the dual model).
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from .config import (COLD_START_CERT_FRACTIONS, COLD_START_CERT_MIN_BLOCK, COLD_START_CERT_MIN_ROWS, COLD_START_CERT_MIN_TRAIN, COLD_START_CERT_TAU, COLD_START_CERT_Z)

STATE_CERTIFIED = "CERTIFIED_PRIOR_ANCHORED"
STATE_NOT_CERTIFIED = "NOT_CERTIFIED"
STATE_NOT_APPLICABLE = "NOT_APPLICABLE"

R_TOO_FEW_ROWS = "too_few_rows"
R_NOT_PRIOR_ANCHORED = "not_prior_anchored"
R_CONVERGED = "fit_already_converged"
R_OTHER = "other_withhold_reasons"
R_STATE = "em_state_not_eligible"
R_REFIT = "refit_failed"
R_DEGRADED_REFIT = "refit_degraded"
R_NONFINITE = "non_finite_scores"
R_GAIN = "gain_not_demonstrated"
R_UNSTABLE = "parameters_unstable"
R_ERROR = "certification_error"

ELIGIBLE_EM_STATES = ("REJECTED_UPDATE", "MAX_ITER_REACHED")


# ------------------------------------------------------------------------------------------------------------------- pure parts
def origin_blocks(T: int, fractions: Sequence[float] = COLD_START_CERT_FRACTIONS, min_train: int = COLD_START_CERT_MIN_TRAIN, min_block: int = COLD_START_CERT_MIN_BLOCK):
    """[(origin, end), ...]: the model is fitted on rows [0, origin) and scored on rows [origin, end); the last block ends at T. Returns (blocks, None) or (None, reason)."""
    if T < COLD_START_CERT_MIN_ROWS:
        return None, R_TOO_FEW_ROWS
    origins = [int(math.ceil(f * T - 1e-9)) for f in fractions]
    ends = origins[1:] + [int(T)]
    blocks = list(zip(origins, ends))
    if origins[0] < min_train or any(e - o < min_block for o, e in blocks) or any(b <= a for a, b in zip(origins, origins[1:])):
        return None, R_TOO_FEW_ROWS
    return blocks, None


def hac_se(d: np.ndarray, bandwidth: int | None = None) -> float:
    """Standard error of the mean of a (possibly autocorrelated) series: Bartlett-kernel long-run variance / n, bandwidth ceil(n / 4) by default, capped at n - 1."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 3 or not np.all(np.isfinite(d)):
        return float("nan")
    m = min(int(math.ceil(n / 4.0)) if bandwidth is None else int(bandwidth), n - 1)
    c = d - d.mean()
    lrv = float(c @ c) / n
    for j in range(1, m + 1):
        lrv += 2.0 * (1.0 - j / (m + 1.0)) * float(c[:-j] @ c[j:]) / n
    return math.sqrt(max(lrv, 0.0) / n)


def gaussian_heldout(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Log density of the held-out rows under the single Gaussian fitted on the training rows (mean, covariance ddof 1). Raises ValueError when the covariance is singular."""
    from scipy.stats import multivariate_normal
    mu = X_train.mean(axis=0)
    cov = np.atleast_2d(np.cov(X_train, rowvar=False, ddof=1))
    try:
        return np.asarray(multivariate_normal(mean=mu, cov=cov, allow_singular=False).logpdf(X_test), dtype=float).reshape(-1)
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise ValueError(f"single Gaussian is not defined on the earlier rows: {exc}") from exc


def stability_distance(means_full: np.ndarray, means_last: np.ndarray, feature_sd: np.ndarray) -> float:
    """Root mean square over regimes and features of the regime observation mean difference in single-Gaussian standard deviations, regimes matched by the permutation of least distance."""
    import itertools
    a, b, sd = np.asarray(means_full, float), np.asarray(means_last, float), np.asarray(feature_sd, float)
    if a.shape != b.shape or a.ndim != 2 or sd.shape != (a.shape[1],) or not np.all(sd > 0) or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("nan")
    best = min(float((((a - b[list(p)]) / sd) ** 2).mean()) for p in itertools.permutations(range(a.shape[0])))
    return math.sqrt(best)


def decide(increments: np.ndarray | None, stability: float | None, z: float = COLD_START_CERT_Z, tau: float = COLD_START_CERT_TAU) -> dict:
    """The certification verdict from the pooled increments and the stability distance. Certified iff every increment is finite, mean - z * se > 0 (strictly) and stability <= tau."""
    reasons: list[str] = []
    d = None if increments is None else np.asarray(increments, dtype=float)
    stats = dict(n_increments=None if d is None else int(len(d)), mean_gain=None, se=None, lower=None, z=float(z), stability=None if stability is None else _f(stability), tau=float(tau))
    if d is None or len(d) < 3 or not np.all(np.isfinite(d)):
        reasons.append(R_NONFINITE)
    else:
        mean, se = float(d.mean()), hac_se(d)
        stats.update(mean_gain=mean, se=_f(se))
        if not math.isfinite(se):
            reasons.append(R_NONFINITE)
        else:
            lower = mean - float(z) * se
            stats["lower"] = lower
            if not lower > 0.0:
                reasons.append(R_GAIN)
    if stability is None or not math.isfinite(float(stability)):
        reasons.append(R_NONFINITE)
    elif not float(stability) <= float(tau):
        reasons.append(R_UNSTABLE)
    reasons = list(dict.fromkeys(reasons))
    return dict(state=STATE_NOT_CERTIFIED if reasons else STATE_CERTIFIED, reasons=reasons, **stats)


def _f(x):
    x = float(x)
    return x if math.isfinite(x) else None


# ------------------------------------------------------------------------------------------------------------------- the procedure
def certify_prior_anchored(X: np.ndarray, full_model, refit_fn: Callable[[np.ndarray], object], *, eligibility: dict | None = None, z: float = COLD_START_CERT_Z,
                           tau: float = COLD_START_CERT_TAU, fractions: Sequence[float] = COLD_START_CERT_FRACTIONS) -> dict:
    """Certify one prior-anchored fit. `full_model` and the models returned by `refit_fn(X_train)` are adapters with `predictive_increments(X) -> per-step log p(y_t | y_<t)` (frozen
    parameters), `regime_observation_means() -> (K, F)` and `degraded` (bool). `eligibility` is what the caller knows about the full fit: used_shared_prior, n0, em_state,
    other_reasons (the withhold reasons other than non-convergence). Never raises: any failure is a NOT_CERTIFIED verdict with a reason."""
    el = eligibility or {}
    base = dict(state=STATE_NOT_CERTIFIED, reasons=[], z=float(z), tau=float(tau), n_increments=None, mean_gain=None, se=None, lower=None, stability=None, blocks=None, schema_version=1)
    try:
        if not el.get("used_shared_prior") or not (el.get("n0") or 0) > 0:
            return dict(base, state=STATE_NOT_APPLICABLE, reasons=[R_NOT_PRIOR_ANCHORED])
        if el.get("em_state") == "CONVERGED":
            return dict(base, state=STATE_NOT_APPLICABLE, reasons=[R_CONVERGED])
        if el.get("em_state") not in ELIGIBLE_EM_STATES:
            return dict(base, reasons=[R_STATE])
        if el.get("other_reasons"):
            return dict(base, reasons=[R_OTHER], other_reasons=list(el["other_reasons"]))
        X = np.asarray(X, dtype=float)
        blocks, why = origin_blocks(len(X), fractions)
        if blocks is None:
            return dict(base, reasons=[why])
        if not np.all(np.isfinite(X)):
            return dict(base, reasons=[R_NONFINITE])
        parts, last = [], None
        for o, e in blocks:
            try:
                m = refit_fn(X[:o])
            except Exception:                                                                                      # noqa: BLE001  an unfinished refit is a refusal
                return dict(base, reasons=[R_REFIT], blocks=[list(b) for b in blocks])
            if getattr(m, "degraded", False):
                return dict(base, reasons=[R_DEGRADED_REFIT], blocks=[list(b) for b in blocks])
            lp = np.asarray(m.predictive_increments(X[:e]), dtype=float)[o:e]
            parts.append(lp - gaussian_heldout(X[:o], X[o:e]))
            last = m
        d = np.concatenate(parts)
        sd = X.std(axis=0, ddof=1)
        stab = stability_distance(full_model.regime_observation_means(), last.regime_observation_means(), sd)
        out = decide(d, stab, z=z, tau=tau)
        out.update(blocks=[list(b) for b in blocks], schema_version=1)
        return out
    except Exception as exc:                                                                                       # noqa: BLE001  fail closed
        return dict(base, reasons=[R_ERROR], error=f"{type(exc).__name__}: {str(exc)[:120]}")


# ------------------------------------------------------------------------------------------------------------------- the dual model side
class DualAdapter:
    """Adapter of a fitted dual model (DualTimescaleHSSM / ShrinkageDualHSSM) to the interface of `certify_prior_anchored`."""

    def __init__(self, model):
        self.model = model

    @property
    def degraded(self) -> bool:
        return bool(getattr(self.model, "fit_degraded_", False))

    def predictive_increments(self, X: np.ndarray) -> np.ndarray:
        from scipy.special import logsumexp
        fwd = self.model._forward_pass(np.asarray(X, dtype=float))
        lse = logsumexp(fwd.alpha.reshape(len(X), -1), axis=1)
        return np.diff(np.concatenate([[0.0], lse]))

    def regime_observation_means(self) -> np.ndarray:
        from .shrinkage import extract_params, regime_observation_means
        return regime_observation_means(extract_params(self.model))


def make_refit_fn(*, n_regimes: int, n_features: int, latent_dim: int, n_init: int, max_iter: int, max_duration: int, base_seed: int, shared_prior, shared_prior_n0) -> Callable:
    """The refit procedure of the certification: exactly the production dual fit (`fit_dual_with_random_restarts`, the same seed, prior, n0, iterations, Dmax) on fewer rows, with the
    cold-start gate bypassed (these are internal validation fits, never published). Raises when the fit does not finish (the caller treats that as a refusal)."""
    def refit(X_train: np.ndarray) -> DualAdapter:
        from .fitting import fit_dual_with_random_restarts
        model, _ = fit_dual_with_random_restarts(X_train, n_regimes=n_regimes, n_features=n_features, latent_dim=latent_dim, n_init=n_init, n_iter=max_iter, max_duration=max_duration,
                                                 base_seed=base_seed, bypass_init_gate=True, shared_prior=shared_prior, shared_prior_n0=shared_prior_n0)
        return DualAdapter(model)
    return refit
