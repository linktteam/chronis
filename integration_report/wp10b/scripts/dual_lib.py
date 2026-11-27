"""WP10B dual-engine study library (B1 to B5): data of an identity-stream user for a train/test split, one dual fit through the PUBLIC path with a named variant, the reasons a fit is withheld, the EM-creep diagnostics, and the
held-out one-step predictive scores of the dual fit, the baseline engine and the single Gaussian (frozen parameters, forward filter over train + test). Study use only. Design: tests/PREDECLARED_CRITERIA_WP10B.md Part 3.

Setting S0 (the study setting; every item of B1 to B5 starts from it): identity stream user, K = 3, F = 7, train days [0, 120) with the per-window z-scoring of the integration pipeline, test days [120, 150) scored with the TRAIN
scaler; dual fit = `fit_hssm(model_type="dual_timescale", candidate_ks=(3,), n_initializations=1, max_iter=200, max_duration=45, latent_dim=2, allow_fast_test_fit=True)` (production defaults except the fast-test switch and
one restart, as the WP9/WP10A dual studies); baseline = `fit_hssm(candidate_ks=(3,), n_initializations=10, max_iter=60)` (production defaults)."""
from __future__ import annotations

import contextlib
import io
import json
import logging
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integration_report" / "wp9" / "scripts"))

K_STUDY, T_TRAIN, T_END = 3, 120, 150
S0 = dict(n_initializations=1, max_iter=200, max_duration=45, latent_dim=2, allow_fast_test_fit=True, model_type="dual_timescale")
# variants of the dual fit: keyword overrides of S0 that need no engine change (B2 adds opt-in engine switches as they are implemented)
# `_flags` = opt-in class switches of `DualTimescaleHSSM` (WP10B B2), set for the duration of one fit and restored afterwards; every other key is a `fit_hssm` keyword override of S0.
VARIANTS = {"S0": {}, "D60": dict(max_duration=60), "I600": dict(max_iter=600, _flags=dict(MAX_FIT_SECONDS=900.0)), "REG": dict(_flags=dict(DURATION_PRIOR_SEGMENTS=1.0)),
            "ALL": dict(max_duration=60, max_iter=600, _flags=dict(MAX_FIT_SECONDS=900.0, DURATION_PRIOR_SEGMENTS=1.0))}
CONVERGE_TOL = 1e-4


# ------------------------------------------------------------------------------------------------- data
def user_data(user: int, stratum: str, t_train: int = T_TRAIN, t_end: int = T_END) -> dict:
    """The matrix the pipeline would build for days [0, t_end) with the scaler of the TRAIN days only: z-scored per feature, a single missing feature filled with 0, an all-missing day left as a NaN row."""
    import wp9_fit_lib as F
    st = F.stream_for(dict(user=user, stratum=stratum))
    raw = st.X_obs[:t_end]
    mean = np.nanmean(raw[:t_train], axis=0)
    std = np.nanstd(raw[:t_train], axis=0)
    std = np.where(std > 0, std, 1.0)
    z = (raw - mean) / std
    day_missing = np.isnan(raw).all(axis=1)
    z = np.where(np.isnan(z), 0.0, z)
    z[day_missing] = np.nan
    return dict(X=z, T_train=t_train, day_missing=day_missing, scaler_mean=mean, scaler_std=std, labels=st.labels[:t_end], raw=raw)


# ------------------------------------------------------------------------------------------------- frozen-parameter one-step scores
def dual_increments(model, X: np.ndarray) -> np.ndarray:
    """lp[t] = log p(y_t | y_1:t-1) under the dual model's own forward filter with frozen parameters (the harness of ledger N18: `tests/utils/heldout_prediction.predictive_increments`)."""
    fwd = model._forward_pass(X)
    lse = logsumexp(fwd.alpha.reshape(len(X), -1), axis=1)
    return np.diff(np.concatenate([[0.0], lse]))


def baseline_increments(model, X: np.ndarray) -> np.ndarray:
    """The same for the baseline Gaussian HSMM: the forward recursion of `GaussianHSMM._forward_backward` on the expanded (regime, remaining duration) state space, frozen parameters; lp[t] = log p(y_t | y_1:t-1)."""
    T, K, D = X.shape[0], model.K, model.Dmax
    emis = model._emission_loglik(X)
    dur_logpmf = model._duration_logpmf()
    logA = np.log(np.clip(model.A, 1e-300, None))
    log_pi = np.log(np.clip(model.pi, 1e-300, None))
    NEG = -np.inf
    alpha = np.full((T, K, D), NEG)
    for k in range(K):
        alpha[0, k, :] = log_pi[k] + dur_logpmf[k] + emis[0][k]
    for t in range(1, T):
        cont = np.full((K, D), NEG)
        cont[:, : D - 1] = alpha[t - 1, :, 1:D]
        trans_in = logsumexp(alpha[t - 1, :, 0][:, None] + logA, axis=0)
        entry = trans_in[:, None] + dur_logpmf
        alpha[t] = np.logaddexp(cont, entry) + emis[t][:, None]
    lse = logsumexp(alpha.reshape(T, -1), axis=1)
    return np.diff(np.concatenate([[0.0], lse]))


def gaussian_increments(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Log density of each test row under the single Gaussian fitted on the train rows (all-missing rows dropped before the fit; mean, covariance ddof 1); NaN rows of the test block give NaN."""
    tr = X_train[~np.isnan(X_train).all(axis=1)]
    mu = tr.mean(axis=0)
    cov = np.atleast_2d(np.cov(tr, rowvar=False, ddof=1))
    out = np.full(len(X_test), np.nan)
    ok = ~np.isnan(X_test).any(axis=1)
    if ok.any():
        out[ok] = multivariate_normal(mean=mu, cov=cov, allow_singular=False).logpdf(X_test[ok])
    return out


def test_mean(lp_full: np.ndarray, t_train: int, day_missing: np.ndarray) -> tuple:
    """(mean over the OBSERVED test days, number of such days, all finite) of the test-block increments."""
    lp = np.asarray(lp_full, dtype=float)[t_train:]
    keep = ~np.asarray(day_missing)[t_train:len(lp_full)]
    x = lp[keep]
    return (float(np.mean(x)) if len(x) else float("nan")), int(len(x)), bool(len(x) > 0 and np.all(np.isfinite(x)))


# ------------------------------------------------------------------------------------------------- diagnosis
def reason_categories(reasons: list) -> list:
    """The categories of the strings of `HSSMResult._withheld_reasons()`: iteration_cap (MAX_ITER_REACHED), rejected_update (REJECTED_UPDATE), other_non_convergence, collapsed_regime, degenerate_duration, numerical
    (the Cholesky fallback, NUMERICAL_FAILURE, the states pass, late fallbacks), other. A fit can have several."""
    cats = []
    for r in reasons:
        s = str(r)
        if s.startswith("fit_state_not_converged"):
            cats.append("iteration_cap" if "MAX_ITER_REACHED" in s else ("rejected_update" if "REJECTED_UPDATE" in s else "other_non_convergence"))
        elif s.startswith("collapsed_regime"):
            cats.append("collapsed_regime")
        elif s.startswith("degenerate_duration"):
            cats.append("degenerate_duration")
        elif s.startswith(("fit_degraded", "fit_state is NUMERICAL_FAILURE", "states_pass", "late_fallbacks")):
            cats.append("numerical")
        else:
            cats.append("other")
    return sorted(set(cats))


def creep_diagnostics(hist: list, tol: float = CONVERGE_TOL, window: int = 50) -> dict:
    """EM creep (WP7-R S1): the geometric rate of the log-likelihood increments over the last `window` iterations (median ratio of consecutive positive increments) and the number of FURTHER iterations the rate
    extrapolates to bring the increment below `tol` (None when the rate is not below 1 or there are too few increments)."""
    h = np.asarray(hist, dtype=float)
    d = np.diff(h)
    out = dict(n_iter=int(len(h)), last_increment=float(d[-1]) if len(d) else float("nan"), gain_last10=float(h[-1] - h[-11]) if len(h) > 10 else float("nan"), rate=None, extra_iterations=None)
    tail = d[-window:]
    pos = [(a, b) for a, b in zip(tail[:-1], tail[1:]) if a > 0 and b > 0]
    if len(pos) >= 5:
        r = float(np.median([b / a for a, b in pos]))
        out["rate"] = r
        if 0 < r < 1 and d[-1] > 0:
            out["extra_iterations"] = 0 if d[-1] < tol else int(math.ceil(math.log(tol / d[-1]) / math.log(r)))
    return out


def fit_dual(Xtr: np.ndarray, seed: int, variant: str = "S0", extra: dict | None = None):
    """One dual fit through the public path with the named variant; returns the HSSMResult (raises what fit_hssm raises)."""
    from backbone.hssm.dual_model import DualTimescaleHSSM
    from backbone.hssm.fitting import fit_hssm
    kw = dict(S0)
    kw.update(VARIANTS[variant])
    kw.update(extra or {})
    flags = dict(kw.pop("_flags", {}))
    old = {k: getattr(DualTimescaleHSSM, k) for k in flags}
    try:
        for k, v in flags.items():
            setattr(DualTimescaleHSSM, k, v)
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            return fit_hssm(matrix=Xtr, candidate_ks=(K_STUDY,), random_seed=seed, **kw)
    finally:
        for k, v in old.items():
            setattr(DualTimescaleHSSM, k, v)


def fit_baseline(Xtr: np.ndarray, seed: int):
    from backbone.hssm.fitting import fit_hssm
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        return fit_hssm(matrix=Xtr, candidate_ks=(K_STUDY,), n_initializations=10, max_iter=60, random_seed=seed)


def dual_record(res, model) -> dict:
    """Everything B1 to B5 read from one dual fit."""
    from backbone.hssm.fitting import dual_duration_degeneracy
    reasons = list(res._withheld_reasons())
    deg = dual_duration_degeneracy(model)
    hist = [float(x) for x in model.log_likelihood_history_]
    coll = dict(res.collapse_report or {})
    box = model.duration_box_refit_
    return dict(fit_state=str(res.fit_state.value), publishable=bool(res.withheld_reason is None), reasons=[str(r)[:160] for r in reasons], categories=reason_categories(reasons),
                n_iter=int(model.n_iter_), criterion=model.convergence_criterion_, window_gain_at_cap=None if model.window_gain_at_cap_ is None else float(model.window_gain_at_cap_),
                ll_final=None if model.log_likelihood_ is None else float(model.log_likelihood_), ll_history=hist, creep=creep_diagnostics(hist),
                dur_mu=[float(x) for x in model.dur_mu], dur_sigma=[float(x) for x in model.dur_sigma], Dmax=int(model.Dmax), at_bound={int(k): bool(v) for k, v in model.duration_at_bound_.items()},
                pmf_rising=[bool(x) for x in deg["pmf_rising_at_end"]], degenerate_regimes=[int(k) for k in deg["degenerate_regimes"]],
                min_effective_days=None if "min_effective_days" not in coll else float(coll["min_effective_days"]), fit_collapsed=bool(res.fit_collapsed), fit_degraded=bool(res.fit_degraded),
                cholesky_fallbacks=int(res.fit_cholesky_fallbacks), exceptions=[str(e.get("exception_type")) for e in res.fit_exceptions],
                box_refit=None if box is None else dict(first_state=str(getattr(box.get("first_fit_state"), "value", box.get("first_fit_state"))), first_n_iter=int(box.get("first_fit_n_iter", -1))),
                optimizer_success=bool(res.optimizer_success), wall_clock_stop=bool(getattr(model, "wall_clock_stop_", False)))


def fit_job(job: dict) -> dict:
    """job: user, stratum, variant (a key of VARIANTS), seed offset 0. One record: the dual fit (diagnosis and held-out one-step mean), the baseline engine's and the single Gaussian's held-out mean on the same days."""
    logging.disable(logging.CRITICAL)
    t0 = time.time()
    rec = dict(job=job, ok=False)
    d = user_data(job["user"], job["stratum"], job.get("t_train", T_TRAIN), job.get("t_end", T_END))
    X, T_tr, dm = d["X"], d["T_train"], d["day_missing"]
    Xtr = X[:T_tr]
    rec.update(T_train=int(T_tr), n_test_days=int((~dm[T_tr:]).sum()))
    try:
        t1 = time.time()
        res = fit_dual(Xtr, job["user"], job.get("variant", "S0"), job.get("extra"))
        model = res.model
        rec["dual"] = dual_record(res, model)
        rec["dual"]["fit_seconds"] = round(time.time() - t1, 2)
        m, n, fin = test_mean(dual_increments(model, X), T_tr, dm)
        rec["dual"].update(test_mean=m, test_finite=fin)
        rec["ok"] = True
    except Exception as exc:                                                   # a failed fit is data too
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    if job.get("with_baseline", True):
        try:
            b = fit_baseline(Xtr, job["user"])
            m, n, fin = test_mean(baseline_increments(b.model, X), T_tr, dm)
            rec["baseline"] = dict(fit_state=str(b.fit_state.value), publishable=bool(b.withheld_reason is None), test_mean=m, test_finite=fin)
        except Exception as exc:
            rec["baseline"] = dict(error=f"{type(exc).__name__}: {str(exc)[:200]}", test_mean=float("nan"), test_finite=False)
        try:
            m, n, fin = test_mean(np.concatenate([np.full(T_tr, np.nan), gaussian_increments(Xtr, X[T_tr:])]), T_tr, dm)
            rec["gauss"] = dict(test_mean=m, test_finite=fin)
        except Exception as exc:
            rec["gauss"] = dict(error=f"{type(exc).__name__}: {str(exc)[:200]}", test_mean=float("nan"), test_finite=False)
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def job_key(job: dict) -> str:
    return json.dumps(job, sort_keys=True)


def load_records(path: str) -> list:
    return [json.loads(l) for l in open(path) if l.strip()]


def run_jobs(jobs: list, out_path: str, workers: int = 2) -> None:
    """Resumable: a job already in `out_path` is skipped; each record is appended, flushed and fsynced as it arrives."""
    import multiprocessing as mp
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(job_key(json.loads(line)["job"]))
            except Exception:
                pass
    todo = [j for j in jobs if job_key(j) not in done]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run", flush=True)
    t0 = time.time()
    with open(out_path, "a") as fh, mp.Pool(min(workers, max(1, len(todo)))) as pool:
        for i, rec in enumerate(pool.imap_unordered(fit_job, todo), 1):
            fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())
            dd = rec.get("dual", {})
            print(i, "/", len(todo), rec["job"]["user"], rec["job"].get("variant", "S0"), "ok" if rec["ok"] else rec.get("error"), dd.get("fit_state"), "publishable" if dd.get("publishable") else dd.get("categories"),
                  rec["seconds"], "s", f"{time.time() - t0:.0f}s elapsed", flush=True)
