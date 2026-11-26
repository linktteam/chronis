"""WP9 study library: baseline fits with ALL restarts kept, so that "4, 10 or 20 restarts" and every selection / consensus rule can be derived offline from the same run.

Why this exists. `fitting.fit_with_random_restarts` keeps only the selected restart. The WP9 questions (A1: how often do two seeds of the same data give different partitions, as a
function of the number of restarts; A2: does agreement of the top restarts help the identifiability gate) need every restart's labelling and log-likelihood. This module repeats the
production restart loop one restart at a time with exactly the same seeds (`base_seed + k * 1000 + i`, as `select_k_by_bic` does) and the same model, and the SELECTION is done offline
by `select_restart`, which implements the production rule (best log-likelihood among CONVERGED restarts that are not collapsed; if every converged restart is collapsed, the best one).
`tests/test_wp9_fit_lib.py` checks, against `fit_hssm` itself, that the selected restart and its labelling are the same as production for the same seeds.

Nothing here changes `backbone/`. Records are JSON-able and one fit is one line of a jsonl file (resumable runner `run_jobs`, as in WP8).
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import time
import warnings
from typing import Iterable

import numpy as np

from backbone.hssm.config import COLLAPSE_MIN_EFFECTIVE_DAYS
from backbone.hssm.fitting import adjusted_rand_index, collapse_report
from backbone.hssm.label_switching import canonicalize_labels
from backbone.hssm.model import FitState, GaussianHSMM
from backbone.hssm.shrinkage import duration_pmf

MAX_DURATION = 45                       # fit_hssm default for the baseline model; the WP8 evaluation used it too
FIT_STATES_WITH_LABELS = (FitState.CONVERGED, FitState.REJECTED_UPDATE, FitState.MAX_ITER_REACHED)

# strata of the WP9 generator: the planted separation (in within-regime sd, log-uniform in [lo, hi]) is an ASSUMPTION of the study, never a measurement
STRATA = {
    "wp8": None,                        # the WP8 generator default, sep in [0.5, 2.0]
    "hard": (0.4, 0.9),
    "medium": (0.9, 1.6),
    "easy": (1.6, 3.0),
}


def spec_for(stratum: str):
    from data.synthetic_identity import IdentitySpec
    bounds = STRATA[stratum]
    return IdentitySpec() if bounds is None else IdentitySpec(sep_lo=bounds[0], sep_hi=bounds[1])


def stream_for(job: dict):
    from data.synthetic_identity import generate_identity_stream
    return generate_identity_stream(job["user"], n_days=job.get("n_days", 150), spec=spec_for(job.get("stratum", "wp8")), drift_sd=job.get("drift", 0.0), k_change=job.get("kchange"))


def labels_to_str(labels) -> str:
    return "".join(str(int(x)) for x in labels)


def str_to_labels(s: str) -> np.ndarray:
    return np.fromiter((int(c) for c in s), dtype=int, count=len(s))


def fit_restarts(X: np.ndarray, K: int, fitseed: int, n_init: int, max_iter: int = 60, max_duration: int = MAX_DURATION) -> list[dict]:
    """Every restart of the production loop for one K: restart i uses seed fitseed + K * 1000 + i. Returns one record per restart (a restart whose fit raised is recorded as FAILED)."""
    F = X.shape[1]
    out = []
    for i in range(n_init):
        seed = int(fitseed) + int(K) * 1000 + i
        model = GaussianHSMM(n_regimes=K, n_features=F, max_duration=max_duration, seed=seed)
        rec = dict(i=i, seed=seed)
        try:
            model.fit(X, n_iter=max_iter, verbose=False)
            state = model.fit_state_
        except Exception as exc:                                       # a restart that fails is data too
            rec.update(state="FAILED", error=f"{type(exc).__name__}: {exc}"[:120])
            out.append(rec)
            continue
        rec.update(state=str(state.value), n_iter=int(model.n_iter_), ll=None if model.log_likelihood_ is None else float(model.log_likelihood_))
        if state in FIT_STATES_WITH_LABELS:
            report, _ = collapse_report(model, X)
            rec.update(collapsed=bool(report["collapsed"]), min_eff=float(report["min_effective_days"]))
        if state == FitState.CONVERGED:
            canon = canonicalize_labels(model)                          # as the production loop does after selection; in place
            _, labels = collapse_report(canon, X)
            post = canon._forward_backward(X)[0]
            pmf = duration_pmf(np.array(canon.dur_mu), np.array(canon.dur_sigma), int(canon.Dmax))
            rec.update(labels=labels_to_str(labels), z_means=np.array(canon.mu, dtype=float).tolist(), max_post=float(post.max(axis=1).mean()),
                       eff=[float(x) for x in post.sum(axis=0)], exp_dur=(pmf * np.arange(1, int(canon.Dmax) + 1)).sum(axis=1).tolist(), dmax=int(canon.Dmax))
        out.append(rec)
    return out


def select_restart(restarts: list[dict], n: int | None = None):
    """The production selection rule on the first `n` restarts: the CONVERGED restart with the highest log-likelihood that is not collapsed; if every converged restart is collapsed, the
    best one of them. Returns the chosen restart record or None (no restart converged: the fit fails, as in production)."""
    pool = [r for r in (restarts if n is None else restarts[:n]) if r.get("state") == FitState.CONVERGED.value and r.get("ll") is not None]
    if not pool:
        return None
    healthy = [r for r in pool if not r["collapsed"]]
    return max(healthy or pool, key=lambda r: r["ll"])


def fit_job(job: dict) -> dict:
    """job: user, start, end, fitseed, K, n_init, max_iter, stratum, drift, kchange. One record per fit with all restarts."""
    logging.disable(logging.CRITICAL)
    t0 = time.time()
    st = stream_for(job)
    w = st.window(job["start"], job["end"])
    rec = dict(job=job, ok=False)
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            restarts = fit_restarts(w["matrix"], job["K"], job["fitseed"], job["n_init"], job.get("max_iter", 60))
        rec.update(ok=True, restarts=restarts, scaler_mean=w["scaler_mean"].tolist(), scaler_std=w["scaler_std"].tolist(), coverage=w["coverage"].tolist(),
                   labels_true=labels_to_str(w["labels"]), sep=float(st.truth["sep"]) if "sep" in st.truth else None)
    except Exception as exc:
        rec.update(error=f"{type(exc).__name__}: {exc}"[:200])
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def job_key(job: dict) -> str:
    return json.dumps({k: job[k] for k in sorted(job) if k not in ("n_init", "max_iter")}, sort_keys=True)


def run_jobs(jobs: Iterable[dict], out_path: str, workers: int = 2) -> None:
    """Resumable: a job whose key is already in `out_path` is skipped; each record is appended, flushed and fsynced as soon as it arrives."""
    import multiprocessing as mp
    jobs = list(jobs)
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(job_key(json.loads(line)["job"]))
            except Exception:
                pass
    todo = [j for j in jobs if job_key(j) not in done]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run", flush=True)
    with open(out_path, "a") as fh, mp.Pool(workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(fit_job, todo), 1):
            fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())
            print(i, "/", len(todo), rec["job"]["user"], rec["job"]["start"], rec["job"]["end"], rec["job"]["fitseed"], "ok" if rec["ok"] else rec.get("error"), rec["seconds"], "s", flush=True)


def load_records(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


# ------------------------------------------------------------------------------------------------- partition agreement
def partition_ari(a: str, b: str) -> float:
    return adjusted_rand_index(str_to_labels(a), str_to_labels(b))


def top_restarts(restarts: list[dict], n: int | None, epsilon: float | None = None, m: int | None = None) -> list[dict]:
    """Converged, non-collapsed restarts among the first `n`, best log-likelihood first; optionally only those within `epsilon` nats of the best, or only the best `m`."""
    pool = [r for r in (restarts if n is None else restarts[:n]) if r.get("state") == FitState.CONVERGED.value and r.get("ll") is not None and not r["collapsed"]]
    pool.sort(key=lambda r: -r["ll"])
    if epsilon is not None and pool:
        pool = [r for r in pool if pool[0]["ll"] - r["ll"] <= epsilon]
    if m is not None:
        pool = pool[:m]
    return pool


def consensus_ari(restarts: list[dict], n: int | None, m: int = 3) -> float | None:
    """The smallest pairwise ARI among the best `m` converged non-collapsed restarts of the first `n` (None if fewer than 2 such restarts)."""
    top = top_restarts(restarts, n, m=m)
    if len(top) < 2:
        return None
    return min(partition_ari(top[i]["labels"], top[j]["labels"]) for i in range(len(top)) for j in range(i + 1, len(top)))
