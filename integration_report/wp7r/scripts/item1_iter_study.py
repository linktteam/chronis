"""WP7-R item 1: does more EM make the dual model predict worse? (design: tests/PREDECLARED_CRITERIA_WP7R.md section 1; committed before this script was run on any scored seed)

usage: PYTHONPATH=. python3 integration_report/wp7r/scripts/item1_iter_study.py --group N18|DPOP --seeds 5201:5221 --out wp7r_evidence/item1_iter.jsonl [--workers 2] [--max-iter 160]
       (--seeds accepts a:b ranges and comma lists; one JSON line per seed is appended as soon as the seed is finished; seeds already in the file are skipped: resumable)

One EM trajectory per seed (DURATION_BOX_REFIT off: the pure EM path, so "more iterations" is literally the same path continued), parameters snapshotted after the
20th, 40th, 80th and 160th M-step. That is what `fit(n_iter=N)` returns for N = 20, 40, 80, 160 (the loop is deterministic in N; a fit that stops early by abs_tol or
by the guard returns the same parameters for every larger N, and the snapshot of the missing checkpoints is that final state). Every checkpoint is scored here, with
this file's own code, as in tests/utils/heldout_prediction.py: forward filter over train + held-out, parameters frozen, mean one-step-ahead predictive log density per
held-out step; plus the training log-likelihood per row and the single-Gaussian baseline score.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import sys
import time
import warnings

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

CHECKPOINTS = (20, 40, 80, 160)
K, F, L, DMAX = 2, 4, 2, 25


def _parse_seeds(txt: str) -> list[int]:
    out: list[int] = []
    for part in txt.split(","):
        if ":" in part:
            a, b = part.split(":")
            out += list(range(int(a), int(b)))
        elif part:
            out.append(int(part))
    return out


def _run_lengths(path) -> list[int]:
    path = np.asarray(path)
    cuts = np.flatnonzero(np.diff(path) != 0) + 1
    return [int(len(seg)) for seg in np.split(path, cuts)]


def one(job):
    group, seed, max_iter = job
    from backbone.hssm.dual_model import DualTimescaleHSSM

    class CheckpointDual(DualTimescaleHSSM):
        DURATION_BOX_REFIT = False
        snaps: dict

        def _m_step_continuous(self, X, bwd, smoother):
            out = super()._m_step_continuous(X, bwd, smoother)
            if not hasattr(self, "snaps"):
                self.snaps = {}
            n_done = self._em_iteration + 1
            if n_done in CHECKPOINTS:
                self.snaps[n_done] = self._snapshot_params()
            return out

    rec: dict = {"group": group, "seed": int(seed), "ok": False}
    t0 = time.time()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        try:
            if group == "N18":
                from tests.utils.heldout_prediction import continued_data
                d = continued_data(seed)
                X_tr, X_te, p_tr = d.X_train, d.X_test, d.p_train
                mseed = seed
            else:
                from data.synthetic_population import generate_population_user
                u = generate_population_user(seed, 12, 2)
                X_tr, X_te, p_tr = u.X_train, u.X_test, u.p_train
                mseed = seed + 2000
            X_full = np.vstack([X_tr, X_te])
            T_tr = len(X_tr)
            m = CheckpointDual(n_regimes=K, n_features=F, latent_dim=L, max_duration=DMAX, seed=mseed)
            m.fit(X_tr, n_iter=max_iter)
            snaps = dict(getattr(m, "snaps", {}))
            final = m._snapshot_params()
            lens = _run_lengths(p_tr)
            rec.update(ok=True, fit_state=m.fit_state_.value, n_iter_run=int(m.n_iter_), criterion=m.convergence_criterion_, T_train=int(T_tr), T_test=int(len(X_te)),
                       ll_hist=[float(x) for x in m.log_likelihood_history_], seg_lengths=lens, max_seg=int(max(lens)), n_seg_over_dmax=int(sum(1 for x in lens if x > DMAX)),
                       gauss_heldout=float(np.mean(multivariate_normal(mean=X_tr.mean(axis=0), cov=np.cov(X_tr, rowvar=False, ddof=1)).logpdf(X_te))))
            mu_lo, mu_hi, sg_hi = m._duration_box()
            cps = {}
            for n in CHECKPOINTS:
                snap = snaps.get(n, final)          # the fit stopped before this checkpoint: its final parameters are what fit(n_iter=n) returns
                m._restore_params(snap)
                fwd = m._forward_pass(X_full)
                lse = logsumexp(np.asarray(fwd.alpha).reshape(len(X_full), -1), axis=1)
                lp = np.diff(np.concatenate([[0.0], lse]))
                fwd_tr = m._forward_pass(X_tr)
                mu, sg = np.asarray(m.dur_mu, float), np.asarray(m.dur_sigma, float)
                cps[str(n)] = dict(heldout=float(np.mean(lp[T_tr:])), heldout_finite=bool(np.all(np.isfinite(lp[T_tr:]))), train_ll_per_row=float(fwd_tr.log_likelihood) / T_tr,
                                   dur_mu=[float(x) for x in mu], dur_sigma=[float(x) for x in sg],
                                   degenerate=bool(np.any(~np.isfinite(mu) | ~np.isfinite(sg) | (mu < mu_lo) | (mu > mu_hi) | (sg > sg_hi))), reached=bool(n in snaps))
            rec["checkpoints"] = cps
        except Exception as exc:  # noqa: BLE001
            rec.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:300]}")
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["N18", "DPOP"])
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-iter", type=int, default=160)
    a = ap.parse_args()
    seeds = _parse_seeds(a.seeds)
    done = set()
    if os.path.exists(a.out):
        for ln in open(a.out):
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            if d.get("ok"):
                done.add((d["group"], d["seed"]))
    jobs = [(a.group, s, a.max_iter) for s in seeds if (a.group, s) not in done]
    print(f"{len(jobs)} jobs ({len(done)} already done)", flush=True)
    if not jobs:
        return
    with mp.Pool(min(a.workers, len(jobs))) as pool:
        for rec in pool.imap_unordered(one, jobs):
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(rec["group"], rec["seed"], rec.get("fit_state"), rec.get("n_iter_run"), f"{rec['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
