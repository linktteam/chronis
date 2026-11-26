"""WP9 A3: fits for the K-change study (design: tests/PREDECLARED_CRITERIA_WP9.md Part 7, protocol K). usage:
  PYTHONPATH=. python3 integration_report/wp9/scripts/a3_run.py --range dev|eval --out FILE.jsonl [--workers 1]
dev = users 9221..9232, eval = users 9241..9252 (refused without tests/DESIGN_FREEZE_WP9_A3.txt). Per user three streams (split, merge, none; the default WP8 generator, planted K = 3), seed = user:
  anchor   the K = 3 fit of [0, 90) (10 restarts, production selection and publish rule: `k_change.fit_recent_window`)
  refit    at t = 118, 134, 150 the recent window R = [t - 45, t) fitted at K' = 2, 3, 4 (the same function): BIC, accepted, labels, regime means
  ksel     the WP8 comparison: K selection (candidates 2, 3, 4) on the final window [0, 150) through `fit_hssm`, split and merge streams only
The fits are the PRODUCTION functions; the BIC margins, the decisions and the candidates are computed offline from these records by `a3_analyze.py` with the production `k_change` code."""
import argparse
import contextlib
import io
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wp9_fit_lib as W                                                   # noqa: E402

from backbone.hssm import k_change as KC                                  # noqa: E402

RANGES = {"dev": range(9221, 9233), "eval": range(9241, 9253)}
STREAMS = ("split", "merge", "none")
REFIT_ENDS = (118, 134, 150)
ANCHOR_END = 90
RECENT = 45


def jobs_for(users):
    out = []
    for u in users:
        for s in STREAMS:
            base = dict(user=u, stream=s, kchange=None if s == "none" else s, stratum="wp8", drift=0.0, fitseed=u, n_init=10, max_iter=60)
            out.append(dict(base, kind="anchor", start=0, end=ANCHOR_END, K=3))
            for t in REFIT_ENDS:
                for K in (2, 3, 4):
                    out.append(dict(base, kind="refit", start=t - RECENT, end=t, K=K))
            if s != "none":
                out.append(dict(base, kind="ksel", start=0, end=150, K=[2, 3, 4]))
    return out


def job_key(job):
    return json.dumps({k: job[k] for k in sorted(job) if k not in ("n_init", "max_iter")}, sort_keys=True)


def fit_job(job):
    logging.disable(logging.CRITICAL)
    t0 = time.time()
    st = W.stream_for(job)
    w = st.window(job["start"], job["end"])
    rec = dict(job=job, ok=False, scaler_mean=w["scaler_mean"].tolist(), scaler_std=w["scaler_std"].tolist(), labels_true=W.labels_to_str(w["labels"]))
    try:
        if job["kind"] == "ksel":
            from backbone.hssm.fitting import fit_hssm
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                r = fit_hssm(matrix=w["matrix"], candidate_ks=tuple(job["K"]), n_initializations=job["n_init"], max_iter=job["max_iter"], random_seed=job["fitseed"], user_id=f"u{job['user']}", dataset_hash=f"h{job['user']}")
            rec.update(ok=True, k_selected=int(r.k_selected), accepted=not bool(r.insights_withheld))
        else:
            f = KC.fit_recent_window(w["raw"], job["K"], seed=job["fitseed"], n_init=job["n_init"], max_iter=job["max_iter"])
            if f is None:
                rec.update(error="fit raised (no accepted K fit)")
            else:
                rec.update(ok=True, bic=f["bic"], accepted=bool(f["accepted"]), labels_fit=W.labels_to_str(f["labels"]), raw_means=np.asarray(f["raw_means"], dtype=float).tolist())
    except Exception as exc:                                              # noqa: BLE001
        rec.update(error=f"{type(exc).__name__}: {exc}"[:200])
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def run(jobs, out_path, workers=1):
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
    with open(out_path, "a") as fh, mp.Pool(workers) as pool:
        for rec in pool.imap_unordered(fit_job, todo):
            fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--range", required=True, choices=list(RANGES)); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.range == "eval" and not Path("tests/DESIGN_FREEZE_WP9_A3.txt").exists():
        sys.exit("the evaluation runs only after tests/DESIGN_FREEZE_WP9_A3.txt is committed")
    run(jobs_for(list(RANGES[a.range])), a.out, a.workers)


if __name__ == "__main__":
    main()
