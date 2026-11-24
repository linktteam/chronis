"""WP7-R item 3: shrinkage under fail-closed (design: tests/PREDECLARED_CRITERIA_WP7R.md section 3; committed before this script was run on any scored seed)

usage: PYTHONPATH=. python3 integration_report/wp7r/scripts/item3_shrinkage_study.py --stage A|B --seeds 7901:7925 --out wp7r_evidence/item3_stageA.jsonl [--workers 2]
       Stage A: max_iter 20, arms gauss shared personal shrink5.  Stage B: max_iter 200, arms personal shrink5.
       One JSON line per (seed, length, arm), appended when the seed is finished; seeds already complete in the file are skipped (resumable).

`personal` and `shrink5` go through the PUBLIC path fit_hssm(model_type="dual_timescale", ...), so `withheld` is the real HSSMResult.insights_withheld, read BEFORE
scoring. The cold-start gate is switched off for this study only (ColdStartConfig(min_present_sessions=1)); the default is untouched.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import multiprocessing as mp
import os
import time
import warnings

import numpy as np

K, F, L, DMAX, N0 = 2, 4, 2, 25, 5.0
LENGTHS = (30, 60, 120)
SEGMENTS = {30: 2, 60: 4, 120: 8}
ARMS = {"A": ("gauss", "shared", "personal", "shrink5"), "B": ("personal", "shrink5")}
MAX_ITER = {"A": 20, "B": 200}
PRIOR_PATH = "wp7r_evidence/prior_main_wp5.json"


def _parse_seeds(txt):
    out = []
    for part in txt.split(","):
        if ":" in part:
            a, b = part.split(":")
            out += list(range(int(a), int(b)))
        elif part:
            out.append(int(part))
    return out


def _run_arm(arm, user, seed, max_iter, prior):
    from backbone.hssm.config import ColdStartConfig
    from backbone.hssm.fitting import fit_hssm
    from backbone.hssm.shrinkage import ShrinkageDualHSSM
    from tests.utils import n18v2 as N

    t0 = time.time()
    rec = dict(arm=arm, withheld=False, withheld_reason=None, fit_state=None, error=None)
    try:
        if arm == "gauss":
            rec.update(N.gauss_scores(user))
            rec["fit_state"] = "NOT_A_FIT"
        elif arm == "shared":
            m = ShrinkageDualHSSM(n_regimes=K, n_features=F, latent_dim=L, max_duration=DMAX, seed=seed, shared_prior=prior, n0=0.0).become_shared_model()
            rec.update(N.score_model(m, user))
            rec.update(N.fit_info(m))
            rec["not_a_fit"] = True
        else:
            kw = dict(matrix=user.X_train, candidate_ks=(2,), n_initializations=1, max_iter=max_iter, random_seed=seed - 2000, allow_fast_test_fit=True,
                      config=ColdStartConfig(min_present_sessions=1), max_duration=DMAX, model_type="dual_timescale", latent_dim=L)
            if arm == "shrink5":
                kw.update(shared_prior=prior, shared_prior_n0=N0)
            res = fit_hssm(**kw)
            rec["withheld"] = bool(res.insights_withheld)                      # read BEFORE scoring: scoring can add fallbacks to the model counter
            rec["withheld_reason"] = res.withheld_reason
            rec["fit_state"] = str(res.fit_state.value if hasattr(res.fit_state, "value") else res.fit_state)
            meta = res.convergence_metadata or {}
            rec.update(n_iter_run=int(getattr(res.model, "n_iter_", -1)), criterion=meta.get("convergence_criterion"), window_gain=meta.get("window_gain_at_cap"),
                       fit_degraded=bool(res.fit_degraded), used_shared_prior=bool(res.used_shared_prior), shared_prior_n0=res.shared_prior_n0)
            rec.update(N.score_model(res.model, user))
    except Exception as exc:  # noqa: BLE001  recorded, never swallowed: an exception means no publishable output for that fit
        rec.update(mean=float("nan"), finite=False, error=f"{type(exc).__name__}: {str(exc)[:300]}", withheld=(arm in ("personal", "shrink5")),
                   fit_state=rec["fit_state"] or "EXCEPTION")
    rec["wall_s"] = round(time.time() - t0, 2)
    return rec


def one(job):
    stage, seed, prior_path = job
    from backbone.hssm.shrinkage import SharedPrior
    from data.synthetic_population import PopulationSpec, generate_population_user

    logging.disable(logging.CRITICAL)
    prior = SharedPrior.from_dict(json.load(open(prior_path)))
    out = []
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        for length in LENGTHS:
            user = generate_population_user(seed, SEGMENTS[length], 2, PopulationSpec())
            for arm in ARMS[stage]:
                rec = _run_arm(arm, user, seed, MAX_ITER[stage], prior)
                rec.update(stage=stage, seed=int(seed), length=length, n_segments=SEGMENTS[length], T_train=int(len(user.X_train)), T_test=int(len(user.X_test)),
                           max_iter=MAX_ITER[stage])
                out.append(rec)
    return seed, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "B"])
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--prior", default=PRIOR_PATH)
    a = ap.parse_args()
    seeds = _parse_seeds(a.seeds)
    need = {(length, arm) for length in LENGTHS for arm in ARMS[a.stage]}
    have: dict = {}
    if os.path.exists(a.out):
        for ln in open(a.out):
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            have.setdefault(d["seed"], set()).add((d["length"], d["arm"]))
    jobs = [(a.stage, s, a.prior) for s in seeds if not need <= have.get(s, set())]
    print(f"stage {a.stage}: {len(jobs)} seeds to run ({len(seeds) - len(jobs)} complete)", flush=True)
    if not jobs:
        return
    t0 = time.time()
    with mp.Pool(min(a.workers, len(jobs))) as pool:
        for seed, recs in pool.imap_unordered(one, jobs):
            with open(a.out, "a") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
            n_w = sum(1 for r in recs if r["arm"] in ("personal", "shrink5") and r["withheld"])
            print(f"seed {seed} done ({sum(r['wall_s'] for r in recs):.0f}s of fits, {time.time() - t0:.0f}s elapsed, {n_w} withheld fits)", flush=True)


if __name__ == "__main__":
    main()
