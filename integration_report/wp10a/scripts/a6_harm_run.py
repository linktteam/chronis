"""WP10A A6 part 2: a prior that is provably harmful, and does `certify_prior_anchored` refuse it? (design: tests/PREDECLARED_CRITERIA_WP10A.md Part 6 section 18 and Part 6b). usage:
  PYTHONPATH=. python3 integration_report/wp10a/scripts/a6_harm_run.py --range dev  --candidates H1[,H2,H3] --out FILE.jsonl [--workers 2]
  PYTHONPATH=. python3 integration_report/wp10a/scripts/a6_harm_run.py --range eval --out FILE.jsonl [--workers 2]      (refused without tests/DESIGN_FREEZE_WP10A_A6.txt: first line {"candidate": "H1"|"H2"|"H3"})
The study of WP9 B1 (`wp9/scripts/b1_run.py`): K = 2, F = 4, L = 2, lengths 30 / 60 / 120 days (2 / 4 / 8 segments), 200 iterations, 1 restart, the fast-test switch and the cold-start gate off for the study only,
n0 = 5, the public path `fit_hssm(..., certify_prior_anchored=True)`. dev = seeds 9631..9642 (the WP9 development users), candidates H1/H2/H3 = the true prior with every intercept + 4 / + 8 / + 16;
eval = seeds 10771..10794: arm H (the frozen candidate), arm T (the true prior). Per (seed, length) also the single Gaussian. One line per (seed, length, arm), resumable."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

K, F, L, DMAX, N0 = 2, 4, 2, 25, 5.0
LENGTHS = (30, 60, 120)
SEGMENTS = {30: 2, 60: 4, 120: 8}
MAX_ITER = 200
PRIOR_PATH = "wp7r_evidence/prior_main_wp5.json"
CANDIDATES = {"H1": 4.0, "H2": 8.0, "H3": 16.0}                                           # added to every intercept d_obs of the true prior
RANGES = {"dev": list(range(9631, 9643)), "eval": list(range(10771, 10795))}
FREEZE = Path("tests/DESIGN_FREEZE_WP10A_A6.txt")
ELIGIBLE_EM_STATES = ("REJECTED_UPDATE", "MAX_ITER_REACHED")


def harmful_prior(prior_path, shift: float):
    from backbone.hssm.shrinkage import SharedPrior
    d = json.load(open(prior_path))
    d["d_obs"] = (np.asarray(d["d_obs"], dtype=float) + float(shift)).tolist()
    return SharedPrior.from_dict(d)


def true_prior(prior_path):
    from backbone.hssm.shrinkage import SharedPrior
    return SharedPrior.from_dict(json.load(open(prior_path)))


def arms_for(range_name: str, candidates=None, frozen=None) -> list:
    """The arms of a range: dev -> the named candidates; eval -> H (the frozen candidate) and T (the true prior)."""
    if range_name == "dev":
        return list(candidates)
    return ["H", "T"]


def fit_arm(arm, user, seed, priors):
    from backbone.hssm.config import ColdStartConfig
    from backbone.hssm.fitting import fit_hssm
    from tests.utils import n18v2 as N
    t0 = time.time()
    rec = dict(arm=arm, error=None)
    try:
        kw = dict(matrix=user.X_train, candidate_ks=(K,), n_initializations=1, max_iter=MAX_ITER, random_seed=seed - 2000, allow_fast_test_fit=True, config=ColdStartConfig(min_present_sessions=1),
                  max_duration=DMAX, model_type="dual_timescale", latent_dim=L, shared_prior=priors[arm], shared_prior_n0=N0, certify_prior_anchored=True)
        res = fit_hssm(**kw)
        cert = dict(res.certification or {})
        saved = res.certification
        res.certification = {}
        em_reason = res.withheld_reason                                                   # the EM-rule verdict, certification off
        res.certification = saved
        em_state = str(getattr(res.fit_state, "value", res.fit_state))
        other = list(res._withheld_reasons(ignore_non_convergence=True))
        rec.update(withheld_em=em_reason is not None, em_reason=em_reason, published_after=res.withheld_reason is None, certification=cert, fit_state=em_state,
                   other_reasons=[str(x)[:80] for x in other], eligible=bool(em_state in ELIGIBLE_EM_STATES and not other), fit_degraded=bool(res.fit_degraded))
        rec.update(N.score_model(res.model, user))
    except Exception as exc:                                                              # noqa: BLE001  recorded: an exception means no publishable output
        rec.update(mean=float("nan"), finite=False, error=f"{type(exc).__name__}: {str(exc)[:300]}", withheld_em=True, published_after=False, eligible=False)
    rec["wall_s"] = round(time.time() - t0, 2)
    return rec


def one(job):
    seed, length, arms, prior_path, shifts = job
    logging.disable(logging.CRITICAL)
    from data.synthetic_population import PopulationSpec, generate_population_user
    from tests.utils import n18v2 as N
    priors = {}
    for a in arms:
        priors[a] = true_prior(prior_path) if a == "T" else harmful_prior(prior_path, shifts[a])
    out = []
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        user = generate_population_user(seed, SEGMENTS[length], 2, PopulationSpec())
        base = dict(seed=int(seed), length=length, n_segments=SEGMENTS[length], T_train=int(len(user.X_train)), T_test=int(len(user.X_test)), max_iter=MAX_ITER)
        g = dict(arm="gauss", **N.gauss_scores(user)); g.update(base); out.append(g)
        for arm in arms:
            r = fit_arm(arm, user, seed, priors); r.update(base); out.append(r)
    return job, out


def read_have(path):
    have = {}
    if os.path.exists(path):
        for ln in open(path):
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            have.setdefault((d["seed"], d["length"]), set()).add(d["arm"])
    return have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", required=True, choices=list(RANGES)); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--candidates", default=""); ap.add_argument("--prior", default=PRIOR_PATH)
    a = ap.parse_args()
    shifts = dict(CANDIDATES)
    if a.range == "eval":
        if not FREEZE.exists():
            sys.exit("the evaluation runs only after tests/DESIGN_FREEZE_WP10A_A6.txt is committed")
        cand = json.loads(FREEZE.read_text().splitlines()[0])["candidate"]
        if cand not in CANDIDATES:
            sys.exit(f"unknown candidate {cand!r} in the freeze file")
        shifts = dict(H=CANDIDATES[cand])
        arms = arms_for("eval")
    else:
        cands = [c for c in a.candidates.split(",") if c]
        if not cands or any(c not in CANDIDATES for c in cands):
            sys.exit("--candidates must name H1, H2 and/or H3")
        arms = arms_for("dev", cands)
    need = {"gauss"} | set(arms)
    have = read_have(a.out)
    jobs = [(s, ln, arms, a.prior, shifts) for s in RANGES[a.range] for ln in LENGTHS if not need <= have.get((s, ln), set())]
    print(f"{len(jobs)} jobs ({len(RANGES[a.range]) * len(LENGTHS) - len(jobs)} complete), arms {arms}", flush=True)
    if not jobs:
        return
    t0 = time.time()
    with mp.Pool(min(a.workers, len(jobs))) as pool:
        for job, recs in pool.imap_unordered(one, jobs):
            with open(a.out, "a") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
                fh.flush(); os.fsync(fh.fileno())
            c = sum(1 for r in recs if (r.get("certification") or {}).get("state") == "CERTIFIED_PRIOR_ANCHORED")
            print(f"seed {job[0]} length {job[1]} done ({sum(r.get('wall_s', 0) for r in recs):.0f}s, {time.time() - t0:.0f}s elapsed, {c} certified)", flush=True)


if __name__ == "__main__":
    main()
