"""WP10B C1(2): a MILDLY wrong prior, and does `certify_prior_anchored` refuse it? (design: tests/PREDECLARED_CRITERIA_WP10B.md section 4). usage:
  PYTHONPATH=. python3 integration_report/wp10b/scripts/c1_mild_run.py --range dev  --candidates M1,M2,M3 --out FILE.jsonl [--workers 2]
  PYTHONPATH=. python3 integration_report/wp10b/scripts/c1_mild_run.py --range eval --out FILE.jsonl [--workers 2]     (refused without tests/DESIGN_FREEZE_WP10B_C1.txt: first line {"candidate": "M1"|"M2"|"M3"})
  PYTHONPATH=. python3 integration_report/wp10b/scripts/c1_mild_run.py --range ext  --out FILE.jsonl --eval-fits EVAL.jsonl   (the pre-declared extension; refused unless the blind count of eligible harmful fits of EVAL is below 12)
The WP10A A6 study (`wp10a/scripts/a6_harm_run.py`): K = 2, F = 4, L = 2, lengths 30 / 60 / 120 days, 200 iterations, 1 restart, fast-test switch and cold-start gate off for the study only, n0 = 5, public path
`fit_hssm(..., certify_prior_anchored=True)`. Candidates are built from the true prior `wp7r_evidence/prior_main_wp5.json`: M1 every intercept + 8, M2 + 12, M3 the observation model (d_obs, C_obs, R_obs) of regime 0 and
regime 1 exchanged. dev = seeds 11001..11012; eval = 11021..11060 with arms M (the frozen candidate) and T (the true prior); ext = 11061..11080. Per (seed, length) also the single Gaussian. One line per (seed, length, arm); resumable."""
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
sys.path.insert(0, str(HERE))

K, F, L, DMAX, N0 = 2, 4, 2, 25, 5.0
LENGTHS = (30, 60, 120)
SEGMENTS = {30: 2, 60: 4, 120: 8}
MAX_ITER = 200
PRIOR_PATH = "wp7r_evidence/prior_main_wp5.json"
CANDIDATE_ORDER = ("M1", "M2", "M3")
H3_SHIFT = 16.0
RANGES = {"dev": list(range(11001, 11013)), "eval": list(range(11021, 11061)), "ext": list(range(11061, 11081))}
FREEZE = Path("tests/DESIGN_FREEZE_WP10B_C1.txt")
ELIGIBLE_EM_STATES = ("REJECTED_UPDATE", "MAX_ITER_REACHED")


def mild_dict(d: dict, cand: str) -> dict:
    """The candidate prior as a parameter dict (pure; `d` is the true prior's JSON dict and is not modified)."""
    out = {k: (np.array(v, dtype=float, copy=True) if isinstance(v, list) else v) for k, v in d.items()}
    if cand == "M1":
        out["d_obs"] = out["d_obs"] + 8.0
    elif cand == "M2":
        out["d_obs"] = out["d_obs"] + 12.0
    elif cand == "M3":
        for k in ("d_obs", "C_obs", "R_obs"):
            out[k] = out[k][::-1].copy()
    else:
        raise ValueError(f"unknown candidate {cand!r}")
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in out.items()}


def dobs_norm(d: dict, cand: str) -> float:
    """Frobenius norm of the change of d_obs (the declared mildness measure; H3, every intercept + 16, has 16 * sqrt(K * F))."""
    new = np.asarray(mild_dict(d, cand)["d_obs"], dtype=float)
    return float(np.linalg.norm(new - np.asarray(d["d_obs"], dtype=float)))


def h3_norm(d: dict) -> float:
    return float(H3_SHIFT * np.sqrt(np.asarray(d["d_obs"]).size))


def true_dict(prior_path) -> dict:
    return json.load(open(prior_path))


def prior_of(prior_path, cand: str | None):
    from backbone.hssm.shrinkage import SharedPrior
    d = true_dict(prior_path)
    return SharedPrior.from_dict(d if cand is None else mild_dict(d, cand))


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
    seed, length, arms, prior_path, cands = job
    logging.disable(logging.CRITICAL)
    from data.synthetic_population import PopulationSpec, generate_population_user
    from tests.utils import n18v2 as N
    priors = {a: prior_of(prior_path, None if a == "T" else cands[a]) for a in arms}
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
    ap.add_argument("--candidates", default=""); ap.add_argument("--prior", default=PRIOR_PATH); ap.add_argument("--eval-fits", default="")
    a = ap.parse_args()
    if a.range == "dev":
        cands = [c for c in a.candidates.split(",") if c]
        if not cands or any(c not in CANDIDATE_ORDER for c in cands):
            sys.exit("--candidates must name M1, M2 and/or M3")
        arms, cmap = list(cands), {c: c for c in cands}
    else:
        if not FREEZE.exists():
            sys.exit("the evaluation runs only after tests/DESIGN_FREEZE_WP10B_C1.txt is committed")
        cand = json.loads(FREEZE.read_text().splitlines()[0])["candidate"]
        if cand not in CANDIDATE_ORDER:
            sys.exit(f"unknown candidate {cand!r} in the freeze file")
        arms, cmap = ["M", "T"], {"M": cand}
        if a.range == "ext":
            import c1_mild_analyze as AN
            if not a.eval_fits or not os.path.exists(a.eval_fits):
                sys.exit("--eval-fits (the stored evaluation file) is required for the extension")
            n = AN.eligible_harmful_count(AN.read_jsonl(a.eval_fits))
            if not AN.needs_extension(n):
                sys.exit(f"the extension is refused: the blind count of eligible harmful fits is {n} (the extension is allowed only below {AN.REQUIRED_EVAL})")
    need = {"gauss"} | set(arms)
    have = read_have(a.out)
    jobs = [(s, ln, arms, a.prior, cmap) for s in RANGES[a.range] for ln in LENGTHS if not need <= have.get((s, ln), set())]
    d = true_dict(a.prior)
    print(f"{len(jobs)} jobs ({len(RANGES[a.range]) * len(LENGTHS) - len(jobs)} complete), arms {arms}; norm of the d_obs change: " + ", ".join(f"{c} {dobs_norm(d, c):.2f}" for c in CANDIDATE_ORDER) + f" (H3 {h3_norm(d):.2f})", flush=True)
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
