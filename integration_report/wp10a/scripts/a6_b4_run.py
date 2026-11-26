"""WP10A A6 part 1: B4, does the posterior sd of the fast state (`m_t_std`) keep its coverage outside the favourable case? (design: tests/PREDECLARED_CRITERIA_WP10A.md Part 6, sections 16-19). usage:
  PYTHONPATH=. python3 integration_report/wp10a/scripts/a6_b4_run.py --out FILE.jsonl [--workers 2] [--limit N] [--seeds 10701,10709] [--pilot]
64 users, seeds 10701..10764 (16 per condition): C0 favourable control, C1 off-centre prior (+1 / +2 population sd), C2 atypical users (spread 3x), C3 misspecified fit (feature dropped, Dmax 8, Dmax 30, K too small). One fit per user: the production dual fit
`fit_hssm(model_type="dual_timescale", shared_prior=..., shared_prior_n0=5, certify_prior_anchored=True)` with the production defaults for restarts and iterations (no fast-test switch, production gate).
Stored per user: the fit verdict (withheld or not, reasons, EM state, certification), the coverage of the true fast state in 1 and 2 posterior sd on the held-out days (and on the training days, for
reference), the mean posterior sd. One line per user, resumable; every line is written as soon as the user is done (partial results). `--pilot`: one favourable user, prints CPU seconds."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
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
N_TRAIN_SEG, N_TEST_SEG = 6, 2
PRIOR_PATH = "wp7r_evidence/prior_main_wp5.json"
SD_INTERCEPT = 0.6                                                                       # PopulationSpec().sd_intercept: one population sd of the intercepts
NOMINAL1, NOMINAL2 = math.erf(1.0 / math.sqrt(2.0)), math.erf(2.0 / math.sqrt(2.0))
CONDITIONS = ("C0", "C1", "C2", "C3")
SEEDS = list(range(10701, 10765))
PER_CONDITION = 16
C3_VARIANTS = ("drop_feature",) * 4 + ("dmax_8",) * 4 + ("dmax_30",) * 4 + ("k_too_small",) * 4


# ------------------------------------------------------------------------------------------------------------------- the design (pure, tested)
def job_for(seed: int) -> dict:
    """The declared design for one seed: the condition, what is changed, and what the fit is told."""
    if seed not in SEEDS:
        raise ValueError(f"seed {seed} is not a B4 seed")
    i = seed - SEEDS[0]
    cond, j = CONDITIONS[i // PER_CONDITION], i % PER_CONDITION
    job = dict(seed=seed, condition=cond, variant="favourable", prior_shift_sd=0.0, spread=1.0, true_K=K, drop_feature=None, dmax=DMAX)
    if cond == "C1":
        job.update(variant="off_centre_1sd" if j < 8 else "off_centre_2sd", prior_shift_sd=1.0 if j < 8 else 2.0)
    elif cond == "C2":
        job.update(variant="atypical_3x", spread=3.0)
    elif cond == "C3":
        v = C3_VARIANTS[j]
        job.update(variant=v)
        if v == "drop_feature":
            job["drop_feature"] = F - 1
        elif v == "dmax_8":
            job["dmax"] = 8
        elif v == "dmax_30":
            job["dmax"] = 30
        elif v == "k_too_small":
            job["true_K"] = 3
    return job


def population_user_with_states(seed, n_train_segments, n_test_segments, spec, spread_factor=1.0):
    """`data.synthetic_population.generate_population_user` with the true fast states returned too (a copy of its code: the draws are the same, so X is bit-identical; tested)."""
    from data.synthetic_population import PopulationUser, draw_perturbation
    Kk, Ff, Ll = spec.n_regimes, spec.n_features, spec.latent_dim
    truth, eps = draw_perturbation(seed, spec, spread_factor)
    A, C, d = truth["A"], truth["C"], truth["d"]
    Q_scale = np.sqrt(np.diag(truth["q"] * np.eye(Ll)))
    R_scale = np.sqrt(np.full(Ff, truth["r"]))
    rng = np.random.default_rng(int(seed))

    def segments(n, current):
        regimes = []
        for _ in range(n):
            dur = int(np.clip(rng.lognormal(mean=np.log(truth["mean_dwell"]), sigma=0.3), 8, 30))
            regimes.extend([current] * dur)
            current = rng.choice([k for k in range(Kk) if k != current])
        return np.array(regimes, dtype=int), current

    p_tr, nxt = segments(n_train_segments, 0)
    T_tr = len(p_tr)
    m_tr = np.zeros((T_tr, Ll))
    m_curr = rng.normal(scale=0.2, size=Ll)
    m_tr[0] = m_curr
    for t in range(1, T_tr):
        m_curr = A @ m_curr + rng.normal(scale=Q_scale, size=Ll)
        m_tr[t] = m_curr
    X_tr = np.zeros((T_tr, Ff))
    for t in range(T_tr):
        X_tr[t] = C @ m_tr[t] + d[p_tr[t]] + rng.normal(scale=R_scale, size=Ff)
    p_te, _ = segments(n_test_segments, nxt)
    T_te = len(p_te)
    m_te = np.zeros((T_te, Ll))
    for t in range(T_te):
        m_curr = A @ m_curr + rng.normal(scale=Q_scale, size=Ll)
        m_te[t] = m_curr
    X_te = np.zeros((T_te, Ff))
    for t in range(T_te):
        X_te[t] = C @ m_te[t] + d[p_te[t]] + rng.normal(scale=R_scale, size=Ff)
    user = PopulationUser(seed=int(seed), spread_factor=float(spread_factor), X_train=X_tr, X_test=X_te, p_train=p_tr, p_test=p_te, truth=truth, eps=eps)
    return user, m_tr, m_te


def shifted_prior(prior, shift_sd: float):
    """The prior with every intercept `d_obs` moved by `shift_sd` population sds of the intercepts (the centre of the population moved; as the `shifted` control of WP9 B1)."""
    from backbone.hssm.shrinkage import SharedPrior
    p = prior.params()
    p["d_obs"] = np.asarray(p["d_obs"], dtype=float) + shift_sd * SD_INTERCEPT
    return SharedPrior.from_params(p, n_users=prior.n_users, n_excluded=prior.n_excluded)


def drop_feature_prior(prior, f: int):
    """The prior restricted to the features other than `f` (the fit sees one feature fewer)."""
    from backbone.hssm.shrinkage import SharedPrior
    p = prior.params()
    keep = [i for i in range(prior.F) if i != f]
    for k in ("C_obs", "d_obs", "R_obs"):
        p[k] = np.asarray(p[k], dtype=float)[:, keep]
    return SharedPrior.from_params(p, n_users=prior.n_users, n_excluded=prior.n_excluded)


def coverage_on(m_hat, std, true_m, true_p, rows) -> dict:
    """Share of the rows `rows` (a slice or index array) on which the true fast state, mapped into the fitted basis by the N17 per-true-regime affine map (in sample, all rows), lies within 1 and 2
    posterior sd, per latent dimension; plus the mean z^2."""
    from tests.test_dual_model_latent_calibration import per_regime_affine_prediction
    ref = per_regime_affine_prediction(true_m, m_hat, true_p, min_points=3 * (m_hat.shape[1] + 1))
    z = ((m_hat - ref) / std)[rows]
    return dict(c1=np.mean(np.abs(z) <= 1.0, axis=0).tolist(), c2=np.mean(np.abs(z) <= 2.0, axis=0).tolist(), z2=np.mean(z ** 2, axis=0).tolist(), n=int(z.shape[0]))


# ------------------------------------------------------------------------------------------------------------------- one user
def one(seed):
    logging.disable(logging.CRITICAL)
    from backbone.hssm.fitting import fit_hssm
    from backbone.hssm.shrinkage import SharedPrior
    from data.synthetic_population import PopulationSpec
    job = job_for(seed)
    rec = dict(job, ok=False, error=None)
    t0, c0 = time.time(), time.process_time()
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            spec = PopulationSpec(n_regimes=job["true_K"])
            user, m_tr, m_te = population_user_with_states(seed, N_TRAIN_SEG, N_TEST_SEG, spec, spread_factor=job["spread"])
            prior = SharedPrior.from_dict(json.load(open(ROOT / PRIOR_PATH)))
            if job["prior_shift_sd"]:
                prior = shifted_prior(prior, job["prior_shift_sd"])
            X_tr, X_te = user.X_train, user.X_test
            if job["drop_feature"] is not None:
                prior = drop_feature_prior(prior, job["drop_feature"])
                keep = [i for i in range(X_tr.shape[1]) if i != job["drop_feature"]]
                X_tr, X_te = X_tr[:, keep], X_te[:, keep]
            res = fit_hssm(matrix=X_tr, candidate_ks=(K,), random_seed=seed, max_duration=job["dmax"], model_type="dual_timescale", latent_dim=L,
                           shared_prior=prior, shared_prior_n0=N0, certify_prior_anchored=True)
            cert = dict(res.certification or {})
            rec.update(withheld=bool(res.insights_withheld), withheld_reason=res.withheld_reason, fit_state=str(getattr(res.fit_state, "value", res.fit_state)),
                       certification_state=cert.get("state"), certification_reasons=cert.get("reasons"), certified=bool(getattr(res, "certified_prior_anchored", False)),
                       n_train=int(len(X_tr)), n_test=int(len(X_te)))
            X_all = np.vstack([X_tr, X_te])
            true_m, true_p = np.vstack([m_tr, m_te]), np.concatenate([user.p_train, user.p_test])
            m_hat, std, _, _ = res.model.estimate_states(X_all)
            t_tr = len(X_tr)
            rec["heldout"] = coverage_on(m_hat, std, true_m, true_p, slice(t_tr, None))
            rec["train"] = coverage_on(m_hat, std, true_m, true_p, slice(0, t_tr))
            rec["std_mean"] = float(np.mean(std))
            rec["ok"] = True
    except Exception as exc:                                                                    # noqa: BLE001  recorded: an exception means no output for that user
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    rec["wall_s"], rec["cpu_s"] = round(time.time() - t0, 1), round(time.process_time() - c0, 1)
    return rec


def read_done(path):
    out = {}
    if os.path.exists(path):
        for ln in open(path):
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            if d.get("ok"):
                out[d["seed"]] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out"); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--seeds", default="")
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.pilot:
        rec = one(SEEDS[0])
        print(json.dumps({k: rec.get(k) for k in ("seed", "ok", "error", "withheld", "withheld_reason", "fit_state", "certification_state", "heldout", "cpu_s", "wall_s")}, indent=1))
        return
    if not a.out:
        sys.exit(__doc__)
    done = read_done(a.out)
    seeds = [int(s) for s in a.seeds.split(",") if s] or SEEDS
    todo = [s for s in seeds if s not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(todo)} users to run ({len(done)} done)", flush=True)
    if not todo:
        return
    t0 = time.time()
    with mp.Pool(min(a.workers, len(todo))) as pool:
        for rec in pool.imap_unordered(one, todo):
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())
            print(f"seed {rec['seed']} {rec['condition']}/{rec['variant']}: {'ok' if rec['ok'] else rec['error']}; withheld={rec.get('withheld')}; cpu {rec['cpu_s']}s, {time.time() - t0:.0f}s elapsed", flush=True)


if __name__ == "__main__":
    main()
