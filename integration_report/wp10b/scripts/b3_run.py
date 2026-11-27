"""WP10B B3: the dual predictive certificate on dual fits that are not prior-anchored (design: tests/PREDECLARED_CRITERIA_WP10B.md, the B3 part). usage:
  PYTHONPATH=. python3.12 integration_report/wp10b/scripts/b3_run.py --arm s0|breaker --range smoke|eval|ext --variant S0|ALL|... --out FILE.jsonl [--workers 2]
Arms.
  s0      : the S0 setting of B1 (identity-stream users, K = 3, F = 7, train days [0, 120) with the train-only scaler, test days [120, 150)), one dual fit with the named variant (`dual_lib.VARIANTS`), the baseline engine on the same
            rows, the certificate (`dual_cert_lib.certify_dual`, refits = the same fit on the rows before each origin, baseline refits = the production baseline fit), and the held-out one-step scores of dual, baseline and single Gaussian
            on the 30 test days (the per-day paired differences are stored). The certificate is asked only about eligible fits (EM state REJECTED_UPDATE or MAX_ITER_REACHED and no other withhold reason).
  breaker : wrong-engine controls on the N18 generator (`tests/utils/heldout_prediction.continued_synthetic`, here with 8 training segments so that the rows (about 120) match S0): the real engine and the two breakers
            (`stops` = M-step disabled, `B2` = A_fast x 0.1 after every M-step) are fitted and REFITTED with the same class; the certificate is applied whatever the EM state says (a breaker that stops learning converges at once),
            and the held-out score on the 2 continuation segments is recorded.
Seeds: smoke (spare, never scored) 11901..11904; eval s0 11421..11456 (12 per stratum: hard 11421..11432, medium 11433..11444, easy 11445..11456); ext s0 11457..11480 (only if the declared blind-count rule says so);
breaker eval 11501..11530 per breaker engine (30 fits each; the harmful ones, held-out gain over the Gaussian below 0, are the control) and 11531..11550 for the real engine. Refused for eval/ext unless the criteria file holds the dated B3 part. One line per job; resumable."""
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

CRITERIA = ROOT / "tests" / "PREDECLARED_CRITERIA_WP10B.md"
CRITERIA_MARK = "B3 (dual certification): declared before any B3 run"
STRATA = ("hard", "medium", "easy")
S0_USERS = {"smoke": [11901, 11902, 11903, 11904], "eval": list(range(11421, 11457)), "ext": list(range(11457, 11481))}
BREAKER_ENGINES = {"stops": list(range(11501, 11531)), "B2": list(range(11501, 11531)), "real": list(range(11531, 11551))}
BREAKER_SMOKE = {"stops": [11905], "B2": [11905], "real": [11905]}
N_TRAIN_SEGMENTS, N_TEST_SEGMENTS, BREAKER_ITER = 8, 2, 20


def stratum_of(user: int) -> str:
    """eval: 12 users per stratum in order; ext: 8 per stratum; smoke: hard, medium, easy, hard."""
    if user in S0_USERS["smoke"]:
        return STRATA[(user - 11901) % 3]
    if 11421 <= user <= 11456:
        return STRATA[(user - 11421) // 12]
    if 11457 <= user <= 11480:
        return STRATA[(user - 11457) // 8]
    raise ValueError(f"user {user} is not in a B3 range")


def paired_summary(d: np.ndarray) -> dict:
    """mean, HAC standard error (the certificate's) and n of a paired difference over observed days."""
    from backbone.hssm import cold_start_cert as C
    d = np.asarray(d, dtype=float)
    ok = bool(len(d) >= 3 and np.all(np.isfinite(d)))
    return dict(n=int(len(d)), mean=float(d.mean()) if len(d) and np.all(np.isfinite(d)) else None, se=C._f(C.hac_se(d)) if ok else None, finite=ok)


# --------------------------------------------------------------------------------------------------------------- s0 arm
def dual_refit(user: int, variant: str):
    """The refit procedure: `fit_dual_with_random_restarts` with the settings of the variant, the same base seed as the production fit of this user (user + K * 1000), no prior, the cold-start gate bypassed, the opt-in class switches of the variant set for the call."""
    import dual_lib as L
    from backbone.hssm import cold_start_cert as C
    from backbone.hssm.dual_model import DualTimescaleHSSM
    from backbone.hssm.fitting import fit_dual_with_random_restarts
    kw = dict(L.S0)
    kw.update(L.VARIANTS[variant])
    flags = dict(kw.pop("_flags", {}))

    def refit(X_train: np.ndarray):
        old = {k: getattr(DualTimescaleHSSM, k) for k in flags}
        try:
            for k, v in flags.items():
                setattr(DualTimescaleHSSM, k, v)
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                model, _ = fit_dual_with_random_restarts(X_train, n_regimes=L.K_STUDY, n_features=X_train.shape[1], latent_dim=kw["latent_dim"], n_init=kw["n_initializations"], n_iter=kw["max_iter"],
                                                         max_duration=kw["max_duration"], base_seed=user + L.K_STUDY * 1000, bypass_init_gate=True, shared_prior=None, shared_prior_n0=None)
        finally:
            for k, v in old.items():
                setattr(DualTimescaleHSSM, k, v)
        return C.DualAdapter(model)
    return refit


def s0_job(job: dict) -> dict:
    logging.disable(logging.CRITICAL)
    import dual_cert_lib as DC
    import dual_lib as L
    from backbone.hssm import cold_start_cert as C
    t0 = time.time()
    user, stratum, variant = job["user"], job["stratum"], job["variant"]
    rec = dict(job=job, ok=False)
    d = L.user_data(user, stratum)
    X, T_tr, dm = d["X"], d["T_train"], d["day_missing"]
    Xtr = X[:T_tr]
    keep = ~dm[T_tr:]
    rec.update(T_train=int(T_tr), n_test_days=int(keep.sum()), n_train_missing_days=int(dm[:T_tr].sum()))
    try:
        t1 = time.time()
        res = L.fit_dual(Xtr, user, variant)
        model = res.model
        rec["dual"] = L.dual_record(res, model)
        rec["dual"]["fit_seconds"] = round(time.time() - t1, 2)
        lp = L.dual_increments(model, X)
        g = L.gaussian_increments(Xtr, X[T_tr:])
        rec["test_g_diff"] = [float(x) for x in (lp[T_tr:][keep] - g[keep])]
        rec["dual"]["test_mean"] = float(np.mean(lp[T_tr:][keep]))
        em_state = rec["dual"]["fit_state"]
        other = list(res._withheld_reasons(ignore_non_convergence=True))
        rec["eligibility"] = dict(em_state=em_state, other_reasons=[str(x)[:120] for x in other])
        rec["eligible"] = bool(em_state in C.ELIGIBLE_EM_STATES and not other)
        try:
            b = L.fit_baseline(Xtr, user)
            lb = L.baseline_increments(b.model, X)
            rec["baseline"] = dict(fit_state=str(b.fit_state.value), publishable=bool(b.withheld_reason is None), test_mean=float(np.mean(lb[T_tr:][keep])))
            rec["test_b_diff"] = [float(x) for x in (lp[T_tr:][keep] - lb[T_tr:][keep])]
        except Exception as exc:                                                                    # noqa: BLE001
            rec["baseline"] = dict(error=f"{type(exc).__name__}: {str(exc)[:160]}")
        rec["ok"] = True
        if rec["eligible"]:
            t2 = time.time()
            cert = DC.certify_dual(Xtr, C.DualAdapter(model), dual_refit(user, variant), DC.make_baseline_refit(user), eligibility=rec["eligibility"])
            cert["cert_seconds"] = round(time.time() - t2, 1)
            rec["certificate"] = cert
        else:
            rec["certificate"] = DC.certify_dual(Xtr, C.DualAdapter(model), lambda Xt: (_ for _ in ()).throw(RuntimeError("not asked")), None, eligibility=rec["eligibility"])
    except Exception as exc:                                                                        # noqa: BLE001  a failed job is data too
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


# --------------------------------------------------------------------------------------------------------------- breaker arm
def breaker_job(job: dict) -> dict:
    logging.disable(logging.CRITICAL)
    import dual_cert_lib as DC
    from backbone.hssm import cold_start_cert as C
    from tests.utils import heldout_prediction as HP
    t0 = time.time()
    engine, seed = job["engine"], job["seed"]
    rec = dict(job=job, ok=False)
    try:
        data = HP.continued_synthetic(seed, N_TRAIN_SEGMENTS, N_TEST_SEGMENTS)
        cls = HP.ENGINE_CLASSES[engine]
        Xtr, Xte = data.X_train, data.X_test

        def make(X_train):
            m = cls(n_regimes=HP.K, n_features=HP.F, latent_dim=HP.L, max_duration=HP.MAX_DURATION, seed=seed)
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                m.fit(X_train, n_iter=BREAKER_ITER)
            return m
        model = make(Xtr)
        full = np.vstack([Xtr, Xte])
        ad = C.DualAdapter(model)
        lp = ad.predictive_increments(full)
        g = C.gaussian_heldout(Xtr, Xte)
        rec["test_g_diff"] = [float(x) for x in (lp[len(Xtr):] - g)]
        rec.update(T_train=int(len(Xtr)), n_test_days=int(len(Xte)), fit_state=str(model.fit_state_.value), fit_degraded=bool(model.fit_degraded_))
        el = dict(apply_to_any_state=True, em_state=rec["fit_state"], other_reasons=[])
        rec["certificate"] = DC.certify_dual(Xtr, ad, lambda Xt: C.DualAdapter(make(Xt)), None, eligibility=el)
        rec["ok"] = True
    except Exception as exc:                                                                        # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


# --------------------------------------------------------------------------------------------------------------- driver
def job_key(job: dict) -> str:
    return json.dumps(job, sort_keys=True)


def run(jobs: list, fn, out_path: str, workers: int) -> None:
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(job_key(json.loads(line)["job"]))
            except Exception:                                                                       # noqa: BLE001
                pass
    todo = [j for j in jobs if job_key(j) not in done]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run", flush=True)
    t0 = time.time()
    if not todo:
        return
    with open(out_path, "a") as fh, mp.Pool(min(workers, len(todo))) as pool:
        for i, rec in enumerate(pool.imap_unordered(fn, todo), 1):
            fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())
            c = (rec.get("certificate") or {})
            print(f"{i} / {len(todo)} {rec['job']} ok={rec.get('ok')} cert={c.get('state')} {c.get('reasons')} {rec.get('seconds')}s {time.time() - t0:.0f}s elapsed", flush=True)


def build_jobs(arm: str, rng: str, variant: str) -> list:
    if arm == "s0":
        return [dict(user=u, stratum=stratum_of(u), variant=variant) for u in S0_USERS[rng]]
    table = BREAKER_SMOKE if rng == "smoke" else BREAKER_ENGINES
    return [dict(engine=e, seed=s) for e, seeds in table.items() for s in seeds]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("s0", "breaker")); ap.add_argument("--range", required=True, choices=("smoke", "eval", "ext"))
    ap.add_argument("--variant", default="S0"); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.range != "smoke" and (not CRITERIA.exists() or CRITERIA_MARK not in CRITERIA.read_text()):
        sys.exit("the scored B3 runs start only after the dated B3 part is committed in tests/PREDECLARED_CRITERIA_WP10B.md")
    if a.arm == "breaker" and a.range == "ext":
        sys.exit("the breaker arm has no extension")
    import dual_lib as L
    if a.variant not in L.VARIANTS:
        sys.exit(f"unknown variant {a.variant!r}")
    run(build_jobs(a.arm, a.range, a.variant), s0_job if a.arm == "s0" else breaker_job, a.out, a.workers)


if __name__ == "__main__":
    main()
