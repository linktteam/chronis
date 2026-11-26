"""WP9 B1: cold-start certification study (design: tests/PREDECLARED_CRITERIA_WP9.md section 16). usage:
  PYTHONPATH=. python3 integration_report/wp9/scripts/b1_run.py --range dev|eval --out FILE.jsonl [--workers 2]
dev = seeds 9631..9642 (mine, unreserved), eval = seeds 9601..9624 (refused without tests/DESIGN_FREEZE_WP9_B1.txt). Job = (seed, length): the WP7-R item 3 stage B settings (200 iterations,
gate off for the study only), arms shrink5 (true prior), scrambled and shifted (control priors), personal (control, no prior), plus the shared-only and single-Gaussian scores. shrink5, scrambled and
shifted go through the PUBLIC path `fit_hssm(..., certify_prior_anchored=True)`; the personal control has no prior, so `cold_start_cert.certify_prior_anchored` is applied to it directly (a test of the
test: production never certifies a fit without a prior). The EM-rule verdict (`withheld_em`) is read with the certification switched off. One jsonl line per (seed, length, arm), resumable."""
import argparse, contextlib, io, json, logging, multiprocessing as mp, os, sys, time, warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp7r" / "scripts"))
import item3_shrinkage_study as I3                                       # noqa: E402

RANGES = {"dev": list(range(9631, 9643)), "eval": list(range(9601, 9625))}
LENGTHS, SEGMENTS = I3.LENGTHS, I3.SEGMENTS
K, F, L, DMAX, N0 = I3.K, I3.F, I3.L, I3.DMAX, I3.N0
MAX_ITER = 200
ARMS = ("shrink5", "scrambled", "shifted", "personal")


def control_priors(prior_path):
    """True prior, scrambled prior (feature axis of d_obs, C_obs, R_obs rolled by one, duration parameters of the two regimes swapped), shifted prior (every d_obs + 1.5)."""
    from backbone.hssm.shrinkage import SharedPrior
    d = json.load(open(prior_path))
    def variant(fn):
        e = json.loads(json.dumps(d)); fn(e)
        return SharedPrior.from_dict(e)
    def scramble(e):
        for k in ("d_obs", "C_obs", "R_obs"):
            e[k] = np.roll(np.asarray(e[k], dtype=float), 1, axis=1).tolist()
        e["dur_mu"] = list(reversed(e["dur_mu"])); e["dur_sigma"] = list(reversed(e["dur_sigma"]))
    def shift(e):
        e["d_obs"] = (np.asarray(e["d_obs"], dtype=float) + 1.5).tolist()
    return dict(true=SharedPrior.from_dict(d), scrambled=variant(scramble), shifted=variant(shift))


def _arm(arm, user, seed, priors):
    from backbone.hssm import cold_start_cert as C
    from backbone.hssm.config import ColdStartConfig
    from backbone.hssm.fitting import fit_hssm
    from tests.utils import n18v2 as N
    t0 = time.time()
    rec = dict(arm=arm, error=None, withheld_em=None, em_reason=None, certification=None)
    try:
        prior = {"shrink5": priors["true"], "scrambled": priors["scrambled"], "shifted": priors["shifted"], "personal": None}[arm]
        kw = dict(matrix=user.X_train, candidate_ks=(2,), n_initializations=1, max_iter=MAX_ITER, random_seed=seed - 2000, allow_fast_test_fit=True, config=ColdStartConfig(min_present_sessions=1),
                  max_duration=DMAX, model_type="dual_timescale", latent_dim=L)
        if prior is not None:
            kw.update(shared_prior=prior, shared_prior_n0=N0, certify_prior_anchored=True)
        res = fit_hssm(**kw)
        cert = dict(res.certification or {})
        if prior is None:                                                  # control: the same function, applied to a fit that has no prior
            em_state = str(res.fit_state.value)
            elig = dict(used_shared_prior=True, n0=1.0, em_state=em_state, other_reasons=res._withheld_reasons(ignore_non_convergence=True))
            refit = C.make_refit_fn(n_regimes=K, n_features=F, latent_dim=L, n_init=1, max_iter=MAX_ITER, max_duration=DMAX, base_seed=(seed - 2000) + K * 1000, shared_prior=None, shared_prior_n0=None)
            cert = C.certify_prior_anchored(user.X_train, C.DualAdapter(res.model), refit, eligibility=elig)
            res.certification = cert
            res.used_shared_prior = True                                   # only so that the property below reads the control's verdict; the object is not kept
        published = res.withheld_reason is None                            # certified or EM-publishable
        saved = res.certification
        res.certification = {}
        em_reason = res.withheld_reason                                    # the EM-rule verdict, certification off
        res.certification = saved
        rec.update(withheld_em=em_reason is not None, em_reason=em_reason, published_after=bool(published), certification=cert, fit_state=str(res.fit_state.value),
                   n_iter_run=int(getattr(res.model, "n_iter_", -1)), fit_degraded=bool(res.fit_degraded))
        rec.update(N.score_model(res.model, user))
    except Exception as exc:                                               # noqa: BLE001  recorded: an exception means no publishable output
        rec.update(mean=float("nan"), finite=False, error=f"{type(exc).__name__}: {str(exc)[:300]}", withheld_em=True, published_after=False)
    rec["wall_s"] = round(time.time() - t0, 2)
    return rec


def one(job):
    seed, length, prior_path = job
    from tests.utils import n18v2 as N
    from data.synthetic_population import PopulationSpec, generate_population_user
    from backbone.hssm.shrinkage import ShrinkageDualHSSM
    logging.disable(logging.CRITICAL)
    priors = control_priors(prior_path)
    out = []
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        user = generate_population_user(seed, SEGMENTS[length], 2, PopulationSpec())
        base = dict(seed=int(seed), length=length, n_segments=SEGMENTS[length], T_train=int(len(user.X_train)), T_test=int(len(user.X_test)), max_iter=MAX_ITER)
        g = dict(arm="gauss", **N.gauss_scores(user)); g.update(base); out.append(g)
        for name, pr in priors.items():                                    # shared-only scores (not fits): validity of the control priors
            m = ShrinkageDualHSSM(n_regimes=K, n_features=F, latent_dim=L, max_duration=DMAX, seed=seed, shared_prior=pr, n0=0.0).become_shared_model()
            r = dict(arm=f"shared_{name}", **N.score_model(m, user)); r.update(base); out.append(r)
        for arm in ARMS:
            r = _arm(arm, user, seed, priors); r.update(base); out.append(r)
    return job, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--range", required=True, choices=list(RANGES)); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--prior", default=I3.PRIOR_PATH)
    a = ap.parse_args()
    if a.range == "eval" and not Path("tests/DESIGN_FREEZE_WP9_B1.txt").exists():
        sys.exit("the evaluation runs only after tests/DESIGN_FREEZE_WP9_B1.txt is committed")
    need = {"gauss", "shared_true", "shared_scrambled", "shared_shifted"} | set(ARMS)
    have = {}
    if os.path.exists(a.out):
        for ln in open(a.out):
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            have.setdefault((d["seed"], d["length"]), set()).add(d["arm"])
    jobs = [(s, ln, a.prior) for s in RANGES[a.range] for ln in LENGTHS if not need <= have.get((s, ln), set())]
    print(f"{len(jobs)} jobs ({len(RANGES[a.range]) * len(LENGTHS) - len(jobs)} complete)", flush=True)
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
