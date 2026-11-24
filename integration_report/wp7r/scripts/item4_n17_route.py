"""WP7-R item 4: N17, a practical route for m_t_std (design: tests/PREDECLARED_CRITERIA_WP7R.md section 4; committed before this script ran on any scored seed)

usage (from the repo root):
  PYTHONPATH=. python3 integration_report/wp7r/scripts/item4_n17_route.py coverage  --seeds 9101:9113 --out wp7r_evidence/item4_coverage.jsonl  [--workers 2]
  PYTHONPATH=. python3 integration_report/wp7r/scripts/item4_n17_route.py bootstrap --seeds 9101:9107 --out wp7r_evidence/item4_bootstrap.jsonl [--workers 2] [--B 30]
  PYTHONPATH=. python3 integration_report/wp7r/scripts/item4_n17_route.py cost      --out wp7r_evidence/item4_cost.json
Each of coverage / bootstrap is resumable (one JSON line per seed, seeds already present are skipped).
NO engine change: everything here is script level. `n17_proto_wp6.py` is the WP6 prototype (pack / unpack / numeric Hessian / Jacobian), reused unchanged.
"""
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

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

K, F, L, DMAX = 2, 4, 2, 25
Z = 2.0
NOMINAL2, NOMINAL1 = math.erf(2.0 / math.sqrt(2.0)), math.erf(1.0 / math.sqrt(2.0))
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


def truth_model(X):
    """The engine at the generator's true parameters (WP5 n17_diag.truth_model, copied unchanged)."""
    from backbone.hssm.dual_model import DualTimescaleHSSM
    from backbone.hssm.model import FitState
    m = DualTimescaleHSSM(K, F, L, DMAX, seed=0)
    m._init_params(X)
    m.pi = np.array([1 - 1e-4, 1e-4]); m.A_slow = np.array([[0.0, 1.0], [1.0, 0.0]])
    m.dur_mu = np.full(K, math.log(15.0)); m.dur_sigma = np.full(K, 0.3)
    m.A_fast = np.stack([0.75 * np.eye(L)] * K); m.b_fast = np.zeros((K, L)); m.Q_fast = np.stack([0.3 * np.eye(L)] * K)
    m.m0 = np.zeros((K, L)); m.P0 = np.stack([0.04 * np.eye(L)] * K)
    C = np.zeros((F, L))
    for f in range(F):
        C[f, f % L] = 1.5
    m.C_obs = np.stack([C] * K); m.d_obs = np.array([np.full(F, -2.0), np.full(F, 2.0)]); m.R_obs = np.full((K, F), 0.05)
    m._is_fitted = True; m.fit_state_ = FitState.CONVERGED
    return m


def coverage_stats(m_hat, std, true_m, true_p, aligned=True):
    """Per latent dimension: share of rows with |z| <= 2 and <= 1, and mean z^2. z = (m_hat - map(true)) / std with the N17 per-true-regime in-sample affine map."""
    from tests.test_dual_model_latent_calibration import per_regime_affine_prediction
    ref = per_regime_affine_prediction(true_m, m_hat, true_p, min_points=3 * (L + 1)) if aligned else true_m
    z = (m_hat - ref) / std
    return dict(c2=np.mean(np.abs(z) <= 2.0, axis=0).tolist(), c1=np.mean(np.abs(z) <= 1.0, axis=0).tolist(), z2=np.mean(z ** 2, axis=0).tolist())


def _quiet():
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")


# ------------------------------------------------------------------------------------------------------------------------ part (a): coverage
def coverage_seed(job):
    seed, prior_path = job
    from backbone.hssm.fitting import fit_hssm
    from backbone.hssm.shrinkage import SharedPrior
    from data.synthetic_dual import generate_synthetic_dual_timescale_data
    rec = dict(seed=int(seed), arms={})
    t0 = time.time()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _quiet()
        prior = SharedPrior.from_dict(json.load(open(prior_path)))
        d = generate_synthetic_dual_timescale_data(n_regimes=K, n_features=F, latent_dim=L, n_segments=4, mean_dwell=15.0, seed=seed)
        rec["T"] = int(len(d.X))
        for arm, (use_prior, n_it) in {"shrink5_200": (True, 200), "shrink5_20": (True, 20), "personal_200": (False, 200), "personal_20": (False, 20)}.items():
            t1 = time.time()
            a = dict(error=None)
            try:
                kw = dict(matrix=d.X, candidate_ks=(2,), n_initializations=1, max_iter=n_it, random_seed=seed - 2000, allow_fast_test_fit=True, max_duration=DMAX,
                          model_type="dual_timescale", latent_dim=L)
                if use_prior:
                    kw.update(shared_prior=prior, shared_prior_n0=5.0)
                res = fit_hssm(**kw)
                a["withheld"] = bool(res.insights_withheld)                        # read BEFORE estimate_states
                a["withheld_reason"] = res.withheld_reason
                a["fit_state"] = str(res.fit_state.value if hasattr(res.fit_state, "value") else res.fit_state)
                a["n_iter_run"] = int(getattr(res.model, "n_iter_", -1))
                m_hat, std, _, _ = res.model.estimate_states(d.X)
                a.update(coverage_stats(m_hat, std, d.true_m_t, d.true_p_t))
                a["std_mean"] = float(np.mean(std))
            except Exception as exc:  # noqa: BLE001  recorded; an exception means no output for that arm
                a.update(error=f"{type(exc).__name__}: {str(exc)[:200]}", withheld=True)
            a["wall_s"] = round(time.time() - t1, 2)
            rec["arms"][arm] = a
        tm = truth_model(d.X)
        m_hat, std, _, _ = tm.estimate_states(d.X)
        rec["arms"]["T0"] = dict(error=None, withheld=False, fit_state="TRUTH", **coverage_stats(m_hat, std, d.true_m_t, d.true_p_t, aligned=False))
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


# ------------------------------------------------------------------------------------------------------------------------ part (b): parametric bootstrap
def simulate(m, T, rng):
    """One series of length T from the fitted dual-timescale parameters: regime path from pi, A_slow (zero diagonal) and the fitted truncated lognormal durations;
    latent state continuous across segments from N(m0_k, P0_k) with the regime's AR(1); emissions C m + d + N(0, R)."""
    from backbone.hssm.shrinkage import duration_pmf
    pmf = duration_pmf(np.asarray(m.dur_mu, float), np.asarray(m.dur_sigma, float), m.Dmax)
    pi = np.clip(np.asarray(m.pi, float), 0, None); pi = pi / pi.sum()
    A = np.clip(np.asarray(m.A_slow, float), 0, None)
    path = []
    k = int(rng.choice(m.K, p=pi))
    while len(path) < T:
        d = int(rng.choice(np.arange(1, m.Dmax + 1), p=pmf[k]))
        path += [k] * d
        p = A[k].copy(); p[k] = 0.0
        k = int(rng.choice(m.K, p=p / p.sum())) if p.sum() > 0 else int(rng.choice(m.K))
    path = np.array(path[:T])
    Lh, Fh = m.L, m.F
    mt = np.zeros((T, Lh))
    cur = rng.multivariate_normal(m.m0[path[0]], m.P0[path[0]])
    mt[0] = cur
    for t in range(1, T):
        k = path[t]
        cur = m.A_fast[k] @ cur + m.b_fast[k] + rng.multivariate_normal(np.zeros(Lh), m.Q_fast[k])
        mt[t] = cur
    X = np.zeros((T, Fh))
    for t in range(T):
        k = path[t]
        X[t] = m.C_obs[k] @ mt[t] + m.d_obs[k] + rng.normal(0.0, np.sqrt(m.R_obs[k]))
    return X, path


def bootstrap_seed(job):
    seed, B = job
    from backbone.hssm.dual_model import DualTimescaleHSSM
    from data.synthetic_dual import generate_synthetic_dual_timescale_data
    from tests.test_dual_model_latent_calibration import per_regime_affine_prediction

    class WarmDual(DualTimescaleHSSM):
        start = None

        def _init_params(self, X):
            super()._init_params(X)
            self._restore_params(self.start)

    rec = dict(seed=int(seed), B=int(B))
    t0 = time.time()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _quiet()
        d = generate_synthetic_dual_timescale_data(n_regimes=K, n_features=F, latent_dim=L, n_segments=4, mean_dwell=15.0, seed=seed)
        X, T = d.X, len(d.X)
        t1 = time.time()
        m = DualTimescaleHSSM(K, F, L, DMAX, seed=seed).fit(X, n_iter=20)
        rec["t_fit_s"] = round(time.time() - t1, 2)
        rec.update(T=int(T), fit_state=m.fit_state_.value)
        m_hat, std, labels, _ = m.estimate_states(X)
        snap = m._snapshot_params()
        rng = np.random.default_rng(10_000 + seed)
        aligned, t_rep, n_fail = [], [], 0
        for b in range(B):
            t2 = time.time()
            try:
                Xb, _ = simulate(m, T, rng)
                rep = WarmDual(K, F, L, DMAX, seed=1000 * seed + b)
                rep.start = snap
                rep.fit(Xb, n_iter=20)
                m_rep = rep.estimate_states(X)[0]                                    # the ORIGINAL series under the replicate's parameters
                aligned.append(per_regime_affine_prediction(m_rep, m_hat, labels, min_points=3 * (L + 1)))     # output-space frame removal, MAP labels of the original fit
            except Exception:  # noqa: BLE001
                n_fail += 1
            t_rep.append(time.time() - t2)
        rec.update(n_fail=int(n_fail), n_ok=int(len(aligned)), t_rep_s=round(float(np.mean(t_rep)), 2))
        rec["E0"] = coverage_stats(m_hat, std, d.true_m_t, d.true_p_t)
        if len(aligned) >= 5:
            var_b = np.var(np.array(aligned), axis=0, ddof=1)
            std_b = np.sqrt(std ** 2 + var_b)
            rec["EB"] = coverage_stats(m_hat, std_b, d.true_m_t, d.true_p_t)
            rec["ratio"] = float(np.mean(std_b / std))
        else:
            rec["EB"] = None
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


# ------------------------------------------------------------------------------------------------------------------------ part (c): cost
def cost(out_path):
    import n17_proto_wp6 as P
    from backbone.hssm.dual_model import DualTimescaleHSSM
    from data.synthetic_dual import generate_synthetic_dual_timescale_data
    res = {}
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _quiet()
        for name, (k, f, l_) in {"P50": (2, 4, 2), "P204": (3, 10, 3)}.items():
            d = generate_synthetic_dual_timescale_data(n_regimes=k, n_features=f, latent_dim=l_, n_segments=4, mean_dwell=15.0, seed=9101)
            X = d.X
            t0 = time.time(); m = DualTimescaleHSSM(k, f, l_, DMAX, seed=9101).fit(X, n_iter=20); t_fit = time.time() - t0
            th = P.pack(m)
            Pn = len(th)
            reps = 20
            t0 = time.time()
            for _ in range(reps):
                P.loglik(m, X, th)
            t_pass = (time.time() - t0) / reps
            t0 = time.time()
            for _ in range(5):
                P.smoothed_mean(m, X, th)
            t_smooth = (time.time() - t0) / 5
            n_pass_hess = int(Pn * Pn / 2 + 2 * Pn + 1)
            r = dict(K=k, F=f, L=l_, T=int(len(X)), P=int(Pn), t_fit20_s=round(t_fit, 2), t_pass_s=round(t_pass, 4), t_smooth_s=round(t_smooth, 4), n_pass_hessian=n_pass_hess,
                     n_smooth_jacobian=2 * Pn, est_hessian_s=round(n_pass_hess * t_pass, 1), est_jacobian_s=round(2 * Pn * t_smooth, 1),
                     est_bootstrap_30_s=round(30 * t_fit, 1), est_bootstrap_100_s=round(100 * t_fit, 1), measured_full_hessian=False)
            if Pn <= 60:
                t0 = time.time(); P.numeric_hessian(m, X, th); r["hessian_s_measured"] = round(time.time() - t0, 1)
                t0 = time.time(); P.jacobian_m(m, X, th); r["jacobian_s_measured"] = round(time.time() - t0, 1)
                r["measured_full_hessian"] = True
            res[name] = r
    json.dump(res, open(out_path, "w"), indent=1)
    for k_, v in res.items():
        print(k_, v)


def run_jobs(fn, jobs, out, workers, key="seed"):
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)[key] for l in open(out) if l.strip()}
    jobs = [j for j in jobs if j[0] not in done]
    print(f"{len(jobs)} seeds to run ({len(done)} done)", flush=True)
    if not jobs:
        return
    with mp.Pool(min(workers, len(jobs))) as pool:
        for rec in pool.imap_unordered(fn, jobs):
            with open(out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print("seed", rec["seed"], f"{rec['seconds']}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["coverage", "bootstrap", "cost"])
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--B", type=int, default=30)
    ap.add_argument("--prior", default=PRIOR_PATH)
    a = ap.parse_args()
    if a.cmd == "cost":
        cost(a.out)
    elif a.cmd == "coverage":
        run_jobs(coverage_seed, [(s, a.prior) for s in _parse_seeds(a.seeds)], a.out, a.workers)
    else:
        run_jobs(bootstrap_seed, [(s, a.B) for s in _parse_seeds(a.seeds)], a.out, a.workers)


if __name__ == "__main__":
    main()
