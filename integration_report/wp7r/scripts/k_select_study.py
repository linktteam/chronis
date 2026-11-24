"""WP7-R item 2: K selection with the collapse rule (baseline model, candidate_ks=(2,3,4)). Design: tests/PREDECLARED_CRITERIA_WP7R.md section 2.
usage: PYTHONPATH=. python3 integration_report/wp7r/scripts/k_select_study.py --seeds 7801:7825 --out wp7r_evidence/item2_kselect.jsonl [--workers 2] [--planted-k 3] [--designs easy,hard]
One JSON line per dataset (resumable). Planted K by seed: 2 for the first 8 seeds of 7801..7824, 3 for the next 8, 4 for the last 8 (unless --planted-k forces one).
The per-K restart logs are captured by wrapping fitting.fit_with_random_restarts, so the K-selection can be re-derived offline for other thresholds / BIC formulas (analyze_k_select.py)."""
from __future__ import annotations
import argparse, contextlib, io, json, multiprocessing as mp, os, time, warnings
import numpy as np

F, T = 7, 75


def planted(seed: int, K: int, design: str = "easy", F: int = F, T: int = T):
    """easy = the integration-style case of scripts/repro_collapse.py: segments of 9 to 22 rows, means up to 2.5 apart, noise 0.5, z-scored; hard = the same with means up to 1.25 and noise 0.8."""
    scale, noise = (2.5, 0.5) if design == "easy" else (1.25, 0.8)
    rng = np.random.default_rng(seed)
    means = rng.normal(0, 1.0, (K, F)); means *= scale / np.abs(means).max()
    path, cur = [], 0
    while len(path) < T:
        path += [cur] * int(rng.integers(9, 23)); cur = (cur + int(rng.integers(1, K))) % K
    path = np.array(path[:T]); X = means[path] + rng.normal(0, noise, (T, F))
    return (X - X.mean(0)) / X.std(0), path


def _seeds(txt):
    out = []
    for p in txt.split(","):
        if ":" in p:
            a, b = p.split(":"); out += list(range(int(a), int(b)))
        elif p:
            out.append(int(p))
    return out


def one(job):
    seed, planted_k, design = job
    from sklearn.metrics import adjusted_rand_score as ari
    from backbone.hssm import fitting
    from backbone.hssm.fitting import fit_hssm
    X, path = planted(seed, planted_k, design); days = np.arange(T, dtype=float)
    cap = {}
    orig = fitting.fit_with_random_restarts

    def wrapped(*a, **kw):
        model, run_log = orig(*a, **kw)
        k = kw.get("n_regimes")
        labels = fitting.collapse_report(model, X, kw.get("timestamps"))[1]
        cr = model.collapse_report_ or {}
        cap[int(k)] = dict(run_log=[dict(fit_state=str(r["fit_state"]).split(".")[-1], ll=r["log_likelihood"], min_eff=r.get("min_effective_days"), n_iter=r["n_iter"]) for r in run_log],
                           ll=float(model.log_likelihood_), n_params=int(model.n_params()), min_eff=cr.get("min_effective_days"), eff=cr.get("effective_days"), collapsed=cr.get("collapsed"),
                           ari=float(ari(path, labels)), dwell=[float(x) for x in np.bincount(labels, minlength=int(k))])
        return model, run_log

    rec = dict(seed=seed, planted_k=planted_k, design=design, ok=False)
    t0 = time.time()
    fitting.fit_with_random_restarts = wrapped
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            r = fit_hssm(matrix=X, candidate_ks=(2, 3, 4), n_initializations=10, max_iter=60, random_seed=seed, timestamps=days, user_id=f"u{seed}{design}", dataset_hash=f"h{seed}{design}")
        lab = np.asarray(r.p_t_posterior).argmax(axis=1)
        rec.update(ok=True, selected_k=int(r.K), fit_state=str(r.fit_state.value), model_fit_state=str((r.convergence_metadata or {}).get("fit_state")).split(".")[-1],
                   insights_withheld=bool(r.insights_withheld), withheld_reason=r.withheld_reason, fit_collapsed=bool(r.fit_collapsed),
                   bic_by_k={str(k): v for k, v in r.bic_by_k.items()}, loglik_by_k={str(k): v for k, v in (r.convergence_metadata or {}).get("loglik_by_k", {}).items()},
                   served_ari=float(ari(path, lab)), n_obs=int(T), per_k={str(k): v for k, v in cap.items()})
    except Exception as exc:  # noqa: BLE001
        rec.update(error=f"{type(exc).__name__}: {str(exc)[:300]}")
    finally:
        fitting.fit_with_random_restarts = orig
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--planted-k", type=int, default=0); ap.add_argument("--designs", default="easy,hard")
    a = ap.parse_args()
    seeds = _seeds(a.seeds)
    done = set()
    if os.path.exists(a.out):
        for ln in open(a.out):
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            if d.get("ok"):
                done.add((d["design"], d["seed"]))
    jobs = [(s, a.planted_k or 2 + (s - 7801) // 8, dz) for dz in a.designs.split(",") for s in seeds if (dz, s) not in done]
    print(f"{len(jobs)} jobs ({len(done)} done)", flush=True)
    if not jobs:
        return
    with mp.Pool(min(a.workers, len(jobs))) as pool:
        for rec in pool.imap_unordered(one, jobs):
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(rec["design"], rec["seed"], "planted", rec["planted_k"], "selected", rec.get("selected_k"), rec.get("fit_state"), rec.get("error", ""), f"{rec['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
