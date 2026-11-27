"""WP10B B2: bit-identity of the DUAL engine between two trees (reference = a clean export of the WP10A final tree 95dd125; new = a clean export of the WP10B head). The WP9 batteries (`wp9/scripts/a1_battery.py`) fit only the
baseline engine, so a change inside `dual_model.py` needs its own check. Every fit goes through the PUBLIC path `fit_hssm(model_type="dual_timescale", ...)` with all B2 switches at their defaults (nothing here sets them).
Variants: `pop` = 12 population users (seeds 11911..11922; K 2, F 4, L 2, Dmax 25, 30 days, 60 iterations, 1 restart; the first 6 without a shared prior, the last 6 prior-anchored with the true prior, n0 = 5)
          `ident` = 4 identity users (seeds 11931..11934, one per stratum plus one more easy; K 3, F 7, L 2, Dmax 45, 150 days, 60 iterations, 1 restart).
usage: python3 dual_battery.py fit --tree TREE --variant pop|ident --out FILE.pkl [--workers 2]
       python3 dual_battery.py compare REF.pkl NEW.pkl
`compare` prints PASS only if every fitted parameter array, the whole log-likelihood history, the EM state, the iteration count, the stopping rule, the withhold reason and the duration-box refit record are bit-identical in every fit."""
import argparse, contextlib, io, logging, multiprocessing as mp, os, pickle, sys, time, warnings
import numpy as np

PARAMS = ("pi", "A_slow", "dur_mu", "dur_sigma", "A_fast", "b_fast", "Q_fast", "C_obs", "d_obs", "R_obs")
POP_SEEDS = list(range(11911, 11923))
IDENT = [(11931, "easy"), (11932, "medium"), (11933, "hard"), (11934, "easy")]
PRIOR_PATH = "wp7r_evidence/prior_main_wp5.json"


def one(job):
    tree, variant, seed, extra = job
    tree = os.path.abspath(tree)
    os.chdir(tree)                                  # the true prior is read relative to the tree
    sys.path.insert(0, tree)
    logging.disable(logging.CRITICAL)
    from backbone.hssm.config import ColdStartConfig
    from backbone.hssm.fitting import fit_hssm
    t0 = time.time()
    rec = dict(variant=variant, seed=seed)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        if variant == "pop":
            import json
            from data.synthetic_population import PopulationSpec, generate_population_user
            from backbone.hssm.shrinkage import SharedPrior
            u = generate_population_user(seed, 2, 0, PopulationSpec())
            kw = dict(matrix=u.X_train, candidate_ks=(2,), n_initializations=1, max_iter=60, random_seed=seed - 2000, allow_fast_test_fit=True, config=ColdStartConfig(min_present_sessions=1),
                      max_duration=25, model_type="dual_timescale", latent_dim=2)
            if extra == "prior":
                kw.update(shared_prior=SharedPrior.from_dict(json.load(open(PRIOR_PATH))), shared_prior_n0=5)
            r = fit_hssm(**kw)
        else:
            sys.path.insert(0, os.path.join(tree, "integration_report", "wp9", "scripts"))
            import wp9_fit_lib as F
            st = F.stream_for(dict(user=seed, stratum=extra))
            w = st.window(0, 120)
            r = fit_hssm(matrix=w["matrix"], candidate_ks=(3,), n_initializations=1, max_iter=60, random_seed=seed, allow_fast_test_fit=True, max_duration=45, model_type="dual_timescale", latent_dim=2)
    m = r.model
    for n in PARAMS:
        rec[n] = np.array(getattr(m, n), copy=True)
    box = m.duration_box_refit_
    rec.update(ll_history=[float(x) for x in m.log_likelihood_history_], fit_state=str(r.fit_state.value), n_iter=int(m.n_iter_), criterion=m.convergence_criterion_, withheld_reason=r.withheld_reason,
               Dmax=int(m.Dmax), box_refit=None if box is None else {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in box.items() if not isinstance(v, list)},
               window_gain_at_cap=m.window_gain_at_cap_, seconds=round(time.time() - t0, 1))
    return rec


def jobs_for(tree, variant):
    if variant == "pop":
        return [(tree, "pop", s, "prior" if i >= 6 else "none") for i, s in enumerate(POP_SEEDS)]
    return [(tree, "ident", s, st) for s, st in IDENT]


def cmd_fit(a):
    with mp.Pool(a.workers) as pool:
        recs = pool.map(one, jobs_for(a.tree, a.variant))
    pickle.dump(sorted(recs, key=lambda r: r["seed"]), open(a.out, "wb"))
    for r in sorted(recs, key=lambda r: r["seed"]):
        print(r["seed"], r["fit_state"], "withheld" if r["withheld_reason"] else "publishable", r["n_iter"], r["seconds"], "s")


def compare_records(ref, new):
    """Returns (ok, rows). Every field of a record except the wall time must be bit-identical."""
    assert [r["seed"] for r in ref] == [r["seed"] for r in new]
    rows, ok = [], True
    for r, n in zip(ref, new):
        diffs = []
        for k in PARAMS:
            if not np.array_equal(r[k], n[k]):
                diffs.append(k)
        for k in ("ll_history", "fit_state", "n_iter", "criterion", "withheld_reason", "Dmax", "box_refit", "window_gain_at_cap"):
            if r[k] != n[k]:
                diffs.append(k)
        rows.append((r["seed"], r["fit_state"], len(r["ll_history"]), diffs))
        ok = ok and not diffs
    return ok, rows


def cmd_compare(a):
    ref, new = pickle.load(open(a.ref, "rb")), pickle.load(open(a.new, "rb"))
    ok, rows = compare_records(ref, new)
    n_pub = sum(1 for r in ref if r["withheld_reason"] is None)
    for seed, st, nl, diffs in rows:
        print(seed, st, nl, "identical" if not diffs else "DIFFERS: " + ",".join(diffs))
    print(f"{len(rows)} fits ({n_pub} publishable in the reference); " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--tree", required=True); f.add_argument("--variant", choices=["pop", "ident"], required=True); f.add_argument("--out", required=True); f.add_argument("--workers", type=int, default=2)
    c = sub.add_parser("compare"); c.add_argument("ref"); c.add_argument("new")
    a = ap.parse_args()
    cmd_fit(a) if a.cmd == "fit" else cmd_compare(a)
