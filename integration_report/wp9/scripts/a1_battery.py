"""WP9 A1 (bit-identity): the baseline battery through the PUBLIC path in two trees (reference = a clean export of the WP8 final tree 7887118; new = a clean export of the WP9 head).
Same users as the WP8 Item 8 battery (population users 5301..5312), two variants: `fast` = the WP8 variant (K 2, 3 restarts, fast-test fit) and `full` = K in (2, 3), the production 10 restarts.
usage: python3 a1_battery.py fit --tree TREE --variant fast|full --out FILE.pkl [--workers 2]
       python3 a1_battery.py compare REF.pkl NEW.pkl
`compare` checks that every fitted output, the BIC, the status fields and every OLD key of `restart_summary` are bit-identical, and that the new tree only ADDS keys to `restart_summary`."""
import argparse, contextlib, io, logging, multiprocessing as mp, pickle, sys, time, warnings
import numpy as np

NUMERIC = ("m_t_posterior_mean", "m_t_uncertainty", "p_t_estimate", "p_t_posterior")
SEEDS = list(range(5301, 5313))
NEW_KEYS = {"consensus_m", "consensus_tau", "n_healthy_restarts", "consensus_min_ari", "fit_unstable", "fit_unstable_reason"}


def one(job):
    tree, seed, variant = job
    sys.path.insert(0, tree)
    logging.disable(logging.CRITICAL)
    from backbone.hssm.config import ColdStartConfig
    from backbone.hssm.fitting import fit_hssm
    from data.synthetic_population import PopulationSpec, generate_population_user
    t0 = time.time()
    rec = dict(seed=seed)
    kw = dict(candidate_ks=(2,), n_initializations=3, max_iter=60, allow_fast_test_fit=True) if variant == "fast" else dict(candidate_ks=(2, 3), n_initializations=10, max_iter=60)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        u = generate_population_user(seed, 12, 0, PopulationSpec())
        r = fit_hssm(matrix=u.X_train, random_seed=seed - 2000, config=ColdStartConfig(min_present_sessions=1), **kw)
    for n in NUMERIC:
        rec[n] = np.array(getattr(r, n), copy=True)
    rec["regime_means"] = np.array(r.model.mu, copy=True)
    rec.update(duration_parameters={k: list(v) for k, v in r.duration_parameters.items()}, BIC=float(r.BIC), fit_state=str(r.fit_state.value), Dmax=int(r.Dmax), K=int(r.K),
               insights_withheld=bool(r.insights_withheld), withheld_reason=r.withheld_reason, fit_collapsed=bool(getattr(r, "fit_collapsed", False)),
               restart_summary=dict(r.restart_summary), bic_by_k={int(k): float(v) for k, v in r.bic_by_k.items()}, seconds=round(time.time() - t0, 1))
    return rec


def cmd_fit(a):
    with mp.Pool(a.workers) as pool:
        recs = pool.map(one, [(a.tree, s, a.variant) for s in SEEDS])
    pickle.dump(sorted(recs, key=lambda r: r["seed"]), open(a.out, "wb"))
    for r in sorted(recs, key=lambda r: r["seed"]):
        print(r["seed"], r["fit_state"], "withheld" if r["insights_withheld"] else "publishable", r["K"], r["seconds"], "s")


def compare_records(ref, new):
    assert [r["seed"] for r in ref] == [r["seed"] for r in new]
    n_fit = n_bic = n_status = n_old_keys = n_added_only = n_healthy = 0
    rows = []
    for r, n in zip(ref, new):
        fit_same = all(np.array_equal(r[k], n[k]) for k in NUMERIC + ("regime_means",)) and r["duration_parameters"] == n["duration_parameters"] and r["fit_state"] == n["fit_state"] \
            and r["Dmax"] == n["Dmax"] and r["K"] == n["K"]
        bic_same = r["BIC"] == n["BIC"] and r["bic_by_k"] == n["bic_by_k"]
        status_same = (r["insights_withheld"], r["withheld_reason"], r["fit_collapsed"]) == (n["insights_withheld"], n["withheld_reason"], n["fit_collapsed"])
        old_keys_same = all(k in n["restart_summary"] and (r["restart_summary"][k] == n["restart_summary"][k]) for k in r["restart_summary"])
        added_only = set(n["restart_summary"]) - set(r["restart_summary"]) == NEW_KEYS
        n_fit += fit_same; n_bic += bic_same; n_status += status_same; n_old_keys += old_keys_same; n_added_only += added_only
        healthy = not r["insights_withheld"]
        n_healthy += bool(healthy and fit_same and bic_same and status_same)
        rows.append((r["seed"], r["fit_state"], "publishable" if healthy else "withheld", fit_same, bic_same, status_same, old_keys_same, added_only, n["restart_summary"].get("fit_unstable")))
    N = len(ref)
    ok = n_fit == N and n_bic == N and n_status == N and n_old_keys == N and n_added_only == N
    return dict(n=N, fit=n_fit, bic=n_bic, status=n_status, old_keys=n_old_keys, added_only=n_added_only, healthy_identical=n_healthy, n_previously_healthy=sum(1 for r in ref if not r["insights_withheld"]), rows=rows, ok=ok)


def cmd_compare(a):
    ref, new = (pickle.load(open(p, "rb")) for p in (a.ref, a.new))
    c = compare_records(ref, new)
    print("seed  fit_state            ref_status   fitted outputs identical  BIC identical  status identical  old restart_summary keys identical  only new keys added  fit_unstable (new tree)")
    for row in c["rows"]:
        print(f"{row[0]}  {row[1]:20s} {row[2]:11s}  {str(row[3]):24s}  {str(row[4]):13s}  {str(row[5]):16s}  {str(row[6]):35s}  {str(row[7]):19s}  {row[8]}")
    print(f"\nfitted outputs (m_t, uncertainty, p_t, posteriors, regime means, duration parameters, fit_state, Dmax, K) bit-identical: {c['fit']} of {c['n']}")
    print(f"BIC and BIC by K identical: {c['bic']} of {c['n']}; withheld status, reason and collapse flag identical: {c['status']} of {c['n']}")
    print(f"old restart_summary keys identical: {c['old_keys']} of {c['n']}; exactly the six new keys added: {c['added_only']} of {c['n']}")
    print(f"previously healthy (publishable) fits unchanged in every output: {c['healthy_identical']} of {c['n_previously_healthy']}")
    print("PASS" if c["ok"] else "FAIL")
    sys.exit(0 if c["ok"] else 1)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit"); p.add_argument("--tree", required=True); p.add_argument("--variant", choices=("fast", "full"), required=True); p.add_argument("--out", required=True); p.add_argument("--workers", type=int, default=2); p.set_defaults(f=cmd_fit)
    p = sub.add_parser("compare"); p.add_argument("ref"); p.add_argument("new"); p.set_defaults(f=cmd_compare)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
