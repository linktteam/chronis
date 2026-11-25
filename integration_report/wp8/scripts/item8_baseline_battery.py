"""WP8 Item 8, criterion 10: the BASELINE identity battery, a copy of integration_report/wp7r/scripts/item5_identity_battery.py (dual) with the baseline engine in place of the dual model.
Same users (population users, seeds 5301..5312), same public path, same numeric outputs; the reference tree is a clean export of 27c591c.

usage:  fit    PYTHONPATH=<TREE> python3 item5_identity_battery.py fit --tree <TREE> --seeds 5301:5313 --out FILE.pkl [--workers 2]     (TREE = a clean export of c95cb5d, or the working tree)
        compare python3 item5_identity_battery.py compare REF.pkl NEW.pkl

Fits each user through the PUBLIC path in the given tree (dual model, one restart, max_iter 200, Dmax 25, cold-start gate off for the study) and stores the numeric outputs and
the status fields; `compare` checks bit identity of the numeric outputs and classifies every fit by its status in the reference tree and in the new tree.
"""
import argparse, contextlib, io, logging, multiprocessing as mp, pickle, sys, time, warnings
import numpy as np

NUMERIC = ("m_t_posterior_mean", "m_t_uncertainty", "p_t_estimate", "p_t_posterior")


def _parse(txt):
    out = []
    for part in txt.split(","):
        if ":" in part:
            a, b = part.split(":"); out += list(range(int(a), int(b)))
        elif part:
            out.append(int(part))
    return out


def one(job):
    tree, seed, prior_path = job
    sys.path.insert(0, tree)
    logging.disable(logging.CRITICAL)
    from backbone.hssm.config import ColdStartConfig
    from backbone.hssm.fitting import fit_hssm
    from data.synthetic_population import PopulationSpec, generate_population_user
    t0 = time.time()
    rec = dict(seed=seed)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        u = generate_population_user(seed, 12, 0, PopulationSpec())
        kw = {}
        if prior_path:                                                    # supplementary battery 5.5: the shared-prior path (prior loaded with the tree's own SharedPrior class)
            import json
            from backbone.hssm.shrinkage import SharedPrior
            kw = dict(shared_prior=SharedPrior.from_dict(json.load(open(prior_path))), shared_prior_n0=5.0)
        r = fit_hssm(matrix=u.X_train, candidate_ks=(2,), n_initializations=3, max_iter=60, random_seed=seed - 2000, allow_fast_test_fit=True,
                     config=ColdStartConfig(min_present_sessions=1), **kw)
    for n in NUMERIC:
        rec[n] = np.array(getattr(r, n), copy=True)
    rec["regime_means"] = np.array(r.model.mu, copy=True)
    rec.update(duration_parameters={k: list(v) for k, v in r.duration_parameters.items()}, BIC=float(r.BIC), fit_state=str(r.fit_state.value), Dmax=int(r.Dmax),
               insights_withheld=bool(r.insights_withheld), withheld_reason=r.withheld_reason, T=int(len(u.X_train)),
               fit_collapsed=bool(getattr(r, "fit_collapsed", False)), fit_duration_degenerate=bool(getattr(r, "fit_duration_degenerate", False)),
               collapse_report=dict(getattr(r, "collapse_report", {}) or {}), duration_degenerate_report=dict(getattr(r, "duration_degenerate_report", {}) or {}),
               seconds=round(time.time() - t0, 1))
    return rec


def cmd_fit(a):
    jobs = [(a.tree, s, a.shared_prior) for s in _parse(a.seeds)]
    with mp.Pool(min(a.workers, len(jobs))) as pool:
        recs = pool.map(one, jobs)
    pickle.dump(sorted(recs, key=lambda r: r["seed"]), open(a.out, "wb"))
    for r in sorted(recs, key=lambda r: r["seed"]):
        print(r["seed"], r["fit_state"], "withheld" if r["insights_withheld"] else "publishable", r["seconds"], "s")


def compare_records(ref, new):
    """Bit identity of two batteries (same seeds). Fitted outputs (m_t, uncertainty, p_t, posteriors, regime means, duration parameters, fit state, Dmax) must be identical; the BIC may differ,
    but only by exactly K ln(T): the owner decision B1 (WP8 Item 0b) changed the default parameter count from the legacy one to the free one, so legacy BIC - free BIC = K ln(T) per fit.
    Returned: counts and rows. `ok` needs every fit identical in the fitted outputs, every BIC shift exact and no unexplained status change."""
    assert [r["seed"] for r in ref] == [r["seed"] for r in new]
    classes = {"previously healthy, unchanged": [], "previously publishable, newly withheld by R1": [], "previously publishable, newly withheld by R2": [],
               "previously publishable, newly withheld by R1 and R2": [], "already withheld (unchanged status)": [], "NOT EXPLAINED": []}
    rows, n_strict, n_fit_id, n_bic_ok = [], 0, 0, 0
    for r, n in zip(ref, new):
        fit_same = all(np.array_equal(r[k], n[k]) for k in NUMERIC + ("regime_means",)) and r["duration_parameters"] == n["duration_parameters"] \
            and r["fit_state"] == n["fit_state"] and r["Dmax"] == n["Dmax"]
        K = int(np.asarray(r["regime_means"]).shape[0])
        shift = float(r["BIC"]) - float(n["BIC"])
        bic_same = r["BIC"] == n["BIC"]
        bic_shift_ok = bool(abs(shift - K * np.log(max(int(r["T"]), 1))) <= 1e-8)       # Item 0b measured a largest deviation of 1.3e-13
        n_strict += bool(fit_same and bic_same); n_fit_id += bool(fit_same); n_bic_ok += bool(bic_shift_ok or bic_same)
        old_pub, new_pub = not r["insights_withheld"], not n["insights_withheld"]
        r1, r2 = n["fit_collapsed"], n["fit_duration_degenerate"]
        if not old_pub:
            c = "already withheld (unchanged status)" if not new_pub else "NOT EXPLAINED"
        elif new_pub and not (r1 or r2):
            c = "previously healthy, unchanged"
        elif (not new_pub) and r1 and r2:
            c = "previously publishable, newly withheld by R1 and R2"
        elif (not new_pub) and r1:
            c = "previously publishable, newly withheld by R1"
        elif (not new_pub) and r2:
            c = "previously publishable, newly withheld by R2"
        else:
            c = "NOT EXPLAINED"
        classes[c].append(r["seed"])
        rows.append((r["seed"], r["fit_state"], "pub" if old_pub else "withheld", "pub" if new_pub else "withheld", r1, r2, fit_same, bic_shift_ok))
    n_flag = sum(1 for r in new if r["fit_collapsed"] or r["fit_duration_degenerate"])
    ok = n_fit_id == len(ref) and n_bic_ok == len(ref) and not classes["NOT EXPLAINED"]
    return dict(n=len(ref), n_strict=n_strict, n_fit_identical=n_fit_id, n_bic_shift_exact=n_bic_ok, classes=classes, rows=rows, n_flagged=n_flag, ok=ok)


def cmd_compare(a):
    ref, new = (pickle.load(open(p, "rb")) for p in (a.ref, a.new))
    c = compare_records(ref, new)
    print("seed  fit_state            ref_status  new_status  R1(collapsed) R2(degenerate)  fitted outputs bit-identical  BIC shift = K ln T (B1)")
    for row in c["rows"]:
        print(f"{row[0]}  {row[1]:20s} {row[2]:10s}  {row[3]:10s}  {str(row[4]):12s}  {str(row[5]):13s}  {str(row[6]):28s}  {row[7]}")
    print(f"\nfitted outputs bit-identical (m_t, m_t_std, p_t, p_t_post, regime means, duration parameters, fit_state, Dmax): {c['n_fit_identical']} of {c['n']}")
    print(f"BIC differs by exactly K ln(T), the owner-decided B1 change of the parameter count (or is equal): {c['n_bic_shift_exact']} of {c['n']}")
    print(f"strictly identical including the BIC value: {c['n_strict']} of {c['n']} (the BIC value changes by design, see B1)")
    for k, v in c["classes"].items():
        print(f"{k}: {len(v)} {v}")
    print(f"share of fits flagged by R1 or R2: {c['n_flagged']} of {c['n']}")
    print("PASS (5.2a)" if c["ok"] else "FAIL (5.2a)")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit"); p.add_argument("--tree", required=True); p.add_argument("--seeds", required=True); p.add_argument("--out", required=True); p.add_argument("--workers", type=int, default=2)
    p.add_argument("--shared-prior", default=None, help="absolute path of a SharedPrior json: fit through the shared-prior path with shared_prior_n0 = 5.0 (supplementary battery, section 5.5)"); p.set_defaults(f=cmd_fit)
    p = sub.add_parser("compare"); p.add_argument("ref"); p.add_argument("new"); p.set_defaults(f=cmd_compare)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
