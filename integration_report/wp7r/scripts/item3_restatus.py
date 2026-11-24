"""WP7-R item 3, section 3.9: refit every NOT-withheld stage B fit with the final engine and take its withheld status from the refit.
usage: PYTHONPATH=. python3 item3_restatus.py --in wp7r_evidence/item3_stageB.jsonl --out wp7r_evidence/item3_stageB_final_engine.jsonl [--workers 2]
The output has the same records as the input; for re-fitted records `withheld`, `withheld_reason`, `fit_state` come from the refit and `restatus` holds the check
(stored vs refit fit_state, n_iter_run, score mean). Resumable per record (a partial file `OUT.partial.jsonl` holds the refits done)."""
import argparse, json, multiprocessing as mp, os, sys, time
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "integration_report/wp7r/scripts"))
import item3_shrinkage_study as S  # noqa: E402


def refit(job):
    import contextlib, io, logging, warnings
    from backbone.hssm.shrinkage import SharedPrior
    from data.synthetic_population import PopulationSpec, generate_population_user
    rec = job
    logging.disable(logging.CRITICAL)
    prior = SharedPrior.from_dict(json.load(open(S.PRIOR_PATH)))
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        user = generate_population_user(rec["seed"], S.SEGMENTS[rec["length"]], 2, PopulationSpec())
        new = S._run_arm(rec["arm"], user, rec["seed"], rec["max_iter"], prior)
    return rec["seed"], rec["length"], rec["arm"], new


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.inp) if l.strip()]
    part = a.out + ".partial.jsonl"
    done = {}
    if os.path.exists(part):
        for l in open(part):
            d = json.loads(l); done[(d["seed"], d["length"], d["arm"])] = d["new"]
    todo = [r for r in rows if r["arm"] in ("personal", "shrink5") and not r["withheld"] and (r["seed"], r["length"], r["arm"]) not in done]
    print(f"{len(todo)} fits to refit ({len(done)} already done)", flush=True)
    t0 = time.time()
    with mp.Pool(a.workers) as pool, open(part, "a") as fh:
        for seed, length, arm, new in pool.imap_unordered(refit, todo):
            fh.write(json.dumps(dict(seed=seed, length=length, arm=arm, new=new)) + "\n"); fh.flush()
            done[(seed, length, arm)] = new
            print(f"{arm} seed {seed} len {length}: withheld {new['withheld']} ({time.time() - t0:.0f}s elapsed)", flush=True)
    out, n_diff, n_flip = [], 0, 0
    for r in rows:
        key = (r["seed"], r["length"], r["arm"])
        if key in done:
            n = done[key]
            same = (n.get("fit_state") == r.get("fit_state") and n.get("n_iter_run") == r.get("n_iter_run") and n.get("mean") == r.get("mean"))
            n_diff += (not same)
            n_flip += bool(n["withheld"])
            r = dict(r, withheld=bool(n["withheld"]), withheld_reason=n["withheld_reason"], fit_state=n["fit_state"],
                     restatus=dict(reproduced=bool(same), stored_state=r.get("fit_state"), stored_iter=r.get("n_iter_run"), stored_mean=r.get("mean"), refit_iter=n.get("n_iter_run"), refit_mean=n.get("mean")))
        out.append(r)
    with open(a.out, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"refit {len(done)} fits; newly withheld {n_flip}; NOT reproduced (state, iterations or score differ from the stored record): {n_diff}")


if __name__ == "__main__":
    main()
