"""WP9 B2: why were 11 of 12 shrinkage fits publishable in WP7-R item 4 but 21 to 38% in item 3? (design: tests/PREDECLARED_CRITERIA_WP9.md section 15). usage:
  PYTHONPATH=. python3 integration_report/wp9/scripts/b2_run.py --out wp9_evidence/b2_fits.jsonl [--workers 1]
Seeds 9681..9692, spread factor f in {0, 0.5, 1.0}, 4 training segments, the Item 3 call (`_run_arm("shrink5", ...)`: public path fit_hssm, 200 iterations). Resumable, one line per fit."""
import argparse, contextlib, io, json, logging, multiprocessing as mp, os, sys, time, warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp7r" / "scripts"))
import item3_shrinkage_study as I3                                      # noqa: E402

SEEDS = list(range(9681, 9693))
SPREADS = (0.0, 0.5, 1.0)


def one(job):
    seed, f, prior_path = job
    from backbone.hssm.shrinkage import SharedPrior
    from data.synthetic_dual import generate_synthetic_dual_timescale_data
    from data.synthetic_population import PopulationSpec, generate_population_user
    logging.disable(logging.CRITICAL)
    prior = SharedPrior.from_dict(json.load(open(prior_path)))
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        user = generate_population_user(seed, 4, 2, PopulationSpec(), spread_factor=f)
        check = None
        if f == 0.0:                                                    # the Item 4 data: the generator's centre
            d = generate_synthetic_dual_timescale_data(n_regimes=2, n_features=4, latent_dim=2, n_segments=4, mean_dwell=15.0, seed=seed)
            check = bool(np.array_equal(np.asarray(d.X), user.X_train))
        rec = I3._run_arm("shrink5", user, seed, 200, prior)
    rec.update(seed=int(seed), spread=f, T_train=int(len(user.X_train)), centre_equals_item4_generator=check)
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=1); ap.add_argument("--prior", default=I3.PRIOR_PATH)
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        for ln in open(a.out):
            try:
                d = json.loads(ln); done.add((d["seed"], d["spread"]))
            except ValueError:
                pass
    jobs = [(s, f, a.prior) for f in SPREADS for s in SEEDS if (s, f) not in done]
    print(len(jobs), "to run", flush=True)
    with mp.Pool(a.workers) as pool:
        for rec in pool.imap_unordered(one, jobs):
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())
            print(rec["spread"], rec["seed"], "T", rec["T_train"], rec["fit_state"], "withheld" if rec["withheld"] else "publishable", rec["wall_s"], "s", flush=True)


if __name__ == "__main__":
    main()
