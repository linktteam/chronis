"""WP9 B3: K-adequacy study (design: tests/PREDECLARED_CRITERIA_WP9.md Part 4). usage:
  PYTHONPATH=. python3 integration_report/wp9/scripts/b3_run.py --range dev|eval --out FILE.jsonl [--workers 2]
dev = seeds 9531..9554, eval = seeds 9501..9524 (refused without tests/DESIGN_FREEZE_WP9_B3.txt). One line per (seed, design), resumable. The public path `fit_hssm` is used; the record keeps the
selected K, the BIC by K, the regime means and variances and the restart summary so that other thresholds can be applied offline."""
import argparse, contextlib, io, json, logging, multiprocessing as mp, os, sys, time, warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp7r" / "scripts"))
from k_select_study import planted                                   # noqa: E402

RANGES = {"dev": range(9531, 9555), "eval": range(9501, 9525)}
CANDIDATES = (2, 3, 4, 5)


def planted_k(seed, lo):
    return 2 + (seed - lo) // 8


def one(job):
    seed, design, lo = job
    from sklearn.metrics import adjusted_rand_score as ari
    from backbone.hssm.fitting import fit_hssm
    logging.disable(logging.CRITICAL)
    K = planted_k(seed, lo)
    X, path = planted(seed, K, design)
    days = np.arange(len(X), dtype=float)
    rec = dict(seed=seed, design=design, planted_k=K, ok=False)
    t0 = time.time()
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            r = fit_hssm(matrix=X, candidate_ks=CANDIDATES, n_initializations=10, max_iter=60, random_seed=seed, timestamps=days, user_id=f"b3-{seed}-{design}", dataset_hash=f"b3h{seed}{design}")
        lab = np.asarray(r.p_t_posterior).argmax(axis=1)
        rs = r.restart_summary
        rec.update(ok=True, selected_k=int(r.K), bic_by_k={str(k): float(v) for k, v in r.bic_by_k.items()}, mu=np.asarray(r.model.mu).tolist(), var=np.asarray(r.model.var).tolist(),
                   fit_unstable=rs.get("fit_unstable"), consensus_min_ari=rs.get("consensus_min_ari"), withheld=bool(r.insights_withheld), served_ari=float(ari(path, lab)), k_adequacy=r.k_adequacy,
                   fit_state=str(r.fit_state.value))
    except Exception as exc:  # noqa: BLE001
        rec.update(error=f"{type(exc).__name__}: {str(exc)[:300]}")
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--range", required=True, choices=list(RANGES)); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.range == "eval" and not Path("tests/DESIGN_FREEZE_WP9_B3.txt").exists():
        sys.exit("the evaluation runs only after tests/DESIGN_FREEZE_WP9_B3.txt is committed")
    seeds = list(RANGES[a.range]); lo = seeds[0]
    done = set()
    if os.path.exists(a.out):
        for ln in open(a.out):
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            if d.get("ok"):
                done.add((d["design"], d["seed"]))
    jobs = [(s, dz, lo) for dz in ("easy", "hard") for s in seeds if (dz, s) not in done]
    print(f"{len(jobs)} jobs ({len(done)} done)", flush=True)
    with mp.Pool(min(a.workers, max(len(jobs), 1))) as pool:
        for rec in pool.imap_unordered(one, jobs):
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())
            print(rec["design"], rec["seed"], "planted", rec["planted_k"], "selected", rec.get("selected_k"), rec.get("error", ""), f"{rec['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
