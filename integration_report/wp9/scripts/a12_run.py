"""WP9 A1/A2 fits (design: tests/PREDECLARED_CRITERIA_WP9.md Part 2). Baseline engine, K = 3, 60 EM iterations, every restart kept (`wp9_fit_lib`). Resumable (a finished job is skipped).
Groups (users, windows, fit seed, restarts, number of fits):
  DEV-A   9301..9360 (9301..9320 hard, 9321..9340 medium, 9341..9360 easy), windows [0, t) for t = 90, 97, ..., 146, 150, fit seed = user, 10 restarts      600
  DEV-B   the same users, t in the 14-day schedule 90, 104, 118, 132, 146, 150, fit seed = user + 1000, 10 restarts                                          360
  EVAL-A  9401..9448 (9401..9416 hard, 9417..9432 medium, 9433..9448 easy), the weekly windows, fit seed = user, 20 restarts                                 480   (run ONCE, after the design freeze)
  EVAL-B  the same users, the 14-day windows, fit seed = user + 1000, 20 restarts                                                                            288
usage: PYTHONPATH=. python3 integration_report/wp9/scripts/a12_run.py wp9_evidence/a12_dev_fits.jsonl --only DEV-A,DEV-B [--workers 2] [--count]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wp9_fit_lib as L

G_T = list(range(90, 147, 7)) + [150]
B_T = [90, 104, 118, 132, 146, 150]                     # the 14-day schedule of WP8 (`lineage_study_lib.schedule_days(14)`)
DEV_USERS = range(9301, 9361)
EVAL_USERS = range(9401, 9449)


def stratum_dev(u):
    return "hard" if u <= 9320 else ("medium" if u <= 9340 else "easy")


def stratum_eval(u):
    return "hard" if u <= 9416 else ("medium" if u <= 9432 else "easy")


def _job(user, t, fitseed, stratum, n_init):
    return dict(user=user, start=0, end=t, fitseed=fitseed, K=3, n_init=n_init, max_iter=60, stratum=stratum, drift=0.0, kchange=None)


def groups():
    return {
        "DEV-A": [_job(u, t, u, stratum_dev(u), 10) for u in DEV_USERS for t in G_T],
        "DEV-B": [_job(u, t, u + 1000, stratum_dev(u), 10) for u in DEV_USERS for t in B_T],
        "EVAL-A": [_job(u, t, u, stratum_eval(u), 20) for u in EVAL_USERS for t in G_T],
        "EVAL-B": [_job(u, t, u + 1000, stratum_eval(u), 20) for u in EVAL_USERS for t in B_T],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--only", required=True); ap.add_argument("--count", action="store_true")
    a = ap.parse_args()
    g = groups()
    names = [n for n in a.only.split(",") if n]
    unknown = [n for n in names if n not in g]
    if unknown:
        sys.exit(f"unknown groups {unknown}; known: {sorted(g)}")
    if not a.count and any(n.startswith("EVAL") for n in names) and not Path("tests/DESIGN_FREEZE_WP9_A12.txt").exists():
        sys.exit("EVAL groups run only after the design freeze file tests/DESIGN_FREEZE_WP9_A12.txt is committed")
    js = [j for n in names for j in g[n]]
    print({n: len(g[n]) for n in names}, "total", len(js), flush=True)
    if not a.count:
        L.run_jobs(js, a.out, a.workers)
