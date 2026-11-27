"""WP10B C2 fits (design: tests/PREDECLARED_CRITERIA_WP10B.md section 5). Baseline engine, K = 3, 60 EM iterations, 10 restarts = the production default, every restart kept (`wp9_fit_lib`), the WP10A A3 seed-A chain:
windows [0, t) for t = 90, 97, ..., 146, 150 (10 fits per user), fit seed = user. Resumable (a finished job is skipped). Groups (users, number of fits):
  DEV-C2   11101..11136 (11101..11112 hard, 11113..11124 medium, 11125..11136 easy)   360
  EVAL-C2  11141..11188 (11141..11156 hard, 11157..11172 medium, 11173..11188 easy)    480   (after tests/DESIGN_FREEZE_WP10B_C2.txt is committed)
usage: PYTHONPATH=. python3 integration_report/wp10b/scripts/c2_run.py OUT.jsonl --only DEV-C2|EVAL-C2 [--workers 2] [--count] [--limit N]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp9" / "scripts"))
import wp9_fit_lib as L                                                               # noqa: E402

G_T = list(range(90, 147, 7)) + [150]
DEV_USERS, EVAL_USERS = range(11101, 11137), range(11141, 11189)
FREEZE = "tests/DESIGN_FREEZE_WP10B_C2.txt"


def stratum_dev(u: int) -> str:
    return "hard" if u <= 11112 else ("medium" if u <= 11124 else "easy")


def stratum_eval(u: int) -> str:
    return "hard" if u <= 11156 else ("medium" if u <= 11172 else "easy")


def stratum_of(u: int) -> str:
    return stratum_dev(u) if u < 11140 else stratum_eval(u)


def _job(user, t, stratum):
    return dict(user=user, start=0, end=t, fitseed=user, K=3, n_init=10, max_iter=60, stratum=stratum, drift=0.0, kchange=None)


def groups():
    return {"DEV-C2": [_job(u, t, stratum_dev(u)) for u in DEV_USERS for t in G_T],
            "EVAL-C2": [_job(u, t, stratum_eval(u)) for u in EVAL_USERS for t in G_T]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--only", required=True); ap.add_argument("--count", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    g = groups()
    if a.only not in g:
        sys.exit(f"unknown group {a.only!r}; known: {sorted(g)}")
    if not a.count and not a.limit and a.only == "EVAL-C2" and not Path(FREEZE).exists():
        sys.exit(f"the evaluation runs only after the design freeze file {FREEZE} is committed")
    jobs = g[a.only]
    if a.limit:
        if a.only == "EVAL-C2":
            sys.exit("--limit is for development only")
        jobs = jobs[:a.limit]
    if a.count:
        print(len(jobs), "fits")
        sys.exit(0)
    L.run_jobs(jobs, a.out, workers=a.workers)
