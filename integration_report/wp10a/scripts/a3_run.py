"""WP10A A3 fits (design: tests/PREDECLARED_CRITERIA_WP10A.md section 3). Baseline engine, K = 3, 60 EM iterations, 10 restarts, every restart kept (`wp9_fit_lib`). Resumable (a finished job is skipped).
Groups (users, windows, fit seed, number of fits):
  DEV-A    10001..10048 (10001..10016 hard, 10017..10032 medium, 10033..10048 easy), weekly t = 90, 97, ..., 146, 150, fit seed = user      480
  DEV-B    the same users, t = 90, 104, 118, 132, 146, 150, fit seed = user + 1000                                                          288
  EVAL-A   10101..10148 (same split), weekly, fit seed = user                                                                               480   (after the freeze file)
  EVAL-B   the same users, 14-day, fit seed = user + 1000                                                                                   288   (after the freeze file)
  SAFE-A   10151..10210 (10151..10170 hard, 10171..10190 medium, 10191..10210 easy), weekly, fit seed = user                                600   (after the freeze file)
  SAFE-B   the same users, 14-day, fit seed = user + 1000                                                                                   360   (after the freeze file)
usage: PYTHONPATH=. python3 integration_report/wp10a/scripts/a3_run.py OUT.jsonl --only DEV-A,DEV-B [--workers 2] [--count] [--limit N]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp9" / "scripts"))
import wp9_fit_lib as L                                                               # noqa: E402

G_T = list(range(90, 147, 7)) + [150]
B_T = [90, 104, 118, 132, 146, 150]
DEV_USERS, EVAL_USERS, SAFE_USERS = range(10001, 10049), range(10101, 10149), range(10151, 10211)


def stratum_dev(u):
    return "hard" if u <= 10016 else ("medium" if u <= 10032 else "easy")


def stratum_eval(u):
    return "hard" if u <= 10116 else ("medium" if u <= 10132 else "easy")


def stratum_safe(u):
    return "hard" if u <= 10170 else ("medium" if u <= 10190 else "easy")


def stratum_of(u):
    return stratum_dev(u) if u < 10100 else (stratum_eval(u) if u < 10150 else stratum_safe(u))


def _job(user, t, fitseed, stratum):
    return dict(user=user, start=0, end=t, fitseed=fitseed, K=3, n_init=10, max_iter=60, stratum=stratum, drift=0.0, kchange=None)


def groups():
    g = {}
    for name, users, st in (("DEV", DEV_USERS, stratum_dev), ("EVAL", EVAL_USERS, stratum_eval), ("SAFE", SAFE_USERS, stratum_safe)):
        g[name + "-A"] = [_job(u, t, u, st(u)) for u in users for t in G_T]
        g[name + "-B"] = [_job(u, t, u + 1000, st(u)) for u in users for t in B_T]
    return g


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--only", required=True); ap.add_argument("--count", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    g = groups()
    names = [n for n in a.only.split(",") if n]
    unknown = [n for n in names if n not in g]
    if unknown:
        sys.exit(f"unknown groups {unknown}; known: {sorted(g)}")
    if not a.count and not a.limit and any(not n.startswith("DEV") for n in names) and not Path("tests/DESIGN_FREEZE_WP10A_A3.txt").exists():
        sys.exit("EVAL and SAFE groups run only after the design freeze file tests/DESIGN_FREEZE_WP10A_A3.txt is committed")
    if a.limit and any(not n.startswith("DEV") for n in names):
        sys.exit("--limit (timing pilot) is for DEV groups only")
    js = [j for n in names for j in g[n]]
    if a.limit:
        js = js[:a.limit]
    print({n: len(g[n]) for n in names}, "total", len(js), flush=True)
    if not a.count:
        L.run_jobs(js, a.out, a.workers)
