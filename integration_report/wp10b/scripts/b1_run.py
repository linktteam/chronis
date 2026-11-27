"""WP10B B1 fits: why dual fits do not publish (design: tests/PREDECLARED_CRITERIA_WP10B.md Part 3). Setting S0 of `dual_lib.py` (identity streams, K = 3, F = 7, train days [0, 120), test days [120, 150), the production defaults
except the fast-test switch and one restart). Groups (users, fits): DEV-B1 11201..11236 (11201..11212 hard, 11213..11224 medium, 11225..11236 easy), variant S0 = 36 dual fits + 36 baseline fits.
usage: PYTHONPATH=. python3 integration_report/wp10b/scripts/b1_run.py OUT.jsonl [--workers 2] [--count] [--limit N]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dual_lib as L                                                                  # noqa: E402

DEV_USERS = range(11201, 11237)


def stratum_dev(u: int) -> str:
    return "hard" if u <= 11212 else ("medium" if u <= 11224 else "easy")


def groups():
    return {"DEV-B1": [dict(user=u, stratum=stratum_dev(u), variant="S0") for u in DEV_USERS]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--count", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    jobs = groups()["DEV-B1"]
    if a.limit:
        jobs = jobs[:a.limit]
    if a.count:
        print(len(jobs), "fits")
        sys.exit(0)
    L.run_jobs(jobs, a.out, a.workers)
