"""WP10B B2 fits: the opt-in fixes on the development users of B1 (design: tests/PREDECLARED_CRITERIA_WP10B.md Part 4). Setting S0 of `dual_lib.py`; the same 36 users as B1 (11201..11236: 11201..11212 hard, 11213..11224 medium, 11225..11236 easy)
so that every variant is PAIRED with the S0 fit of `wp10b_evidence/b1_dev_fits.jsonl` (same seed, same data). Variants: D60 (Dmax 60), I600 (600 iterations and a 900 s wall-clock guard), REG (duration prior, 1 pseudo-segment), ALL (the three together).
The baseline engine and the single Gaussian are NOT refitted (the S0 records have them). usage: PYTHONPATH=. python3 integration_report/wp10b/scripts/b2_run.py OUT.jsonl --variants D60,I600,REG,ALL [--workers 2] [--count] [--limit N]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dual_lib as L                                                                  # noqa: E402

DEV_USERS = range(11201, 11237)
VARIANT_ORDER = ("I600", "REG", "D60", "ALL")


def stratum_dev(u: int) -> str:
    return "hard" if u <= 11212 else ("medium" if u <= 11224 else "easy")


def jobs_for(variants):
    return [dict(user=u, stratum=stratum_dev(u), variant=v, with_baseline=False) for v in variants for u in DEV_USERS]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--variants", default=",".join(VARIANT_ORDER)); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--count", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    vs = [v for v in a.variants.split(",") if v]
    if any(v not in L.VARIANTS or v == "S0" for v in vs):
        sys.exit("--variants must name D60, I600, REG and/or ALL")
    jobs = jobs_for(vs)
    if a.limit:
        jobs = jobs[:a.limit]
    if a.count:
        print(len(jobs), "fits")
        sys.exit(0)
    L.run_jobs(jobs, a.out, a.workers)
