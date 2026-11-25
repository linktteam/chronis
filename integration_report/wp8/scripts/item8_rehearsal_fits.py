"""WP8 Item 8 REHEARSAL fits on DEVELOPMENT seeds only (8201, 8202): K-change streams (criterion 6 plumbing), same windows and candidate K as the evaluation group B-K. No evaluation seed.
usage: PYTHONPATH=. python3 integration_report/wp8/scripts/item8_rehearsal_fits.py wp8_evidence/item8_rehearsal_fits.jsonl [--workers 2]"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from identity_lib import run_jobs

G_T = list(range(90, 147, 7)) + [150]
USERS = (8201, 8202)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    assert all(8201 <= u <= 8212 for u in USERS)                         # development seeds only
    jobs = [dict(user=u, engine="baseline", start=0, end=t, fitseed=u, K=[2, 3, 4], drift=0.0, kchange=kc, group="B-K") for u in USERS for kc in ("split", "merge") for t in G_T]
    print(len(jobs), "rehearsal jobs", flush=True)
    run_jobs(jobs, a.out, a.workers)
