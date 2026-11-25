"""WP8 item 3 development fits (seeds 8201..8212 only; never the evaluation seeds). Resumable.
usage: PYTHONPATH=. python3 integration_report/wp8/scripts/item3_run.py wp8_evidence/item3_dev_fits.jsonl [--workers 2] [--only B,D]
B: baseline fits of the 12 development users on windows (0,90), (0,120), (0,150), (30,90), (60,120).
D: dual settings check (users 8201..8204, window (0,90)): max_iter 200 with 1 and 2 restarts, to see whether any dual fit can be accepted and what it costs."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from identity_lib import run_jobs

USERS = list(range(8201, 8213))


def J(user, engine, start, end, fitseed, K=3, group="", fit_kw=None):
    j = dict(user=user, engine=engine, start=start, end=end, fitseed=fitseed, K=K, drift=0.0, kchange=None, group=group)
    if fit_kw: j["fit_kw"] = fit_kw
    return j


def jobs(only):
    out = []
    if "B" in only:
        out += [J(u, "baseline", s, e, u, group="B") for u in USERS for (s, e) in ((0, 90), (0, 120), (0, 150), (30, 90), (60, 120))]
    if "D" in only:
        out += [J(u, "dual", 0, 90, u, group="D", fit_kw=dict(max_iter=200, n_initializations=n)) for u in USERS[:4] for n in (1, 2)]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--only", default="B,D")
    a = ap.parse_args()
    run_jobs(jobs(set(a.only.split(","))), a.out, a.workers)
