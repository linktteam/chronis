"""WP8 item 2 diagnosis: all fits (seeds 8101..8112; fit seeds = user seed, or user seed + 20000 for the second seed). Resumable: re-run the same command to continue.
usage: PYTHONPATH=. python3 integration_report/wp8/scripts/item2_run.py wp8_evidence/item2_fits.jsonl [--workers 2] [--only A,B,D,F,C]
Groups: A baseline protocol G (growing window 90..146 weekly plus 120 and 150, drift 0); B same windows 90 and 150 with a second fit seed; D dual (users 8101..8106; windows 90, 120, 150; plus 90 with a
second seed); F K change (split 8101..8106, merge 8107..8112; baseline); C protocol R (rolling 60-day windows every 15 days) with drift 1 and 2.5."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from identity_lib import run_jobs

USERS = list(range(8101, 8113))


def J(user, engine, start, end, fitseed, K=3, drift=0.0, kchange=None, group=""):
    return dict(user=user, engine=engine, start=start, end=end, fitseed=fitseed, K=K, drift=drift, kchange=kchange, group=group)


def jobs(only):
    out = []
    if "A" in only:
        out += [J(u, "baseline", 0, w, u, group="A") for u in USERS for w in sorted(set(range(90, 151, 7)) | {120, 150})]
    if "B" in only:
        out += [J(u, "baseline", 0, w, u + 20000, group="B") for u in USERS for w in (90, 150)]
    if "D" in only:
        out += [J(u, "dual", 0, w, u, group="D") for u in USERS[:6] for w in (90, 120, 150)] + [J(u, "dual", 0, 90, u + 20000, group="D") for u in USERS[:6]]
    if "F" in only:
        for i, u in enumerate(USERS):
            kind, k_after = ("split", 4) if i < 6 else ("merge", 2)
            out += [J(u, "baseline", 0, 90, u, 3, kchange=kind, group="F"), J(u, "baseline", 30, 150, u, 3, kchange=kind, group="F"), J(u, "baseline", 30, 150, u, k_after, kchange=kind, group="F")]
    if "C" in only:
        out += [J(u, "baseline", t - 60, t, u, drift=d, group="C") for d in (1.0, 2.5) for u in USERS for t in range(60, 151, 15)]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--only", default="A,B,D,F,C")
    a = ap.parse_args()
    run_jobs(jobs(set(a.only.split(","))), a.out, a.workers)
