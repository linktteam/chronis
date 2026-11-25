"""WP8 item 4 development fits (seeds 8201..8212 only). Resumable. Design: tests/PREDECLARED_CRITERIA_WP8.md section 8, 'Pre-declaration (Item 4)'.
usage: PYTHONPATH=. python3 integration_report/wp8/scripts/item4_run.py wp8_evidence/item4_dev_fits.jsonl [--workers 2] [--only baseline,dual]
Protocol G: [0, t), t = 90, 97, ..., 146, 150. Protocol R: [t-60, t), t = 60, 70, ..., 150. Baseline: drifts 0, 1, 2.5 for every user. Dual (frozen: max_iter 200, one restart): one drift per user."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from identity_lib import run_jobs

USERS = list(range(8201, 8213))
G_T = list(range(90, 147, 7)) + [150]
R_T = list(range(60, 151, 10))
DUAL_KW = dict(max_iter=200, n_initializations=1)


def dual_drift(user):
    return (0.0, 1.0, 2.5)[(user - 8201) // 4]


def jobs(only):
    out = []
    def add(engine, user, drift, kw):
        for proto, wins in (("G", [(0, t) for t in G_T]), ("R", [(t - 60, t) for t in R_T])):
            for s, e in wins:
                j = dict(user=user, engine=engine, start=s, end=e, fitseed=user, K=3, drift=drift, kchange=None, group=f"{engine}-{proto}")
                if kw: j["fit_kw"] = kw
                out.append(j)
    if "dual" in only:                                  # dual first: it is the slow part and the engine-switch pairs need it
        for u in USERS: add("dual", u, dual_drift(u), DUAL_KW)
    if "baseline" in only:
        for d in (0.0, 1.0, 2.5):
            for u in USERS: add("baseline", u, d, None)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--only", default="baseline,dual")
    a = ap.parse_args()
    js = jobs(set(a.only.split(",")))
    print(f"{len(js)} jobs", flush=True)
    run_jobs(js, a.out, a.workers)
