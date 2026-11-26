"""A1 diagnosis fits (seeds 9201..9212, read-only study; plan in tests/PREDECLARED_CRITERIA_WP9.md section 4). Two fit seeds (user, user+1000) x windows [0,90) and [0,150) x 20 restarts.
usage: PYTHONPATH=. python3 integration_report/wp9/scripts/a1_diag_run.py wp9_evidence/a1_diag_fits.jsonl [--workers 2]   (resumable)"""
import sys
sys.path.insert(0, "integration_report/wp9/scripts")
import wp9_fit_lib as L

STRATUM = {**{u: "hard" for u in range(9201, 9205)}, **{u: "medium" for u in range(9205, 9209)}, **{u: "easy" for u in range(9209, 9213)}}


def jobs():
    out = []
    for u in range(9201, 9213):
        for end in (90, 150):
            for fs in (u, u + 1000):
                out.append(dict(user=u, start=0, end=end, fitseed=fs, K=3, n_init=20, max_iter=60, stratum=STRATUM[u], drift=0.0, kchange=None))
    return out


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 2
    L.run_jobs(jobs(), sys.argv[1], workers=workers)
