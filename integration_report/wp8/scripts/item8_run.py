"""WP8 Item 8 benchmark fits (EVALUATION seeds 8301..8324, 8401..8424, 8501..8512, 8601..8612). Run ONCE, after the Item 4 design freeze is committed. Resumable (a finished job is skipped).
Design: tests/PREDECLARED_CRITERIA_WP8.md section 5 and the dated section 'Item 8 run plan'.
usage: PYTHONPATH=. python3 integration_report/wp8/scripts/item8_run.py wp8_evidence/item8_fits.jsonl [--workers 2] [--only B-G,B-G2,...] [--count]
Groups (engine, users, windows, number of fits):
  B-G      baseline, 8301..8324, drift 0, windows [0, t) t = 90, 97, ..., 146, 150                        240  (criteria 1, 2, 4, L, 8 weekly and 14-day)
  B-G2     the same windows with fit seed user + 1000                                                      240  (criterion 3)
  D-G      dual (max_iter 200, one restart), 8301..8312, the same windows                                  120  (dual criteria; compute)
  B-cad    baseline, 8301..8312, every other day t = 90..150 not in B-G                                    612  (criterion 8 daily)
  B-R1     baseline, 8401..8424, drift 1.0, protocol R [t-60, t) t = 60, 67, ..., 150                      336  (criterion 5)
  B-R25    the same with drift 2.5                                                                          336
  D-R1     dual, 8401..8412, drift 1.0, protocol R at t = 60, 81, 102, 123, 144 (every third week; compute) 60
  D-R25    the same with drift 2.5                                                                           60
  B-K      baseline, 8501..8512, k_change split and merge (change day 90), candidate K (2, 3, 4), windows [0, t) t = 90, 97, ..., 146, 150   240  (criterion 6)
  X-B      baseline, 8601..8612, drift 0, windows [0, t) t = 90, 120, 150                                    36  (criterion 7)
  X-D      dual, the same windows                                                                            36
Fit seed = user for every group but B-G2."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from identity_lib import run_jobs

G_T = list(range(90, 147, 7)) + [150]
R_T = list(range(60, 145, 7)) + [150]
R_T_DUAL = [60, 81, 102, 123, 144]
X_T = [90, 120, 150]
DUAL_KW = dict(max_iter=200, n_initializations=1)          # frozen in the Item 3 design freeze


def _job(group, engine, user, s, e, drift=0.0, kchange=None, K=3, fitseed=None, kw=None):
    j = dict(user=user, engine=engine, start=s, end=e, fitseed=user if fitseed is None else fitseed, K=K, drift=drift, kchange=kchange, group=group)
    if kw:
        j["fit_kw"] = kw
    return j


def groups():
    g = {}
    g["B-G"] = [_job("B-G", "baseline", u, 0, t) for u in range(8301, 8325) for t in G_T]
    g["B-G2"] = [_job("B-G2", "baseline", u, 0, t, fitseed=u + 1000) for u in range(8301, 8325) for t in G_T]
    g["D-G"] = [_job("D-G", "dual", u, 0, t, kw=DUAL_KW) for u in range(8301, 8313) for t in G_T]
    g["B-cad"] = [_job("B-cad", "baseline", u, 0, t) for u in range(8301, 8313) for t in range(90, 151) if t not in G_T]
    for tag, d in (("R1", 1.0), ("R25", 2.5)):
        g[f"B-{tag}"] = [_job(f"B-{tag}", "baseline", u, t - 60, t, drift=d) for u in range(8401, 8425) for t in R_T]
        g[f"D-{tag}"] = [_job(f"D-{tag}", "dual", u, t - 60, t, drift=d, kw=DUAL_KW) for u in range(8401, 8413) for t in R_T_DUAL]
    g["B-K"] = [_job("B-K", "baseline", u, 0, t, kchange=kc, K=[2, 3, 4]) for u in range(8501, 8513) for kc in ("split", "merge") for t in G_T]
    g["X-B"] = [_job("X-B", "baseline", u, 0, t) for u in range(8601, 8613) for t in X_T]
    g["X-D"] = [_job("X-D", "dual", u, 0, t, kw=DUAL_KW) for u in range(8601, 8613) for t in X_T]
    return g


ORDER = ["B-G", "D-G", "X-B", "X-D", "B-G2", "B-K", "B-R1", "B-R25", "D-R1", "D-R25", "B-cad"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--only", default=",".join(ORDER)); ap.add_argument("--count", action="store_true")
    a = ap.parse_args()
    g = groups()
    names = [n for n in a.only.split(",") if n]
    unknown = [n for n in names if n not in g]
    if unknown:
        sys.exit(f"unknown groups {unknown}; known: {sorted(g)}")
    js = [j for n in ORDER if n in names for j in g[n]]
    print({n: len(g[n]) for n in names}, "total", len(js), flush=True)
    if not a.count:
        run_jobs(js, a.out, a.workers)
