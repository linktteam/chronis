"""WP10A A4 fits for designs D and E (design: tests/PREDECLARED_CRITERIA_WP10A.md Part 2, sections 5-7; design F needs no new fit, it replays the A3 fits). usage:
  PYTHONPATH=. python3 integration_report/wp10a/scripts/a4_run.py --range dev|eval --out FILE.jsonl [--workers 2] [--designs D,E] [--count]
dev = users 10301..10316 (designs D and E on the same users); eval = users 10321..10336, refused without tests/DESIGN_FREEZE_WP10A_A4.txt (first line `{"design": "D"}` or "E" or "F"); only the chosen design is run
(D also runs its no-change comparison; E has none; F has nothing to run: its evaluation replays the A3 evaluation fits).
Common: easy stratum, K = 3, 150-day streams, 10 restarts, 60 iterations, fit seed = user.
D: fresh 45-day windows ending at 45, 60, ..., 150 (groups A4D-shared for the windows that end at 45 and 60, which contain no changed day, A4D-D with the planted change, A4D-none without), 14 fits per user.
E: anchor window [0, 90) (group A4E-anchor) and short windows [t - L, t), t in 105, 120, 135, 150, L in 20, 30 (groups A4E-L20, A4E-L30), 9 fits per user."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp9" / "scripts"))
import wp9_fit_lib as W                                                            # noqa: E402

RANGES = {"dev": range(10301, 10317), "eval": range(10321, 10337)}
ENDS = (45, 60, 75, 90, 105, 120, 135, 150)
D_CHANGE = dict(regime=2, features=[0, 1, 2, 3], factor=3.0, start=60)
E_T, E_L = (105, 120, 135, 150), (20, 30)


def _job(user, start, end, group, regime_change=None):
    return dict(user=user, start=start, end=end, fitseed=user, K=3, n_init=10, max_iter=60, stratum="easy", drift=0.0, kchange=None, scale_change=None, regime_change=regime_change, group=group)


def jobs_d(users):
    out = []
    for u in users:
        for e in ENDS:
            if e <= 60:
                out.append(_job(u, e - 45, e, "A4D-shared"))
            else:
                out.append(_job(u, e - 45, e, "A4D-D", D_CHANGE))
                out.append(_job(u, e - 45, e, "A4D-none"))
    return out


def jobs_e(users):
    out = []
    for u in users:
        out.append(_job(u, 0, 90, "A4E-anchor"))
        for t in E_T:
            for L in E_L:
                out.append(_job(u, t - L, t, f"A4E-L{L}"))
    return out


def jobs_for(users, designs):
    out = []
    if "D" in designs:
        out += jobs_d(users)
    if "E" in designs:
        out += jobs_e(users)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", required=True, choices=list(RANGES)); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--designs", default="D,E"); ap.add_argument("--count", action="store_true")
    a = ap.parse_args()
    designs = a.designs.split(",")
    if a.range == "eval":
        fz = Path("tests/DESIGN_FREEZE_WP10A_A4.txt")
        if not fz.exists():
            sys.exit("the evaluation runs only after tests/DESIGN_FREEZE_WP10A_A4.txt is committed")
        chosen = json.loads(fz.read_text().splitlines()[0])["design"]
        if chosen == "F":
            sys.exit("design F is replay-only: nothing to fit (its evaluation replays the A3 evaluation fits)")
        designs = [chosen]
    jobs = jobs_for(list(RANGES[a.range]), designs)
    if a.count:
        print(len(jobs), "fits")
        return
    W.run_jobs(jobs, a.out, workers=a.workers)


if __name__ == "__main__":
    main()
