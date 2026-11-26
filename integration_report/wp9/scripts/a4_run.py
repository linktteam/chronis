"""WP9 A4: fits for the negative control that can fail (design: tests/PREDECLARED_CRITERIA_WP9.md Part 6, section 18). usage:
  PYTHONPATH=. python3 integration_report/wp9/scripts/a4_run.py --range dev|eval --out FILE.jsonl [--workers 2] [--designs A,B,C,none]
dev = users 9261..9276 (all designs), eval = users 9281..9296 (refused without tests/DESIGN_FREEZE_WP9_A4.txt: `{"design": "A"}`; only that design and `none` are run).
Windows of 45 days ending at 45, 60, ..., 150; K = 3, 10 restarts, easy stratum. Windows ending at 45 and 60 contain no changed day and are shared by all designs."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wp9_fit_lib as W                                                   # noqa: E402

RANGES = {"dev": range(9261, 9277), "eval": range(9281, 9297)}
ENDS = (45, 60, 75, 90, 105, 120, 135, 150)
DESIGNS = {"A": dict(features=(5, 6), factor=0.1, start=60), "B": dict(features=(3, 4, 5, 6), factor=0.1, start=60), "C": dict(features=(0, 1, 2), factor=4.0, start=60), "none": None}


def jobs_for(users, designs):
    out = []
    for u in users:
        for e in ENDS:
            for d in designs:
                if e <= 60 and d != designs[0]:
                    continue                                                   # shared windows: run once
                sc = None if e <= 60 else DESIGNS[d]
                out.append(dict(user=u, start=e - 45, end=e, fitseed=u, K=3, n_init=10, max_iter=60, stratum="easy", drift=0.0, kchange=None, scale_change=sc,
                                group="A4-shared" if e <= 60 else f"A4-{d}"))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--range", required=True, choices=list(RANGES)); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--designs", default=None)
    a = ap.parse_args()
    designs = a.designs.split(",") if a.designs else ["A", "B", "C", "none"]
    if a.range == "eval":
        fz = Path("tests/DESIGN_FREEZE_WP9_A4.txt")
        if not fz.exists():
            sys.exit("the evaluation runs only after tests/DESIGN_FREEZE_WP9_A4.txt is committed")
        chosen = json.loads(fz.read_text().splitlines()[0])["design"]
        designs = [chosen, "none"] if chosen != "none" else ["none"]
    W.run_jobs(jobs_for(list(RANGES[a.range]), designs), a.out, workers=a.workers)


if __name__ == "__main__":
    main()
