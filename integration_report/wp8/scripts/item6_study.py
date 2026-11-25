"""WP8 Item 6 cadence study on DEVELOPMENT seeds 8201..8204 (the evaluation version is criterion 8 of Item 8). The same 150-day baseline streams (drift 0, protocol G windows [0, t))
are refitted daily, weekly and every 14 days; every schedule is replayed through the production lineage with the frozen defaults and scored against planted truth.
  run:     PYTHONPATH=. python3 integration_report/wp8/scripts/item6_study.py run     wp8_evidence/item6_dev_fits.jsonl [--workers 2]   (resumable; skips fits that item4_dev_fits.jsonl already has)
  analyze: PYTHONPATH=. python3 integration_report/wp8/scripts/item6_study.py analyze wp8_evidence/item6_analysis.txt
The dual engine is not refitted daily (about 60 s per fit); this study is baseline only, as declared in the Item 3 design freeze."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from identity_lib import job_key, load_records, run_jobs
import lineage_study_lib as S

USERS = [8201, 8202, 8203, 8204]
FIRST, LAST = 90, 150
ITEM4_FITS = "wp8_evidence/item4_dev_fits.jsonl"
INTERVALS = (1, 7, 14)


def all_jobs():
    return [dict(user=u, engine="baseline", start=0, end=t, fitseed=u, K=3, drift=0.0, kchange=None, group="baseline-G") for u in USERS for t in range(FIRST, LAST + 1)]


def todo_jobs():
    have = set()
    if Path(ITEM4_FITS).exists():
        for l in open(ITEM4_FITS):
            try: have.add(job_key(json.loads(l)["job"]))
            except Exception: pass
    return [j for j in all_jobs() if job_key(j) not in have]


def pooled(paths):
    out = {}
    for p in paths:
        if Path(p).exists():
            for r in load_records(p):
                j = r["job"]
                if j["engine"] == "baseline" and j["user"] in USERS and j["start"] == 0 and j["drift"] == 0.0 and j.get("kchange") is None and j["group"] == "baseline-G":
                    out[(j["user"], j["end"])] = r
    return out


def analyze(out_path):
    L = []
    def P(*a):
        s = " ".join(str(x) for x in a); L.append(s); print(s)
    recs = pooled([ITEM4_FITS, "wp8_evidence/item6_dev_fits.jsonl"])
    P("WP8 Item 6 cadence study, development seeds 8201..8204, baseline, drift 0, windows [0, t), K = 3. Production lineage with the defaults at the time of this run.")
    from backbone.hssm.lineage import AlignmentConfig
    cfg = AlignmentConfig()
    P("config:", cfg)
    sched = {i: S.schedule_days(i, FIRST, LAST) for i in INTERVALS}
    P("schedules (window ends):", {i: (len(d), d[:4] + ["..."] + d[-3:]) for i, d in sched.items()})
    rows_by, ids150, n_fits = {i: [] for i in INTERVALS}, {}, {i: 0 for i in INTERVALS}
    agree = []
    for u in USERS:
        mine = [r for (uu, t), r in sorted(recs.items()) if uu == u]
        ends = {r["job"]["end"] for r in mine}
        need = set(sched[1])
        if not need <= ends:
            P(f"user {u}: {len(need - ends)} of the daily fits are missing: run `item6_study.py run` first"); continue
        ids = {}
        for i in INTERVALS:
            fits = S.fits_for_days(mine, sched[i])
            rows, lrs, _ = S.replay(fits, cfg)
            rows_by[i] += rows
            n_fits[i] += len(rows)
            ids[i] = lrs[-1].stable_regime_id
            P(f"  user {u} interval {i:2d}: fed {len(rows)} accepted fits, states", "".join(r["state"][0] for r in rows), "last window", rows[-1]["window"], "last state", rows[-1]["state"])
        pairs = {"daily-weekly": (1, 7), "daily-14d": (1, 14), "weekly-14d": (7, 14)}
        ag = {k: S.day_agreement(ids[a], ids[b]) for k, (a, b) in pairs.items()}
        agree.append((u, ag))
        P(f"  user {u} day-150 stable-id agreement: " + ", ".join(f"{k} {v:.3f}" for k, v in ag.items()))
    P("")
    names = {1: "daily", 7: "weekly", 14: "every 14 days"}
    for i in INTERVALS:
        t = S.tally(rows_by[i])
        P(f"{names[i]:14s} fits fed {t['steps']}, alignments {t['alignments']}, served (ALIGNED) {S.rate(t['served'], t['alignments'])}, served and correct {S.rate(t['served_correct'], t['served'])}, "
          f"wrong-and-unflagged {S.rate(t['wrong_unflagged'], t['alignments'])}, flagged {t['flagged']} (would have been wrong {t['flagged_would_be_wrong']}, right {t['flagged_would_be_right']})")
    if agree:
        for k in ("daily-weekly", "daily-14d", "weekly-14d"):
            vals = [a[k] for _, a in agree]
            P(f"day-150 agreement {k}: min {min(vals):.3f} mean {sum(vals) / len(vals):.3f}; users at or above 0.95: {sum(v >= 0.95 for v in vals)}/{len(vals)}")
    P("Counts are of fits and alignments of 4 users; the windows of one user are not independent. Development data: no claim beyond these 4 streams.")
    Path(out_path).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["run", "analyze"]); ap.add_argument("path"); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.mode == "run":
        js = todo_jobs()
        print(f"{len(js)} new daily fits needed", flush=True)
        run_jobs(js, a.path, a.workers)
    else:
        analyze(a.path)
