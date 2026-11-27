"""WP10B replay of the A3 safety demonstration on a SUBSET, for opa to run on any machine (an AMD EPYC 7763, an AVX-512 runner, ...). Design and the declared failure outcomes: tests/PREDECLARED_CRITERIA_WP10B.md section 14.
What it does: for 12 users per stratum (hard, medium, easy; the first 12 of the SAFE-EXTRA users 10151..10162, 10171..10182, 10191..10202), single seed (the weekly seed-A chain: windows [0, t) for t = 90, 97, ..., 146, 150, fit seed = user,
10 fits per user), it FITS every window on THIS machine (baseline engine, K = 3, 60 EM iterations, the production 10 restarts, `wp9_fit_lib.fit_job`: the same function that made the stored WP10A fits), replays the chain through the
PRODUCTION lineage at the chosen operating point of tests/DESIGN_FREEZE_WP10A_A3.txt (consensus flag ON, sharpness floor 0.95, gate h = 2.5) and counts, as WP10A did, the gate-eligible alignments, the served ones and the
wrong-and-unflagged ones (a served alignment whose names are wrong against the planted truth). It also compares every fit with the STORED WP10A fit of the same job (made on the development machine; `--stored`), field by field in the
DISCRETE outputs (state, collapse flag and iteration count of every restart, the labels of the selected restart, the lineage state of the window) and reports the largest relative difference of the restart log-likelihoods.
One line per finished user goes to OUT/users.jsonl, one line per fit to OUT/fits.jsonl (crash-safe, resumable: a finished fit and a finished user are skipped), and the running totals are printed after every user.
usage:
  PYTHONPATH=. python integration_report/wp10b/scripts/a3_safety_replay.py run --out DIR [--users-per-stratum 12] [--workers 2] [--stored wp10a_evidence/a3_eval_fits.jsonl]
  PYTHONPATH=. python integration_report/wp10b/scripts/a3_safety_replay.py summary --out DIR           (the totals of what exists so far; marked PARTIAL until every user is done)
  PYTHONPATH=. python integration_report/wp10b/scripts/a3_safety_replay.py reference                  (the same numbers from the STORED fits alone, no fitting, seconds; used before the run to state the reference)
Time (RUN estimate, from the stored fits of these 36 users: 360 fits, 5931 s of fit time in total, mean 16.5 s, median 14.7 s, maximum 53.5 s on the development 2-core machine): about 99 minutes of CPU time, so about 99 / workers
minutes of wall clock on a quiet machine (two workers: about 50 minutes; four: about 25). Stop with Ctrl-C at any time and start the same command again: it continues."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")                                  # one BLAS thread per worker: no oversubscription, the same setting for every machine
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parents[1] / "wp9" / "scripts"))

G_T = list(range(90, 147, 7)) + [150]                                               # the weekly seed-A windows of WP10A A3
STRATUM_USERS = {"hard": 10151, "medium": 10171, "easy": 10191}                      # first SAFE-EXTRA user of each stratum (20 per stratum exist)
FREEZE = ROOT / "tests" / "DESIGN_FREEZE_WP10A_A3.txt"
OPERATING_POINT = (2.5, True, 0.95)                                                 # gate h, consensus flag ON, sharpness floor: the chosen point of WP10A A3 (checked against the freeze file)
STORED_DEFAULT = ROOT / "wp10a_evidence" / "a3_eval_fits.jsonl"
SAFETY_BOUND = 0.01
LL_REPORT = 1e-6                                                                    # restart log-likelihood differences above this relative size are listed (reported; no verdict)


# --------------------------------------------------------------------------------------------------------------------------------- pure helpers (tested)
def users_for(per_stratum: int) -> list:
    """[(user, stratum)] in round-robin order over the strata (hard, medium, easy, hard, ...), so partial results cover every stratum early."""
    if not 1 <= per_stratum <= 20:
        raise ValueError("users-per-stratum must be between 1 and 20 (SAFE-EXTRA has 20 users per stratum)")
    return [(first + i, st) for i in range(per_stratum) for st, first in STRATUM_USERS.items()]


def job_for(user: int, stratum: str, t: int) -> dict:
    """The WP10A A3 job (same fields as `a3_run._job`): window [0, t), fit seed = user, K = 3, 10 restarts, 60 iterations."""
    return dict(user=user, start=0, end=t, fitseed=user, K=3, n_init=10, max_iter=60, stratum=stratum, drift=0.0, kchange=None)


def check_operating_point(text: str) -> tuple:
    """(h, flag, sharpness) from the first line of the freeze file; the replay refuses any other point (it is the demonstration at the CHOSEN point)."""
    d = json.loads(text.splitlines()[0])
    point = (float(d["h"]), bool(d["u"]), float(d["s"]))
    if point != OPERATING_POINT:
        raise SystemExit(f"the freeze file says {point}, this replay is defined at {OPERATING_POINT}")
    return point


def exact_upper(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided exact (Clopper-Pearson) upper bound for k events in n trials; k = 0 gives 1 - (1 - conf)^(1/n)."""
    if n <= 0:
        return float("nan")
    if k == 0:
        return 1.0 - (1.0 - conf) ** (1.0 / n)
    from scipy.stats import beta
    return float(beta.ppf(conf, k + 1, n - k)) if k < n else 1.0


def discrete_view(rec: dict, n: int = 10) -> dict:
    """What must be identical between two machines: whether the fit worked, the state / collapse flag / iteration count of every restart, and the selected restart's index and labels."""
    if not rec.get("ok"):
        return dict(ok=False)
    import wp9_fit_lib as L
    rs = rec["restarts"]
    sel = L.select_restart(rs, n)
    return dict(ok=True, restarts=[(r.get("state"), bool(r.get("collapsed")), r.get("n_iter")) for r in rs], selected=None if sel is None else (sel.get("i"), sel.get("labels")))


def max_rel_ll(a: dict, b: dict) -> float:
    """Largest relative difference of the final log-likelihood over the restarts of two records of the same job (0.0 when there is nothing to compare)."""
    if not (a.get("ok") and b.get("ok")) or len(a["restarts"]) != len(b["restarts"]):
        return 0.0
    worst = 0.0
    for x, y in zip(a["restarts"], b["restarts"]):
        lx, ly = x.get("ll"), y.get("ll")
        if lx is None or ly is None:
            continue
        worst = max(worst, abs(lx - ly) / max(abs(lx), abs(ly), 1e-300))
    return worst


def compare_fit(new: dict, stored: dict | None) -> dict:
    """The comparison of one fit with the stored one: identical discrete view, the list of what differs and the largest relative log-likelihood difference."""
    if stored is None:
        return dict(compared=False, identical=None, diffs=[], max_rel_ll=None)
    a, b = discrete_view(new), discrete_view(stored)
    diffs = []
    if a.get("ok") != b.get("ok"):
        diffs.append("ok")
    elif a.get("ok"):
        if a["restarts"] != b["restarts"]:
            diffs.append("restart_states")
        if a["selected"] != b["selected"]:
            diffs.append("selected_restart")
    return dict(compared=True, identical=not diffs, diffs=diffs, max_rel_ll=max_rel_ll(new, stored))


def verdict_lines(totals: dict, complete: bool) -> list:
    """The declared outcome words (criteria 14.3). `totals`: users, users_expected, alignments, served, wrong_unflagged, compared, identical, state_diffs, max_rel_ll."""
    out = []
    tag = "" if complete else "PARTIAL (not every user is done): "
    n, w = totals["alignments"], totals["wrong_unflagged"]
    bound = exact_upper(w, n)
    if w > 0:
        out.append(f"{tag}SAFETY FAILURE: {w} wrong-and-unflagged alignment(s) in {n} (the WP10A demonstration had 0 in 972)")
    else:
        out.append(f"{tag}SAFETY: no wrong-and-unflagged alignment in {n}; one-sided exact 95% upper bound {100 * bound:.2f}% ({'at most' if bound <= SAFETY_BOUND else 'ABOVE'} 1%; a transfer check on a subset, not a new per-stratum demonstration)")
    if totals["compared"] == 0:
        out.append(f"{tag}REPRODUCTION: not compared (no stored fits)")
    elif totals["identical"] == totals["compared"] and totals["state_diffs"] == 0:
        out.append(f"{tag}REPRODUCTION: the discrete outcomes of all {totals['compared']} fits and the lineage state of every window are identical to the stored WP10A fits; largest relative log-likelihood difference {totals['max_rel_ll']:.3g}")
    else:
        out.append(f"{tag}REPRODUCTION FAILURE: {totals['compared'] - totals['identical']} of {totals['compared']} fits differ in a discrete outcome and {totals['state_diffs']} window(s) differ in the lineage state; largest relative log-likelihood difference {totals['max_rel_ll']:.3g}")
    return out


def sum_users(lines: list, users_expected: int) -> dict:
    t = dict(users=len(lines), users_expected=users_expected, alignments=0, served=0, wrong_unflagged=0, compared=0, identical=0, state_diffs=0, max_rel_ll=0.0, refits=0)
    for d in lines:
        t["alignments"] += d["alignments"]; t["served"] += d["served"]; t["wrong_unflagged"] += d["wrong_unflagged"]; t["refits"] += d["refits"]
        t["compared"] += d["compared"]; t["identical"] += d["identical"]; t["state_diffs"] += d["state_diffs"]; t["max_rel_ll"] = max(t["max_rel_ll"], d["max_rel_ll"])
    return t


# --------------------------------------------------------------------------------------------------------------------------------- the work
def load_stored(path) -> dict:
    """{(user, end): record} of the seed-A weekly fits in a stored WP10A file ({} when the file does not exist)."""
    out = {}
    if path and Path(path).exists():
        for ln in open(path):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            j = r["job"]
            if j["fitseed"] == j["user"] and j["end"] in G_T:
                out[(j["user"], j["end"])] = r
    return out


def replay_user(records: list, stored: dict, h: float, flag: bool, sharp: float) -> dict:
    """The production lineage over the chain of one user's records, the WP8 tally, and the comparison with the stored fits (and with the stored replay's lineage states)."""
    import a12_analyze as A
    recs = sorted(records, key=lambda r: r["job"]["end"])
    user, stratum = recs[0]["job"]["user"], recs[0]["job"].get("stratum")
    res = A.run_candidate({(user, user): recs}, h, sharp, flag, 10)
    t = A.tally_rows(res)
    rows, _ = res[(user, user)]
    cmps = [compare_fit(r, stored.get((user, r["job"]["end"]))) for r in recs]
    state_diffs = 0
    if all((user, r["job"]["end"]) in stored for r in recs):
        sres = A.run_candidate({(user, user): [stored[(user, r["job"]["end"])] for r in recs]}, h, sharp, flag, 10)
        srows, _ = sres[(user, user)]
        state_diffs = sum(1 for a, b in zip(rows, srows) if (a["state"], a["served"], a["correct"]) != (b["state"], b["served"], b["correct"])) + abs(len(rows) - len(srows))
    return dict(user=user, stratum=stratum, fits=len(recs), fits_ok=sum(1 for r in recs if r.get("ok")), refits=t["refits"], alignments=t["alignments"], served=t["served"], wrong_unflagged=t["wrong_unflagged"],
                states=[r["state"] for r in rows], compared=sum(1 for c in cmps if c["compared"]), identical=sum(1 for c in cmps if c["identical"]), state_diffs=state_diffs,
                diffs=[(r["job"]["end"], c["diffs"]) for r, c in zip(recs, cmps) if c["diffs"]], max_rel_ll=max([c["max_rel_ll"] or 0.0 for c in cmps] + [0.0]),
                rel_ll_listed=[(r["job"]["end"], round(c["max_rel_ll"], 12)) for r, c in zip(recs, cmps) if (c["max_rel_ll"] or 0.0) > LL_REPORT])


def user_task(args):
    user, stratum, done_fits, stored_path, out_dir = args
    import wp9_fit_lib as L
    h, flag, sharp = OPERATING_POINT
    stored = load_stored(stored_path)
    recs = []
    for t in G_T:
        job = job_for(user, stratum, t)
        key = L.job_key(job)
        rec = done_fits.get(key)
        if rec is None:
            rec = L.fit_job(job)
            with open(Path(out_dir) / "fits.jsonl", "a") as fh:                    # one line per fit, flushed: a stopped run loses at most the fit in progress
                fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())
        recs.append(rec)
    return replay_user(recs, stored, h, flag, sharp)


def machine_info() -> dict:
    flags = []
    try:
        for ln in open("/proc/cpuinfo"):
            if ln.startswith("flags"):
                flags = sorted({f for f in ln.split(":", 1)[1].split() if f.startswith(("avx", "fma", "sse4"))})
                break
    except OSError:
        pass
    info = dict(platform=platform.platform(), machine=platform.machine(), processor=platform.processor(), python=platform.python_version(), cpu_flags=flags)
    try:
        import numpy
        import scipy
        info.update(numpy=numpy.__version__, scipy=scipy.__version__)
        from threadpoolctl import threadpool_info
        info["blas"] = [dict(api=d["internal_api"], arch=d.get("architecture"), threads=d["num_threads"]) for d in threadpool_info()]
    except Exception as exc:                                                         # noqa: BLE001  informational only
        info["info_error"] = type(exc).__name__
    return info


def read_jsonl(path) -> list:
    out = []
    if Path(path).exists():
        for ln in open(path):
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    return out


def print_totals(lines: list, expected: int):
    t = sum_users(lines, expected)
    by = {}
    for d in lines:
        by.setdefault(d["stratum"], []).append(d)
    print(f"  running: users {t['users']}/{expected}; alignments {t['alignments']}, served {t['served']}, wrong-and-unflagged {t['wrong_unflagged']}; fits identical to stored {t['identical']}/{t['compared']}, lineage-state differences {t['state_diffs']}", flush=True)
    return t


def cmd_run(a):
    h, flag, sharp = check_operating_point(FREEZE.read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "env.json").write_text(json.dumps(machine_info(), indent=1))
    users = users_for(a.users_per_stratum)
    done_users = {d["user"] for d in read_jsonl(out / "users.jsonl")}
    import wp9_fit_lib as L
    done_fits = {L.job_key(r["job"]): r for r in read_jsonl(out / "fits.jsonl") if r.get("ok")}
    todo = [(u, st, {L.job_key(job_for(u, st, t)): done_fits[L.job_key(job_for(u, st, t))] for t in G_T if L.job_key(job_for(u, st, t)) in done_fits}, a.stored, str(out)) for u, st in users if u not in done_users]
    stored_n = len(load_stored(a.stored))
    print(f"machine: {json.loads((out / 'env.json').read_text()).get('blas')}; operating point h={h}, flag ON, sharpness {sharp}; {len(users)} users ({len(done_users)} done), {len(todo) * len(G_T)} fits to run at most; stored fits for comparison: {stored_n}", flush=True)
    t0 = time.time()
    lines = read_jsonl(out / "users.jsonl")
    if todo:
        with mp.Pool(max(1, min(a.workers, len(todo)))) as pool:
            for d in pool.imap_unordered(user_task, todo):
                with open(out / "users.jsonl", "a") as fh:
                    fh.write(json.dumps(d) + "\n"); fh.flush(); os.fsync(fh.fileno())
                lines.append(d)
                print(f"user {d['user']} ({d['stratum']}) done in {time.time() - t0:.0f} s: alignments {d['alignments']}, served {d['served']}, wrong-and-unflagged {d['wrong_unflagged']}, identical to stored {d['identical']}/{d['compared']}, state differences {d['state_diffs']}", flush=True)
                print_totals(lines, len(users))
    cmd_summary(argparse.Namespace(out=str(out), users_per_stratum=a.users_per_stratum))


def cmd_summary(a):
    out = Path(a.out)
    lines = read_jsonl(out / "users.jsonl")
    expected = len(users_for(getattr(a, "users_per_stratum", 12)))
    t = sum_users(lines, expected)
    print(f"summary of {out} ({time.strftime('%Y-%m-%d %H:%M')}): users {t['users']}/{expected}")
    for st in ("hard", "medium", "easy"):
        ls = [d for d in lines if d["stratum"] == st]
        s = sum_users(ls, 0)
        print(f"  {st:7s}: users {s['users']}, alignments {s['alignments']}, served {s['served']}, wrong-and-unflagged {s['wrong_unflagged']}, identical to stored {s['identical']}/{s['compared']}")
    for ln in verdict_lines(t, complete=t["users"] == expected):
        print(ln)
    bad = [(d["user"], d["diffs"]) for d in lines if d["diffs"]]
    if bad:
        print("fits that differ from the stored ones (user, [(window end, what)]): " + json.dumps(bad)[:2000])
    listed = [(d["user"], d["rel_ll_listed"]) for d in lines if d["rel_ll_listed"]]
    if listed:
        print(f"windows with a restart log-likelihood differing by more than {LL_REPORT:g} relative (reported, no verdict): " + json.dumps(listed)[:1500])


def cmd_reference(a):
    """The numbers of the STORED fits of the same 36 users (no fitting): the reference the replay is compared with."""
    check_operating_point(FREEZE.read_text())
    stored = load_stored(a.stored)
    lines = []
    for u, st in users_for(a.users_per_stratum):
        recs = [stored[(u, t)] for t in G_T if (u, t) in stored]
        if len(recs) != len(G_T):
            raise SystemExit(f"stored fits missing for user {u}: {len(recs)} of {len(G_T)}")
        d = replay_user(recs, stored, *OPERATING_POINT)
        lines.append(d)
    t = sum_users(lines, len(lines))
    print(f"REFERENCE from the stored WP10A fits ({a.stored}), {t['users']} users, operating point {OPERATING_POINT}: alignments {t['alignments']}, served {t['served']} of {t['refits']} refits, wrong-and-unflagged {t['wrong_unflagged']}; exact 95% upper bound {100 * exact_upper(t['wrong_unflagged'], t['alignments']):.2f}%")
    for st in ("hard", "medium", "easy"):
        s = sum_users([d for d in lines if d["stratum"] == st], 0)
        print(f"  {st:7s}: alignments {s['alignments']}, served {s['served']} of {s['refits']}, wrong-and-unflagged {s['wrong_unflagged']}")
    print(f"  self-comparison (stored against stored): identical {t['identical']}/{t['compared']}, state differences {t['state_diffs']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--out", required=True); p.add_argument("--users-per-stratum", type=int, default=12); p.add_argument("--workers", type=int, default=2)
    p.add_argument("--stored", default=str(STORED_DEFAULT)); p.set_defaults(f=cmd_run)
    p = sub.add_parser("summary"); p.add_argument("--out", required=True); p.add_argument("--users-per-stratum", type=int, default=12); p.set_defaults(f=cmd_summary)
    p = sub.add_parser("reference"); p.add_argument("--stored", default=str(STORED_DEFAULT)); p.add_argument("--users-per-stratum", type=int, default=12); p.set_defaults(f=cmd_reference)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
