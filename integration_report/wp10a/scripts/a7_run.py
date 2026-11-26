"""WP10A A7: a re-certification arm that is not vacuous (design: tests/PREDECLARED_CRITERIA_WP10A.md Part 5, sections 12-15). usage:
  PYTHONPATH=. python3 integration_report/wp10a/scripts/a7_run.py --arm W1|W2 --out FILE.jsonl [--workers 2]
Arm W1: seeds 10601..10648, candidates (5,) forced (a misspecified K that withholds fits). Arm W2: seeds 10651..10698, the production candidates (2, 3, 4), separation 0.3 to 0.6.
Users are run in seed order in chunks of 6 until 12 are withheld at day 30 or the seeds are used up (the analysis applies the same rule in seed order). One line per user, resumable.
Per user: the WP9 B6 procedure (`wp9/scripts/b6_run.py`: the day-30 fit, `recertify` with the rows to day 60 and 90, a fresh fit on the same matrix per call = F1; no F2), plus the settings probes of
every withheld user and the checker control (the attributes `independent_publish_check` reads, altered five ways)."""
from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import io
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

CHUNK, TARGET = 6, 12
ARMS = {
    "W1": dict(seeds=list(range(10601, 10649)), candidates=(5,), sep=lambda u: (0.5, 1.0) if u % 2 == 0 else (1.0, 2.0), stratum=lambda u: "hard" if u % 2 == 0 else "medium"),
    "W2": dict(seeds=list(range(10651, 10699)), candidates=None, sep=lambda u: (0.3, 0.6), stratum=lambda u: "very_hard"),
}
PROBES = (("n_initializations_3", dict(n_initializations=3)), ("unknown_allow_fast_test_fit", dict(allow_fast_test_fit=True)), ("unknown_config", dict(config="relaxed")))
ALTERATIONS = ("fit_collapsed", "fit_degraded", "fit_duration_degenerate", "states_pass_fallbacks", "fit_state_not_converged")
CHECKED_ATTRS = ("fit_state", "convergence_metadata", "fit_collapsed", "fit_duration_degenerate", "fit_degraded", "states_pass_fallbacks", "late_fallbacks")


def _load_wp9_b6():
    spec = importlib.util.spec_from_file_location("wp9_b6_run_for_a7", ROOT / "integration_report" / "wp9" / "scripts" / "b6_run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wp9_b6_run_for_a7"] = mod
    spec.loader.exec_module(mod)
    return mod


B6 = _load_wp9_b6()


# ------------------------------------------------------------------------------------------------------------------- pure helpers (tested)
def chunks(seq, n):
    seq = list(seq)
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def withheld_count(recs) -> int:
    return sum(1 for r in recs if r.get("ok") and r["day30"]["withheld"])


def should_stop(recs, target: int = TARGET) -> bool:
    """The declared stopping rule: at least `target` users withheld at the day-30 check among the users run so far."""
    return withheld_count(recs) >= target


def snapshot(result):
    """The attributes `independent_publish_check` reads, as a plain object that can be altered without touching the result."""
    return SimpleNamespace(**{k: copy.copy(getattr(result, k, None)) for k in CHECKED_ATTRS})


def alter(snap, name: str):
    """A copy of `snap` with one rule violated."""
    from backbone.hssm.model import FitState
    s = SimpleNamespace(**vars(snap))
    if name == "fit_collapsed":
        s.fit_collapsed = True
    elif name == "fit_degraded":
        s.fit_degraded = True
    elif name == "fit_duration_degenerate":
        s.fit_duration_degenerate = True
    elif name == "states_pass_fallbacks":
        s.states_pass_fallbacks = 1
    elif name == "fit_state_not_converged":
        s.fit_state = FitState.MAX_ITER_REACHED
        s.convergence_metadata = None
    else:
        raise ValueError(name)
    return s


def checker_control(snap) -> dict:
    """{unaltered: bool, altered: {name: bool}}: the independent check on the snapshot of a PUBLISHED result (must be True) and on each alteration of it (each must be False)."""
    return dict(unaltered=bool(B6.independent_publish_check(snap)), altered={n: bool(B6.independent_publish_check(alter(snap, n))) for n in ALTERATIONS})


def settings_probes(stored, new, meta, ts, new_ts) -> list:
    """`recertify` with settings that would relax a rule: each call must be refused (NOT_ELIGIBLE, no result, rules_relaxed False) before anything is fitted."""
    from backbone.hssm.recertify import recertify
    out = []
    for name, extra in PROBES:
        m = dict(meta, settings=dict(meta["settings"], **extra))
        r = recertify(stored, new, metadata=m, timestamps=ts, new_timestamps=new_ts, today=None)
        out.append(dict(probe=name, state=r["state"], reasons=[str(x)[:80] for x in r["reasons"]], result_returned=r["result"] is not None, rules_relaxed=r["rules_relaxed"]))
    return out


# ------------------------------------------------------------------------------------------------------------------- the user
def stream_for(arm, user):
    from data.synthetic_identity import IdentitySpec, generate_identity_stream
    lo, hi = ARMS[arm]["sep"](user)
    return generate_identity_stream(user, 90, IdentitySpec(n_regimes=3, sep_lo=lo, sep_hi=hi)), (lo, hi)


def one(job):
    arm, user = job
    logging.disable(logging.CRITICAL)
    from backbone.hssm.fitting import fit_hssm
    from backbone.hssm.gating import ColdStartError
    from backbone.hssm.recertify import recertify
    st, (lo, hi) = stream_for(arm, user)
    t0 = time.time()
    rec = dict(arm=arm, user=int(user), ok=False, stratum=ARMS[arm]["stratum"](user), sep=[lo, hi], probes=None, snapshot_control=None, recert={})
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            end0 = 30
            while int((~np.isnan(st.X_obs[:end0]).all(axis=1)).sum()) < 30:
                end0 += 1
            rec["end0"] = end0
            w0 = st.window(0, end0)
            mean, std = w0["scaler_mean"], w0["scaler_std"]
            kw = dict(B6.SETTINGS, random_seed=user, user_id=f"a7-{user}", dataset_hash=f"a7h{user}")
            if ARMS[arm]["candidates"]:
                kw["candidate_ks"] = tuple(ARMS[arm]["candidates"])
            r0 = fit_hssm(matrix=w0["matrix"], timestamps=np.arange(end0, dtype=float), **kw)
            rec["day30"] = B6.summarize(r0)
            snap = snapshot(r0) if not r0.insights_withheld else None
            stored, ts = w0["matrix"], np.arange(end0, dtype=float)
            meta = dict(settings=dict(kw), earlier_state="STILL_WITHHELD", last_check_day=end0)
            cur = "PUBLISHABLE" if not r0.insights_withheld else "STILL_WITHHELD"
            for end in (60, 90):
                new = B6.z_with(st, mean, std, len(stored), end)
                new_ts = np.arange(len(stored), end, dtype=float)
                if cur == "STILL_WITHHELD":
                    if rec["probes"] is None:
                        rec["probes"] = settings_probes(stored, new, meta, ts, new_ts)
                    out = recertify(stored, new, metadata=meta, timestamps=ts, new_timestamps=new_ts, today=None)
                    entry = dict(state=out["state"], reasons=[x[:60] for x in out["reasons"]], rules_relaxed=out["rules_relaxed"], n_rows=out.get("n_rows"), result_returned=out["result"] is not None)
                    if out["result"] is not None:
                        entry["result_withheld"] = bool(out["result"].insights_withheld)
                        entry["independent_publishable"] = bool(B6.independent_publish_check(out["result"]))
                        if snap is None:
                            snap = snapshot(out["result"])
                    f1 = fit_hssm(matrix=np.vstack([stored, new]), timestamps=np.concatenate([ts, new_ts]), **kw)
                    entry["f1_withheld"] = bool(f1.insights_withheld)
                    rec["recert"][str(end)] = entry
                    cur = out["state"]
                else:
                    rec["recert"][str(end)] = dict(state="PUBLISHABLE_EARLIER")
                stored, ts = np.vstack([stored, new]), np.concatenate([ts, new_ts])
            rec["snapshot_control"] = checker_control(snap) if snap is not None else None
            rec["ok"] = True
    except ColdStartError as exc:
        rec["error"] = f"ColdStartError: {exc}"[:200]
    except Exception as exc:                                                                    # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def read_done(path):
    recs = []
    if os.path.exists(path):
        for ln in open(path):
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            if d.get("ok"):
                recs.append(d)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS)); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    recs = read_done(a.out)
    done = {r["user"] for r in recs}
    todo = [s for s in ARMS[a.arm]["seeds"] if s not in done]
    print(f"arm {a.arm}: {len(recs)} users done ({withheld_count(recs)} withheld at day 30), {len(todo)} seeds left", flush=True)
    with mp.Pool(a.workers) as pool:
        for ch in chunks(todo, CHUNK):
            if should_stop(recs):
                print(f"stopping rule met: {withheld_count(recs)} withheld at day 30", flush=True)
                break
            for rec in pool.imap(one, [(a.arm, s) for s in ch]):
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())
                print(rec["user"], "ok" if rec["ok"] else rec.get("error"), rec["seconds"], "s", "withheld" if rec["ok"] and rec["day30"]["withheld"] else "", flush=True)
                if rec["ok"]:
                    recs.append(rec)


if __name__ == "__main__":
    main()
