"""WP10A A5 fits (design: tests/PREDECLARED_CRITERIA_WP10A.md Part 4, sections 8-11). The WP9 B5 protocol (`wp9/scripts/b5_run.py`) on new users, storing what the candidate monitors need.
usage: PYTHONPATH=. python3 integration_report/wp10a/scripts/a5_run.py --range dev|eval --out FILE.jsonl [--workers 2] [--limit N] [--count]
dev = users 10401..10448; eval = users 10451..10546 (refused without tests/DESIGN_FREEZE_WP10A_A5.txt). One line per user, resumable (a user already in the file with ok = true is skipped).
Per user: a 180-day stream of the default generator (K = 3, separation 1.0 to 2.5); the production `fit_hssm` fits the baseline K = 3 model on days [0, 90) (10 restarts, 60 iterations, fit seed = user) and freezes it;
for each of three streams (none; regime shift = every regime mean moved by 2.0 within-regime sd along a fixed random direction from day 135; sensor change = noise of features 0, 1, 2 x 4 from day 135) the
frozen model's forward pass over days 0..179 gives the per-day increments and the filtered regime posterior. Stored: increments (180), posterior (180 x 3), missing-day mask, per stream."""
import argparse
import contextlib
import io
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
RANGES = {"dev": range(10401, 10449), "eval": range(10451, 10547)}
CHANGE_DAY, N_DAYS, TRAIN_END = 135, 180, 90
SPEC_KW = dict(n_regimes=3, sep_lo=1.0, sep_hi=2.5)
FREEZE = Path("tests/DESIGN_FREEZE_WP10A_A5.txt")


def z_rows(stream, mean, std, start, end):
    raw = stream.X_obs[start:end]
    z = (raw - mean) / std
    day_missing = np.isnan(raw).all(axis=1)
    z = np.where(np.isnan(z), 0.0, z)
    z[day_missing] = np.nan
    return z, day_missing


def one(user):
    logging.disable(logging.CRITICAL)
    from backbone.hssm.fitting import fit_hssm
    from backbone.hssm.monitoring import baseline_forward_scores
    from data.synthetic_identity import SCALE, IdentitySpec, draw_user, generate_identity_stream
    spec = IdentitySpec(**SPEC_KW)
    t0 = time.time()
    base = generate_identity_stream(user, N_DAYS, spec)
    sensor = generate_identity_stream(user, N_DAYS, spec, feature_scale_change=dict(features=(0, 1, 2), factor=4.0, start=CHANGE_DAY))
    shift = generate_identity_stream(user, N_DAYS, spec)
    u = draw_user(user, spec)
    vec = np.random.default_rng([user, 99]).standard_normal(len(u["noise"]))
    vec /= np.linalg.norm(vec)
    shift.X_obs = shift.X_obs.copy()
    shift.X_obs[CHANGE_DAY:] = shift.X_obs[CHANGE_DAY:] + 2.0 * vec * (SCALE * u["noise"])         # NaNs stay NaN
    w = base.window(0, TRAIN_END)
    mean, std = w["scaler_mean"], w["scaler_std"]
    rec = dict(user=user, ok=False)
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            r = fit_hssm(matrix=w["matrix"], candidate_ks=(3,), n_initializations=10, max_iter=60, random_seed=user, timestamps=np.arange(TRAIN_END, dtype=float), user_id=f"a5-{user}", dataset_hash=f"a5h{user}")
        model = r.model
        rec.update(ok=True, fit_state=str(r.fit_state.value), withheld=bool(r.insights_withheld), K=int(r.K), sep=float(u["sep"]))
        streams = {}
        for name, st in (("none", base), ("regime_shift", shift), ("sensor_change", sensor)):
            Z, miss = z_rows(st, mean, std, 0, N_DAYS)
            inc, post = baseline_forward_scores(model, Z)
            streams[name] = dict(inc=[float(x) for x in inc], post=[[float(x) for x in row] for row in post], miss=[bool(x) for x in miss])
        rec["streams"] = streams
    except Exception as exc:  # noqa: BLE001
        rec.update(error=f"{type(exc).__name__}: {str(exc)[:300]}")
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def done_users(path):
    done = set()
    if os.path.exists(path):
        for ln in open(path):
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            if d.get("ok"):
                done.add(d["user"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", required=True, choices=list(RANGES)); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--count", action="store_true")
    a = ap.parse_args()
    if a.range == "eval" and not FREEZE.exists():
        sys.exit("the evaluation runs only after tests/DESIGN_FREEZE_WP10A_A5.txt is committed")
    todo = [u for u in RANGES[a.range] if u not in done_users(a.out)]
    if a.limit is not None:
        todo = todo[:a.limit]
    print(f"{len(RANGES[a.range])} users, {len(todo)} to run", flush=True)
    if a.count:
        return
    with open(a.out, "a") as fh, mp.Pool(a.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(one, todo), 1):
            fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())
            print(i, "/", len(todo), rec["user"], "ok" if rec["ok"] else rec.get("error"), rec["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
