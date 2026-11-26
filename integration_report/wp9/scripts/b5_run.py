"""WP9 B5: monitor simulation (design: tests/PREDECLARED_CRITERIA_WP9.md sections 24 and 26). usage:
  PYTHONPATH=. python3 integration_report/wp9/scripts/b5_run.py --out wp9_evidence/b5_sim.jsonl
Seeds 9901..9912, one line per user, resumable. 180-day streams (default generator, K = 3, separation 1.0 to 2.5); the baseline K = 3 model is fitted with `fit_hssm` on days [0, 90) and frozen;
the held-out one-step score of every day 90..179 is computed on three streams that are identical before day 135: none, regime shift (every regime mean shifted by 2.0 within-regime sd along a fixed
random direction from day 135), sensor change (generator option feature_scale_change: noise of features 0, 1, 2 x 4 from day 135). Two series: S1 = the score, S2 = the score minus the single Gaussian of days [0, 90).
The monitor is the production `predictability_monitor` at its default (frozen) parameters."""
import argparse, contextlib, io, json, logging, os, sys, time, warnings
from pathlib import Path

import numpy as np

SEEDS = range(9901, 9913)
CHANGE_DAY, N_DAYS, TRAIN_END = 135, 180, 90
SPEC_KW = dict(n_regimes=3, sep_lo=1.0, sep_hi=2.5)


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
    from backbone.hssm.cold_start_cert import gaussian_heldout
    from backbone.hssm.monitoring import baseline_predictive_increments, predictability_monitor
    from data.synthetic_identity import SCALE, IdentitySpec, draw_user, generate_identity_stream
    spec = IdentitySpec(**SPEC_KW)
    t0 = time.time()
    base = generate_identity_stream(user, N_DAYS, spec)
    sensor = generate_identity_stream(user, N_DAYS, spec, feature_scale_change=dict(features=(0, 1, 2), factor=4.0, start=CHANGE_DAY))
    shift = generate_identity_stream(user, N_DAYS, spec)
    u = draw_user(user, spec)
    vec = np.random.default_rng([user, 99]).standard_normal(len(u["noise"])); vec /= np.linalg.norm(vec)
    shift.X_obs = shift.X_obs.copy()
    shift.X_obs[CHANGE_DAY:] = shift.X_obs[CHANGE_DAY:] + 2.0 * vec * (SCALE * u["noise"])         # NaNs stay NaN
    w = base.window(0, TRAIN_END)
    mean, std = w["scaler_mean"], w["scaler_std"]
    rec = dict(user=user, ok=False)
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            r = fit_hssm(matrix=w["matrix"], candidate_ks=(3,), n_initializations=10, max_iter=60, random_seed=user, timestamps=np.arange(TRAIN_END, dtype=float), user_id=f"b5-{user}", dataset_hash=f"b5h{user}")
        model = r.model
        rec.update(ok=True, fit_state=str(r.fit_state.value), withheld=bool(r.insights_withheld), K=int(r.K), sep=float(u["sep"]))
        train = w["matrix"][~np.isnan(w["matrix"]).all(axis=1)]
        out = {}
        for name, st in (("none", base), ("regime_shift", shift), ("sensor_change", sensor)):
            Z, miss = z_rows(st, mean, std, 0, N_DAYS)
            inc = baseline_predictive_increments(model, Z)
            s1 = inc[TRAIN_END:].copy(); s1[miss[TRAIN_END:]] = np.nan
            held = Z[TRAIN_END:]
            g = np.full(len(held), np.nan)
            ok = ~np.isnan(held).all(axis=1)
            g[ok] = gaussian_heldout(train, held[ok])
            s2 = s1 - g
            res = {}
            for sname, series in (("S1", s1), ("S2", s2)):
                m = predictability_monitor(series)
                res[sname] = dict(status=m["status"], event_days=[TRAIN_END + e["day"] for e in m["events"]], n_scores=m["n_scores"], mean_before=float(np.nanmean(series[:CHANGE_DAY - TRAIN_END])),
                                  mean_after=float(np.nanmean(series[CHANGE_DAY - TRAIN_END:])))
            out[name] = res
        rec["streams"] = out
    except Exception as exc:  # noqa: BLE001
        rec.update(error=f"{type(exc).__name__}: {str(exc)[:300]}")
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        for ln in open(a.out):
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            if d.get("ok"):
                done.add(d["user"])
    for user in SEEDS:
        if user in done:
            continue
        rec = one(user)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())
        print(user, "ok" if rec["ok"] else rec.get("error"), rec["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
