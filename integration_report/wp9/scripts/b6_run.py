"""WP9 B6: re-certification simulation (design: tests/PREDECLARED_CRITERIA_WP9.md section 25). usage:
  PYTHONPATH=. python3 integration_report/wp9/scripts/b6_run.py --out wp9_evidence/b6_sim.jsonl
Seeds 9801..9812 (9801..9806 hard: separation 0.5..1.0; 9807..9812 medium: 1.0..2.0), K = 3, 90-day streams, production public path `fit_hssm` (candidates (2, 3, 4), 10 restarts, 100 iterations).
Day-30 fit: the shortest prefix [0, end0) with at least 30 present days (end0 >= 30; the cold-start gate counts present days). The stored matrix is z-scored with the scaler of [0, end0); the days
that arrive later are z-scored with that stored scaler. Of the fits withheld at day 30: `recertify` with the rows to day 60 and, if still withheld, again with the rows to day 90.
For every user: a fresh fit on the same matrix (F1, equals the re-certification by construction: determinism check) and a fresh window fit with its own z-scoring (F2) at days 60 and 90.
One line per user, resumable."""
import argparse, contextlib, io, json, logging, os, sys, time, warnings

import numpy as np

SEEDS = range(9801, 9813)
SETTINGS = dict(candidate_ks=(2, 3, 4), n_initializations=10, max_iter=100)


def stream_for(user):
    from data.synthetic_identity import IdentitySpec, generate_identity_stream
    lo, hi = (0.5, 1.0) if user <= 9806 else (1.0, 2.0)
    return generate_identity_stream(user, 90, IdentitySpec(n_regimes=3, sep_lo=lo, sep_hi=hi)), (lo, hi)


def z_with(stream, mean, std, start, end):
    raw = stream.X_obs[start:end]
    z = (raw - mean) / std
    miss = np.isnan(raw).all(axis=1)
    z = np.where(np.isnan(z), 0.0, z)
    z[miss] = np.nan
    return z


def independent_publish_check(r):
    """Recomputes the publish decision from the raw attributes of a result (not from `_withheld_reasons`). True = publishable."""
    from backbone.hssm.fitting import _NOT_CONVERGED_STATES
    from backbone.hssm.model import FitState
    underlying = r.fit_state
    if underlying == FitState.MODEL_SELECTION_CONTESTED:
        underlying = (r.convergence_metadata or {}).get("fit_state", FitState.CONVERGED)
    ok = underlying not in _NOT_CONVERGED_STATES and underlying != FitState.NUMERICAL_FAILURE
    ok = ok and not r.fit_collapsed and not r.fit_duration_degenerate and not r.fit_degraded and r.states_pass_fallbacks == 0 and r.late_fallbacks == 0
    return bool(ok)


def summarize(r):
    return dict(withheld=bool(r.insights_withheld), reasons=[str(x)[:60] for x in r._withheld_reasons()], fit_state=str(r.fit_state.value), K=int(r.K), independent_publishable=independent_publish_check(r))


def one(user):
    logging.disable(logging.CRITICAL)
    from backbone.hssm.fitting import fit_hssm
    from backbone.hssm.recertify import recertify
    from backbone.hssm.gating import ColdStartError
    st, (lo, hi) = stream_for(user)
    t0 = time.time()
    rec = dict(user=user, ok=False, stratum="hard" if user <= 9806 else "medium")
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            end0 = 30
            while int((~np.isnan(st.X_obs[:end0]).all(axis=1)).sum()) < 30:
                end0 += 1
            rec["end0"] = end0
            w0 = st.window(0, end0)
            mean, std = w0["scaler_mean"], w0["scaler_std"]
            kw = dict(SETTINGS, random_seed=user, user_id=f"b6-{user}", dataset_hash=f"b6h{user}")
            r0 = fit_hssm(matrix=w0["matrix"], timestamps=np.arange(end0, dtype=float), **kw)
            rec["day30"] = summarize(r0)
            stored = w0["matrix"]
            ts = np.arange(end0, dtype=float)
            meta = dict(settings=dict(kw), earlier_state="STILL_WITHHELD", last_check_day=end0)
            rec["recert"], violations, outputs = {}, 0, 0
            pub = not r0.insights_withheld
            outputs += int(pub); violations += int(pub and not rec["day30"]["independent_publishable"])
            cur_state = "PUBLISHABLE" if pub else "STILL_WITHHELD"
            for end in (60, 90):
                new = z_with(st, mean, std, len(stored), end)
                new_ts = np.arange(len(stored), end, dtype=float)
                if cur_state == "STILL_WITHHELD":
                    meta["earlier_state"] = "STILL_WITHHELD"
                    out = recertify(stored, new, metadata=meta, timestamps=ts, new_timestamps=new_ts, today=None)
                    entry = dict(state=out["state"], reasons=[r_[:60] for r_ in out["reasons"]], rules_relaxed=out["rules_relaxed"], n_rows=out.get("n_rows"))
                    if out["state"] == "PUBLISHABLE":
                        entry["independent_publishable"] = independent_publish_check(out["result"]); outputs += 1; violations += int(not entry["independent_publishable"] or out["result"].insights_withheld)
                    violations += int(out["rules_relaxed"] is not False or (out["state"] == "PUBLISHABLE" and out["result"] is None) or (out["state"] != "PUBLISHABLE" and out["result"] is not None))
                    # F1: the same matrix, a fresh call (determinism)
                    f1 = fit_hssm(matrix=np.vstack([stored, new]), timestamps=np.concatenate([ts, new_ts]), **kw)
                    entry["f1_withheld"] = bool(f1.insights_withheld); entry["f1_equals_recert"] = (bool(f1.insights_withheld) == (out["state"] != "PUBLISHABLE"))
                    rec["recert"][str(end)] = entry
                    cur_state = out["state"]
                    stored, ts = np.vstack([stored, new]), np.concatenate([ts, new_ts])
                else:
                    stored, ts = np.vstack([stored, new]), np.concatenate([ts, new_ts])
                    rec["recert"][str(end)] = dict(state="PUBLISHABLE_EARLIER")
                wf = st.window(0, end)
                f2 = fit_hssm(matrix=wf["matrix"], timestamps=np.arange(end, dtype=float), **kw)
                rec.setdefault("fresh", {})[str(end)] = summarize(f2)
                outputs += int(not f2.insights_withheld); violations += int((not f2.insights_withheld) and not independent_publish_check(f2))
            rec.update(ok=True, outputs_published=outputs, violations=violations)
    except ColdStartError as exc:
        rec.update(error=f"ColdStartError: {exc}"[:200])
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
