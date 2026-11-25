"""WP8 study library (Items 2, 3, 4, 8): fitting a window of an identity stream, summarising a fit, and measuring alignments against planted truth.
Used by the study scripts only. The production versions of the ordering and the alignment live in backbone/hssm/ordering.py and lineage.py (Items 3 and 4) and are
cross-checked against this library. Design: tests/PREDECLARED_CRITERIA_WP8.md."""
from __future__ import annotations

import contextlib, io, itertools, json, logging, os, time, warnings
import numpy as np
from scipy.optimize import linear_sum_assignment

FIT_DEFAULTS = {
    "baseline": dict(n_initializations=10, max_iter=60),
    "dual": dict(n_initializations=2, max_iter=60, allow_fast_test_fit=True, model_type="dual_timescale", latent_dim=2, max_duration=30),
}


def stream_for(job):
    from data.synthetic_identity import generate_identity_stream
    return generate_identity_stream(job["user"], n_days=job.get("n_days", 150), drift_sd=job.get("drift", 0.0), k_change=job.get("kchange"))


def fit_window(job: dict) -> dict:
    """job: user, engine ('baseline'|'dual'), start, end, fitseed, K, drift, kchange. Returns a JSON-able record (regime summaries, decoded labels, status, timing)."""
    from backbone.hssm.fitting import fit_hssm
    from backbone.hssm.shrinkage import duration_pmf, extract_params, regime_observation_means
    logging.disable(logging.CRITICAL)
    t0 = time.time()
    st = stream_for(job)
    w = st.window(job["start"], job["end"])
    kw = dict(FIT_DEFAULTS[job["engine"]]); kw.update(job.get("fit_kw", {}))
    K = job["K"]
    rec = dict(job=job, ok=False)
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            r = fit_hssm(matrix=w["matrix"], candidate_ks=(K,) if np.isscalar(K) else tuple(K), random_seed=job["fitseed"], user_id=f"u{job['user']}", dataset_hash=f"h{job['user']}", **kw)
        m = r.model
        Kf = int(m.K)
        if job["engine"] == "baseline":
            z_means = np.array(m.mu, dtype=float)
        else:
            z_means = regime_observation_means(extract_params(m))
        raw_means = w["scaler_mean"] + w["scaler_std"] * z_means
        dmax = int(m.Dmax)
        pmf = duration_pmf(np.array(m.dur_mu), np.array(m.dur_sigma), dmax)
        exp_dur = (pmf * np.arange(1, dmax + 1)).sum(axis=1)
        post = np.asarray(r.p_t_posterior)
        rec.update(ok=True, K=Kf, z_means=z_means.tolist(), raw_means=raw_means.tolist(), scaler_mean=w["scaler_mean"].tolist(), scaler_std=w["scaler_std"].tolist(),
                   coverage=w["coverage"].tolist(), exp_dur=exp_dur.tolist(), labels_fit=post.argmax(axis=1).tolist(), labels_true=w["labels"].tolist(),
                   fit_state=str(r.fit_state.value), accepted=not bool(r.insights_withheld), withheld_reason=r.withheld_reason, log_likelihood=float(getattr(m, "log_likelihood_", float("nan"))),
                   max_post=float(post.max(axis=1).mean()))
    except Exception as exc:                                                       # a failed fit is data too
        rec.update(error=f"{type(exc).__name__}: {exc}")
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def job_key(job: dict) -> str:
    return json.dumps({k: job[k] for k in sorted(job) if k != "fit_kw"} | {"fit_kw": job.get("fit_kw", {})}, sort_keys=True)


def run_jobs(jobs, out_path, workers=2):
    """Resumable: a job whose key is already in `out_path` is skipped; each record is appended and flushed as soon as it arrives."""
    import multiprocessing as mp
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(job_key(json.loads(line)["job"]))
            except Exception:
                pass
    todo = [j for j in jobs if job_key(j) not in done]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run", flush=True)
    with open(out_path, "a") as fh, mp.Pool(workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(fit_window, todo), 1):
            fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())
            print(i, "/", len(todo), rec["job"]["engine"], rec["job"]["user"], rec["job"]["start"], rec["job"]["end"], "ok" if rec["ok"] else rec.get("error"), rec["seconds"], "s", flush=True)


def load_records(path):
    return [json.loads(l) for l in open(path) if l.strip()]


# ------------------------------------------------------------------------------------------------- agreement and truth
def contingency(a, b, Ka, Kb):
    M = np.zeros((Ka, Kb))
    for x, y in zip(a, b):
        M[x, y] += 1
    return M


def best_relabel_agreement(a, b, Ka, Kb) -> float:
    """Share of days on which a and b agree after the best one-to-one relabelling (Hungarian on the contingency table)."""
    M = contingency(a, b, Ka, Kb)
    r, c = linear_sum_assignment(-M)
    return float(M[r, c].sum() / max(len(a), 1))


def overlap_map(fit_labels, true_labels, K_fit, K_true):
    """For each fit regime, the planted regime with the largest day overlap; recoverable iff the map is one-to-one onto the planted regimes present in the fit's window."""
    M = contingency(fit_labels, true_labels, K_fit, K_true)
    mp = [int(M[j].argmax()) if M[j].sum() > 0 else -1 for j in range(K_fit)]
    present = {int(t) for t in set(true_labels)}
    recoverable = len(set(mp)) == len(mp) and -1 not in mp and set(mp) == present
    return mp, bool(recoverable)


def l2_order_matches(mp) -> bool:
    """True iff raw regime index order equals the planted activity order, i.e. the map raw index -> planted index is increasing."""
    return all(mp[i] < mp[i + 1] for i in range(len(mp) - 1))


# ------------------------------------------------------------------------------------------------- distances, alignment and margins
def regime_vectors(rec, mode, ref_scale=None):
    """The vectors that are compared. mode 'z': the fit's own z-space means (negative control, no stored scaler); 'raw': raw units divided by the reference scale;
    'raw_dur': raw plus the log expected duration as one more coordinate (weight 1)."""
    if mode == "z":
        return np.array(rec["z_means"])
    raw = np.array(rec["raw_means"]) / np.asarray(ref_scale)
    if mode == "raw":
        return raw
    if mode == "raw_dur":
        return np.hstack([raw, np.log(np.array(rec["exp_dur"]))[:, None]])
    raise ValueError(mode)


def assignment_costs(ref_vec, new_vec):
    """Hungarian assignment and the exhaustive second best (K <= 6). Returns perm (perm[i] = new regime matched to reference regime i), C1, C2, per-regime distances."""
    K = ref_vec.shape[0]
    cost = ((ref_vec[:, None, :] - new_vec[None, :, :]) ** 2).sum(axis=2)
    rows, cols = linear_sum_assignment(cost)
    perm = tuple(int(c) for c in cols[np.argsort(rows)])
    C1 = float(sum(cost[i, perm[i]] for i in range(K)))
    others = [sum(cost[i, q[i]] for i in range(K)) for q in itertools.permutations(range(K)) if tuple(q) != perm]
    C2 = float(min(others)) if others else float("inf")
    matched = [float(cost[i, perm[i]]) for i in range(K)]
    nearest_other = [float(min(cost[i, j] for j in range(K) if j != perm[i])) if K > 1 else float("inf") for i in range(K)]
    return perm, C1, C2, matched, nearest_other


def align_records(prev, new, mode):
    """Align `new` to `prev` in the given distance mode; reference scale = prev's scaler_std. Returns the alignment with its margins."""
    ref_scale = np.array(prev["scaler_std"])
    pv, nv = regime_vectors(prev, mode, ref_scale), regime_vectors(new, mode, ref_scale)
    perm, C1, C2, matched, nearest_other = assignment_costs(pv, nv)
    K = pv.shape[0]
    return dict(perm=perm, C1=C1, C2=C2, margin_diff=(C2 - C1) / K, margin_ratio=(C1 / C2) if C2 > 0 else 1.0, matched=matched, nearest_other=nearest_other)


def alignment_correct(prev, new, perm, K_true=3):
    """Correct iff every regime of `new` gets the planted identity of the previous fit's regime it was matched to. Both fits must be recoverable."""
    mp_prev, ok_prev = overlap_map(prev["labels_fit"], prev["labels_true"], prev["K"], K_true + 1)
    mp_new, ok_new = overlap_map(new["labels_fit"], new["labels_true"], new["K"], K_true + 1)
    if prev["K"] != new["K"]:
        return None, ok_prev and ok_new
    good = all(mp_prev[i] == mp_new[perm[i]] for i in range(prev["K"]))
    return bool(good), bool(ok_prev and ok_new)
