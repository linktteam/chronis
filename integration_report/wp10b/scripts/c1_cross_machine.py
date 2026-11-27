"""WP10B C1(3) (b): is the BASELINE engine affected by the numpy AVX-512 dispatch? The WP9 12-user battery (`wp9/scripts/a1_battery.py`, population users 5301..5312, variants `fast` = K 2, 3 restarts and `full` = K in (2, 3),
10 restarts, 60 iterations, the public path `fit_hssm`) is run in two processes of the SAME tree on the SAME machine: `default` (numpy dispatches AVX-512) and `NO_AVX512` (NPY_DISABLE_CPU_FEATURES = every AVX-512 name numpy dispatches on,
the configuration that flipped the F2 probe outcome in `c1_cpu_features.py`). The comparison reports what the PRODUCT sees: fit state, K, BIC, withheld reasons, the stability flag, the decoded regime labels, and the served lineage
(`backbone/hssm/lineage.step`: the labels the lineage would serve as stable ids, the stable names, and whether a second fit of the same window is ALIGNED with the first), plus the size of every numeric difference.
usage: PYTHONPATH=. python3.12 integration_report/wp10b/scripts/c1_cross_machine.py drive --outdir DIR [--variants fast,full] [--workers 2]
       PYTHONPATH=. python3.12 integration_report/wp10b/scripts/c1_cross_machine.py fit --tree TREE --variant fast|full --out F.pkl [--workers 2]      (run it under the environment of the configuration)
       PYTHONPATH=. python3.12 integration_report/wp10b/scripts/c1_cross_machine.py compare REF.pkl NEW.pkl [--out TXT]"""
from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "integration_report" / "wp9" / "scripts"))
import a1_battery as B                                                                                 # noqa: E402

AVX512_ALL = "AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"
CONFIGS = {"default": {}, "NO_AVX512": {"NPY_DISABLE_CPU_FEATURES": AVX512_ALL}, "NO_AVX512_ZEN": {"NPY_DISABLE_CPU_FEATURES": AVX512_ALL, "OPENBLAS_CORETYPE": "ZEN"}}      # the last: closest to the AMD EPYC 7763 (no AVX-512; OpenBLAS Zen/Haswell kernels)


# ------------------------------------------------------------------------------------------------- fit (one process = one configuration)
def one2(job):
    """The battery record of `a1_battery.one` plus what the product derives from the result: decoded labels, the stability flag, the lineage summary pieces and the training window."""
    tree, seed, variant = job
    tree = os.path.abspath(tree)
    sys.path.insert(0, tree)
    import backbone.hssm.fitting as FM
    orig, cap = FM.fit_hssm, {}

    def wrapper(*a, **k):
        r = orig(*a, **k)
        cap["r"], cap["kw"] = r, k
        return r
    FM.fit_hssm = wrapper
    try:
        rec = B.one((tree, seed, variant))
    finally:
        FM.fit_hssm = orig
    r = cap["r"]
    from backbone.hssm.lineage import summarize_fit
    X = np.asarray(cap["kw"]["matrix"], dtype=float)
    F = X.shape[1]
    s = summarize_fit(r, X, np.zeros(F), np.ones(F))
    rec.update(labels=np.asarray(s.labels, dtype=int), raw_means=np.asarray(s.raw_means), exp_dur=np.asarray(s.exp_durations), separation_min=float(s.separation_min), unstable=bool(s.fit_unstable),
               unstable_reason=s.fit_unstable_reason, raw_window=X, scaler_mean=np.zeros(F), scaler_std=np.ones(F), engine=s.engine)
    return rec


def cmd_fit(a):
    with mp.Pool(a.workers) as pool:
        recs = pool.map(one2, [(a.tree, s, a.variant) for s in B.SEEDS])
    pickle.dump(sorted(recs, key=lambda r: r["seed"]), open(a.out, "wb"))
    from numpy._core._multiarray_umath import __cpu_features__ as f
    print("numpy AVX-512 features enabled in this process:", sorted(k for k, v in f.items() if v and k.startswith("AVX512")) or "none")
    for r in sorted(recs, key=lambda r: r["seed"]):
        print(r["seed"], r["fit_state"], "withheld" if r["insights_withheld"] else "publishable", r["K"], r["seconds"], "s")


def cmd_drive(a):
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for variant in a.variants.split(","):
        for name, extra in CONFIGS.items():
            out = outdir / f"battery_{variant}_{name}.pkl"
            if out.exists():
                print("have", out, flush=True)
                continue
            env = dict(os.environ, PYTHONPATH=str(ROOT))
            for k in ("NPY_DISABLE_CPU_FEATURES", "NPY_ENABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS"):
                env.pop(k, None)
            env.update(extra)
            cmd = [sys.executable, str(Path(__file__).resolve()), "fit", "--tree", str(ROOT), "--variant", variant, "--out", str(out), "--workers", str(a.workers)]
            print(f"[{variant} {name}]", flush=True)
            r = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True, text=True)
            print(r.stdout[-1500:], r.stderr[-300:] if r.returncode else "", flush=True)
            if r.returncode:
                raise SystemExit(f"fit failed for {variant} {name}")


# ------------------------------------------------------------------------------------------------- comparison (pure functions, tested)
def label_agreement(a: np.ndarray, b: np.ndarray) -> tuple:
    """(fraction of sessions with the same label under the best one-to-one renaming of b's labels, exactly equal sequences?)."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    if a.shape != b.shape:
        return float("nan"), False
    ka, kb = int(a.max()) + 1, int(b.max()) + 1
    K = max(ka, kb)
    best = 0.0
    for perm in itertools.permutations(range(K)):
        best = max(best, float(np.mean(np.asarray(perm)[b] == a)))
    return best, bool(np.array_equal(a, b))


def max_abs_diff(x, y) -> float:
    """Largest absolute difference; NaNs must sit in the same places (the baseline engine has no latent state: its m_t arrays are all NaN in both runs, difference 0); a different NaN pattern or shape is infinite."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or not np.array_equal(np.isnan(x), np.isnan(y)):
        return float("inf")
    ok = ~np.isnan(x)
    return float(np.abs(x[ok] - y[ok]).max()) if ok.any() else 0.0


def duration_diff(ref: dict, new: dict) -> float:
    ks = sorted(set(ref) | set(new))
    if set(ref) != set(new):
        return float("inf")
    return max((max_abs_diff(ref[k], new[k]) for k in ks), default=0.0)


def lineage_outcome(ref: dict, new: dict) -> dict:
    """What the served lineage does with the two fits of the same window: (1) the stable ids it would serve per session when each fit starts a lineage (names, fraction of sessions with the same served id), (2) the continuity state when the
    fit of the other configuration is fed as the next fit (production `step`, default `AlignmentConfig`), and whether the product may claim 'same as before'."""
    from backbone.hssm.lineage import AlignmentConfig, make_summary, may_claim_same_as_before, step
    cfg = AlignmentConfig()

    def summ(r):
        return make_summary("baseline", r["raw_means"], r["scaler_mean"], r["scaler_std"], r["labels"], r["raw_window"], exp_durations=r["exp_dur"], fit_unstable=r["unstable"], fit_unstable_reason=r["unstable_reason"])
    out = {}
    try:
        sa, sb = summ(ref), summ(new)
        ra, state_a = step(None, sa, cfg, None, "x")
        rb, _ = step(None, sb, cfg, None, "x")
        ida, idb = np.asarray(ra.stable_regime_id), np.asarray(rb.stable_regime_id)
        out.update(served_ids_equal=bool(np.array_equal(ida, idb)), served_ids_agree=float(np.mean(ida == idb)) if ida.shape == idb.shape else float("nan"), names_a=tuple(ra.raw_to_stable), names_b=tuple(rb.raw_to_stable))
        nxt, _ = step(state_a, sb, cfg, None)
        own, _ = step(state_a, sa, cfg, None)                                          # the same fit fed twice: what the gate itself does, whatever the machine
        out.update(next_state=str(nxt.continuity_state), may_claim=bool(may_claim_same_as_before(nxt)), self_state=str(own.continuity_state), self_may_claim=bool(may_claim_same_as_before(own)), ok=True)
    except Exception as exc:                                                       # noqa: BLE001  recorded, not dropped
        out.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:160]}")
    return out


NUMERIC = ("m_t_posterior_mean", "m_t_uncertainty", "p_t_estimate", "p_t_posterior", "regime_means")


def compare_one(r: dict, n: dict) -> dict:
    row = dict(seed=r["seed"], fit_state=(r["fit_state"], n["fit_state"]), K=(r["K"], n["K"]), Dmax=(r["Dmax"], n["Dmax"]), withheld=(r["withheld_reason"], n["withheld_reason"]), collapsed=(r["fit_collapsed"], n["fit_collapsed"]),
               unstable=((r["unstable"], r["unstable_reason"]), (n["unstable"], n["unstable_reason"])))
    row["same_discrete"] = all(row[k][0] == row[k][1] for k in ("fit_state", "K", "Dmax", "withheld", "collapsed", "unstable"))
    row["bic_abs"] = abs(float(r["BIC"]) - float(n["BIC"]))
    row["bic_rel"] = row["bic_abs"] / max(abs(float(r["BIC"])), 1e-300)
    row["bic_by_k_abs"] = max([abs(r["bic_by_k"][k] - n["bic_by_k"][k]) for k in r["bic_by_k"] if k in n["bic_by_k"]] + [0.0]) if set(r["bic_by_k"]) == set(n["bic_by_k"]) else float("inf")
    row["bit_identical_outputs"] = all(np.array_equal(r[k], n[k]) for k in NUMERIC) and r["duration_parameters"] == n["duration_parameters"]
    row["max_abs"] = {k: max_abs_diff(r[k], n[k]) for k in NUMERIC}
    row["duration_params_abs"] = duration_diff({k: np.asarray(v) for k, v in r["duration_parameters"].items()}, {k: np.asarray(v) for k, v in n["duration_parameters"].items()})
    row["label_agree"], row["labels_equal"] = label_agreement(r["labels"], n["labels"])
    row["restart_summary_diff"] = sorted(k for k in set(r["restart_summary"]) | set(n["restart_summary"]) if r["restart_summary"].get(k) != n["restart_summary"].get(k))
    row["restart_summary_float_abs"] = {k: abs(float(r["restart_summary"][k]) - float(n["restart_summary"][k])) for k in row["restart_summary_diff"]
                                        if isinstance(r["restart_summary"].get(k), (int, float)) and isinstance(n["restart_summary"].get(k), (int, float)) and not isinstance(r["restart_summary"].get(k), bool)}
    row["separation"] = (r["separation_min"], n["separation_min"])
    row["lineage"] = lineage_outcome(r, n)
    return row


def compare_all(ref: list, new: list) -> dict:
    assert [r["seed"] for r in ref] == [r["seed"] for r in new]
    rows = [compare_one(r, n) for r, n in zip(ref, new)]
    N = len(rows)
    lin = [r["lineage"] for r in rows]
    return dict(n=N, rows=rows, bit_identical=sum(r["bit_identical_outputs"] for r in rows), same_discrete=sum(r["same_discrete"] for r in rows), labels_equal=sum(r["labels_equal"] for r in rows),
                max_bic_abs=max(r["bic_abs"] for r in rows), max_bic_rel=max(r["bic_rel"] for r in rows), min_label_agree=min(r["label_agree"] for r in rows),
                max_abs={k: max(r["max_abs"][k] for r in rows) for k in NUMERIC}, max_duration_abs=max(r["duration_params_abs"] for r in rows),
                served_ids_equal=sum(bool(l.get("served_ids_equal")) for l in lin), lineage_ok=sum(bool(l.get("ok")) for l in lin), aligned=sum(l.get("next_state") == "ALIGNED" for l in lin),
                may_claim=sum(bool(l.get("may_claim")) for l in lin), self_aligned=sum(l.get("self_state") == "ALIGNED" for l in lin), aligned_other_not_self=sum(l.get("next_state") == "ALIGNED" and l.get("self_state") != "ALIGNED" for l in lin),
                aligned_self_not_other=sum(l.get("self_state") == "ALIGNED" and l.get("next_state") != "ALIGNED" for l in lin),
                states_self=sorted({str(l.get("self_state")) for l in lin}), states_other=sorted({str(l.get("next_state")) for l in lin}),
                restart_summary_float_abs={k: max(r["restart_summary_float_abs"].get(k, 0.0) for r in rows) for k in sorted({k for r in rows for k in r["restart_summary_float_abs"]})}, states_ref=sorted({r["fit_state"][0] for r in rows}), n_withheld_ref=sum(1 for r in rows if r["withheld"][0] is not None),
                restart_summary_keys_differing=sorted({k for r in rows for k in r["restart_summary_diff"]}))


def render(name: str, c: dict) -> str:
    L = [f"{name}: {c['n']} users"]
    L.append(f"  bit-identical fitted outputs (m_t, uncertainty, p_t, posteriors, regime means, duration parameters): {c['bit_identical']} of {c['n']}")
    L.append(f"  same fit state, K, Dmax, withheld reason, collapse flag and stability flag/reason: {c['same_discrete']} of {c['n']}")
    L.append(f"  BIC: largest absolute difference {c['max_bic_abs']:.3g} (relative {c['max_bic_rel']:.3g})")
    L.append(f"  decoded regime labels: exactly equal sequences {c['labels_equal']} of {c['n']}; lowest agreement after renaming {c['min_label_agree']:.4f}")
    L.append("  largest absolute difference of an output over the users: " + ", ".join(f"{k} {v:.3g}" for k, v in c["max_abs"].items()) + f"; duration parameters {c['max_duration_abs']:.3g}")
    L.append(f"  restart_summary keys that differ in at least one user: {c['restart_summary_keys_differing'] or 'none'}" + (f" (largest absolute difference of a numeric one: {c['restart_summary_float_abs']})" if c["restart_summary_float_abs"] else ""))
    L.append(f"  served lineage: served stable ids equal when each fit starts a lineage {c['served_ids_equal']} of {c['n']}; the other configuration's fit is ALIGNED with the first {c['aligned']} of {c['n']} (may claim 'same as before': {c['may_claim']}); lineage step ran without error {c['lineage_ok']} of {c['n']}")
    L.append(f"  the SAME fit fed twice to the lineage (what the gate does on its own, whatever the machine): ALIGNED {c['self_aligned']} of {c['n']} (states {c['states_self']}); ALIGNED only with the other configuration's fit: {c['aligned_other_not_self']}; ALIGNED only with itself: {c['aligned_self_not_other']} (states with the other configuration's fit: {c['states_other']})")
    L.append(f"  (reference fits: states {c['states_ref']}, withheld {c['n_withheld_ref']} of {c['n']})")
    bad = [r for r in c["rows"] if not r["bit_identical_outputs"] or not r["same_discrete"] or not r["labels_equal"]]
    for r in bad[:12]:
        L.append(f"    user {r['seed']}: state {r['fit_state']}, K {r['K']}, withheld {r['withheld']}, unstable {r['unstable']}, label agreement {r['label_agree']:.3f}, BIC diff {r['bic_abs']:.3g}, lineage {r['lineage'].get('next_state')}")
    return "\n".join(L)


def cmd_compare(a):
    ref, new = (pickle.load(open(p, "rb")) for p in (a.ref, a.new))
    c = compare_all(ref, new)
    txt = render(f"{Path(a.ref).name} vs {Path(a.new).name}", c)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n")
    ok = c["bit_identical"] == c["n"] and c["same_discrete"] == c["n"]
    print("BIT-IDENTICAL on every user" if ok else "NOT bit-identical")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit"); p.add_argument("--tree", required=True); p.add_argument("--variant", choices=("fast", "full"), required=True); p.add_argument("--out", required=True); p.add_argument("--workers", type=int, default=2); p.set_defaults(f=cmd_fit)
    p = sub.add_parser("drive"); p.add_argument("--outdir", required=True); p.add_argument("--variants", default="fast,full"); p.add_argument("--workers", type=int, default=2); p.set_defaults(f=cmd_drive)
    p = sub.add_parser("compare"); p.add_argument("ref"); p.add_argument("new"); p.add_argument("--out", default=""); p.set_defaults(f=cmd_compare)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
