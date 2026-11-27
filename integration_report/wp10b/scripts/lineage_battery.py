"""WP10B F3 (bit-identity of healthy fits): 48 chains x 8 fits of HEALTHY summaries through `lineage.step` / `apply_user_choice` in two trees (reference = a clean `git archive 95dd125`; new = a clean export of the head).
Healthy = no crash anywhere: every K-change record is None, or comes from the real `assess_k_change` with a deterministic stand-in for the recent-window fit that never raises. The chains cover the four states
(ALIGNED, K_CHANGED, NEW_LINEAGE, CONTINUITY_UNVERIFIED by dual engine / unstable fit / suspected K change), both engines, user choices, and the record without any assessment.
usage: python3 lineage_battery.py fit --tree TREE --out FILE.json     |     python3 lineage_battery.py compare REF.json NEW.json     (compare prints PASS or FAIL; exit code 0 / 1)
       python3 lineage_battery.py compare --tolerant REF.json NEW.json   (F4: two machines or OpenBLAS kernels; every non-float leaf must be identical, floats are reported by relative difference; prints SAME DISCRETE OUTPUTS or DIFFERENT DISCRETE OUTPUTS)
Compared: EVERY field of every LineageRecord (all dataclass fields present in BOTH trees; fields only in the new tree are listed), `may_claim_same_as_before`, `effective_choice`, and the state after each step (epoch, anchor fit
index, pending fit index, epoch-history length). Floats are compared exactly (repr round trip)."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import Counter

import numpy as np

N_CHAINS, N_STEPS = 48, 8


def canon(x):
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: canon(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, np.ndarray):
        return dict(__nd__=list(x.shape), v=[canon(v) for v in x.ravel().tolist()])
    if isinstance(x, (np.floating, float)):
        return float(x).hex() if np.isfinite(x) else repr(float(x))
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (list, tuple)):
        return [canon(v) for v in x]
    if isinstance(x, dict):
        return {str(k): canon(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    return x


def stub_fit_factory(anchor_means, case):
    """A deterministic, never-raising stand-in for the recent-window fit: K centres = the anchor means (K-1: the first K-1; K+1: plus the window mean shifted), BIC = squared distance + 20 K, with a case-dependent discount that
    makes the K+1 fit better (a split), the K-1 fit better (a merge) or neither."""
    Ka = anchor_means.shape[0]

    def fit(recent, K):
        recent = np.asarray(recent, dtype=float)
        if K <= Ka:
            centres = anchor_means[:K]
        else:
            centres = np.vstack([anchor_means, recent.mean(axis=0) + 3.0])
        d = ((recent[:, None, :] - centres[None]) ** 2).sum(-1)
        labels = d.argmin(1)
        bic = float(d.min(1).sum() + 20.0 * K)
        if case == "split" and K == Ka + 1:
            bic -= 1e4
        if case == "merge" and K == Ka - 1:
            bic -= 1e4
        return dict(bic=bic, labels=labels, raw_means=np.asarray(centres, dtype=float), accepted=True)
    return fit


def chain_summaries(c, L, KC, SCALE, draw_user):
    from itertools import cycle
    rng = np.random.default_rng(7000 + c)
    K = 2 + c % 3
    spec_means = draw_user(7000 + c, __import__("data.synthetic_identity", fromlist=["IdentitySpec"]).IdentitySpec(n_regimes=K, sep_lo=2.5, sep_hi=2.5, offset_sd=0.3))["means"]
    outs = []
    for t in range(N_STEPS):
        m = spec_means.copy()
        Kt = K
        engine = "baseline"
        unstable, why = False, None
        if c % 4 == 1 and t == 3:                                    # a K change: one regime more
            m = np.vstack([m, m[0] + 4.0 * SCALE]); Kt = K + 1
        if c % 6 == 2 and t == 5:                                    # far away: a new lineage
            m = m + 25.0 * SCALE
        if c % 5 == 0 and t == 4:
            engine = "dual"
        if c % 7 == 0 and t == 2:
            unstable, why = True, "restarts_disagree"
        T = 240
        labels = rng.permutation(np.repeat(np.arange(Kt), T // Kt + 1)[:T])
        window = m[labels] + 0.35 * SCALE * rng.standard_normal((T, m.shape[1]))
        case = ("split", "merge", "none")[(c + t) % 3]
        if (c + t) % 4 == 0 or (c % 2 == 0 and t % 2 == 0):
            kc = None
        else:
            var = np.ones(m.shape) * np.asarray(SCALE, dtype=float) ** 2
            kc = KC.assess_k_change(m, var, window, fit_fn=stub_fit_factory(m, case), k_anchor=Kt)
        outs.append((engine, m, window.mean(0), window.std(0), labels, window, unstable, why, kc, c % 3 == 0 and t == 6))
    return outs


def cmd_fit(a):
    sys.path.insert(0, a.tree)
    from backbone.hssm import k_change as KC
    from backbone.hssm import lineage as L
    from backbone.hssm.ordering import LadderConfig
    from data.synthetic_identity import REGISTRY, SCALE, draw_user
    cfg = L.AlignmentConfig(min_separation=1.0, ladder=LadderConfig(n_bootstrap=40, min_gap=0.5))
    out = dict(chains=[], new_fields=[], states=Counter())
    for c in range(N_CHAINS):
        state, recs = None, []
        for t, (engine, m, mu, sd, labels, window, unstable, why, kc, choose) in enumerate(chain_summaries(c, L, KC, SCALE, draw_user)):
            s = L.make_summary(engine, m, mu, sd, labels, window, fit_unstable=unstable, fit_unstable_reason=why, k_change=kc)
            rec, state = L.step(state, s, cfg, REGISTRY, lineage_id=f"c{c}")
            row = dict(t=t, record=canon(rec), may_claim=bool(L.may_claim_same_as_before(rec)), effective=str(L.effective_choice(rec)), state=dict(
                epoch=state.epoch, anchor=state.anchor.fit_index, pending=None if state.pending is None else state.pending.fit_index, history=len(state.epoch_history)))
            out["states"][rec.continuity_state] += 1
            if choose and rec.continuity_state in (L.CONTINUITY_UNVERIFIED, L.K_CHANGED) and state.pending is not None:
                rec2, state = L.apply_user_choice(state, rec, "NEW_NAMES" if c % 2 else "KEEP_OLD")
                row["after_choice"] = dict(record=canon(rec2), state=dict(epoch=state.epoch, anchor=state.anchor.fit_index, pending=None if state.pending is None else state.pending.fit_index, history=len(state.epoch_history)))
            recs.append(row)
        out["chains"].append(recs)
    out["states"] = dict(out["states"])
    out["record_fields"] = [f.name for f in dataclasses.fields(L.LineageRecord)]
    json.dump(out, open(a.out, "w"), sort_keys=True)
    print(f"{N_CHAINS} chains x {N_STEPS} fits; states: {out['states']}; record fields: {len(out['record_fields'])}")


def compare_dumps(ref, new):
    common = [f for f in ref["record_fields"] if f in new["record_fields"]]
    only_new = [f for f in new["record_fields"] if f not in ref["record_fields"]]
    only_ref = [f for f in ref["record_fields"] if f not in new["record_fields"]]

    def strip(rec):
        return {k: rec[k] for k in common}
    diffs, n_rec = [], 0
    for c, (cr, cn) in enumerate(zip(ref["chains"], new["chains"])):
        for t, (r, n) in enumerate(zip(cr, cn)):
            n_rec += 1
            if strip(r["record"]) != strip(n["record"]):
                diffs.append((c, t, [k for k in common if r["record"][k] != n["record"][k]]))
            for k in ("may_claim", "effective", "state"):
                if r[k] != n[k]:
                    diffs.append((c, t, [k]))
            if ("after_choice" in r) != ("after_choice" in n) or ("after_choice" in r and (strip(r["after_choice"]["record"]) != strip(n["after_choice"]["record"]) or r["after_choice"]["state"] != n["after_choice"]["state"])):
                diffs.append((c, t, ["after_choice"]))
    shape = len(ref["chains"]) == len(new["chains"]) and all(len(a) == len(b) for a, b in zip(ref["chains"], new["chains"]))
    ok = shape and not diffs and not only_ref and ref["states"] == new["states"]
    return dict(ok=ok, n_records=n_rec, diffs=diffs, only_new=only_new, only_ref=only_ref, states_ref=ref["states"], states_new=new["states"], common=len(common))


def leaves(x, path=""):
    """(path, value) of every leaf of a canonical dump; a float leaf is a hex string that parses with `float.fromhex` (or `nan`/`inf`)."""
    if isinstance(x, dict):
        for k in sorted(x):
            yield from leaves(x[k], f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, x


def as_float(v):
    if isinstance(v, str) and (v.startswith(("0x", "-0x")) or v in ("nan", "inf", "-inf")):
        try:
            return float.fromhex(v) if v[:2] in ("0x", "-0") else float(v)
        except ValueError:
            return None
    return None


def compare_tolerant(ref, new):
    """Machine / kernel comparison: every leaf that is not a float must be identical (states, reasons, names, flags, counts, shapes); floats are compared by relative difference."""
    n_rec = n_discrete_diff = n_float_diff = 0
    worst, worst_at = 0.0, None
    discrete_examples = []
    for c, (cr, cn) in enumerate(zip(ref["chains"], new["chains"])):
        for t, (r, n) in enumerate(zip(cr, cn)):
            n_rec += 1
            lr, ln = list(leaves(r)), list(leaves(n))
            d_diff = f_diff = False
            if [p for p, _ in lr] != [p for p, _ in ln]:
                d_diff = True
            else:
                for (p, a), (_, b) in zip(lr, ln):
                    fa, fb = as_float(a), as_float(b)
                    if fa is not None and fb is not None:
                        if fa != fb and not (fa != fa and fb != fb):
                            f_diff = True
                            rel = abs(fa - fb) / max(abs(fa), abs(fb), 1e-300)
                            if rel > worst:
                                worst, worst_at = rel, (c, t, p)
                    elif a != b:
                        d_diff = True
                        if len(discrete_examples) < 5:
                            discrete_examples.append((c, t, p, a, b))
            n_discrete_diff += d_diff
            n_float_diff += f_diff
    return dict(n_records=n_rec, n_discrete_diff=n_discrete_diff, n_float_diff=n_float_diff, worst_rel=worst, worst_at=worst_at, examples=discrete_examples, ok=n_discrete_diff == 0)


def cmd_compare(a):
    ref, new = (json.load(open(p)) for p in (a.ref, a.new))
    if a.tolerant:
        c = compare_tolerant(ref, new)
        print(f"records compared: {c['n_records']}; records with any non-float leaf different (state, reasons, names, flags, counts): {c['n_discrete_diff']}; records with any float leaf different: {c['n_float_diff']}; "
              f"largest relative difference of a float: {c['worst_rel']:.3g}" + (f" at {c['worst_at']}" if c["worst_at"] else ""))
        if c["examples"]:
            print("first discrete differences: " + str(c["examples"]))
        print("SAME DISCRETE OUTPUTS" if c["ok"] else "DIFFERENT DISCRETE OUTPUTS")
        sys.exit(0 if c["ok"] else 1)
    c = compare_dumps(ref, new)
    print(f"records compared: {c['n_records']} (+ the records after a user choice); fields compared per record: {c['common']}; fields only in the new tree: {c['only_new']}; only in the reference: {c['only_ref']}")
    print(f"states reached (reference): {c['states_ref']}\nstates reached (new):       {c['states_new']}")
    print(f"records with any difference: {len(c['diffs'])}" + (f"; first: {c['diffs'][:5]}" if c["diffs"] else ""))
    print("PASS" if c["ok"] else "FAIL")
    sys.exit(0 if c["ok"] else 1)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit"); p.add_argument("--tree", required=True); p.add_argument("--out", required=True); p.set_defaults(f=cmd_fit)
    p = sub.add_parser("compare"); p.add_argument("ref"); p.add_argument("new"); p.add_argument("--tolerant", action="store_true"); p.set_defaults(f=cmd_compare)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
