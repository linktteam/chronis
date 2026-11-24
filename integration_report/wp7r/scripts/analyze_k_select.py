"""Applies the words of tests/PREDECLARED_CRITERIA_WP7R.md section 2 to wp7r_evidence/item2_kselect.jsonl. usage: python3 analyze_k_select.py [jsonl]"""
import json, math, sys, collections
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/work/hssm/wp7r_evidence/item2_kselect.jsonl"
recs = [json.loads(l) for l in open(path)]
bad = [r for r in recs if not r.get("ok")]
recs = [r for r in recs if r.get("ok")]
print("datasets ok:", len(recs), "errors:", [(r["design"], r["seed"], r.get("error")) for r in bad])

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 2), round(c + h, 2)

def derive(r, thr=4.0, conv="legacy"):
    n_obs = r["n_obs"]; out = {}
    for ks, v in r["per_k"].items():
        k = int(ks)
        conv_runs = [x for x in v["run_log"] if x["fit_state"] == "CONVERGED" and x["ll"] is not None]
        if not conv_runs:
            out[k] = None; continue
        healthy = [x for x in conv_runs if x["min_eff"] is not None and x["min_eff"] >= thr]
        pool = healthy if healthy else conv_runs
        sel = max(pool, key=lambda x: x["ll"])            # first maximum, as the strict ">" of the harness
        npar = v["n_params"] - (k if conv == "free" else 0)
        out[k] = dict(ll=sel["ll"], min_eff=sel["min_eff"], collapsed=bool(sel["min_eff"] is not None and sel["min_eff"] < thr), bic=-2 * sel["ll"] + npar * math.log(max(n_obs, 1)))
    ok = {k: v for k, v in out.items() if v}
    healthy = {k: v for k, v in ok.items() if not v["collapsed"]}
    pool = healthy if healthy else ok
    sel_k = min(pool, key=lambda k: pool[k]["bic"])
    raw_k = min(ok, key=lambda k: ok[k]["bic"])
    return dict(sel=sel_k, raw=raw_k, per=out, withheld_collapse=ok[sel_k]["collapsed"])

designs = sorted(set(r["design"] for r in recs))
# validity gate at 4 days
gate = []
for r in recs:
    d = derive(r, 4.0, "legacy")
    same_k = d["sel"] == r["selected_k"]
    same_flags = all(bool(d["per"][int(k)]["collapsed"]) == bool(v["collapsed"]) for k, v in r["per_k"].items() if d["per"][int(k)])
    same_bic = all(abs(d["per"][int(k)]["bic"] - r["bic_by_k"][k]) < 1e-6 for k in r["per_k"] if d["per"][int(k)])
    same_withheld = d["withheld_collapse"] == any(s.startswith("collapsed_regime") for s in str(r["withheld_reason"] or "").split("; "))
    gate.append(same_k and same_flags and same_bic and same_withheld)
print(f"\nVALIDITY GATE (offline re-derivation at 4 days reproduces the actual selected K, collapse flags, BICs and collapsed-withheld status): {sum(gate)}/{len(gate)}")

for dz in designs:
    R = [r for r in recs if r["design"] == dz]
    print(f"\n================ design {dz} (n={len(R)})")
    tot = sum(r["selected_k"] == r["planted_k"] for r in R)
    byk = {pk: (sum(r["selected_k"] == pk for r in R if r["planted_k"] == pk), sum(1 for r in R if r["planted_k"] == pk)) for pk in (2, 3, 4)}
    print("planted-K recovery:", f"{tot}/{len(R)}", wilson(tot, len(R)), {pk: f"{a}/{b} {wilson(a, b)}" for pk, (a, b) in byk.items()})
    print("selected K distribution by planted K:", {pk: dict(collections.Counter(r["selected_k"] for r in R if r["planted_k"] == pk)) for pk in (2, 3, 4)})
    print("WORD recovery:", "BIC RECOVERS the planted K" if tot >= 20 and all(a >= 5 for a, b in byk.values()) else "BIC does NOT reliably recover the planted K")
    # collapse handling
    served_collapsed = [r["seed"] for r in R if not r["insights_withheld"] and (r["fit_collapsed"] or r["per_k"][str(r["selected_k"])]["collapsed"])]
    print("hard (i) served collapsed fits:", len(served_collapsed), served_collapsed)
    viol = []
    for r in R:
        coll = {int(k): bool(v["collapsed"]) for k, v in r["per_k"].items()}
        if coll.get(r["selected_k"]) and any(not c for k, c in coll.items()): viol.append(r["seed"])
    print("hard (ii) collapsed K selected while a healthy candidate exists:", len(viol), viol)
    anyc = [r for r in R if any(v["collapsed"] for v in r["per_k"].values())]
    changed = []; helps = harms = 0
    for r in R:
        d = derive(r, 4.0, "legacy")
        if d["sel"] != d["raw"]:
            changed.append(r["seed"]); helps += (d["raw"] != r["planted_k"] and d["sel"] == r["planted_k"]); harms += (d["raw"] == r["planted_k"] and d["sel"] != r["planted_k"])
    print(f"(iii) datasets with some collapsed candidate K: {len(anyc)}; lowest-BIC K collapsed (demotion changes the choice): {len(changed)} {changed}; helps {helps}, harms {harms}")
    print("WORD demotion:", "never triggered" if not changed else "helps" if helps and not harms else "harms" if harms and not helps else "mixed" if helps and harms else "triggered, neutral for planted-K recovery")
    sv = [r for r in R if not r["insights_withheld"]]
    ar = [r["served_ari"] for r in sv]
    print(f"served (not withheld): {len(sv)}/{len(R)}; served ARI median {np.median(ar):.3f} min {min(ar):.3f} (n={len(ar)})" if ar else "no served fits")
    print("WORD ARI:", ("ADEQUATE (median >= 0.80)" if ar and np.median(ar) >= 0.80 else "NOT adequate (median < 0.80)") + ("" if dz == "easy" else "  [hard design: reported only]"))
    wr = collections.Counter(str(r["withheld_reason"]).split(":")[0] if r["insights_withheld"] else "served" for r in R)
    print("withheld/served by reason:", dict(wr), " by planted K:", {pk: sum(r["insights_withheld"] for r in R if r["planted_k"] == pk) for pk in (2, 3, 4)})
    print("fit states:", dict(collections.Counter(r["fit_state"] for r in R)))
    wa = [r["served_ari"] for r in R if r["insights_withheld"]]
    if wa: print("ARI of the withheld fits (descriptive): median %.3f min %.3f" % (np.median(wa), min(wa)))
    # P1-01
    dfree = [derive(r, 4.0, "free") for r in R]; dleg = [derive(r, 4.0, "legacy") for r in R]
    ch = [r["seed"] for r, a, b in zip(R, dfree, dleg) if a["sel"] != b["sel"]]
    rec_free = sum(a["sel"] == r["planted_k"] for r, a in zip(R, dfree)); rec_leg = sum(b["sel"] == r["planted_k"] for r, b in zip(R, dleg))
    print(f"P1-01: legacy recovery {rec_leg}/{len(R)}, free-count recovery {rec_free}/{len(R)}; datasets whose selected K changes under the free count: {len(ch)} {ch}")
    print("WORD P1-01:", "THE COUNT MOVES MODEL SELECTION (owner approval needed; default stays)" if ch else "the count does not move selection in this study")
    # 4-day sensitivity
    for thr in (3.0, 4.0, 6.0):
        dd = [derive(r, thr, "legacy") for r in R]
        rec_t = sum(x["sel"] == r["planted_k"] for r, x in zip(R, dd)); diff = [r["seed"] for r, x, y in zip(R, dd, dleg) if x["sel"] != y["sel"]]
        print(f"  threshold {thr:g} d: recovery {rec_t}/{len(R)}, withheld by collapse {sum(x['withheld_collapse'] for x in dd)}/{len(R)}, selected K differs from the 4-day choice in {len(diff)} {diff}")
fr = [(r["design"], r["seed"], r["planted_k"], r["selected_k"]) for r in recs if r["selected_k"] != r["planted_k"]]
print("\nwrong-K datasets:", fr)
