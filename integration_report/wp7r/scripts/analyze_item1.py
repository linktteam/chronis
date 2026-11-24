"""Applies the words of tests/PREDECLARED_CRITERIA_WP7R.md section 1 to wp7r_evidence/item1_iter.jsonl. usage: python3 analyze_item1.py [jsonl] [s3.jsonl]"""
import json, sys, math
import numpy as np
from scipy.stats import fisher_exact

path = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/work/hssm/wp7r_evidence/item1_iter.jsonl"
recs = {}
for ln in open(path):
    d = json.loads(ln)
    if d.get("ok"):
        recs[(d["group"], d["seed"])] = d
CP = ["20", "40", "80", "160"]
groups = {"N18": [r for k, r in sorted(recs.items()) if k[0] == "N18"], "DPOP": [r for k, r in sorted(recs.items()) if k[0] == "DPOP"]}
groups["POOLED"] = groups["N18"] + groups["DPOP"]
print("seeds done:", {g: len(v) for g, v in groups.items()})
bad = [(r["group"], r["seed"], c) for r in groups["POOLED"] for c in CP if not r["checkpoints"][c]["heldout_finite"]]
print("non-finite held-out scores:", bad)
def stat(d):
    d = np.asarray(d, float); n = len(d)
    return (d.mean(), d.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan"), n, int((d > 0).sum()))
for g, rs in groups.items():
    if not rs: continue
    print(f"\n== {g} (n={len(rs)})")
    print("mean held-out by checkpoint:", {c: round(float(np.mean([r["checkpoints"][c]["heldout"] for r in rs])), 4) for c in CP})
    print("mean train ll/row:          ", {c: round(float(np.mean([r["checkpoints"][c]["train_ll_per_row"] for r in rs])), 4) for c in CP})
    print("share dual > Gaussian:      ", {c: round(float(np.mean([r["checkpoints"][c]["heldout"] > r["gauss_heldout"] for r in rs])), 3) for c in CP}, " mean dual-Gauss:", {c: round(float(np.mean([r["checkpoints"][c]["heldout"] - r["gauss_heldout"] for r in rs])), 3) for c in CP})
    print("degenerate at checkpoint:   ", {c: int(sum(r["checkpoints"][c]["degenerate"] for r in rs)) for c in CP}, " fit states:", {s: sum(1 for r in rs if r["fit_state"] == s) for s in set(r["fit_state"] for r in rs)})
    for a, b in (("20", "40"), ("40", "80"), ("80", "160"), ("20", "160")):
        h = stat([r["checkpoints"][b]["heldout"] - r["checkpoints"][a]["heldout"] for r in rs]); t = stat([r["checkpoints"][b]["train_ll_per_row"] - r["checkpoints"][a]["train_ll_per_row"] for r in rs])
        print(f"  {a}->{b}: held-out diff mean {h[0]:+.4f} se {h[1]:.4f} n={h[2]} better:{h[3]}   | train ll/row diff mean {t[0]:+.4f} (higher in {t[3]})")
x = np.log2(np.array([20, 40, 80, 160], float)); xc = x - x.mean()
def slopes(rs): return np.array([float(np.dot(xc, [r["checkpoints"][c]["heldout"] for c in CP]) / np.dot(xc, xc)) for r in rs])
sl = {g: slopes(rs) for g, rs in groups.items() if rs}
P = sl["POOLED"]; S = P.mean(); se = P.std(ddof=1) / math.sqrt(len(P))
print(f"\nslope (nats/step per doubling): pooled mean {S:+.4f} se {se:.4f} t {S/se:+.2f}; negative {int((P<0).sum())}/{len(P)}; N18 mean {sl['N18'].mean():+.4f}; DPOP mean {sl['DPOP'].mean():+.4f}")
lower = S < 0 and S / se <= -2 and (P < 0).sum() >= 20 and sl["N18"].mean() < 0 and sl["DPOP"].mean() < 0
raise_ = S > 0 and S / se >= 2 and (P > 0).sum() >= 20 and sl["N18"].mean() > 0 and sl["DPOP"].mean() > 0
print("WORD (1.3):", "MORE EM LOWERS the held-out score (a trend)" if lower else "MORE EM RAISES the held-out score" if raise_ else "NO CONSISTENT TREND")
# S1
rates, ext, conv = [], [], 0
for r in groups["POOLED"]:
    h = np.array(r["ll_hist"]); inc = np.diff(h)
    if r["fit_state"] == "CONVERGED" and r["n_iter_run"] < 160: conv += 1; continue
    rr = [inc[t] / inc[t - 1] for t in range(99, len(inc)) if inc[t] > 0 and inc[t - 1] > 0]
    if rr:
        rates.append(float(np.median(rr)))
        d160 = inc[-1]
        if 0 < rates[-1] < 1 and d160 > 1e-4: ext.append(160 + math.log(1e-4 / d160) / math.log(rates[-1]))
print(f"\nS1: seeds stopped before 160 by abs_tol: {conv}; median rate {np.median(rates):.4f} (n={len(rates)}); median extrapolated iterations {np.median(ext):.0f} (n={len(ext)}); range {min(ext):.0f}..{max(ext):.0f}")
print("S1 word:", "RE-VERIFIED (rate and count in range); the observation-block share is NOT RE-VERIFIED" if 0.99 <= np.median(rates) <= 0.999 and 700 <= np.median(ext) <= 2500 else "NOT RE-VERIFIED")
# S2
deg = {(r["group"], r["seed"]): r["checkpoints"]["160"]["degenerate"] for r in groups["POOLED"]}
lng = {(r["group"], r["seed"]): r["n_seg_over_dmax"] > 0 for r in groups["POOLED"]}
a = sum(deg[k] and lng[k] for k in deg); b = sum(deg[k] and not lng[k] for k in deg); c = sum((not deg[k]) and lng[k] for k in deg); d = sum((not deg[k]) and (not lng[k]) for k in deg)
orr, p = fisher_exact([[a, b], [c, d]])
print(f"\nS2: degenerate&long {a}, degenerate&not-long {b}, ok&long {c}, ok&not-long {d}; Fisher two-sided p {p:.4f} odds ratio {orr:.2f}")
print("S2 word:", "RE-VERIFIED" if p < 0.05 and orr > 1 else "NOT RE-VERIFIED")
print("degenerate seeds at 160 (S3 candidates):", sorted(k[1] for k in deg if deg[k]), "| by group:", {g: sorted(k[1] for k in deg if deg[k] and k[0] == g) for g in ("N18", "DPOP")})
if len(sys.argv) > 2:
    s3 = [json.loads(l) for l in open(sys.argv[2])]
    cured = sum(1 for r in s3 if r.get("ok") and not r["checkpoints"]["160"]["degenerate"])
    print(f"\nS3: cured {cured} of {len(s3)} at Dmax=60; word:", "RE-VERIFIED" if len(s3) >= 5 and cured / len(s3) >= 0.75 else "NOT RE-VERIFIED")
