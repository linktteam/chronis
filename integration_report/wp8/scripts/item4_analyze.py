"""WP8 item 4: development analysis (seeds 8201..8212 only; never the evaluation seeds). Applies the rules declared in tests/PREDECLARED_CRITERIA_WP8.md section 8
('Pre-declaration (Item 4)' and its clarification) to the fits of item4_run.py:
  1. every consecutive-refit alignment (and every engine-switch pair) with all margins, for each distance candidate, stored in wp8_evidence/item4_alignments.jsonl;
  2. the identifiability gate h by the declared rule (baseline development alignments, raw distance);
  3. margin thresholds per pair class and candidate, the coverage they give, and the selection.
usage: PYTHONPATH=. python3 integration_report/wp8/scripts/item4_analyze.py wp8_evidence/item4_dev_fits.jsonl wp8_evidence/item4_analysis.txt wp8_evidence/item4_alignments.jsonl wp8_evidence/item4_thresholds.json"""
import json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identity_lib import load_records, overlap_map, stream_for
from backbone.hssm.lineage import AMBIGUITY_RATIO, alignment, separation_min

fits_path, out_path, ali_path, thr_path = sys.argv[1:5]
L = []
def P(*a):
    s = " ".join(str(x) for x in a); L.append(s); print(s)
def pct(a, b): return f"{a}/{b} = {100.0 * a / b:.1f}%" if b else "0/0"

# ------------------------------------------------------------------------------------------------- fits
_streams = {}
def window_raw(j):
    key = (j["user"], j["drift"])
    if key not in _streams:
        _streams[key] = stream_for(j)
    return _streams[key].window(j["start"], j["end"])["raw"]

recs = [r for r in load_records(fits_path) if r["ok"] and r["fit_state"] not in ("FAILED", "NUMERICAL_FAILURE")]
info = {}
for r in recs:
    j = r["job"]
    mp, rec = overlap_map(r["labels_fit"], r["labels_true"], r["K"], 4)
    raw_means = np.array(r["raw_means"])
    info[(j["engine"], j["user"], j["drift"], j["group"][-1], j["start"], j["end"])] = dict(
        engine=j["engine"], user=j["user"], drift=j["drift"], proto=j["group"][-1], start=j["start"], end=j["end"], K=r["K"], mp=mp, recoverable=rec,
        raw_means=raw_means, z_means=np.array(r["z_means"]), scale=np.array(r["scaler_std"]), exp_dur=np.array(r["exp_dur"]), accepted=bool(r["accepted"]),
        sep=separation_min(raw_means, window_raw(j), np.array(r["labels_fit"])))
P(f"development fits read: {len(recs)} (baseline {sum(i['engine'] == 'baseline' for i in info.values())}, dual {sum(i['engine'] == 'dual' for i in info.values())}); accepted dual {sum(i['accepted'] for i in info.values() if i['engine'] == 'dual')}")

MODES = ("z", "raw", "dur0.5", "dur1")
def vec(f, mode, scale):
    if mode == "z":
        return f["z_means"]
    v = f["raw_means"] / scale
    if mode.startswith("dur"):
        v = np.hstack([v, np.sqrt(float(mode[3:])) * np.log(f["exp_dur"])[:, None]])
    return v

def make_pair(cls, a, n):
    row = dict(cls=cls, engine_anchor=a["engine"], engine_new=n["engine"], user=a["user"], drift=a["drift"], proto=a["proto"], anchor_window=[a["start"], a["end"]], new_window=[n["start"], n["end"]],
               anchor_recoverable=a["recoverable"], new_recoverable=n["recoverable"], sep_anchor=a["sep"], sep_new=n["sep"], modes={})
    for mode in MODES:
        al = alignment(vec(a, mode, a["scale"]), vec(n, mode, a["scale"]))
        perm = al["perm"]
        correct = bool(a["recoverable"] and n["recoverable"] and all(a["mp"][i] == n["mp"][perm[i]] for i in range(a["K"])))
        row["modes"][mode] = dict(perm=list(perm), correct=correct, margin_diff=al["margin_diff"], margin_ratio=al["margin_ratio"], C1=al["C1"], C2=al["C2"])
    return row

rows = []
streams = defaultdict(list)
for key, f in info.items():
    streams[(f["engine"], f["user"], f["drift"], f["proto"])].append(f)
for (eng, user, drift, proto), fl in sorted(streams.items()):
    fl.sort(key=lambda f: f["end"])
    for a, n in zip(fl[:-1], fl[1:]):
        rows.append(make_pair("baseline" if eng == "baseline" else "dual", a, n))
for (eng, user, drift, proto, s, e), f in info.items():                  # engine switch: the same window fitted by the two engines, both directions
    if eng == "baseline":
        d = info.get(("dual", user, drift, proto, s, e))
        if d is not None:
            rows.append(make_pair("cross", f, d)); rows.append(make_pair("cross", d, f))
with open(ali_path, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
by_cls = {c: [r for r in rows if r["cls"] == c] for c in ("baseline", "dual", "cross")}
for c, rr in by_cls.items():
    n_both = sum(r["anchor_recoverable"] and r["new_recoverable"] for r in rr)
    err = sum(not r["modes"]["raw"]["correct"] for r in rr if r["anchor_recoverable"] and r["new_recoverable"])
    P(f"class {c}: {len(rr)} alignments; both fits recoverable {pct(n_both, len(rr))}; permutation errors among those (raw distance): {err}; strict-correct overall {pct(sum(r['modes']['raw']['correct'] for r in rr), len(rr))}")

# ------------------------------------------------------------------------------------------------- the gate
GRID = [0.5 * i for i in range(0, 11)]
def passes(r, h): return r["sep_anchor"] >= h and r["sep_new"] >= h
P("\n== identifiability gate h (rule: smallest grid value with no strict-wrong alignment among those whose two fits are both >= h; baseline class, raw distance)")
P("  h     pass share (all alignments)   strict-wrong among passing   recoverable-pair permutation errors among passing")
base = by_cls["baseline"]
chosen_h = None; one_pct_h = None
for h in GRID:
    ps = [r for r in base if passes(r, h)]
    wrong = sum(not r["modes"]["raw"]["correct"] for r in ps)
    perm_err = sum(not r["modes"]["raw"]["correct"] for r in ps if r["anchor_recoverable"] and r["new_recoverable"])
    P(f"  {h:3.1f}   {pct(len(ps), len(base)):>22}   {wrong:>8}   {perm_err:>8}")
    if chosen_h is None and ps and wrong == 0:
        chosen_h = h
    if one_pct_h is None and ps and wrong / len(ps) <= 0.01:
        one_pct_h = h
if chosen_h is None:
    chosen_h = one_pct_h if one_pct_h is not None else 3.0
    P(f"  no grid value gives 0 strict-wrong passing alignments; {'smallest value with <= 1% strict-wrong: ' + str(chosen_h) if one_pct_h is not None else 'no value reaches 1% either: h = 3.0 (declared fallback), SHORTFALL'}")
else:
    P(f"  h = {chosen_h} (the smallest grid value with 0 strict-wrong passing alignments)")
h = chosen_h
for c in ("dual", "cross"):
    rr = by_cls[c]
    if rr:
        ps = [r for r in rr if passes(r, h)]
        P(f"  same h on class {c}: pass share {pct(len(ps), len(rr))}; strict-wrong among passing {sum(not r['modes']['raw']['correct'] for r in ps)}")

# ------------------------------------------------------------------------------------------------- margin thresholds
def candidates(rr, mode):
    ps = [r for r in rr if passes(r, h)]
    wrong = [r["modes"][mode] for r in ps if not r["modes"][mode]["correct"]]
    out = {}
    tau = 0.0 if not wrong else math.ceil((max(w["margin_diff"] for w in wrong) + 1e-9) * 100) / 100
    out["a"] = dict(tau=tau, vacuous=(tau == 0.0))
    rho_star = AMBIGUITY_RATIO if not wrong else min(AMBIGUITY_RATIO, math.floor((min(w["margin_ratio"] for w in wrong) - 1e-9) * 100) / 100)
    out["b"] = dict(rho=max(rho_star, 0.0), vacuous=False)
    out["c"] = dict(tau=tau, rho=max(rho_star, 0.0), vacuous=(tau == 0.0 and rho_star >= AMBIGUITY_RATIO))
    for k, v in out.items():
        def acc(r):
            m = r["modes"][mode]
            if not passes(r, h) or not m["C1"] < m["C2"]:
                return False
            ok_a = m["margin_diff"] >= v.get("tau", 0.0); ok_b = m["margin_ratio"] <= v.get("rho", AMBIGUITY_RATIO)
            return {"a": ok_a, "b": ok_b, "c": ok_a and ok_b}[k]
        served = [r for r in rr if acc(r)]
        v["coverage"] = len(served) / len(rr) if rr else float("nan")
        v["wrong_served"] = sum(not r["modes"][mode]["correct"] for r in served)
        v["served"] = len(served); v["n_passing_gate"] = len(ps)
    return out

summary = dict(h=h, classes={})
SIMPLE = {"raw": 0, "dur0.5": 1, "dur1": 2}; ACC = {"b": 0, "a": 1, "c": 2}
for c in ("baseline", "dual", "cross"):
    rr = by_cls[c]
    P(f"\n== margin thresholds, class {c}: {len(rr)} alignments, {sum(passes(r, h) for r in rr)} pass the gate (h = {h})")
    if sum(passes(r, h) for r in rr) < 20:
        P(f"  fewer than 20 alignments pass the gate: thresholds for this class are UNKNOWN (reported, not chosen)")
        summary["classes"][c] = dict(unknown=True, n=len(rr), n_passing=sum(passes(r, h) for r in rr))
        continue
    best = None
    for mode in ("z", "raw", "dur0.5", "dur1"):
        cand = candidates(rr, mode)
        for k in ("a", "b", "c"):
            v = cand[k]
            P(f"  distance {mode:6} rule ({k}) {'vacuous (excluded)' if v['vacuous'] else '':18} tau={v.get('tau', '-')} rho={v.get('rho', '-')}: coverage {100 * v['coverage']:.1f}% ({v['served']}/{len(rr)}), wrong served {v['wrong_served']}")
            if mode != "z" and not v["vacuous"] and v["wrong_served"] == 0:
                key = (-round(v["coverage"], 6), SIMPLE[mode], ACC[k])
                if best is None or key < best[0]:
                    best = (key, mode, k, v)
    if best is None:
        P("  no non-vacuous candidate with 0 wrong served"); summary["classes"][c] = dict(unknown=True, n=len(rr)); continue
    _, mode, k, v = best
    P(f"  CHOSEN: distance {mode}, rule ({k}), tau={v.get('tau', '-')}, rho={v.get('rho', '-')}: coverage {100 * v['coverage']:.1f}% ({v['served']}/{len(rr)}), 0 wrong served on development"
      f"{'' if v['coverage'] >= 0.8 else '  SHORTFALL: below the declared 80%, reported, criteria not loosened'}")
    # reported next to it: strict versus conditional
    cond = [r for r in rr if r["anchor_recoverable"] and r["new_recoverable"]]
    P(f"  context: both-fits-recoverable pairs {len(cond)}; among them permutation errors {sum(not r['modes'][mode]['correct'] for r in cond)}; ALIGNED among all alignments {pct(v['served'], len(rr))}")
    summary["classes"][c] = dict(unknown=False, distance=mode, rule=k, tau=v.get("tau"), rho=v.get("rho"), coverage=v["coverage"], n=len(rr), served=v["served"], n_passing=v["n_passing_gate"])

# ------------------------------------------------------------------------------------------------- sensitivity of the gate (reported)
P("\n== reported: coverage and wrong-served of the chosen baseline rule for other gate values")
chosen = summary["classes"].get("baseline")
if chosen and not chosen.get("unknown"):
    for hh in sorted({0.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, h}):
        served = []
        for r in base:
            m = r["modes"][chosen["distance"]]
            if passes(r, hh) and m["C1"] < m["C2"] and (chosen["rule"] == "a" and m["margin_diff"] >= chosen["tau"] or chosen["rule"] == "b" and m["margin_ratio"] <= chosen["rho"]
                                                         or chosen["rule"] == "c" and m["margin_diff"] >= chosen["tau"] and m["margin_ratio"] <= chosen["rho"]):
                served.append(r)
        P(f"  h={hh:3.1f}: coverage {pct(len(served), len(base))}; strict-wrong served {sum(not r['modes'][chosen['distance']]['correct'] for r in served)}; permutation errors served {sum(not r['modes'][chosen['distance']]['correct'] for r in served if r['anchor_recoverable'] and r['new_recoverable'])}")
P("\n== by drift (baseline class, chosen rule): ALIGNED share and strict-wrong served")
if chosen and not chosen.get("unknown"):
    for d in (0.0, 1.0, 2.5):
        rr = [r for r in base if r["drift"] == d]
        served = []
        for r in rr:
            m = r["modes"][chosen["distance"]]
            if passes(r, h) and m["C1"] < m["C2"] and (chosen["rule"] == "a" and m["margin_diff"] >= chosen["tau"] or chosen["rule"] == "b" and m["margin_ratio"] <= chosen["rho"]
                                                         or chosen["rule"] == "c" and m["margin_diff"] >= chosen["tau"] and m["margin_ratio"] <= chosen["rho"]):
                served.append(r)
        P(f"  drift {d}: {len(rr)} alignments, ALIGNED {pct(len(served), len(rr))}, strict-wrong served {sum(not r['modes'][chosen['distance']]['correct'] for r in served)}")
Path(out_path).write_text("\n".join(L) + "\n")
Path(thr_path).write_text(json.dumps(summary, indent=1))
