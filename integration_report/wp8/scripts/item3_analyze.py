"""WP8 item 3: development analysis of the ordering ladder on seeds 8201..8212 (never the evaluation seeds).
usage: PYTHONPATH=. python3 integration_report/wp8/scripts/item3_analyze.py wp8_evidence/item3_dev_fits.jsonl wp8_evidence/item3_analysis.txt [wp8_evidence/item3_ladder_dev.jsonl]
Runs the ladder with permissive thresholds on every development fit, stores every rung result (order, modal frequency, smallest gap, weights), then applies the declared rule
(tests/PREDECLARED_CRITERIA_WP8.md 4.2) to fix g_min and reports the accuracy of the ladder, of each rung alone, and of the variants that justify each design choice."""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identity_lib import load_records, overlap_map, stream_for
from backbone.hssm.ordering import LadderConfig, order_regimes, RUNG_ROLES
from data.synthetic_identity import REGISTRY

fits_path, out_path = sys.argv[1:3]
store = sys.argv[3] if len(sys.argv) > 3 else None
L = []
def P(*a):
    s = " ".join(str(x) for x in a); L.append(s); print(s)
def pct(a, b): return f"{a}/{b} = {100.0 * a / b:.1f}%" if b else "0/0"

recs = [r for r in load_records(fits_path) if r["ok"] and r["job"]["engine"] == "baseline"]
P(f"development fits: {len(recs)} baseline fits of {len({r['job']['user'] for r in recs})} users (seeds 8201..8212); accepted {sum(r['accepted'] for r in recs)}")
permissive = LadderConfig(min_gap=0.0, min_modal_freq=0.51)
rows = []
for r in recs:
    j = r["job"]; w = stream_for(j).window(j["start"], j["end"])
    res = order_regimes(w["raw"], np.array(r["labels_fit"]), r["K"], REGISTRY, permissive)
    mp, recov = overlap_map(r["labels_fit"], r["labels_true"], r["K"], 4)
    row = dict(user=j["user"], window=[j["start"], j["end"]], recoverable=recov, mp=mp, sep=stream_for(j).truth["user"]["sep"], rungs={})
    for k, rr in res.rungs.items():
        if rr.available:
            correct = bool(recov and [mp[x] for x in rr.order] == [0, 1, 2])
            row["rungs"][k] = dict(order=list(rr.order), modal_order=list(rr.modal_order) if rr.modal_order else None, modal_freq=rr.modal_freq, min_gap=rr.min_gap, correct=correct, weights=rr.weights)
    rows.append(row)
if store:
    with open(store, "w") as fh:
        for row in rows: fh.write(json.dumps(row) + "\n")
n_rec = sum(r["recoverable"] for r in rows)
P(f"fits whose regimes are one-to-one recoverable: {pct(n_rec, len(rows))}")
P("\n== each rung alone: point-estimate ordering equals the planted order (all fits, recoverable required)")
for k in (1, 2, 3):
    x = [r["rungs"][k] for r in rows if k in r["rungs"]]
    P(f"  rung {k} (roles {RUNG_ROLES[k]}): available {len(x)}/{len(rows)}; ordering correct {pct(sum(v['correct'] for v in x), len(x))}")

def is_reliable(v, freq, gap):
    return v["modal_order"] == v["order"] and v["modal_freq"] >= freq and v["min_gap"] >= gap

# the declared rule for g_min: pool (fit, rung) results that pass the modal-frequency condition, find the smallest g above which every one is correct, round up to 0.1
pool = [v for r in rows for v in r["rungs"].values() if v["modal_order"] == v["order"] and v["modal_freq"] >= 0.90]
P(f"\n== g_min rule: pooled rung results with modal frequency >= 0.90: {len(pool)}; correct {pct(sum(v['correct'] for v in pool), len(pool))}")
gaps_wrong = [v["min_gap"] for v in pool if not v["correct"]]
if not gaps_wrong:
    g_min_rule = 0.0
    P("  no pooled result is wrong: any g_min >= 0 satisfies the rule; the rule gives 0.0, the declared fallback (0.5) is kept because zero wrong results in this sample does not show that small gaps are safe")
    g_min = 0.5
else:
    g_star = max(gaps_wrong)
    g_min_rule = float(np.ceil((g_star + 1e-9) * 10) / 10)
    g_min = g_min_rule if g_min_rule <= 1.0 else 0.5
    P(f"  largest min-gap among wrong pooled results {g_star:.3f}; rule gives g_min = {g_min_rule:.1f}; used {g_min:.1f}{' (rule value above 1.0, fallback 0.5 as declared)' if g_min_rule > 1.0 else ''}")
rec_pool = [v for r in rows if r["recoverable"] for v in r["rungs"].values() if v["modal_order"] == v["order"] and v["modal_freq"] >= 0.90]
w_rec = [v["min_gap"] for v in rec_pool if not v["correct"]]
P(f"  (reported, not used for the rule) among fits whose regimes are recoverable: pooled {len(rec_pool)}, correct {pct(sum(v['correct'] for v in rec_pool), len(rec_pool))}; largest min-gap among the wrong ones: {max(w_rec):.3f}" if w_rec else f"  (reported, not used for the rule) among recoverable fits: pooled {len(rec_pool)}, all correct")
P(f"  distribution of min_gap among correct pooled results: median {np.median([v['min_gap'] for v in pool if v['correct']]):.2f}, min {min([v['min_gap'] for v in pool if v['correct']]):.2f}")

def ladder(row, freq, gap, agreement=True, rungs=(1, 2, 3)):
    rel = [k for k in rungs if k in row["rungs"] and is_reliable(row["rungs"][k], freq, gap)]
    if not rel: return 0, False, None
    top = rel[0]
    if agreement and any(row["rungs"][k]["order"] != row["rungs"][top]["order"] for k in rel[1:]): return 0, False, "conflict"
    return top, True, row["rungs"][top]["correct"]

def report(label, **kw):
    out = [ladder(r, 0.90, g_min, **kw) for r in rows]
    ver = [o for o in out if o[1]]
    c = Counter(o[0] for o in ver)
    P(f"  {label}: verified {pct(len(ver), len(rows))}; correct among verified {pct(sum(bool(o[2]) for o in ver), len(ver))}; per rung {dict(sorted(c.items()))}; conflicts {sum(o[2] == 'conflict' for o in out)}")
P(f"\n== ladder with modal frequency >= 0.90 and g_min = {g_min:.1f} (development, {len(rows)} fits)")
report("full ladder with agreement check")
report("no agreement check", agreement=False)
report("rung 3 only (movement)", rungs=(3,))
report("rungs 1 and 3 only", rungs=(1, 3))
P("  sensitivity to the modal-frequency threshold:")
for f in (0.80, 0.90, 0.95):
    out = [ladder(r, f, g_min) for r in rows]; ver = [o for o in out if o[1]]
    P(f"    modal frequency >= {f:.2f}: verified {pct(len(ver), len(rows))}; correct among verified {pct(sum(bool(o[2]) for o in ver), len(ver))}")
P("  sensitivity to g_min:")
for g in (0.0, 0.25, 0.5, 0.75, 1.0):
    out = [ladder(r, 0.90, g) for r in rows]; ver = [o for o in out if o[1]]
    P(f"    g_min {g:.2f}: verified {pct(len(ver), len(rows))}; correct among verified {pct(sum(bool(o[2]) for o in ver), len(ver))}")
P("\n== by planted separation of the user (sep): verified share and accuracy")
seps = sorted({r["user"]: r["sep"] for r in rows}.items(), key=lambda kv: kv[1])
for lo, hi in ((0.0, 0.8), (0.8, 1.3), (1.3, 9.0)):
    sub = [r for r in rows if lo <= r["sep"] < hi]
    out = [ladder(r, 0.90, g_min) for r in sub]; ver = [o for o in out if o[1]]
    P(f"  sep in [{lo:.1f}, {hi:.1f}): fits {len(sub)}; verified {pct(len(ver), len(sub))}; correct among verified {pct(sum(bool(o[2]) for o in ver), len(ver))}")
Path(out_path).write_text("\n".join(L) + "\n")
