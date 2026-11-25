"""WP8 item 2 diagnosis: analysis of wp8_evidence/item2_fits.jsonl (written by item2_run.py). Writes the text report and every alignment with all its margins.
usage: PYTHONPATH=. python3 integration_report/wp8/scripts/item2_analyze.py wp8_evidence/item2_fits.jsonl wp8_evidence/item2_analysis.txt wp8_evidence/item2_alignments.jsonl"""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identity_lib import (align_records, alignment_correct, assignment_costs, best_relabel_agreement, l2_order_matches, load_records, overlap_map, regime_vectors, stream_for)

fits_path, out_path, al_path = sys.argv[1:4]
recs = [r for r in load_records(fits_path) if r["ok"]]
bad = [r for r in load_records(fits_path) if not r["ok"]]
L = []
def P(*a):
    s = " ".join(str(x) for x in a); L.append(s); print(s)

def pct(a, b): return f"{a}/{b} = {100.0 * a / b:.1f}%" if b else "0/0"

by = defaultdict(dict)                     # (group, engine, user, drift, kchange) -> {(start,end,fitseed): rec}
for r in recs:
    j = r["job"]; by[(j["group"], j["engine"], j["user"], j["drift"], j["kchange"], j["K"])][(j["start"], j["end"], j["fitseed"])] = r
P(f"fits read: {len(recs)} ok, {len(bad)} failed ({sorted({b.get('error','?')[:60] for b in bad})[:3]})")
for g in "ABDFC":
    rr = [r for r in recs if r["job"]["group"] == g]
    if rr:
        P(f"group {g}: {len(rr)} fits, accepted {sum(r['accepted'] for r in rr)}, CONVERGED {sum(r['fit_state']=='CONVERGED' for r in rr)}, mean seconds {np.mean([r['seconds'] for r in rr]):.1f}")

# ---------------------------------------------------------------- 1/2. raw instability under window growth, same data different seed, L2 convention
def raw_agree(a, b): n = min(len(a), len(b)); return float(np.mean(np.array(a[:n]) == np.array(b[:n])))
for eng, grp, users in (("baseline", "A", range(8101, 8113)), ("dual", "D", range(8101, 8107))):
    P(f"\n== {eng}: window growth (fit on days 0..w versus the first fit on days 0..90, compared on days 0..89), raw ids and after the best relabelling")
    tab = defaultdict(list)
    for u in users:
        d = by.get((grp, eng, u, 0.0, None, 3), {})
        first = d.get((0, 90, u))
        if not first: continue
        for (s, e, fs), r in sorted(d.items()):
            if fs == u and e > 90:
                tab[e].append((raw_agree(first["labels_fit"], r["labels_fit"][:90]), best_relabel_agreement(first["labels_fit"], r["labels_fit"][:90], 3, 3)))
    for e in sorted(tab):
        raw = np.array([x[0] for x in tab[e]]); rel = np.array([x[1] for x in tab[e]])
        P(f"  w={e}: users {len(raw)}  raw agreement mean {raw.mean():.3f} (min {raw.min():.3f}); users with >5% of days relabelled {pct(int((raw < 0.95).sum()), len(raw))}; after best relabelling mean {rel.mean():.3f}")
    P(f"== {eng}: same data, different fit seed")
    for w in (90, 150):
        pairs = []
        for u in users:
            d = by.get((grp if eng == 'dual' else 'A', eng, u, 0.0, None, 3), {}); d2 = by.get(("D" if eng == "dual" else "B", eng, u, 0.0, None, 3), {})
            a, b = d.get((0, w, u)), d2.get((0, w, u + 20000))
            if a and b: pairs.append((raw_agree(a["labels_fit"], b["labels_fit"]), best_relabel_agreement(a["labels_fit"], b["labels_fit"], 3, 3)))
        if pairs:
            raw = np.array([x[0] for x in pairs]); rel = np.array([x[1] for x in pairs])
            P(f"  w={w}: pairs {len(raw)}  raw agreement mean {raw.mean():.3f} (min {raw.min():.3f}), identical raw labels in {pct(int((raw > 0.999).sum()), len(raw))}; after best relabelling mean {rel.mean():.3f}")
for eng, grp in (("baseline", ("A", "B")), ("dual", ("D",))):
    rr = [r for r in recs if r["job"]["group"] in grp and r["job"]["engine"] == eng]
    n_all = n_rec = n_match_rec = n_match_all = 0
    for r in rr:
        mp, rec_ok = overlap_map(r["labels_fit"], r["labels_true"], r["K"], 4)
        n_all += 1; n_match_all += l2_order_matches(mp)
        if rec_ok: n_rec += 1; n_match_rec += l2_order_matches(mp)
    P(f"== {eng}: raw regime order equals the planted activity order (chance 1/6 = 16.7%): all fits {pct(n_match_all, n_all)}; fits whose regimes are one-to-one recoverable {pct(n_match_rec, n_rec)} (recoverable fits: {pct(n_rec, n_all)})")
    if eng == "baseline":
        # L2 norm order versus activity order is what canonicalize_labels claims
        P("   (the baseline raw order IS the L2 order: canonicalize_labels runs inside select_k_by_bic)")

# ---------------------------------------------------------------- baseline: labels consistent with mu
ok_n = tot = 0
for r in [r for r in recs if r["job"]["engine"] == "baseline" and r["job"]["group"] == "A"][:60]:
    j = r["job"]; w = stream_for(j).window(j["start"], j["end"])["matrix"]
    lab = np.array(r["labels_fit"]); mu = np.array(r["z_means"]); good = True
    for k in range(r["K"]):
        sel = (lab == k) & ~np.isnan(w).any(axis=1)
        if sel.sum() == 0: good = False; continue
        mk = w[sel].mean(axis=0)
        good &= int(np.argmin(((mu - mk) ** 2).sum(axis=1))) == k
    ok_n += good; tot += 1
P(f"== baseline output/parameter consistency: the data mean of the days decoded as regime k is closest to model.mu[k] for every k in {pct(ok_n, tot)} of the first {tot} group-A fits")

# ---------------------------------------------------------------- 3. alignments and margins
alignments = []
def add_pairs(group, engine, drift, kchange, step_pairs, kind):
    for u in range(8101, 8113):
        d = by.get((group, engine, u, drift, kchange, 3), {})
        for (a, b) in step_pairs(d, u):
            for mode in ("z", "raw", "raw_dur"):
                al = align_records(a, b, mode)
                good, recov = alignment_correct(a, b, al["perm"])
                alignments.append(dict(user=u, engine=engine, kind=kind, drift=drift, mode=mode, start_prev=a["job"]["start"], end_prev=a["job"]["end"], start_new=b["job"]["start"], end_new=b["job"]["end"],
                                       correct=good, recoverable=recov, accepted_prev=a["accepted"], accepted_new=b["accepted"], **{k: al[k] for k in ("perm", "C1", "C2", "margin_diff", "margin_ratio", "matched", "nearest_other")}))
def G_consecutive(d, u):
    ws = sorted(k for k in d if k[2] == u and k[0] == 0)
    return [(d[x], d[y]) for x, y in zip(ws, ws[1:])]
def G_first(d, u):
    ws = sorted(k for k in d if k[2] == u and k[0] == 0)
    return [(d[ws[0]], d[y]) for y in ws[1:]]
def R_consecutive(d, u):
    ws = sorted(k for k in d if k[2] == u)
    return [(d[x], d[y]) for x, y in zip(ws, ws[1:])]
add_pairs("A", "baseline", 0.0, None, G_consecutive, "G_consecutive")
add_pairs("A", "baseline", 0.0, None, G_first, "G_first_to_w")
add_pairs("C", "baseline", 1.0, None, R_consecutive, "R_consecutive")
add_pairs("C", "baseline", 2.5, None, R_consecutive, "R_consecutive")
with open(al_path, "w") as fh:
    for a in alignments: fh.write(json.dumps(a) + "\n")
P(f"\n== alignments: {len(alignments) // 3} per distance, each stored with every margin in {al_path}")
def _extra(a):
    a = dict(a)
    a["max_matched"] = max(a["matched"])
    a["max_regime_ratio"] = max(m / n if n > 0 else 1.0 for m, n in zip(a["matched"], a["nearest_other"]))
    return a
MARGINS = (("margin_diff", True), ("margin_ratio", False), ("max_matched", False), ("max_regime_ratio", False))
def summarize(sel, label):
    for mode in ("z", "raw", "raw_dur"):
        x = [_extra(a) for a in sel if a["mode"] == mode]
        if not x: continue
        correct = [a for a in x if a["correct"] and a["recoverable"]]
        wrong = [a for a in x if not (a["correct"] and a["recoverable"])]
        wrong_rec = [a for a in x if a["recoverable"] and not a["correct"]]
        line = f"  {label:34s} {mode:8s} n={len(x):3d} correct {pct(len(correct), len(x))}; wrong-and-recoverable {len(wrong_rec)}, not recoverable {len(x) - len(correct) - len(wrong_rec)}"
        P(line)
        for mname, better_high in MARGINS:
            cm = np.array([a[mname] for a in correct]) if correct else np.array([np.nan]); wm = np.array([a[mname] for a in wrong]) if wrong else np.array([np.nan])
            if better_high:
                thr = np.nanmax(wm) if wrong else -np.inf
                cov = np.mean([a[mname] > thr for a in x]) if wrong else 1.0
                P(f"      {mname:17s} correct: min {np.nanmin(cm):.3g} median {np.nanmedian(cm):.3g} | wrong: max {np.nanmax(wm) if wrong else float('nan'):.3g} | share above the largest wrong value {100*cov:.0f}%")
            else:
                thr = np.nanmin(wm) if wrong else np.inf
                cov = np.mean([a[mname] < thr for a in x]) if wrong else 1.0
                P(f"      {mname:17s} correct: max {np.nanmax(cm):.3g} median {np.nanmedian(cm):.3g} | wrong: min {np.nanmin(wm) if wrong else float('nan'):.3g} | share below the smallest wrong value {100*cov:.0f}%")
for kind in ("G_consecutive", "G_first_to_w"):
    summarize([a for a in alignments if a["kind"] == kind], f"protocol G {kind[2:]}")
for dr in (1.0, 2.5):
    summarize([a for a in alignments if a["kind"] == "R_consecutive" and a["drift"] == dr], f"protocol R drift {dr} sd")
summarize(alignments, "ALL")
P("  (wrong = not correct, or a fit that is not one-to-one recoverable; 'above the largest wrong margin' = the share that a threshold just above every wrong alignment would still serve)")

# ---------------------------------------------------------------- 4. cross engine
P("\n== cross-engine alignment (baseline fit versus dual fit on the same window, users 8101..8106)")
cross = []
for u in range(8101, 8107):
    for w in (90, 120, 150):
        b = by.get(("A", "baseline", u, 0.0, None, 3), {}).get((0, w, u)); d = by.get(("D", "dual", u, 0.0, None, 3), {}).get((0, w, u))
        if not (b and d): continue
        for mode in ("raw", "raw_dur"):
            if mode == "raw_dur": pass
            al = align_records(b, d, mode); good, recov = alignment_correct(b, d, al["perm"])
            cross.append(dict(user=u, w=w, mode=mode, correct=good, recoverable=recov, dual_accepted=d["accepted"], margin_diff=al["margin_diff"], margin_ratio=al["margin_ratio"]))
for mode in ("raw", "raw_dur"):
    x = [c for c in cross if c["mode"] == mode]
    if x:
        P(f"  {mode}: pairs {len(x)}; both fits recoverable {pct(sum(c['recoverable'] for c in x), len(x))}; correct among recoverable {pct(sum(bool(c['correct']) for c in x if c['recoverable']), sum(c['recoverable'] for c in x))}; dual fit accepted {pct(sum(c['dual_accepted'] for c in x), len(x))}")
        P("     margins (diff / ratio, correct?, recoverable): " + "; ".join(f"{c['user']}/{c['w']}: {c['margin_diff']:.2f}/{c['margin_ratio']:.2f} {'ok' if c['correct'] else 'WRONG'}{'' if c['recoverable'] else ' (unrecoverable)'}" for c in x))
d_rec = [r for r in recs if r["job"]["group"] == "D"]
if d_rec:
    P(f"  dual fits: recoverable (one-to-one with planted regimes) {pct(sum(overlap_map(r['labels_fit'], r['labels_true'], r['K'], 4)[1] for r in d_rec), len(d_rec))}; mean best-relabel agreement with planted labels {np.mean([best_relabel_agreement(r['labels_fit'], r['labels_true'], r['K'], 4) for r in d_rec]):.3f}")
b_rec = [r for r in recs if r["job"]["group"] == "A"]
P(f"  baseline fits (group A): recoverable {pct(sum(overlap_map(r['labels_fit'], r['labels_true'], r['K'], 4)[1] for r in b_rec), len(b_rec))}; mean best-relabel agreement with planted labels {np.mean([best_relabel_agreement(r['labels_fit'], r['labels_true'], r['K'], 4) for r in b_rec]):.3f}")

# ---------------------------------------------------------------- 5. K change
P("\n== K change (baseline). Fit A: days 0..90 with K = 3. Fit B: days 30..150.")
rows = []
for i, u in enumerate(range(8101, 8113)):
    kind, k_after = ("split", 4) if i < 6 else ("merge", 2)
    d = by.get(("F", "baseline", u, 0.0, kind, 3), {}); dk = by.get(("F", "baseline", u, 0.0, kind, k_after), {})
    A = d.get((0, 90, u)); B3 = d.get((30, 150, u)); Bk = dk.get((30, 150, u))
    if not (A and B3 and Bk): continue
    st = stream_for(dict(user=u, drift=0.0, kchange=kind)); user = st.truth["user"]
    mpA, recA = overlap_map(A["labels_fit"], A["labels_true"], 3, 4)
    inv = {p_: j for j, p_ in enumerate(mpA)}
    ref = regime_vectors(A, "raw", np.array(A["scaler_std"])); new = regime_vectors(Bk, "raw", np.array(A["scaler_std"]))
    cost = ((ref[:, None, :] - new[None, :, :]) ** 2).sum(axis=2)
    if kind == "split":
        nearest = cost.argmin(axis=0)                                  # old regime nearest to each new regime
        shared = [int(o) for o in set(nearest.tolist()) if (nearest == o).sum() >= 2]
        parent_fit = inv.get(user["split_parent"], -1)
        rows.append(dict(user=u, kind=kind, candidate=shared, truth=parent_fit, hit=(len(shared) == 1 and shared[0] == parent_fit), recA=recA, same_K_correct=None))
    else:
        nearest = cost.argmin(axis=1)                                  # new regime nearest to each old regime
        shared = [tuple(sorted(i_ for i_ in range(3) if nearest[i_] == n)) for n in set(nearest.tolist()) if (nearest == n).sum() >= 2]
        pair_fit = tuple(sorted(inv.get(x, -1) for x in user["merge_pair"]))
        rows.append(dict(user=u, kind=kind, candidate=shared, truth=pair_fit, hit=(len(shared) == 1 and shared[0] == pair_fit), recA=recA))
    al3 = align_records(A, B3, "raw"); g3, rec3 = alignment_correct(A, B3, al3["perm"])
    rows[-1].update(same_K_margin_ratio=al3["margin_ratio"], same_K_correct=g3, same_K_recoverable=rec3, B_selected_k_fit=Bk["K"], accepted=Bk["accepted"])
for kind in ("split", "merge"):
    x = [r for r in rows if r["kind"] == kind]
    if x:
        P(f"  {kind}: users {len(x)}; the matching of the K-changed fit names exactly the true {'parent regime' if kind == 'split' else 'merged pair'} in {pct(sum(r['hit'] for r in x), len(x))}; fit A recoverable in {pct(sum(r['recA'] for r in x), len(x))}")
        P("     per user (candidate vs truth): " + "; ".join(f"{r['user']}: {r['candidate']} vs {r['truth']}{' ok' if r['hit'] else ''}" for r in x))
        P(f"     if K had NOT been changed (K = 3 forced on days 30..150), a plain alignment to fit A (WP5 rule: ratio <= 0.5) would be served in {sum(r['same_K_margin_ratio'] <= 0.5 for r in x)} of {len(x)} cases and is correct in {pct(sum(bool(r['same_K_correct']) for r in x), len(x))} (margin ratio median {np.median([r['same_K_margin_ratio'] for r in x]):.2f}): the matching cannot tell that the world changed")

# ---------------------------------------------------------------- 6. per-feature activity signal
P("\n== per-feature activity signal (12 users; fits on days 0..90; ordering the fit's regimes by ONE feature, signed by the registry direction, equals the planted order?)")
from data.synthetic_identity import REGISTRY
names = [f.name for f in REGISTRY]; dirs = [f.direction for f in REGISTRY]
acc = defaultdict(list); snr_true = defaultdict(list); cov = defaultdict(list); noise = defaultdict(list)
for u in range(8101, 8113):
    r = by.get(("A", "baseline", u, 0.0, None, 3), {}).get((0, 90, u))
    if not r: continue
    mp, recov = overlap_map(r["labels_fit"], r["labels_true"], 3, 4)
    raw = np.array(r["raw_means"])
    for f, nm in enumerate(names):
        if dirs[f] == 0: continue
        order = np.argsort(dirs[f] * raw[:, f])
        acc[nm].append(recov and [mp[j] for j in order] == [0, 1, 2])
    st = stream_for(dict(user=u, drift=0.0, kchange=None)); lab = st.labels[:90]; X = st.X_full[:90]
    for f, nm in enumerate(names):
        means = np.array([X[lab == k, f].mean() for k in range(3) if (lab == k).any()]); w = np.array([(lab == k).sum() for k in range(3) if (lab == k).any()])
        within = np.sqrt(sum(((X[lab == k, f] - X[lab == k, f].mean()) ** 2).sum() for k in range(3) if (lab == k).any()) / len(lab))
        snr_true[nm].append(float(np.average((means - np.average(means, weights=w)) ** 2, weights=w) / within ** 2)); cov[nm].append(float(r["coverage"][f])); noise[nm].append(float(within / X[:, f].std()))
P("  feature      direction  ordering correct (recoverable fits)   planted SNR (between/within) median [min..max]   coverage median [min..max]   day-to-day noise / total sd median [min..max]")
for f, nm in enumerate(names):
    if dirs[f] == 0:
        P(f"  {nm:11s}  {dirs[f]:+d}        (no activity direction)"); continue
    a = acc[nm]
    P(f"  {nm:11s}  {dirs[f]:+d}        {pct(sum(a), len(a)):>14s}                      {np.median(snr_true[nm]):6.2f} [{min(snr_true[nm]):.2f}..{max(snr_true[nm]):.2f}]                {np.median(cov[nm]):.2f} [{min(cov[nm]):.2f}..{max(cov[nm]):.2f}]            {np.median(noise[nm]):.2f} [{min(noise[nm]):.2f}..{max(noise[nm]):.2f}]")

# ---------------------------------------------------------------- 7. fit-level identifiability signal (exploratory, for the design of the lineage gate)
P("\n== fit-level signal: can a fit tell, without planted truth, that its regimes do not correspond one-to-one to planted regimes? (baseline fits of groups A and C)")
def min_separation(r):
    j = r["job"]; w = stream_for(j).window(j["start"], j["end"])["matrix"]
    lab = np.array(r["labels_fit"]); mu = np.array(r["z_means"]); K = r["K"]
    ok = ~np.isnan(w).any(axis=1)
    resid = np.vstack([w[ok & (lab == k)] - mu[k] for k in range(K) if (ok & (lab == k)).sum() > 1])
    sd = np.sqrt((resid ** 2).mean(axis=0)); sd = np.where(sd > 0, sd, 1.0)
    return min(float(np.linalg.norm((mu[i] - mu[k]) / sd)) for i in range(K) for k in range(i + 1, K))
def min_days(r):
    return float(min(np.bincount(np.array(r["labels_fit"]), minlength=r["K"])))
from itertools import product
for grp_label, grp in (("A (growing window, drift 0)", "A"), ("C (rolling 60-day, drift 1 and 2.5)", "C")):
    rr = [r for r in recs if r["job"]["group"] == grp and r["job"]["engine"] == "baseline"]
    rec_flags = [overlap_map(r["labels_fit"], r["labels_true"], r["K"], 4)[1] for r in rr]
    for name, fn in (("min pairwise regime separation (within-regime sd units)", min_separation), ("fewest decoded days in a regime", min_days)):
        v = np.array([fn(r) for r in rr]); good = v[np.array(rec_flags)]; bad = v[~np.array(rec_flags)]
        if len(bad) and len(good):
            auc = np.mean([(g > b) + 0.5 * (g == b) for g in good for b in bad])
            P(f"  {grp_label}: {name}: recoverable fits median {np.median(good):.2f} [{good.min():.2f}..{good.max():.2f}] (n={len(good)}), unrecoverable median {np.median(bad):.2f} [{bad.min():.2f}..{bad.max():.2f}] (n={len(bad)}), AUC (recoverable higher) {auc:.2f}")
        else:
            P(f"  {grp_label}: {name}: recoverable n={len(good)}, unrecoverable n={len(bad)} (no comparison)")

Path(out_path).write_text("\n".join(L) + "\n")
