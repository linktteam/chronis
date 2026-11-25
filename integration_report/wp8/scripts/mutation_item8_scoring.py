"""Mutation check for tests/test_wp8_item8_scoring.py (the scoring library of the Item 6 and Item 8 studies). usage: PYTHONPATH=. python3 integration_report/wp8/scripts/mutation_item8_scoring.py [out.txt]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mutation_runner import run

LB = "integration_report/wp8/scripts/lineage_study_lib.py"
MUTANTS = [
    ("W01 wilson centre without the z^2/2n term", LB, "c = p + z * z / (2 * n)", "c = p"),
    ("W02 wilson half width without the correction", LB, "h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))", "h = z * math.sqrt(p * (1 - p) / n)"),
    ("W03 wilson with n = 0 returns zeros", LB, "return (float(\"nan\"), float(\"nan\"))", "return (0.0, 0.0)"),
    ("W04 wilson z of one", LB, "def wilson(k: int, n: int, z: float = 1.959964)", "def wilson(k: int, n: int, z: float = 1.0)"),
    ("W05b wilson lower bound not clamped at zero", LB, "return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))", "return ((c - h) / d, min(1.0, (c + h) / d))"),
    ("W05c wilson upper bound not clamped at one", LB, "return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))", "return (max(0.0, (c - h) / d), (c + h) / d)"),
    ("W05 rate prints no interval", LB, 'return f"{k}/{n} = {100.0 * k / n:.1f}% (95% CI {100 * lo:.1f}-{100 * hi:.1f}%)"', 'return f"{k}/{n} = {100.0 * k / n:.1f}%"'),
    ("W06 day agreement counts disagreements", LB, "return float((a == b).mean())", "return float((a != b).mean())"),
    ("W07 day agreement accepts different lengths", LB, "if a.shape != b.shape or a.size == 0:", "if False:"),
    ("W08 correctness ignores recoverability", LB, "return bool(both and good), both", "return bool(good), both"),
    ("W09 correctness uses the anchor's raw order, not its stable ids", LB, "stable_to_anchor_raw = {s: i for i, s in enumerate(anchor_r2s)}", "stable_to_anchor_raw = {s: s for s in anchor_r2s}"),
    ("W10 correctness compares raw index with planted regime", LB, "good = all(anchor_info.mp[stable_to_anchor_raw[new_r2s[j]]] == info.mp[j] for j in range(info.K))", "good = all(stable_to_anchor_raw[new_r2s[j]] == info.mp[j] for j in range(info.K))"),
    ("W11 a changed K is scored", LB, "    if anchor_info.K != info.K:\n        return None, both", "    if False:\n        return None, both"),
    ("W12 K_CHANGED steps are scored", LB, "        if lr.continuity_state != K_CHANGED:", "        if True:"),
    ("W31 the anchor K is the new fit's K", LB, 'row["K_anchor"] = ai.K', 'row["K_anchor"] = info.K'),
    ("W13 served is any state but NEW_LINEAGE", LB, "served=lr.continuity_state == ALIGNED)", "served=lr.continuity_state != NEW_LINEAGE)"),
    ("W14 scoring uses the first fit as the anchor", LB, "ai = infos[state_before.anchor.fit_index]", "ai = infos[0]"),
    ("W15 scoring uses the new names for the anchor", LB, "state_before.anchor.raw_to_stable, tuple(lr.raw_to_stable))", "tuple(lr.raw_to_stable), tuple(lr.raw_to_stable))"),
    ("W16 replay ignores the publish rule", LB, "if not usable(rec) or (accepted_only and not rec.get(\"accepted\")):", "if not usable(rec):"),
    ("W17 replay feeds failed fits", LB, "if not usable(rec) or (accepted_only", "if (accepted_only"),
    ("W18 usable ignores the fit state", LB, 'return bool(rec.get("ok")) and rec.get("fit_state") not in NOT_USABLE_STATES', 'return bool(rec.get("ok"))'),
    ("W19 usable ignores ok", LB, 'return bool(rec.get("ok")) and rec.get("fit_state") not in NOT_USABLE_STATES', 'return rec.get("fit_state") not in NOT_USABLE_STATES'),
    ("W20 replay does not apply the patch", LB, "    with ctx:\n        for rec in recs:", "    with contextlib.nullcontext():\n        for rec in recs:"),
    ("W21 tally counts wrong among all alignments", LB, "wrong_unflagged = [r for r in served if not r[\"correct\"]]", "wrong_unflagged = [r for r in al if not r[\"correct\"]]"),
    ("W22 tally swaps would-be-wrong and would-be-right", LB, 'flagged_would_be_wrong=sum(1 for r in al if not r["served"] and not r["correct"]), flagged_would_be_right=sum(1 for r in al if not r["served"] and r["correct"])',
     'flagged_would_be_wrong=sum(1 for r in al if not r["served"] and r["correct"]), flagged_would_be_right=sum(1 for r in al if not r["served"] and not r["correct"])'),
    ("W23 conditional set ignores both-recoverable", LB, 'cond = [r for r in al if r["both_recoverable"]]', "cond = al"),
    ("W24 the summary uses z means as raw means", LB, 'np.asarray(rec["raw_means"], dtype=float), np.asarray(rec["scaler_mean"], dtype=float)', 'np.asarray(rec["raw_means"], dtype=float) * 0.5, np.asarray(rec["scaler_mean"], dtype=float)'),
    ("W25 the summary uses the true labels", LB, 'np.asarray(rec["labels_fit"], dtype=int), wr, exp_durations', 'np.asarray(rec["labels_true"], dtype=int) % rec["K"], wr, exp_durations'),
    ("W26 stable ids taken from the end of the window", LB, "return [np.asarray(lr.stable_regime_id)[:n_days] for lr in lrs]", "return [np.asarray(lr.stable_regime_id)[-n_days:] for lr in lrs]"),
    ("W27 the schedule does not add the last window", LB, "return sorted(set(days) | {last})", "return sorted(set(days))"),
    ("W28 the schedule ignores the production policy", LB, "days = refit_schedule(first, last, RefitPolicy(interval))", "days = list(range(first, last + 1, max(interval, 1)))"),
    ("W29 a missing fit is skipped silently", LB, "    if missing:\n        raise KeyError", "    if False:\n        raise KeyError"),
    ("W30 fits are returned in file order", LB, "return [by[d] for d in days]", "return list(by.values())"),
    ("W32 a candidate uses the stable id as the planted regime", LB, "return -1 if raw is None else int(anchor_info.mp[raw])", "return -1 if raw is None else int(stable_id)"),
    ("W33 merge candidates are not sorted", LB, 'planted=sorted(planted(i) for i in c["old_stable_ids"])', 'planted=[planted(i) for i in c["old_stable_ids"]]'),
    ("W34 candidates attached to aligned rows too", LB, '        else:\n            row["candidates"] =', '        if True:\n            row["candidates"] ='),
    ("W35 an unknown stable id is mapped to regime 0", LB, "raw = list(anchor_r2s).index(stable_id) if stable_id in anchor_r2s else None", "raw = list(anchor_r2s).index(stable_id) if stable_id in anchor_r2s else 0"),
    ("W36 verdict: below the threshold is met", LB, "        if p < thr:\n            return \"NOT MET\"", "        if p > thr:\n            return \"NOT MET\""),
    ("W37 verdict: an interval crossing the threshold is a plain MET", LB, 'return "MET" if lo >= thr else "MET on the point estimate, NOT DEMONSTRATED (the 95% interval reaches below the threshold)"', 'return "MET"'),
    ("W38 verdict: the upper interval is not used", LB, 'return "MET" if hi <= thr else "MET on the point estimate, NOT DEMONSTRATED (the 95% interval reaches above the threshold)"', 'return "MET"'),
    ("W39 verdict: n = 0 counts as met", LB, 'return "UNKNOWN (nothing to count)"', 'return "MET"'),
    ("W40 verdict: an unknown comparison is accepted", LB, "    if kind not in (\">=\", \"<=\"):\n        raise ValueError(\"kind must be '>=' or '<='\")\n", ""),
    ("W41 verdict: the at-most rule uses the lower bound", LB, 'return "MET" if hi <= thr else "MET on', 'return "MET" if lo <= thr else "MET on'),
]

if __name__ == "__main__":
    sys.exit(run(MUTANTS, ["tests/test_wp8_item8_scoring.py"], sys.argv[1] if len(sys.argv) > 1 else None, workers=1))
