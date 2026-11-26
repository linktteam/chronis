"""Mutation check for A5: the candidate monitors (`a5_monitors.py`), the frozen development rule and tallies (`a5_analyze.py`) and the additive forward pass of `backbone/hssm/monitoring.py`
(`baseline_forward_scores`) against tests/test_wp10a_a5_monitors.py. usage: PYTHONPATH=. python3 integration_report/wp10a/scripts/mutation_wp10a_a5.py [out.txt]"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp8" / "scripts"))
import mutation_runner as R

_base = shutil.ignore_patterns(".git", "__pycache__", "wp7r_evidence", "wp8_evidence", "wp9_evidence", "wp10a_evidence", ".pytest_cache")
R.IGNORE = lambda d, names: set(_base(d, names))

MN = "integration_report/wp10a/scripts/a5_monitors.py"
AN = "integration_report/wp10a/scripts/a5_analyze.py"
PM = "backbone/hssm/monitoring.py"
MUT = [
    ("N01 the block length is 5", MN, "BLOCK, DRAWS = 7, 2000", "BLOCK, DRAWS = 5, 2000"),
    ("N02 1000 draws", MN, "BLOCK, DRAWS = 7, 2000", "BLOCK, DRAWS = 7, 1000"),
    ("N03 drop_z 3.0", MN, "WINDOW, REFERENCE, COOLDOWN, MIN_DROP_SD, DROP_Z = 14, 28, 14, 0.75, 3.5", "WINDOW, REFERENCE, COOLDOWN, MIN_DROP_SD, DROP_Z = 14, 28, 14, 0.75, 3.0"),
    ("N04 a regime needs 3 days", MN, "MIN_REGIME_DAYS = 5", "MIN_REGIME_DAYS = 3"),
    ("N05 the bootstrap seed is 1", MN, "rng = np.random.default_rng(0)\n    n_starts", "rng = np.random.default_rng(1)\n    n_starts"),
    ("N06 the reference has one block too many", MN, "nb_recent, nb_ref = window // block, len(ref) // block", "nb_recent, nb_ref = window // block, len(ref) // block + 1"),
    ("N07 the block mean divides by block + 1", MN, "/ block\n    nb_recent", "/ (block + 1)\n    nb_recent"),
    ("N08 the smaller standard error wins", MN, "se = max(se_ar, _lag1_free_se_block(ref, window))", "se = min(se_ar, _lag1_free_se_block(ref, window))"),
    ("N09 the autocorrelation is capped at 0.9", MN, "        r = min(max(_lag1(ref), 0.0), 0.8)\n        g = (1.0 + r) / (1.0 - r)\n        se_ar", "        r = min(max(_lag1(ref), 0.0), 0.9)\n        g = (1.0 + r) / (1.0 - r)\n        se_ar"),
    ("N10 the event needs the drop or the z", MN, "if drop >= min_drop_sd * s0 and z >= drop_z:", "if drop >= min_drop_sd * s0 or z >= drop_z:"),
    ("N11 the cooldown is exclusive", MN, "if last is not None and day - last < cooldown_days:", "if last is not None and day - last <= cooldown_days:"),
    ("N12 41 scores are enough", MN, "if len(v) < need:\n        return dict(out, status=\"not_enough_scores\")", "if len(v) < need - 1:\n        return dict(out, status=\"not_enough_scores\")"),
    ("N13 the day after a transition is kept", MN, "    out[1:] |= change[:-1]\n", "    pass\n"),
    ("N14 a transition is no change of regime", MN, "change[1:] = k[1:] != k[:-1]", "change[1:] = k[1:] == k[:-1]"),
    ("N15 the row before is not dropped again", MN, "return out[1:] if prev is not None else out", "return out"),
    ("N16 the decoded regime is the least probable", MN, "return np.argmax(np.asarray(post, dtype=float), axis=1)", "return np.argmin(np.asarray(post, dtype=float), axis=1)"),
    ("N17 a regime needs more than 5 days", MN, "if sel.sum() >= min_days:", "if sel.sum() > min_days:"),
    ("N18 the regime sd has ddof 0", MN, "sds.append(float(s[sel].std(ddof=1)))", "sds.append(float(s[sel].std(ddof=0)))"),
    ("N19 the residual multiplies by the sd", MN, "e = (s - np.asarray(means)[k]) / sd", "e = (s - np.asarray(means)[k]) * sd"),
    ("N20 a zero sd gives inf", MN, "e[~(np.isfinite(sd) & (sd > 0))] = np.nan", "e[~np.isfinite(sd)] = np.nan"),
    ("N21 MB keeps only the transition days", MN, "y[mask] = np.nan", "y[~mask] = np.nan"),
    ("N22 MC fixes its means on the held-out days", MN, "means, sds = regime_stats(train_scores, train_post)", "means, sds = regime_stats(scores, post)"),
    ("N23 MA is not dispatched", MN, 'if name == "MA":', 'if name == "MX":'),
    ("N24 MB is not given the row before", MN, "return monitor_transition_excluded(scores, post, prev)", "return monitor_transition_excluded(scores, post)"),
    ("A01 the false-alarm level is 20%", AN, "LEVEL_FA, LEVEL_DET = 0.10, 0.70", "LEVEL_FA, LEVEL_DET = 0.20, 0.70"),
    ("A02 the detection level is 60%", AN, "LEVEL_FA, LEVEL_DET = 0.10, 0.70", "LEVEL_FA, LEVEL_DET = 0.10, 0.60"),
    ("A03 the change day is 130", AN, "CHANGE_DAY, END_DAY, TRAIN_END = 135, 179, 90", "CHANGE_DAY, END_DAY, TRAIN_END = 130, 179, 90"),
    ("A04 the last day is 170", AN, "CHANGE_DAY, END_DAY, TRAIN_END = 135, 179, 90", "CHANGE_DAY, END_DAY, TRAIN_END = 135, 170, 90"),
    ("A05 training ends at day 89", AN, "CHANGE_DAY, END_DAY, TRAIN_END = 135, 179, 90", "CHANGE_DAY, END_DAY, TRAIN_END = 135, 179, 89"),
    ("A06 the order is reversed", AN, 'ORDER = ("MA", "MB", "MC")', 'ORDER = ("MC", "MB", "MA")'),
    ("A07 a false alarm of exactly 10% is not eligible", AN, "return fa <= LEVEL_FA and", "return fa < LEVEL_FA and"),
    ("A08 a detection of exactly 70% is not eligible", AN, "det_shift >= LEVEL_DET and det_sensor >= LEVEL_DET", "det_shift > LEVEL_DET and det_sensor >= LEVEL_DET"),
    ("A09 eligibility needs one detection only", AN, "det_shift >= LEVEL_DET and det_sensor >= LEVEL_DET", "det_shift >= LEVEL_DET or det_sensor >= LEVEL_DET"),
    ("A10 the false-alarm shortfall is inverted", AN, "return max(0.0, fa - LEVEL_FA) +", "return max(0.0, LEVEL_FA - fa) +"),
    ("A11 the highest false alarm is chosen", AN, 'best = min(el, key=lambda c: (summary[c]["fa"], -summary[c]["det_shift"], ORDER.index(c)))', 'best = max(el, key=lambda c: (summary[c]["fa"], -summary[c]["det_shift"], ORDER.index(c)))'),
    ("A12 ties go to the lower regime-shift detection", AN, 'best = min(el, key=lambda c: (summary[c]["fa"], -summary[c]["det_shift"], ORDER.index(c)))', 'best = min(el, key=lambda c: (summary[c]["fa"], summary[c]["det_shift"], ORDER.index(c)))'),
    ("A13 ties go to the last candidate", AN, 'best = min(el, key=lambda c: (summary[c]["fa"], -summary[c]["det_shift"], ORDER.index(c)))', 'best = min(el, key=lambda c: (summary[c]["fa"], -summary[c]["det_shift"], -ORDER.index(c)))'),
    ("A14 the largest shortfall is chosen", AN, "best = min(cands, key=lambda c: (shortfall(", "best = max(cands, key=lambda c: (shortfall("),
    ("A15 the fallback ties go to the last candidate", AN, 'summary[c]["det_sensor"]), ORDER.index(c)))', 'summary[c]["det_sensor"]), -ORDER.index(c)))'),
    ("A16 the alarmed users are those without an event", AN, 'k = sum(1 for r in users if r["event_days"])', 'k = sum(1 for r in users if not r["event_days"])'),
    ("A17 an event on day 135 is not a detection", AN, 'post = [r for r in users if any(CHANGE_DAY <= d <= END_DAY for d in r["event_days"])]', 'post = [r for r in users if any(CHANGE_DAY < d <= END_DAY for d in r["event_days"])]'),
    ("A18 an event on day 134 is not a pre-change event", AN, 'pre = sum(1 for r in users if any(d < CHANGE_DAY for d in r["event_days"]))', 'pre = sum(1 for r in users if any(d < CHANGE_DAY - 1 for d in r["event_days"]))'),
    ("A19 the latency is the shortest", AN, "latency_median=(lat[len(lat) // 2] if lat else None)", "latency_median=(lat[0] if lat else None)"),
    ("A20 missing days keep their score", AN, "    s[miss] = np.nan\n", "    pass\n"),
    ("A21 the row before is the first held-out row", AN, "post[TRAIN_END:], post[TRAIN_END - 1], s[:TRAIN_END]", "post[TRAIN_END:], post[TRAIN_END], s[:TRAIN_END]"),
    ("A22 the evaluation accepts development users", AN, "if not rehearsal and any(r[\"user\"] < 10451 for r in recs):", "if False:"),
    ("A23 an unknown monitor is accepted", AN, "if chosen not in ORDER:", "if False:"),
    ("A24 event days are relative to the held-out series", AN, 'event_days=[TRAIN_END + e["day"] for e in res["events"]]', 'event_days=[e["day"] for e in res["events"]]'),
    ("A25 the evaluation runs without the freeze file", AN, "    if not FREEZE.exists():\n        sys.exit(\"the evaluation needs", "    if False:\n        sys.exit(\"the evaluation needs"),
    ("P01 the regime posterior sums over the wrong axis", PM, "per_regime = logsumexp(alpha, axis=2)", "per_regime = logsumexp(alpha, axis=1)[:, :alpha.shape[1]]"),
    ("P02 the posterior is not normalised", PM, "post = np.exp(per_regime - lse[:, None])", "post = np.exp(per_regime)"),
    ("P03 the posterior is returned reversed", PM, "return np.diff(np.concatenate([[0.0], lse])), post", "return np.diff(np.concatenate([[0.0], lse])), post[::-1]"),
]
if __name__ == "__main__":
    sys.exit(R.run(MUT, ["tests/test_wp10a_a5_monitors.py"], sys.argv[1] if len(sys.argv) > 1 else None, workers=1))
