"""Mutation check for tests/test_wp9_a1_consensus.py (WP9 A1: FIT_UNSTABLE consensus). usage: PYTHONPATH=. python3 integration_report/wp9/scripts/mutation_a1.py [out.txt]
A mutant edits ONE thing in a throw-away copy of the repository; the tests must FAIL (KILLED). A pattern that does not occur exactly once is NOT APPLIED and counts as not killed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp8" / "scripts"))
import shutil

import mutation_runner as R

_base = shutil.ignore_patterns(".git", "__pycache__", "wp7r_evidence", "wp8_evidence", "wp9_evidence", ".pytest_cache")
R.IGNORE = lambda d, names: set(_base(d, names))                 # the throw-away copy leaves out the (large) evidence folders; the tests below read none of them

FS, CFG, FIT, LIN = "backbone/hssm/fit_stability.py", "backbone/hssm/config.py", "backbone/hssm/fitting.py", "backbone/hssm/lineage.py"
MUTANTS = [
    ("A01 m = 2 instead of 3", CFG, "FIT_CONSENSUS_M = 3", "FIT_CONSENSUS_M = 2"),
    ("A02 tau 0.9 instead of 0.99", CFG, "FIT_CONSENSUS_TAU = 0.99", "FIT_CONSENSUS_TAU = 0.9"),
    ("A03 tau 0.999 instead of 0.99", CFG, "FIT_CONSENSUS_TAU = 0.99", "FIT_CONSENSUS_TAU = 0.999"),
    ("A04 the level is exclusive at tau", FS, "unstable = min_ari < tau", "unstable = min_ari <= tau"),
    ("A05 the largest pairwise ARI instead of the smallest", FS, "min_ari = min(float(ari(", "min_ari = max(float(ari("),
    ("A06 the worst restarts by log-likelihood are taken", FS, "healthy.sort(key=lambda t: (-t[1][\"ll\"], t[0]))", "healthy.sort(key=lambda t: (t[1][\"ll\"], t[0]))"),
    ("A07 ties go to the higher restart index", FS, "healthy.sort(key=lambda t: (-t[1][\"ll\"], t[0]))", "healthy.sort(key=lambda t: (-t[1][\"ll\"], -t[0]))"),
    ("A08 collapsed restarts are not excluded", FS, "healthy = [(i, r) for i, r in enumerate(records) if not r[\"collapsed\"]]", "healthy = [(i, r) for i, r in enumerate(records)]"),
    ("A09 the too-few rule is removed", FS, "    if len(top) < m:\n        return out", "    if len(top) < 1:\n        return out"),
    ("A10 the reasons are swapped", FS, "fit_unstable_reason=REASON_DISAGREE if unstable else None", "fit_unstable_reason=REASON_TOO_FEW if unstable else None"),
    ("A11 only the agreement with the best restart is compared", FS, "for i in range(m) for j in range(i + 1, m))", "for i in range(1) for j in range(i + 1, m))"),
    # (a mutant "all healthy restarts instead of the best m" is EQUIVALENT: the comparison loops index only the first m restarts of the sorted list; it is not counted)
    ("A13 a fit that cannot be assessed is reported stable", FS, "fit_unstable=None, fit_unstable_reason=REASON_NOT_ASSESSED)", "fit_unstable=False, fit_unstable_reason=REASON_NOT_ASSESSED)"),
    ("A14 the consensus keys are left out of the restart summary", FIT, "        **consensus,\n", ""),
    ("A15 a collapsed selected fit is assessed as if healthy", FIT, "consensus = not_assessed() if selected[\"collapsed\"] else assess_consensus(", "consensus = assess_consensus("),
    ("A16 the consensus compares with a constant ARI", FIT, "assess_consensus(converged_records, adjusted_rand_index)", "assess_consensus(converged_records, lambda a, b: 1.0)"),
    ("A17 the lineage ignores the flag of the new fit", LIN, "        if summary.fit_unstable:  ", "        if False:  "),
    ("A18 the lineage ignores the flag of the anchor", LIN, "        if anchor.summary.fit_unstable:\n", "        if False:\n"),
    ("A19 a baseline result without an assessment is trusted", LIN, "        return True, rs.get(\"fit_unstable_reason\") or REASON_NOT_ASSESSED", "        return False, None"),
    ("A20 the dual engine is flagged", LIN, "        return False, None\n    rs = getattr(result", "        return True, None\n    rs = getattr(result"),
    ("A21 the record drops the flags", LIN, "fit_unstable=bool(summary.fit_unstable), anchor_fit_unstable=False if anchor is None else bool(anchor.summary.fit_unstable), k_change=", "fit_unstable=False, anchor_fit_unstable=False, k_change="),
    ("A22 a summary without a flag counts as flagged", LIN, "    fit_unstable: bool = False                      # WP9 A1", "    fit_unstable: bool = True                       # WP9 A1"),
    ("A23 make_summary drops the flag", LIN, "ctx, bool(fit_unstable), fit_unstable_reason,", "ctx, False, None,"),
    ("A24 summarize_fit drops the flag", LIN, "fit_unstable=unstable, fit_unstable_reason=reason, k_change=k_change)", "fit_unstable=False, fit_unstable_reason=None, k_change=k_change)"),
    ("A26 an unstable anchor is never replaced", LIN, "    if anchor.summary.fit_unstable and not summary.fit_unstable and al is not None:", "    if False:"),
    ("A27 an unstable anchor is replaced by an unstable fit", LIN, "    if anchor.summary.fit_unstable and not summary.fit_unstable and al is not None:", "    if anchor.summary.fit_unstable and al is not None:"),
    ("A28 every unverified step moves the anchor", LIN, "    if anchor.summary.fit_unstable and not summary.fit_unstable and al is not None:", "    if not summary.fit_unstable and al is not None:"),
    ("A29 the replacing anchor carries the identity names instead of the best guess", LIN, "new_anchor = Anchor(summary, tuple(int(x) for x in r2s), fit_index, anchor.activity_rung, anchor.ordering_verified)\n        return rec, replace(state, anchor=new_anchor",
     "new_anchor = Anchor(summary, tuple(range(summary.K)), fit_index, anchor.activity_rung, anchor.ordering_verified)\n        return rec, replace(state, anchor=new_anchor"),
    ("A25 the healthy count includes collapsed restarts", FS, "n_healthy_restarts=len(healthy),", "n_healthy_restarts=len(records),"),
]

if __name__ == "__main__":
    sys.exit(R.run(MUTANTS, ["tests/test_wp9_a1_consensus.py"], sys.argv[1] if len(sys.argv) > 1 else None, workers=1))
