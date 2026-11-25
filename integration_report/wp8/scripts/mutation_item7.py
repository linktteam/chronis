"""Mutation check for tests/test_wp8_item7_expert_review.py (WP8 item 7). usage: PYTHONPATH=. python3 integration_report/wp8/scripts/mutation_item7.py [out.txt]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mutation_runner import run

LN, OR, RP = "backbone/hssm/lineage.py", "backbone/hssm/ordering.py", "backbone/hssm/refit_policy.py"
SC, REG, DOC = "integration_report/wp8/scripts/expert_review_register.py", "docs/EXPERT_REVIEW_REGISTER.md", "docs/INTEGRATION_CONTRACT_HSSM.md"
T = "EXPERT_REVIEW(identity-distance): "
MUTANTS = [
    ("E01 tag removed from step", LN, "# " + T + "the gate, the distance, the thresholds and the continuity state are applied together here.\n", ""),
    ("E02 tag removed from the margin_ok docstring", LN, '"""' + T + "the acceptance rule.", '"""the acceptance rule.'),
    ("E03 status marked as reviewed", LN, 'EXPERT_REVIEW_STATUS = "PENDING"', 'EXPERT_REVIEW_STATUS = "REVIEWED"'),
    ("E04 a threshold default changes without the register", LN, "min_separation: float = ", "min_separation: float = 1.0 + "),
    ("E05 register status line changed", REG, "**Status: PENDING.**", "**Status: REVIEWED.**"),
    ("E06 register claims one more site", REG, "## Tagged sites (", "## Tagged sites (9"),
    ("E07 tag text edited, register not regenerated", LN, "the acceptance rule. A tie between", "the acceptance test. A tie between"),
    ("E08 contract does not name the register", DOC, "`docs/EXPERT_REVIEW_REGISTER.md` lists every tagged site", "the register lists every tagged site"),
    ("E09 a docstring that only mentions the tag counts", SC, 'r"^[ \\t]*" + re.escape(TAG)', 're.escape(TAG)'),
    ("E10 a comment attaches to the definition above it", SC, "nxt = [s for s in stmts if s[0] > line]", "nxt = [s for s in stmts if s[0] < line][::-1]"),
    ("E11 comment tags ignored", SC, "if tok.type == tokenize.COMMENT and", "if tok.type == -1 and"),
    ("E12 record default status empty", LN, "expert_review_status: str = EXPERT_REVIEW_STATUS", 'expert_review_status: str = ""'),
    ("E13 the scan ignores attribute uses", SC, "elif isinstance(sub, ast.Attribute) and sub.attr in watched:", "elif False:"),
    ("E14 the scan ignores name uses", SC, "if isinstance(sub, ast.Name) and sub.id in watched:", "if False:"),
    ("E15 --check always passes", SC, "if not REGISTER.exists() or REGISTER.read_text() != text:", "if False:"),
    ("E16 annotate_result drops the status", LN, "result.lineage = record.as_dict()", 'result.lineage = {k: v for k, v in record.as_dict().items() if k != "expert_review_status"}'),
    ("E17 the register prints no values", SC, "for item, val, where in rows:", "for item, val, where in rows[:0]:"),
    ("E18 qualified names flattened", SC, "q = prefix + child.name", "q = child.name"),
    ("E19 a watched name is renamed", SC, '"alignment", "alignment_from_cost", "margin_ok",', '"alignment", "alignment_from_cost", "margin_okk",'),
    ("E20 a hard ceiling value changes without the register", LN, "AMBIGUITY_RATIO = 0.5 ", "AMBIGUITY_RATIO = 0.4 "),
    ("E21 tag removed from the refit module docstring", RP, T + "the cadence changes", "the cadence changes"),
    ("E22 a dangling tag comment is skipped", SC, '                raise ValueError(f"{rel}:{line}: a {TAG} comment is not followed by a definition or assignment")', "                continue"),
    ("E23 tag removed from the ladder weights", OR, '"""' + T + "the ladder weights are computed here", '"""the ladder weights are computed here'),
    ("E25 the dual class threshold is not watched", SC, '"same_engine", "dual_engine", "cross_engine"', '"same_engine", "cross_engine"'),
    ("E24 the register status text is dropped", SC, '"**Status: PENDING.** No expert has reviewed', '"**Status: open.** No expert has reviewed'),
]

if __name__ == "__main__":
    sys.exit(run(MUTANTS, ["tests/test_wp8_item7_expert_review.py"], sys.argv[1] if len(sys.argv) > 1 else None, workers=1))
