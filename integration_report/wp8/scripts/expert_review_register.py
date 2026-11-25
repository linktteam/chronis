"""WP8 Item 7 (owner decision A7): the expert-review register.

Every place in `backbone/hssm/` that defines or uses the matching distance, the ladder weights or an alignment threshold carries the tag
`EXPERT_REVIEW(identity-distance)` (in a docstring that starts with it, or in a comment line directly above the definition). This script finds
every tag with the `ast` and `tokenize` modules, names the enclosing definition, and writes `docs/EXPERT_REVIEW_REGISTER.md`. The register also prints the numbers
the expert is asked to review, read from the code (so the register cannot show a value the code does not have).

  python3 integration_report/wp8/scripts/expert_review_register.py --write    # regenerate docs/EXPERT_REVIEW_REGISTER.md
  python3 integration_report/wp8/scripts/expert_review_register.py --check    # exit 1 if the committed register differs from the code

`tests/test_wp8_item7_expert_review.py` runs the same check. Line numbers are left out on purpose, so that an unrelated edit does not make the register stale;
a site is identified by file, enclosing definition and kind (docstring or comment). The tag text is printed, so a changed tag also makes the register stale."""
from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from dataclasses import fields, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "backbone" / "hssm"
REGISTER = ROOT / "docs" / "EXPERT_REVIEW_REGISTER.md"
TAG = "EXPERT_REVIEW(identity-distance)"
TAG_RE = re.compile(re.escape(TAG) + r":?\s*(.*)", re.S)
DOC_TAG_RE = re.compile(r"^[ \t]*" + re.escape(TAG) + r":?[ \t]*(.*?)(?:\n[ \t]*\n|\Z)", re.S | re.M)


def _qualified(tree: ast.Module):
    """Yields (qualified name, node) for every class, function and simple assignment: `Class.method`, `outer.inner`, module level names bare."""
    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = prefix + child.name
                yield q, child
                yield from visit(child, q + ".")
            elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                tgt = child.targets[0] if isinstance(child, ast.Assign) else child.target
                if isinstance(tgt, ast.Name):
                    yield prefix + tgt.id, child
            else:
                yield from visit(child, prefix)
    yield from visit(tree, "")


def _owner_of_docstring(tree: ast.Module):
    """Yields (owner name, docstring text) for the module and every class and function that has a docstring."""
    mod_doc = ast.get_docstring(tree, clean=True)
    if mod_doc:
        yield "<module>", mod_doc
    for q, node in _qualified(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            d = ast.get_docstring(node, clean=True)
            if d:
                yield q, d


def _statements(tree: ast.Module):
    """(first line including decorators, qualified name) of every statement that a comment can describe: definitions and assignments."""
    out = []
    for q, node in _qualified(tree):
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        out.append((first, q))
    return sorted(out)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def find_sites(path: Path, root: Path = ROOT) -> list[dict]:
    """Every tag in one file: {file, symbol, kind, text}. A tag comment belongs to the first definition or assignment that starts after it."""
    src = path.read_text()
    tree = ast.parse(src)
    rel = path.relative_to(root).as_posix()
    sites = []
    for owner, doc in _owner_of_docstring(tree):
        # a docstring tag counts only where a line of the docstring starts with it, which keeps prose that merely mentions the tag out of the register
        for m in DOC_TAG_RE.finditer(doc):
            sites.append(dict(file=rel, symbol=owner, kind="docstring", text=_clean(m.group(1))))
    stmts = _statements(tree)
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT and tok.string.lstrip("# ").startswith(TAG) and tok.line.strip().startswith("#"):
            line = tok.start[0]
            nxt = [s for s in stmts if s[0] > line]
            if not nxt:
                raise ValueError(f"{rel}:{line}: a {TAG} comment is not followed by a definition or assignment")
            sites.append(dict(file=rel, symbol=nxt[0][1], kind="comment", text=_clean(TAG_RE.match(tok.string.lstrip("# ")).group(1))))
    return sites


# Names whose use means "this code uses the matching distance, a ladder weight or an alignment threshold". A function that mentions one of them must carry the tag.
WATCHED = frozenset({
    "alignment", "alignment_from_cost", "margin_ok", "order_regimes", "separation_min", "hellinger2_matrix", "_vectors", "_scale", "_mean_pair", "_correspondence",
    "split_merge_candidates", "name_fit", "_combined_alignment", "_rung_weights_scores", "_bootstrap_modal", "_stats",
    "min_separation", "same_engine", "dual_engine", "cross_engine", "duration_weight", "min_modal_freq", "min_gap", "context_threshold",
})


def untagged_uses(path: Path, watched=WATCHED) -> list[tuple[str, str]]:
    """(function, watched name) for every function in `path` that mentions a watched name but carries no tag (neither in its docstring nor in a comment directly above it).
    A function that is itself a watched name is a definition and is checked the same way."""
    src = path.read_text()
    tree = ast.parse(src)
    tagged = {s["symbol"] for s in find_sites(path, path.parent)}
    bad = []
    for q, node in _qualified(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            used = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and sub.id in watched:
                    used.add(sub.id)
                elif isinstance(sub, ast.Attribute) and sub.attr in watched:
                    used.add(sub.attr)
            used.discard(node.name)          # a function is not a "use" of itself
            if (used or node.name in watched) and q not in tagged:
                bad.append((q, ", ".join(sorted(used)) or "(definition)"))
    return sorted(bad)


def all_sites() -> list[dict]:
    sites = []
    for p in sorted(PACKAGE.glob("*.py")):
        sites.extend(find_sites(p))
    # one row per (file, symbol, kind); a second tag on the same symbol and kind is numbered
    seen: dict = {}
    for s in sites:
        k = (s["file"], s["symbol"], s["kind"])
        seen[k] = seen.get(k, 0) + 1
        s["n"] = seen[k]
    return sorted(sites, key=lambda s: (s["file"], s["symbol"] != "<module>", s["symbol"], s["kind"], s["n"]))


def _fmt(v) -> str:
    if is_dataclass(v) and not isinstance(v, type):
        return type(v).__name__ + "(" + ", ".join(f"{f.name}={_fmt(getattr(v, f.name))}" for f in fields(v)) + ")"
    return repr(v)


def reviewed_values() -> list[tuple[str, str, str]]:
    """(item, value, where) read from the code. These are the numbers the expert is asked to review."""
    sys.path.insert(0, str(ROOT))
    from backbone.hssm import lineage as L, ordering as O, refit_policy as R
    rows = [
        ("Ladder configuration (`ordering.LadderConfig()`)", _fmt(O.LadderConfig()), "backbone/hssm/ordering.py"),
        ("Rung definitions (`ordering.RUNG_ROLES`)", _fmt(O.RUNG_ROLES), "backbone/hssm/ordering.py"),
        ("Alignment configuration (`lineage.AlignmentConfig()`)", _fmt(L.AlignmentConfig()), "backbone/hssm/lineage.py"),
        ("Channel configuration (`lineage.ChannelConfig()`)", _fmt(L.ChannelConfig()), "backbone/hssm/lineage.py"),
        ("Hard ceiling of the ratio rule (`lineage.AMBIGUITY_RATIO`)", _fmt(L.AMBIGUITY_RATIO), "backbone/hssm/lineage.py"),
        ("Largest K compared by enumeration (`lineage.MAX_MATCH_K`)", _fmt(L.MAX_MATCH_K), "backbone/hssm/lineage.py"),
        ("Refit cadence (`refit_policy.RefitPolicy()`), bounds", _fmt(R.RefitPolicy()) + f", bounds {R.MIN_REFIT_INTERVAL_DAYS}..{R.MAX_REFIT_INTERVAL_DAYS} days", "backbone/hssm/refit_policy.py"),
        ("Status written on every record (`lineage.EXPERT_REVIEW_STATUS`)", _fmt(L.EXPERT_REVIEW_STATUS), "backbone/hssm/lineage.py"),
    ]
    return rows


def render() -> str:
    sites = all_sites()
    rows = reviewed_values()
    out = [
        "# Expert-review register: identity distance (WP8, owner decision A7)",
        "",
        "<!-- GENERATED by integration_report/wp8/scripts/expert_review_register.py --write. Do not edit by hand. tests/test_wp8_item7_expert_review.py fails when this file is stale. -->",
        "",
        "**Status: PENDING.** No expert has reviewed any of the items below. Every lineage record carries `expert_review_status = \"PENDING\"` until the owner records a sign-off here "
        "and in `lineage.EXPERT_REVIEW_STATUS`.",
        "",
        "What the expert is asked to review: (1) the **matching distance** (regime means in raw units, divided by the anchor's stored per-feature scale, optionally with a duration "
        "coordinate and a second, contextual channel) and the **identifiability gate** that decides whether a fit can be matched at all; (2) the **ladder weights and the rung "
        "definitions** that give regimes their stable names at a lineage's first fit; (3) **every threshold** that turns a distance margin into ALIGNED or CONTINUITY_UNVERIFIED, and the refit cadence that sets how far apart two matched fits are.",
        "",
        "Marking rule: the tag `EXPERT_REVIEW(identity-distance)` stands at the start of a docstring or in a comment directly above a definition or assignment, at each place that "
        "**defines or uses** the matching distance, a ladder weight or an alignment threshold. Line numbers are left out on purpose.",
        "",
        "## Values under review (read from the code when this file is generated)",
        "",
        "| Item | Value | File |",
        "|---|---|---|",
    ]
    for item, val, where in rows:
        out.append(f"| {item} | `{val}` | `{where}` |")
    out += ["", f"## Tagged sites ({len(sites)})", "", "| # | File | Definition | Tag in | What is under review |", "|---|---|---|---|---|"]
    for i, s in enumerate(sites, 1):
        sym = "module" if s["symbol"] == "<module>" else f"`{s['symbol']}`"
        text = s["text"].replace("|", "\\|")
        out.append(f"| {i} | `{s['file']}` | {sym} | {s['kind']} | {text} |")
    out += ["", "## Sign-off", "", "Recorded by: nobody yet. Date: not signed off. Scope: not signed off.", ""]
    return "\n".join(out)


def main(argv: list[str]) -> int:
    text = render()
    if "--write" in argv:
        REGISTER.write_text(text)
        print(f"wrote {REGISTER} ({len(all_sites())} sites)")
        return 0
    if "--check" in argv:
        if not REGISTER.exists() or REGISTER.read_text() != text:
            print("EXPERT_REVIEW_REGISTER.md is missing or stale: run --write", file=sys.stderr)
            return 1
        print("register is current")
        return 0
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
