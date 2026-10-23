"""Scan the third-party imports under backbone/, data/ and tests/ and compare them with requirements.txt (WP4 item 0d).

Usage (from the repository root):
    python scripts/scan_imports.py [--requirements requirements.txt] [--optional requirements-optional.txt]

Imports inside functions and try/except blocks count; relative imports are first-party. The exit status is 1 if a third-party
distribution imported by the scanned directories is not pinned in the requirements file, 0 otherwise.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

SCANNED_DIRS = ("backbone", "data", "tests")
FIRST_PARTY = {"backbone", "data", "tests", "scripts"}
# import name -> distribution name, for the cases where they differ
IMPORT_TO_DIST = {
    "sklearn": "scikit-learn", "dateutil": "python-dateutil", "yaml": "PyYAML", "PIL": "Pillow", "cv2": "opencv-python",
    "bs4": "beautifulsoup4", "attr": "attrs", "_pytest": "pytest", "pyannote": "pyannote.audio", "whisper": "openai-whisper",
    "typing_extensions": "typing_extensions", "mypy_extensions": "mypy_extensions",
}


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    if not path.exists():
        return pins
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*([^\s;]+)", line)
        if m:
            pins[norm(m.group(1))] = m.group(2)
    return pins


def scan(root: Path) -> dict[str, dict]:
    """distribution (normalised) -> {'imports': set of import names, 'files': {path: first line}}"""
    found: dict[str, dict] = defaultdict(lambda: {"imports": set(), "files": {}})
    for d in SCANNED_DIRS:
        for path in sorted((root / d).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module.split(".")[0]]
                for n in names:
                    if n in sys.stdlib_module_names or n in FIRST_PARTY or n == "__future__":
                        continue
                    dist = norm(IMPORT_TO_DIST.get(n, n))
                    rec = found[dist]
                    rec["imports"].add(n)
                    rec["files"].setdefault(str(path.relative_to(root)), node.lineno)
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--requirements", default="requirements.txt")
    ap.add_argument("--optional", default="requirements-optional.txt")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    req = read_pins(root / args.requirements)
    opt = read_pins(root / args.optional)
    found = scan(root)
    uncovered = []
    print(f"third-party distributions imported under {', '.join(SCANNED_DIRS)}: {len(found)}")
    for dist in sorted(found):
        rec = found[dist]
        where = "requirements.txt" if dist in req else ("requirements-optional.txt" if dist in opt else "NOT PINNED")
        if where == "NOT PINNED":
            uncovered.append(dist)
        dirs = sorted({Path(f).parts[0] for f in rec["files"]})
        first = next(iter(rec["files"].items()))
        print(f"  {dist:<16} imports {sorted(rec['imports'])} in {len(rec['files'])} file(s) ({', '.join(dirs)}), e.g. {first[0]}:{first[1]} -> {where}"
              + (f" =={req[dist]}" if dist in req else ""))
    only_pinned = sorted(d for d in (set(req) | set(opt)) if d not in found)
    print(f"pinned but not imported by the scanned directories ({len(only_pinned)}): {', '.join(only_pinned) if only_pinned else '-'}")
    if opt:
        for dist in sorted(opt):
            needs = sorted(found[dist]["files"]) if dist in found else []
            print(f"  optional {dist}=={opt[dist]}: " + (f"imported by {', '.join(needs)}" if needs else "not imported anywhere in this repo"))
    if uncovered:
        print("FAIL: imported but not pinned:", ", ".join(uncovered))
        return 1
    print("OK: every third-party import is pinned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
