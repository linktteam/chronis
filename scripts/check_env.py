"""Check that the active Python environment matches requirements.txt (WP4 item 0d).

Usage (from the repository root, inside the virtual environment):
    python scripts/check_env.py [--requirements requirements.txt]

Prints the Python version and, for every pinned library, the pinned and the installed version. Exit status 1 if any library is missing or its installed
version differs from the pin, 0 if all match. A Python version different from `.python-version` is printed as a warning and does not fail the check.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import re
import sys
from pathlib import Path

try:  # packaging is itself pinned; the fallback keeps the script usable in a bare interpreter
    from packaging.version import InvalidVersion, Version
except Exception:  # pragma: no cover
    Version = None  # type: ignore[assignment]
    InvalidVersion = Exception  # type: ignore[assignment,misc]


def read_pins(path: Path) -> list[tuple[str, str]]:
    pins = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*([^\s;]+)", line)
        if m:
            pins.append((m.group(1), m.group(2)))
    return pins


def same_version(installed: str, pinned: str) -> bool:
    if Version is not None:
        try:
            return Version(installed) == Version(pinned)
        except InvalidVersion:
            pass
    return installed == pinned


def compare(pins: list[tuple[str, str]]) -> list[tuple[str, str, str | None, str]]:
    """Returns (name, pinned, installed or None, status) for every pin; status is OK, MISMATCH or MISSING."""
    rows = []
    for name, pinned in pins:
        try:
            installed = md.version(name)
        except md.PackageNotFoundError:
            rows.append((name, pinned, None, "MISSING"))
            continue
        rows.append((name, pinned, installed, "OK" if same_version(installed, pinned) else "MISMATCH"))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--requirements", default=None, help="default: requirements.txt in the repository root")
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    req_path = Path(args.requirements) if args.requirements else root / "requirements.txt"
    pins = read_pins(req_path)
    have = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python {sys.version.split()[0]} ({sys.executable})")
    pv = root / ".python-version"
    if pv.exists() and pv.read_text().strip() != have:
        print(f"WARNING: target Python is {pv.read_text().strip()} (.python-version), this is {have}; not an error by itself")
    rows = compare(pins)
    width = max((len(r[0]) for r in rows), default=10)
    for name, pinned, installed, status in rows:
        print(f"  {name:<{width}}  pinned {pinned:<14} installed {installed or '-':<14} {status}")
    bad = [r for r in rows if r[3] != "OK"]
    if bad:
        print(f"FAIL: {len(bad)} of {len(rows)} pinned libraries differ: " + ", ".join(f"{r[0]} ({r[3].lower()})" for r in bad))
        return 1
    print(f"OK: all {len(rows)} pinned libraries match {req_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
