"""Check that sbom/chronis-ml.cdx.json agrees with requirements.txt and requirements-optional.txt (WP4 item 0d).

Usage (from the repository root): python scripts/check_sbom.py
Exit status 1 if a pin has no SBOM component, an SBOM component is in neither requirements file, or a version differs.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(path: Path) -> dict[str, str]:
    pins = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*([^\s;#]+)", line.split("#", 1)[0].strip())
            if m:
                pins[norm(m.group(1))] = m.group(2)
    return pins


def problems(root: Path = ROOT) -> list[str]:
    sbom = {norm(c["name"]): c["version"] for c in json.loads((root / "sbom" / "chronis-ml.cdx.json").read_text(encoding="utf-8"))["components"]}
    pins = {**read_pins(root / "requirements-optional.txt"), **read_pins(root / "requirements.txt")}
    out = []
    for name, v in sorted(pins.items()):
        if name not in sbom:
            out.append(f"{name}=={v} is pinned but has no SBOM component")
        elif sbom[name] != v:
            out.append(f"{name}: pinned {v}, SBOM {sbom[name]}")
    for name in sorted(set(sbom) - set(pins)):
        out.append(f"{name}=={sbom[name]} is an SBOM component but is in neither requirements file")
    return out


if __name__ == "__main__":
    p = problems()
    print("\n".join(p) if p else "OK: SBOM and requirements files agree")
    sys.exit(1 if p else 0)
