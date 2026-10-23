#!/usr/bin/env bash
# N20 and "hardened" line 8: a FRESH virtual environment on Python 3.12, install exactly requirements.txt, then run the whole suite from the repository root with no extra flags.
# Needs access to PyPI. NOT RUN in the Claude sessions (PyPI unreachable there: DNS failure), so the install step and everything after it are UNTESTED.
# Usage: bash scripts/n20_run_under_pins.sh [path-to-repository-root]      (PYTHON=/path/to/python3.12 overrides the interpreter; works in Git Bash on Windows too)
set -euo pipefail
REPO="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
PY="${PYTHON:-python3.12}"
command -v "$PY" >/dev/null 2>&1 || { echo "ERROR: $PY not found; install Python 3.12 or set PYTHON=..."; exit 2; }
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || { echo "ERROR: $PY is not Python 3.12"; exit 2; }
VENV="$(mktemp -d)/venv"
"$PY" -m venv "$VENV"
if [ -d "$VENV/Scripts" ]; then BIN="$VENV/Scripts"; else BIN="$VENV/bin"; fi
"$BIN/python" -m pip install -r requirements.txt
"$BIN/python" scripts/check_env.py          # fails if any installed version differs from the pin
"$BIN/python" scripts/scan_imports.py       # fails if the code imports a library that is not pinned
"$BIN/python" scripts/check_sbom.py         # fails if the SBOM and the requirements files disagree
"$BIN/python" -m pytest -q 2>&1 | tee n20_pinned_full_run.log | tail -n 30    # no extra flags: pytest.ini sets testpaths = tests
