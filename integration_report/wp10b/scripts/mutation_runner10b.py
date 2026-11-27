"""WP10B mutation runner (the brief's rule, section 4): a mutant is CAUGHT only if a NAMED test fails for the PLANTED reason. Stricter than the WP8 runner (`wp8/scripts/mutation_runner.py`, which counts any non-zero exit):
  * the UNMUTATED control runs first, in the same invocation, on the same named tests; if any of them does not pass, nothing is reported as caught (CONTROL FAILED);
  * a mutant carries the node ids of the test(s) that are supposed to catch it; only those tests are run, with a junit report; the mutant is CAUGHT iff at least one named test has outcome `failure` whose type is
    AssertionError (the test's own check failed). An `error` (import error, collection error, fixture error), a failure of another type (the planted code crashed: TypeError, NameError, SyntaxError, ...), a timeout, or
    a named test that did not run, is NOT caught (status CRASH / TIMEOUT / NOT RUN);
  * a pattern that does not occur exactly once is NOT APPLIED and counts as not caught;
  * SURVIVED = every named test passed.
Limit, stated plainly: "for the planted reason" is approximated by "the named test's own assertion failed, not a crash". Each mutant names the one rule it breaks and the test written for that rule; whether a test that
passes for the wrong reason exists is judged when the survivors are read.
usage from a script: sys.exit(run(MUTANTS, "out.txt", workers=2)); MUTANTS = [(name, relative_path, old, new, [node ids]), ...]  (node ids relative to the repository root, e.g. "tests/test_x.py::test_y")"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_BASE_IGNORE = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "wp7r_evidence", "wp8_evidence", "wp9_evidence", "wp10a_evidence", "wp10b_evidence")
TIMEOUT = 600


def IGNORE(dirpath, names):
    return set(_BASE_IGNORE(dirpath, names))


def classify(junit_xml: str, named: list) -> tuple:
    """Pure function (tested): the status of a mutant from a junit report and the node ids that were meant to catch it. Returns (status, detail).
    status: CAUGHT (a named test failed with AssertionError), SURVIVED (all named tests ran and passed), CRASH (a named test failed or errored for another reason), NOT RUN (a named test is absent)."""
    root = ET.fromstring(junit_xml)
    cases = {}
    for tc in root.iter("testcase"):
        key = (tc.get("classname", "") + "::" + tc.get("name", ""))
        fail = tc.find("failure")
        err = tc.find("error")
        skip = tc.find("skipped")
        if fail is not None:
            head = (fail.get("message") or "").lstrip()
            msg = head + " " + (fail.text or "")[:400]
            # the test's OWN verdict: a bare `assert` (message "assert ..."), an AssertionError, or `Failed:` (pytest.fail / DID NOT RAISE); any other exception type is the planted code crashing
            kind = "assert" if head.startswith(("assert", "AssertionError", "Failed:")) else "other"
            cases[key] = ("failure_" + kind, msg.strip()[:200])
        elif err is not None:
            cases[key] = ("error", ((err.get("message") or "") + " " + (err.text or ""))[:200].strip())
        elif skip is not None:
            cases[key] = ("skipped", "")
        else:
            cases[key] = ("passed", "")
    want = []
    for nid in named:
        path, _, name = nid.partition("::")
        mod = path[:-3].replace("/", ".") if path.endswith(".py") else path
        want.append((mod, name))
    found = []
    for mod, name in want:
        hits = [(k, v) for k, v in cases.items() if k.startswith(mod + "::") and (k.split("::", 1)[1] == name or k.split("::", 1)[1].startswith(name + "["))]
        found.append(hits)
    if any(not h for h in found):
        return "NOT RUN", "a named test is absent from the report"
    flat = [v for h in found for _, v in h]
    if any(s == "failure_assert" for s, _ in flat):
        return "CAUGHT", next(d for s, d in flat if s == "failure_assert")
    if any(s in ("failure_other", "error") for s, _ in flat):
        return "CRASH", next(d for s, d in flat if s in ("failure_other", "error"))
    if any(s == "skipped" for s, _ in flat):
        return "NOT RUN", "a named test was skipped"
    return "SURVIVED", ""


def _pytest(dst: Path, named: list, td: str, junit: Path, timeout: int) -> tuple:
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", *named, "-q", "-p", "no:cacheprovider", f"--junitxml={junit}"], cwd=dst,
                           env={"PYTHONPATH": str(dst), "PATH": "/usr/bin:/usr/local/bin:/bin", "HOME": td}, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout[-600:]
    except subprocess.TimeoutExpired:
        return None, "timeout"


def _one(args):
    (name, rel, old, new, named), timeout = args
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "repo"
        shutil.copytree(ROOT, dst, ignore=IGNORE)
        f = dst / rel
        txt = f.read_text()
        if txt.count(old) != 1:
            return name, "NOT APPLIED", f"pattern count {txt.count(old)}"
        f.write_text(txt.replace(old, new, 1))
        junit = Path(td) / "junit.xml"
        rc, tail = _pytest(dst, named, td, junit, timeout)
        if rc is None:
            return name, "TIMEOUT", "not caught"
        if not junit.exists():
            return name, "CRASH", "no junit report (collection or import error): " + tail.replace("\n", " ")[-200:]
        status, detail = classify(junit.read_text(), named)
        return name, status, detail


def control(all_named: list, timeout: int = TIMEOUT) -> tuple:
    """The unmutated control: every named test must pass on the unmutated copy, in this same run."""
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "repo"
        shutil.copytree(ROOT, dst, ignore=IGNORE)
        junit = Path(td) / "junit.xml"
        rc, tail = _pytest(dst, all_named, td, junit, timeout)
        if rc != 0:
            return False, f"exit {rc}: " + tail.replace("\n", " ")[-300:]
        return True, "all named tests pass unmutated"


def run(mutants: list, out_path: str | None = None, workers: int = 2, timeout: int = TIMEOUT) -> int:
    out = open(out_path, "w") if out_path else None

    def P(s):
        print(s, flush=True)
        if out:
            out.write(s + "\n"); out.flush()
    all_named = sorted({n for m in mutants for n in m[4]})
    ok, detail = control(all_named, timeout)
    P(f"CONTROL (unmutated, {len(all_named)} named tests, same run): {'PASSED' if ok else 'FAILED'}: {detail}")
    if not ok:
        P("0 caught (control failed: no mutant result is valid)")
        if out:
            out.close()
        return 2
    caught = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for name, status, det in ex.map(_one, [(m, timeout) for m in mutants]):
            caught += status == "CAUGHT"
            P(f"{name}: {status}" + (f"  [{det[:140]}]" if det and status != "CAUGHT" else ""))
    P(f"{caught}/{len(mutants)} caught")
    if out:
        out.close()
    return 0 if caught == len(mutants) else 1
