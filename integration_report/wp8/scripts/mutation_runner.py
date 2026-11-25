"""WP8: shared mutation runner. A mutant edits ONE thing (code or doc) in a throw-away copy of the repository; the given tests must FAIL (KILLED).
A mutant whose pattern does not occur exactly once is reported NOT APPLIED and counts as not killed (a silently inert mutant is a defect, WP7-R D14).
usage from a script: sys.exit(run(MUTANTS, ["tests/x.py", ...], "out.txt", workers=2)); MUTANTS = [(name, relative_path, old, new), ...]"""
import shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
IGNORE = shutil.ignore_patterns(".git", "__pycache__", "wp7r_evidence", "wp8_evidence", ".pytest_cache")


def _one(args):
    (name, rel, old, new), tests = args
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "repo"
        shutil.copytree(ROOT, dst, ignore=IGNORE)
        f = dst / rel
        txt = f.read_text()
        if txt.count(old) != 1:
            return name, "NOT APPLIED (pattern count %d)" % txt.count(old)
        f.write_text(txt.replace(old, new, 1))
        r = subprocess.run([sys.executable, "-m", "pytest", *tests, "-q", "-x", "-p", "no:cacheprovider"], cwd=dst,
                           env={"PYTHONPATH": str(dst), "PATH": "/usr/bin:/usr/local/bin:/bin", "HOME": td}, capture_output=True, text=True)
        return name, ("KILLED" if r.returncode != 0 else "SURVIVED")


def run(mutants, tests, out_path=None, workers=2):
    out = open(out_path, "w") if out_path else None
    killed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for name, status in ex.map(_one, [(m, tests) for m in mutants]):
            killed += status == "KILLED"
            line = f"{name}: {status}"
            print(line, flush=True)
            if out:
                out.write(line + "\n"); out.flush()
    summary = f"{killed}/{len(mutants)} killed"
    print(summary)
    if out:
        out.write(summary + "\n"); out.close()
    return 0 if killed == len(mutants) else 1
