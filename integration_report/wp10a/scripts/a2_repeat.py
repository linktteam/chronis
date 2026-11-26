"""WP10A A2 (finding F2): run `tests/test_wp8_item0_metadata.py` N times per BLAS configuration and count passes; every failure is recorded with the test id and the first lines of its message. Resumable (a finished
(config, run) is skipped); one JSON line per run is appended and fsynced. Run i of EVERY configuration uses PYTHONHASHSEED = i, so that a failure is reproducible from (configuration, i) and the same hash seed
can be compared across configurations.
Configurations: name:threads:kernel, threads in 1, 2 or unset (= OpenBLAS default = all cores), kernel = OPENBLAS_CORETYPE or `default`.
usage: PYTHONPATH=. python3.12 integration_report/wp10a/scripts/a2_repeat.py OUT.jsonl --runs 50 --configs t1:1:default,t2:2:default,tdef:unset:default,haswell:1:HASWELL,skylakex:1:SKYLAKEX [--workers 2] [--target FILE]"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def parse(cfg: str) -> dict:
    name, th, kernel = cfg.split(":")
    return dict(name=name, threads=th, kernel=kernel)


def run_once(job):
    cfg, i, target = job
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONHASHSEED=str(i))
    for k in ("OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        env.pop(k, None)
    if cfg["kernel"] != "default":
        env["OPENBLAS_CORETYPE"] = cfg["kernel"]
    if cfg["threads"] != "unset":
        env["OPENBLAS_NUM_THREADS"] = cfg["threads"]
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", target, "-q", "-p", "no:cacheprovider", "-rf", "--no-header", "--tb=line"], env=env, cwd=str(ROOT), capture_output=True, text=True)
    out = r.stdout
    last = out.strip().splitlines()[-1] if out.strip() else ""
    failed = [l[len("FAILED "):][:400] for l in out.splitlines() if l.startswith("FAILED ")]
    tb = [l[:300] for l in out.splitlines() if re.match(r"^/|^[\w/\.]+\.py:\d+:", l)][:6]
    return dict(config=cfg["name"], threads=cfg["threads"], kernel=cfg["kernel"], run=i, hashseed=i, returncode=r.returncode, summary=last, failed=failed, tb=tb, seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--runs", type=int, default=50); ap.add_argument("--configs", required=True); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--target", default="tests/test_wp8_item0_metadata.py"); ap.add_argument("--count", action="store_true")
    a = ap.parse_args()
    cfgs = [parse(c) for c in a.configs.split(",")]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line); done.add((r["config"], r["run"]))
            except Exception:
                pass
    todo = [(c, i, a.target) for i in range(a.runs) for c in cfgs if (c["name"], i) not in done]
    print(f"{len(cfgs) * a.runs} runs, {len(cfgs) * a.runs - len(todo)} done, {len(todo)} to run", flush=True)
    if a.count:
        return
    with open(a.out, "a") as fh, ThreadPoolExecutor(max_workers=a.workers) as ex:
        for n, rec in enumerate(ex.map(run_once, todo), 1):
            fh.write(json.dumps(rec) + "\n"); fh.flush(); os.fsync(fh.fileno())
            print(n, "/", len(todo), rec["config"], rec["run"], rec["summary"], f"{rec['seconds']}s", rec["failed"][:1], flush=True)


if __name__ == "__main__":
    main()
