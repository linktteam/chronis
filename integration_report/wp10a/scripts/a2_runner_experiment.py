"""WP10A A2 (finding F2): THE EXACT EXPERIMENT FOR THE INTEGRATION RUNNER. Self-contained (standard library plus the repo's own pins); one command, all output also written to OUT_DIR for upload as a CI artifact.
usage (from the repository root, with the pinned environment active):
  PYTHONPATH=. python integration_report/wp10a/scripts/a2_runner_experiment.py OUT_DIR [--runs 20] [--kernels default,HASWELL,SKYLAKEX,ZEN,SANDYBRIDGE] [--threads 1,2,4,unset] [--skip-repeat]
Phase 0  environment: CPU model and vector flags, python, numpy, scipy, the BLAS libraries and the kernel each one actually uses (threadpoolctl), thread counts, OPENBLAS_* variables.
Phase 1  probe: one fresh process per (kernel, threads) builds the fits of tests/test_wp8_item0_metadata.py (converged and not-converged dual fixtures, public-path fits of seeds 1 and 2) and fingerprints them;
         every configuration is compared bit-wise with the first and with the reference fingerprint recorded in this repository (`wp10a_evidence/a2_probe_drive.json`, the default kernel of the development container);
         the DISCRETE outcomes that decide the test (fit states, duration-optimiser fallback of seeds 1 and 2, boundary flags) are printed per configuration: a flipped outcome IS the failure mechanism.
Phase 2  repeat: `tests/test_wp8_item0_metadata.py` is run --runs times in the runner's own default environment with PYTHONHASHSEED = run index, every failure is recorded with its test id and message.
Reading the result: (a) a configuration whose discrete outcomes differ from the reference reproduces the failure mechanism (a CPU-kernel dependent fixture); (b) differing bits but equal discrete outcomes does not; (c) a failure in
phase 2 with equal outcomes in phase 1 means another cause: the recorded test id and message are the starting point."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import a2_probe as PROBE                                                              # noqa: E402
import a2_repeat as REP                                                               # noqa: E402

DISCRETE = ("converged_state", "not_converged_state", "seed1_fallback", "seed2_fallback", "converged_at_bound", "converged_degraded")


def environment() -> dict:
    cpu = {}
    try:
        txt = open("/proc/cpuinfo").read()
        first = txt.split("\n\n")[0]
        cpu["model_name"] = next((l.split(":", 1)[1].strip() for l in first.splitlines() if l.startswith("model name")), None)
        flags = next((l.split(":", 1)[1].split() for l in first.splitlines() if l.startswith("flags")), [])
        cpu["vector_flags"] = [f for f in flags if f.startswith(("avx", "fma", "sse4", "amx"))]
    except OSError:
        pass
    env = dict(platform=platform.platform(), machine=platform.machine(), python=sys.version.split()[0], cpu_count=os.cpu_count(), cpu=cpu,
               openblas_env={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OPENBLAS_CORETYPE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED")})
    try:
        import numpy
        import scipy
        from threadpoolctl import threadpool_info
        env.update(numpy=numpy.__version__, scipy=scipy.__version__, blas=[dict(api=d["internal_api"], arch=d.get("architecture"), threads=d["num_threads"], version=d.get("version"), lib=Path(d["filepath"]).name) for d in threadpool_info()])
    except Exception as exc:                                                          # noqa: BLE001
        env["error"] = repr(exc)
    return env


def discrete_of(result: dict) -> dict:
    return {k: result["outcomes"].get(k) for k in DISCRETE}


def summarise(probe: list, reference: dict | None, repeat: list) -> dict:
    """probe: list of child outputs, each with `config`; reference: a child output (or None); repeat: list of `a2_repeat` records. Pure function (tested)."""
    out = dict(configs=[], discrete_flips=[], bit_differences=[], repeat=dict(runs=len(repeat), failures=[]))
    first = probe[0] if probe else None
    for r in probe:
        name = f"{r['config']['kernel']}/threads={r['config']['threads']}"
        d_ref = PROBE.compare(reference, r) if reference else None
        d_first = PROBE.compare(first, r) if first is not r else []
        disc_differs_ref = [x for x in (d_ref or []) if x[0].startswith("outcomes.")]
        out["configs"].append(dict(config=name, arch=[b["arch"] for b in r["blas"]][:1], threads_effective=[b["threads"] for b in r["blas"]][:1], discrete=discrete_of(r),
                                   fields_differing_from_first=len(d_first), fields_differing_from_reference=(len(d_ref) if d_ref is not None else None),
                                   discrete_differs_from_reference=disc_differs_ref))
        if disc_differs_ref:
            out["discrete_flips"].append(dict(config=name, differs=disc_differs_ref))
        if d_first:
            out["bit_differences"].append(name)
    for rec in repeat:
        if rec["returncode"] != 0 or rec["failed"]:
            out["repeat"]["failures"].append(dict(run=rec["run"], hashseed=rec["hashseed"], failed=rec["failed"], summary=rec["summary"], tb=rec["tb"][:3]))
    out["verdict"] = ("MECHANISM REPRODUCED: a discrete outcome that the test depends on differs between configurations (see discrete_flips)" if out["discrete_flips"] else
                      ("FAILURE REPRODUCED IN THE REPEAT WITHOUT A DISCRETE FLIP IN THE PROBE: another cause (see repeat.failures)" if out["repeat"]["failures"] else
                       "NOT REPRODUCED: no discrete outcome differs and the repeat had no failure"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--kernels", default="default,HASWELL,SKYLAKEX,ZEN,SANDYBRIDGE"); ap.add_argument("--threads", default="1,2,4,unset")
    ap.add_argument("--skip-repeat", action="store_true"); ap.add_argument("--target", default="tests/test_wp8_item0_metadata.py")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    env = environment()
    print("PHASE 0 environment:\n" + json.dumps(env, indent=1), flush=True)
    ref_path = ROOT / "wp10a_evidence" / "a2_probe_drive.json"
    reference = None
    if ref_path.exists():
        ref_runs = json.load(open(ref_path))["runs"]
        reference = next((r for r in ref_runs if r["config"]["kernel"] == "default" and r["config"]["threads"] == "1"), None)
    probe = []
    print("\nPHASE 1 probe (one fresh process per configuration)", flush=True)
    for kernel in a.kernels.split(","):
        for th in a.threads.split(","):
            try:
                r = PROBE.run_one(kernel, th)
            except Exception as exc:                                                  # noqa: BLE001
                print(f"  {kernel}/{th}: FAILED TO RUN: {str(exc)[:200]}", flush=True)
                continue
            r["config"] = dict(kernel=kernel, threads=th, rep=0)
            probe.append(r)
            print(f"  {kernel:11s} threads={th:5s} blas {[(b['arch'], b['threads']) for b in r['blas']][:1]} discrete {discrete_of(r)}", flush=True)
    json.dump(dict(environment=env, probe=probe), open(out / "probe.json", "w"), indent=1)
    repeat = []
    if not a.skip_repeat:
        print(f"\nPHASE 2 repeat: {a.runs} runs of {a.target} in the default environment", flush=True)
        cfg = dict(name="runner_default", threads="unset", kernel="default")
        with open(out / "repeat.jsonl", "w") as fh:
            for i in range(a.runs):
                rec = REP.run_once((cfg, i, a.target))
                repeat.append(rec)
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(f"  run {i}: {rec['summary']} {rec['seconds']}s {rec['failed'][:1]}", flush=True)
    res = summarise(probe, reference, repeat)
    res["environment"] = env
    json.dump(res, open(out / "summary.json", "w"), indent=1)
    print("\nSUMMARY\n" + json.dumps({k: v for k, v in res.items() if k != "environment"}, indent=1), flush=True)
    print(f"\nVERDICT: {res['verdict']}")


if __name__ == "__main__":
    main()
