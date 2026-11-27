"""WP10B C1(3) (a): does the numpy AVX-512 dispatch decide the WP10A F2 probe outcome `seed1_fallback`? (request of opa after the AMD EPYC 7763 runner experiment: seed1_fallback True there in every configuration, False on this machine and on an EPYC 9V74.)
Each configuration is one FRESH child process that (i) reports which CPU features numpy actually enabled and (ii) runs the WP10A probe child (`integration_report/wp10a/scripts/a2_probe.py --child`: the fits of tests/test_wp8_item0_metadata.py and the public-path fits of the optimiser-success test, seeds 1 and 2).
Configurations: default | NO_AVX512 (NPY_DISABLE_CPU_FEATURES = every AVX-512 name numpy dispatches on) | NO_AVX512_ZEN (the same plus OPENBLAS_CORETYPE=ZEN, closest to the 7763's dispatch) | SKX_ONLY_OFF (AVX512_SKX/CLX/CNL/ICL off, F and CD on).
Compares each configuration with `default`: the discrete outcomes and every fingerprint field, and says which fits differ and by how much (largest relative difference of the stored floats).
usage: PYTHONPATH=. python3.12 integration_report/wp10b/scripts/c1_cpu_features.py OUT.json [--configs default,NO_AVX512,...] [--repeat N]
       child mode (internal): ... --child"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "integration_report" / "wp10a" / "scripts"))

AVX512_ALL = "AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"
CONFIGS = {
    "default": {},
    "NO_AVX512": {"NPY_DISABLE_CPU_FEATURES": AVX512_ALL},
    "NO_AVX512_ZEN": {"NPY_DISABLE_CPU_FEATURES": AVX512_ALL, "OPENBLAS_CORETYPE": "ZEN"},
    "SKX_ONLY_OFF": {"NPY_DISABLE_CPU_FEATURES": "AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"},
    "CD_AND_UP_OFF": {"NPY_DISABLE_CPU_FEATURES": "AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"},      # AVX512F stays on
}


def child():
    import numpy as np
    from numpy._core._multiarray_umath import __cpu_features__ as f
    import a2_probe as P
    sys.path.insert(0, str(ROOT))
    enabled = sorted(k for k, v in f.items() if v)
    print("CPU_JSON " + json.dumps(dict(numpy=np.__version__, enabled=enabled, avx512_enabled=[k for k in enabled if k.startswith("AVX512")])), flush=True)
    P.child()


def run_one(name):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    for k in ("OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NPY_DISABLE_CPU_FEATURES", "NPY_ENABLE_CPU_FEATURES"):
        env.pop(k, None)
    env.update(CONFIGS[name])
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child"], env=env, capture_output=True, text=True, cwd=str(ROOT))
    cpu = next((l for l in r.stdout.splitlines() if l.startswith("CPU_JSON ")), None)
    pr = next((l for l in r.stdout.splitlines() if l.startswith("PROBE_JSON ")), None)
    if cpu is None or pr is None:
        raise RuntimeError(f"child failed for {name}: {r.stderr[-600:]}")
    out = json.loads(pr[len("PROBE_JSON "):])
    out["cpu"] = json.loads(cpu[len("CPU_JSON "):])
    out["config"] = dict(name=name, env=CONFIGS[name])
    out["stderr_tail"] = r.stderr[-300:]
    return out


def float_rel_diff(a, b):
    try:
        x, y = float.fromhex(a), float.fromhex(b)
    except Exception:
        return None
    return abs(x - y) / max(abs(x), abs(y), 1e-300)


def main():
    import a2_probe as P
    args = sys.argv[1:]
    out_path = args[0]
    names = list(CONFIGS)
    repeat = 1
    if "--configs" in args:
        names = args[args.index("--configs") + 1].split(",")
    if "--repeat" in args:
        repeat = int(args[args.index("--repeat") + 1])
    runs = []
    for n in names:
        for rep in range(repeat):
            r = run_one(n)
            r["config"]["rep"] = rep
            runs.append(r)
            print(f"[{n:14s} rep {rep}] numpy AVX-512 enabled: {r['cpu']['avx512_enabled'] or 'none'}; BLAS {[(b['api'], b['arch'], b['threads']) for b in r['blas']]}\n    outcomes {r['outcomes']}", flush=True)
    ref = runs[0]
    summary = []
    print(f"\ncomparison with the first configuration ({ref['config']['name']}):")
    for r in runs[1:]:
        diffs = P.compare(ref, r)
        fits = sorted({d[0].split(".")[0] for d in diffs})
        outc = [d for d in diffs if d[0].startswith("outcomes.")]
        rel = {}
        for k, a, b in diffs:
            if isinstance(a, str) and isinstance(b, str):
                d = float_rel_diff(a, b)
                if d is not None:
                    fit = k.split(".")[0]
                    rel[fit] = max(rel.get(fit, 0.0), d)
        summary.append(dict(config=r["config"], n_differing_fields=len(diffs), fits_differing=fits, outcomes_differ=outc, max_rel_diff_by_fit=rel))
        print(f"  {r['config']['name']} rep {r['config']['rep']}: {'BIT-IDENTICAL on every fit' if not diffs else str(len(diffs)) + ' fields differ in ' + str(fits)}; discrete outcomes that differ: {[(d[0], d[1], d[2]) for d in outc] or 'none'}; largest relative difference of a stored float per fit: " + (", ".join(f"{f} {v:.1e}" for f, v in sorted(rel.items())) or "n/a"))
    json.dump(dict(runs=runs, summary=summary), open(out_path, "w"), indent=1)


if __name__ == "__main__":
    child() if "--child" in sys.argv else main()
