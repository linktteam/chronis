"""WP10A A2 (finding F2): probe of the fits that `tests/test_wp8_item0_metadata.py` builds, under forced BLAS settings.
Child mode (one fresh process = one configuration): builds the fits of the test file (the CONVERGED and the MAX_ITER_REACHED dual models of the module fixtures, and the two public-path fits of the
optimiser-success test, seeds 1 and 2) and prints one JSON line: the numerical fingerprints (sha256 of every array, hex of every float) and the discrete outcomes (fit state, iterations, duration optimiser
fallback flags, boundary flags) next to the BLAS library actually in use (threadpoolctl).
Driver mode: runs the child once per configuration (kernel x threads), compares every configuration with the first and prints, per fit, whether it is BIT-IDENTICAL, the largest absolute difference
and any discrete outcome that differs.
usage: PYTHONPATH=. python3.12 integration_report/wp10a/scripts/a2_probe.py --child
       PYTHONPATH=. python3.12 integration_report/wp10a/scripts/a2_probe.py --drive OUT.json [--kernels default,HASWELL,SKYLAKEX,SANDYBRIDGE,NEHALEM] [--threads 1,2,unset] [--repeat 1]"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def fingerprint(obj, depth=0):
    """Canonical, comparable summary of a fitted object: arrays -> (sha256 of the bytes, shape, dtype, first values as hex); floats -> hex; containers -> recursion; everything else -> repr."""
    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        return dict(sha=hashlib.sha256(a.tobytes()).hexdigest()[:16], shape=list(a.shape), dtype=str(a.dtype), head=[float(x).hex() for x in a.ravel()[:6]] if a.dtype.kind == "f" else None)
    if isinstance(obj, (float, np.floating)):
        return dict(f=float(obj).hex())
    if isinstance(obj, (bool, np.bool_, int, np.integer, str, type(None))):
        return repr(obj)
    if isinstance(obj, dict) and depth < 3:
        return {str(k): fingerprint(v, depth + 1) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)) and depth < 3 and len(obj) < 400:
        return [fingerprint(v, depth + 1) for v in obj]
    return f"<{type(obj).__name__}>"


def model_fp(m):
    skip = {"rng"}
    return {k: fingerprint(v) for k, v in sorted(vars(m).items()) if k not in skip and not k.startswith("__") and not callable(v)}


def child():
    logging.disable(logging.CRITICAL)
    sys.path.insert(0, str(ROOT))
    from tests.test_wp8_item0_metadata import _dual
    from tests.test_wp7r_item5_dual_honesty import _fit, _series
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conv = _dual(tol=1.0)
        out["converged_fixture"] = dict(model=model_fp(conv))
        notc = _dual(tol=1e-4)
        out["not_converged_fixture"] = dict(model=model_fp(notc))
        for seed in (1, 2):
            r = _fit(_series([20] * 5, seed=seed))
            out[f"optimizer_success_seed{seed}"] = dict(model=model_fp(r.model), optimizer_success=repr(r.optimizer_success), duration_optimizer_success=repr(r.duration_optimizer_success),
                                                         duration_fallback_used=repr(r.model.duration_fallback_used_))
    from threadpoolctl import threadpool_info
    out["blas"] = [dict(api=d["internal_api"], arch=d.get("architecture"), threads=d["num_threads"], lib=Path(d["filepath"]).name) for d in threadpool_info()]
    out["env"] = {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OPENBLAS_CORETYPE", "OMP_NUM_THREADS")}
    out["outcomes"] = dict(converged_state=str(conv.fit_state_), not_converged_state=str(notc.fit_state_), seed1_fallback=out["optimizer_success_seed1"]["duration_fallback_used"],
                           seed2_fallback=out["optimizer_success_seed2"]["duration_fallback_used"], converged_at_bound=repr(conv.duration_at_bound_), converged_degraded=repr(conv.fit_degraded_))
    print("PROBE_JSON " + json.dumps(out))


def run_one(kernel, threads):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    for k in ("OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        env.pop(k, None)
    if kernel != "default":
        env["OPENBLAS_CORETYPE"] = kernel
    if threads != "unset":
        env["OPENBLAS_NUM_THREADS"] = threads
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child"], env=env, capture_output=True, text=True, cwd=str(ROOT))
    line = next((l for l in r.stdout.splitlines() if l.startswith("PROBE_JSON ")), None)
    if line is None:
        raise RuntimeError(f"child failed for {kernel}/{threads}: {r.stderr[-400:]}")
    return json.loads(line[len("PROBE_JSON "):])


def flat(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flat(v, f"{prefix}{k}.")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from flat(v, f"{prefix}{i}.")
    else:
        yield prefix.rstrip("."), d


def compare(a, b):
    """Differences between two child outputs restricted to the fits (not the BLAS description, the environment or the `config` label): the list of (path, a, b) that differ."""
    fa, fb = dict(flat({k: v for k, v in a.items() if k not in ("blas", "env", "config")})), dict(flat({k: v for k, v in b.items() if k not in ("blas", "env", "config")}))
    return [(k, fa.get(k), fb.get(k)) for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]


def drive(out_path, kernels, threads, repeat):
    res = []
    for kernel in kernels:
        for th in threads:
            for rep in range(repeat):
                r = run_one(kernel, th)
                r["config"] = dict(kernel=kernel, threads=th, rep=rep)
                res.append(r)
                print(f"[{kernel:11s} threads={th:5s} rep {rep}] BLAS {[(b['api'], b['arch'], b['threads']) for b in r['blas']]} outcomes {r['outcomes']}", flush=True)
    ref = res[0]
    print(f"\ncomparison with the first configuration ({ref['config']}):")
    summary = []
    for r in res[1:]:
        diffs = compare(ref, r)
        fits = sorted({d[0].split(".")[0] for d in diffs})
        summary.append(dict(config=r["config"], n_differing_fields=len(diffs), fits_differing=fits, outcomes_differ=[d for d in diffs if d[0].startswith("outcomes.")]))
        print(f"  {r['config']}: {'BIT-IDENTICAL on every fit' if not diffs else str(len(diffs)) + ' fields differ in ' + str(fits)}" + (f"; discrete outcomes that differ: {[d for d in diffs if d[0].startswith('outcomes.')]}" if any(d[0].startswith("outcomes.") for d in diffs) else ""))
    json.dump(dict(runs=res, summary=summary), open(out_path, "w"), indent=1)


if __name__ == "__main__":
    if "--child" in sys.argv:
        child()
    elif "--drive" in sys.argv:
        out = sys.argv[sys.argv.index("--drive") + 1]
        kernels = (sys.argv[sys.argv.index("--kernels") + 1] if "--kernels" in sys.argv else "default,HASWELL,SKYLAKEX,SANDYBRIDGE,NEHALEM").split(",")
        threads = (sys.argv[sys.argv.index("--threads") + 1] if "--threads" in sys.argv else "1,2,unset").split(",")
        repeat = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else 1
        drive(out, kernels, threads, repeat)
    else:
        sys.exit(__doc__)
