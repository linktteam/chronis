"""WP10B F4: which function of the data generator gives CPU-dependent numbers (opa's finding: `test_default_streams_are_bit_identical_to_wp8[9996-drift]` fails on an AMD EPYC 7763, hash d66af22924722a01 against 41355ac8122f9a17).
Parent: runs a child process per configuration (the default; numpy's AVX-512 dispatch off; each forced OpenBLAS kernel; both); `--child` is the child. Child: the hashes of the eight WP8/WP9 test streams (seeds 9996, 9997; default,
split, merge, drift) and a TRACE of every `np.linalg.norm` call made while the streams are generated (call site, input hash, output as hex), plus the BLAS architecture reported by threadpoolctl.
Parent output: the table of stream hashes per configuration, and for the drift stream of seed 9996 the first norm call whose output differs from the default configuration (call site and the two values).
usage: PYTHONPATH=. python3.12 integration_report/wp10b/scripts/f4_generator_probe.py [OUT.json]"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AVX512_ALL = "AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"
CONFIGS = {
    "default": {},
    "NO_AVX512_numpy": {"NPY_DISABLE_CPU_FEATURES": AVX512_ALL},
    "OPENBLAS_SKYLAKEX": {"OPENBLAS_CORETYPE": "SKYLAKEX"},
    "OPENBLAS_HASWELL": {"OPENBLAS_CORETYPE": "HASWELL"},
    "OPENBLAS_ZEN": {"OPENBLAS_CORETYPE": "ZEN"},
    "OPENBLAS_NEHALEM": {"OPENBLAS_CORETYPE": "NEHALEM"},
    "HASWELL_and_NO_AVX512_numpy": {"OPENBLAS_CORETYPE": "HASWELL", "NPY_DISABLE_CPU_FEATURES": AVX512_ALL},
}
STREAMS = [(s, k) for s in (9996, 9997) for k in ("default", "split", "merge", "drift")]
KW = {"default": {}, "split": dict(k_change="split"), "merge": dict(k_change="merge"), "drift": dict(drift_sd=1.0)}


def h16(a) -> str:
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def child():
    import numpy as np
    from threadpoolctl import threadpool_info
    from data.synthetic_identity import generate_identity_stream
    real = np.linalg.norm
    trace = []

    def traced(x, *a, **k):
        out = real(x, *a, **k)
        f = sys._getframe(1)
        trace.append(dict(site=f"{Path(f.f_code.co_filename).name}:{f.f_lineno} in {f.f_code.co_name}", n=int(np.size(x)), axis=k.get("axis", a[1] if len(a) > 1 else None), inp=h16(np.asarray(x)),
                          out=float(out).hex() if np.ndim(out) == 0 else h16(out)))
        return out
    np.linalg.norm = traced
    res = {}
    for seed, kind in STREAMS:
        trace.clear()
        X = generate_identity_stream(seed, **KW[kind]).X_full
        res[f"{seed}-{kind}"] = dict(hash=h16(X), trace=list(trace))
    np.linalg.norm = real
    print("PROBE_JSON " + json.dumps(dict(numpy=np.__version__, blas=[dict(api=d["internal_api"], arch=d.get("architecture"), threads=d["num_threads"]) for d in threadpool_info()], streams=res)), flush=True)


def run_child(cfg):
    env = {k: v for k, v in os.environ.items() if k not in ("OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES")}
    env.update(PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS="1")
    env.update(CONFIGS[cfg])
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child"], env=env, capture_output=True, text=True, cwd=str(ROOT))
    line = next((x for x in r.stdout.splitlines() if x.startswith("PROBE_JSON ")), None)
    if line is None:
        raise RuntimeError(f"child {cfg} failed: {r.stderr[-400:]}")
    return json.loads(line[len("PROBE_JSON "):])


def first_difference(base, other, key):
    """The first `np.linalg.norm` call (in order) of stream `key` whose output differs between two children, or None."""
    for i, (a, b) in enumerate(zip(base["streams"][key]["trace"], other["streams"][key]["trace"])):
        assert a["site"] == b["site"] and a["inp"] == b["inp"], "the call sequence or an input differs: the difference starts before the norm"
        if a["out"] != b["out"]:
            return i, a, b
    return None


def main():
    out = {cfg: run_child(cfg) for cfg in CONFIGS}
    ref = out["default"]
    print(f"numpy {ref['numpy']}; BLAS of the default configuration: {ref['blas']}")
    print("config".ljust(30) + "BLAS arch".ljust(12) + " ".join(f"{s}-{k}".ljust(18) for s, k in STREAMS))
    for cfg, r in out.items():
        print(cfg.ljust(30) + str([b["arch"] for b in r["blas"]]).ljust(12) + " ".join(r["streams"][f"{s}-{k}"]["hash"].ljust(18) for s, k in STREAMS))
    print("\nnorm calls per stream (default configuration): " + ", ".join(f"{k}: {len(v['trace'])}" for k, v in ref["streams"].items()))
    sites = sorted({t["site"] for v in ref["streams"].values() for t in v["trace"]})
    print("call sites of np.linalg.norm in the generator: " + "; ".join(sites))
    for cfg in ("NO_AVX512_numpy", "OPENBLAS_HASWELL", "OPENBLAS_ZEN", "OPENBLAS_NEHALEM"):
        for key in ("9996-drift", "9996-split", "9997-drift"):
            d = first_difference(ref, out[cfg], key)
            same = out[cfg]["streams"][key]["hash"] == ref["streams"][key]["hash"]
            print(f"{cfg:20s} {key:11s} stream hash {'identical' if same else 'DIFFERENT'}; first norm output that differs: " + ("none" if d is None else f"call {d[0]} at {d[1]['site']}: default {d[1]['out']} ({float.fromhex(d[1]['out']):.17g}) vs {d[2]['out']} ({float.fromhex(d[2]['out']):.17g})"))
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        json.dump(out, open(sys.argv[1], "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    child() if "--child" in sys.argv else main()
