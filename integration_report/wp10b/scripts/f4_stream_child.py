"""WP10B F4 child process: the eight WP8 test streams under the OpenBLAS kernel this process was started with, generated twice: with the real `np.linalg.norm` (a BLAS `ddot`) and with a norm that does not use BLAS
(`sqrt(fsum(x * x))`, exactly rounded, so it cannot depend on the CPU). Prints one `CHILD_JSON` line: the BLAS architecture, the probe norm, the stream hashes of both variants and the largest relative difference between them."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f4_lib as F                                                                           # noqa: E402
from data.synthetic_identity import generate_identity_stream                                 # noqa: E402

KW = {"default": {}, "split": dict(k_change="split"), "merge": dict(k_change="merge"), "drift": dict(drift_sd=1.0)}
STREAMS = [(s, k) for s in (9996, 9997) for k in KW]


def h16(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def pure_norm(v, *a, **k):
    return math.sqrt(math.fsum(float(t) * float(t) for t in np.ravel(v)))


def main():
    from threadpoolctl import threadpool_info
    arch = [d.get("architecture") for d in threadpool_info() if d["internal_api"] == "openblas"]
    real_norm = np.linalg.norm
    real = {f"{s}-{k}": generate_identity_stream(s, **KW[k]).X_full for s, k in STREAMS}
    np.linalg.norm = pure_norm
    try:
        pure = {f"{s}-{k}": generate_identity_stream(s, **KW[k]).X_full for s, k in STREAMS}
    finally:
        np.linalg.norm = real_norm
    worst = {key: float(np.max(np.abs(real[key] - pure[key]) / np.maximum(np.abs(real[key]), 1e-300))) for key in real}
    ref = np.load(Path(__file__).resolve().parents[3] / "tests" / "data" / "wp8_identity_streams.npz")
    to_wp8 = {key: float(np.max(np.abs(real[key] - ref[key]) / np.maximum(np.abs(ref[key]), 1e-300))) for key in real}
    print("CHILD_JSON " + json.dumps(dict(arch=arch[0] if arch else None, probe_norm=F.probe_norm_hex(), real={k: h16(v) for k, v in real.items()}, pure={k: h16(v) for k, v in pure.items()}, rel_diff=worst, rel_to_wp8=to_wp8)), flush=True)


if __name__ == "__main__":
    main()
