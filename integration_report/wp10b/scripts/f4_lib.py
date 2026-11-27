"""WP10B F4 helpers (pure functions; criteria section 13): the kernel condition of a machine, the WP8 BLAS-norm probe, and the last-bit comparison of two streams."""
from __future__ import annotations

import numpy as np

# The seven values of the first `_unit` vector of the seed-9996 drift stream (the second `np.linalg.norm` call of `generate_identity_stream(9996, drift_sd=1.0)`), in hex. Its norm is 0x1.621c576b7e1f2p+0 with the
# SkylakeX OpenBLAS kernel (the kernel that reproduces the WP8 stream hashes) and 0x1.621c576b7e1f3p+0 with the Haswell and Nehalem kernels.
PROBE_VECTOR_HEX = ("-0x1.87c82a98adf22p-4", "0x1.a0eef00e24542p-4", "-0x1.3268e66fd9f50p-2", "0x1.8abcd73f90ef3p-1", "-0x1.2e1c2dbae5f9dp-2", "0x1.d4df545ab3ce0p-6", "-0x1.0f2c8798f3840p+0")
WP8_PROBE_NORM_HEX = "0x1.621c576b7e1f2p+0"
NORM_DEPENDENT_KINDS = ("drift", "split")             # the stream kinds whose arrays use a vector made with `np.linalg.norm`; `default` and `merge` never do
LAST_BIT_RTOL = 1e-15
X86 = ("x86_64", "amd64", "x86", "i686", "i386")


def probe_vector() -> np.ndarray:
    return np.array([float.fromhex(h) for h in PROBE_VECTOR_HEX])


def probe_norm_hex() -> str:
    """The norm of the probe vector on THIS machine, computed the way the generator computes it (`np.linalg.norm`, which is `sqrt(ddot)` and so depends on the OpenBLAS kernel)."""
    return float(np.linalg.norm(probe_vector())).hex()


def same_blas_norm_as_wp8(norm_hex: str | None = None) -> bool:
    """True when this machine's BLAS norm of the probe is exactly the value of the machine that produced the WP8 hashes: then the streams must reproduce the WP8 hashes exactly."""
    return (probe_norm_hex() if norm_hex is None else norm_hex) == WP8_PROBE_NORM_HEX


def kernel_condition(archs: dict, machine: str) -> str:
    """The condition of a machine for the cross-kernel test. `archs` = the OpenBLAS architecture reported by a fresh process for the kernels "default", "HASWELL" and "NEHALEM" (the last two forced with
    OPENBLAS_CORETYPE); `machine` = `platform.machine()`. Returns
      "not_x86"                    the kernel names are x86 names: the test does not apply (the only skip),
      "three_distinct"             three different kernels,
      "default_is_a_forced_kernel" fewer than three: the default kernel IS the Haswell or the Nehalem one (for example an AMD EPYC 7763, where OpenBLAS runs its Haswell kernel); two distinct kernels remain,
      "forcing_not_honoured"       a forced kernel was not applied (HASWELL did not give Haswell, NEHALEM did not give Nehalem, or the two are the same): a failure, never a skip."""
    if str(machine).lower() not in X86:
        return "not_x86"
    if archs.get("HASWELL") != "Haswell" or archs.get("NEHALEM") != "Nehalem":
        return "forcing_not_honoured"
    if len(set(archs.values())) == 3:
        return "three_distinct"
    return "default_is_a_forced_kernel"                  # the two forced kernels are honoured and differ, so with fewer than three distinct the default one is one of them


def within_last_bit(x, ref, rtol: float = LAST_BIT_RTOL) -> bool:
    """Element-wise relative difference <= rtol, NaN at the same places (a NaN against a number is a difference)."""
    x, ref = np.asarray(x, dtype=float), np.asarray(ref, dtype=float)
    if x.shape != ref.shape:
        return False
    nan_x, nan_r = np.isnan(x), np.isnan(ref)
    if (nan_x != nan_r).any():
        return False
    ok = ~nan_x
    return bool((np.abs(x[ok] - ref[ok]) <= rtol * np.maximum(np.abs(x[ok]), np.abs(ref[ok]))).all())


def run_child(kernel: str, script, root) -> dict:
    """Run `f4_stream_child.py` in a fresh process under the OpenBLAS kernel `kernel` ("default" = none forced); returns its CHILD_JSON dict. A child that fails is an AssertionError with its stderr."""
    import json
    import os
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if k not in ("OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}
    env.update(PYTHONPATH=str(root), OPENBLAS_NUM_THREADS="1")
    if kernel != "default":
        env["OPENBLAS_CORETYPE"] = kernel
    r = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, cwd=str(root))
    line = next((x for x in r.stdout.splitlines() if x.startswith("CHILD_JSON ")), None)
    assert line is not None, f"child failed for {kernel}: {r.stderr[-400:]}"
    return json.loads(line[len("CHILD_JSON "):])
