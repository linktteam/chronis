"""WP10B F4 (opa's package P02A; criteria section 13): which function of the data generator gives CPU-dependent numbers, and the helpers that make the WP8-identity and the cross-kernel tests honest on a machine with another OpenBLAS kernel.
Diagnosis (RUN, wp10b_evidence/f4_generator_probe.txt): `np.linalg.norm` of a 1-D float64 vector is `sqrt(ddot)`; the OpenBLAS kernel changes its last bit. The tests below run the generator in FRESH processes under the default,
the HASWELL and the NEHALEM kernel."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "integration_report" / "wp10b" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import f4_lib as F  # noqa: E402
from tests.test_wp9_a4_generator import HASHES  # noqa: E402

KERNELS = ("default", "HASWELL", "NEHALEM")
SKYLAKE_CLASS = ("SkylakeX",)


@pytest.fixture(scope="module")
def children():
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as ex:
        return dict(zip(KERNELS, ex.map(lambda k: F.run_child(k, SCRIPTS / "f4_stream_child.py", ROOT), KERNELS)))


def sha16(a):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------- the diagnosis, in fresh processes
def test_the_blas_norm_is_the_only_cpu_dependent_function_of_the_generator(children):
    ref = children["default"]["pure"]
    for k in KERNELS[1:]:
        assert children[k]["pure"] == ref, (k, {s: (children[k]["pure"][s], ref[s]) for s in ref if children[k]["pure"][s] != ref[s]})
    assert set(ref) == {f"{s}-{kind}" for (s, kind) in HASHES}


def test_the_effect_of_the_blas_norm_is_a_last_bit_effect(children):
    for k in KERNELS:
        assert max(children[k]["rel_diff"].values()) <= F.LAST_BIT_RTOL, (k, children[k]["rel_diff"])
    assert children["default"]["real"].keys() == children["default"]["pure"].keys()


def test_the_forced_kernels_are_honoured_by_the_children(children):
    for kernel, arch in (("HASWELL", "Haswell"), ("NEHALEM", "Nehalem")):
        if F.kernel_condition({k: children[k]["arch"] for k in KERNELS}, __import__("platform").machine()) == "not_x86":
            pytest.skip("not an x86-64 machine")
        assert children[kernel]["arch"] == arch, (kernel, children[kernel]["arch"])


def test_every_kernel_gives_streams_within_the_last_bit_of_the_stored_wp8_arrays(children):
    for k in KERNELS:
        assert max(children[k]["rel_to_wp8"].values()) <= F.LAST_BIT_RTOL, (k, children[k]["rel_to_wp8"])
        for key in ("9996-default", "9997-default", "9996-merge", "9997-merge"):
            assert children[k]["rel_to_wp8"][key] == 0.0, (k, key)


def test_the_default_and_merge_streams_do_not_depend_on_the_kernel(children):
    for key in ("9996-default", "9997-default", "9996-merge", "9997-merge"):
        assert {children[k]["real"][key] for k in KERNELS} == {HASHES[(int(key.split("-")[0]), key.split("-")[1])]}, key


def test_the_wp8_norm_is_the_one_of_the_skylakex_kernel_and_the_forced_avx2_kernels_differ_from_it(children):
    for k in KERNELS:
        arch = children[k]["arch"]
        if arch in SKYLAKE_CLASS:
            assert children[k]["probe_norm"] == F.WP8_PROBE_NORM_HEX, (k, arch)                      # the kernel that reproduced the WP8 hashes
        elif arch in ("Haswell", "Nehalem"):
            assert children[k]["probe_norm"] != F.WP8_PROBE_NORM_HEX, (k, arch)                      # the AVX2 / SSE kernels round the probe's norm differently
    if children["default"]["arch"] in SKYLAKE_CLASS:
        assert F.same_blas_norm_as_wp8()                                                              # (this process uses the default kernel)


def test_a_stream_made_with_the_wp8_kernel_reproduces_the_wp8_hash_exactly_and_the_others_are_within_the_last_bit(children):
    for k in KERNELS:
        for key, h in children[k]["real"].items():
            seed, kind = int(key.split("-")[0]), key.split("-")[1]
            if children[k]["probe_norm"] == F.WP8_PROBE_NORM_HEX:
                assert h == HASHES[(seed, kind)], (k, key)
            else:
                assert kind in F.NORM_DEPENDENT_KINDS or h == HASHES[(seed, kind)], (k, key)


def test_the_stored_reference_streams_are_the_wp8_streams():
    ref = np.load(ROOT / "tests" / "data" / "wp8_identity_streams.npz")
    assert set(ref.files) == {f"{s}-{kind}" for (s, kind) in HASHES}
    for (s, kind), h in HASHES.items():
        assert sha16(ref[f"{s}-{kind}"]) == h, (s, kind)


# ---------------------------------------------------------------------------------------------- the helpers (pure)
def test_the_probe_vector_has_seven_values_and_its_norm_is_close_to_the_exact_one():
    v = F.probe_vector()
    assert v.shape == (7,)
    exact = math.sqrt(math.fsum(float(t) * float(t) for t in v))
    assert abs(float.fromhex(F.probe_norm_hex()) - exact) <= 4 * math.ulp(exact)
    assert abs(float.fromhex(F.WP8_PROBE_NORM_HEX) - exact) <= 4 * math.ulp(exact)                    # the WP8 value is within a few ulp of the exact norm too (which of the two kernels is the exact one is not claimed)


def test_same_blas_norm_as_wp8_compares_exactly():
    assert F.same_blas_norm_as_wp8(F.WP8_PROBE_NORM_HEX) is True
    assert F.same_blas_norm_as_wp8("0x1.621c576b7e1f3p+0") is False
    assert F.same_blas_norm_as_wp8("") is False


@pytest.mark.parametrize("archs,machine,expected", [
    (dict(default="SkylakeX", HASWELL="Haswell", NEHALEM="Nehalem"), "x86_64", "three_distinct"),
    (dict(default="Haswell", HASWELL="Haswell", NEHALEM="Nehalem"), "x86_64", "default_is_a_forced_kernel"),               # an AMD EPYC 7763 (INFERENCE)
    (dict(default="Nehalem", HASWELL="Haswell", NEHALEM="Nehalem"), "AMD64", "default_is_a_forced_kernel"),
    (dict(default="SkylakeX", HASWELL="SkylakeX", NEHALEM="Nehalem"), "x86_64", "forcing_not_honoured"),                  # HASWELL was not applied
    (dict(default="SkylakeX", HASWELL="Haswell", NEHALEM="SkylakeX"), "x86_64", "forcing_not_honoured"),                  # NEHALEM was not applied
    (dict(default="Haswell", HASWELL="Haswell", NEHALEM="Haswell"), "x86_64", "forcing_not_honoured"),                    # both forced kernels collapsed to the default
    (dict(default="Zen", HASWELL="Haswell", NEHALEM="Nehalem"), "x86_64", "three_distinct"),                              # an arch name of another kernel is a third kernel
    (dict(default="Zen", HASWELL="Haswell", NEHALEM="Nehalem"), "arm64", "not_x86"),
    (dict(default=None, HASWELL=None, NEHALEM=None), "aarch64", "not_x86"),
    (dict(default=None, HASWELL=None, NEHALEM=None), "x86_64", "forcing_not_honoured"),
])
def test_kernel_condition(archs, machine, expected):
    assert F.kernel_condition(archs, machine) == expected


def _within(x, ref):
    try:
        return F.within_last_bit(x, ref)
    except Exception as exc:                                                                          # noqa: BLE001  a comparison that raises is not an answer
        return f"raised {type(exc).__name__}"


def test_within_last_bit():
    x = np.array([1.0, 2.0, np.nan, 1e4])
    assert F.within_last_bit(x, x.copy())
    one_ulp = np.array([np.nextafter(1.0, 2.0), 2.0, np.nan, 1e4])
    assert F.within_last_bit(one_ulp, x)                                                              # 1 ulp is 2.2e-16 relative
    assert not F.within_last_bit(np.array([1.0 + 2e-15, 2.0, np.nan, 1e4]), x)                        # 2e-15 is above the tolerance
    assert not F.within_last_bit(np.array([1.0 + 1e-13, 2.0, np.nan, 1e4]), x)
    assert not F.within_last_bit(np.array([1.0, 2.0, 3.0, 1e4]), x)                                   # a number where the reference has NaN
    assert not F.within_last_bit(np.array([1.0, 2.0, np.nan, np.nan]), x)                             # NaN where the reference has a number
    assert _within(x[:3], x) is False                                                                 # shape: a clear False, not an exception
    assert F.within_last_bit(np.array([0.0, 0.0]), np.array([0.0, 0.0]))                             # exact zeros
    assert not F.within_last_bit(np.array([1e-300, 0.0]), np.array([0.0, 0.0]))


def test_the_kinds_that_may_deviate_are_exactly_drift_and_split():
    assert tuple(sorted(F.NORM_DEPENDENT_KINDS)) == ("drift", "split")
    assert "default" not in F.NORM_DEPENDENT_KINDS and "merge" not in F.NORM_DEPENDENT_KINDS and F.LAST_BIT_RTOL == 1e-15
