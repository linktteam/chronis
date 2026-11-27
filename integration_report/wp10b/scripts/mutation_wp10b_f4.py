"""WP10B F4: mutation control of `f4_lib.py`, `f4_stream_child.py` and the data generator against the F4 tests (set declared in tests/PREDECLARED_CRITERIA_WP10B.md section 13.2(6); named test must fail by its own assertion;
control first; runner `mutation_runner10b.py`). usage: PYTHONPATH=. python3.12 integration_report/wp10b/scripts/mutation_wp10b_f4.py [out.txt]
Two limits, stated: (1) the stored reference arrays are binary, so "a corrupted reference accepted" is checked by hand (see tests/WP10B_F4_RESULTS.md), not here; (2) G14 (the probe vector mistyped; run 1 used a one-ulp change of one value, which does not change the rounded norm: an EQUIVALENT mutant, SURVIVED, replaced) is caught only on a machine whose
default kernel is SkylakeX-class (this one); on another machine the same test takes its other branch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutation_runner10b as R

LIB = "integration_report/wp10b/scripts/f4_lib.py"
CH = "integration_report/wp10b/scripts/f4_stream_child.py"
GEN = "data/synthetic_identity.py"
K = "tests/test_wp10b_f4_generator_kernels.py::"
W9 = "tests/test_wp9_a4_generator.py::test_default_streams_are_bit_identical_to_wp8"
MUT = [
    ("G01 the HASWELL forcing is not checked", LIB, 'if archs.get("HASWELL") != "Haswell" or archs.get("NEHALEM") != "Nehalem":', 'if archs.get("NEHALEM") != "Nehalem":', [K + "test_kernel_condition"]),
    ("G02 the NEHALEM forcing is not checked", LIB, 'if archs.get("HASWELL") != "Haswell" or archs.get("NEHALEM") != "Nehalem":', 'if archs.get("HASWELL") != "Haswell":', [K + "test_kernel_condition"]),
    ("G03 three distinct kernels accepted with two", LIB, "    if len(set(archs.values())) == 3:", "    if len(set(archs.values())) >= 2:", [K + "test_kernel_condition"]),
    ("G04 the three-distinct condition is never reported", LIB, '    if len(set(archs.values())) == 3:\n        return "three_distinct"\n', "", [K + "test_kernel_condition"]),
    ("G05 a non-x86 machine is treated as x86", LIB, "    if str(machine).lower() not in X86:", "    if False:", [K + "test_kernel_condition"]),
    ("G06 AMD64 is not recognised as x86", LIB, 'X86 = ("x86_64", "amd64", "x86", "i686", "i386")', 'X86 = ("x86_64", "x86", "i686", "i386")', [K + "test_kernel_condition"]),
    ("G07 the WP8 norm comparison is inverted", LIB, "    return (probe_norm_hex() if norm_hex is None else norm_hex) == WP8_PROBE_NORM_HEX", "    return (probe_norm_hex() if norm_hex is None else norm_hex) != WP8_PROBE_NORM_HEX", [K + "test_same_blas_norm_as_wp8_compares_exactly"]),
    ("G08 the tolerance is widened to 1e-12", LIB, "LAST_BIT_RTOL = 1e-15", "LAST_BIT_RTOL = 1e-12", [K + "test_within_last_bit"]),
    ("G09 default may deviate", LIB, 'NORM_DEPENDENT_KINDS = ("drift", "split")', 'NORM_DEPENDENT_KINDS = ("drift", "split", "default")', [K + "test_the_kinds_that_may_deviate_are_exactly_drift_and_split"]),
    ("G10 merge may deviate", LIB, 'NORM_DEPENDENT_KINDS = ("drift", "split")', 'NORM_DEPENDENT_KINDS = ("drift", "split", "merge")', [K + "test_the_kinds_that_may_deviate_are_exactly_drift_and_split"]),
    ("G11 NaN placement is ignored", LIB, "    if (nan_x != nan_r).any():\n        return False\n", "", [K + "test_within_last_bit"]),
    ("G12 the shape is ignored", LIB, "    if x.shape != ref.shape:\n        return False\n", "", [K + "test_within_last_bit"]),
    ("G13 the tolerance is exclusive", LIB, "<= rtol * np.maximum", "< rtol * np.maximum", [K + "test_within_last_bit"]),
    ("G14 the probe vector is mistyped (an exponent)", LIB, '"0x1.8abcd73f90ef3p-1"', '"0x1.8abcd73f90ef3p-2"', [K + "test_the_wp8_norm_is_the_one_of_the_skylakex_kernel_and_the_forced_avx2_kernels_differ_from_it"]),
    ("G15 the children do not force the kernel", LIB, '        env["OPENBLAS_CORETYPE"] = kernel', "        pass", [K + "test_the_forced_kernels_are_honoured_by_the_children"]),
    ("G16 the probe norm is computed exactly instead of with np.linalg.norm", LIB, "    return float(np.linalg.norm(probe_vector())).hex()", "    import math\n    return math.sqrt(math.fsum(float(t) * float(t) for t in probe_vector())).hex()",
     [K + "test_the_wp8_norm_is_the_one_of_the_skylakex_kernel_and_the_forced_avx2_kernels_differ_from_it"]),
    ("G17 the 'pure' norm of the child uses BLAS", CH, "    return math.sqrt(math.fsum(float(t) * float(t) for t in np.ravel(v)))", "    return float(np.sqrt(np.dot(np.ravel(v), np.ravel(v))))", [K + "test_the_blas_norm_is_the_only_cpu_dependent_function_of_the_generator"]),
    ("G18 the generator's drift stream moves by 1e-13 (WP8-kernel machine)", GEN, "            means_t[idx] += frac[idx] * drift_sd * wsd * vec\n", "            means_t[idx] += frac[idx] * drift_sd * wsd * vec * (1.0 + 1e-13)\n", [W9 + "[9996-drift]"]),
    ("G19 the generator's merge stream moves by 1e-13", GEN, "        means_t[later_b] = u[\"means\"][a]\n", "        means_t[later_b] = u[\"means\"][a] * (1.0 + 1e-13)\n", [W9 + "[9996-merge]"]),
    ("G20 the generator's drift stream moves by 1e-13 (any kernel)", GEN, "            means_t[idx] += frac[idx] * drift_sd * wsd * vec\n", "            means_t[idx] += frac[idx] * drift_sd * wsd * vec * (1.0 + 1e-13)\n", [K + "test_every_kernel_gives_streams_within_the_last_bit_of_the_stored_wp8_arrays"]),
    ("G21 the generator's default stream moves by 1e-13", GEN, "    X_full = means_t + wsd * e\n", "    X_full = (means_t + wsd * e) * (1.0 + 1e-13)\n", [K + "test_the_default_and_merge_streams_do_not_depend_on_the_kernel"]),
]

if __name__ == "__main__":
    sys.exit(R.run(MUT, sys.argv[1] if len(sys.argv) > 1 else None, workers=2))
