"""WP10A A2 (finding F2): the cross-kernel test and the documented tolerance. The fixtures of tests/test_wp8_item0_metadata.py (the two dual fixtures and the two public-path fits of the optimiser-success test) are
built in FRESH processes under three different OpenBLAS kernels (the default one, HASWELL, NEHALEM: three distinct kernels of the bundled OpenBLAS; `a2_probe.py --child` is the child). Claims, all measured in
wp10a_evidence/a2_probe_allkernels_summary.txt on the development container (5 distinct kernels, threads 1):
  1. the DISCRETE outcomes the tests depend on are identical across kernels (fit states, duration-optimiser fallback of seeds 1 and 2, boundary flags);
  2. the stored numbers of the three HEALTHY fits (converged fixture and the two public-path fits) agree within a relative 1e-8 (observed maximum 1.5e-11);
  3. the stored numbers of the NOT-CONVERGED fixture (40 iterations, tol 1e-4: an unhealthy fit that the tests never compare numerically) agree within a relative 5e-2 (observed maximum 5.2e-3).
Bit identity across kernels is NOT claimed: it does not hold (300 of 655 stored fields differ)."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "integration_report" / "wp10a" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from a2_probe_summary import fitfields, floats  # noqa: E402

sys.path.insert(0, str(ROOT / "integration_report" / "wp10b" / "scripts"))
from f4_lib import kernel_condition  # noqa: E402

KERNELS = ("default", "HASWELL", "NEHALEM")
HEALTHY = ("converged_fixture", "optimizer_success_seed1", "optimizer_success_seed2")
REL_HEALTHY, REL_UNHEALTHY = 1e-8, 5e-2


def _child(kernel):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    for k in ("OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        env.pop(k, None)
    env["OPENBLAS_NUM_THREADS"] = "1"
    if kernel != "default":
        env["OPENBLAS_CORETYPE"] = kernel
    r = subprocess.run([sys.executable, str(SCRIPTS / "a2_probe.py"), "--child"], env=env, capture_output=True, text=True, cwd=str(ROOT))
    line = next((x for x in r.stdout.splitlines() if x.startswith("PROBE_JSON ")), None)
    assert line is not None, f"child failed for {kernel}: {r.stderr[-400:]}"
    return json.loads(line[len("PROBE_JSON "):])


@pytest.fixture(scope="module")
def children():
    with ThreadPoolExecutor(max_workers=2) as ex:
        return dict(zip(KERNELS, ex.map(_child, KERNELS)))


def test_the_forced_kernels_are_really_different_ones_and_a_machine_with_fewer_than_three_is_named(children, record_property):
    """WP10B F4 (criteria section 13). This used to SKIP whenever the machine had fewer than three distinct kernels (an AMD EPYC 7763 runs OpenBLAS's Haswell kernel by default, so default and forced HASWELL are the same
    kernel: two distinct kernels, not three). A skip hid that the property can be tested there. The property: the two FORCED kernels are honoured and are two different kernels (so the HASWELL-against-NEHALEM comparison
    below compares two really different kernels on every x86-64 machine), and the machine is in one of the two documented conditions `three_distinct` or `default_is_a_forced_kernel`. A forced kernel that is not honoured
    FAILS. The only skip left is a machine that is not x86-64 (the kernel names are x86 names). The condition is recorded as a test property."""
    archs = {k: children[k]["blas"][0]["arch"] for k in KERNELS}
    condition = kernel_condition(archs, platform.machine())
    record_property("kernel_condition", condition)
    record_property("kernels", json.dumps(archs, sort_keys=True))
    if condition == "not_x86":
        pytest.skip(f"not an x86-64 machine ({platform.machine()}): OPENBLAS_CORETYPE=HASWELL / NEHALEM name x86 kernels; reported: {archs}")
    assert condition in ("three_distinct", "default_is_a_forced_kernel"), f"a forced OpenBLAS kernel was not honoured: {archs}"
    assert archs["HASWELL"] == "Haswell" and archs["NEHALEM"] == "Nehalem" and archs["HASWELL"] != archs["NEHALEM"]


@pytest.mark.parametrize("pair", [("HASWELL", "NEHALEM")])
def test_two_forced_kernels_agree_with_each_other_within_the_documented_tolerances(children, pair):
    """WP10B F4: the cross-kernel claims (healthy fits within a relative 1e-8, the not-converged fixture within 5e-2, the discrete outcomes identical) tested between the two forced kernels, which are two distinct kernels
    on every x86-64 machine (also when the default kernel is one of them, so that default-against-forced is a comparison of a kernel with itself)."""
    a, b = pair
    if kernel_condition({k: children[k]["blas"][0]["arch"] for k in KERNELS}, platform.machine()) == "not_x86":
        pytest.skip("not an x86-64 machine: the forced kernels are not x86 kernels")
    assert children[a]["outcomes"] == children[b]["outcomes"], (children[a]["outcomes"], children[b]["outcomes"])
    worst, n = _max_rel(children[a], children[b], HEALTHY)
    assert n > 100 and worst <= REL_HEALTHY, f"{pair}: relative difference {worst:.3g} > {REL_HEALTHY} (n={n})"
    worst, n = _max_rel(children[a], children[b], ("not_converged_fixture",))
    assert n > 50 and worst <= REL_UNHEALTHY, f"{pair}: relative difference {worst:.3g} > {REL_UNHEALTHY} (n={n})"


def test_the_discrete_outcomes_are_identical_across_kernels(children):
    ref = children["default"]["outcomes"]
    for k in KERNELS[1:]:
        assert children[k]["outcomes"] == ref, (k, children[k]["outcomes"], ref)
    assert ref["converged_state"] == "FitState.CONVERGED" and ref["not_converged_state"] == "FitState.MAX_ITER_REACHED"
    assert ref["seed1_fallback"] in ("True", "False") and ref["seed2_fallback"] in ("True", "False")


def _max_rel(a, b, prefixes):
    fa, fb = floats(fitfields(a)), floats(fitfields(b))
    worst = 0.0
    n = 0
    for key in fa:
        if key.split(".")[0] in prefixes and key in fb:
            n += 1
            worst = max(worst, abs(fa[key] - fb[key]) / max(abs(fa[key]), abs(fb[key]), 1e-300))
    return worst, n


@pytest.mark.parametrize("kernel", KERNELS[1:])
def test_healthy_fits_agree_across_kernels_within_the_documented_tolerance(children, kernel):
    worst, n = _max_rel(children["default"], children[kernel], HEALTHY)
    assert n > 100, "too few stored numbers were compared"
    assert worst <= REL_HEALTHY, f"{kernel}: relative difference {worst:.3g} > {REL_HEALTHY}"


@pytest.mark.parametrize("kernel", KERNELS[1:])
def test_the_not_converged_fixture_agrees_within_its_wider_tolerance(children, kernel):
    worst, n = _max_rel(children["default"], children[kernel], ("not_converged_fixture",))
    assert n > 50 and worst <= REL_UNHEALTHY, f"{kernel}: relative difference {worst:.3g} > {REL_UNHEALTHY}"


def test_the_tolerance_is_not_vacuous_a_planted_difference_would_be_seen():
    a = {"converged_fixture": {"model": {"x": {"f": (1.0).hex()}}}}
    b = {"converged_fixture": {"model": {"x": {"f": (1.0 + 1e-6).hex()}}}}
    fa, fb = floats(fitfields(a)), floats(fitfields(b))
    rel = max(abs(fa[k] - fb[k]) / max(abs(fa[k]), abs(fb[k])) for k in fa)
    assert rel > REL_HEALTHY
