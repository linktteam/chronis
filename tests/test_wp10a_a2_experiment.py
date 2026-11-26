"""WP10A A2: the pure parts of the runner experiment (`a2_runner_experiment.py`) and of the probe (`a2_probe.py`) on fabricated child outputs. The experiment itself needs the runner's CPU."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "integration_report" / "wp10a" / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PROBE = _load("a2_probe", SCRIPTS / "a2_probe.py")
EXP = _load("wp10a_a2_runner_experiment", SCRIPTS / "a2_runner_experiment.py")

OUTCOMES = dict(converged_state="FitState.CONVERGED", not_converged_state="FitState.MAX_ITER_REACHED", seed1_fallback="False", seed2_fallback="True", converged_at_bound="{0: False, 1: False}", converged_degraded="False")


def child(kernel="default", threads="1", bits=0.0, outcomes=None):
    return dict(converged_fixture=dict(model=dict(mu=PROBE.fingerprint(np.array([1.0 + bits, 2.0])), ll=PROBE.fingerprint(-3.5 + bits))), outcomes=dict(OUTCOMES, **(outcomes or {})),
                blas=[dict(api="openblas", arch="SkylakeX", threads=1, lib="x.so")], env={}, config=dict(kernel=kernel, threads=threads, rep=0))


def test_fingerprint_is_bitwise_and_hex_for_floats():
    a, b = PROBE.fingerprint(np.array([1.0, 2.0])), PROBE.fingerprint(np.array([1.0, np.nextafter(2.0, 3.0)]))
    assert a["sha"] != b["sha"] and a["head"][0] == b["head"][0] and a["head"][1] != b["head"][1]
    assert PROBE.fingerprint(1.0) == dict(f=float(1.0).hex())


def test_compare_ignores_the_config_label_and_the_blas_description():
    a, b = child("default", "1"), child("HASWELL", "2")
    b["blas"][0]["arch"] = "Haswell"
    assert PROBE.compare(a, b) == []
    assert PROBE.compare(a, child(bits=1e-12))                                           # a bit-level difference is seen


def test_a_discrete_flip_is_the_reproduced_mechanism():
    ref = child()
    probe = [child("default", "1"), child("ZEN", "1", bits=1e-9, outcomes=dict(seed2_fallback="False"))]
    res = EXP.summarise(probe, ref, [])
    assert res["verdict"].startswith("MECHANISM REPRODUCED")
    assert [f["config"] for f in res["discrete_flips"]] == ["ZEN/threads=1"] and res["bit_differences"] == ["ZEN/threads=1"]
    assert res["configs"][1]["discrete"]["seed2_fallback"] == "False" and res["configs"][0]["discrete_differs_from_reference"] == []


def test_bit_differences_without_a_flip_do_not_reproduce_it():
    res = EXP.summarise([child("default", "1"), child("HASWELL", "1", bits=1e-9)], child(), [])
    assert res["verdict"].startswith("NOT REPRODUCED") and res["discrete_flips"] == [] and res["bit_differences"] == ["HASWELL/threads=1"]


def test_a_failure_in_the_repeat_without_a_flip_is_reported_as_another_cause():
    rep = [dict(run=0, hashseed=0, returncode=0, failed=[], summary="21 passed", tb=[]), dict(run=1, hashseed=1, returncode=1, failed=["FAILED tests/x.py::t - assert"], summary="1 failed, 20 passed", tb=["x.py:3: assert"])]
    res = EXP.summarise([child()], child(), rep)
    assert res["verdict"].startswith("FAILURE REPRODUCED IN THE REPEAT") and res["repeat"]["runs"] == 2 and [f["run"] for f in res["repeat"]["failures"]] == [1]


def test_without_a_reference_only_the_first_configuration_is_the_baseline():
    res = EXP.summarise([child("default", "1"), child("default", "2")], None, [])
    assert res["configs"][0]["fields_differing_from_reference"] is None and res["verdict"].startswith("NOT REPRODUCED")


def test_the_environment_capture_has_the_fields_the_reading_needs():
    env = EXP.environment()
    assert {"python", "numpy", "scipy", "blas", "cpu", "cpu_count", "openblas_env"} <= set(env)
    assert all({"api", "arch", "threads"} <= set(b) for b in env["blas"])
