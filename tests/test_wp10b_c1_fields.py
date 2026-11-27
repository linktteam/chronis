"""WP10B C1(1): the field-kind analysis of the cross-kernel probe (`integration_report/wp10b/scripts/c1_optimizer_fields.py`). Synthetic probe records (the format of `wp10a/scripts/a2_probe.py`), no stored file is read:
a differing float in the first values of an array is reported as a differing MODEL ATTRIBUTE and not as a flag or a discrete outcome; a flipped flag or a differing counter would be reported as such."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("wp10b_c1_fields_under_test", ROOT / "integration_report" / "wp10b" / "scripts" / "c1_optimizer_fields.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def arr(sha, head):
    return dict(sha=sha, shape=[2, 2], dtype="float64", head=[float(x).hex() for x in head])


def probe(kernel, *, flag="True", it="'18'", head=(1.0, 2.0), sha="aa", label="optimizer_success_seed1"):
    return {label: dict(model=dict(A_fast=arr(sha, head), _em_iteration=it, K="'2'", log_likelihood_=dict(f=float(-10.5).hex())), optimizer_success=flag, duration_optimizer_success=flag, duration_fallback_used="False"),
            "config": dict(kernel=kernel, threads="1", rep=0)}


def result(runs):
    return C.analyse(runs)[0]["labels"]["optimizer_success_seed1"]


def test_a_float_difference_is_a_model_attribute_not_a_flag_and_not_a_discrete_outcome():
    r = result([probe("default"), probe("HASWELL", head=(1.0, 2.0 + 1e-12), sha="bb")])
    assert r["differing_model_attributes"] == ["A_fast"]
    assert r["flag_values"]["optimizer_success"] == ["True"]
    assert r["discrete_subfields_that_differ"] == []
    assert r["differing_by_kind"].get("float", 0) >= 1 and "flag" not in r["differing_by_kind"]
    assert 4e-13 < r["max_rel_diff_of_differing_floats"]["A_fast"] < 6e-13                # relative: 1e-12 / 2.0, not the absolute 1e-12


def test_identical_probes_have_no_difference_at_all():
    r = result([probe("default"), probe("HASWELL")])
    assert r["differing_model_attributes"] == [] and r["differing_by_kind"] == {}


def test_a_flipped_flag_is_reported_as_a_flag():
    r = result([probe("default"), probe("HASWELL", flag="False")])
    assert r["flag_values"]["optimizer_success"] == ["False", "True"]
    assert r["differing_by_kind"].get("flag") == 2


def test_a_differing_iteration_counter_is_reported_as_a_discrete_outcome():
    r = result([probe("default"), probe("HASWELL", it="'19'")])
    assert r["discrete_subfields_that_differ"] == ["model._em_iteration"]


def test_the_two_labels_are_analysed_separately():
    runs = [probe("default"), probe("HASWELL")]
    for p in runs:
        p["optimizer_success_seed2"] = p["optimizer_success_seed1"]
    both = C.analyse(runs)[0]["labels"]
    assert set(both) == {"optimizer_success_seed1", "optimizer_success_seed2"}
