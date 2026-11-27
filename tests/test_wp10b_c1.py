"""WP10B C1(2): the rules of the mild harmful-prior control (`c1_mild_run.py` construction, `c1_mild_analyze.py` rules; tests/PREDECLARED_CRITERIA_WP10B.md section 4 and Part 1b). Synthetic row sets; no fit."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "integration_report" / "wp10b" / "scripts"
sys.path.insert(0, str(SC))
sys.path.insert(0, str(ROOT))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, SC / fname)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


A = _load("c1_mild_analyze", "c1_mild_analyze.py")        # the runner imports it under this name (extension rule)
R = _load("c1_mild_run", "c1_mild_run.py")

PRIOR = {"d_obs": [[-0.5, -0.2, -0.4, -0.1], [0.2, 0.6, 0.0, 0.2]], "C_obs": [[[1.0, 0.1]] * 4, [[2.0, 0.2]] * 4], "R_obs": [[0.05, 0.05, 0.06, 0.06], [0.07, 0.07, 0.08, 0.08]],
         "A_fast": [[[0.7, 0], [0, 0.6]], [[0.5, 0], [0, 0.4]]], "dur_mu": [2.6, 2.7], "pi": [1.0, 0.0], "K": 2}


# ---- construction -------------------------------------------------------------------------------------------------------------------------------------------------------------
def test_m1_and_m2_shift_every_intercept_and_nothing_else():
    for c, s in (("M1", 8.0), ("M2", 12.0)):
        out = R.mild_dict(PRIOR, c)
        assert np.allclose(np.array(out["d_obs"]) - np.array(PRIOR["d_obs"]), s)
        for k in ("C_obs", "R_obs", "A_fast", "dur_mu", "pi"):
            assert np.array_equal(np.array(out[k]), np.array(PRIOR[k]))


def test_m3_exchanges_the_observation_model_of_the_two_regimes_and_keeps_the_dynamics():
    out = R.mild_dict(PRIOR, "M3")
    for k in ("d_obs", "C_obs", "R_obs"):
        assert np.array_equal(np.array(out[k]), np.array(PRIOR[k])[::-1])
    for k in ("A_fast", "dur_mu", "pi"):
        assert np.array_equal(np.array(out[k]), np.array(PRIOR[k]))


def test_the_true_prior_dict_is_not_modified():
    import copy
    before = copy.deepcopy(PRIOR)
    for c in R.CANDIDATE_ORDER:
        R.mild_dict(PRIOR, c)
    assert PRIOR == before


def test_every_candidate_is_milder_than_h3_by_the_declared_norm():
    h3 = R.h3_norm(PRIOR)
    assert math.isclose(h3, 16.0 * math.sqrt(8))
    n = {c: R.dobs_norm(PRIOR, c) for c in R.CANDIDATE_ORDER}
    assert math.isclose(n["M1"], 8.0 * math.sqrt(8)) and math.isclose(n["M2"], 12.0 * math.sqrt(8))
    assert all(v < h3 for v in n.values())
    assert n["M3"] < n["M1"] < n["M2"]


def test_an_unknown_candidate_is_an_error():
    with pytest.raises(ValueError):
        R.mild_dict(PRIOR, "M9")


# ---- validity (b), (c) ---------------------------------------------------------------------------------------------------------------------------------------------------------
def row(seed, length, arm, mean, eligible=False, cert=None, finite=True, published=False):
    return dict(seed=seed, length=length, arm=arm, mean=mean, finite=finite, eligible=eligible, certification={"state": cert} if cert else {}, published_after=published)


def fits_and_gauss(n_worse, n_worse_elig, n=36):
    """n fits scored against a Gaussian of -1.0: the first n_worse are below it (-2.0), of which the first n_worse_elig are eligible."""
    rows, gauss = [], {}
    for i in range(n):
        seed, length = 11001 + i // 3, (30, 60, 120)[i % 3]
        gauss[(seed, length)] = -1.0
        worse = i < n_worse
        rows.append(row(seed, length, "M1", -2.0 if worse else -0.5, eligible=i < n_worse_elig))
    return rows, gauss


@pytest.mark.parametrize("nw,ne,ok", [(6, 4, True), (5, 4, False), (6, 3, False), (36, 36, True), (0, 0, False)])
def test_validity_needs_six_below_the_gaussian_and_four_of_them_eligible(nw, ne, ok):
    rows, g = fits_and_gauss(nw, ne)
    v = A.validity(rows, g)
    assert v["verified"] is ok
    assert v["worse_than_gauss"] == nw and v["worse_and_eligible"] == ne


def test_a_fit_equal_to_the_gaussian_is_not_worse_and_a_nonfinite_fit_is():
    g = {(1, 30): -1.0}
    assert not A.worse_than_gauss(row(1, 30, "M1", -1.0), -1.0)
    assert A.worse_than_gauss(row(1, 30, "M1", float("nan"), finite=False), -1.0)
    assert A.worse_than_gauss(row(1, 30, "M1", -1.0000001), -1.0)


def test_an_eligible_fit_that_is_not_worse_does_not_count_for_rule_c():
    rows, g = fits_and_gauss(6, 0)
    for r in rows[10:14]:
        r["eligible"] = True                                                                        # eligible but better than the Gaussian
    assert A.validity(rows, g)["worse_and_eligible"] == 0


def test_a_pair_without_a_finite_gaussian_is_unusable_and_not_counted():
    rows, g = fits_and_gauss(36, 36)
    g[(11001, 30)] = float("nan")
    v = A.validity(rows, g)
    assert v["unusable_pairs"] == 1 and v["worse_than_gauss"] == 35


# ---- choice of the frozen candidate ---------------------------------------------------------------------------------------------------------------------------------------------
def test_the_mildest_verified_candidate_is_frozen():
    res = {"M1": dict(verified=True), "M2": dict(verified=True), "M3": dict(verified=True)}
    norms = {"M1": 22.6, "M2": 33.9, "M3": 1.7}
    assert A.choose(res, norms) == "M3"
    res["M3"]["verified"] = False
    assert A.choose(res, norms) == "M1"
    res["M1"]["verified"] = False
    assert A.choose(res, norms) == "M2"
    res["M2"]["verified"] = False
    assert A.choose(res, norms) is None


def test_a_candidate_that_was_not_run_is_never_chosen():
    assert A.choose({"M2": dict(verified=True)}, {"M1": 1.0, "M2": 5.0, "M3": 0.1}) == "M2"


# ---- the evaluation count, the extension and the verdict --------------------------------------------------------------------------------------------------------------------------
def eval_rows(n_eligible_harmful, n_cert=0, extra_better=3):
    rows = []
    for i in range(n_eligible_harmful + extra_better):
        seed, length = 11021 + i, 60
        rows.append(row(seed, length, "gauss", -1.0))
        harmful = i < n_eligible_harmful
        rows.append(row(seed, length, "M", -2.0 if harmful else -0.5, eligible=True, cert=A.CERTIFIED if i < n_cert else "NOT_CERTIFIED"))
    return rows


def test_the_blind_count_counts_only_eligible_fits_below_the_gaussian_in_arm_m():
    rows = eval_rows(11)
    rows.append(row(11999, 60, "gauss", -1.0)); rows.append(row(11999, 60, "T", -2.0, eligible=True))      # arm T is not arm M
    rows.append(row(11998, 60, "gauss", -1.0)); rows.append(row(11998, 60, "M", -2.0, eligible=False))     # worse but not eligible
    assert A.eligible_harmful_count(rows) == 11


def test_the_blind_count_does_not_depend_on_the_certification_field():
    a, b = eval_rows(15, n_cert=0), eval_rows(15, n_cert=15)
    assert A.eligible_harmful_count(a) == A.eligible_harmful_count(b) == 15
    for r in a:
        r.pop("certification", None)
    assert A.eligible_harmful_count(a) == 15


@pytest.mark.parametrize("n,ext", [(11, True), (12, False), (0, True), (40, False)])
def test_the_extension_is_allowed_only_below_twelve(n, ext):
    assert A.needs_extension(n) is ext


@pytest.mark.parametrize("k,n,v", [(0, 12, "MET"), (0, 30, "MET"), (0, 11, "NOT EVALUABLE"), (1, 30, "NOT MET"), (1, 5, "NOT MET"), (3, 12, "NOT MET")])
def test_verdict_e1(k, n, v):
    assert A.verdict_e1(k, n).startswith(v)


def test_evaluate_reports_the_counts_and_the_bound(tmp_path):
    import json
    rows = eval_rows(14, n_cert=0)
    rows += [row(11200 + i, 60, "gauss", -1.0) for i in range(4)] + [row(11200 + i, 60, "T", -0.5, eligible=True, cert=A.CERTIFIED if i < 2 else "NOT_CERTIFIED") for i in range(4)]
    p = tmp_path / "f.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    A.OUT.clear()
    out = tmp_path / "o.json"
    e1 = A.evaluate([str(p)], str(out))
    d = json.loads(out.read_text())
    assert e1 == "MET" and d["eligible_and_worse"] == 14 and d["certified"] == 0 and d["T_certified"] == 2 and d["E0"] is True
    assert math.isclose(d["upper"], 1 - 0.05 ** (1 / 14), rel_tol=1e-6)


def test_cp_upper_matches_the_closed_form_for_zero_events():
    for n in (12, 14, 40):
        assert math.isclose(A.cp_upper(0, n), 1 - 0.05 ** (1 / n), rel_tol=1e-9)
    assert A.cp_upper(5, 5) == 1.0 and math.isnan(A.cp_upper(0, 0))
