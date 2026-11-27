"""WP10B B3 / B4 analysis (`integration_report/wp10b/scripts/b3_analyze.py`): truth of a fit, exact bound, Wilson interval, verdict words, blind harmful count and extension rule, paired per-user margin, published share, publish rule."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "integration_report" / "wp10b" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SC))
_spec = importlib.util.spec_from_file_location("wp10b_b3_analyze_under_test", SC / "b3_analyze.py")
A = importlib.util.module_from_spec(_spec)
sys.modules["wp10b_b3_analyze_under_test"] = A
_spec.loader.exec_module(A)
from backbone.hssm import cold_start_cert as C  # noqa: E402


def rec(user=1, stratum="easy", g=0.5, state="NOT_CERTIFIED", eligible=True, publishable=False, b=None, ok=True, n=30, noise=0.2):
    rng = np.random.default_rng(user)
    gd = g + noise * rng.standard_normal(n)
    gd = gd - gd.mean() + g
    r = dict(job=dict(user=user, stratum=stratum, variant="S0"), ok=ok, eligible=eligible, dual=dict(publishable=publishable), test_g_diff=[float(x) for x in gd], certificate=dict(state=state, reasons=[]))
    if b is not None:
        bd = b + noise * rng.standard_normal(n)
        r["test_b_diff"] = [float(x) for x in (bd - bd.mean() + b)]
    return r


CERT = C.STATE_CERTIFIED


def test_truth_is_the_point_estimate_and_the_strict_version_uses_1_645_se():
    t = A.truth(rec(g=0.5))
    assert t["harmful"] is False and t["good"] is True and t["strict_good"] is True and t["strict_harmful"] is False
    t = A.truth(rec(g=-0.02, noise=1.0))
    assert t["harmful"] is True and t["good"] is False and t["strict_harmful"] is False                       # a small loss in noisy data is harmful by the point estimate only
    t = A.truth(rec(g=0.02, noise=1.0))
    assert t["good"] is True and t["strict_good"] is False
    t = A.truth(rec(g=-1.0))
    assert t["strict_harmful"] is True


def test_truth_is_none_for_too_few_or_non_finite_days():
    r = rec(); r["test_g_diff"] = [0.1, 0.2]
    assert A.truth(r) is None
    r["test_g_diff"] = [0.1, float("nan"), 0.3]
    assert A.truth(r) is None
    r["test_g_diff"] = []
    assert A.truth(r) is None


def test_exact_upper_bound():
    assert A.exact_upper(0, 14) == pytest.approx(1 - 0.05 ** (1 / 14)) and 0.19 < A.exact_upper(0, 14) < 0.20
    assert A.exact_upper(0, 59) == pytest.approx(1 - 0.05 ** (1 / 59)) and 0.049 < A.exact_upper(0, 59) < 0.050
    assert A.exact_upper(1, 14) > A.exact_upper(0, 14)
    assert A.exact_upper(5, 5) == 1.0 and A.exact_upper(0, 0) is None


def test_wilson_interval():
    lo, hi = A.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = A.wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    assert A.wilson(3, 0) == (None, None)


def test_refusal_verdict_words():
    assert A.refusal_verdict(0, 14) == "MET" and A.refusal_verdict(0, 13) == "NOT DEMONSTRATED" and A.refusal_verdict(1, 40) == "NOT MET"


def test_share_verdict_words():
    assert A.share_verdict(11, 12) == "MET" and A.share_verdict(8, 12) == "MET on the point estimate" and A.share_verdict(5, 12) == "NOT MET" and A.share_verdict(11, 11) == "NOT DEMONSTRATED"
    assert A.share_verdict(6, 12) == "MET on the point estimate" and A.share_verdict(5, 11) == "NOT DEMONSTRATED"
    assert A.share_verdict(0, 12) == "NOT MET"


def test_refusal_block_counts_only_harmful_asked_fits_and_the_certified_among_them():
    recs = [rec(i, g=-0.5, state=CERT if i < 2 else "NOT_CERTIFIED") for i in range(10)] + [rec(20 + i, g=0.5, state=CERT) for i in range(3)] + [rec(40, g=-0.5, eligible=False, state=CERT)] + [rec(41, g=-0.5, ok=False)]
    b = A.refusal_block(recs, "x")
    assert b["n_asked"] == 13 and b["n_harmful"] == 10 and b["certified_among_harmful"] == 2 and b["certified_all"] == 5
    assert b["verdict"] == "NOT MET" and b["upper95"] == pytest.approx(A.exact_upper(2, 10))
    assert b["harmful_users"] == list(range(10))


def test_share_block_counts_good_asked_fits():
    recs = [rec(i, g=0.5, state=CERT) for i in range(7)] + [rec(10 + i, g=0.5) for i in range(5)] + [rec(30, g=-0.5, state=CERT)] + [rec(31, g=0.5, eligible=False, state=CERT)]
    b = A.share_block(recs, "x")
    assert b["n_asked"] == 13 and b["n_good"] == 12 and b["certified_among_good"] == 7 and b["share"] == pytest.approx(7 / 12) and b["verdict"] == "MET on the point estimate"


def test_the_extension_rule_reads_harm_without_the_certificate():
    recs = [rec(i, g=-0.5, state=CERT if i % 2 else "NOT_CERTIFIED") for i in range(13)] + [rec(30 + i, g=0.5) for i in range(9)] + [rec(50, g=-0.5, eligible=False)]
    assert A.blind_harmful_count(recs) == 13 and A.needs_extension(recs) is True
    assert A.needs_extension(recs + [rec(60, g=-0.5)]) is False


def test_paired_users_margin_is_across_users_at_three_standard_errors():
    recs = [rec(i, b=0.4 + 0.1 * (i % 3 - 1)) for i in range(12)]
    p = A.paired_users(recs, "test_b_diff")
    v = np.array([np.mean(r["test_b_diff"]) for r in recs])
    assert p["n"] == 12 and p["mean"] == pytest.approx(v.mean()) and p["se"] == pytest.approx(v.std(ddof=1) / math.sqrt(12)) and p["lower3"] == pytest.approx(v.mean() - 3 * v.std(ddof=1) / math.sqrt(12))
    assert A.paired_users(recs[:2], "test_b_diff")["lower3"] is None


def test_published_is_converged_publishable_or_certified():
    assert A.published(rec(publishable=True)) and A.published(rec(state=CERT)) and not A.published(rec()) and not A.published(rec(publishable=True, ok=False))


def test_b4_rule_needs_both_the_baseline_margin_and_the_published_share():
    good_b = [rec(i, "easy" if i % 2 else "medium", state=CERT, b=0.5) for i in range(20)]
    r = A.b4_block(good_b)
    assert r["n_users"] == 20 and r["published"] == 20 and r["beats_baseline_3se"] is True and r["published_verdict"] == "MET" and r["rule_met"] and r["rule_met_strict"]
    behind = [rec(i, "easy", state=CERT, b=-0.5) for i in range(20)]
    r = A.b4_block(behind)
    assert r["beats_baseline_3se"] is False and r["published_verdict"] == "MET" and r["rule_met"] is False
    few = [rec(i, "easy", state=CERT if i < 3 else "NOT_CERTIFIED", b=0.5) for i in range(20)]
    r = A.b4_block(few)
    assert r["beats_baseline_3se"] is True and r["published"] == 3 and r["published_verdict"] == "NOT MET" and r["rule_met"] is False


def test_b4_uses_easy_and_medium_users_only():
    recs = [rec(i, "hard", state=CERT, b=0.5) for i in range(10)] + [rec(20 + i, "easy", b=0.5) for i in range(4)]
    r = A.b4_block(recs)
    assert r["n_users"] == 4 and r["published"] == 0


def test_the_point_estimate_verdict_of_the_published_share():
    recs = [rec(i, "easy", state=CERT if i < 15 else "NOT_CERTIFIED", b=0.5) for i in range(24)]
    r = A.b4_block(recs)
    assert r["published"] == 15 and r["published_verdict"] == "MET on the point estimate" and r["rule_met"] is True and r["rule_met_strict"] is False


def test_stratum_table_and_baseline_margin():
    recs = [rec(1, "hard", state=CERT, b=0.1), rec(2, "hard"), rec(3, "easy", publishable=True)]
    recs[0]["certificate"]["baseline"] = dict(beats_baseline=True)
    t = A.stratum_table(recs)
    assert t["hard"]["n"] == 2 and t["hard"]["certified"] == 1 and t["hard"]["published"] == 1 and t["easy"]["converged_publishable"] == 1 and t["medium"]["n"] == 0
    assert A.baseline_margin_block(recs) == dict(certified=1, certified_and_beats_baseline_refits=1)


def test_analyse_marks_breaker_fits_as_asked_and_reads_the_real_engine_as_a_share():
    br = [dict(job=dict(engine="stops", seed=s), ok=True, test_g_diff=rec(s, g=-3.0)["test_g_diff"], certificate=dict(state="NOT_CERTIFIED", reasons=[])) for s in range(14)]
    br += [dict(job=dict(engine="B2", seed=s), ok=True, test_g_diff=rec(s, g=0.5)["test_g_diff"], certificate=dict(state="NOT_CERTIFIED", reasons=[])) for s in range(5)]
    br += [dict(job=dict(engine="real", seed=s), ok=True, test_g_diff=rec(s, g=0.5)["test_g_diff"], certificate=dict(state=CERT, reasons=[])) for s in range(12)]
    a = A.analyse([rec(1, "hard", g=-0.5), rec(2, "easy", g=0.5, state=CERT)], br)
    assert a["breakers"]["stops"]["refusal"]["n_harmful"] == 14 and a["breakers"]["stops"]["refusal"]["verdict"] == "MET"
    assert a["breakers"]["B2"]["refusal"]["n_harmful"] == 0 and a["breakers"]["B2"]["refusal"]["verdict"] == "NOT DEMONSTRATED"
    assert a["breakers"]["real"]["share"]["verdict"] == "MET" and a["breakers"]["real"]["n"] == 12
    assert "REFUSAL" in A.render(a) and "B4 DATA" in A.render(a)
