"""WP9 B3: `k_adequacy` (the "unsure number of phases" tag) and the developer alert events. Design: tests/PREDECLARED_CRITERIA_WP9.md Part 4.
Mutation check: integration_report/wp9/scripts/mutation_b3.py."""
from __future__ import annotations

import contextlib
import io
import json
import logging
import warnings

import numpy as np
import pytest

from backbone.hssm import alerts
from backbone.hssm import k_adequacy as KA
from backbone.hssm.config import K_ADEQUACY_BIC_MARGIN, K_ADEQUACY_MIN_SEPARATION
from backbone.hssm.fitting import HSSMResult, fit_hssm


# ------------------------------------------------------------------------------------------------------------------- the separation measure
def test_pairwise_separation_is_the_rms_standardised_difference_of_the_closest_pair():
    mu = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 0.5]])
    var = np.ones((3, 2))
    d, pair = KA.pairwise_separation(mu, var)
    assert pair == (1, 2) and d == pytest.approx(np.sqrt(0.25 / 2))                                   # sqrt(mean([0, 0.25]))
    d01, _ = KA.pairwise_separation(mu[:2], var[:2])
    assert d01 == pytest.approx(np.sqrt(4.0 / 2))
    d_var, _ = KA.pairwise_separation(mu[:2], np.array([[1.0, 1.0], [3.0, 1.0]]))                     # pooled variance (1 + 3) / 2 = 2 in the first feature
    assert d_var == pytest.approx(np.sqrt((4.0 / 2.0 + 0.0) / 2))
    assert KA.pairwise_separation(mu[:1], var[:1]) == (float("inf"), None)


def test_the_separation_does_not_grow_with_the_number_of_features():
    one = KA.pairwise_separation(np.array([[0.0], [1.5]]), np.ones((2, 1)))[0]
    many = KA.pairwise_separation(np.array([[0.0] * 9, [1.5] * 9]), np.ones((2, 9)))[0]
    assert one == pytest.approx(many) == pytest.approx(1.5)


def test_bad_shapes_and_variances_are_refused():
    with pytest.raises(ValueError):
        KA.pairwise_separation(np.zeros((2, 3)), np.ones((2, 2)))
    with pytest.raises(ValueError):
        KA.pairwise_separation(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        KA.pairwise_separation(np.zeros(3), np.ones(3))


# ------------------------------------------------------------------------------------------------------------------- the assessment
MU = np.array([[0.0, 0.0], [4.0, 4.0], [8.0, 0.0]])
VAR = np.ones((3, 2))
BIC = {2: 120.0, 3: 100.0, 4: 130.0, 5: 150.0}


def ad(**kw):
    args = dict(selected_k=3, bic_by_k=BIC, mu=MU, var=VAR, fit_unstable=False)
    args.update(kw)
    return KA.assess_k_adequacy(**args)


def test_the_defaults_are_the_starting_values_of_the_declaration():
    assert K_ADEQUACY_MIN_SEPARATION == 1.0 and K_ADEQUACY_BIC_MARGIN == 6.0


def test_a_clean_fit_is_not_unsure_and_the_best_guess_is_shown():
    out = ad()
    assert out["k_unsure"] is False and out["reasons"] == [] and out["selected_k"] == 3 and out["best_guess_shown"] is True
    assert out["signals"]["runner_up_k"] == 2 and out["signals"]["runner_up_bic_gap"] == pytest.approx(20.0) and out["signals"]["candidates"] == [2, 3, 4, 5]
    assert out["thresholds"] == dict(min_separation=1.0, bic_margin=6.0, flag_largest=True)


def test_overlapping_regimes_are_flagged_and_the_level_is_exclusive():
    close = np.array([[0.0, 0.0], [1.0, 1.0], [8.0, 0.0]])                                              # closest pair: rms difference exactly 1.0
    assert KA.REASON_OVERLAP not in ad(mu=close)["reasons"]                                              # equal to d0: not flagged
    assert KA.REASON_OVERLAP in ad(mu=close, min_separation=1.0000001)["reasons"]
    assert ad(mu=close, min_separation=1.0000001)["k_unsure"] is True


def test_a_runner_up_within_the_bic_margin_is_flagged_and_the_margin_is_exclusive():
    assert KA.REASON_NOT_BEST not in ad(bic_by_k={2: 106.0, 3: 100.0, 4: 130.0, 5: 150.0})["reasons"]       # gap 6 = margin: not flagged
    assert KA.REASON_NOT_BEST in ad(bic_by_k={2: 105.9, 3: 100.0, 4: 130.0, 5: 150.0})["reasons"]
    assert KA.REASON_NOT_BEST in ad(bic_by_k={2: 120.0, 3: 100.0, 4: 103.0, 5: 150.0})["reasons"]           # a larger K counts as well
    assert KA.REASON_NOT_BEST in ad(bic_by_k={2: 120.0, 3: 100.0, 4: 99.0, 5: 150.0})["reasons"]            # a better BIC that was demoted (collapsed) is the strongest case


def test_an_infinite_bic_of_a_failed_candidate_is_ignored():
    out = ad(bic_by_k={2: float("inf"), 3: 100.0, 4: 130.0, 5: float("inf")})
    assert out["signals"]["runner_up_k"] == 4 and out["reasons"] == []


def test_the_largest_candidate_is_flagged_only_when_there_is_something_to_compare():
    top = ad(selected_k=5, bic_by_k={2: 150.0, 3: 140.0, 4: 130.0, 5: 100.0}, mu=np.array([[0.0, 0], [4, 4], [8, 0], [0, 8], [8, 8]]), var=np.ones((5, 2)))
    assert KA.REASON_UPPER in top["reasons"]
    single = KA.assess_k_adequacy(3, {3: 100.0}, MU, VAR, False)
    assert single["reasons"] == [] and single["signals"]["runner_up_k"] is None and single["signals"]["runner_up_bic_gap"] is None
    assert KA.REASON_UPPER not in ad(candidates=[2, 3, 4, 5])["reasons"]


def test_the_largest_candidate_signal_can_be_switched_off_and_the_default_is_on():
    from backbone.hssm.config import K_ADEQUACY_FLAG_LARGEST
    assert K_ADEQUACY_FLAG_LARGEST is True
    kw = dict(selected_k=5, bic_by_k={2: 150.0, 3: 140.0, 4: 130.0, 5: 100.0}, mu=np.array([[0.0, 0], [4, 4], [8, 0], [0, 8], [8, 8]]), var=np.ones((5, 2)))
    off = ad(flag_largest=False, **kw)
    assert KA.REASON_UPPER not in off["reasons"] and off["k_unsure"] is False and off["thresholds"]["flag_largest"] is False
    assert KA.REASON_UPPER in ad(**kw)["reasons"] and ad(**kw)["thresholds"]["flag_largest"] is True


def test_restart_disagreement_and_a_missing_stability_assessment_are_flagged():
    assert ad(fit_unstable=True)["reasons"] == [KA.REASON_RESTARTS]
    assert ad(fit_unstable=None)["reasons"] == [KA.REASON_RESTARTS]                                      # fail closed
    assert ad(fit_unstable=False)["reasons"] == []


def test_reasons_accumulate_in_a_fixed_order():
    close = np.array([[0.0, 0.0], [0.5, 0.5], [8.0, 0.0]])
    out = ad(mu=close, fit_unstable=True, bic_by_k={2: 101.0, 3: 100.0, 4: 130.0, 5: 150.0})
    assert out["reasons"] == [KA.REASON_OVERLAP, KA.REASON_NOT_BEST, KA.REASON_RESTARTS] and out["k_unsure"] is True


# ------------------------------------------------------------------------------------------------------------------- alerts
@pytest.fixture(autouse=True)
def _clean_handlers():
    alerts.clear_handlers()
    yield
    alerts.clear_handlers()


def test_an_event_carries_a_hash_of_the_user_and_never_the_user_id(caplog):
    caplog.set_level(logging.WARNING, logger="chronis.hssm.alerts")
    got = []
    alerts.register_handler(got.append)
    ev = alerts.emit("k_unsure", "alice@example.org", K=3, reasons=["regimes_overlap"])
    assert ev == got[0] and ev["event"] == "k_unsure" and ev["schema_version"] == 1 and len(ev["user_id_hash"]) == 16
    assert ev["user_id_hash"] == alerts.user_id_hash("alice@example.org") != alerts.user_id_hash("bob@example.org")
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "alice" not in text and "example.org" not in text and "CHRONIS_DEV_ALERT" in text and json.loads(text.split("CHRONIS_DEV_ALERT ", 1)[1])["K"] == 3
    assert len(caplog.records) == 1 and caplog.records[0].levelno == logging.WARNING


def test_personal_or_raw_fields_are_refused():
    for bad in ("email", "matrix", "raw", "data", "X", "name", "device_id"):
        with pytest.raises(ValueError):
            alerts.make_event("k_unsure", "u", **{bad: 1})
    with pytest.raises((ValueError, TypeError)):                                                        # `user_id` is the positional argument: a second value for it is refused as well
        alerts.make_event("k_unsure", "u", **{"user_id": "x"})


def test_a_failing_handler_never_breaks_the_caller_and_is_counted():
    seen = []
    alerts.register_handler(lambda ev: 1 / 0)
    alerts.register_handler(seen.append)
    before = alerts.handler_failures
    alerts.emit("k_unsure", "u", K=2)
    assert len(seen) == 1 and alerts.handler_failures == before + 1


# ------------------------------------------------------------------------------------------------------------------- inside fit_hssm
def _planted(seed, K=3):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration_report" / "wp7r" / "scripts"))
    from k_select_study import planted
    X, _ = planted(seed, K, "easy")
    return X


def _fit(seed=9998, cands=(2, 3, 4), n_init=10, user="u-9998", **kw):
    logging.disable(logging.CRITICAL)
    X = _planted(seed)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        return fit_hssm(matrix=X, candidate_ks=cands, n_initializations=n_init, max_iter=25, random_seed=seed, timestamps=np.arange(len(X), dtype=float), user_id=user, dataset_hash="h", **kw)


@pytest.fixture(scope="module")
def live():
    return _fit()


def test_a_live_fit_carries_the_adequacy_record_and_the_property(live):
    ad_ = live.k_adequacy
    assert set(ad_) >= {"k_unsure", "reasons", "selected_k", "best_guess_shown", "signals", "thresholds"} and ad_["selected_k"] == live.K == live.selected_k
    assert live.k_unsure == ad_["k_unsure"] and isinstance(live.k_unsure, bool)
    assert HSSMResult.__dataclass_fields__["k_adequacy"].default_factory is dict
    json.dumps(ad_)


def test_the_selected_k_is_the_bic_choice_whatever_the_tag_says(live, monkeypatch):
    healthy = {k: b for k, b in live.bic_by_k.items() if np.isfinite(b)}
    assert live.K == min(healthy, key=healthy.get)                                                      # unchanged BIC rule (no collapsed demotion in this fit)
    import backbone.hssm.k_adequacy as mod
    monkeypatch.setattr(mod, "assess_k_adequacy", lambda *a, **k: dict(k_unsure=True, reasons=["x"], selected_k=-1, best_guess_shown=True, signals={}, thresholds={}))
    other = _fit()
    assert other.K == live.K and np.array_equal(np.asarray(other.p_t_posterior), np.asarray(live.p_t_posterior))      # only the additive field differs


def test_an_unsure_fit_raises_one_developer_alert_with_a_hashed_user(monkeypatch):
    got = []
    alerts.register_handler(got.append)
    import backbone.hssm.k_adequacy as mod
    monkeypatch.setattr(mod, "assess_k_adequacy", lambda *a, **k: dict(k_unsure=True, reasons=[KA.REASON_OVERLAP], selected_k=3, best_guess_shown=True,
                                                                        signals=dict(min_separation_sd=0.4, runner_up_bic_gap=2.0), thresholds={}))
    r = _fit(user="person-17")
    assert r.k_unsure and len(got) == 1 and got[0]["event"] == "k_unsure" and got[0]["K"] == 3 and got[0]["reasons"] == [KA.REASON_OVERLAP]
    assert got[0]["user_id_hash"] == alerts.user_id_hash("person-17") and "person-17" not in json.dumps(got[0])


def test_a_clean_fit_raises_no_alert(monkeypatch):
    got = []
    alerts.register_handler(got.append)
    import backbone.hssm.k_adequacy as mod
    monkeypatch.setattr(mod, "assess_k_adequacy", lambda *a, **k: dict(k_unsure=False, reasons=[], selected_k=3, best_guess_shown=True, signals={}, thresholds={}))
    assert _fit().k_unsure is False and got == []


def test_an_exception_in_the_assessment_fails_closed_and_does_not_break_the_fit(monkeypatch):
    got = []
    alerts.register_handler(got.append)
    import backbone.hssm.k_adequacy as mod

    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(mod, "assess_k_adequacy", boom)
    r = _fit()
    assert r.k_unsure is True and r.k_adequacy["reasons"] == ["assessment_failed"] and r.K in (2, 3, 4) and len(got) == 1


def test_a_result_without_an_assessment_is_not_unsure():
    assert HSSMResult.__dataclass_fields__["k_adequacy"].default_factory() == {}
    stub = type("S", (), {"k_adequacy": {}})()
    assert HSSMResult.k_unsure.fget(stub) is False
