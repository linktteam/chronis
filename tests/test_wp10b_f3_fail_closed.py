"""WP10B F3: a crashed or malformed assessment never gives ALIGNED (opa's package P02A with the owner's amendment). Pre-declared in tests/PREDECLARED_CRITERIA_WP10B.md section 12 (T1 to T14).
Stages: k_change, fit_stability, scaler, matching. The alert is `alerts.ASSESSMENT_CRASHED` with user_id_hash, stage, data_revision_id, reason and error_class (the class name only)."""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from backbone.hssm import alerts
from backbone.hssm import k_change as KC
from backbone.hssm import lineage as L
from backbone.hssm.lineage import ALIGNED, CONTINUITY_UNVERIFIED, K_CHANGED, NEW_LINEAGE, AlignmentConfig, make_summary, may_claim_same_as_before, step
from backbone.hssm.model import FittingConvergenceError
from backbone.hssm.ordering import LadderConfig
from data.synthetic_identity import REGISTRY, SCALE, IdentitySpec, draw_user

CFG = AlignmentConfig(min_separation=1.0, ladder=LadderConfig(n_bootstrap=40, min_gap=0.5))
SECRET = "SECRET-MESSAGE-9f3a 123.456 patient_value"
OK_RECORD = dict(schema_version=1, suspected=False, direction=None, k_best=3, margin=None, reason="k_anchor_is_best", bic_by_k={}, candidates=(), recent_days=45, thresholds=dict(delta_up=10.0, delta_down=10.0))


@pytest.fixture(autouse=True)
def fresh_alerts():
    alerts.reset_dedup()
    alerts.clear_handlers()
    got = []
    alerts.register_handler(got.append)
    yield got
    alerts.clear_handlers()
    alerts.reset_dedup()


def means(seed=3, K=3, sep=2.5):
    return draw_user(seed, IdentitySpec(n_regimes=K, sep_lo=sep, sep_hi=sep, offset_sd=0.3))["means"]


def summary(m, seed=0, T=240, noise=0.35, engine="baseline", **kw):
    rng = np.random.default_rng(seed)
    K, F = m.shape
    labels = rng.permutation(np.repeat(np.arange(K), T // K)[:T])
    window = m[labels] + noise * SCALE * rng.standard_normal((len(labels), F))
    return make_summary(engine, m, window.mean(axis=0), window.std(axis=0), labels, window, **kw)


def started(m=None, **kw):
    m = means() if m is None else m
    rec, st = step(None, summary(m, seed=1, **kw), CFG, REGISTRY, lineage_id="t")
    return m, rec, st


def crash_record(name="ValueError"):
    return dict(schema_version=1, suspected=False, direction=None, k_best=3, margin=None, reason="assessment_crashed", crashed=True, error_class=name, bic_by_k={}, candidates=())


def safe_step(st, n, cfg=CFG):
    """`step`, with the test's own verdict if an exception escapes (a crashed assessment must give a record, not an exception)."""
    try:
        return step(st, n, cfg, REGISTRY)
    except Exception as exc:                                                              # noqa: BLE001
        pytest.fail(f"an exception escaped the lineage: {type(exc).__name__}")


def assert_failed_closed(rec, state_name=CONTINUITY_UNVERIFIED):
    assert rec.continuity_state == state_name and "assessment_crashed" in rec.reasons and rec.continuity_state != ALIGNED and not may_claim_same_as_before(rec)


# ------------------------------------------------------------------------------------------------ T1, T2, T3
@pytest.mark.parametrize("rec_in", [dict(OK_RECORD, reason="assessment_error", error="ValueError: x"), crash_record(), dict(OK_RECORD, crashed=True)])
def test_t1_a_crashed_k_change_record_is_never_aligned(rec_in):
    m, _, st = started()
    r, new = step(st, summary(m, seed=2, k_change=rec_in), CFG, REGISTRY)
    assert_failed_closed(r)
    assert r.reasons.count("assessment_crashed") == 1
    healthy, _ = step(st, summary(m, seed=2, k_change=OK_RECORD), CFG, REGISTRY)
    assert healthy.continuity_state == ALIGNED and "assessment_crashed" not in healthy.reasons          # the same fit with a healthy record is aligned: only the crash changed the result


@pytest.mark.parametrize("bad", ["garbage", 7, ["suspected"], dict(OK_RECORD, suspected="yes"), dict(OK_RECORD, suspected=1), dict(OK_RECORD, suspected=[True]), dict(OK_RECORD, suspected=None),
                                 {k: v for k, v in OK_RECORD.items() if k != "suspected"}])
def test_t2_a_malformed_k_change_record_is_never_aligned(bad):
    m, _, st = started()
    n = summary(m, seed=2)
    n = L.FitSummary(n.engine, n.raw_means, n.scaler_mean, n.scaler_std, n.labels, n.separation_min, n.raw_window, n.exp_durations, None, False, None, bad)       # a record that bypasses make_summary
    r, _ = safe_step(st, n)
    assert_failed_closed(r)
    if not isinstance(bad, dict):
        assert r.k_change is not None and r.k_change.get("crashed") is True and r.k_change.get("error_class") == "MalformedRecord"     # shown in the record as a crash record
    r2, _ = safe_step(st, summary(m, seed=2, k_change=bad))                                                      # through make_summary
    assert_failed_closed(r2)


def test_t3_the_unverified_result_keeps_the_old_names_and_carries_a_proposal():
    m, first, st = started()
    r, new = step(st, summary(m, seed=2, k_change=crash_record()), CFG, REGISTRY)
    assert_failed_closed(r)
    assert tuple(r.raw_to_stable) is not None and r.proposed_new_names is not None and new.pending is not None and new.anchor is st.anchor
    assert r.message.startswith("cannot confirm this is the same as before") and r.k_change["crashed"] is True


# ------------------------------------------------------------------------------------------------ T4, T5
def test_t4_assess_k_change_marks_an_unexpected_exception_as_crashed_and_a_non_convergence_as_unavailable():
    mu = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]); var = np.ones((3, 2))

    def make(exc):
        def fit(recent, K):
            if K == 4:
                raise exc
            return dict(bic=300.0, labels=np.zeros(len(recent), dtype=int), raw_means=np.zeros((K, 2)), accepted=True)
        return fit
    for exc in (RuntimeError(SECRET), TypeError(SECRET), KeyError(SECRET), ValueError(SECRET)):
        r = KC.assess_k_change(mu, var, np.zeros((45, 2)), fit_fn=make(exc))
        assert r.get("crashed") is True and r.get("reason") == "assessment_crashed" and r["suspected"] is False and r.get("error_class") == type(exc).__name__
        assert SECRET not in json.dumps(r, default=str) and "error" not in r
    for exc in (FittingConvergenceError(SECRET),):
        r = KC.assess_k_change(mu, var, np.zeros((45, 2)), fit_fn=make(exc))
        assert "crashed" not in r and r["bic_by_k"]["4"] is None and r["reason"] != "assessment_crashed"
    bad = KC.assess_k_change(mu, var, "not an array", fit_fn=make(RuntimeError("x")))
    assert bad.get("crashed") is True and bad.get("error_class") in ("ValueError", "TypeError", "IndexError", "AttributeError")


def test_t4b_the_anchor_variance_failure_is_a_crash_record_with_the_class_name_only():
    m = means()
    s = summary(m, seed=1)
    state = SimpleNamespace(anchor=SimpleNamespace(summary=SimpleNamespace(raw_window="not an array", labels=s.labels, K=3, raw_means=m)))
    r = KC.assess_against_anchor(state, np.zeros((45, 3)), fit_fn=lambda a, k: None)
    assert r.get("crashed") is True and r.get("reason") == "assessment_crashed" and "error" not in r and str(r.get("error_class")).isidentifier()


def test_t5_fit_recent_window_returns_none_only_for_the_two_expected_classes_and_lets_any_other_exception_through(monkeypatch):
    import backbone.hssm.fitting as FIT
    from backbone.hssm.gating import ColdStartError
    X = np.random.default_rng(0).normal(size=(50, 3))
    for exc, expect_none in ((FittingConvergenceError("x"), True), (ColdStartError("x"), True), (RuntimeError(SECRET), False), (ValueError(SECRET), False), (TypeError("x"), False), (KeyError("x"), False)):
        def boom(**kw):
            raise exc
        monkeypatch.setattr(FIT, "fit_hssm", boom)
        if expect_none:
            assert KC.fit_recent_window(X, 2, seed=0, n_init=3, max_iter=5, allow_fast_test_fit=True) is None
        else:
            with pytest.raises(type(exc)):
                KC.fit_recent_window(X, 2, seed=0, n_init=3, max_iter=5, allow_fast_test_fit=True)


# ------------------------------------------------------------------------------------------------ T6, T7, T8
def ids(user="u-1", rev="rev-1", **kw):
    return dict(user_id=user, data_revision_id=rev, **kw)


def test_t6_the_alert_has_the_declared_kind_and_exactly_the_declared_fields(fresh_alerts):
    m, _, st = started()
    step(st, summary(m, seed=2, k_change=crash_record("ValueError"), **ids()), CFG, REGISTRY)
    assert len(fresh_alerts) == 1
    ev = fresh_alerts[0]
    assert alerts.ASSESSMENT_CRASHED == "ASSESSMENT_CRASHED" and ev["event"] == "ASSESSMENT_CRASHED"
    assert set(ev) == {"event", "schema_version", "user_id_hash", "stage", "data_revision_id", "reason", "error_class"}
    assert ev["stage"] == "k_change" and ev["data_revision_id"] == "rev-1" and ev["reason"] == "assessment_crashed" and ev["error_class"] == "ValueError" and ev["user_id_hash"] == alerts.user_id_hash("u-1")
    assert "u-1" not in json.dumps(ev)


def test_t7_a_message_text_a_traceback_or_a_data_value_never_reaches_the_alert_the_log_or_the_record(fresh_alerts, caplog):
    mu = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]); var = np.ones((3, 2))

    def fit(recent, K):
        raise RuntimeError(SECRET)
    rec = KC.assess_k_change(mu, var, np.zeros((45, 2)), fit_fn=fit)
    m, _, st = started()
    with caplog.at_level(logging.DEBUG):
        r, new = step(st, summary(m, seed=2, k_change=rec, **ids()), CFG, REGISTRY)
    blob = json.dumps(fresh_alerts, default=str) + " ".join(x.getMessage() for x in caplog.records) + json.dumps(r.k_change, default=str) + repr(r.reasons) + r.message
    assert fresh_alerts and "SECRET" not in blob and "123.456" not in blob and "patient_value" not in blob and "Traceback" not in blob
    assert alerts.exception_class_name("Value Error: with a message") == "Exception" and alerts.exception_class_name("a" * 200) == "Exception" and alerts.exception_class_name(KeyError("x")) == "KeyError"
    legacy = dict(OK_RECORD, reason="assessment_error", error=f"ValueError: {SECRET}")
    step(st, summary(m, seed=2, k_change=legacy, **ids(rev="rev-legacy")), CFG, REGISTRY)
    assert "SECRET" not in json.dumps(fresh_alerts, default=str) and fresh_alerts[-1]["error_class"] == "ValueError"


def test_t8_at_most_one_alert_per_user_stage_and_revision(fresh_alerts):
    m, _, st = started()
    def go(**kw):
        step(st, summary(m, seed=2, k_change=crash_record(), **kw), CFG, REGISTRY)
    go(**ids()); go(**ids())
    assert len(fresh_alerts) == 1
    go(**ids(rev="rev-2"))
    assert len(fresh_alerts) == 2
    go(**ids(user="u-2"))
    assert len(fresh_alerts) == 3
    go(**ids(user="u-2", rev="rev-1"))
    assert len(fresh_alerts) == 3
    # another stage, same user and revision: a fit-stability crash on the same data gives its own alert
    n = summary(m, seed=2, k_change=crash_record(), assessment_crashes=(("fit_stability", "KeyError"),), **ids())
    step(st, n, CFG, REGISTRY)
    assert [e["stage"] for e in fresh_alerts][-1] == "fit_stability" and len(fresh_alerts) == 4
    # two crashes of the SAME stage in one step give one alert
    n = summary(m, seed=2, assessment_crashes=(("scaler", "KeyError"), ("scaler", "ValueError")), **ids(rev="rev-9"))
    step(st, n, CFG, REGISTRY)
    assert [e["stage"] for e in fresh_alerts].count("scaler") == 1


def test_t8b_the_dedup_memory_is_bounded(monkeypatch):
    monkeypatch.setattr(alerts, "DEDUP_MAX", 3)
    got = []
    alerts.register_handler(got.append)
    for i in range(5):
        alerts.emit_once("k", "u", ("rev", i), x=1)
    assert len(alerts._seen) == 3 and alerts.emit_once("k", "u", ("rev", 4), x=1) is None and alerts.emit_once("k", "u", ("rev", 0), x=1) is not None


# ------------------------------------------------------------------------------------------------ T9
class Raiser:
    def __init__(self, exc=RuntimeError("delivery broke")):
        self.exc = exc

    def __call__(self, *a, **k):
        raise self.exc


def run_closed(st, m, **kw):
    try:
        r, new = step(st, summary(m, seed=2, k_change=crash_record(), **kw), CFG, REGISTRY)
    except Exception as exc:                                                              # the test's own verdict: an exception escaped
        pytest.fail(f"the alert failure escaped the lineage: {type(exc).__name__}")
    return r


def test_t9_a_failure_to_deliver_the_alert_cannot_change_the_result(monkeypatch, caplog):
    m, _, st = started()
    # a handler that raises (swallowed by emit)
    alerts.register_handler(Raiser(RuntimeError(SECRET)))
    before = alerts.handler_failures
    with caplog.at_level(logging.WARNING, logger="chronis.hssm.alerts"):
        assert_failed_closed(run_closed(st, m, **ids()))
    assert alerts.handler_failures == before + 1
    logged = [x.getMessage() for x in caplog.records if "handler failed" in x.getMessage()]
    assert len(logged) == 1 and "RuntimeError" in logged[0] and "SECRET-MESSAGE" not in logged[0] and "123.456" not in logged[0]          # logged, class name only
    caplog.clear()
    # emit_once itself raising
    monkeypatch.setattr(alerts, "emit_once", Raiser(RuntimeError("down")))
    with caplog.at_level(logging.WARNING, logger="chronis.hssm.lineage"):
        r = run_closed(st, m, **ids(rev="r2"))
    assert_failed_closed(r)
    assert any("not delivered" in x.getMessage() and "RuntimeError" in x.getMessage() and "down" not in x.getMessage() for x in caplog.records)
    # the alert logger raising
    monkeypatch.undo()
    monkeypatch.setattr(alerts, "logger", SimpleNamespace(warning=Raiser(OSError("disk"))))
    assert_failed_closed(run_closed(st, m, **ids(rev="r3")))
    # and the logger of the lineage raising as well
    monkeypatch.setattr(L, "logger", SimpleNamespace(warning=Raiser(OSError("disk"))))
    monkeypatch.setattr(alerts, "emit_once", Raiser(RuntimeError("down")))
    assert_failed_closed(run_closed(st, m, **ids(rev="r4")))


# ------------------------------------------------------------------------------------------------ T10, T11
def baseline_result(m, post=None, restart_summary=None, user_id="u-1", dataset_hash="h-1"):
    s = summary(m, seed=2)
    p = np.eye(m.shape[0])[s.labels] * 0.98 + 0.01 if post is None else post
    model = SimpleNamespace(mu=(m - s.scaler_mean) / s.scaler_std, dur_mu=np.array([2.0] * m.shape[0]), dur_sigma=np.array([0.5] * m.shape[0]), Dmax=30)
    rs = dict(fit_unstable=False, fit_unstable_reason=None) if restart_summary is None else restart_summary
    return SimpleNamespace(model=model, p_t_posterior=p, is_baseline_model=True, restart_summary=rs, user_id=user_id, fit_dataset_hash=dataset_hash), s


def test_t10_a_crash_in_the_fit_stability_assessment_and_a_malformed_posterior_fail_closed(fresh_alerts):
    m, _, st = started()
    res, s = baseline_result(m)
    good = L.summarize_fit(res, s.raw_window, s.scaler_mean, s.scaler_std)
    assert good.assessment_crashes == () and good.fit_unstable is False and good.user_id == "u-1" and good.data_revision_id == "h-1"

    class Exploding(SimpleNamespace):
        @property
        def restart_summary(self):
            raise KeyError(SECRET)
    res2, _ = baseline_result(m)
    boom = Exploding(**{k: v for k, v in vars(res2).items() if k != "restart_summary"})
    crashed = L.summarize_fit(boom, s.raw_window, s.scaler_mean, s.scaler_std)
    assert crashed.fit_unstable is True and crashed.fit_unstable_reason == "assessment_crashed" and crashed.assessment_crashes == (("fit_stability", "KeyError"),)
    r, _ = safe_step(st, crashed)
    assert_failed_closed(r)
    assert "fit_unstable" in r.reasons
    assert [e["stage"] for e in fresh_alerts] == ["fit_stability"] and fresh_alerts[0]["error_class"] == "KeyError" and SECRET not in json.dumps(fresh_alerts)
    # an unusable posterior: not finite / empty / one-dimensional
    for k, post in enumerate((np.full((240, 3), np.nan), np.zeros((0, 3)), np.ones(240))):
        res3, _ = baseline_result(m, post=post if post.ndim == 2 and post.shape[0] == 240 else np.nan_to_num(np.eye(3)[s.labels]))
        res3.p_t_posterior = post
        try:
            sm = L.summarize_fit(res3, s.raw_window, s.scaler_mean, s.scaler_std)
        except Exception:                                                                 # the labels cannot even be derived from this posterior: no summary exists, nothing can be served
            continue
        assert sm.fit_unstable is True and sm.assessment_crashes and sm.assessment_crashes[0][0] == "fit_stability"
        assert_failed_closed(safe_step(st, sm)[0])


def test_t11_a_crash_in_the_scaler_or_the_matching_step_fails_closed(fresh_alerts, monkeypatch):
    m, _, st = started()
    n = summary(m, seed=2, **ids())
    healthy, _ = step(st, n, CFG, REGISTRY)
    assert healthy.continuity_state == ALIGNED and fresh_alerts == []
    for stage, name, exc in (("scaler", "_scale", KeyError("x")), ("matching", "_combined_alignment", RuntimeError("y")), ("matching", "margin_ok", ValueError("z"))):
        alerts.reset_dedup(); fresh_alerts.clear()
        def boom(*a, _e=exc, **k):
            raise _e
        monkeypatch.setattr(L, name, boom)
        r, new = safe_step(st, summary(m, seed=2, **ids()))
        assert_failed_closed(r)
        assert [e["stage"] for e in fresh_alerts] == [stage] and fresh_alerts[0]["error_class"] == type(exc).__name__
        assert new.anchor is st.anchor                                                   # the crashed step does not move the anchor
        monkeypatch.undo()


def test_t11b_a_crash_in_the_matching_of_a_k_change_is_recorded_and_the_state_stays_k_changed(fresh_alerts, monkeypatch):
    m, _, st = started()
    m4 = means(seed=4, K=4)
    monkeypatch.setattr(L, "split_merge_candidates", lambda *a: (_ for _ in ()).throw(RuntimeError("q")))
    r, _ = safe_step(st, summary(m4, seed=2, **ids()))
    assert r.continuity_state == K_CHANGED and "assessment_crashed" in r.reasons and "k_changed" in r.reasons and [e["stage"] for e in fresh_alerts] == ["matching"]


def test_t11c_a_crashed_k_change_does_not_promote_a_fit_to_anchor_when_the_anchor_was_unstable():
    m = means()
    _, st = step(None, summary(m, seed=1, fit_unstable=True), CFG, REGISTRY, lineage_id="t")
    r, new = safe_step(st, summary(m, seed=2, k_change=crash_record()))
    assert_failed_closed(r)
    assert new.anchor is st.anchor
    r, new = safe_step(st, summary(m, seed=3))                                  # a healthy stable fit still takes over an unstable anchor (WP9 A1)
    assert r.continuity_state == CONTINUITY_UNVERIFIED and new.anchor is not st.anchor


def test_t11d_the_duration_weight_without_durations_stays_a_loud_configuration_error():
    m, _, st = started()
    with pytest.raises(ValueError):
        step(st, summary(m, seed=2), AlignmentConfig(min_separation=1.0, duration_weight=1.0, ladder=LadderConfig(n_bootstrap=40, min_gap=0.5)), REGISTRY)


# ------------------------------------------------------------------------------------------------ T12, T13, T14
def test_t12_a_healthy_fit_and_a_record_that_was_never_asked_for_give_no_alert_and_no_crash_reason(fresh_alerts):
    m, _, st = started()
    for kc in (None, OK_RECORD, dict(OK_RECORD, reason=None), dict(OK_RECORD, reason="reference_fit_unavailable"), dict(OK_RECORD, reason="no_anchor"), dict(OK_RECORD, reason="anchor_has_no_window")):
        r, _ = step(st, summary(m, seed=2, k_change=kc, **ids()), CFG, REGISTRY)
        assert r.continuity_state == ALIGNED and "assessment_crashed" not in r.reasons, kc
    assert fresh_alerts == []
    assert L.k_change_crash(summary(m, seed=2)) is None and L.k_change_crash(summary(m, seed=2, k_change=OK_RECORD)) is None


def test_t13_the_first_fit_carries_the_reason_and_alerts_and_still_starts_the_lineage(fresh_alerts):
    m = means()
    rec, st = step(None, summary(m, seed=1, assessment_crashes=(("fit_stability", "KeyError"),), **ids()), CFG, REGISTRY, lineage_id="t")
    assert rec.continuity_state == NEW_LINEAGE and rec.reasons == ("first_fit", "assessment_crashed") and st.anchor is not None
    assert [e["stage"] for e in fresh_alerts] == ["fit_stability"]


def test_t14_a_crash_without_a_user_or_a_revision_is_never_suppressed(fresh_alerts):
    m, _, st = started()
    for _ in range(3):
        step(st, summary(m, seed=2, k_change=crash_record()), CFG, REGISTRY)
    assert len(fresh_alerts) == 3
    for kw in (dict(user_id="u-1"), dict(data_revision_id="r-1")):
        for _ in range(2):
            step(st, summary(m, seed=2, k_change=crash_record(), **kw), CFG, REGISTRY)
    assert len(fresh_alerts) == 7
    n0 = len(fresh_alerts)
    step(st, summary(m, seed=2, assessment_crashes=(("scaler", "KeyError"), ("scaler", "ValueError"))), CFG, REGISTRY)           # two crashes of one stage in one step: one alert
    assert len(fresh_alerts) == n0 + 1


def test_the_new_summary_fields_default_to_nothing_and_a_summary_without_them_still_works():
    s = summary(means(), seed=1)
    assert s.assessment_crashes == () and s.user_id is None and s.data_revision_id is None
    assert L.FitSummary.__dataclass_fields__["assessment_crashes"].default == ()


def test_the_record_does_not_carry_the_user_or_the_revision():
    m, _, st = started()
    r, _ = step(st, summary(m, seed=2, k_change=crash_record(), **ids(user="u-secret", rev="rev-secret")), CFG, REGISTRY)
    blob = repr(r)
    assert "u-secret" not in blob and "rev-secret" not in blob
