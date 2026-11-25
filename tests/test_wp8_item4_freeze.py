"""WP8 Item 4, design freeze (2026-11-25): the frozen defaults of the lineage, and the fail-closed handling of the classes whose threshold the development data could not calibrate.
The numbers come from `wp8_evidence/item4_thresholds.json` (written by `integration_report/wp8/scripts/item4_analyze.py` from the 960 development fits); these tests tie the code to that file."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from backbone.hssm import lineage as L
from backbone.hssm.lineage import (ALIGNED, CONTINUITY_UNVERIFIED, AlignmentConfig, PairThreshold, make_summary, may_claim_same_as_before, step)
from backbone.hssm.ordering import LadderConfig
from data.synthetic_identity import REGISTRY, SCALE, IdentitySpec, draw_user

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "wp8_evidence" / "item4_thresholds.json"
LADDER = LadderConfig(n_bootstrap=40, min_gap=0.5)                  # test speed only: the freeze concerns the alignment defaults, not the ladder
DEFAULTS = AlignmentConfig()
CAL = PairThreshold("ratio", rho=0.5)                                # a threshold calibrated BY THE TEST, to show that the reason a pair is not served is the missing calibration


def _means(seed, sep=6.0):
    return draw_user(seed, IdentitySpec(n_regimes=3, sep_lo=sep, sep_hi=sep, offset_sd=0.3))["means"]


def _summary(means, raw_perm=None, engine="baseline", seed=0, T=240, noise=0.25, scaler=None):
    rng = np.random.default_rng(seed)
    K, F = means.shape
    raw_perm = list(range(K)) if raw_perm is None else list(raw_perm)
    fit_means = means[raw_perm]
    labels = rng.permutation(np.repeat(np.arange(K), T // K)[:T])
    window = fit_means[labels] + noise * SCALE * rng.standard_normal((len(labels), F))
    mean, std = (window.mean(axis=0), window.std(axis=0)) if scaler is None else scaler
    return make_summary(engine, fit_means, mean, std, labels, window)


def _cfg(**kw):
    return replace(DEFAULTS, ladder=LADDER, **kw)


def _first(engine="baseline", seed=4, cfg=None):
    a = _summary(_means(seed), engine=engine, seed=1)
    _, st = step(None, a, _cfg() if cfg is None else cfg, REGISTRY, lineage_id="u")
    return a, st


# ------------------------------------------------------------------------------------------------------------------- the frozen numbers are the ones of the evidence file
def test_the_frozen_defaults_are_pinned():
    c = AlignmentConfig()
    assert c.min_separation == 4.0 and c.duration_weight == 0.0
    assert c.same_engine == PairThreshold("ratio", 0.0, 0.5) and c.same_engine.calibrated is True
    assert c.dual_engine.calibrated is False and c.cross_engine.calibrated is False
    assert (c.dual_engine.rule, c.dual_engine.rho) == ("ratio", 0.5) and (c.cross_engine.rule, c.cross_engine.rho) == ("ratio", 0.5)


def test_the_code_defaults_equal_the_development_evidence():
    ev = json.loads(EVIDENCE.read_text())
    c = AlignmentConfig()
    assert c.min_separation == ev["h"]
    base = ev["classes"]["baseline"]
    assert not base["unknown"] and c.same_engine.calibrated and c.same_engine.rule == {"a": "diff", "b": "ratio", "c": "both"}[base["rule"]]
    assert c.same_engine.rho == base["rho"] and base["tau"] is None and c.same_engine.tau == 0.0
    assert c.duration_weight == 0.0 and base["distance"] == "raw"
    assert ev["classes"]["dual"]["unknown"] is True and c.dual_engine.calibrated is False and ev["classes"]["dual"]["n_passing"] < 20
    assert ev["classes"]["cross"]["unknown"] is True and c.cross_engine.calibrated is False and ev["classes"]["cross"]["n_passing"] < 20


# ------------------------------------------------------------------------------------------------------------------- uncalibrated thresholds accept nothing
def test_an_uncalibrated_threshold_accepts_no_margin_however_good():
    perfect = dict(C1=0.0, C2=100.0, margin_diff=33.0, margin_ratio=0.0)
    assert L.margin_ok(CAL, perfect) and not L.margin_ok(PairThreshold("ratio", rho=0.5, calibrated=False), perfect)
    only_one = dict(C1=0.0, C2=float("inf"), margin_diff=0.0, margin_ratio=0.0)                       # a single regime is unambiguous but still a class that is not calibrated
    assert L.margin_ok(CAL, only_one) and not L.margin_ok(PairThreshold("ratio", rho=0.5, calibrated=False), only_one)
    for rule in ("diff", "ratio", "both"):
        assert not L.margin_ok(PairThreshold(rule, tau=0.0, rho=0.5, calibrated=False), perfect)


def test_a_clear_baseline_pair_is_aligned_with_the_defaults():
    a, st = _first()
    assert a.separation_min >= DEFAULTS.min_separation
    n = _summary(_means(4), raw_perm=[2, 0, 1], seed=2, scaler=(a.scaler_mean, a.scaler_std))
    rec, _ = step(st, n, _cfg(), REGISTRY)
    assert rec.continuity_state == ALIGNED and may_claim_same_as_before(rec) and rec.threshold["calibrated"] is True and rec.reasons == ()


def test_a_clear_dual_pair_is_not_aligned_with_the_defaults_but_is_with_a_calibrated_threshold():
    a, st = _first(engine="dual")
    n = _summary(_means(4), raw_perm=[2, 0, 1], engine="dual", seed=2, scaler=(a.scaler_mean, a.scaler_std))
    rec, st1 = step(st, n, _cfg(), REGISTRY)
    assert rec.continuity_state == CONTINUITY_UNVERIFIED and not may_claim_same_as_before(rec) and rec.reasons == ("threshold_not_calibrated",)
    assert rec.threshold["calibrated"] is False and st1.anchor.summary is a                                # the claim is withheld and the anchor does not move
    assert rec.message.startswith("cannot confirm this is the same as before")
    control, _ = step(st, n, _cfg(dual_engine=CAL), REGISTRY)
    assert control.continuity_state == ALIGNED and control.reasons == ()


@pytest.mark.parametrize("first_engine, second_engine", [("baseline", "dual"), ("dual", "baseline")])
def test_a_clear_engine_switch_is_not_aligned_with_the_defaults_but_is_with_a_calibrated_threshold(first_engine, second_engine):
    a, st = _first(engine=first_engine)
    n = _summary(_means(4), raw_perm=[1, 2, 0], engine=second_engine, seed=2, scaler=(a.scaler_mean, a.scaler_std))
    rec, _ = step(st, n, _cfg(), REGISTRY)
    assert rec.continuity_state == CONTINUITY_UNVERIFIED and rec.reasons == ("threshold_not_calibrated",) and rec.anchor_engine == first_engine and not may_claim_same_as_before(rec)
    control, _ = step(st, n, _cfg(cross_engine=CAL), REGISTRY)
    assert control.continuity_state == ALIGNED


def test_the_old_names_are_kept_and_the_best_guess_is_recorded_on_an_uncalibrated_pair():
    a, st = _first(engine="dual")
    perm = [2, 0, 1]
    n = _summary(_means(4), raw_perm=perm, engine="dual", seed=2, scaler=(a.scaler_mean, a.scaler_std))
    rec, _ = step(st, n, _cfg(), REGISTRY)
    best, _ = step(st, n, _cfg(dual_engine=CAL), REGISTRY)
    assert rec.raw_to_stable == best.raw_to_stable and rec.margin_ratio == best.margin_ratio     # the withheld claim still carries the margins and the best guess
    assert rec.proposed_new_names is not None


# ------------------------------------------------------------------------------------------------------------------- the gate h and the class of pair
@pytest.mark.parametrize("sep, flagged", [(3.99, True), (4.0, False), (4.01, False), (0.0, True)])
def test_the_identifiability_gate_is_at_h_equal_4(sep, flagged):
    a, st = _first()
    n = replace(_summary(_means(4), raw_perm=[1, 2, 0], seed=2, scaler=(a.scaler_mean, a.scaler_std)), separation_min=sep)
    rec, _ = step(st, n, _cfg(), REGISTRY)
    assert ("fit_not_identifiable" in rec.reasons) is flagged and (rec.continuity_state == CONTINUITY_UNVERIFIED) is flagged


def test_the_anchor_is_gated_with_the_same_h():
    a = replace(_summary(_means(4), seed=1), separation_min=3.5)
    _, st = step(None, a, _cfg(), REGISTRY, lineage_id="u")
    n = _summary(_means(4), raw_perm=[1, 2, 0], seed=2, scaler=(a.scaler_mean, a.scaler_std))
    rec, _ = step(st, n, _cfg(), REGISTRY)
    assert "anchor_not_identifiable" in rec.reasons and rec.continuity_state == CONTINUITY_UNVERIFIED


def test_each_class_of_pair_uses_its_own_threshold():
    cfg = _cfg(same_engine=PairThreshold("ratio", rho=0.11), dual_engine=PairThreshold("ratio", rho=0.22), cross_engine=PairThreshold("ratio", rho=0.33))
    sc = None
    for first, second, expected in (("baseline", "baseline", 0.11), ("dual", "dual", 0.22), ("baseline", "dual", 0.33), ("dual", "baseline", 0.33)):
        a, st = _first(engine=first, cfg=cfg)
        n = _summary(_means(4), raw_perm=[1, 2, 0], engine=second, seed=2, scaler=(a.scaler_mean, a.scaler_std))
        rec, _ = step(st, n, cfg, REGISTRY)
        assert rec.threshold["rho"] == expected and rec.threshold["calibrated"] is True, (first, second)


def test_the_record_threshold_dict_stays_json_ready_and_additive():
    a, st = _first()
    n = _summary(_means(4), raw_perm=[1, 2, 0], seed=2, scaler=(a.scaler_mean, a.scaler_std))
    rec, _ = step(st, n, _cfg(), REGISTRY)
    d = json.loads(json.dumps(rec.as_dict()))
    assert {"rule", "tau", "rho"} <= set(d["threshold"]) and d["threshold"]["calibrated"] is True                 # the old keys are still there
