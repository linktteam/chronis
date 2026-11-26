"""WP9 A2: the posterior-sharpness floor (`FIT_MIN_POSTERIOR_SHARPNESS`, a second reason for FIT_UNSTABLE) and the frozen gate default. Design: tests/PREDECLARED_CRITERIA_WP9.md section 8 and Part 8."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backbone.hssm import fit_stability as FS
from backbone.hssm import lineage as L
from backbone.hssm.config import FIT_MIN_POSTERIOR_SHARPNESS
from backbone.hssm.lineage import AlignmentConfig

ROOT = Path(__file__).resolve().parents[1]


def post(maxes):
    """A (T, 3) posterior whose row maxima are `maxes`."""
    out = np.zeros((len(maxes), 3))
    for i, m in enumerate(maxes):
        out[i] = [m, (1 - m) / 2, (1 - m) / 2]
    return out


def result(maxes, restart_summary, baseline=True):
    return SimpleNamespace(is_baseline_model=baseline, restart_summary=restart_summary, p_t_posterior=None if maxes is None else post(maxes))


STABLE = dict(fit_unstable=False, fit_unstable_reason=None)


def test_the_frozen_values_are_the_ones_in_the_freeze_file():
    frozen = json.loads((ROOT / "tests" / "DESIGN_FREEZE_WP9_A12.txt").read_text().splitlines()[0])
    assert frozen == {"h": 2.5, "s": 0.95, "u": True}
    assert AlignmentConfig().min_separation == frozen["h"] and FIT_MIN_POSTERIOR_SHARPNESS == frozen["s"]


def test_the_mean_of_the_row_maxima_is_the_sharpness():
    assert FS.posterior_sharpness(post([1.0, 0.9, 0.8])) == pytest.approx(0.9)
    assert FS.posterior_sharpness(np.array([[0.5, 0.5]])) == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [None, np.zeros((0, 3)), np.array([1.0, 0.0]), np.array([[np.nan, 0.5]]), np.array([[np.inf, 0.0]])])
def test_a_missing_or_unusable_posterior_has_no_sharpness_and_is_not_flagged_here(bad):
    assert FS.posterior_sharpness(bad) is None and FS.sharpness_unstable(bad) is False


def test_the_floor_is_strict_and_zero_switches_it_off():
    p = post([0.95, 0.95])
    assert FS.sharpness_unstable(p, 0.95) is False                                         # equal to the floor: not flagged
    assert FS.sharpness_unstable(p, 0.9500001) is True
    assert FS.sharpness_unstable(post([0.5, 0.5]), 0.0) is False and FS.sharpness_unstable(post([0.5, 0.5]), -1.0) is False
    assert FS.sharpness_unstable(post([0.94, 0.94])) is True and FS.sharpness_unstable(post([0.96, 0.96])) is False        # the default floor is the frozen 0.95


def test_a_stable_but_blurry_baseline_fit_is_flagged_with_the_sharpness_reason():
    assert L.fit_stability_flag(result([0.9] * 10, STABLE)) == (True, FS.REASON_NOT_SHARP)
    assert L.fit_stability_flag(result([0.99] * 10, STABLE)) == (False, None)


def test_the_consensus_reason_comes_first_and_a_fit_without_assessment_still_fails_closed():
    assert L.fit_stability_flag(result([0.9] * 10, dict(fit_unstable=True, fit_unstable_reason=FS.REASON_DISAGREE))) == (True, FS.REASON_DISAGREE)
    assert L.fit_stability_flag(result([0.99] * 10, {})) == (True, FS.REASON_NOT_ASSESSED)
    assert L.fit_stability_flag(result([0.99] * 10, dict(fit_unstable=None, fit_unstable_reason=None))) == (True, FS.REASON_NOT_ASSESSED)


def test_a_dual_result_is_never_flagged_by_the_floor():
    assert L.fit_stability_flag(result([0.5] * 10, STABLE, baseline=False)) == (False, None)


def _summary(means, seed=0, T=240, noise=0.35, **kw):
    from data.synthetic_identity import SCALE
    rng = np.random.default_rng(seed)
    K, F = means.shape
    labels = rng.permutation(np.repeat(np.arange(K), T // K)[:T])
    window = means[labels] + noise * SCALE * rng.standard_normal((len(labels), F))
    return L.make_summary("baseline", means, window.mean(axis=0), window.std(axis=0), labels, window, **kw)


def test_the_flag_reaches_the_lineage_as_a_withheld_alignment_and_the_unflagged_pair_is_aligned():
    from backbone.hssm.ordering import LadderConfig
    from data.synthetic_identity import REGISTRY, IdentitySpec, draw_user
    m = draw_user(3, IdentitySpec(n_regimes=3, sep_lo=2.5, sep_hi=2.5, offset_sd=0.3))["means"]
    cfg = AlignmentConfig(min_separation=1.0, ladder=LadderConfig(n_bootstrap=40, min_gap=0.5))
    _, st = L.step(None, _summary(m, seed=1), cfg, REGISTRY, lineage_id="u")
    ok, _ = L.step(st, _summary(m, seed=2), cfg, REGISTRY)
    blurry, _ = L.step(st, _summary(m, seed=2, fit_unstable=True, fit_unstable_reason=FS.REASON_NOT_SHARP), cfg, REGISTRY)
    assert ok.continuity_state == L.ALIGNED
    assert blurry.continuity_state == L.CONTINUITY_UNVERIFIED and "fit_unstable" in blurry.reasons and not L.may_claim_same_as_before(blurry)
