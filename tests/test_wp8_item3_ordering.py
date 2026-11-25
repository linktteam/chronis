"""WP8 item 3: the ordering score ladder (backbone/hssm/ordering.py). Design: tests/PREDECLARED_CRITERIA_WP8.md section 4.2.
Hand-built windows with a known planted activity order; an independent re-implementation of the weights and scores is the oracle; every behaviour has an opposite case.
Mutation check: integration_report/wp8/scripts/mutation_item3.py."""
from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from backbone.hssm.feature_registry import FeatureSpec
from backbone.hssm.ordering import LadderConfig, RUNG_ROLES, order_regimes
from data.synthetic_identity import BASE, REGISTRY, SCALE

FAST = LadderConfig(n_bootstrap=60)
DIR = np.array([f.direction for f in REGISTRY], dtype=float)
LOAD_ALL = np.array([1.0, 0.9, 0.6, 0.5, 0.3, 0.0, 0.2])
NOISE_ALL = np.array([0.35, 0.45, 0.7, 0.9, 1.0, 1.1, 0.8])


def _window(seed=0, K=3, T=150, sep=2.0, load=LOAD_ALL, noise=NOISE_ALL, direction=DIR, missing=None, alpha=None):
    """Raw features for regimes with activity levels alpha (default evenly spaced, planted order = index order); labels in blocks of 6..12 days, every regime present."""
    rng = np.random.default_rng(seed)
    labels = []
    cur = 0
    while len(labels) < T:
        labels += [cur] * int(rng.integers(6, 13)); cur = (cur + int(rng.integers(1, K))) % K
    labels = np.array(labels[:T])
    a = np.linspace(-1, 1, K) * sep if alpha is None else np.asarray(alpha, dtype=float)
    raw = BASE + SCALE * (direction * load * a[labels][:, None] + noise * rng.standard_normal((T, len(BASE))))
    if missing is not None:
        raw = np.where(rng.random(raw.shape) < missing, np.nan, raw)
    return raw, labels


def _oracle(raw, labels, K, registry, rung):
    """Independent implementation of the declared weights and scores."""
    roles = RUNG_ROLES[rung]
    obs = ~np.isnan(raw)
    mean_f = np.nanmean(raw, axis=0); sd_f = np.nanstd(raw, axis=0)
    w = np.zeros(raw.shape[1]); s = np.zeros(raw.shape[1])
    for f, spec in enumerate(registry):
        if spec.direction == 0 or (roles is not None and spec.role not in roles):
            continue
        groups = [raw[(labels == k) & obs[:, f], f] for k in range(K)]
        n = np.array([len(g) for g in groups], dtype=float); m = np.array([g.mean() for g in groups])
        within = sum(((g - g.mean()) ** 2).sum() for g in groups) / n.sum()
        grand = (n * m).sum() / n.sum(); between = (n * (m - grand) ** 2).sum() / n.sum()
        snr = between / within
        w[f] = obs[:, f].mean() * snr / (1 + snr); s[f] = spec.direction
    w = w / w.sum()
    z = np.array([[(raw[(labels == k) & obs[:, f], f].mean() - mean_f[f]) / sd_f[f] for f in range(raw.shape[1])] for k in range(K)])
    return w, (z * (w * s)).sum(axis=1)


# ------------------------------------------------------------------------------------------------------------------- weights and scores (oracle)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_weights_and_scores_equal_an_independent_computation_for_every_rung(seed):
    raw, labels = _window(seed, missing=0.1)
    res = order_regimes(raw, labels, 3, REGISTRY, FAST)
    for rung in (1, 2, 3):
        w, score = _oracle(raw, labels, 3, REGISTRY, rung)
        got = res.rungs[rung]
        assert np.allclose([got.weights.get(f.name, 0.0) for f in REGISTRY], w, atol=1e-12)
        assert np.allclose(got.scores, score, atol=1e-12)
        assert got.order == tuple(np.argsort(score, kind="stable"))
        assert np.isclose(got.min_gap, np.diff(np.sort(score)).min())


def test_weights_sum_to_one_and_features_without_a_direction_or_outside_the_rung_get_none():
    raw, labels = _window(4)
    res = order_regimes(raw, labels, 3, REGISTRY, FAST)
    for rung in (1, 2, 3):
        assert np.isclose(sum(res.rungs[rung].weights.values()), 1.0)
        assert "pitch" not in res.rungs[rung].weights
    assert set(res.rungs[2].weights) <= {"steps", "time_moving"} and set(res.rungs[3].weights) == {"steps"}
    assert {"mean_hr", "hrv", "talking", "sleep"} & set(res.rungs[1].weights)


def test_a_more_reliable_feature_gets_a_larger_weight_and_so_does_a_better_covered_one():
    noise_hi = NOISE_ALL.copy(); noise_hi[1] = 3.0                                 # time_moving made very noisy
    raw, labels = _window(5, noise=noise_hi)
    w = order_regimes(raw, labels, 3, REGISTRY, FAST).rungs[2].weights
    assert w["steps"] > w["time_moving"]
    clean, labels = _window(6, noise=np.full(7, 0.3), load=np.ones(7))              # equal SNR everywhere: coverage decides
    clean = clean.copy(); rng = np.random.default_rng(0); clean[rng.random(len(clean)) < 0.5, 1] = np.nan     # time_moving observed on half the days
    w2 = order_regimes(clean, labels, 3, REGISTRY, FAST).rungs[2].weights
    assert w2["steps"] > w2["time_moving"]


def test_the_weights_do_not_depend_on_the_regime_names_or_the_day_order():
    raw, labels = _window(7)
    base = order_regimes(raw, labels, 3, REGISTRY, FAST)
    perm = np.array([2, 0, 1])                                                       # rename the regimes
    renamed = order_regimes(raw, perm[labels], 3, REGISTRY, FAST)
    for r in (1, 2, 3):
        assert base.rungs[r].weights == pytest.approx(renamed.rungs[r].weights)
        assert renamed.rungs[r].order == tuple(int(perm[o]) for o in base.rungs[r].order)      # the same physical ordering under the new names
    idx = np.random.default_rng(1).permutation(len(raw))                            # shuffle the days jointly: point estimates unchanged
    shuffled = order_regimes(raw[idx], labels[idx], 3, REGISTRY, FAST)
    for r in (1, 2, 3):
        assert shuffled.rungs[r].scores == pytest.approx(base.rungs[r].scores) and shuffled.rungs[r].order == base.rungs[r].order


# ------------------------------------------------------------------------------------------------------------------- the ladder
def test_clear_data_is_verified_at_rung_1_and_gives_the_planted_order():
    raw, labels = _window(8, sep=2.0)
    res = order_regimes(raw, labels, 3, REGISTRY, FAST)
    assert res.verified and res.activity_rung == 1 and res.order == (0, 1, 2) and "rung 1" in res.reason
    renamed = order_regimes(raw, np.array([1, 2, 0])[labels], 3, REGISTRY, FAST)     # planted regime 0 is now called 1, 1 is 2, 2 is 0
    assert renamed.verified and renamed.order == (1, 2, 0)


def test_flipping_every_registry_direction_reverses_the_order():
    raw, labels = _window(9)
    flipped = tuple(FeatureSpec(f.name, f.unit, f.source, f.role, -f.direction) for f in REGISTRY)
    assert order_regimes(raw, labels, 3, flipped, FAST).order == (2, 1, 0)


def test_no_activity_signal_means_unverified_first_fit_order_and_no_rung():
    raw, labels = _window(10, load=np.zeros(7))                                      # regimes do not differ in activity at all
    res = order_regimes(raw, labels, 3, REGISTRY, FAST)
    assert not res.verified and res.activity_rung == 0 and res.order == (0, 1, 2) and "first-fit" in res.reason
    renamed = order_regimes(raw, np.array([2, 1, 0])[labels], 3, REGISTRY, FAST)
    assert renamed.order == (0, 1, 2)                                                # first-fit order is the index order, whatever the names


def test_conflicting_reliable_rungs_are_unverified_not_served():
    flip = DIR.copy(); flip[[2, 3, 4, 6]] *= -1                                      # four non-movement features move AGAINST their registry direction
    raw, labels = _window(11, sep=3.0, direction=flip, noise=np.full(7, 0.25), load=np.array([1, 1, 1, 1, 1, 0.0, 1]))
    res = order_regimes(raw, labels, 3, REGISTRY, LadderConfig(n_bootstrap=60, min_gap=0.2))
    assert res.rungs[3].reliable and res.rungs[2].reliable and res.rungs[1].reliable
    assert res.rungs[1].order != res.rungs[3].order                                  # the misleading features reverse the rung-1 order
    assert not res.verified and res.activity_rung == 0 and "disagree" in res.reason and res.order == (0, 1, 2)


def test_the_highest_reliable_rung_names_the_regimes_when_rung_1_is_not_reliable():
    flip = DIR.copy(); flip[[2, 3, 4, 6]] *= -1                                      # the other features pull against movement, so rung 1 has a small gap
    raw, labels = _window(11, sep=3.0, direction=flip, noise=np.full(7, 0.25), load=np.array([1, 1, 1, 1, 1, 0.0, 1]))
    res = order_regimes(raw, labels, 3, REGISTRY, LadderConfig(n_bootstrap=60, min_gap=0.8))
    assert not res.rungs[1].reliable and res.rungs[2].reliable and res.rungs[3].reliable
    assert res.verified and res.activity_rung == 2 and res.order == (0, 1, 2)         # rung 2 is the highest reliable rung; rung 3 agrees


def test_the_gap_condition_is_what_makes_close_regimes_unverified():
    raw, labels = _window(13, T=3000, alpha=[-0.1, 0.0, 0.1], noise=np.full(7, 1.0))   # a long window: a small difference (0.1 sd) is stable but tiny
    tight = order_regimes(raw, labels, 3, REGISTRY, LadderConfig(n_bootstrap=60, min_gap=0.5))
    assert tight.rungs[3].modal_freq >= 0.9 and tight.rungs[3].min_gap < 0.5 and not tight.rungs[3].reliable and not tight.verified
    loose = order_regimes(raw, labels, 3, REGISTRY, LadderConfig(n_bootstrap=60, min_gap=0.0))
    assert loose.rungs[3].reliable and loose.verified                                # the same data pass when the gap condition is off


def test_the_frequency_condition_is_what_makes_unstable_orderings_unverified():
    raw, labels = _window(15, sep=0.25, noise=np.full(7, 1.5), load=np.ones(7))      # noisy: the order flips between resamples
    strict = order_regimes(raw, labels, 3, REGISTRY, LadderConfig(n_bootstrap=120, min_gap=0.0, min_modal_freq=0.95))
    lax = order_regimes(raw, labels, 3, REGISTRY, LadderConfig(n_bootstrap=120, min_gap=0.0, min_modal_freq=0.51))
    assert strict.rungs[3].modal_freq < 0.95 and strict.rungs[3].reliable is False and not strict.verified
    assert lax.rungs[3].modal_freq == strict.rungs[3].modal_freq                     # same bootstrap, only the threshold differs
    assert lax.rungs[3].reliable is True


def test_a_point_estimate_order_that_is_not_the_modal_bootstrap_order_is_not_reliable():
    raw, labels = _window(100, T=90, sep=0.1, noise=np.full(7, 1.2), load=np.ones(7))
    r = order_regimes(raw, labels, 3, REGISTRY, LadderConfig(n_bootstrap=40, min_gap=0.0, min_modal_freq=0.51, seed=0)).rungs[3]
    assert r.modal_freq >= 0.51 and r.min_gap >= 0.0 and r.modal_order != r.order    # frequency and gap alone would accept it
    assert r.reliable is False


def test_a_rung_without_usable_data_is_not_available():
    raw, labels = _window(16)
    raw = raw.copy(); raw[:, 0] = np.nan                                             # steps never observed
    res = order_regimes(raw, labels, 3, REGISTRY, FAST)
    assert res.rungs[3].available is False and res.rungs[2].available and set(res.rungs[2].weights) == {"time_moving"}
    raw2 = raw.copy(); raw2[labels == 1, 1] = np.nan                                 # time_moving never observed on regime 1: dropped from the rungs
    res2 = order_regimes(raw2, labels, 3, REGISTRY, FAST)
    assert res2.rungs[2].available is False and "time_moving" not in res2.rungs[1].weights


# ------------------------------------------------------------------------------------------------------------------- interface
def test_the_function_takes_no_outcome_or_planted_label_and_is_deterministic():
    names = list(inspect.signature(order_regimes).parameters)
    assert names == ["raw", "labels", "K", "registry", "config"]
    raw, labels = _window(17)
    a = order_regimes(raw, labels, 3, REGISTRY, FAST).as_dict(); b = order_regimes(raw, labels, 3, REGISTRY, FAST).as_dict()
    assert a == b and json.loads(json.dumps(a)) == a                                  # same input, same output, JSON-able (string keys)


def test_invalid_input_and_config_raise():
    raw, labels = _window(18)
    with pytest.raises(ValueError):
        order_regimes(raw, labels + 1, 3, REGISTRY, FAST)                             # label 3 with K = 3
    with pytest.raises(ValueError):
        order_regimes(raw, labels[:-1], 3, REGISTRY, FAST)
    with pytest.raises(ValueError):
        order_regimes(raw[:, :6], labels, 3, REGISTRY, FAST)                          # registry has 7 features
    for bad in (dict(n_bootstrap=5), dict(block_days=0), dict(min_modal_freq=0.5), dict(min_modal_freq=1.1), dict(min_gap=-0.1)):
        with pytest.raises(ValueError):
            LadderConfig(**bad)
    LadderConfig(n_bootstrap=10, block_days=1, min_modal_freq=1.0, min_gap=0.0)


def test_the_default_config_is_the_frozen_design():
    c = LadderConfig()                                                                # tests/PREDECLARED_CRITERIA_WP8.md, "Design freeze (Item 3)"
    assert (c.n_bootstrap, c.block_days, c.min_modal_freq, c.min_gap) == (200, 7, 0.90, 0.7)
    assert RUNG_ROLES == {1: None, 2: ("movement", "time_moving"), 3: ("movement",)}
