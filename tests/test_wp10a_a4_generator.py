"""WP10A A4 design D: the additive `regime_scale_change` option of data/synthetic_identity.py (default None = bit-identical to WP8/WP9; the pinned default hashes are in test_wp9_a4_generator.py)."""
import numpy as np
import pytest

from data.synthetic_identity import generate_identity_stream

SEED = 9996                                         # a WP9 seed used only for generator unit tests, no result depends on it
OPT = dict(regime=2, features=(0, 1, 2, 3), factor=3.0, start=60)


def test_the_option_none_or_empty_changes_nothing():
    a = generate_identity_stream(SEED)
    for opt in (None, {}):
        b = generate_identity_stream(SEED, regime_scale_change=opt)
        assert np.array_equal(a.X_full, b.X_full) and np.array_equal(a.X_obs, b.X_obs, equal_nan=True) and np.array_equal(a.truth["means_t"], b.truth["means_t"])


def test_factor_one_changes_nothing():
    a = generate_identity_stream(SEED)
    b = generate_identity_stream(SEED, regime_scale_change=dict(OPT, factor=1.0))
    assert np.array_equal(a.X_full, b.X_full)


def test_only_the_listed_regime_from_the_start_day_on_the_listed_features_changes():
    base, ch = generate_identity_stream(SEED), generate_identity_stream(SEED, regime_scale_change=OPT)
    assert np.array_equal(base.labels, ch.labels) and np.array_equal(np.isnan(base.X_obs), np.isnan(ch.X_obs))          # nothing else drawn
    t = np.arange(base.n_days)
    sel = (base.labels == 2) & (t >= 60)
    assert sel.sum() > 5
    changed = np.zeros(base.X_full.shape, dtype=bool)
    changed[np.ix_(sel, [0, 1, 2, 3])] = True
    assert np.array_equal(base.X_full[~changed], ch.X_full[~changed])                                                  # everything outside the selection is identical
    assert not np.allclose(base.X_full[changed], ch.X_full[changed])
    centre = base.truth["user"]["means"].mean(axis=0)
    mb, mc = base.truth["means_t"], ch.truth["means_t"]
    for f in (0, 1, 2, 3):
        assert np.allclose(mc[sel, f] - centre[f], 3.0 * (mb[sel, f] - centre[f]))                                      # the offset from the regime-average mean is scaled
        assert np.allclose(ch.X_full[sel, f] - mc[sel, f], 3.0 * (base.X_full[sel, f] - mb[sel, f]))                    # the noise is scaled by the same factor
    assert np.array_equal(mb[~changed.any(axis=1)], mc[~changed.any(axis=1)])
    assert ch.truth["regime_scale_change"]["regime"] == 2 and base.truth["regime_scale_change"] is None


def test_a_later_start_leaves_earlier_days_identical():
    base, ch = generate_identity_stream(SEED), generate_identity_stream(SEED, regime_scale_change=dict(OPT, start=100))
    assert np.array_equal(base.X_full[:100], ch.X_full[:100]) and not np.array_equal(base.X_full[100:], ch.X_full[100:])


def test_it_composes_with_the_noise_only_option():
    both = generate_identity_stream(SEED, regime_scale_change=OPT, feature_scale_change=dict(features=(5,), factor=0.5, start=60))
    only_regime = generate_identity_stream(SEED, regime_scale_change=OPT)
    only_noise = generate_identity_stream(SEED, feature_scale_change=dict(features=(5,), factor=0.5, start=60))
    assert np.array_equal(both.X_full[:, :4], only_regime.X_full[:, :4]) and np.array_equal(both.X_full[:, 5], only_noise.X_full[:, 5])


@pytest.mark.parametrize("bad", [dict(regime=2, features=(0,), factor=3.0), dict(regime=2, factor=3.0, start=1)])
def test_a_malformed_option_fails_loudly(bad):
    with pytest.raises(KeyError):
        generate_identity_stream(SEED, regime_scale_change=bad)
