"""WP8 item 2: the identity-stream generator (data/synthetic_identity.py) and the feature registry. Design: tests/PREDECLARED_CRITERIA_WP8.md section 3.
Every test has an opposite case; mutation check: integration_report/wp8/scripts/mutation_item2_generator.py."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from backbone.hssm.feature_registry import FeatureSpec, validate_registry
from data.synthetic_identity import BASE, COVERAGE, REGISTRY, SCALE, IdentitySpec, draw_user, generate_identity_stream


# ------------------------------------------------------------------------------------------------------------------- registry
def test_registry_accepts_a_valid_feature_and_rejects_each_invalid_field():
    FeatureSpec("steps", "steps/day", "movement", "movement", 1)
    for bad in (dict(name="", unit="u"), dict(name="a", unit=""), dict(source="radar"), dict(role="mood"), dict(direction=2), dict(direction=-2)):
        kw = dict(name="a", unit="u", source="movement", role="movement", direction=0); kw.update(bad)
        with pytest.raises(ValueError):
            FeatureSpec(**kw)


def test_validate_registry_checks_unique_names_and_width():
    assert validate_registry(REGISTRY, 7) == REGISTRY
    with pytest.raises(ValueError):
        validate_registry(REGISTRY + (REGISTRY[0],))                    # duplicate name
    with pytest.raises(ValueError):
        validate_registry(REGISTRY, 6)                                  # model input has another width


def test_the_registry_has_units_sources_roles_and_the_declared_directions():
    assert [f.name for f in REGISTRY] == ["steps", "time_moving", "mean_hr", "hrv", "talking", "pitch", "sleep"]
    assert [f.direction for f in REGISTRY] == [1, 1, 1, -1, 1, 0, -1]
    assert [f.role for f in REGISTRY][:2] == ["movement", "time_moving"]
    assert [f.unit for f in REGISTRY] == ["steps/day", "min/day", "bpm", "ms", "min/day", "Hz", "h"]
    assert [f.source for f in REGISTRY] == ["movement", "movement", "heart_rate", "heart_rate", "voice", "voice", "other"]


# ------------------------------------------------------------------------------------------------------------------- determinism and independence
def test_same_seed_same_stream_and_other_seed_other_stream():
    a, b, c = generate_identity_stream(8101), generate_identity_stream(8101), generate_identity_stream(8102)
    assert np.array_equal(a.X_obs, b.X_obs, equal_nan=True) and np.array_equal(a.labels, b.labels)
    assert not np.array_equal(a.labels, c.labels) or not np.allclose(a.X_full, c.X_full)


def test_the_user_does_not_change_with_stream_length_or_drift_and_drift_does_not_change_the_labels():
    base = generate_identity_stream(8101, n_days=150)
    longer = generate_identity_stream(8101, n_days=200)
    drift = generate_identity_stream(8101, n_days=150, drift_sd=2.5)
    for k in ("means", "coverage", "sep", "loading"):
        assert np.array_equal(base.truth["user"][k], longer.truth["user"][k]) and np.array_equal(base.truth["user"][k], drift.truth["user"][k])
    assert np.array_equal(base.labels, drift.labels)                                 # drift moves means only
    assert np.array_equal(base.labels, longer.labels[:150])                         # a longer stream continues the same path
    assert np.allclose(base.X_full[0], drift.X_full[0]) and not np.allclose(base.X_full[-1], drift.X_full[-1])


def test_shapes_labels_and_dwell_lengths():
    s = generate_identity_stream(8105, n_days=150)
    assert s.X_obs.shape == s.X_full.shape == (150, 7) and s.labels.shape == (150,) and set(s.labels) <= {0, 1, 2}
    runs = np.diff(np.r_[0, np.flatnonzero(np.diff(s.labels)) + 1, 150])
    assert runs[:-1].min() >= 5 and runs.max() <= 30                                # the last run is cut by the stream end
    assert all(s.labels[i] != s.labels[i + 1] for i in np.flatnonzero(np.diff(s.labels) != 0))


def test_dwell_clipping_holds_even_when_the_dwell_distribution_is_very_wide():
    wide = IdentitySpec(dwell_log_sd=2.0)                                            # without the clip, runs of 1 day and of 100+ days would be common
    for seed in (8101, 8102, 8103):
        s = generate_identity_stream(seed, n_days=600, spec=wide)
        runs = np.diff(np.r_[0, np.flatnonzero(np.diff(s.labels)) + 1, 600])
        assert runs[:-1].min() >= 5 and runs.max() <= 30
    assert (runs[:-1] == 5).any() or (np.diff(np.r_[0, np.flatnonzero(np.diff(generate_identity_stream(8101, n_days=600, spec=wide).labels)) + 1, 600])[:-1] == 30).any()


# ------------------------------------------------------------------------------------------------------------------- planted activity order
def test_activity_levels_are_ordered_and_features_follow_their_registry_direction():
    spec = IdentitySpec(offset_sd=0.0)                                              # without non-activity offsets the order is exact
    for seed in (8101, 8102, 8103):
        u = draw_user(seed, spec)
        assert np.all(np.diff(u["alpha"]) > 0) and np.isclose(u["alpha"][-1], u["sep"])
        m = u["means"]
        assert np.all(np.diff(m[:, 0]) > 0) and np.all(np.diff(m[:, 1]) > 0)         # steps and time moving rise with activity
        assert np.all(np.diff(m[:, 3]) < 0) and np.all(np.diff(m[:, 6]) < 0)         # hrv and sleep fall
        assert np.allclose(m[:, 5], m[0, 5])                                         # pitch has no activity direction: no offsets, so the regimes coincide on it


def test_offsets_make_regimes_differ_in_non_activity_directions():
    u = draw_user(8101, IdentitySpec())
    assert not np.allclose(u["means"][:, 5], u["means"][0, 5])                      # pitch differs between regimes once offsets are on


# ------------------------------------------------------------------------------------------------------------------- the model input (window)
def test_window_is_z_scored_per_feature_with_zero_fill_and_nan_rows_for_missing_days():
    s = generate_identity_stream(8110, n_days=400)
    w = s.window(0, 400)
    z = w["matrix"]
    day_missing = np.isnan(s.X_obs).all(axis=1)
    assert np.array_equal(np.isnan(z).all(axis=1), day_missing) and not np.isnan(z[~day_missing]).any()
    observed = ~np.isnan(s.X_obs[~day_missing])
    for f in range(7):
        col = z[~day_missing][:, f]
        assert np.isclose(col[observed[:, f]].mean(), 0.0, atol=1e-9) and np.isclose(col[observed[:, f]].std(), 1.0, atol=1e-9)
        assert np.all(col[~observed[:, f]] == 0.0)                                  # a missing single feature is filled with the person's mean
    assert np.allclose(w["scaler_mean"], np.nanmean(s.X_obs, axis=0)) and np.allclose(w["scaler_std"], np.nanstd(s.X_obs, axis=0))
    assert w["labels"].shape == (400,) and w["coverage"].shape == (7,)


def test_window_scaler_depends_on_the_window():
    s = generate_identity_stream(8110, n_days=150)
    a, b = s.window(0, 90), s.window(60, 150)
    assert not np.allclose(a["scaler_mean"], b["scaler_mean"]) and not np.allclose(a["scaler_std"], b["scaler_std"])


def test_coverage_matches_the_user_coverage_on_a_long_stream():
    s = generate_identity_stream(8111, n_days=6000)
    present = (~np.isnan(s.X_obs[~np.isnan(s.X_obs).all(axis=1)])).mean(axis=0)
    assert np.allclose(present, s.truth["user"]["coverage"], atol=0.03)
    assert abs(np.isnan(s.X_obs).all(axis=1).mean() - 0.03) < 0.01                  # whole missing days


def test_day_to_day_noise_has_the_declared_persistence_and_scale():
    s = generate_identity_stream(8112, n_days=6000)
    e = (s.X_full - s.truth["means_t"]) / s.truth["within_sd"]
    assert np.allclose(e.std(axis=0), 1.0, atol=0.06)
    lag1 = np.array([np.corrcoef(e[:-1, f], e[1:, f])[0, 1] for f in range(7)])
    assert np.allclose(lag1, 0.4, atol=0.06)


# ------------------------------------------------------------------------------------------------------------------- drift, split, merge
def test_drift_moves_a_regime_by_the_declared_number_of_within_sd_over_the_stream():
    s = generate_identity_stream(8101, n_days=150, drift_sd=2.5)
    for k in range(3):
        idx = np.flatnonzero(s.labels == k)
        i0, i1 = idx[0], idx[-1]
        d = (s.truth["means_t"][i1] - s.truth["means_t"][i0]) / s.truth["within_sd"]
        assert np.isclose(np.linalg.norm(d), 2.5 * (i1 - i0) / 149, rtol=1e-6)
    still = generate_identity_stream(8101, n_days=150)
    assert np.allclose(still.truth["means_t"][s.labels == 0], still.truth["means_t"][s.labels == 0][0])           # no drift: constant means


@pytest.mark.parametrize("seed", range(8101, 8109))
def test_split_creates_a_new_regime_only_after_the_change_day_two_sd_from_its_parent(seed):
    s = generate_identity_stream(seed, n_days=150, k_change="split", change_day=90)
    u = s.truth["user"]
    child = s.labels == 3
    assert s.truth["K_after"] == 4 and np.all(s.labels[:90] < 3)                     # nothing is split before the change day
    if child.any():
        assert np.flatnonzero(child).min() >= 90
        off = (s.truth["means_t"][child] - u["means"][u["split_parent"]]) / s.truth["within_sd"]
        assert np.allclose(np.linalg.norm(off, axis=1), 2.0, atol=1e-9)
    assert not (generate_identity_stream(seed, n_days=150).labels == 3).any()       # opposite case: no k_change, no new regime


def test_some_stream_does_split_and_the_split_regime_comes_from_the_parent_segments():
    n_child = [int((generate_identity_stream(sd, n_days=150, k_change="split").labels == 3).sum()) for sd in range(8101, 8109)]
    assert max(n_child) > 0
    s = generate_identity_stream(8101, n_days=150, k_change="split"); u = s.truth["user"]
    base = generate_identity_stream(8101, n_days=150)
    child = s.labels == 3
    assert np.all(base.labels[child] == u["split_parent"])                           # the new regime only takes days of the parent
    assert np.array_equal(s.labels[~child], base.labels[~child])                     # every other day keeps its planted regime


def test_merge_makes_one_regime_equal_to_another_after_the_change_day_only():
    s = generate_identity_stream(8101, n_days=150, k_change="merge", change_day=90)
    u = s.truth["user"]; a, b = u["merge_pair"]
    after = (s.labels == b) & (np.arange(150) >= 90); before = (s.labels == b) & (np.arange(150) < 90)
    assert s.truth["K_after"] == 2
    if after.any():
        assert np.allclose(s.truth["means_t"][after], u["means"][a])
    if before.any():
        assert np.allclose(s.truth["means_t"][before], u["means"][b]) and not np.allclose(u["means"][a], u["means"][b])
