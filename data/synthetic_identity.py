"""WP8 identity-stream generator (synthetic; every size below is an ASSUMPTION, there are no real population data here).
Design: tests/PREDECLARED_CRITERIA_WP8.md section 3 (committed before this file ran). A stream is `n_days` of raw-unit daily features for one user with planted regimes whose
index order IS the planted activity order (regime 0 = lowest activity). Features have different reliability per user (loading, noise, coverage), regimes differ in non-activity
directions too, and the stream can drift, split or merge a regime.

The user's parameters come from `default_rng([seed, 1])` and the data stream from `default_rng(seed)` (the WP5 convention), so a user does not change when a study asks for another
stream length or drift.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backbone.hssm.feature_registry import FeatureSpec

REGISTRY: tuple[FeatureSpec, ...] = (
    FeatureSpec("steps", "steps/day", "movement", "movement", +1),
    FeatureSpec("time_moving", "min/day", "movement", "time_moving", +1),
    FeatureSpec("mean_hr", "bpm", "heart_rate", "heart_rate", +1),
    FeatureSpec("hrv", "ms", "heart_rate", "heart_rate", -1),
    FeatureSpec("talking", "min/day", "voice", "talking", +1),
    FeatureSpec("pitch", "Hz", "voice", "other", 0),
    FeatureSpec("sleep", "h", "other", "other", -1),
)
BASE = np.array([7000.0, 70.0, 72.0, 45.0, 90.0, 180.0, 7.0])       # centre of each feature in its physical unit
SCALE = np.array([2500.0, 25.0, 6.0, 12.0, 35.0, 20.0, 1.0])        # one "sd" in physical units
LOADING = np.array([1.0, 0.9, 0.6, 0.5, 0.3, 0.0, 0.2])             # how strongly activity moves the feature (in SCALE units per unit of activity)
NOISE = np.array([0.35, 0.45, 0.7, 0.9, 1.0, 1.1, 0.8])             # within-regime day-to-day sd (in SCALE units)
COVERAGE = np.array([0.98, 0.97, 0.90, 0.85, 0.75, 0.75, 0.90])     # share of days a feature is present


@dataclass(frozen=True)
class IdentitySpec:
    n_regimes: int = 3
    sep_lo: float = 0.5
    sep_hi: float = 2.0
    sd_log_loading: float = 0.25
    sd_log_noise: float = 0.25
    offset_sd: float = 0.3            # regime-specific offsets NOT along the activity order, in SCALE units
    base_shift_sd: float = 0.5
    coverage_jitter: float = 0.05
    ar: float = 0.4                   # day-to-day persistence of the noise (marginal sd stays NOISE)
    dwell_log_mean: float = float(np.log(12.0))
    dwell_log_sd: float = 0.35
    dwell_min: int = 5
    dwell_max: int = 30
    whole_day_missing: float = 0.03


@dataclass
class IdentityStream:
    seed: int
    spec: IdentitySpec
    registry: tuple
    X_full: np.ndarray              # (n_days, F) raw units, nothing missing (truth)
    X_obs: np.ndarray               # (n_days, F) raw units, NaN where a feature (or the whole day) is missing
    labels: np.ndarray              # (n_days,) planted regime per day; indices 0..K-1 in planted activity order, K = the split-off regime after a split
    truth: dict = field(default_factory=dict)

    @property
    def n_days(self) -> int:
        return self.X_obs.shape[0]

    def window(self, start: int, end: int) -> dict:
        """The model input the integration pipeline would build for days [start, end): per-person z-scoring over the window's observed days, a missing single feature
        filled with 0 (the person's mean; the declared stand-in for 'handled upstream'), a missing day left as a NaN row. Returns the matrix, the scaler and the coverage."""
        raw = self.X_obs[start:end]
        mean = np.nanmean(raw, axis=0)
        std = np.nanstd(raw, axis=0)
        std = np.where(std > 0, std, 1.0)
        z = (raw - mean) / std
        day_missing = np.isnan(raw).all(axis=1)
        z = np.where(np.isnan(z), 0.0, z)
        z[day_missing] = np.nan
        coverage = (~np.isnan(raw[~day_missing])).mean(axis=0) if (~day_missing).any() else np.zeros(raw.shape[1])
        return dict(matrix=z, scaler_mean=mean, scaler_std=std, coverage=coverage, labels=self.labels[start:end], raw=raw, start=start, end=end)


def _unit(rng, n):
    v = rng.standard_normal(n)
    return v / np.linalg.norm(v)


def draw_user(seed: int, spec: IdentitySpec) -> dict:
    K, F = spec.n_regimes, len(REGISTRY)
    rng = np.random.default_rng([int(seed), 1])
    sep = float(np.exp(rng.uniform(np.log(spec.sep_lo), np.log(spec.sep_hi))))
    loading = LOADING * np.exp(spec.sd_log_loading * rng.standard_normal(F))
    noise = NOISE * np.exp(spec.sd_log_noise * rng.standard_normal(F))
    coverage = np.clip(COVERAGE + rng.uniform(-spec.coverage_jitter, spec.coverage_jitter, F), 0.5, 1.0)
    base = BASE + SCALE * spec.base_shift_sd * rng.standard_normal(F)
    offsets = spec.offset_sd * rng.standard_normal((K, F))
    direction = np.array([f.direction for f in REGISTRY], dtype=float)
    alpha = np.linspace(-1.0, 1.0, K) * sep
    means = base + SCALE * (direction * loading * alpha[:, None] + offsets)          # (K, F) raw units
    # drift / split / merge directions are drawn here so that the user is the same whatever stream is asked for
    drift_common = _unit(rng, F)
    drift_own = np.array([_unit(rng, F) for _ in range(K)])
    split_dir = _unit(rng, F)
    split_parent, merge_pair = int(rng.integers(0, K)), tuple(int(x) for x in rng.choice(K, size=2, replace=False))
    return dict(sep=sep, loading=loading, noise=noise, coverage=coverage, base=base, offsets=offsets, alpha=alpha, means=means, direction=direction,
                drift_common=drift_common, drift_own=drift_own, split_dir=split_dir, split_parent=split_parent, merge_pair=merge_pair)


def generate_identity_stream(seed: int, n_days: int = 150, spec: IdentitySpec = IdentitySpec(), drift_sd: float = 0.0, k_change: str | None = None,
                             change_day: int = 90, feature_scale_change: dict | None = None, regime_scale_change: dict | None = None) -> IdentityStream:
    """k_change in (None, "split", "merge"). drift_sd: total displacement over the stream of every regime mean, a vector of Euclidean norm `drift_sd` in units of the
    user's within-regime sd (half common to all regimes, half regime-specific).
    WP9 A4 (additive, default None = the stream is bit-identical to WP8's): `feature_scale_change = dict(features=(i, ...), factor=f, start=d)` multiplies the within-regime NOISE of the
    listed features by `f` from day `d` on. The regime means do not move, so the raw-unit identity of the regimes is unchanged, but the within-window scale of those features is: per-window
    z-scoring changes the basis, the stored scaler does not. No random number is drawn for it, so a stream with the option differs from the default one only in those noise values.
    WP10A A4 design D (additive, default None = bit-identical): `regime_scale_change = dict(regime=k, features=(i, ...), factor=f, start=d)` multiplies, from day `d` on and only on the days of the
    planted regime `k`, BOTH the offset of that regime's mean from the regime-average mean and the within-regime noise of the listed features by `f` (a scale change that is correlated with a
    regime-mean change, confined to one regime). The planted labels, the missingness and every random number are unchanged; the regimes keep their order in raw units for f > 0."""
    K, F = spec.n_regimes, len(REGISTRY)
    u = draw_user(seed, spec)
    rng = np.random.default_rng(int(seed))
    wsd = SCALE * u["noise"]                                                          # within-regime sd per feature in physical units
    # planted regime sequence
    labels, cur = [], int(rng.integers(0, K))
    while len(labels) < n_days:
        dur = int(np.clip(rng.lognormal(spec.dwell_log_mean, spec.dwell_log_sd), spec.dwell_min, spec.dwell_max))
        labels += [cur] * dur
        cur = int(rng.choice([k for k in range(K) if k != cur]))
    labels = np.array(labels[:n_days], dtype=int)
    # a split makes some later segments of the parent regime a new regime with the index K (parent kept in truth)
    if k_change == "split":
        parent = u["split_parent"]
        seg_start = np.r_[0, np.flatnonzero(np.diff(labels)) + 1]
        seg_end = np.r_[seg_start[1:], n_days]
        flip = rng.random(len(seg_start)) < 0.5
        for s, e, fl in zip(seg_start, seg_end, flip):
            if s >= change_day and labels[s] == parent and fl:
                labels[s:e] = K
    # per-day means
    t = np.arange(n_days)
    means_t = u["means"][np.minimum(labels, K - 1)].copy()                            # (n_days, F)
    if k_change == "split":
        child = labels == K
        means_t[child] = u["means"][u["split_parent"]] + 2.0 * wsd * u["split_dir"]
    if k_change == "merge":
        a, b = u["merge_pair"]
        later_b = (labels == b) & (t >= change_day)
        means_t[later_b] = u["means"][a]
    if drift_sd:
        frac = (t / max(n_days - 1, 1))[:, None]
        for k in range(K):
            idx = np.flatnonzero(np.minimum(labels, K - 1) == k) if k_change != "split" else np.flatnonzero(labels == k)
            vec = (u["drift_common"] + u["drift_own"][k]) / np.sqrt(2.0)
            vec = vec / np.linalg.norm(vec)
            means_t[idx] += frac[idx] * drift_sd * wsd * vec
        if k_change == "split":
            idx = np.flatnonzero(labels == K)
            vec = (u["drift_common"] + u["drift_own"][u["split_parent"]]) / np.sqrt(2.0)
            means_t[idx] += frac[idx] * drift_sd * wsd * vec / np.linalg.norm(vec)
    if regime_scale_change:                                                           # WP10A A4 design D: the regime's offset from the regime-average mean is scaled (noise: below)
        centre = u["means"].mean(axis=0)
        sel = (np.minimum(labels, K - 1) == int(regime_scale_change["regime"])) & (t >= int(regime_scale_change["start"]))
        for f in regime_scale_change["features"]:
            means_t[sel, int(f)] = centre[int(f)] + float(regime_scale_change["factor"]) * (means_t[sel, int(f)] - centre[int(f)])
    # AR(1) noise with the marginal sd `wsd`
    e = np.zeros((n_days, F))
    eps = rng.standard_normal((n_days, F))
    e[0] = eps[0]
    c = np.sqrt(1.0 - spec.ar ** 2)
    for i in range(1, n_days):
        e[i] = spec.ar * e[i - 1] + c * eps[i]
    gain = None
    if feature_scale_change:
        gain = np.ones((n_days, F))
        for f in feature_scale_change["features"]:
            gain[np.arange(n_days) >= int(feature_scale_change["start"]), int(f)] = float(feature_scale_change["factor"])
    if regime_scale_change:
        gain = np.ones((n_days, F)) if gain is None else gain
        sel = (np.minimum(labels, K - 1) == int(regime_scale_change["regime"])) & (t >= int(regime_scale_change["start"]))
        for f in regime_scale_change["features"]:
            gain[sel, int(f)] *= float(regime_scale_change["factor"])
    if gain is not None:
        e = e * gain
    X_full = means_t + wsd * e
    present = rng.random((n_days, F)) < u["coverage"]
    day_missing = rng.random(n_days) < spec.whole_day_missing
    X_obs = np.where(present, X_full, np.nan)
    X_obs[day_missing] = np.nan
    truth = dict(user=u, means_t=means_t, within_sd=wsd, drift_sd=float(drift_sd), k_change=k_change, change_day=int(change_day), feature_scale_change=feature_scale_change, regime_scale_change=regime_scale_change,
                 planted_order=list(range(K)), K=K, K_after=(K + 1 if k_change == "split" else K - 1 if k_change == "merge" else K))
    return IdentityStream(seed=int(seed), spec=spec, registry=REGISTRY, X_full=X_full, X_obs=X_obs, labels=labels, truth=truth)
