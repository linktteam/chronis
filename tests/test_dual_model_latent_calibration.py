"""Ledger A2: replacements for the two old tests of the fast latent state.

Criteria were pre-declared in tests/PREDECLARED_CRITERIA_WP2.md BEFORE this file was written or run.
They are not edited here after seeing results. Read that file for the reasons behind every number.

WHY THE OLD TESTS WERE REPLACED (not loosened)
  * test 39 regressed the true latent on the fitted latent with ONE global affine map. The model class gives
    every regime its own C_k, A_k, Q_k, so the latent is identifiable only up to a map that can differ per
    regime. A global map is not something this model class can satisfy. The new check aligns per TRUE regime.
  * test 41 never read `m_t_std`; it compared an error with a multiple of its own spread. The new check
    asks whether the engine's reported standard deviation matches its actual error.

WHAT IS AND IS NOT CLAIMED
  * Alignment is fitted IN SAMPLE on the same data (3 parameters per output per regime), which flatters both
    statistics. Parameter uncertainty is not part of `m_t_std`. This is a coarse screen, not a certificate of calibration.
  * With T = 60..64 a coverage of 1.0 is not detectably different from the nominal 0.9545, so this test cannot
    detect intervals that are too WIDE, only too narrow.
"""
import copy
import math

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from tests.utils.synthetic_dual_fits import CELLS, fitted

# ---- pre-declared constants (tests/PREDECLARED_CRITERIA_WP2.md sections 1.2 and 1.3) ----
R2_MIN = 0.80
Z = 2.0
P0 = math.erf(Z / math.sqrt(2.0))      # P(|N(0,1)| <= 2) = 0.954500
SE_MULT = 3.0


def coverage_floor(T: int) -> float:
    return P0 - SE_MULT * math.sqrt(P0 * (1.0 - P0) / T)


# ------------------------------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------------------------------
def per_regime_affine_prediction(src: np.ndarray, dst: np.ndarray, labels: np.ndarray, min_points: int) -> np.ndarray:
    """For each label r: least squares dst ~ [src, 1] W_r fitted in sample; returns the prediction for every row.
    A label with fewer than `min_points` rows raises; nothing is skipped or dropped silently."""
    pred = np.zeros_like(dst, dtype=float)
    for r in np.unique(labels):
        idx = labels == r
        n = int(idx.sum())
        if n < min_points:
            raise ValueError(f"regime {r} has only {n} time steps (< {min_points}); alignment would be meaningless")
        A = np.hstack([src[idx], np.ones((n, 1))])
        W, *_ = np.linalg.lstsq(A, dst[idx], rcond=None)
        pred[idx] = A @ W
    return pred


def aligned_r2(m_hat: np.ndarray, true_m: np.ndarray, labels: np.ndarray) -> float:
    """Pooled R^2 of true_m explained by a per-true-regime affine map of m_hat (in-sample)."""
    L = m_hat.shape[1]
    pred = per_regime_affine_prediction(m_hat, true_m, labels, min_points=3 * (L + 1))
    sse = float(np.sum((true_m - pred) ** 2))
    sst = float(np.sum((true_m - true_m.mean(axis=0)) ** 2))
    return 1.0 - sse / sst


def old_global_affine_r2(m_hat: np.ndarray, true_m: np.ndarray) -> float:
    """The statistic the OLD test 39 used (sklearn: uniform average of per-column R^2 of ONE global affine map)."""
    return float(LinearRegression().fit(m_hat, true_m).score(m_hat, true_m))


def coverage_in_fitted_coordinates(m_hat, m_std, true_m, labels) -> np.ndarray:
    """Per-dimension fraction of time steps with |m_hat - map(true_m)| <= Z * m_std, where map is the per-true-regime
    in-sample affine map from the true latent into the FITTED coordinates (so m_std is read in its own coordinates)."""
    L = m_hat.shape[1]
    pred = per_regime_affine_prediction(true_m, m_hat, labels, min_points=3 * (L + 1))
    z = (m_hat - pred) / m_std
    return np.mean(np.abs(z) <= Z, axis=0)


# ------------------------------------------------------------------------------------------
# metric tests (no engine involved): positive and negative controls on the statistics themselves
# ------------------------------------------------------------------------------------------
def _two_regime_labels(T):
    return (np.arange(T) // (T // 4)) % 2


def test_H1_per_regime_alignment_accepts_regime_specific_coordinates_where_the_old_global_map_fails():
    rng = np.random.default_rng(0)
    T = 240
    labels = _two_regime_labels(T)
    true_m = np.zeros((T, 2))
    for t in range(1, T):
        true_m[t] = 0.75 * true_m[t - 1] + rng.normal(0, 0.5, 2)
    R1 = np.array([[0.0, -1.0], [1.0, 0.0]])              # a 90 degree rotation, plus a flip below
    fitted_m = np.where(labels[:, None] == 0, true_m, -(true_m @ R1.T) + np.array([4.0, -4.0]))
    fitted_m = fitted_m + rng.normal(0, 0.05, fitted_m.shape)
    assert aligned_r2(fitted_m, true_m, labels) >= 0.95
    assert old_global_affine_r2(fitted_m, true_m) < 0.50     # the old assertion cannot be met by this (correct) fit


def test_H2_alignment_cannot_manufacture_a_good_fit_from_unrelated_coordinates():
    rng = np.random.default_rng(1)
    T = 64
    labels = _two_regime_labels(T)
    true_m = np.zeros((T, 2))
    for t in range(1, T):
        true_m[t] = 0.75 * true_m[t - 1] + rng.normal(0, 0.5, 2)
    unrelated = true_m[rng.permutation(T)]
    assert aligned_r2(unrelated, true_m, labels) < 0.30


def test_H3_coverage_statistic_is_nominal_for_a_correct_std_and_collapses_for_a_wrong_one():
    rng = np.random.default_rng(2)
    T = 20000
    labels = np.zeros(T, dtype=int)
    true_m = rng.normal(0, 1, (T, 1))
    sd = rng.uniform(0.1, 2.0, (T, 1))
    m_hat = 2.0 * true_m + 1.0 + sd * rng.standard_normal((T, 1))
    ok = coverage_in_fitted_coordinates(m_hat, sd, true_m, labels)[0]
    assert abs(ok - P0) < 0.01
    too_small = coverage_in_fitted_coordinates(m_hat, 0.25 * sd, true_m, labels)[0]
    assert too_small < 0.50


def test_alignment_refuses_a_regime_with_too_few_points():
    labels = np.array([0] * 30 + [1] * 5)
    with pytest.raises(ValueError, match="regime 1"):
        aligned_r2(np.random.default_rng(0).normal(size=(35, 2)), np.random.default_rng(1).normal(size=(35, 2)), labels)


# ------------------------------------------------------------------------------------------
# engine tests: the real engine on both cells
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cell", sorted(CELLS))
def test_latent_recovery_per_regime_aligned(cell):
    data, _, m_t, _, _, _ = fitted("real", cell)
    r2 = aligned_r2(m_t, data.true_m_t, data.true_p_t)
    assert r2 >= R2_MIN, (
        f"cell {cell}: per-regime-aligned R^2 {r2:.3f} < {R2_MIN} "
        f"(old global-affine statistic on the same fit: {old_global_affine_r2(m_t, data.true_m_t):.3f})")


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_latent_recovery_control_time_permuted_m_t_fails_the_criterion(cell):
    """Negative control G (REQUIRED by the pre-declaration): the same fit with its m_t shuffled in time must violate the criterion."""
    data, _, m_t, _, _, _ = fitted("real", cell)
    shuffled = m_t[np.random.default_rng(0).permutation(len(m_t))]
    r2 = aligned_r2(shuffled, data.true_m_t, data.true_p_t)
    assert r2 < R2_MIN, f"cell {cell}: shuffled m_t still gives R^2 {r2:.3f}; the criterion would not detect a broken engine"


N17_REASON = (
    "FINDING N17 (ledger 7.D): the engine's reported m_t_std under-covers on cell B, dimension 0. First run: coverage "
    "0.8281 (53/64) vs the pre-declared floor 0.8764 (nominal 0.9545, T=64); the ledger's reviewer reports the same 0.828. "
    "The criterion is NOT loosened. strict=True so that a later engine fix turns this into a visible XPASS failure "
    "that forces the mark to be removed. Owner may prefer to leave the cell red instead (see report).")
_CALIBRATION_CELLS = [
    ("A", 0), ("A", 1),
    pytest.param("B", 0, marks=pytest.mark.xfail(strict=True, reason=N17_REASON)),
    ("B", 1),
]


@pytest.mark.parametrize("cell,dim", _CALIBRATION_CELLS)
def test_m_t_std_calibration_coverage(cell, dim):
    data, _, m_t, m_std, _, _ = fitted("real", cell)
    assert np.all(np.isfinite(m_std)) and np.all(m_std > 0)
    cov = coverage_in_fitted_coordinates(m_t, m_std, data.true_m_t, data.true_p_t)[dim]
    floor = coverage_floor(len(data.X))
    assert cov >= floor, (
        f"cell {cell} dim {dim}: coverage of +-{Z:g}*m_t_std is {cov:.4f}, below the pre-declared floor {floor:.4f} "
        f"(nominal {P0:.4f}, T={len(data.X)})")


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("cell", sorted(CELLS))
def test_m_t_std_calibration_control_std_scaled_down_fails_the_criterion(cell, dim):
    """Negative control S (REQUIRED by the pre-declaration): reported std x 0.25 must violate the calibration criterion in
    every cell and dimension. This is also the proof that the calibration test really reads `m_t_std`."""
    data, _, m_t, m_std, _, _ = fitted("real", cell)
    cov = coverage_in_fitted_coordinates(m_t, 0.25 * copy.copy(m_std), data.true_m_t, data.true_p_t)[dim]
    assert cov < coverage_floor(len(data.X)), f"cell {cell} dim {dim}: coverage {cov:.3f} with std x 0.25 is still above the floor"
