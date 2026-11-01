"""WP6 item 1: `timestamps` are calendar DAYS; the entry points refuse anything else before allocating.
Pre-declaration: tests/PREDECLARED_CRITERIA_WP6.md section 2. Every check has a negative control (the valid neighbour passes and gives the same fit as before)."""
import numpy as np
import pytest

from backbone.hssm import fitting
from backbone.hssm.config import MAX_TIMESTAMP_SPAN_DAYS
from backbone.hssm.fitting import (fit_dual_with_random_restarts, fit_hssm, fit_with_random_restarts, validate_timestamps_days)
from backbone.hssm.gating import ColdStartError, fit_hssm_gated_result
from backbone.hssm.model import GaussianHSMM

LIMIT = MAX_TIMESTAMP_SPAN_DAYS


def _data(T=45, F=3, seed=0):
    rng = np.random.default_rng(seed)
    means = np.array([[-1.5] * F, [1.5] * F])
    path = np.repeat([0, 1, 0], [T // 3, T // 3, T - 2 * (T // 3)])
    return means[path] + rng.normal(0, 0.5, (T, F))


FAST = dict(n_initializations=1, max_iter=5, allow_fast_test_fit=True, max_duration=12)


def test_the_limit_is_declared_and_is_twenty_thousand_days():
    assert LIMIT == 20000


@pytest.mark.parametrize("ts", [
    np.arange(45.0),                                    # one row per day
    np.arange(45.0) * 2 + 100.0,                        # every second day, offset start
    np.linspace(100.0, 160.0, 45),                      # fractional days, spacing > 1
    list(range(45)),                                    # a plain list of ints
    np.cumsum(np.r_[0, np.random.default_rng(3).integers(1, 5, 44)]).astype(float),   # irregular gaps
    np.r_[0.0, 1.0, 1.6, 3.0, np.arange(4.0, 45.0)],    # 1.6 rounds to day 2 (round, not floor): distinct days, valid
])
def test_valid_timestamps_are_accepted_and_returned_as_floats(ts):
    out = validate_timestamps_days(ts, len(ts))
    assert out.dtype == np.float64 and out.shape == (len(ts),)
    np.testing.assert_array_equal(out, np.asarray(ts, dtype=float))


@pytest.mark.parametrize("name,ts,needle", [
    ("seconds", np.arange(45.0) * 86400.0, "seconds or milliseconds"),
    ("milliseconds", np.arange(45.0) * 86400.0 * 1000.0, "seconds or milliseconds"),
    ("nan", np.r_[np.arange(44.0), np.nan], "NaN or infinite"),
    ("inf", np.r_[np.arange(44.0), np.inf], "NaN or infinite"),
    ("decreasing", np.arange(45.0)[::-1].copy(), "strictly increasing"),
    ("duplicate day", np.r_[np.arange(44.0), 43.0], "strictly increasing"),
    ("equal neighbours", np.r_[0.0, 0.0, np.arange(2.0, 45.0)], "strictly increasing"),
    ("same day after rounding", np.r_[np.arange(40.0), 40.2, 40.4, 41.0, 42.0, 43.0], "same calendar day"),
    ("two dimensional", np.arange(90.0).reshape(45, 2), "1-dimensional"),
    ("wrong length", np.arange(44.0), "45 rows"),
    ("empty", np.array([]), "rows"),
])
def test_each_invalid_class_raises_value_error_naming_the_unit(name, ts, needle):
    with pytest.raises(ValueError, match=needle):
        validate_timestamps_days(ts, 45)


def test_empty_timestamps_are_refused_even_without_a_row_count():
    with pytest.raises(ValueError, match="empty"):
        validate_timestamps_days(np.array([]))


def test_strings_are_refused_with_value_error():
    with pytest.raises(ValueError, match="numbers in calendar DAYS"):
        validate_timestamps_days(["2020-01-01"] * 3, 3)


def test_the_unit_is_named_in_the_span_message():
    with pytest.raises(ValueError) as e:
        validate_timestamps_days(np.arange(45.0) * 86400.0, 45)
    msg = str(e.value)
    assert "calendar DAYS" in msg and str(LIMIT) in msg and "years" in msg


def test_span_boundary_exactly_at_the_limit_passes_and_just_above_fails():
    ok = np.r_[0.0, 10.0, float(LIMIT)]
    validate_timestamps_days(ok, 3)                                  # negative control: exactly the limit
    for bad in (np.r_[0.0, 10.0, LIMIT + 0.5], np.r_[0.0, 10.0, LIMIT + 1.0]):
        with pytest.raises(ValueError, match="limit of"):
            validate_timestamps_days(bad, 3)


def test_the_check_does_not_modify_its_input():
    ts = np.arange(45.0) * 1.5
    before = ts.copy()
    validate_timestamps_days(ts, 45)
    np.testing.assert_array_equal(ts, before)
    lst = list(range(45))
    validate_timestamps_days(lst, 45)
    assert lst == list(range(45))


# ---------- the entry points ----------

SECONDS = np.arange(45.0) * 86400.0


class _AllocationReached(AssertionError):
    pass


def _forbid_the_lattice(monkeypatch):
    """Fail if the baseline builds a calendar lattice: the guard must come first."""
    def boom(*a, **k):
        raise _AllocationReached("the calendar lattice was built")
    monkeypatch.setattr(GaussianHSMM, "_build_calendar_grid", staticmethod(boom))


def test_fit_hssm_baseline_refuses_seconds_before_any_lattice(monkeypatch):
    _forbid_the_lattice(monkeypatch)
    with pytest.raises(ValueError, match="seconds or milliseconds"):
        fit_hssm(matrix=_data(), candidate_ks=(2,), timestamps=SECONDS, **FAST)


def test_fit_hssm_dual_refuses_seconds():
    with pytest.raises(ValueError, match="seconds or milliseconds"):
        fit_hssm(matrix=_data(), candidate_ks=(2,), timestamps=SECONDS, model_type="dual_timescale", latent_dim=1, **FAST)


def test_fit_with_random_restarts_refuses_seconds_before_any_lattice(monkeypatch):
    _forbid_the_lattice(monkeypatch)
    with pytest.raises(ValueError, match="seconds or milliseconds"):
        fit_with_random_restarts(_data(), 2, 3, n_init=1, n_iter=3, max_duration=12, bypass_init_gate=True, timestamps=SECONDS)


def test_fit_dual_with_random_restarts_refuses_seconds():
    with pytest.raises(ValueError, match="seconds or milliseconds"):
        fit_dual_with_random_restarts(_data(), 2, 3, latent_dim=1, n_init=1, n_iter=3, max_duration=12, bypass_init_gate=True, timestamps=SECONDS)


def test_gated_entry_points_refuse_seconds():
    with pytest.raises(ValueError, match="seconds or milliseconds"):
        fit_hssm_gated_result(_data(), n_regimes=2, n_features=3, n_init=1, max_duration=12, timestamps=SECONDS)


def test_the_guard_runs_before_the_cold_start_gate():
    X = _data(T=10)
    with pytest.raises(ValueError, match="strictly increasing"):
        fit_hssm(matrix=X, candidate_ks=(2,), timestamps=np.zeros(10), **FAST)          # malformed AND too short: ValueError first
    with pytest.raises(ColdStartError):                                                  # negative control: valid timestamps, too short
        fit_hssm(matrix=X, candidate_ks=(2,), timestamps=np.arange(10.0), **FAST)


def test_the_model_refuses_a_huge_span_before_allocating(monkeypatch):
    X = _data(T=5)
    ts = np.array([0.0, 1.0, 2.0, 3.0, 86400.0 * 74])                                    # a short record in seconds
    real_full = np.full

    def watch(shape, *a, **k):
        assert int(np.prod(shape)) < 10_000_000, f"an allocation of shape {shape} was attempted"
        return real_full(shape, *a, **k)

    monkeypatch.setattr("backbone.hssm.model.np.full", watch)
    with pytest.raises(ValueError, match="calendar days"):
        GaussianHSMM._build_calendar_grid(X, ts)
    grid = GaussianHSMM._build_calendar_grid(X, np.array([0.0, 1.0, 2.0, 4.0, 7.0]))    # negative control
    assert grid.shape == (8, 3) and np.isnan(grid[3]).all() and np.isnan(grid[5]).all()


def test_the_limit_applies_to_the_model_grid_at_the_boundary():
    X = _data(T=3)
    GaussianHSMM._build_calendar_grid(X, np.array([0.0, 10.0, float(LIMIT)]))            # exactly the limit: builds (20001, 3), 0.5 MB
    with pytest.raises(ValueError, match="limit of"):
        GaussianHSMM._build_calendar_grid(X, np.array([0.0, 10.0, float(LIMIT + 1)]))


def test_valid_timestamps_give_the_same_fit_as_calling_the_model_directly():
    """The guard has no side effect: the harness fit equals a direct fit of the same seed."""
    X = _data()
    ts = np.arange(45.0) * 1.0 + 7.0
    model, _ = fit_with_random_restarts(X, 2, 3, n_init=1, n_iter=8, max_duration=12, base_seed=11, bypass_init_gate=True, timestamps=ts)
    direct = GaussianHSMM(n_regimes=2, n_features=3, max_duration=12, seed=11)
    direct.fit(X, n_iter=8, verbose=False, timestamps=ts)
    assert model.log_likelihood_ == direct.log_likelihood_
    assert model.fit_state_ == direct.fit_state_ and model.n_iter_ == direct.n_iter_
    np.testing.assert_array_equal(np.sort(model.mu, axis=0), np.sort(direct.mu, axis=0))


def test_fit_hssm_with_valid_timestamps_still_works_end_to_end():
    # (the baseline harness raises FittingConvergenceError when nothing converged in 5 iterations, so use enough iterations)
    r = fit_hssm(matrix=_data(), candidate_ks=(2,), timestamps=np.arange(45.0), n_initializations=2, max_iter=60, allow_fast_test_fit=True,
                 max_duration=12, random_seed=3)
    assert r.K == 2 and r.calendar_start == 0.0 and r.calendar_end == 44.0


def test_docstrings_and_schema_state_the_unit():
    for obj in (fit_hssm, fit_dual_with_random_restarts, validate_timestamps_days):
        assert "CALENDAR DAYS" in (obj.__doc__ or "")
    import pathlib
    schema = (pathlib.Path(__file__).resolve().parents[1] / "docs" / "backbone" / "HSSM_RESULT_SCHEMA.md").read_text()
    contract = (pathlib.Path(__file__).resolve().parents[1] / "docs" / "INTEGRATION_CONTRACT_HSSM.md").read_text()
    assert "calendar DAYS" in schema and f"{LIMIT:,}" in schema
    assert "calendar DAYS" in contract and f"{LIMIT:,}" in contract
