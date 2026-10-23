"""Ledger N18: held-out one-step-ahead prediction, the "real-fire drill" for the engine.

Criteria and design: tests/PREDECLARED_CRITERIA_WP3.md section 1 (committed before this file was run).

Question the tests ask: on data the engine has NOT been fitted to, does it predict better than (C1) a single Gaussian, (C2) itself with learning
switched off, (C3) itself with A_fast driven toward 0 after every M-step? "Better" means the mean one-step-ahead predictive log density
is higher by at least 3.0 paired standard errors over 8 seeds. The test of the test substitutes each breaker (with a different random
initialisation than the opponent copy) for the engine; the conjunction C1 and C2 and C3 must then fail.

WHAT THIS DOES NOT CATCH. B1 (smoother aggregated with the filtered posterior) changes `estimate_states`, not the forward filter, so these
tests cannot detect it. Nothing here claims otherwise. See integration_report/wp3 for what does detect B1.
Not tested: calibration of m_t_std (N17), anything about the smoother.
"""
import numpy as np
import pytest

from tests.utils import heldout_prediction as H


# ------------------------------------------------------------------------------------------
# data and metric tests: no engine involved or a single cached fit
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", H.SEEDS)
def test_continuation_reproduces_the_generator_on_the_training_part_and_extends_it(seed):
    data = H.continued_data(seed)                  # raises if the replayed training part differs from the generator
    assert 32 <= len(data.X_train) <= 120 and 16 <= len(data.X_test) <= 60      # the generator's logical ranges: 4 segments of 8..30, 2 segments of 8..30
    assert data.X_train.shape[1] == data.X_test.shape[1] == 4
    assert data.p_test[0] != data.p_train[-1]      # the generator always switches to a DIFFERENT regime after a segment
    assert np.all(np.isfinite(data.X_full))


def test_per_step_predictive_increments_sum_to_full_minus_train_log_likelihood():
    s = H.SEEDS[0]
    sc = H.engine_scores("real", s)
    assert abs(np.sum(sc["lp_test"]) - (sc["ll_full"] - sc["ll_train"])) < 1e-9


def test_aggregator_accepts_a_consistent_advantage_and_rejects_noise_and_identity():
    rng = np.random.default_rng(0)
    assert H.paired_margin(1.0 + 0.1 * rng.standard_normal(8))[2] > 0            # clear advantage passes
    assert H.paired_margin(0.1 * rng.standard_normal(8))[2] <= 0                 # zero-mean noise fails (this draw)
    assert H.paired_margin(np.zeros(8))[2] <= 0                                  # identical scores fail
    assert H.paired_margin(np.full(8, 0.2))[2] > 0                               # a constant positive advantage passes (se = 0)
    mixed = np.array([5.0, -4.0, 6.0, -5.0, 4.0, -3.0, 5.0, -4.0])               # positive mean but huge spread: not 3 se above zero
    assert H.paired_margin(mixed)[0] > 0 and H.paired_margin(mixed)[2] <= 0


# ------------------------------------------------------------------------------------------
# health gate: a score that contains the -1e6 constant is not a score
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("engine", ["real", "stops", "B2", "stops_alt", "B2_alt"])
def test_no_fit_or_scoring_pass_used_the_cholesky_fallback(engine):
    for s in H.SEEDS:
        sc = H.engine_scores(engine, s)
        assert sc["fit_degraded"] is False and sc["fit_fallbacks"] == 0, (engine, s, sc["fit_state"])
        assert sc["scoring_fallbacks"] == 0, (engine, s)
        assert np.all(np.isfinite(sc["lp_test"])) and np.min(sc["lp_test"]) > -1e5, (engine, s)


# ------------------------------------------------------------------------------------------
# the three criteria on the real engine
# ------------------------------------------------------------------------------------------
def _report(engine, opponent):
    d = H.per_seed_differences(engine, opponent)
    mean, se, lower = H.paired_margin(d)
    return d, mean, se, lower


# WP4 item 2 (owner decision): C1 and C3 are known findings about the engine, recorded in WP3 (RUN). They are marked xfail(strict=True) so the suite is
# green while the findings stay visible: an unexpected pass FAILS the suite and forces this mark to be removed. The criterion itself is unchanged:
# margin 3.0 paired standard errors, 8 seeds, same data and opponents (see tests/PREDECLARED_CRITERIA_WP3.md section 1).
N18_C1_REASON = (
    "N18 C1 (real engine vs a single Gaussian) fails on the engine: mean held-out step log-density advantage -0.7087 nats, paired se 0.4920, "
    "mean - 3 se = -2.1846 (needs > 0); the real engine is ahead in 3 of 8 seeds (WP3, RUN). "
    "Owner's reading: C1 is a cold-start finding, addressed in WP5; C3 is a near-miss that clears with 24 seeds (see WP3 sensitivity V1)."
)
N18_C3_REASON = (
    "N18 C3 (real engine vs A_fast driven to 0) narrowly fails: mean advantage +2.2200 nats, paired se 0.7444, mean - 3 se = -0.0131 "
    "(needs > 0); the real engine is ahead in 8 of 8 seeds (WP3, RUN). "
    "Owner's reading: C1 is a cold-start finding, addressed in WP5; C3 is a near-miss that clears with 24 seeds (see WP3 sensitivity V1)."
)


@pytest.mark.parametrize("opponent,label", [
    pytest.param("baseline", "C1 single Gaussian", marks=pytest.mark.xfail(strict=True, reason=N18_C1_REASON)),
    ("stops", "C2 stops learning"),
    pytest.param("B2", "C3 A_fast driven to 0", marks=pytest.mark.xfail(strict=True, reason=N18_C3_REASON)),
])
def test_real_engine_beats_opponent_on_held_out_data(opponent, label):
    d, mean, se, lower = _report("real", opponent)
    assert lower > 0.0, (
        f"{label}: mean held-out step log-density advantage {mean:.4f} nats, paired se {se:.4f}, mean - 3 se = {lower:.4f} (needs > 0); "
        f"per seed {np.round(d, 3).tolist()}")


# ------------------------------------------------------------------------------------------
# test of the test: a breaker substituted for the engine must fail the conjunction
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("breaker", ["stops_alt", "B2_alt"])
def test_control_a_breaker_in_place_of_the_engine_fails_the_conjunction(breaker):
    verdicts = {opp: H.beats(breaker, opp) for opp in H.OPPONENTS}
    assert not all(verdicts.values()), f"{breaker} passed C1, C2 and C3 ({verdicts}); the held-out test cannot tell it from the real engine"
