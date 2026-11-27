"""WP10B B5: the feasibility arithmetic and the facts the declared rule reads (criteria Part 10, section 16). No fit is run; the tests use only code and synthetic records (no evidence file), so they also run in the mutation copy of the repository."""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration_report" / "wp10b" / "scripts"))
import b5_feasibility as F  # noqa: E402

RATES = {"hard": 0.5, "medium": 0.5, "easy": 0.5}
PASSING_OK = {"dual_to_dual": (25, 216), "cross_engine": (30, 480)}
PASSING_WP8 = {"dual_to_dual": (4, 216), "cross_engine": (16, 480)}


def test_the_required_n_is_299_for_a_one_percent_bound():
    assert F.n_required() == 299
    assert F.exact_upper_zero(298) > 0.01 >= F.exact_upper_zero(299)
    assert F.exact_upper_zero(299) == pytest.approx(0.0099686, abs=1e-6)
    assert F.exact_upper_zero(0) == 1.0


def test_other_bounds_give_other_n():
    assert F.n_required(0.05) == 59          # the "rule of three": 3 / 0.05 is about 59
    assert F.n_required(0.01, 0.95) > F.n_required(0.01, 0.90)


def test_chains_needed_counts_nine_alignments_per_chain_and_rounds_up():
    assert F.ALIGNMENTS_PER_CHAIN == 9 and F.FITS_PER_CHAIN == 10
    assert F.chains_needed(299, 1.0) == 34                         # 299 / 9 = 33.2
    assert F.chains_needed(299, 0.5) == 67                         # 299 / 4.5 = 66.4
    assert F.chains_needed(18, 1.0) == 2 and F.chains_needed(19, 1.0) == 3
    try:
        zero = F.chains_needed(299, 0.0)
    except ZeroDivisionError:
        assert False, "a zero publishable share must give an infinite number of chains, not a crash"
    assert math.isinf(zero)


def test_cost_hours_divides_by_the_workers():
    assert F.cost_hours(1020, 60.0) == pytest.approx(8.5)
    assert F.cost_hours(1020, 60.0, workers=1) == pytest.approx(17.0)
    assert F.cost_hours(0, 60.0) == 0.0


def test_the_s0_rates_are_infeasible_and_the_numbers_are_pinned():
    ev = F.evaluate({"hard": 1 / 6, "medium": 1 / 6, "easy": 0.0}, 60.0, PASSING_WP8, dual_signal=False)
    assert ev["evaluation_fits_all_pairs"] == 1020
    assert ev["hours_all_pairs"] == pytest.approx(8.5)
    assert math.isinf(ev["evaluation_fits_published_pairs"])        # a stratum with publishable share 0 cannot be reached by published pairs
    assert ev["hours_favourable"] == pytest.approx(8.5)             # the favourable count is the finite one
    assert ev["a_cost_ok"] is False and ev["run_b5"] is False
    assert ev["verdict"] == "NOT DEMONSTRATED (infeasible in this package)"


def test_the_favourable_count_is_the_smaller_one():
    ev = F.evaluate({"hard": 1.0, "medium": 1.0, "easy": 1.0}, 60.0, PASSING_OK, dual_signal=True)
    assert ev["evaluation_fits_published_pairs"] == ev["evaluation_fits_all_pairs"] == 1020
    ev = F.evaluate(RATES, 60.0, PASSING_OK, dual_signal=True)
    assert ev["evaluation_fits_all_pairs"] == 1020 and ev["evaluation_fits_published_pairs"] == 3 * 67 * 10
    assert ev["hours_favourable"] == pytest.approx(F.cost_hours(1020, 60.0))     # 1020 < 2010: the favourable count is the smaller


def test_all_three_conditions_must_hold_to_run_b5():
    ok = F.evaluate(RATES, 1.0, PASSING_OK, dual_signal=True)
    assert ok["a_cost_ok"] and ok["c_passing_ok"] and ok["run_b5"] and ok["verdict"] == "run the A3-style calibration"
    assert F.evaluate(RATES, 1.0, PASSING_OK, dual_signal=False)["run_b5"] is False          # no dual stability signal
    assert F.evaluate(RATES, 1.0, PASSING_WP8, dual_signal=True)["run_b5"] is False          # 4 and 16 passing development alignments are below 20
    assert F.evaluate(RATES, 1.0, {"dual_to_dual": (25, 216), "cross_engine": (19, 480)}, dual_signal=True)["run_b5"] is False   # one pair type below 20 is enough
    assert F.evaluate(RATES, 60.0, PASSING_OK, dual_signal=True)["run_b5"] is False          # too expensive


def test_achievable_alignments_per_stratum_in_the_budget_are_whole_chains_shared_by_the_strata():
    assert F.achievable_per_stratum(6.0, 60.0) == 216                 # 720 fits, 72 chains, 24 per stratum, 9 alignments each
    assert F.achievable_per_stratum(6.0, 191.0) == 63                 # 226.2 fits, 22 chains, 7 per stratum
    assert F.achievable_per_stratum(6.0, 60.0, workers=1) == 108
    assert F.achievable_per_stratum(0.0, 60.0) == 0
    assert F.exact_upper_zero(216) == pytest.approx(0.013780, abs=1e-5) and F.exact_upper_zero(63) == pytest.approx(0.046440, abs=1e-4)


def test_the_cost_budget_is_inclusive():
    hours = F.cost_hours(1020, 60.0)
    assert F.evaluate(RATES, 60.0, PASSING_OK, True, budget=hours)["a_cost_ok"] is True
    assert F.evaluate(RATES, 60.0, PASSING_OK, True, budget=hours - 1e-6)["a_cost_ok"] is False
    assert F.BUDGET_HOURS == 6.0 and F.MIN_PASSING_DEV == 20 and F.WORKERS == 2


def _rec(user, stratum, variant, publishable, seconds, ok=True):
    r = {"job": {"user": user, "stratum": stratum, "variant": variant}, "ok": ok}
    if ok:
        r["dual"] = {"publishable": publishable, "fit_seconds": seconds}
    return r


def test_publishable_shares_and_median_seconds_by_stratum_and_variant():
    recs = [_rec(1, "hard", "I600", True, 100.0), _rec(2, "hard", "I600", False, 200.0), _rec(3, "medium", "I600", True, 300.0), _rec(4, "easy", "I600", False, 400.0),
            _rec(5, "easy", "S0", True, 10.0), _rec(6, "hard", "I600", False, 0.0, ok=False)]
    try:
        rates, med = F.publishable_by_stratum(recs, "I600")
    except KeyError:
        assert False, "a fit that failed to run has no dual record and must be skipped, not crash the summary"
    assert rates == {"hard": 0.5, "medium": 1.0, "easy": 0.0}               # the failed fit and the S0 fit are not counted
    assert med == 250.0
    rates_all, med_all = F.publishable_by_stratum(recs, None)
    assert rates_all["easy"] == 0.5 and med_all == 200.0                      # every variant, failed fits still left out
    assert F.publishable_by_stratum([], "I600")[0] == {"hard": 0.0, "medium": 0.0, "easy": 0.0}


def test_the_production_code_keeps_both_pair_types_uncalibrated_and_the_dual_without_a_flag():
    facts = F.code_facts()
    assert facts["dual_engine_calibrated"] is False and facts["cross_engine_calibrated"] is False
    assert facts["dual_stability_flag"] == (False, None)


def test_main_prints_the_verdict_for_both_configurations(tmp_path, capsys):
    b1 = tmp_path / "b1.jsonl"
    b2 = tmp_path / "b2.jsonl"
    b1.write_text("\n".join(json.dumps(_rec(i, s, "S0", i % 6 == 0, 60.0)) for i, s in enumerate(["hard", "medium", "easy"] * 4)) + "\n")
    b2.write_text("\n".join(json.dumps(_rec(i, s, "I600", i % 3 != 0, 190.0)) for i, s in enumerate(["hard", "medium", "easy"] * 4)) + "\n")
    out = tmp_path / "out.txt"
    assert F.main(["x", str(b1), str(b2), str(out), "I600"]) == 0
    text = out.read_text()
    assert text.count("NOT DEMONSTRATED (infeasible in this package)") == 2
    assert "n required for the A3 level" in text and "299" in text
    assert "a dual stability signal exists: False" in text                  # the production dual result carries no stability flag
    assert "[I600 (B2, the frozen configuration)]" in text and "median dual fit 190 s" in text
    assert capsys.readouterr().out.strip() == text.strip()
