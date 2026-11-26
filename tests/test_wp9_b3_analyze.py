"""WP9 B3: the declared development rule and the evaluation wording of integration_report/wp9/scripts/b3_analyze.py on fabricated records (no real data)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration_report" / "wp9" / "scripts"))
import b3_analyze as A                                           # noqa: E402

MU2 = [[0.0, 0.0], [4.0, 4.0]]
MU_CLOSE = [[0.0, 0.0], [0.3, 0.3]]
VAR = [[1.0, 1.0], [1.0, 1.0]]


def rec(design, planted, selected, mu=MU2, gap=50.0, unstable=False, seed=1):
    bic = {"2": 100.0, "3": 100.0 + gap, "4": 100.0 + gap, "5": 100.0 + gap}
    if selected != 2:
        bic = {"2": 100.0 + gap, "3": 100.0, "4": 100.0 + gap, "5": 100.0 + gap}
        mu = [[0.0, 0.0], [4.0, 4.0], [8.0, 0.0]]
        var = [[1.0, 1.0]] * 3
    else:
        var = VAR
    return dict(seed=seed, design=design, planted_k=planted, selected_k=selected, bic_by_k=bic, mu=mu, var=var, fit_unstable=unstable, ok=True)


def test_clean_right_fits_are_not_tagged_and_wrong_close_fits_are():
    right = rec("easy", 2, 2)
    wrong = rec("hard", 3, 2, mu=MU_CLOSE)
    assert A.unsure(right, 1.0, 6, True)["k_unsure"] is False
    assert A.unsure(wrong, 1.0, 6, True)["k_unsure"] is True


def test_metrics_count_detection_and_false_alarms_by_design():
    recs = [rec("easy", 2, 2), rec("easy", 2, 2, unstable=True), rec("hard", 2, 2), rec("hard", 3, 2, mu=MU_CLOSE), rec("easy", 3, 2)]
    m = A.metrics(recs, 1.0, 6, True)
    assert (m["n_wrong"], m["n_right"], m["det"], m["fa_all"], m["fa_easy"], m["n_easy_right"], m["fa_hard"], m["n_hard_right"]) == (2, 3, 1, 1, 1, 2, 0, 1)


def test_the_rule_takes_the_highest_detection_among_points_within_the_easy_false_alarm_cap_and_breaks_ties_as_declared():
    recs = [rec("easy", 2, 2) for _ in range(4)] + [rec("hard", 3, 2, mu=MU_CLOSE) for _ in range(3)]
    grid, ch, cap = A.pick(recs)
    assert cap is True and ch["det"] == 3 and A.rate(ch["fa_easy"], ch["n_easy_right"]) == 0.0
    # ties on detection and on false alarms: the smallest d0, then the smallest delta, then signal (3) off
    assert (ch["d0"], ch["delta"], ch["sig3"]) == (0.5, 0, False)
    # a different data set where only a larger d0 catches the wrong fit: the point with the highest detection wins over the smaller d0
    close = [rec("hard", 3, 2, mu=[[0.0, 0.0], [0.8, 0.8]]) for _ in range(3)]
    _, ch2, _ = A.pick([rec("easy", 2, 2) for _ in range(4)] + close)
    assert ch2["det"] == 3 and ch2["d0"] == 1.0 and ch2["delta"] == 0 and ch2["sig3"] is False


def test_when_no_point_meets_the_cap_the_lowest_easy_false_alarm_is_taken_and_the_shortfall_is_reported():
    # every easy right-K fit is unstable, so every grid point flags all of them: false alarm 100% everywhere
    recs = [rec("easy", 2, 2, unstable=True) for _ in range(4)] + [rec("hard", 3, 2, mu=MU_CLOSE)]
    grid, ch, cap = A.pick(recs)
    assert cap is False and A.rate(ch["fa_easy"], ch["n_easy_right"]) == 1.0


def test_wilson_wording():
    assert "Wilson" in A.fmt_ci(3, 10)
    assert A.fmt_ci(0, 0).startswith("0/0")
