"""WP9 B1: the declared development rule for z and the evaluation sentences of integration_report/wp9/scripts/b1_analyze.py on fabricated records (no real data)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration_report" / "wp9" / "scripts"))
import b1_analyze as A                                           # noqa: E402

NC = "fit_state_not_converged: fit_state is REJECTED_UPDATE, the EM fit did not reach convergence"


def fit(seed, length, arm, mean, gain=None, se=None, stab=0.1, state=None, em_reason=NC, withheld_em=True, published=None, reasons=None):
    c = {}
    if gain is not None:
        c = dict(mean_gain=gain, se=se, stability=stab, reasons=reasons or [], state=state or "NOT_CERTIFIED")
    elif state:
        c = dict(state=state, reasons=reasons or [])
    return dict(seed=seed, length=length, arm=arm, mean=mean, finite=True, withheld_em=withheld_em, em_reason=em_reason if withheld_em else None, certification=c,
                published_after=(state == "CERTIFIED_PRIOR_ANCHORED") if published is None else published)


def table(rows):
    return {(r["seed"], r["length"], r["arm"]): r for r in rows}


def test_eligibility_means_withheld_for_non_convergence_only():
    assert A.eligible(fit(1, 60, "shrink5", 0.0))
    assert not A.eligible(fit(1, 60, "shrink5", 0.0, withheld_em=False))
    assert not A.eligible(fit(1, 60, "shrink5", 0.0, em_reason=NC + "; collapsed_regime: a regime holds 1.0 effective days"))
    assert not A.eligible(fit(1, 60, "shrink5", 0.0, em_reason="degenerate_duration: x"))


def test_the_offline_verdict_follows_the_margin_and_the_stability_level():
    r = fit(1, 60, "shrink5", 0.0, gain=1.0, se=0.4)                        # lower = 1 - z * 0.4: > 0 for z = 1 and 2, < 0 for z = 3
    assert [A.certified_at(r, z) for z in (1.0, 2.0, 3.0)] == [True, True, False]
    assert A.certified_at(fit(1, 60, "shrink5", 0.0, gain=1.0, se=0.4, stab=0.26), 1.0) is False
    assert A.certified_at(fit(1, 60, "shrink5", 0.0, gain=1.0, se=0.4, stab=0.25), 1.0) is True
    assert A.certified_at(fit(1, 60, "shrink5", 0.0, gain=1.0, se=0.4, reasons=["too_few_rows"]), 1.0) is False        # decision never reached
    assert A.certified_at(fit(1, 60, "shrink5", 0.0), 1.0) is False


def dev_table(shrink_gain, ctrl_gain, n=12):
    rows = []
    for s in range(n):
        for ln in A.LENGTHS:
            rows.append(fit(s, ln, "gauss", -5.0, withheld_em=False))
            rows += [dict(fit(s, ln, "gauss", -5.0), arm="shared_true", mean=-8.0), dict(fit(s, ln, "gauss", -5.0), arm="shared_scrambled", mean=-12.0), dict(fit(s, ln, "gauss", -5.0), arm="shared_shifted", mean=-9.5)]
            rows.append(fit(s, ln, "shrink5", -4.0, gain=shrink_gain, se=0.5))
            for a in A.CONTROLS:
                rows.append(fit(s, ln, a, -9.0, gain=ctrl_gain, se=0.5))
    return table(rows)


def run_dev(tmp_path, recs):
    out = tmp_path / "o.txt"; js = tmp_path / "o.json"
    A.dev_report(recs, str(out), str(js))
    return json.load(open(js)), out.read_text()


def test_the_rule_takes_the_smallest_z_that_separates_the_controls(tmp_path):
    # shrink5 gain 1.2 / se 0.5 (certified for z < 2.4), controls gain 0.75 / se 0.5 (certified for z < 1.5): z = 1 certifies both, z = 2 only shrink5
    res, txt = run_dev(tmp_path, dev_table(1.2, 0.75))
    assert res["chosen_z"] == 2.0 and res["rule_met"] is True and res["valid"] == dict(scrambled=True, shifted=True)


def test_when_no_z_separates_the_controls_z_is_3_and_the_shortfall_is_reported(tmp_path):
    res, txt = run_dev(tmp_path, dev_table(5.0, 5.0))
    assert res["chosen_z"] == 3.0 and res["rule_met"] is False and "shortfall" in txt


def test_a_control_prior_that_is_not_worse_than_the_true_prior_is_reported_as_invalid(tmp_path):
    recs = dev_table(1.2, 0.75)
    for k, r in recs.items():
        if k[2] == "shared_shifted":
            r["mean"] = -8.5                                                  # only 0.5 worse than the true prior
    res, txt = run_dev(tmp_path, recs)
    assert res["valid"] == dict(scrambled=True, shifted=False) and "NOT a valid control" in txt


def eval_table(shrink_state, ctrl_state, n=24, shrink_gain=1.0):
    rows = []
    for s in range(n):
        for ln in A.LENGTHS:
            rows.append(fit(s, ln, "gauss", -5.0, withheld_em=False))
            rows.append(fit(s, ln, "shrink5", -5.0 + shrink_gain, gain=1.0, se=0.2, state=shrink_state))
            for a in A.CONTROLS:
                rows.append(fit(s, ln, a, -7.0, gain=-1.0, se=0.2, state=ctrl_state))
    return table(rows)


def run_eval(tmp_path, recs):
    out = tmp_path / "e.txt"; js = tmp_path / "e.json"
    A.eval_report(recs, str(out), str(js))
    return json.load(open(js)), out.read_text()


def test_everything_certified_with_a_clear_gain_survives_and_controls_that_do_not_certify_are_met(tmp_path):
    res, txt = run_eval(tmp_path, eval_table("CERTIFIED_PRIOR_ANCHORED", "NOT_CERTIFIED"))
    assert res["survives"] is True and res["controls_met"] is True and "survives" in txt and "does not survive" not in txt
    assert res["per_length"]["60"]["certified"] == 24 and res["per_length"]["60"]["published_after"] == 24 and res["per_length"]["60"]["cert_gain"]["n"] == 24


def test_controls_that_certify_as_often_as_shrink5_fail_the_controls_criterion(tmp_path):
    res, txt = run_eval(tmp_path, eval_table("CERTIFIED_PRIOR_ANCHORED", "CERTIFIED_PRIOR_ANCHORED"))
    assert res["controls_met"] is False and "NOT MET" in txt


def test_fewer_than_eight_judged_fits_cannot_survive(tmp_path):
    recs = eval_table("NOT_CERTIFIED", "NOT_CERTIFIED")
    for k, r in recs.items():
        if k[2] == "shrink5" and k[0] < 5:
            r["certification"]["state"] = "CERTIFIED_PRIOR_ANCHORED"; r["published_after"] = True
    res, txt = run_eval(tmp_path, recs)
    assert res["survives"] is False and "NOT judged: n < 8" in txt and "does not survive" in txt


def test_a_publishable_set_with_a_loss_does_not_survive_and_risk_coverage_is_computed_against_the_gaussian(tmp_path):
    recs = eval_table("CERTIFIED_PRIOR_ANCHORED", "NOT_CERTIFIED", shrink_gain=-1.0)
    res, txt = run_eval(tmp_path, recs)
    assert res["survives"] is False and res["per_length"]["30"]["risk_coverage"]["mean"] == pytest.approx(-1.0)
