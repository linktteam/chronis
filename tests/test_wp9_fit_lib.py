"""WP9 study library (integration_report/wp9/scripts/wp9_fit_lib.py): restart selection, agreement summaries, strata, and equality with the production fit."""
import contextlib
import io
import logging
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "integration_report", "wp9", "scripts"))
import wp9_fit_lib as L  # noqa: E402


def R(i, ll, collapsed=False, state="CONVERGED", labels="000111222"):
    r = dict(i=i, seed=i, state=state, ll=ll)
    if state in ("CONVERGED", "REJECTED_UPDATE", "MAX_ITER_REACHED"):
        r["collapsed"] = collapsed
    if state == "CONVERGED":
        r["labels"] = labels
    return r


def test_selection_prefers_the_best_converged_non_collapsed_restart():
    rs = [R(0, -10.0), R(1, -5.0, collapsed=True), R(2, -7.0), R(3, -6.0, state="MAX_ITER_REACHED")]
    assert L.select_restart(rs)["i"] == 2                 # -7 is the best healthy converged one; -5 is collapsed; -6 is not converged


def test_selection_falls_back_to_the_best_collapsed_when_every_converged_restart_is_collapsed():
    rs = [R(0, -10.0, collapsed=True), R(1, -5.0, collapsed=True)]
    assert L.select_restart(rs)["i"] == 1


def test_selection_returns_none_when_nothing_converged():
    assert L.select_restart([R(0, -1.0, state="MAX_ITER_REACHED"), dict(i=1, seed=1, state="FAILED")]) is None
    assert L.select_restart([]) is None


def test_selection_uses_only_the_first_n_restarts():
    rs = [R(0, -10.0), R(1, -9.0), R(2, -1.0)]
    assert L.select_restart(rs, n=2)["i"] == 1
    assert L.select_restart(rs)["i"] == 2
    assert L.select_restart(rs, n=1)["i"] == 0


def test_top_restarts_order_epsilon_and_count():
    rs = [R(0, -10.0), R(1, -9.0), R(2, -9.5), R(3, -2.0, collapsed=True), R(4, -20.0, state="MAX_ITER_REACHED")]
    assert [r["i"] for r in L.top_restarts(rs, None)] == [1, 2, 0]
    assert [r["i"] for r in L.top_restarts(rs, None, epsilon=0.6)] == [1, 2]
    assert [r["i"] for r in L.top_restarts(rs, None, m=1)] == [1]
    assert [r["i"] for r in L.top_restarts(rs, 1)] == [0]


def test_consensus_ari_is_the_smallest_pairwise_agreement_of_the_best_m():
    same, swapped, other = "000111222", "111222000", "012012012"
    assert L.consensus_ari([R(0, -1, labels=same), R(1, -2, labels=swapped)], None) == pytest.approx(1.0)      # a relabelling is the same partition
    mixed = L.consensus_ari([R(0, -1, labels=same), R(1, -2, labels=swapped), R(2, -3, labels=other)], None, m=3)
    assert mixed < 0.5
    assert L.consensus_ari([R(0, -1, labels=same), R(1, -2, labels=swapped), R(2, -3, labels=other)], None, m=2) == pytest.approx(1.0)   # only the best two count
    assert L.consensus_ari([R(0, -1)], None) is None


def test_label_string_round_trip():
    lab = np.array([0, 2, 1, 1, 0])
    assert np.array_equal(L.str_to_labels(L.labels_to_str(lab)), lab)


def test_job_key_ignores_the_number_of_restarts_and_the_iteration_cap():
    a = dict(user=1, start=0, end=90, fitseed=1, K=3, n_init=20, max_iter=60, stratum="hard")
    assert L.job_key(a) == L.job_key(dict(a, n_init=4, max_iter=10))
    assert L.job_key(a) != L.job_key(dict(a, fitseed=1001))
    assert L.job_key(a) != L.job_key(dict(a, stratum="easy"))


@pytest.mark.parametrize("stratum,lo,hi", [("hard", 0.4, 0.9), ("medium", 0.9, 1.6), ("easy", 1.6, 3.0), ("wp8", 0.5, 2.0)])
def test_strata_bound_the_planted_separation(stratum, lo, hi):
    from data.synthetic_identity import draw_user
    seps = [draw_user(9200 + s, L.spec_for(stratum))["sep"] for s in range(40)]
    assert min(seps) >= lo - 1e-9 and max(seps) <= hi + 1e-9
    assert max(seps) - min(seps) > 0.3 * (hi - lo)           # the whole range is used, not one value


def test_a_user_keeps_every_other_parameter_when_only_the_stratum_changes():
    from data.synthetic_identity import draw_user
    a, b = draw_user(9201, L.spec_for("hard")), draw_user(9201, L.spec_for("easy"))
    assert np.allclose(a["loading"], b["loading"]) and np.allclose(a["offsets"], b["offsets"]) and a["sep"] != b["sep"]


def test_the_selected_restart_is_the_production_fit():
    """Same seeds, same model: the study's selected restart equals what `fit_hssm` returns (means bit for bit, decoded labels, fit state and log-likelihood)."""
    from backbone.hssm.fitting import fit_hssm
    logging.disable(logging.CRITICAL)
    job = dict(user=9201, start=0, end=70, fitseed=9201, K=3, n_init=3, max_iter=25, stratum="easy")
    w = L.stream_for(job).window(0, 70)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        rs = L.fit_restarts(w["matrix"], 3, 9201, 3, 25)
        res = fit_hssm(matrix=w["matrix"], candidate_ks=(3,), random_seed=9201, user_id="u", dataset_hash="h", n_initializations=3, max_iter=25, allow_fast_test_fit=True)
    sel = L.select_restart(rs)
    assert sel is not None
    assert np.array_equal(np.array(res.model.mu), np.array(sel["z_means"]))
    assert np.array_equal(res.p_t_posterior.argmax(axis=1), L.str_to_labels(sel["labels"]))
    assert float(res.model.log_likelihood_) == sel["ll"] and str(res.fit_state.value) == sel["state"]
