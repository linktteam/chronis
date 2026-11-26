"""WP9 A1/A2: the study replay (`integration_report/wp9/scripts/wp9_replay_lib.py`) and the scoring in `a12_analyze.py`. Mutation check: integration_report/wp9/scripts/mutation_a12_scoring.py."""
from __future__ import annotations

import contextlib
import io
import logging
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "integration_report" / "wp9" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import a12_analyze as A                                                                 # noqa: E402
import wp9_fit_lib as F                                                                 # noqa: E402
import wp9_replay_lib as R                                                              # noqa: E402

from backbone.hssm.fitting import fit_hssm                                              # noqa: E402
from backbone.hssm.lineage import AlignmentConfig, make_summary                         # noqa: E402
from backbone.hssm.ordering import LadderConfig                                         # noqa: E402
from data.synthetic_identity import REGISTRY, IdentitySpec, draw_user, generate_identity_stream  # noqa: E402

LAB = "0" * 10 + "1" * 10 + "2" * 10


def restart(i, ll, labels=LAB, state="CONVERGED", collapsed=False):
    r = dict(i=i, seed=i, state=state, n_iter=5, ll=ll, collapsed=collapsed, min_eff=9.0)
    if state == "CONVERGED":
        r.update(labels=labels, z_means=[[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]], max_post=0.9, eff=[10.0, 10.0, 10.0], exp_dur=[5.0, 5.0, 5.0], dmax=12)
    return r


def fit_rec(restarts, ok=True):
    return dict(job=dict(user=1, start=0, end=30, fitseed=1, K=3, stratum="easy"), ok=ok, restarts=restarts, scaler_mean=[0.0, 0.0], scaler_std=[1.0, 1.0], coverage=[1.0, 1.0],
                labels_true=LAB, seconds=1.0)


# ------------------------------------------------------------------------------------------------------------------- the adapter
def test_wp8_record_uses_the_production_selection_and_the_publish_rule():
    rec = fit_rec([restart(0, -10.0, collapsed=True), restart(1, -12.0), restart(2, -11.0)])
    r8 = R.wp8_record(rec)
    assert r8["ok"] and r8["restart"] == 2 and r8["accepted"] is True and r8["job"]["engine"] == "baseline" and r8["K"] == 3 and r8["fit_state"] == "CONVERGED"
    assert r8["labels_fit"] == [0] * 10 + [1] * 10 + [2] * 10 and r8["raw_means"] == r8["z_means"]                  # scaler mean 0, std 1
    assert R.wp8_record(rec, n=2)["restart"] == 1                                                                     # only the first two restarts are looked at
    only_collapsed = R.wp8_record(fit_rec([restart(0, -10.0, collapsed=True), restart(1, -12.0, collapsed=True)]))
    assert only_collapsed["ok"] and only_collapsed["accepted"] is False and only_collapsed["withheld_reason"] == "collapsed_regime"
    assert R.wp8_record(fit_rec([restart(0, None, state="FAILED")]))["ok"] is False
    assert R.wp8_record(fit_rec([], ok=False))["ok"] is False
    assert R.wp8_record(rec, restart=rec["restarts"][1])["restart"] == 1


def test_the_scaler_is_applied_to_the_regime_means():
    rec = fit_rec([restart(0, -10.0)])
    rec["scaler_mean"], rec["scaler_std"] = [10.0, 20.0], [2.0, 3.0]
    r8 = R.wp8_record(rec)
    assert np.allclose(r8["raw_means"], np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]]) * [2.0, 3.0] + [10.0, 20.0])


def test_consensus_of_uses_the_production_function_on_the_converged_restarts_in_order():
    other = "0" * 9 + "1" * 11 + "2" * 10
    rec = fit_rec([restart(0, -10.0), restart(1, -11.0, state="MAX_ITER_REACHED"), restart(2, -12.0), restart(3, -13.0, labels=other), restart(4, -9.0)])
    c = R.consensus_of(rec, None)
    assert c["n_healthy_restarts"] == 4 and c["fit_unstable"] is False                                                # best three (-9, -10, -12) agree; the fourth healthy one is not compared
    assert R.consensus_of(rec, 3)["fit_unstable"] is True and R.consensus_of(rec, 3)["n_healthy_restarts"] == 2     # only restarts 0 and 2 are converged among the first three
    rec2 = fit_rec([restart(0, -10.0), restart(1, -11.0), restart(2, -9.0, labels=other)])
    assert R.consensus_of(rec2, None)["fit_unstable"] is True and R.consensus_of(rec2, None)["fit_unstable_reason"] == "best_restarts_disagree"


# ------------------------------------------------------------------------------------------------------------------- the memoised ladder
def test_the_cached_name_fit_equals_the_direct_one():
    d = draw_user(3, IdentitySpec(n_regimes=3, sep_lo=2.0, sep_hi=2.0))["means"]
    rng = np.random.default_rng(0)
    labels = rng.permutation(np.repeat(np.arange(3), 40))
    window = d[labels] + 0.3 * rng.standard_normal((120, d.shape[1]))
    s = make_summary("baseline", d, window.mean(axis=0), window.std(axis=0), labels, window)
    ladder = LadderConfig(n_bootstrap=20)
    a = R._name_fit_orig(s, REGISTRY, ladder)
    b = R._name_fit_cached(s, REGISTRY, ladder)
    c = R._name_fit_cached(s, REGISTRY, ladder)                                                                        # second call: from the cache
    assert tuple(a[0]) == tuple(b[0]) == tuple(c[0]) and a[1:3] == b[1:3] == c[1:3]
    assert (a[3].as_dict() if a[3] is not None else None) == (b[3].as_dict() if b[3] is not None else None)


# ------------------------------------------------------------------------------------------------------------------- live equivalence with the production fit (accepted flag, labels)
@pytest.mark.parametrize("seed,K,end", [(9990, 3, 60), (9991, 6, 40)])
def test_accepted_and_labels_agree_with_fit_hssm(seed, K, end):
    logging.disable(logging.CRITICAL)
    job = dict(user=seed, start=0, end=end, fitseed=seed, K=K, n_init=10, max_iter=25, stratum="easy", drift=0.0, kchange=None)
    rec = F.fit_job(job)
    assert rec["ok"]
    st = F.stream_for(job)
    w = st.window(0, end)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        try:
            r = fit_hssm(matrix=w["matrix"], candidate_ks=(K,), n_initializations=10, max_iter=25, random_seed=seed, user_id="t", dataset_hash="h")
        except Exception as exc:
            assert F.select_restart(rec["restarts"]) is None, exc                                                      # production raised: no restart converged
            assert R.wp8_record(rec)["ok"] is False
            return
    r8 = R.wp8_record(rec)
    assert r8["ok"] and r8["accepted"] == (not r.insights_withheld)
    assert r8["labels_fit"] == np.asarray(r.p_t_posterior).argmax(axis=1).tolist() and np.array_equal(np.array(r8["z_means"]), np.array(r.model.mu))


# ------------------------------------------------------------------------------------------------------------------- the scoring
def lr_(ids, fu=False, afu=False):
    return SimpleNamespace(stable_regime_id=np.asarray(ids), fit_unstable=fu, anchor_fit_unstable=afu)


def row(end, state, flag=False):
    return dict(end=end, state=state, flag=flag)


def run_ident(monkeypatch, a_rows, b_rows):
    """identical_lineage with the replay replaced by prepared rows: a_rows / b_rows = lists of (row, lineage record) for the 14-day windows of user 1."""
    calls = iter([([x[0] for x in a_rows], [x[1] for x in a_rows], None, None), ([x[0] for x in b_rows], [x[1] for x in b_rows], None, None)])
    monkeypatch.setattr(A.R, "replay9", lambda *args, **kw: next(calls))
    chain = [dict(job=dict(user=1, end=e, fitseed=1)) for e in A.B_T]
    chain_b = [dict(job=dict(user=1, end=e, fitseed=1001)) for e in A.B_T]
    return A.identical_lineage({(1, 1): chain}, {(1, 1001): chain_b}, AlignmentConfig(), 10)


ID0, ID1 = [0] * 50 + [1] * 50, [0] * 50 + [1] * 49 + [2]


def test_identical_lineage_counts_a_case_when_at_least_one_seed_serves(monkeypatch):
    a = [(row(90, "NEW_LINEAGE"), lr_(ID0)), (row(104, "ALIGNED"), lr_(ID0)), (row(118, "ALIGNED"), lr_(ID0)), (row(132, "CONTINUITY_UNVERIFIED", True), lr_(ID0, fu=True))]
    b = [(row(90, "NEW_LINEAGE"), lr_(ID0)), (row(104, "ALIGNED"), lr_(ID0)), (row(118, "CONTINUITY_UNVERIFIED", True), lr_(ID0, fu=True)), (row(132, "ALIGNED"), lr_(ID0))]
    out = run_ident(monkeypatch, a, b)
    assert out["strict_cases"] == 3 and out["strict_identical"] == 1                          # 104 both served and the same; 118 and 132 served by one seed only
    assert out["wp8_cases"] == 4 and out["wp8_identical"] == 2                                # 90 (same state, same ids) and 104; 118 and 132 differ in state
    assert out["exactly_one_seed_flagged"] == 2 and out["aligned_steps"] == 4 and out["aligned_with_flagged_fit_or_anchor"] == 0


def test_identical_lineage_needs_99_percent_of_days_to_agree_and_counts_flagged_aligned_steps(monkeypatch):
    a = [(row(90, "NEW_LINEAGE"), lr_(ID0)), (row(104, "ALIGNED"), lr_(ID0))]
    b = [(row(90, "NEW_LINEAGE"), lr_(ID0)), (row(104, "ALIGNED"), lr_(ID1))]                  # one day of 100 differs: agreement 0.99 -> still identical
    assert run_ident(monkeypatch, a, b)["strict_identical"] == 1
    c = [(row(90, "NEW_LINEAGE"), lr_(ID0)), (row(104, "ALIGNED"), lr_([1] * 50 + [0] * 50))]                      # swapped names: 0 agreement
    assert run_ident(monkeypatch, a, c)["strict_identical"] == 0
    bad = [(row(90, "NEW_LINEAGE"), lr_(ID0)), (row(104, "ALIGNED"), lr_(ID0, afu=True))]
    assert run_ident(monkeypatch, a, bad)["aligned_with_flagged_fit_or_anchor"] == 1


def test_the_calibration_half_is_the_even_user_numbers():
    assert A.half_of(9302) == "cal" and A.half_of(9303) == "val" and A.half_of(9360) == "cal"


def test_the_candidate_grid_is_the_declared_one():
    assert A.H_GRID == (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0) and A.S_GRID == (0.0, 0.85, 0.90, 0.95) and A.B_T == [90, 104, 118, 132, 146, 150]
    assert A.G_T == [90, 97, 104, 111, 118, 125, 132, 139, 146, 150]


def test_the_prepared_cache_keeps_fits_of_different_scale_change_streams_apart(monkeypatch):
    import wp9_replay_lib as RL
    calls = []
    monkeypatch.setattr(RL, "wp8_record", lambda rec, n=None, restart=None: (calls.append(rec["job"].get("scale_change")) or dict(ok=False)))
    monkeypatch.setattr(RL.S, "usable", lambda r: False)
    RL._prep_cache.clear()
    base = dict(user=1, fitseed=1, end=75, start=30, stratum="easy")
    RL.prepared(dict(job=dict(base, scale_change=None)), 10)
    RL.prepared(dict(job=dict(base, scale_change=dict(features=(5,), factor=0.1, start=60))), 10)
    RL.prepared(dict(job=dict(base, scale_change=None)), 10)                              # a repeat is served from the cache
    assert calls == [None, dict(features=(5,), factor=0.1, start=60)]
    RL._prep_cache.clear()
