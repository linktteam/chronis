"""WP9 A4: the declared reading of integration_report/wp9/scripts/a4_analyze.py on fabricated rows (no fits)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integration_report" / "wp9" / "scripts"))
import a4_analyze as A                                           # noqa: E402


def rows(n_served_right, n_served_wrong, n_flagged):
    out = []
    for i in range(n_served_right):
        out.append(dict(user=i, end=75, state='ALIGNED', served=True, correct=True, both_recoverable=True))
    for i in range(n_served_wrong):
        out.append(dict(user=1000 + i, end=75, state='ALIGNED', served=True, correct=False, both_recoverable=True))
    for i in range(n_flagged):
        out.append(dict(user=2000 + i, end=75, state='CONTINUITY_UNVERIFIED', served=False, correct=True, both_recoverable=True))
    return out


def test_metrics_and_the_declared_levels():
    m = A.metrics(rows(48, 0, 12))
    assert (m["alignments"], m["served"], m["served_correct"], m["wrong_unflagged"]) == (60, 48, 48, 0) and m["coverage"] == pytest.approx(0.8)
    assert A.raw_works(m) is True and A.z_fails(m) is False


def test_raw_works_needs_enough_alignments_and_the_correctness_levels():
    assert A.raw_works(A.metrics(rows(30, 0, 5))) is False                          # 35 < 40 alignments
    assert A.raw_works(A.metrics(rows(94, 6, 0))) is False                          # 94% correct among served
    assert A.raw_works(A.metrics(rows(95, 5, 0))) is False                          # 5% wrong and unflagged
    assert A.raw_works(A.metrics(rows(500, 4, 0))) is True                          # 99.2% correct, 0.8% wrong-unflagged
    assert A.raw_works(A.metrics(rows(0, 0, 50))) is False                          # nothing served


def test_z_fails_only_through_correctness_not_through_coverage():
    assert A.z_fails(A.metrics(rows(10, 0, 90))) is False                           # low coverage alone is not a failure (declared)
    assert A.z_fails(A.metrics(rows(40, 10, 0))) is True                            # 80% correct among served
    assert A.z_fails(A.metrics(rows(500, 4, 0))) is False
    assert A.z_fails(A.metrics(rows(500, 6, 0))) is True                            # 1.2% wrong-and-unflagged
    assert A.z_fails(A.metrics([])) is False


def test_paired_counts_follow_the_same_user_and_window():
    r = [dict(user=1, end=75, served=True, correct=True), dict(user=2, end=75, served=True, correct=True), dict(user=3, end=75, served=False, correct=True)]
    z = [dict(user=1, end=75, served=True, correct=False), dict(user=2, end=75, served=True, correct=True), dict(user=3, end=75, served=True, correct=False)]
    p = A.paired(r, z)
    assert p == dict(common=3, z_wrong_r_right=1, r_wrong_z_right=0, z_wrong=2, r_wrong=0)


def test_the_z_space_patch_is_restored_and_uses_the_fits_own_scaler():
    from backbone.hssm import lineage as L
    old = L._vectors
    with A.patch_zspace():
        assert L._vectors is not old
        s = type("S", (), dict(raw_means=np.array([[10.0, 20.0]]), scaler_mean=np.array([4.0, 10.0]), scaler_std=np.array([2.0, 0.0])))()
        assert np.allclose(L._vectors(s, None, 0.0), [[3.0, 10.0]])                  # (raw - mean) / sd, a zero sd counts as 1
    assert L._vectors is old


def test_the_run_script_shares_the_windows_that_contain_no_changed_day():
    import a4_run as R
    jobs = R.jobs_for([9261], ["A", "B", "C", "none"])
    shared = [j for j in jobs if j["group"] == "A4-shared"]
    assert sorted(j["end"] for j in shared) == [45, 60] and all(j["scale_change"] is None for j in shared)
    assert len(jobs) == 2 + 6 * 4
    assert all(j["start"] == j["end"] - 45 and j["K"] == 3 and j["stratum"] == "easy" for j in jobs)
    assert {j["group"] for j in jobs if j["end"] > 60} == {"A4-A", "A4-B", "A4-C", "A4-none"}
    assert all((j["scale_change"] is None) == (j["group"] == "A4-none") for j in jobs if j["end"] > 60)
