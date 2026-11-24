"""WP7-R item 2 (P1-01): the BIC parameter count of the baseline model, and the opt-in `n_params_convention` switch (default "legacy" = unchanged)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from backbone.hssm.model import GaussianHSMM


def _fitted(K: int, F: int = 3, T: int = 40, seed: int = 3) -> GaussianHSMM:
    rng = np.random.default_rng(seed)
    path = np.tile(np.repeat(np.arange(K), 8), T // (8 * K) + 1)[:T]
    X = rng.normal(size=(T, F)) + 2.0 * np.eye(K, F)[path % K]
    m = GaussianHSMM(n_regimes=K, n_features=F, max_duration=12, seed=seed)
    m.fit(X, n_iter=8, verbose=False)
    return m


def test_default_count_is_the_legacy_count_bit_for_bit():
    for K, F, expected in ((2, 7, 35), (3, 7, 56), (4, 7, 79), (2, 3, 19)):
        m = _fitted(K, F=F) if F == 3 else _fitted(K, F=F, T=60)
        assert GaussianHSMM.n_params_convention == "legacy" and m.n_params() == expected == (K - 1) + K * (K - 1) + 2 * K * F + 2 * K


def test_free_count_equals_the_structural_count_of_a_fitted_model():
    for K in (2, 3, 4):
        m = _fitted(K)
        assert np.allclose(np.diag(m.A), 0.0), "premise of P1-01: the M-step keeps a zero transition diagonal"
        assert np.allclose(m.A.sum(axis=1), 1.0)
        free = (K - 1) + sum(int(np.count_nonzero(~np.eye(K, dtype=bool)[i])) - 1 for i in range(K)) + m.mu.size + m.var.size + 2 * K
        m.n_params_convention = "free"
        assert m.n_params() == free == (K - 1) + K * (K - 2) + 2 * K * m.F + 2 * K
        m.n_params_convention = "legacy"
        assert m.n_params() - free == K            # the legacy count over-counts by K


def test_bic_and_aic_follow_the_convention_and_nothing_else_changes():
    m = _fitted(3)
    n_obs = 50
    legacy_bic, legacy_aic = m.bic(n_observations=n_obs), m.aic()
    assert legacy_bic == -2 * m.log_likelihood_ + m.n_params() * math.log(n_obs)
    m.n_params_convention = "free"
    assert m.bic(n_observations=n_obs) == pytest.approx(legacy_bic - 3 * math.log(n_obs))
    assert m.aic() == pytest.approx(legacy_aic - 2 * 3)
    m.n_params_convention = "legacy"
    assert m.bic(n_observations=n_obs) == legacy_bic and m.aic() == legacy_aic


def test_unknown_convention_is_rejected():
    m = _fitted(2)
    m.n_params_convention = "kk1"
    with pytest.raises(ValueError, match="n_params_convention"):
        m.n_params()


def test_the_switch_reaches_the_k_selection_bic(monkeypatch):
    from backbone.hssm.fitting import fit_hssm
    rng = np.random.default_rng(5)
    X = np.vstack([rng.normal(0, 0.6, (14, 3)) + [2, 0, 0], rng.normal(0, 0.6, (14, 3)) + [0, 2, 0]] * 2)
    days = np.arange(len(X), dtype=float)
    kw = dict(matrix=X, candidate_ks=(2, 3), n_initializations=3, max_iter=60, random_seed=4, timestamps=days, user_id="u", dataset_hash="h", allow_fast_test_fit=True)
    legacy = fit_hssm(**kw).bic_by_k
    monkeypatch.setattr(GaussianHSMM, "n_params_convention", "free")
    free = fit_hssm(**kw).bic_by_k
    finite = [k for k in (2, 3) if math.isfinite(legacy[k])]
    assert finite, legacy                      # at least one candidate converged, otherwise the test would say nothing
    for k in finite:
        assert legacy[k] - free[k] == pytest.approx(k * math.log(len(X)), abs=1e-6), (k, legacy, free)


def test_the_contract_doc_states_the_count_the_default_and_the_switch():
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[1] / "docs" / "INTEGRATION_CONTRACT_HSSM.md").read_text()
    sec = doc.split("### 9.8 BIC parameter count and model selection")[1]
    assert "(K-1) + K(K-1) + 2KF + 2K" in sec and "`K(K-2)` free entries" in sec and "too large by **K**" in sec
    assert '`n_params_convention = "legacy"`' in sec and '`GaussianHSMM.n_params_convention = "free"`' in sec and "default is unchanged" in sec
    assert "needs the owner's approval" in sec
    assert GaussianHSMM.n_params_convention == "legacy"            # what the doc says is the default
