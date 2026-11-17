"""HSSM contract smoke run (WP7-R item 0). Prints every field that docs/INTEGRATION_CONTRACT_HSSM.md section 9 tells an integrator to read.

    PYTHONPATH=. python3 scripts/hssm_contract_smoke.py          # about 30 s on 2 shared cores; exit code 0 when every contract check holds

Planted case: K = 2, F = 4, T = 60 rows (segments of 12 to 18 rows), z-scored, calendar-day timestamps; baseline model with 10 restarts (the minimum
without `allow_fast_test_fit`), then the dual-timescale model at 15 iterations (it does NOT converge: the point is the withheld contract), then the error paths.
`run()` returns a dict of the facts so that the doc tests can assert them; `main()` prints them.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

from backbone.hssm.fitting import fit_hssm
from backbone.hssm.gating import ColdStartError
from backbone.hssm.model import FitState


def planted(seed: int = 11, T: int = 60, K: int = 2, F: int = 4):
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, 1.0, (K, F))
    means *= 2.5 / np.abs(means).max()
    path, cur = [], 0
    while len(path) < T:
        path += [cur] * int(rng.integers(12, 19))
        cur = (cur + 1) % K
    path = np.array(path[:T])
    X = means[path] + rng.normal(0.0, 0.5, (T, F))
    return (X - X.mean(0)) / X.std(0), path


def _contract_fields(r) -> dict:
    meta = r.convergence_metadata or {}
    po = r.publishable_outputs()
    return {
        "fit_state": r.fit_state.value if isinstance(r.fit_state, FitState) else str(r.fit_state),
        "model_fit_state": meta.get("fit_state"),
        "convergence_criterion": meta.get("convergence_criterion"),
        "insights_withheld": bool(r.insights_withheld),
        "withheld_reason": r.withheld_reason,
        "publishable_outputs_is_none": po is None,
        "fit_degraded": bool(r.fit_degraded),
        "fit_cholesky_fallbacks": int(r.fit_cholesky_fallbacks),
        "fit_collapsed": bool(r.fit_collapsed),
        "collapse_keys": sorted((r.collapse_report or {}).keys()),
        "restart_summary_keys": sorted((r.restart_summary or {}).keys()),
        "duration_at_bound": dict(r.duration_at_bound or {}),
        "duration_box_keys": sorted((meta.get("duration_box") or {}).keys()),
        "used_shared_prior": bool(r.used_shared_prior),
        "shared_prior_n0": r.shared_prior_n0,
        "K": int(r.K),
        "m_t_present": r.m_t_posterior_mean is not None,
        "m_t_uncertainty_present": getattr(r, "m_t_uncertainty", None) is not None,
        "has_session_confidence_field": any(n in dir(r) for n in ("confidence", "session_confidence", "p_t_confidence")),
    }


def run() -> dict:
    out: dict = {}
    X, _path = planted()
    days = np.arange(len(X), dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rb = fit_hssm(matrix=X, candidate_ks=(2,), n_initializations=10, max_iter=60, random_seed=7, timestamps=days, user_id="smoke", dataset_hash="smoke")
        out["baseline"] = _contract_fields(rb)
        rd = fit_hssm(matrix=X, candidate_ks=(2,), n_initializations=1, max_iter=15, random_seed=7, timestamps=days, user_id="smoke", dataset_hash="smoke",
                      allow_fast_test_fit=True, model_type="dual_timescale", latent_dim=2, max_duration=25)
        out["dual_short"] = _contract_fields(rd)
        errors = {}
        try:
            fit_hssm(matrix=X[:20], candidate_ks=(2,), n_initializations=10, max_iter=10, random_seed=7, user_id="smoke", dataset_hash="smoke")
            errors["cold_start"] = "NO ERROR"
        except ColdStartError as exc:
            errors["cold_start"] = type(exc).__name__
        try:
            fit_hssm(matrix=X, candidate_ks=(2,), n_initializations=10, max_iter=10, random_seed=7, timestamps=days * 86400.0, user_id="smoke", dataset_hash="smoke")
            errors["seconds_as_days"] = "NO ERROR"
        except ValueError as exc:
            errors["seconds_as_days"] = "ValueError: " + str(exc)[:70]
        try:
            fit_hssm(matrix=X, candidate_ks=(2,), n_initializations=3, max_iter=10, random_seed=7, user_id="smoke", dataset_hash="smoke")
            errors["too_few_restarts"] = "NO ERROR"
        except ValueError as exc:
            errors["too_few_restarts"] = "ValueError"
        out["errors"] = errors
    return out


def main() -> int:
    out = run()
    for name in ("baseline", "dual_short"):
        print(f"[{name}]")
        for k, v in out[name].items():
            print(f"  {k}: {v}")
    print("[error paths]")
    for k, v in out["errors"].items():
        print(f"  {k}: {v}")
    b, d, e = out["baseline"], out["dual_short"], out["errors"]
    checks = {
        "baseline served only through publishable_outputs": (not b["insights_withheld"]) == (not b["publishable_outputs_is_none"]),
        "dual at 15 iterations is withheld as not converged": d["insights_withheld"] and "fit_state_not_converged" in str(d["withheld_reason"]) and d["publishable_outputs_is_none"],
        "no per-session confidence field": not b["has_session_confidence_field"] and not d["has_session_confidence_field"],
        "cold start raises ColdStartError": e["cold_start"] == "ColdStartError",
        "seconds-as-days raises ValueError": e["seconds_as_days"].startswith("ValueError"),
        "fewer than 10 restarts raises ValueError": e["too_few_restarts"] == "ValueError",
    }
    for k, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + k)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
