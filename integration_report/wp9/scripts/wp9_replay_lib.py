"""WP9 replay library: turns WP9 fit records (all restarts kept, `wp9_fit_lib`) into WP8-format records (`identity_lib.fit_window`) for the selected restart, so that the production
lineage (`backbone/hssm/lineage.step`) and the WP8 scoring (`lineage_study_lib`) can be replayed on them unchanged. Study use only.

`accepted` of the baseline engine = the production publish rule for a baseline fit: the selected restart is CONVERGED and not collapsed (`HSSMResult.withheld_reason`: the other causes
- degenerate duration, Cholesky fallbacks - exist for the dual engine only). `tests/test_wp9_replay_lib.py` checks this against `fit_hssm` on live fits.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp8" / "scripts"))

import wp9_fit_lib as F                                                            # noqa: E402


def wp8_record(rec: dict, n: int | None = None, restart: dict | None = None) -> dict:
    """WP8-format record of one WP9 fit record for the production-selected restart among the first `n` (or for an explicitly given `restart`). A fit whose restarts all failed to converge
    returns ok = False (production raises FittingConvergenceError)."""
    job = dict(rec["job"], engine="baseline")
    out = dict(job=job, ok=False, seconds=rec.get("seconds"))
    if not rec.get("ok"):
        out["error"] = rec.get("error", "job failed")
        return out
    sel = restart if restart is not None else F.select_restart(rec["restarts"], n)
    if sel is None:
        out["error"] = "FittingConvergenceError: no restart converged"
        return out
    z = np.asarray(sel["z_means"], dtype=float)
    sm = np.asarray(rec["scaler_mean"], dtype=float)
    ss = np.asarray(rec["scaler_std"], dtype=float)
    # expected duration per regime, in the canonical order (the library stores them canonical)
    out.update(ok=True, K=int(z.shape[0]), z_means=z.tolist(), raw_means=(sm + ss * z).tolist(), scaler_mean=sm.tolist(), scaler_std=ss.tolist(), coverage=rec["coverage"],
               exp_dur=sel["exp_dur"], labels_fit=F.str_to_labels(sel["labels"]).tolist(), labels_true=F.str_to_labels(rec["labels_true"]).tolist(), fit_state=sel["state"],
               accepted=bool(not sel["collapsed"]), withheld_reason=None if not sel["collapsed"] else "collapsed_regime", log_likelihood=sel["ll"], max_post=sel["max_post"],
               restart=sel["i"])
    return out
