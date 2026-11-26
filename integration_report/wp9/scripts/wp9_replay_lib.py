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


# ------------------------------------------------------------------------------------------------- replay through the production lineage
import lineage_study_lib as S                                                      # noqa: E402  (WP8 scoring: fit_info, score_step, tally, rate, verdict, wilson, day_agreement)

from backbone.hssm.fit_stability import assess_consensus                           # noqa: E402
from backbone.hssm.fitting import adjusted_rand_index                              # noqa: E402
from backbone.hssm.lineage import AlignmentConfig, make_summary, step              # noqa: E402
from data.synthetic_identity import REGISTRY                                       # noqa: E402

_streams: dict = {}


def raw_window9(job: dict) -> np.ndarray:
    key = (job["user"], job.get("stratum", "wp8"), job.get("drift", 0.0), job.get("kchange"), job.get("n_days", 150), repr(job.get("scale_change")), repr(job.get("regime_change")))
    if key not in _streams:
        _streams[key] = F.stream_for(job)
    return _streams[key].window(job["start"], job["end"])["raw"]


def consensus_of(rec: dict, n: int | None) -> dict:
    """The production consensus function (`fit_stability.assess_consensus`) on the CONVERGED restarts of the first n restarts of a stored fit, in restart order."""
    rs = rec["restarts"] if n is None else rec["restarts"][:n]
    conv = [dict(ll=r["ll"], collapsed=r["collapsed"], labels=F.str_to_labels(r["labels"])) for r in rs if r.get("state") == "CONVERGED" and r.get("ll") is not None]
    return assess_consensus(conv, adjusted_rand_index)


_prep_cache: dict = {}


def prepared(rec: dict, n: int | None):
    """Everything about one stored fit that does not depend on the candidate gate: the WP8-format record, the production consensus, the raw window and the scoring info (cached)."""
    j = rec["job"]
    key = (j["user"], j["fitseed"], j["end"], j["start"], j.get("stratum"), n, repr(j.get("scale_change")), repr(j.get("regime_change")), j.get("kchange"), j.get("drift"), j.get("group"))
    if key not in _prep_cache:
        r8 = wp8_record(rec, n)
        if not S.usable(r8) or not r8.get("accepted"):
            _prep_cache[key] = None
        else:
            _prep_cache[key] = dict(r8=r8, cons=consensus_of(rec, n), raw=raw_window9(r8["job"]), info=S.fit_info(r8))
    return _prep_cache[key]


def replay9(chain: list, cfg: AlignmentConfig, n: int | None = None, use_flag: bool = True, sharpness: float = 0.0, lineage_id: str = "wp9"):
    """Feed the fits of ONE chain (one user, one fit seed; records of `wp9_fit_lib`, in feed order) through the production `lineage.step`. Fits the publish rule withholds are skipped (as in
    WP8). A fit is flagged FIT_UNSTABLE when the production consensus function says so (`use_flag`) or when its mean maximum posterior is below `sharpness` (the A2 candidate signal).
    Returns (rows, lineage records, final state, infos) with the WP8 row format of `lineage_study_lib.score_step` plus `flag`, `max_post`, `user`, `stratum`."""
    rows, lrs, infos, state = [], [], [], None
    for rec in chain:
        p = prepared(rec, n)
        if p is None:
            continue
        r8, cons, info = p["r8"], p["cons"], p["info"]
        flag = bool(use_flag and cons["fit_unstable"]) or (sharpness > 0.0 and r8["max_post"] < sharpness)
        j = r8["job"]
        summ = make_summary("baseline", r8["raw_means"], r8["scaler_mean"], r8["scaler_std"], r8["labels_fit"], p["raw"], exp_durations=r8["exp_dur"], fit_unstable=flag)
        lr, new_state = step(state, summ, cfg, REGISTRY, lineage_id if state is None else None)
        row = S.score_step(lr, state, infos, info)
        row.update(flag=flag, max_post=r8["max_post"], user=j["user"], stratum=j.get("stratum"), end=j["end"], consensus_min_ari=cons["consensus_min_ari"])
        rows.append(row); lrs.append(lr); infos.append(info); state = new_state
    return rows, lrs, state, infos


def chains_by_user(records: list, windows: set | None = None) -> dict:
    """{(user, fitseed): [records sorted by window end]}, optionally restricted to the window ends in `windows`."""
    out: dict = {}
    for r in records:
        j = r["job"]
        if windows is not None and j["end"] not in windows:
            continue
        out.setdefault((j["user"], j["fitseed"]), []).append(r)
    for k in out:
        out[k].sort(key=lambda r: r["job"]["end"])
    return out


# ------------------------------------------------------------------------------------------------- speed: memoise the ordering ladder (study scripts only)
from backbone.hssm import lineage as _lineage                                      # noqa: E402

_name_cache: dict = {}
_name_fit_orig = _lineage.name_fit


def _name_fit_cached(summary, registry, ladder):
    """`lineage.name_fit` is a deterministic function of the fit summary, the registry and the ladder config; the study replays the same fits under many gate settings, so the (slow) bootstrap of
    the ladder is computed once per distinct input. Study use only; `tests/test_wp9_replay_lib.py` checks that the cached result equals the direct one."""
    key = (summary.raw_means.tobytes(), summary.labels.tobytes(), None if summary.raw_window is None else summary.raw_window.tobytes(), repr(ladder), id(registry))
    if key not in _name_cache:
        _name_cache[key] = _name_fit_orig(summary, registry, ladder)
    return _name_cache[key]


def install_name_cache() -> None:
    _lineage.name_fit = _name_fit_cached


def uninstall_name_cache() -> None:
    _lineage.name_fit = _name_fit_orig


install_name_cache()
