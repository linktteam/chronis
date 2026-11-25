"""WP8 Items 6 and 8: study library. Replays recorded fits (records of `identity_lib.fit_window`) through the PRODUCTION lineage (`backbone/hssm/lineage.step`, whatever defaults it has
at the time) and scores every step against the planted truth, with the definitions of tests/PREDECLARED_CRITERIA_WP8.md (section 2 and the Item 4 pre-declaration). Used by study scripts only.
Tested by tests/test_wp8_item8_scoring.py (mutation check: integration_report/wp8/scripts/mutation_item8_scoring.py)."""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identity_lib import overlap_map, stream_for                                   # noqa: E402

from backbone.hssm import lineage as L                                             # noqa: E402
from backbone.hssm.lineage import ALIGNED, K_CHANGED, NEW_LINEAGE, AlignmentConfig, make_summary, step  # noqa: E402
from data.synthetic_identity import REGISTRY                                       # noqa: E402

NOT_USABLE_STATES = ("FAILED", "NUMERICAL_FAILURE")      # a fit that failed numerically carries no regime summary worth matching (as in the Item 4 analysis)


def usable(rec: dict) -> bool:
    return bool(rec.get("ok")) and rec.get("fit_state") not in NOT_USABLE_STATES


# --------------------------------------------------------------------------------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson 95% interval for k successes out of n (NaN, NaN when n = 0)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))                   # float noise must not print -0.0% or 100.0000001%


def rate(k: int, n: int) -> str:
    """'k/n = x% (Wilson lo-hi%)', or '0/0' when there is nothing to count."""
    if n == 0:
        return "0/0"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100.0 * k / n:.1f}% (95% CI {100 * lo:.1f}-{100 * hi:.1f}%)"


def verdict(k: int, n: int, thr: float, kind: str) -> str:
    """kind '>=': the rate must be at least thr; '<=': at most thr. Met on the point estimate; NOT DEMONSTRATED when the Wilson interval crosses thr; UNKNOWN when n = 0."""
    if n == 0:
        return "UNKNOWN (nothing to count)"
    p = k / n
    if kind not in (">=", "<="):
        raise ValueError("kind must be '>=' or '<='")
    lo, hi = wilson(k, n)
    if kind == ">=":
        if p < thr:
            return "NOT MET"
        return "MET" if lo >= thr else "MET on the point estimate, NOT DEMONSTRATED (the 95% interval reaches below the threshold)"
    if p > thr:
        return "NOT MET"
    return "MET" if hi <= thr else "MET on the point estimate, NOT DEMONSTRATED (the 95% interval reaches above the threshold)"


def day_agreement(a, b) -> float:
    """Share of days on which two per-day label vectors of the same length are equal."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("per-day label vectors must have the same non-zero length")
    return float((a == b).mean())


# --------------------------------------------------------------------------------------------------------------------------------------------- fits
_streams: dict = {}


def raw_window(job: dict) -> np.ndarray:
    key = (job["user"], job.get("drift", 0.0), job.get("kchange"), job.get("n_days", 150))
    if key not in _streams:
        _streams[key] = stream_for(job)
    return _streams[key].window(job["start"], job["end"])["raw"]


@dataclass
class FitInfo:
    engine: str
    K: int
    mp: list                 # planted regime of every fit regime (largest day overlap), -1 if none
    recoverable: bool
    accepted: bool
    window: tuple
    user: int
    raw_labels: np.ndarray


def fit_info(rec: dict, k_true: int = 4) -> FitInfo:
    mp, recoverable = overlap_map(rec["labels_fit"], rec["labels_true"], rec["K"], k_true)
    j = rec["job"]
    return FitInfo(j["engine"], int(rec["K"]), mp, bool(recoverable), bool(rec.get("accepted")), (j["start"], j["end"]), j["user"], np.asarray(rec["labels_fit"], dtype=int))


def summary_for(rec: dict, window_raw: np.ndarray | None = None):
    """FitSummary of one recorded fit, built the way `lineage.summarize_fit` builds it from a live result (regime means in raw units, the fit's scaler, the decoded labels, the raw window)."""
    j = rec["job"]
    wr = raw_window(j) if window_raw is None else window_raw
    return make_summary(j["engine"], np.asarray(rec["raw_means"], dtype=float), np.asarray(rec["scaler_mean"], dtype=float), np.asarray(rec["scaler_std"], dtype=float),
                        np.asarray(rec["labels_fit"], dtype=int), wr, exp_durations=np.asarray(rec["exp_dur"], dtype=float))


# --------------------------------------------------------------------------------------------------------------------------------------------- scoring one step
def alignment_correct(anchor_info: FitInfo, info: FitInfo, anchor_r2s: tuple, new_r2s: tuple):
    """Permutation-correct (pre-declaration Item 4, point 1): both fits recoverable and every new regime j, which the record names with the stable id s, has the planted regime that
    the anchor regime named s has. Returns (correct or None when K differs, both_recoverable)."""
    both = bool(anchor_info.recoverable and info.recoverable)
    if anchor_info.K != info.K:
        return None, both
    stable_to_anchor_raw = {s: i for i, s in enumerate(anchor_r2s)}
    good = all(anchor_info.mp[stable_to_anchor_raw[new_r2s[j]]] == info.mp[j] for j in range(info.K))
    return bool(both and good), both


def candidates_as_planted(cands, anchor_info: FitInfo, anchor_r2s: tuple) -> list:
    """The split and merge candidates of a K_CHANGED step in planted-regime terms: an old stable id is translated to the planted regime the anchor's regime of that id corresponds to
    (-1 when the anchor fit does not map it). A split candidate becomes dict(kind, planted=[p]), a merge candidate dict(kind, planted=[p, q, ...])."""
    def planted(stable_id):
        raw = list(anchor_r2s).index(stable_id) if stable_id in anchor_r2s else None
        return -1 if raw is None else int(anchor_info.mp[raw])
    out = []
    for c in cands:
        if c["kind"] == "split":
            out.append(dict(kind="split", planted=[planted(c["old_stable_id"])]))
        else:
            out.append(dict(kind="merge", planted=sorted(planted(i) for i in c["old_stable_ids"])))
    return out


def score_step(lr, state_before, infos: list, info: FitInfo) -> dict:
    """One row per fed fit. `correct` is the correctness of the alignment the lineage computed (for a flagged step: of its best guess), None for the first fit and for K_CHANGED."""
    row = dict(fit_index=lr.fit_index, engine=info.engine, window=list(info.window), K=info.K, accepted=info.accepted, recoverable=info.recoverable, state=lr.continuity_state,
               reasons=list(lr.reasons), margin_diff=lr.margin_diff, margin_ratio=lr.margin_ratio, separation_min=lr.separation_min, correct=None, both_recoverable=None, served=lr.continuity_state == ALIGNED)
    if state_before is not None:
        ai = infos[state_before.anchor.fit_index]
        row["K_anchor"] = ai.K
        if lr.continuity_state != K_CHANGED:
            row["correct"], row["both_recoverable"] = alignment_correct(ai, info, state_before.anchor.raw_to_stable, tuple(lr.raw_to_stable))
            row["anchor_engine"] = ai.engine
        else:
            row["candidates"] = candidates_as_planted(getattr(lr, "split_merge_candidates", ()), ai, state_before.anchor.raw_to_stable)
    return row


def replay(recs: list, config: AlignmentConfig | None = None, registry=REGISTRY, lineage_id: str = "study", accepted_only: bool = True, patch=None):
    """Feed `recs` (in feed order) through the production `step`. Returns (rows, lineage records, final state). `accepted_only` drops fits the publish rule withholds (the criterion);
    False feeds every fit that did not fail numerically (the labelled secondary measurement). `patch` is an optional context manager used by the negative controls."""
    import contextlib
    cfg = AlignmentConfig() if config is None else config
    ctx = patch() if patch is not None else contextlib.nullcontext()
    rows, lrs, infos, state = [], [], [], None
    with ctx:
        for rec in recs:
            if not usable(rec) or (accepted_only and not rec.get("accepted")):
                continue
            info = fit_info(rec)
            lr, new_state = step(state, summary_for(rec), cfg, registry, lineage_id if state is None else None)
            rows.append(score_step(lr, state, infos, info))
            lrs.append(lr)
            infos.append(info)
            state = new_state
    return rows, lrs, state


# --------------------------------------------------------------------------------------------------------------------------------------------- criteria on rows
def tally(rows: list) -> dict:
    """Counts of the steps that carry an alignment (same K, not the first fit): served, correct, wrong-and-unflagged, flagged, and the recoverability split."""
    al = [r for r in rows if r["correct"] is not None]
    served = [r for r in al if r["served"]]
    wrong_unflagged = [r for r in served if not r["correct"]]
    cond = [r for r in al if r["both_recoverable"]]
    return dict(alignments=len(al), served=len(served), served_correct=len(served) - len(wrong_unflagged), wrong_unflagged=len(wrong_unflagged), flagged=len(al) - len(served),
                flagged_would_be_wrong=sum(1 for r in al if not r["served"] and not r["correct"]), flagged_would_be_right=sum(1 for r in al if not r["served"] and r["correct"]),
                cond_alignments=len(cond), cond_correct=sum(1 for r in cond if r["correct"]), steps=len(rows), k_changed=sum(1 for r in rows if r["state"] == K_CHANGED))


def stable_ids_first_days(lrs: list, n_days: int):
    """Per fit, the per-day stable ids of days 0..n_days-1 (windows of protocol G all start at day 0)."""
    return [np.asarray(lr.stable_regime_id)[:n_days] for lr in lrs]


# --------------------------------------------------------------------------------------------------------------------------------------------- refit cadence
def schedule_days(interval: int, first: int = 90, last: int = 150) -> list:
    """Window ends (days) of a refit schedule, from the production `RefitPolicy` / `refit_schedule`, with the final window `last` always added so that every schedule ends on the same fit."""
    from backbone.hssm.refit_policy import RefitPolicy, refit_schedule
    days = refit_schedule(first, last, RefitPolicy(interval))
    return sorted(set(days) | {last})


def fits_for_days(recs: list, days: list, key=lambda r: r["job"]["end"]) -> list:
    """The records whose window end is in `days`, in that order; raises when a day is missing (a schedule must not silently lose a fit)."""
    by = {key(r): r for r in recs}
    missing = [d for d in days if d not in by]
    if missing:
        raise KeyError(f"no fit for window ends {missing}")
    return [by[d] for d in days]
