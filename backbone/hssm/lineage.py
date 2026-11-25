"""WP8 Item 4 (owner decisions A3, A4, A6, A7, C2): regime identity across refits, both engines. Pure functions and additive result fields; no UI.

Problem: the raw regime ids of a fit mean nothing across fits (the same data refitted with a longer window or another seed names the regimes differently). This module gives
every accepted fit a SECOND set of names, the stable ids, and says how far that can be trusted. The raw record is never changed (A4).
  * Naming: the first accepted fit of a lineage is named by the ordering ladder (`ordering.py`, which says when it cannot be sure). Later fits INHERIT the names by alignment.
  * Alignment: Hungarian assignment between the anchor fit's regimes and the new fit's regimes on the regime means in RAW physical units (A6: the fit-time scaler is stored with
    every fit), squared distance after dividing by the anchor's per-feature scale. Every alignment stores its margins: `margin_diff = (C2 - C1) / K` and `margin_ratio = C1 / C2`
    (C1 best total cost, C2 best cost of any other one-to-one assignment).
  * States: ALIGNED (identifiable fits and the margin passes), CONTINUITY_UNVERIFIED (ambiguous, or a fit that is not identifiable), K_CHANGED (the fitted K differs from the anchor's;
    the margins are not consulted), NEW_LINEAGE (first fit, or the user chose new names).
  * A3: on CONTINUITY_UNVERIFIED and K_CHANGED the OLD names are kept, the result says "cannot confirm this is the same as before" and carries `proposed_new_names`; `user_choice` is
    None, KEEP_OLD or NEW_NAMES and None behaves as KEEP_OLD. NEW_NAMES starts a new epoch and the old-to-new mapping is kept in the state's history for good. The application makes the
    choice (`apply_user_choice`); nothing here decides for the user.
  * Fail closed: `may_claim_same_as_before(record)` is True only for ALIGNED. The anchor moves only on ALIGNED and NEW_LINEAGE. A class of pair whose threshold could not be
    calibrated on the development data (dual to dual, baseline to dual and back: `PairThreshold.calibrated = False`) is never ALIGNED (reason `threshold_not_calibrated`).
  * Two channels (A6): channel 1 is the physical one above (regime means in real units, interpretable through the feature registry). Channel 2 is OPTIONAL and contextual: a per-regime
    profile (non-negative, sums to 1) that the CALLER supplies, for example topic or scene proportions aggregated per day. Nothing here extracts topics or scenes. The combined cost is
    `w1 * d1 / s1 + w2 * d2 / s2` (d2 = squared Hellinger distance; s = the anchor's own mean distance between its regimes in that channel); the default `w2 = 0` is channel 1 alone,
    bit for bit. If each channel alone gives a confident alignment (its own margin passes its own threshold) and the two assignments differ, the state is CONTINUITY_UNVERIFIED.
    Contextual descriptors are sensitive personal data: they must come from a consented, aggregated, non-identifying pipeline (docs/INTEGRATION_CONTRACT_HSSM.md, 9.10).
  * Engines: the baseline's regime means are `mu`; the dual model's are `regime_observation_means(params)` (the stationary mean d_k + C_k m_k). Both are put into raw units with the
    scaler of the window the fit was made on, so a switch of engine is aligned exactly like a refit, with its own thresholds.
EXPERT_REVIEW(identity-distance): the matching distance, the identifiability gate and every threshold below are defined here and are awaiting expert sign-off (A7); every record says so.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from .ordering import LadderConfig, OrderingResult, order_regimes

ALIGNED = "ALIGNED"
CONTINUITY_UNVERIFIED = "CONTINUITY_UNVERIFIED"
K_CHANGED = "K_CHANGED"
NEW_LINEAGE = "NEW_LINEAGE"
STATES = (ALIGNED, CONTINUITY_UNVERIFIED, K_CHANGED, NEW_LINEAGE)
USER_CHOICES = (None, "KEEP_OLD", "NEW_NAMES")
MESSAGE_UNCONFIRMED = "cannot confirm this is the same as before"
EXPERT_REVIEW_STATUS = "PENDING"            # stays PENDING until the owner records an expert sign-off (A7)
MAX_MATCH_K = 6                             # the second-best assignment is found by enumeration, which is exact for K <= 6 (the same limit as shrinkage.match_regimes)
AMBIGUITY_RATIO = 0.5                       # the WP5 production rule of shrinkage.match_regimes; the ratio threshold is never more lenient than this
RECORD_COLUMNS = ("stable_regime_id", "lineage_id", "lineage_epoch", "continuity_state", "alignment_margin", "activity_rung", "scaler_mean", "scaler_std", "expert_review_status")


# EXPERT_REVIEW(identity-distance): the alignment acceptance thresholds, frozen in the Item 4 design freeze (tests/PREDECLARED_CRITERIA_WP8.md, section 8). `calibrated=False` means the
# development data could not support a value for this class of pair (fewer than 20 alignments passed the gate): an alignment of that class is never served as ALIGNED.
@dataclass(frozen=True)
class PairThreshold:
    rule: str = "ratio"              # "diff": margin_diff >= tau; "ratio": margin_ratio <= rho; "both"
    tau: float = 0.0
    rho: float = AMBIGUITY_RATIO
    calibrated: bool = True          # False = UNKNOWN threshold: fail closed (reason `threshold_not_calibrated`), whatever the margins are

    def __post_init__(self) -> None:
        if self.rule not in ("diff", "ratio", "both"):
            raise ValueError("rule must be 'diff', 'ratio' or 'both'")
        if self.tau < 0 or not (0.0 <= self.rho <= AMBIGUITY_RATIO):
            raise ValueError(f"tau must not be negative and rho must lie in [0, {AMBIGUITY_RATIO}] (never more lenient than the WP5 rule)")


# EXPERT_REVIEW(identity-distance): the channel weights (`w2 = 0` means the contextual channel is off) and the acceptance threshold of the contextual channel on its own.
@dataclass(frozen=True)
class ChannelConfig:
    w1: float = 1.0
    w2: float = 0.0
    context_threshold: PairThreshold = PairThreshold()

    def __post_init__(self) -> None:
        if self.w1 < 0 or self.w2 < 0 or (self.w1 == 0 and self.w2 == 0):
            raise ValueError("channel weights must not be negative and not both zero")


# EXPERT_REVIEW(identity-distance): the identifiability gate `min_separation`, the optional duration coordinate of the matching distance and the thresholds per pair class.
# FROZEN 2026-11-25 (Item 4 design freeze, 960 development fits, seeds 8201..8212; evidence wp8_evidence/item4_analysis.txt):
#   h = 4.0: the smallest grid value at which no alignment between two fits that both pass it was strict-wrong on development (coverage at h = 4.0: 37.3% of all baseline alignments,
#            well below the declared 80%: the shortfall is reported, not hidden by loosening the gate);
#   distance = raw units / anchor scaler, no duration coordinate (raw, dur0.5 and dur1 tied on coverage, ties go to the simpler one);
#   baseline pairs: rule (b), rho = 0.5 (the declared floor: every passing development alignment was correct, so the smallest tau is 0 and is vacuous);
#   dual pairs and cross-engine pairs: UNKNOWN (4 and 16 of 216 and 480 development alignments pass the gate, fewer than 20), so `calibrated=False`: never ALIGNED.
@dataclass(frozen=True)
class AlignmentConfig:
    min_separation: float = 4.0                  # h: a fit whose regimes are closer than this (within-regime sd units) is not identifiable
    duration_weight: float = 0.0                 # lambda: weight of log expected duration as one more coordinate of the matching distance (0 = off)
    same_engine: PairThreshold = PairThreshold("ratio", 0.0, AMBIGUITY_RATIO)                      # baseline to baseline
    cross_engine: PairThreshold = PairThreshold("ratio", 0.0, AMBIGUITY_RATIO, calibrated=False)   # baseline to dual and dual to baseline: UNKNOWN
    ladder: LadderConfig = LadderConfig()
    dual_engine: PairThreshold = PairThreshold("ratio", 0.0, AMBIGUITY_RATIO, calibrated=False)    # dual to dual: UNKNOWN
    channels: ChannelConfig = ChannelConfig()

    # EXPERT_REVIEW(identity-distance): validates the identifiability gate and the duration weight.
    def __post_init__(self) -> None:
        if self.min_separation < 0 or self.duration_weight < 0:
            raise ValueError("min_separation and duration_weight must not be negative")


@dataclass(frozen=True)
class FitSummary:
    """Engine-agnostic view of one accepted fit. `raw_means[k]` is regime k's mean vector in physical units; the scaler is the one of the fit window; `labels` are the decoded raw
    regime of every session of the window; `raw_window` (sessions x features, NaN = missing) is only needed to name a fit with the ordering ladder."""
    engine: str
    raw_means: np.ndarray
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    labels: np.ndarray
    separation_min: float
    raw_window: np.ndarray | None = None
    exp_durations: np.ndarray | None = None
    context_profiles: np.ndarray | None = None      # channel 2 (optional): (K, C), one non-negative row per regime that sums to 1, supplied by the caller

    @property
    def K(self) -> int:
        return int(self.raw_means.shape[0])


@dataclass(frozen=True)
class Anchor:
    summary: FitSummary
    raw_to_stable: tuple
    fit_index: int
    activity_rung: int
    ordering_verified: bool

    @property
    def stable_to_raw(self) -> tuple:
        inv = [0] * len(self.raw_to_stable)
        for raw, stable in enumerate(self.raw_to_stable):
            inv[stable] = raw
        return tuple(inv)


@dataclass(frozen=True)
class Pending:
    """The latest fit whose continuity could not be confirmed, with what NEW_NAMES would do. Replaced by the next unresolved fit, cleared by ALIGNED or a user choice."""
    fit_index: int
    summary: FitSummary
    proposed_raw_to_stable: tuple
    proposed_rung: int
    proposed_verified: bool
    correspondence: tuple                  # per old stable id: dict(old, new, distance); new is None when no counterpart


@dataclass(frozen=True)
class LineageState:
    lineage_id: str
    epoch: int
    anchor: Anchor
    n_fits: int
    pending: Pending | None = None
    epoch_history: tuple = ()


# EXPERT_REVIEW(identity-distance): every record carries `expert_review_status` (PENDING until an expert signs off) next to the distance, margin and threshold that produced its state.
@dataclass(frozen=True)
class LineageRecord:
    fit_index: int
    engine: str
    anchor_engine: str | None
    K: int
    K_anchor: int | None
    lineage_id: str
    lineage_epoch: int
    continuity_state: str
    user_choice: str | None
    message: str
    stable_regime_id: np.ndarray               # per session: the stable id of the regime the session was decoded into (raw ids are untouched)
    raw_to_stable: tuple
    activity_rung: int
    ordering_verified: bool
    alignment_margin: float | None             # = margin_diff
    margin_diff: float | None
    margin_ratio: float | None
    cost_best: float | None
    cost_second: float | None
    matched_distances: tuple | None
    nearest_other_distances: tuple | None
    separation_min: float
    anchor_separation_min: float | None
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    proposed_new_names: dict | None
    split_merge_candidates: tuple
    reasons: tuple
    threshold: dict | None
    expert_review_status: str = EXPERT_REVIEW_STATUS
    channels_used: tuple = ("physical",)           # ("physical",) or ("physical", "context")
    channel_detail: dict | None = None             # per channel: assignment, margins and whether it was confident on its own; None when only channel 1 was used

    def as_dict(self) -> dict:
        def clean(x):
            if isinstance(x, np.ndarray):
                return [clean(v) for v in x.tolist()]
            if isinstance(x, (np.floating, float)):
                return None if not np.isfinite(x) else float(x)
            if isinstance(x, (np.integer,)):
                return int(x)
            if isinstance(x, (tuple, list)):
                return [clean(v) for v in x]
            if isinstance(x, dict):
                return {k: clean(v) for k, v in x.items()}
            return x
        return {k: clean(v) for k, v in self.__dict__.items()}


def may_claim_same_as_before(record) -> bool:
    """Fail closed: True only when the record is a LineageRecord (or its dict) whose continuity_state is exactly ALIGNED."""
    state = record.get("continuity_state") if isinstance(record, dict) else getattr(record, "continuity_state", None)
    return state == ALIGNED


def effective_choice(record) -> str:
    """`None` behaves as KEEP_OLD (A3)."""
    choice = record.get("user_choice") if isinstance(record, dict) else getattr(record, "user_choice", None)
    return "KEEP_OLD" if choice is None else choice


# ----------------------------------------------------------------------------------------------------------------- summaries
def separation_min(raw_means: np.ndarray, raw_window: np.ndarray, labels: np.ndarray) -> float:
    """EXPERT_REVIEW(identity-distance): the identifiability signal. Smallest over regime pairs of the Euclidean norm of the difference of the two regime means divided feature by
    feature by the pooled within-regime sd of the days decoded into the regimes (observed entries, residuals about the model's regime means; sd 0 or no data gives 1). inf for K < 2."""
    means = np.asarray(raw_means, dtype=float)
    K, F = means.shape
    if K < 2:
        return float("inf")
    X = np.asarray(raw_window, dtype=float); lab = np.asarray(labels, dtype=int)
    ss = np.zeros(F); n = np.zeros(F)
    for k in range(K):
        sel = X[lab == k]
        for f in range(F):
            x = sel[:, f]; x = x[~np.isnan(x)]
            ss[f] += ((x - means[k, f]) ** 2).sum(); n[f] += len(x)
    sd = np.sqrt(np.where(n > 0, ss / np.maximum(n, 1), 0.0)); sd = np.where(sd > 0, sd, 1.0)
    return float(min(np.linalg.norm((means[i] - means[j]) / sd) for i in range(K) for j in range(i + 1, K)))


def validate_context_profiles(profiles, K: int) -> np.ndarray:
    """A (K, C) array of non-negative finite rows that sum to 1 (a histogram or a vector of proportions per regime). Anything else is refused."""
    P = np.array(profiles, dtype=float)
    if P.ndim != 2 or P.shape[0] != K or P.shape[1] < 1:
        raise ValueError(f"context profiles must be (K={K}, C) with C >= 1")
    if not np.isfinite(P).all() or (P < 0).any():
        raise ValueError("context profiles must be finite and non-negative")
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("every context profile must sum to 1")
    return P


def context_profiles_from_days(daily_profiles, labels, K: int) -> np.ndarray:
    """Aggregates per-day descriptor proportions (T, C) into one profile per regime: the mean over the days decoded into the regime, renormalised. Days without a descriptor (a NaN row) are
    skipped. A regime without any usable day is refused. The descriptors themselves come from the caller's consented, aggregated pipeline; nothing is extracted here."""
    D = np.array(daily_profiles, dtype=float); lab = np.asarray(labels, dtype=int)
    if D.ndim != 2 or lab.shape != (D.shape[0],) or lab.min() < 0 or lab.max() >= K:
        raise ValueError("daily profiles must be (T, C) and labels (T,) in 0..K-1")
    out = np.zeros((K, D.shape[1]))
    for k in range(K):
        rows = D[(lab == k) & ~np.isnan(D).any(axis=1)]
        if len(rows) == 0:
            raise ValueError(f"regime {k} has no day with a descriptor")
        m = rows.mean(axis=0)
        out[k] = m / m.sum()
    return validate_context_profiles(out, K)


def hellinger2_matrix(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """EXPERT_REVIEW(identity-distance): squared Hellinger distance between every row of P and every row of Q: 0.5 * sum((sqrt p - sqrt q)^2), in [0, 1]."""
    return 0.5 * ((np.sqrt(P)[:, None, :] - np.sqrt(Q)[None, :, :]) ** 2).sum(axis=2)


def regime_table(summary: "FitSummary", registry) -> list[dict]:
    """Channel 1 made readable (A6): per regime, the mean of every feature in its physical unit, from the registry (name, unit, source)."""
    from .feature_registry import validate_registry
    reg = validate_registry(registry, summary.raw_means.shape[1])
    return [{f.name: dict(value=float(summary.raw_means[k, i]), unit=f.unit, source=f.source) for i, f in enumerate(reg)} for k in range(summary.K)]


# EXPERT_REVIEW(identity-distance): computes the identifiability signal (`separation_min`) that the gate `min_separation` is applied to.
def make_summary(engine: str, raw_means, scaler_mean, scaler_std, labels, raw_window, exp_durations=None, context_profiles=None) -> FitSummary:
    raw_means = np.array(raw_means, dtype=float); raw_window = np.array(raw_window, dtype=float); labels = np.array(labels, dtype=int)
    scaler_mean = np.array(scaler_mean, dtype=float); scaler_std = np.array(scaler_std, dtype=float)
    if engine not in ("baseline", "dual"):
        raise ValueError("engine must be 'baseline' or 'dual'")
    K, F = raw_means.shape
    if scaler_mean.shape != (F,) or scaler_std.shape != (F,) or raw_window.shape[1] != F or labels.shape != (raw_window.shape[0],):
        raise ValueError("shapes of raw_means, scaler, raw_window and labels do not agree")
    if labels.min() < 0 or labels.max() >= K:
        raise ValueError(f"labels must lie in 0..{K - 1}")
    if not (scaler_std > 0).all():
        raise ValueError("scaler_std must be positive")
    dur = None if exp_durations is None else np.array(exp_durations, dtype=float)
    ctx = None if context_profiles is None else validate_context_profiles(context_profiles, K)
    return FitSummary(engine, raw_means, scaler_mean, scaler_std, labels, separation_min(raw_means, raw_window, labels), raw_window, dur, ctx)


def summarize_fit(result, raw_window, scaler_mean, scaler_std) -> FitSummary:
    """Summary of an `HSSMResult` of either engine. `raw_window` is the window in physical units that was z-scored with (scaler_mean, scaler_std) to give the model input.
    Baseline regime means: `mu` (z-space). Dual: `regime_observation_means(extract_params(model))` (z-space). Both are converted to raw units with the stored scaler."""
    model = result.model
    labels = np.asarray(result.p_t_posterior).argmax(axis=1)
    if result.is_baseline_model:
        z = np.array(model.mu, dtype=float); engine = "baseline"
    else:
        from .shrinkage import extract_params, regime_observation_means
        z = regime_observation_means(extract_params(model)); engine = "dual"
    scaler_mean = np.asarray(scaler_mean, dtype=float); scaler_std = np.asarray(scaler_std, dtype=float)
    raw_means = scaler_mean + scaler_std * z
    from .shrinkage import duration_pmf
    pmf = duration_pmf(np.array(model.dur_mu), np.array(model.dur_sigma), int(model.Dmax))
    exp_dur = (pmf * np.arange(1, int(model.Dmax) + 1)).sum(axis=1)
    return make_summary(engine, raw_means, scaler_mean, scaler_std, labels, raw_window, exp_dur)


def annotate_result(result, record: LineageRecord):
    """Adds the lineage record as the new, additive field `HSSMResult.lineage` (A4). No raw output is touched. Returns the same result object."""
    result.lineage = record.as_dict()
    return result


# ----------------------------------------------------------------------------------------------------------------- naming (first fit of a lineage or of an epoch)
# EXPERT_REVIEW(identity-distance): applies the ordering ladder (weights, reliability thresholds) to give a fit its stable names.
def name_fit(summary: FitSummary, registry, ladder: LadderConfig) -> tuple[tuple, int, bool, OrderingResult | None]:
    """Stable ids for a fit that has no anchor: the ladder's order when it is verified, else first-fit order. Returns (raw_to_stable, activity_rung, verified, ordering)."""
    K = summary.K
    if summary.raw_window is None or registry is None:
        return tuple(range(K)), 0, False, None
    res = order_regimes(summary.raw_window, summary.labels, K, registry, ladder)
    raw_to_stable = [0] * K
    for pos, raw in enumerate(res.order):
        raw_to_stable[raw] = pos
    return tuple(raw_to_stable), int(res.activity_rung), bool(res.verified), res


# ----------------------------------------------------------------------------------------------------------------- alignment
def _vectors(summary: FitSummary, scale: np.ndarray, duration_weight: float) -> np.ndarray:
    """EXPERT_REVIEW(identity-distance): the matching distance works on these vectors: regime means in raw units divided by the anchor's per-feature scale, plus (weight > 0)
    the log expected duration as one more coordinate."""
    v = summary.raw_means / scale
    if duration_weight > 0:
        if summary.exp_durations is None:
            raise ValueError("duration_weight > 0 needs exp_durations in every summary")
        v = np.hstack([v, np.sqrt(duration_weight) * np.log(summary.exp_durations)[:, None]])
    return v


# EXPERT_REVIEW(identity-distance): the per-feature scale of the matching distance (the anchor's stored scaler).
def _scale(anchor: Anchor) -> np.ndarray:
    s = np.array(anchor.summary.scaler_std, dtype=float)
    return np.where(s > 0, s, 1.0)


def alignment(anchor_vec: np.ndarray, new_vec: np.ndarray) -> dict:
    """EXPERT_REVIEW(identity-distance): Hungarian assignment of the anchor's regimes (rows, in stable-id order) to the new fit's regimes on squared distance, with the best total cost
    C1, the best total cost C2 of any other one-to-one assignment (exhaustive, K <= 6), the margins and, per regime, the matched distance and the distance to the nearest other regime."""
    return alignment_from_cost(((anchor_vec[:, None, :] - new_vec[None, :, :]) ** 2).sum(axis=2))


# EXPERT_REVIEW(identity-distance): the best and the second best assignment and the two margins (`margin_diff`, `margin_ratio`) that the thresholds are applied to.
def alignment_from_cost(cost: np.ndarray) -> dict:
    """Same as `alignment` for a given (K, K) cost matrix (rows: the anchor's regimes in stable-id order, columns: the new fit's raw regimes)."""
    K = cost.shape[0]
    rows, cols = linear_sum_assignment(cost)
    perm = tuple(int(c) for c in cols[np.argsort(rows)])
    C1 = float(sum(cost[i, perm[i]] for i in range(K)))
    others = [sum(cost[i, q[i]] for i in range(K)) for q in itertools.permutations(range(K)) if tuple(q) != perm]
    C2 = float(min(others)) if others else float("inf")
    if C2 == float("inf"):
        ratio = 0.0
    else:
        ratio = (C1 / C2) if C2 > 0 else 1.0
    return dict(perm=perm, C1=C1, C2=C2, margin_diff=((C2 - C1) / K), margin_ratio=ratio,
                matched=tuple(float(cost[i, perm[i]]) for i in range(K)),
                nearest_other=tuple(float(min(cost[i, j] for j in range(K) if j != perm[i])) if K > 1 else float("inf") for i in range(K)))


def margin_ok(th: PairThreshold, al: dict) -> bool:
    """EXPERT_REVIEW(identity-distance): the acceptance rule. A tie between the best and the second best assignment is never accepted, and a threshold that is not calibrated
    (UNKNOWN for its class of pair) accepts nothing, whatever the margins are."""
    if not th.calibrated:
        return False
    if al["C2"] == float("inf"):
        return True
    strict = al["C1"] < al["C2"]
    ok_diff = strict and al["margin_diff"] >= th.tau
    ok_ratio = strict and al["margin_ratio"] <= th.rho
    return {"diff": ok_diff, "ratio": ok_ratio, "both": ok_diff and ok_ratio}[th.rule]


# EXPERT_REVIEW(identity-distance): uses the matching distance to name the split or merge candidates when K changed.
def split_merge_candidates(anchor_vec: np.ndarray, new_vec: np.ndarray) -> tuple:
    """When the fitted K differs from the anchor's: each regime goes to the nearest regime of the other fit. A split candidate is an old (stable) id that is nearest to two or more new
    regimes; a merge candidate is a group of old ids that share the same nearest new regime. Exploratory; precision is reported in the Item 8 study."""
    D = ((anchor_vec[:, None, :] - new_vec[None, :, :]) ** 2).sum(axis=2)
    K_old, K_new = D.shape
    out = []
    if K_new > K_old:
        nearest_old = D.argmin(axis=0)
        for i in range(K_old):
            kids = [int(j) for j in range(K_new) if nearest_old[j] == i]
            if len(kids) >= 2:
                out.append(dict(kind="split", old_stable_id=int(i), new_raw_regimes=kids))
    elif K_new < K_old:
        nearest_new = D.argmin(axis=1)
        for j in range(K_new):
            group = [int(i) for i in range(K_old) if nearest_new[i] == j]
            if len(group) >= 2:
                out.append(dict(kind="merge", old_stable_ids=group, new_raw_regime=int(j)))
    return tuple(out)


# EXPERT_REVIEW(identity-distance): uses the matching distance to record the old-to-new correspondence kept when the user takes new names.
def _correspondence(anchor_vec: np.ndarray, new_vec: np.ndarray, proposed_raw_to_stable: tuple) -> tuple:
    """Old stable id -> new stable id (under the PROPOSED names) of the new regime the assignment pairs it with; None when the old regime has no counterpart (K changed)."""
    D = ((anchor_vec[:, None, :] - new_vec[None, :, :]) ** 2).sum(axis=2)
    rows, cols = linear_sum_assignment(D)
    pair = {int(r): int(c) for r, c in zip(rows, cols)}
    return tuple(dict(old=i, new=(int(proposed_raw_to_stable[pair[i]]) if i in pair else None), distance=(float(D[i, pair[i]]) if i in pair else None), confirmed=False)
                 for i in range(anchor_vec.shape[0]))


# EXPERT_REVIEW(identity-distance): writes the identifiability signal of the fit into the record next to the margins and the state.
def _record(fit_index, summary, anchor, epoch, lineage_id, state_name, message, raw_to_stable, rung, verified, al, th, reasons, proposed, candidates, user_choice=None,
            channels_used=("physical",), channel_detail=None):
    stable = np.asarray(raw_to_stable, dtype=int)[summary.labels]
    return LineageRecord(
        fit_index=fit_index, engine=summary.engine, anchor_engine=None if anchor is None else anchor.summary.engine, K=summary.K, K_anchor=None if anchor is None else anchor.summary.K,
        lineage_id=lineage_id, lineage_epoch=epoch, continuity_state=state_name, user_choice=user_choice, message=message, stable_regime_id=stable, raw_to_stable=tuple(int(x) for x in raw_to_stable),
        activity_rung=int(rung), ordering_verified=bool(verified),
        alignment_margin=None if al is None else al["margin_diff"], margin_diff=None if al is None else al["margin_diff"], margin_ratio=None if al is None else al["margin_ratio"],
        cost_best=None if al is None else al["C1"], cost_second=None if al is None else al["C2"], matched_distances=None if al is None else al["matched"],
        nearest_other_distances=None if al is None else al["nearest_other"], separation_min=summary.separation_min,
        anchor_separation_min=None if anchor is None else anchor.summary.separation_min, scaler_mean=summary.scaler_mean, scaler_std=summary.scaler_std,
        proposed_new_names=proposed, split_merge_candidates=tuple(candidates), reasons=tuple(reasons),
        threshold=None if th is None else dict(rule=th.rule, tau=th.tau, rho=th.rho, calibrated=th.calibrated), channels_used=tuple(channels_used), channel_detail=channel_detail)


# EXPERT_REVIEW(identity-distance): the scale of a channel's distance (mean pairwise inter-regime distance of the anchor) used in the combined cost.
def _mean_pair(M: np.ndarray) -> float:
    K = M.shape[0]
    if K < 2:
        return 1.0
    v = float((M.sum() - np.trace(M)) / (K * (K - 1)))
    return v if v > 0 else 1.0


def _combined_alignment(anchor: Anchor, summary: FitSummary, a_vec: np.ndarray, n_vec: np.ndarray, th: PairThreshold, config: AlignmentConfig):
    """Channel 1 alone (the default, identical to `alignment`), or channel 1 and the optional contextual channel combined as `w1 * d1 / s1 + w2 * d2 / s2` with the disagreement check.
    EXPERT_REVIEW(identity-distance): the combined cost, the scales s1 and s2 and the disagreement rule are defined here. Returns (alignment, channels_used, channel_detail)."""
    ch = config.channels
    D1 = ((a_vec[:, None, :] - n_vec[None, :, :]) ** 2).sum(axis=2)
    a_ctx, n_ctx = anchor.summary.context_profiles, summary.context_profiles
    if ch.w2 == 0:
        return alignment_from_cost(D1), ("physical",), None
    if a_ctx is None or n_ctx is None:
        return alignment_from_cost(D1), ("physical",), dict(context="missing: channel 1 alone was used", weights=(ch.w1, ch.w2))
    if a_ctx.shape[1] != n_ctx.shape[1]:
        raise ValueError("the context profiles of the two fits must have the same categories (same C)")
    a_ctx_s = a_ctx[list(anchor.stable_to_raw)]
    D2 = hellinger2_matrix(a_ctx_s, n_ctx)
    s1 = _mean_pair(((a_vec[:, None, :] - a_vec[None, :, :]) ** 2).sum(axis=2)); s2 = _mean_pair(hellinger2_matrix(a_ctx_s, a_ctx_s))
    al = alignment_from_cost(ch.w1 * D1 / s1 + ch.w2 * D2 / s2)
    al1, al2 = alignment_from_cost(D1), alignment_from_cost(D2)
    conf1, conf2 = margin_ok(th, al1), margin_ok(ch.context_threshold, al2)
    detail = dict(weights=(ch.w1, ch.w2), scales=(s1, s2),
                  physical=dict(perm=al1["perm"], margin_diff=al1["margin_diff"], margin_ratio=al1["margin_ratio"], confident=bool(conf1)),
                  context=dict(perm=al2["perm"], margin_diff=al2["margin_diff"], margin_ratio=al2["margin_ratio"], confident=bool(conf2)),
                  disagree=bool(conf1 and conf2 and al1["perm"] != al2["perm"]))
    return al, ("physical", "context"), detail


# EXPERT_REVIEW(identity-distance): the gate, the distance, the thresholds and the continuity state are applied together here.
def step(state: LineageState | None, summary: FitSummary, config: AlignmentConfig = AlignmentConfig(), registry=None, lineage_id: str | None = None) -> tuple[LineageRecord, LineageState]:
    """Feed the next accepted fit. Returns (record, new state); the old state is not modified. `lineage_id` is required for the first fit (no hidden randomness) and ignored after."""
    if state is None:
        if not lineage_id:
            raise ValueError("the first fit of a lineage needs a lineage_id")
        raw_to_stable, rung, verified, _ = name_fit(summary, registry, config.ladder)
        anchor = Anchor(summary, raw_to_stable, 0, rung, verified)
        msg = ("first fit of this lineage: regimes named by the activity ordering" if verified else
               "first fit of this lineage: the activity order could not be verified, regimes are named in first-fit order (no activity claim)")
        rec = _record(0, summary, None, 0, lineage_id, NEW_LINEAGE, msg, raw_to_stable, rung, verified, None, None, ("first_fit",), None, ())
        return rec, LineageState(lineage_id, 0, anchor, 1)
    anchor = state.anchor
    fit_index = state.n_fits
    scale = _scale(anchor)
    a_vec_raw = _vectors(anchor.summary, scale, config.duration_weight)
    a_vec = a_vec_raw[list(anchor.stable_to_raw)]                          # rows in stable-id order
    n_vec = _vectors(summary, scale, config.duration_weight)
    cross = summary.engine != anchor.summary.engine
    th = config.cross_engine if cross else (config.dual_engine if summary.engine == "dual" else config.same_engine)
    # EXPERT_REVIEW(identity-distance): the proposal for a new fit: the ladder names it and the matching distance gives the old-to-new correspondence.
    def proposal():
        """The names this fit would get on its own (what NEW_NAMES would adopt), computed only when continuity is not confirmed."""
        prop_r2s, prop_rung, prop_verified, _ = name_fit(summary, registry, config.ladder)
        prop = dict(raw_to_stable=tuple(prop_r2s), stable_regime_id=np.asarray(prop_r2s, dtype=int)[summary.labels], activity_rung=prop_rung, ordering_verified=prop_verified)
        return prop, Pending(fit_index, summary, tuple(prop_r2s), prop_rung, prop_verified, _correspondence(a_vec, n_vec, tuple(prop_r2s)))
    if summary.K != anchor.summary.K:
        # one-to-one as far as it goes: matched regimes inherit the old name, the others get fresh ids that were never used by the anchor
        D = ((a_vec[:, None, :] - n_vec[None, :, :]) ** 2).sum(axis=2)
        rows, cols = linear_sum_assignment(D)
        r2s = [-1] * summary.K
        for r, c in zip(rows, cols):
            r2s[int(c)] = int(r)
        nxt = anchor.summary.K
        for k in range(summary.K):
            if r2s[k] < 0:
                r2s[k] = nxt; nxt += 1
        cands = split_merge_candidates(a_vec, n_vec)
        proposed, pending = proposal()
        rec = _record(fit_index, summary, anchor, state.epoch, state.lineage_id, K_CHANGED, MESSAGE_UNCONFIRMED + ": the number of regimes changed from "
                      f"{anchor.summary.K} to {summary.K}; the old names are kept, new names can be chosen", r2s, anchor.activity_rung, anchor.ordering_verified, None, None,
                      ("k_changed",), proposed, cands)
        return rec, replace(state, n_fits=state.n_fits + 1, pending=pending)
    if summary.K > MAX_MATCH_K:
        reasons = ["k_above_matching_ceiling"]; al = None; channels_used, detail = ("physical",), None
    else:
        al, channels_used, detail = _combined_alignment(anchor, summary, a_vec, n_vec, th, config)
        reasons = []
        if anchor.summary.separation_min < config.min_separation:
            reasons.append("anchor_not_identifiable")
        if summary.separation_min < config.min_separation:
            reasons.append("fit_not_identifiable")
        if not th.calibrated:
            reasons.append("threshold_not_calibrated")
        elif not margin_ok(th, al):
            reasons.append("margin_below_threshold")
        if detail is not None and detail.get("disagree"):
            reasons.append("channels_disagree")
    if not reasons:
        r2s = [0] * summary.K
        for i, raw in enumerate(al["perm"]):
            r2s[raw] = i                                                  # the new regime matched to old stable id i takes that name
        rec = _record(fit_index, summary, anchor, state.epoch, state.lineage_id, ALIGNED, "regimes matched to the previous accepted fit", r2s, anchor.activity_rung,
                      anchor.ordering_verified, al, th, (), None, (), channels_used=channels_used, channel_detail=detail)
        new_anchor = Anchor(summary, tuple(r2s), fit_index, anchor.activity_rung, anchor.ordering_verified)
        return rec, replace(state, anchor=new_anchor, n_fits=state.n_fits + 1, pending=None)
    # ambiguous: the best guess keeps the old names but the claim is withheld
    if al is None:
        r2s = list(range(summary.K))
    else:
        r2s = [0] * summary.K
        for i, raw in enumerate(al["perm"]):
            r2s[raw] = i
    proposed, pending = proposal()
    rec = _record(fit_index, summary, anchor, state.epoch, state.lineage_id, CONTINUITY_UNVERIFIED, MESSAGE_UNCONFIRMED + "; the old names are kept, new names can be chosen",
                  r2s, anchor.activity_rung, anchor.ordering_verified, al, th, reasons, proposed, (), channels_used=channels_used, channel_detail=detail)
    return rec, replace(state, n_fits=state.n_fits + 1, pending=pending)


def apply_user_choice(state: LineageState, record: LineageRecord, choice) -> tuple[LineageRecord, LineageState]:
    """The application's answer to an unconfirmed continuity. `None` and `KEEP_OLD` leave the old names (and the anchor) as they are. `NEW_NAMES` adopts `proposed_new_names` of that
    fit: the lineage starts a new epoch, the fit becomes the anchor, the record is returned as NEW_LINEAGE, and the old-to-new mapping is appended to `epoch_history` for good."""
    if choice not in USER_CHOICES:
        raise ValueError(f"user_choice must be one of {USER_CHOICES}")
    if record.continuity_state not in (CONTINUITY_UNVERIFIED, K_CHANGED):
        raise ValueError("a user choice only applies to CONTINUITY_UNVERIFIED or K_CHANGED records")
    if record.user_choice is not None:
        raise ValueError("this record already carries a user choice")
    pend = state.pending
    if pend is None or pend.fit_index != record.fit_index:
        raise ValueError("this record is no longer the pending one (a later fit replaced it or it was resolved)")
    if choice is None:
        return record, state
    if choice == "KEEP_OLD":
        return replace(record, user_choice="KEEP_OLD"), replace(state, pending=None)
    new_epoch = state.epoch + 1
    entry = dict(from_epoch=state.epoch, to_epoch=new_epoch, at_fit_index=record.fit_index, old_K=state.anchor.summary.K, new_K=pend.summary.K,
                 old_to_new=pend.correspondence, old_names_verified=state.anchor.ordering_verified, new_names_verified=pend.proposed_verified, continuity_state_when_chosen=record.continuity_state)
    anchor = Anchor(pend.summary, pend.proposed_raw_to_stable, record.fit_index, pend.proposed_rung, pend.proposed_verified)
    new_rec = _record(record.fit_index, pend.summary, state.anchor, new_epoch, state.lineage_id, NEW_LINEAGE, "new names chosen by the user; the earlier names and the mapping stay in the epoch history",
                      pend.proposed_raw_to_stable, pend.proposed_rung, pend.proposed_verified, None, None, ("user_chose_new_names",), None, (), user_choice="NEW_NAMES")
    new_rec = replace(new_rec, margin_diff=record.margin_diff, margin_ratio=record.margin_ratio, alignment_margin=record.alignment_margin, cost_best=record.cost_best,
                      cost_second=record.cost_second, matched_distances=record.matched_distances, nearest_other_distances=record.nearest_other_distances, threshold=record.threshold,
                      channels_used=record.channels_used, channel_detail=record.channel_detail)
    return new_rec, replace(state, epoch=new_epoch, anchor=anchor, pending=None, epoch_history=state.epoch_history + (entry,))
