"""WP7-R item 3: statistics of the shrinkage study under fail-closed. Criteria: tests/PREDECLARED_CRITERIA_WP7R.md section 3.

Pure functions (no fitting here) so the rules can be tested and mutation-checked. Inputs are plain dicts keyed by seed:

  scores[seed]      mean held-out one-step log density of the arm (may be NaN)
  ref[seed]         the same for the reference arm (single Gaussian, shared-only, ...)
  publishable[seed] True when the arm's fit is NOT withheld by HSSMResult.withheld_reason

  withheld_tally      how many fits were withheld, and why (by fit_state and by reason prefix)
  subset_stats        arm minus reference on the PUBLISHABLE seeds only (the N18 statistic: mean, paired se, mean - 3 se)
  fallback_stats      arm minus reference over ALL seeds, where a withheld fit (or a publishable fit with a non-finite score) counts as "no prediction":
                      the reference is served instead, so that seed contributes a difference of exactly 0 (risk-coverage)
  survives            the pre-declared verdict word over the lengths

A seed whose score (arm or reference) is not finite is dropped from `subset_stats` and counted; nothing is imputed.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from tests.utils.heldout_prediction import MARGIN_SE, paired_margin

MIN_PUBLISHABLE = 8        # at n = 8 the multiplier 3 of "mean - 3 se" equals the one-sided 99% t quantile (2.998, 7 df); below 8 the quantile is larger than 3


def reason_prefixes(reason: str | None) -> list[str]:
    """The prefixes of a withheld_reason string ('fit_state_not_converged', 'collapsed_regime', 'fit_degraded', 'states_pass', 'late_fallbacks', ...);
    'fit_state is NUMERICAL_FAILURE' has no colon and is returned as 'numerical_failure'."""
    if not reason:
        return []
    out = [m.group(1) for m in re.finditer(r"(?:^|; )([a-z_]+):", reason)]
    if "fit_state is NUMERICAL_FAILURE" in reason:
        out.append("numerical_failure")
    return out


def withheld_tally(records: Sequence[Mapping]) -> dict:
    """records: one dict per fit with keys `withheld` (bool), `fit_state` (str) and `withheld_reason` (str or None)."""
    n = len(records)
    n_w = sum(1 for r in records if r["withheld"])
    by_state = Counter(str(r["fit_state"]) for r in records if r["withheld"])
    by_prefix: Counter = Counter()
    for r in records:
        if r["withheld"]:
            for p in set(reason_prefixes(r.get("withheld_reason"))):
                by_prefix[p] += 1
    return dict(n=n, n_withheld=n_w, n_publishable=n - n_w, withheld_rate=(n_w / n) if n else float("nan"), by_fit_state=dict(by_state),
                by_reason_prefix=dict(by_prefix), fit_states_all=dict(Counter(str(r["fit_state"]) for r in records)))


def _finite(x) -> bool:
    return x is not None and math.isfinite(float(x))


def subset_stats(scores: Mapping[int, float], ref: Mapping[int, float], publishable: Mapping[int, bool], min_n: int = MIN_PUBLISHABLE) -> dict:
    """Arm minus reference on the seeds where the arm is publishable and both scores are finite."""
    seeds = sorted(set(scores) & set(ref))
    pub = [s for s in seeds if publishable.get(s, False)]
    use = [s for s in pub if _finite(scores[s]) and _finite(ref[s])]
    n_nonfinite = len(pub) - len(use)
    out = dict(n_seeds=len(seeds), n_publishable=len(pub), n_used=len(use), n_nonfinite_publishable=n_nonfinite,
               coverage=(len(pub) / len(seeds)) if seeds else float("nan"), judged=len(use) >= min_n)
    if len(use) < 2:
        out.update(mean_arm=float("nan"), mean_ref=float("nan"), mean=float("nan"), se=float("nan"), mean_minus_3se=float("nan"), n_favour=0)
        return out
    d = np.array([float(scores[s]) - float(ref[s]) for s in use])
    mean, se, lower = paired_margin(d, MARGIN_SE)
    out.update(mean_arm=float(np.mean([scores[s] for s in use])), mean_ref=float(np.mean([ref[s] for s in use])), mean=mean, se=se, mean_minus_3se=lower,
               n_favour=int(np.sum(d > 0)))
    return out


def fallback_stats(scores: Mapping[int, float], ref: Mapping[int, float], publishable: Mapping[int, bool]) -> dict:
    """Arm minus reference over ALL seeds with a finite reference score. A withheld fit, or a publishable fit whose score is not finite, is "no prediction":
    the reference is served for that seed and its difference is 0."""
    seeds = sorted(s for s in set(scores) & set(ref) if _finite(ref[s]))
    d = []
    n_served = 0
    for s in seeds:
        if publishable.get(s, False) and _finite(scores[s]):
            d.append(float(scores[s]) - float(ref[s]))
            n_served += 1
        else:
            d.append(0.0)
    out = dict(n_seeds=len(seeds), n_served=n_served, coverage=(n_served / len(seeds)) if seeds else float("nan"))
    if len(d) < 2:
        out.update(mean=float("nan"), se=float("nan"), mean_minus_3se=float("nan"))
        return out
    mean, se, lower = paired_margin(np.array(d), MARGIN_SE)
    out.update(mean=mean, se=se, mean_minus_3se=lower, n_favour=int(np.sum(np.array(d) > 0)))
    return out


def survives(stats_by_length: Mapping[int, Mapping]) -> str:
    """The verdict word for 'the gain over the reference survives on publishable fits'. `stats_by_length[L]` is a `subset_stats` dict.

    - 'does not survive'  : at least one judged length (n_used >= MIN_PUBLISHABLE) has mean - 3 se <= 0 (or not finite)
    - 'survives'          : every length is judged and every one has mean - 3 se > 0
    - 'partly judged'     : the judged lengths all pass, but at least one length is not judged
    - 'cannot be judged'  : no length is judged
    """
    judged = [L for L, st in stats_by_length.items() if st["judged"]]
    if not judged:
        return "cannot be judged"
    if any(not (math.isfinite(stats_by_length[L]["mean_minus_3se"]) and stats_by_length[L]["mean_minus_3se"] > 0) for L in judged):
        return "does not survive"
    return "survives" if len(judged) == len(stats_by_length) else "partly judged"
