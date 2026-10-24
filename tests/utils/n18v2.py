"""N18-v2 scoring and decision rules (WP5). Criteria: tests/PREDECLARED_CRITERIA_WP5.md sections 3, 6 and 7.

Pure functions (no fitting here) so the decision rules can be tested and mutation-checked:
  paired_stats        per-seed paired difference of two arms: mean, paired se (ddof 1), mean - 3 se, mean + 1 se, mean + 3 se, seeds favouring
  choose_n0           the freeze rule of section 6 (maximise the minimum over lengths of mean - 3 se; ties go to the smaller n0)
  evaluate_criteria   criteria 1 to 3 for one arm (section 7)
  variant_verdict     criterion 6: a variant must make at least one of criteria 1 to 3 fail
and the scoring of one fitted model on one user (`score_model`) and of the single Gaussian (`gauss_scores`).

`scores[arm][length]` is a dict {seed: mean one-step-ahead predictive log density per held-out step}. A seed whose score for either arm of a comparison is
not finite is dropped from that comparison and COUNTED (`n_dropped`); nothing is imputed.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
from scipy.stats import multivariate_normal

from tests.utils.heldout_prediction import MARGIN_SE, paired_margin, predictive_increments

K, F, L = 2, 4, 2
MAX_DURATION = 25
N_ITER = 20
LENGTHS = (30, 60, 120)
SEGMENTS = {30: 2, 60: 4, 120: 8}
N0_GRID = (5.0, 20.0, 80.0, 320.0)
ATYPICAL_LENGTH = 120


def paired_stats(a: Mapping[int, float], b: Mapping[int, float]) -> dict:
    """Paired differences d_s = a[s] - b[s] over the seeds present in both with finite values."""
    seeds = sorted(set(a) & set(b))
    pairs = [(s, float(a[s]) - float(b[s])) for s in seeds if math.isfinite(a[s]) and math.isfinite(b[s])]
    n_dropped = len(seeds) - len(pairs)
    if len(pairs) < 2:
        return dict(n=len(pairs), n_dropped=n_dropped, mean=float("nan"), se=float("nan"), mean_minus_3se=float("nan"), mean_plus_1se=float("nan"),
                    mean_plus_3se=float("nan"), n_favour=0)
    d = np.array([x for _, x in pairs])
    mean, se, lower = paired_margin(d, MARGIN_SE)                                   # the N18 statistic: mean, paired se (ddof 1), mean - 3 se
    return dict(n=len(d), n_dropped=n_dropped, mean=mean, se=se, mean_minus_3se=lower, mean_plus_1se=mean + se, mean_plus_3se=mean + 3.0 * se,
                n_favour=int(np.sum(d > 0)))


def choose_n0(lowers: Mapping[float, Mapping[int, float]]) -> float:
    """lowers[n0][length] = mean - 3 se of (shrinkage - gauss) on the development seeds. Returns the n0 that maximises the minimum over lengths;
    ties go to the smaller n0 (a later, larger n0 must be STRICTLY better to win). Non-finite entries make that n0 ineligible."""
    best_n0, best_val = None, -math.inf
    for n0 in sorted(lowers):
        vals = [lowers[n0][length] for length in lowers[n0]]
        if not vals or not all(math.isfinite(v) for v in vals):
            continue
        m = min(vals)
        if m > best_val:
            best_n0, best_val = n0, m
    if best_n0 is None:
        raise ValueError("no n0 has finite scores at every length")
    return best_n0


def evaluate_criteria(eval_scores: Mapping[str, Mapping[int, Mapping[int, float]]], atyp_scores: Mapping[str, Mapping[int, Mapping[int, float]]],
                      arm: str = "shrinkage", lengths=LENGTHS, atypical_length: int = ATYPICAL_LENGTH) -> dict:
    """Criteria 1 to 3 for `arm`. Opponents `gauss`, `personal`, `shared` are taken from the same score dictionaries."""
    c1 = {L_: paired_stats(eval_scores[arm][L_], eval_scores["gauss"][L_]) for L_ in lengths}
    c2p = {L_: paired_stats(eval_scores[arm][L_], eval_scores["personal"][L_]) for L_ in lengths}
    c2s = {L_: paired_stats(eval_scores[arm][L_], eval_scores["shared"][L_]) for L_ in lengths}
    c3 = {L_: paired_stats(atyp_scores[arm][L_], atyp_scores["personal"][L_]) for L_ in lengths if L_ in atyp_scores[arm]}
    ok1 = all(math.isfinite(c1[L_]["mean_minus_3se"]) and c1[L_]["mean_minus_3se"] > 0 for L_ in lengths)
    ok2 = all(math.isfinite(c2p[L_]["mean_plus_1se"]) and c2p[L_]["mean_plus_1se"] >= 0 and math.isfinite(c2s[L_]["mean_plus_1se"])
              and c2s[L_]["mean_plus_1se"] >= 0 for L_ in lengths)
    ok3 = atypical_length in c3 and math.isfinite(c3[atypical_length]["mean_plus_3se"]) and c3[atypical_length]["mean_plus_3se"] >= 0
    return dict(arm=arm, c1=ok1, c2=ok2, c3=ok3, all_pass=bool(ok1 and ok2 and ok3), vs_gauss=c1, vs_personal=c2p, vs_shared=c2s, atypical_vs_personal=c3)


def variant_verdict(result: Mapping) -> dict:
    """Criterion 6 for one variant (result = evaluate_criteria of the variant): it discriminates iff at least one of criteria 1 to 3 fails."""
    return dict(arm=result["arm"], discriminates=not (result["c1"] and result["c2"] and result["c3"]), c1=result["c1"], c2=result["c2"], c3=result["c3"])


# ------------------------------------------------------------------------------------------------------------- scoring one model on one user
def score_model(model, user) -> dict:
    """Forward filter over train + held-out with frozen parameters; the held-out increments are the score."""
    before = model.cholesky_fallback_count_
    lp, ll_full = predictive_increments(model, user.X_full)
    t_tr = len(user.X_train)
    lp_test = lp[t_tr:]
    return dict(mean=float(np.mean(lp_test)), n_test=int(len(lp_test)), finite=bool(np.all(np.isfinite(lp_test))), ll_full=float(ll_full),
                scoring_fallbacks=int(model.cholesky_fallback_count_ - before))


def gauss_scores(user) -> dict:
    try:
        mu = user.X_train.mean(axis=0)
        cov = np.cov(user.X_train, rowvar=False, ddof=1)
        lp = multivariate_normal(mean=mu, cov=cov).logpdf(user.X_test)
        return dict(mean=float(np.mean(lp)), n_test=int(len(lp)), finite=bool(np.all(np.isfinite(lp))))
    except (np.linalg.LinAlgError, ValueError):
        return dict(mean=float("nan"), n_test=int(len(user.X_test)), finite=False)


def fit_info(model) -> dict:
    return dict(fit_state=str(model.fit_state_.value), fit_degraded=bool(model.fit_degraded_), n_em_iterations=len(model.log_likelihood_history_),
                ll_train_final=None if model.log_likelihood_ is None else float(model.log_likelihood_))
