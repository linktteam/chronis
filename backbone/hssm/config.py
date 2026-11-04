"""Hyperparameters and thresholds for the HSSM module, kept out of code so they're
inspectable/tunable without touching model logic. Nothing here is a hidden magic number."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class HSSMFitConfig:
    n_init: int = 10          # HARD MINIMUM per directive Day 8 — do not lower
    n_iter: int = 100
    tol: float = 1e-4
    max_duration: int = 45    # Dmax for expanded-state duration truncation, session-index units
    k_candidates: tuple[int, ...] = (2, 3, 4)   # BIC search range, directive Day 8
    mean_dwell_guess: float = 20.0              # init-only prior guess for log-normal duration


@dataclass(frozen=True)
class ColdStartConfig:
    min_present_sessions: int = 30   # Bible 5.1 cold-start gate. Below this: NO output, not low-confidence.


DEFAULT_FIT_CONFIG = HSSMFitConfig()
DEFAULT_COLD_START_CONFIG = ColdStartConfig()

# WP5 (N18-v2): strength of the shrinkage toward a shared prior, in pseudo-steps PER REGIME. Frozen on development seeds 6001..6012 (tests/N0_FROZEN_WP5.md).
# Used only when a shared prior is passed to fit_hssm / fit_dual_with_random_restarts; there is no prior by default.
SHARED_PRIOR_DEFAULT_N0 = 5.0

# WP6 item 1: `timestamps` are calendar DAYS. The span (last - first) may not exceed this many days. Reason (tests/PREDECLARED_CRITERIA_WP6.md section 2):
# 20,000 days is 54.8 years, longer than any person's record; a record in seconds exceeds it after 5.5 hours, in milliseconds after 20 seconds; at the
# limit the baseline lattice is (20000, 4, 45) float64 = 28.8 MB per array, against the 6.43 GiB the owner's run tried to allocate for seconds.
MAX_TIMESTAMP_SPAN_DAYS = 20000

# WP6 item 3 (tests/PREDECLARED_CRITERIA_WP6.md section 4). Reasons are given there; the numbers come from tests/WP6_CONVERGENCE_DIAGNOSIS.md.
# A fit is *collapsed* when some regime holds fewer than this many effective days (sum of its smoothed posterior over the rows). A variance at the
# floor is reported but does not make a fit collapsed (floor-only converged fits are not worse than healthy ones in the diagnosis, section 3).
COLLAPSE_MIN_EFFECTIVE_DAYS = 4.0
# Variance floor of the baseline M-step (unchanged value; named here so that the collapse report and the M-step use the same number).
BASELINE_VARIANCE_FLOOR = 1e-3
# Multi-start summary (ledger N1): restarts within this many nats of the selected log-likelihood count as "within epsilon"; two labellings with an
# adjusted Rand index at least RESTART_SAME_ARI count as the same solution.
RESTART_EPSILON_NATS = 1.0
RESTART_SAME_ARI = 0.9
