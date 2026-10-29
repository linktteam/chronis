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
