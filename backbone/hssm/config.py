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

# WP6 item 3, part 2 (tests/PREDECLARED_CRITERIA_WP6.md section 5; numbers from tests/WP6_CONVERGENCE_DIAGNOSIS.md and the Q4b continuation study).
# Dual stopping rule at the iteration limit: if the log-likelihood gained less than this many nats PER ROW over the last DUAL_CAP_WINDOW iterations, the fit
# that reached the limit is certified CONVERGED (criterion "window_gain_at_cap"); otherwise it stays MAX_ITER_REACHED and is withheld.
DUAL_CAP_WINDOW = 10
DUAL_CAP_CERT_GAIN_PER_ROW = 2.0e-4
# Dual duration box (applied only when the Nelder-Mead update leaves it): mu in [DUAL_DURATION_MU_MIN, ln(DUAL_DURATION_MU_MAX_DMAX_FACTOR * Dmax)], sigma <= DUAL_DURATION_SIGMA_MAX.
DUAL_DURATION_MU_MIN = -5.0
DUAL_DURATION_MU_MAX_DMAX_FACTOR = 10.0
DUAL_DURATION_SIGMA_MAX = 3.0
DUAL_DURATION_BOUND_TOL = 1e-3          # absolute tolerance (in mu and in sigma) for the `duration_at_bound_` flag (clarification 5.5b)

# WP9 A1 (tests/PREDECLARED_CRITERIA_WP9.md section 6): multi-start consensus. A baseline fit is FIT_UNSTABLE when fewer than FIT_CONSENSUS_M healthy restarts exist or when the
# best FIT_CONSENSUS_M healthy restarts (by log-likelihood) do not all agree at adjusted Rand index FIT_CONSENSUS_TAU (0.99 is the level this package calls "the same partition").
FIT_CONSENSUS_M = 3
FIT_CONSENSUS_TAU = 0.99
# WP9 A2 (tests/PREDECLARED_CRITERIA_WP9.md section 8 and Part 8; frozen in tests/DESIGN_FREEZE_WP9_A12.txt): a baseline fit whose mean maximum posterior probability (over the rows of the window) is below this
# is flagged FIT_UNSTABLE (reason `low_posterior_sharpness`) and never served as ALIGNED. 0.95 was chosen by the declared rule on the development seeds 9301..9360 as the largest value tied for the best
# coverage: it flagged 3 of 960 development fits and changed no alignment count (`wp9_evidence/a2_dev_analysis.txt`). 0 switches it off.
FIT_MIN_POSTERIOR_SHARPNESS = 0.95

# WP9 B3 (tests/PREDECLARED_CRITERIA_WP9.md Part 4): thresholds of the `k_adequacy` tag, set by the declared development rule on seeds 9531..9554 (`wp9_evidence/b3_dev_analysis.txt`, frozen in tests/DESIGN_FREEZE_WP9_B3.txt):
# regimes closer than 1.5 pooled within-regime sd per feature (root mean square over features) overlap strongly; delta = 0 means that signal (2) "K not clearly best" never fires (the selected K has the lowest BIC, so
# the gap to the runner-up is never negative: at d0 = 1.5 a larger delta added no detection on development); signal (3) "selected K is the largest candidate" is off (it added no detection on development). Development numbers: detection 15/19 wrong-K fits, false alarm 0/20 on right-K easy fits,
# 9/29 on all right-K fits. Never changes which K is selected.
K_ADEQUACY_MIN_SEPARATION = 1.5
K_ADEQUACY_BIC_MARGIN = 0.0
K_ADEQUACY_FLAG_LARGEST = False                 # signal (3): selected K is the largest candidate (K + 1 untried)

# WP9 B1 (tests/PREDECLARED_CRITERIA_WP9.md section 16): certification of prior-anchored fits (opt-in, `fit_hssm(certify_prior_anchored=True)`).
# `COLD_START_CERT_Z` = 3.0 (the N18 margin), FROZEN by the declared development rule on seeds 9631..9642 (`wp9_evidence/b1_dev_analysis.txt`, `tests/DESIGN_FREEZE_WP9_B1.txt`): NO value of {1, 2, 3} met the declared rule, so 3.0 was taken and the shortfall is reported there.
COLD_START_CERT_FRACTIONS = (0.50, 0.65, 0.80)   # rolling origins, as a share of the rows
COLD_START_CERT_MIN_ROWS = 24                    # fewer rows: not certifiable
COLD_START_CERT_MIN_TRAIN = 12                   # rows before the first origin
COLD_START_CERT_MIN_BLOCK = 3                    # rows of every held-out block
COLD_START_CERT_TAU = 0.25                       # parameter stability: regime-mean movement, in single-Gaussian standard deviations (fixed, not tuned)
COLD_START_CERT_Z = 3.0                          # margin in HAC standard errors of the mean gain over the single Gaussian

# WP9 A3 (tests/PREDECLARED_CRITERIA_WP9.md Part 7): K-change detection on the recent window of a refit (`k_change.assess_k_change`). The two deltas (BIC points that the fit at K_a + 1 / K_a - 1
# must beat the fit at K_a on the recent window by) were set by the declared development rule on seeds 9221..9232 (`wp9_evidence/a3_dev_analysis.txt`, frozen in tests/DESIGN_FREEZE_WP9_A3.txt):
# up = 0 (smallest delta with no-change false alarm <= 10%: 0/12), down = 40 (NO grid value met the 10% cap: the largest is taken and the shortfall is reported; false alarm 4/12 even there).
# Never changes which K is fitted.
K_CHANGE_RECENT_DAYS = 45
K_CHANGE_DELTA_UP = 0.0
K_CHANGE_DELTA_DOWN = 40.0

# WP9 B5 (tests/PREDECLARED_CRITERIA_WP9.md Part 9): monitoring primitives. `N17_PASSED` is False while the shrinkage-uncertainty criterion N17 has not passed at its unchanged floor
# (the N17 xfail in the test-suite stays); `confidence_display_gate` returns "early_estimate" for every input while it is False. Do not set it to True without the evidence (owner decision C1).
N17_PASSED = False
CONFIDENCE_DISPLAY_THRESHOLD = 0.85               # owner decision C1: a percentage only at a calibrated confidence of at least 85%
MONITOR_WINDOW_DAYS = 14                          # recent window of the predictability monitor
MONITOR_REFERENCE_DAYS = 28                       # trailing reference window (the same user's earlier scores)
MONITOR_DROP_Z = 3.5                              # drop in units of the standard error of the difference of the two window means (reference sd, lag-1 autocorrelation inflation; Part 9b)
MONITOR_MIN_DROP_SD = 0.75                        # and at least this many reference standard deviations (scale-free; Part 9b replaced the 0.5 nats of Part 9)
MONITOR_COOLDOWN_DAYS = 14                        # one event per episode

