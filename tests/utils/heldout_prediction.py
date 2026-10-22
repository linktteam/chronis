"""Held-out one-step-ahead prediction harness (ledger N18, the "real-fire drill").

Criteria and design were pre-declared in tests/PREDECLARED_CRITERIA_WP3.md, section 1, before this file was run.

What it provides
  * `continued_synthetic(seed)`: the A2-style training series (generator, n_segments=4) plus a held-out continuation of the SAME
    generative process (2 more segments). The training part is checked to be bit-identical to the generator's own output.
  * engines: the real engine, "stops learning" (M-step disabled), B2 (A_fast x 0.1 after each M-step), and the two breakers with a
    different random initialisation (`stops_alt`, `B2_alt`) used for the test-of-the-test.
  * `engine_scores(engine, seed)`: per-step one-step-ahead predictive log density on the held-out steps with parameters frozen.
  * `baseline_scores(seed)`: a single Gaussian fitted on the training data.
  * `paired_margin(d)`: mean, paired standard error, and mean - 3.0 * se.

Cached objects are plain numbers/arrays (no model is kept alive).
"""
import functools
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from backbone.hssm.dual_model import DualTimescaleHSSM
from data.synthetic_dual import generate_synthetic_dual_timescale_data
from tests.utils.synthetic_dual_fits import EngineADrivenToZero

SEEDS = (202, 404, 505, 606, 707, 808, 909, 1010)
MARGIN_SE = 3.0
K, F, L = 2, 4, 2
MEAN_DWELL = 15.0
N_TRAIN_SEGMENTS, N_TEST_SEGMENTS = 4, 2
MAX_DURATION = 25
ALT_SEED_OFFSET = 1000


def n_iter_for(seed: int) -> int:
    """A2 cell A (seed 202) used 25 iterations, cell B (seed 404) and everything else 20."""
    return 25 if seed == 202 else 20


# ------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ContinuedData:
    X_train: np.ndarray
    X_test: np.ndarray
    p_train: np.ndarray
    p_test: np.ndarray

    @property
    def X_full(self) -> np.ndarray:
        return np.vstack([self.X_train, self.X_test])


def continued_synthetic(seed: int) -> ContinuedData:
    """Replays data/synthetic_dual.py draw by draw for the training part (and checks it equals the generator's output),
    then CONTINUES the same random stream: draws for the extra segments' durations/regimes first, then the extra
    latent innovations, then the extra emission noise (the same order the generator uses within one series)."""
    rng = np.random.default_rng(seed)
    Q_scale = np.sqrt(np.diag(0.3 * np.eye(L)))              # same expressions as the generator, so draws are bit-identical
    R_scale = np.sqrt(np.full(F, 0.05))
    A = 0.75 * np.eye(L)
    C = np.zeros((F, L))
    for f in range(F):
        C[f, f % L] = 1.5
    d = [np.full(F, (k - (K - 1) / 2.0) * 4.0) for k in range(K)]

    def segments(n, current):
        regimes = []
        for _ in range(n):
            dur = int(np.clip(rng.lognormal(mean=np.log(MEAN_DWELL), sigma=0.3), 8, 30))
            regimes.extend([current] * dur)
            current = rng.choice([k for k in range(K) if k != current])
        return np.array(regimes, dtype=int), current

    p_tr, nxt = segments(N_TRAIN_SEGMENTS, 0)
    T_tr = len(p_tr)
    m_tr = np.zeros((T_tr, L))
    m_curr = rng.normal(scale=0.2, size=L)
    m_tr[0] = m_curr
    for t in range(1, T_tr):
        m_curr = A @ m_curr + rng.normal(scale=Q_scale, size=L)
        m_tr[t] = m_curr
    X_tr = np.zeros((T_tr, F))
    for t in range(T_tr):
        X_tr[t] = C @ m_tr[t] + d[p_tr[t]] + rng.normal(scale=R_scale, size=F)

    ref = generate_synthetic_dual_timescale_data(
        n_regimes=K, n_features=F, latent_dim=L, n_segments=N_TRAIN_SEGMENTS, mean_dwell=MEAN_DWELL, seed=seed)
    if not (np.array_equal(ref.X, X_tr) and np.array_equal(ref.true_p_t, p_tr) and np.array_equal(ref.true_m_t, m_tr)):
        raise AssertionError(f"seed {seed}: the replayed training part differs from the generator's output; the generator changed")

    p_te, _ = segments(N_TEST_SEGMENTS, nxt)
    T_te = len(p_te)
    m_te = np.zeros((T_te, L))
    for t in range(T_te):
        m_curr = A @ m_curr + rng.normal(scale=Q_scale, size=L)
        m_te[t] = m_curr
    X_te = np.zeros((T_te, F))
    for t in range(T_te):
        X_te[t] = C @ m_te[t] + d[p_te[t]] + rng.normal(scale=R_scale, size=F)
    return ContinuedData(X_train=X_tr, X_test=X_te, p_train=p_tr, p_test=p_te)


@functools.lru_cache(maxsize=None)
def continued_data(seed: int) -> ContinuedData:
    return continued_synthetic(seed)


# ------------------------------------------------------------------------------------------
# engines
# ------------------------------------------------------------------------------------------
class EngineStopsLearning(DualTimescaleHSSM):
    """The M-step is disabled: parameters stay at their initial values."""

    def _update_discrete_params(self, bwd):
        return None

    def _m_step_continuous(self, X, bwd, smoother):
        return None


ENGINE_CLASSES = {
    "real": DualTimescaleHSSM,
    "stops": EngineStopsLearning,
    "B2": EngineADrivenToZero,
    "stops_alt": EngineStopsLearning,     # same breaker, different random initialisation (test of the test)
    "B2_alt": EngineADrivenToZero,
}


def constructor_seed(engine: str, seed: int) -> int:
    return seed + ALT_SEED_OFFSET if engine.endswith("_alt") else seed


# ------------------------------------------------------------------------------------------
# scoring
# ------------------------------------------------------------------------------------------
def predictive_increments(model, X: np.ndarray):
    """lp[t] = log p(y_t | y_1:t-1) under the model's own forward filter (parameters frozen), and LL of the whole series."""
    fwd = model._forward_pass(X)
    lse = logsumexp(fwd.alpha.reshape(len(X), -1), axis=1)
    lp = np.diff(np.concatenate([[0.0], lse]))
    return lp, float(fwd.log_likelihood)


@functools.lru_cache(maxsize=None)
def engine_scores(engine: str, seed: int) -> dict:
    data = continued_data(seed)
    model = ENGINE_CLASSES[engine](
        n_regimes=K, n_features=F, latent_dim=L, max_duration=MAX_DURATION, seed=constructor_seed(engine, seed))
    model.fit(data.X_train, n_iter=n_iter_for(seed))
    before = model.cholesky_fallback_count_
    lp, ll_full = predictive_increments(model, data.X_full)
    _, ll_train = predictive_increments(model, data.X_train)
    T_tr = len(data.X_train)
    return dict(
        lp_test=lp[T_tr:].copy(),
        ll_full=ll_full, ll_train=ll_train,
        fit_state=str(model.fit_state_.value),
        fit_degraded=bool(model.fit_degraded_),
        fit_fallbacks=int(model.fit_cholesky_fallbacks_),
        scoring_fallbacks=int(model.cholesky_fallback_count_ - before),
    )


@functools.lru_cache(maxsize=None)
def baseline_scores(seed: int) -> np.ndarray:
    data = continued_data(seed)
    mu = data.X_train.mean(axis=0)
    cov = np.cov(data.X_train, rowvar=False, ddof=1)
    return multivariate_normal(mean=mu, cov=cov).logpdf(data.X_test)


def opponent_lp(opponent: str, seed: int) -> np.ndarray:
    return baseline_scores(seed) if opponent == "baseline" else engine_scores(opponent, seed)["lp_test"]


# ------------------------------------------------------------------------------------------
# statistic
# ------------------------------------------------------------------------------------------
def paired_margin(d, mult: float = MARGIN_SE):
    """mean, paired standard error (ddof=1), and mean - mult * se. The criterion is `lower > 0`."""
    d = np.asarray(d, dtype=float)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return mean, se, mean - mult * se


def per_seed_differences(engine: str, opponent: str, seeds=SEEDS) -> np.ndarray:
    """d_s = mean held-out step score of `engine` minus that of `opponent`, one value per seed."""
    return np.array([float(np.mean(engine_scores(engine, s)["lp_test"]) - np.mean(opponent_lp(opponent, s))) for s in seeds])


def beats(engine: str, opponent: str, seeds=SEEDS) -> bool:
    return paired_margin(per_seed_differences(engine, opponent, seeds))[2] > 0.0


OPPONENTS = ("baseline", "stops", "B2")
