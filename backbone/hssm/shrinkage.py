"""
Shrinkage toward a shared prior: the WP5 prototype (cold start). NOT imported by `backbone/hssm/__init__.py` or by any default path.
Criteria and design: tests/PREDECLARED_CRITERIA_WP5.md (sections 4 and 5).

What it does
  * `SharedPrior`: a small object of AGGREGATE parameters (per regime), nothing else. No data, no sequences, no per-user parameters.
  * `ShrinkageDualHSSM(DualTimescaleHSSM)`: the engine initialised at the shared parameters and with a conjugate MAP M-step toward them.
    `n0` is the prior strength in pseudo-steps PER REGIME: the weight of the prior on regime k's parameters is about n0 / (n0 + W_k), W_k = the
    user's expected number of steps in regime k. Without a prior, or with n0 = 0 and no warm start, it is the unchanged engine (a test checks bit-identity).
  * `build_shared_prior`: fits the population users, aligns regimes (Hungarian on frame-free regime means) and the latent frame (linear map) to a
    reference, pools by the mean, in two rounds. Failures are excluded and counted.

Which parameters are shrunk (all closed form or through the existing count/Nelder-Mead structure)
  A_fast, b_fast, Q_fast   pseudo sufficient statistics of the shared dynamics (stationary moments of regime k) added to G, H, S, W  [hook _dynamics_sufficient_stats]
  C_obs, d_obs, R_obs      the same, per feature                                                                                    [_observation_map_step]
  A_slow, pi               pseudo transition / start counts: n0 / Dbar_k per regime (Dbar_k = mean of the shared duration pmf, steps per segment)
  dur_mu, dur_sigma        pseudo segments n0 / Dbar_k distributed as the shared duration pmf are added to the weighted counts of the Nelder-Mead fit
Not shrunk: m0, P0 (the engine never updates them: they stay at the values copied from the prior), and the latent frame (fixed by the warm start).
The prior's uncertainty does NOT enter m_t_std (`prior_uncertainty_in_m_t_std = False`); estimate_states is the unchanged engine method.
The engine's EM guard compares the UNPENALISED log-likelihood, which a MAP step can lower: such a fit can stop early (REJECTED_UPDATE) at the previous parameters.
"""
from __future__ import annotations

import dataclasses
import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import linear_sum_assignment
from scipy.stats import lognorm

from backbone.hssm.dual_model import DualTimescaleHSSM
from backbone.hssm.model import FitState, HSSMError

PARAM_NAMES = ("pi", "A_slow", "dur_mu", "dur_sigma", "A_fast", "b_fast", "Q_fast", "m0", "P0", "C_obs", "d_obs", "R_obs")
PRIOR_WHITELIST = ("K", "F", "L", "n_users", "n_excluded", "prior_uncertainty_in_m_t_std") + PARAM_NAMES
EXCLUSION_KEYS = ("unhealthy", "regime_ambiguous", "frame_failed")
MEDIAN_POOLED = ("dur_mu", "dur_sigma")   # pooled by the median (amendment A8): the engine's own duration M-step can drift to degenerate values
COND_LIMIT = 10.0           # frame map: condition number above this is a failure
RESIDUAL_LIMIT = 0.5        # frame map: relative residual of the loadings above this is a failure
AMBIGUITY_RATIO = 0.5       # regime matching: best cost must be at most this fraction of the second-best cost


class PriorBuildError(HSSMError):
    """The shared prior could not be built (too few usable users, or no healthy reference fit)."""


def expected_shapes(K: int, F: int, L: int) -> dict[str, tuple[int, ...]]:
    return {"pi": (K,), "A_slow": (K, K), "dur_mu": (K,), "dur_sigma": (K,), "A_fast": (K, L, L), "b_fast": (K, L), "Q_fast": (K, L, L),
            "m0": (K, L), "P0": (K, L, L), "C_obs": (K, F, L), "d_obs": (K, F), "R_obs": (K, F)}


# ------------------------------------------------------------------------------------------ the prior object
@dataclass
class SharedPrior:
    K: int
    F: int
    L: int
    pi: np.ndarray
    A_slow: np.ndarray
    dur_mu: np.ndarray
    dur_sigma: np.ndarray
    A_fast: np.ndarray
    b_fast: np.ndarray
    Q_fast: np.ndarray
    m0: np.ndarray
    P0: np.ndarray
    C_obs: np.ndarray
    d_obs: np.ndarray
    R_obs: np.ndarray
    n_users: int = 0                                           # how many users were pooled (a count, not data)
    n_excluded: dict = field(default_factory=dict)             # counts of excluded users by reason
    prior_uncertainty_in_m_t_std: bool = False                 # recorded design property (N17 depends on it)

    def __post_init__(self) -> None:
        shapes = expected_shapes(self.K, self.F, self.L)
        for name in PARAM_NAMES:
            arr = np.array(getattr(self, name), dtype=float, copy=True)
            if arr.shape != shapes[name]:
                raise ValueError(f"SharedPrior.{name} has shape {arr.shape}, expected {shapes[name]}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"SharedPrior.{name} contains non-finite values")
            setattr(self, name, arr)

    def params(self) -> dict[str, np.ndarray]:
        return {name: np.array(getattr(self, name), copy=True) for name in PARAM_NAMES}

    @classmethod
    def from_params(cls, params: dict, n_users: int = 0, n_excluded: dict | None = None) -> "SharedPrior":
        K, L = np.asarray(params["b_fast"]).shape
        F = np.asarray(params["d_obs"]).shape[1]
        return cls(K=K, F=F, L=L, n_users=int(n_users), n_excluded=dict(n_excluded or {}), **{name: params[name] for name in PARAM_NAMES})

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"K": self.K, "F": self.F, "L": self.L, "n_users": int(self.n_users),
                               "n_excluded": {str(k): int(v) for k, v in self.n_excluded.items()},
                               "prior_uncertainty_in_m_t_std": bool(self.prior_uncertainty_in_m_t_std)}
        out.update({name: getattr(self, name).tolist() for name in PARAM_NAMES})
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "SharedPrior":
        return cls(K=d["K"], F=d["F"], L=d["L"], n_users=d.get("n_users", 0), n_excluded=d.get("n_excluded", {}),
                   prior_uncertainty_in_m_t_std=d.get("prior_uncertainty_in_m_t_std", False), **{name: np.array(d[name]) for name in PARAM_NAMES})

    def assert_aggregate_only(self) -> None:
        """Raises AssertionError unless the object holds exactly the whitelisted aggregate fields with the expected shapes and nothing else
        (no data array, no extra attribute). Also checks the serialised form."""
        extra = sorted(set(vars(self)) - set(PRIOR_WHITELIST))
        assert not extra, f"SharedPrior has attributes outside the whitelist: {extra}"
        shapes = expected_shapes(self.K, self.F, self.L)
        for name in PARAM_NAMES:
            assert getattr(self, name).shape == shapes[name], f"{name} has an unexpected shape {getattr(self, name).shape}"
        assert set(self.n_excluded) <= set(EXCLUSION_KEYS) and all(isinstance(v, (int, np.integer)) for v in self.n_excluded.values())
        payload = json.loads(self.to_json())
        assert set(payload) <= set(PRIOR_WHITELIST), f"serialised form has keys outside the whitelist: {sorted(set(payload) - set(PRIOR_WHITELIST))}"
        n_numbers = sum(int(np.prod(s)) for s in shapes.values())
        assert sum(np.asarray(payload[n]).size for n in PARAM_NAMES) == n_numbers


# ------------------------------------------------------------------------------------------ helpers on parameters
def duration_pmf(dur_mu: np.ndarray, dur_sigma: np.ndarray, Dmax: int) -> np.ndarray:
    """(K, Dmax): the truncated, renormalised log-normal duration pmf, computed exactly as DualTimescaleHSSM._duration_logpmf does."""
    d = np.arange(1, Dmax + 1)
    out = np.zeros((len(dur_mu), Dmax))
    for k in range(len(dur_mu)):
        pdf = np.clip(lognorm.pdf(d, s=dur_sigma[k], scale=np.exp(dur_mu[k])), 1e-300, None)
        out[k] = pdf / pdf.sum()
    return out


def stationary_moments(A: np.ndarray, b: np.ndarray, Q: np.ndarray, P0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stationary mean and covariance of m_{t+1} = A m_t + b + w, w ~ N(0, Q). If A is not stable (spectral radius >= 0.999) the fallback is
    mean 0 and covariance P0 (documented fallback; never expected for the pooled parameters of a healthy population)."""
    L = len(b)
    if np.max(np.abs(np.linalg.eigvals(A))) < 0.999:
        mu = np.linalg.solve(np.eye(L) - A, b)
        Sigma = solve_discrete_lyapunov(A, Q)
        return mu, 0.5 * (Sigma + Sigma.T)
    return np.zeros(L), np.array(P0, copy=True)


def regime_observation_means(p: dict) -> np.ndarray:
    """(K, F): frame-free mean of the observations in each regime, d_k + C_k mu_k with mu_k the stationary latent mean."""
    K = p["b_fast"].shape[0]
    out = np.zeros_like(p["d_obs"])
    for k in range(K):
        mu, _ = stationary_moments(p["A_fast"][k], p["b_fast"][k], p["Q_fast"][k], p["P0"][k])
        out[k] = p["d_obs"][k] + p["C_obs"][k] @ mu
    return out


def extract_params(model: DualTimescaleHSSM) -> dict[str, np.ndarray]:
    return {name: np.array(getattr(model, name), copy=True) for name in PARAM_NAMES}


def match_regimes(ref_means: np.ndarray, user_means: np.ndarray) -> tuple[tuple[int, ...], float, float, bool]:
    """Hungarian matching of the user's regimes to the reference's by squared distance of the regime observation means.
    Returns (perm, best_cost, second_best_cost, ambiguous): aligned regime k is the user's regime perm[k]. `ambiguous` is True when the best
    total cost is not at most AMBIGUITY_RATIO times the best cost of any other assignment (checked by enumeration, K <= 6)."""
    K = ref_means.shape[0]
    if K > 6:
        raise ValueError("regime matching enumerates assignments and supports K <= 6")
    cost = ((ref_means[:, None, :] - user_means[None, :, :]) ** 2).sum(axis=2)
    rows, cols = linear_sum_assignment(cost)
    perm = tuple(int(c) for c in cols[np.argsort(rows)])
    best = float(sum(cost[k, perm[k]] for k in range(K)))
    others = [sum(cost[k, q[k]] for k in range(K)) for q in itertools.permutations(range(K)) if tuple(q) != perm]
    second = float(min(others)) if others else float("inf")
    ambiguous = bool(K > 1 and not (best < second and best <= AMBIGUITY_RATIO * second))   # best == second (e.g. all costs 0) is ambiguous
    return perm, best, second, ambiguous


def permute_regimes(p: dict, perm: Sequence[int]) -> dict:
    idx = np.asarray(perm, dtype=int)
    out = {n: np.array(v, copy=True) for n, v in p.items()}
    out["pi"] = p["pi"][idx]
    out["A_slow"] = p["A_slow"][np.ix_(idx, idx)]
    for n in ("dur_mu", "dur_sigma", "A_fast", "b_fast", "Q_fast", "m0", "P0", "C_obs", "d_obs", "R_obs"):
        out[n] = p[n][idx]
    return out


def frame_map(user_p: dict, ref_p: dict) -> tuple[np.ndarray, float, float]:
    """Linear map M (L x L) minimising sum_k ||C_k^user M - C_k^ref||_F^2, its condition number, and the relative residual."""
    Cu = user_p["C_obs"].reshape(-1, user_p["C_obs"].shape[2])
    Cr = ref_p["C_obs"].reshape(-1, ref_p["C_obs"].shape[2])
    M, *_ = np.linalg.lstsq(Cu, Cr, rcond=None)
    return M, float(np.linalg.cond(M)), float(np.linalg.norm(Cu @ M - Cr) / np.linalg.norm(Cr))


def apply_frame(p: dict, M: np.ndarray) -> dict:
    """Re-express the model in latent coordinates m' = M^-1 m (m = M m'): C' = C M, A' = M^-1 A M, b' = M^-1 b, Q' = M^-1 Q M^-T, m0' = M^-1 m0."""
    Mi = np.linalg.inv(M)
    out = {n: np.array(v, copy=True) for n, v in p.items()}
    out["C_obs"] = p["C_obs"] @ M
    out["A_fast"] = np.einsum("ij,kjl,lm->kim", Mi, p["A_fast"], M)
    out["b_fast"] = p["b_fast"] @ Mi.T
    out["Q_fast"] = np.einsum("ij,kjl,ml->kim", Mi, p["Q_fast"], Mi)
    out["m0"] = p["m0"] @ Mi.T
    out["P0"] = np.einsum("ij,kjl,ml->kim", Mi, p["P0"], Mi)
    return out


def align_user(user_p: dict, ref_p: dict) -> tuple[dict | None, dict]:
    """Regime matching, then latent frame map. Returns (aligned parameters or None, info). info['status'] is ok, regime_ambiguous or frame_failed."""
    perm, best, second, ambiguous = match_regimes(regime_observation_means(ref_p), regime_observation_means(user_p))
    info = {"perm": perm, "match_cost": best, "second_cost": second}
    if ambiguous:
        return None, {**info, "status": "regime_ambiguous"}
    permuted = permute_regimes(user_p, perm)
    M, cond, rel = frame_map(permuted, ref_p)
    info.update(cond=cond, rel_residual=rel)
    if not (np.isfinite(cond) and cond <= COND_LIMIT and rel <= RESIDUAL_LIMIT):
        return None, {**info, "status": "frame_failed"}
    return apply_frame(permuted, M), {**info, "status": "ok"}


def pool(aligned: Sequence[dict], median_pooled: Sequence[str] = MEDIAN_POOLED) -> dict:
    """Pool aligned parameters over users: the mean, except `median_pooled` (the two duration parameters) which use the median (bounded influence)."""
    return {n: (np.median if n in median_pooled else np.mean)([p[n] for p in aligned], axis=0) for n in PARAM_NAMES}


# ------------------------------------------------------------------------------------------ the model
class ShrinkageDualHSSM(DualTimescaleHSSM):
    """DualTimescaleHSSM initialised at a shared prior with a conjugate MAP M-step toward it. See the module docstring."""

    def __init__(self, *args, shared_prior: SharedPrior | None = None, n0: float = 0.0, warm_start: bool | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if n0 < 0 or not np.isfinite(n0):
            raise ValueError("n0 must be a finite number >= 0")
        if shared_prior is not None and (shared_prior.K, shared_prior.F, shared_prior.L) != (self.K, self.F, self.L):
            raise ValueError("the shared prior's (K, F, L) differ from the model's")
        self.shared_prior = shared_prior
        self.n0 = float(n0)
        self.warm_start = (shared_prior is not None) if warm_start is None else bool(warm_start)
        self.prior_uncertainty_in_m_t_std = False
        self._pseudo: dict[str, np.ndarray] | None = None

    # -- bookkeeping the integration will report
    @property
    def used_shared_prior(self) -> bool:
        return self.shared_prior is not None and (self.warm_start or self.n0 > 0.0)

    @property
    def _active(self) -> bool:
        return self.shared_prior is not None and self.n0 > 0.0

    # -- initialisation
    def _load_prior_parameters(self) -> None:
        for name, value in self.shared_prior.params().items():
            setattr(self, name, value)
        self.duration_optimizer_success_ = {k: True for k in range(self.K)}
        self.duration_fallback_used_ = False
        self.transition_fallback_rows_ = set()

    def _init_params(self, X: np.ndarray) -> None:
        super()._init_params(X)
        if self.shared_prior is not None and self.warm_start:
            self._load_prior_parameters()
        self._pseudo = self._prepare_pseudo() if self._active else None

    def become_shared_model(self) -> "ShrinkageDualHSSM":
        """The `shared` arm: the prior's parameters in an engine object, no fitting."""
        if self.shared_prior is None:
            raise ValueError("no shared prior")
        self._load_prior_parameters()
        self.fit_state_ = FitState.CONVERGED
        self._is_fitted = True
        return self

    # -- pseudo statistics of the shared model
    def _prepare_pseudo(self) -> dict[str, np.ndarray]:
        pr, K, F, L = self.shared_prior, self.K, self.F, self.L
        G0 = np.zeros((K, L + 1, L + 1))
        H0_dyn = np.zeros((K, L + 1, L))
        S0 = np.zeros((K, L, L))
        H0_obs = np.zeros((K, F, L + 1))
        sy0 = np.zeros((K, F))
        for k in range(K):
            mu, Sigma = stationary_moments(pr.A_fast[k], pr.b_fast[k], pr.Q_fast[k], pr.P0[k])
            G0[k, :L, :L] = Sigma + np.outer(mu, mu)
            G0[k, :L, L] = mu
            G0[k, L, :L] = mu
            G0[k, L, L] = 1.0
            Theta0 = np.vstack([pr.A_fast[k].T, pr.b_fast[k][None, :]])            # (L+1, L): [A^T; b^T], the layout of the engine's theta
            H0_dyn[k] = G0[k] @ Theta0
            S0[k] = Theta0.T @ G0[k] @ Theta0 + pr.Q_fast[k]
            for f in range(F):
                th = np.concatenate([pr.C_obs[k, f], [pr.d_obs[k, f]]])
                H0_obs[k, f] = G0[k] @ th
                sy0[k, f] = th @ G0[k] @ th + pr.R_obs[k, f]
        pmf = duration_pmf(pr.dur_mu, pr.dur_sigma, self.Dmax)
        dbar = pmf @ np.arange(1, self.Dmax + 1)
        return {"G0": G0, "H0_dyn": H0_dyn, "S0": S0, "H0_obs": H0_obs, "sy0": sy0, "pmf": pmf, "dbar": dbar}

    # -- M-step pieces
    def _update_discrete_params(self, bwd):
        if not self._active:
            return super()._update_discrete_params(bwd)
        pr, ps = self.shared_prior, self._pseudo
        c0 = self.n0 / ps["dbar"]                                                   # pseudo-segments per regime
        xi = bwd.xi_counts + c0[:, None] * pr.A_slow
        entry = np.array(bwd.entry_posterior, copy=True)
        entry[0] += c0[:, None] * ps["pmf"]
        gam = np.array(bwd.regime_posterior_smoothed, copy=True)
        gam[0] = gam[0] + c0 * pr.pi
        return super()._update_discrete_params(dataclasses.replace(bwd, xi_counts=xi, entry_posterior=entry, regime_posterior_smoothed=gam))

    def _dynamics_sufficient_stats(self, smoother):
        G, H, S, W = super()._dynamics_sufficient_stats(smoother)
        if self._active:
            ps = self._pseudo
            G = G + self.n0 * ps["G0"]
            H = H + self.n0 * ps["H0_dyn"]
            S = S + self.n0 * ps["S0"]
            W = W + self.n0
        return G, H, S, W

    def _m_step_continuous(self, X, bwd, smoother):
        super()._m_step_continuous(X, bwd, smoother)
        if self._active:
            self._observation_map_step(X, bwd, smoother, self.n0)

    def _observation_map_step(self, X: np.ndarray, bwd, smoother, n0: float) -> None:
        """MAP update of C_obs, d_obs, R_obs: the engine's weighted regressions plus n0 pseudo-steps of the shared observation model per regime."""
        ps, K, F, L = self._pseudo, self.K, self.F, self.L
        T = X.shape[0]
        gamma = bwd.regime_posterior_smoothed
        C_new, d_new, R_new = self.C_obs.copy(), self.d_obs.copy(), self.R_obs.copy()
        for k in range(K):
            Ex = smoother.smoothed_m_k[:, k, :]                                       # (T, L)
            Exx = smoother.smoothed_P_k[:, k] + Ex[:, :, None] * Ex[:, None, :]       # (T, L, L)
            Z = np.zeros((T, L + 1, L + 1))
            Z[:, :L, :L], Z[:, :L, L], Z[:, L, :L], Z[:, L, L] = Exx, Ex, Ex, 1.0
            Ex1 = np.concatenate([Ex, np.ones((T, 1))], axis=1)                       # (T, L+1)
            for f in range(F):
                valid = ~np.isnan(X[:, f])
                w = np.where(valid, gamma[:, k], 0.0)
                y = np.where(valid, X[:, f], 0.0)
                G = np.einsum("t,tij->ij", w, Z) + n0 * ps["G0"][k]
                H = np.einsum("t,ti->i", w * y, Ex1) + n0 * ps["H0_obs"][k, f]
                sum_y2 = float(np.sum(w * y * y)) + n0 * ps["sy0"][k, f]
                W = float(np.sum(w)) + n0
                if W <= 1e-8:
                    continue
                theta = np.linalg.solve(G + 1e-6 * np.eye(L + 1), H)
                C_new[k, f, :], d_new[k, f] = theta[:L], theta[L]
                R_new[k, f] = max((sum_y2 - float(theta @ H)) / W, 1e-6)
        self.C_obs, self.d_obs, self.R_obs = C_new, d_new, R_new


# ------------------------------------------------------------------------------------------ building the prior
def _healthy(model: DualTimescaleHSSM) -> bool:
    ll = model.log_likelihood_
    return (not model.fit_degraded_) and model.fit_state_ != FitState.NUMERICAL_FAILURE and ll is not None and bool(np.isfinite(ll))


def build_shared_prior(series: Sequence[np.ndarray], n_regimes: int, n_features: int, latent_dim: int, max_duration: int,
                       reference_seeds: Sequence[int], n_iter: int = 30, rounds: int = 2, min_users: int = 10,
                       scramble_labels_seed: int | None = None, duration_pooling: str = "median") -> tuple[SharedPrior, dict]:
    """Build the shared prior from the training series of the population users (the series are used here and not stored).
    Returns (prior, report); the report holds counts and parameter changes only. `scramble_labels_seed` is the "alignment broken" variant of
    criterion 6: after alignment every user's regimes are randomly permuted before pooling. `duration_pooling` is "median" (amendment A8, the
    default) or "mean" (the form pre-declared in section 4.4, kept so the two can be compared)."""
    if duration_pooling not in ("median", "mean"):
        raise ValueError("duration_pooling must be 'median' or 'mean'")
    median_pooled = MEDIAN_POOLED if duration_pooling == "median" else ()
    if len(series) < min_users:
        raise PriorBuildError(f"{len(series)} users given, the minimum is {min_users}")
    best, best_ll = None, -np.inf
    for s in reference_seeds:
        m = ShrinkageDualHSSM(n_regimes, n_features, latent_dim, max_duration, seed=int(s)).fit(series[0], n_iter=n_iter)
        if _healthy(m) and m.log_likelihood_ > best_ll:
            best, best_ll = m, m.log_likelihood_
    if best is None:
        raise PriorBuildError("no healthy reference fit")
    ref = SharedPrior.from_params(extract_params(best), n_users=1)
    report: dict[str, Any] = {"n_users_in": len(series), "reference_ll": float(best_ll), "rounds": []}
    rng = np.random.default_rng(scramble_labels_seed) if scramble_labels_seed is not None else None
    for r in range(rounds):
        aligned, counts, conds, rels = [], {k: 0 for k in EXCLUSION_KEYS}, [], []
        n_nonidentity, n_long_sigma = 0, 0
        for X in series:
            m = ShrinkageDualHSSM(n_regimes, n_features, latent_dim, max_duration, seed=0, shared_prior=ref, n0=0.0, warm_start=True).fit(X, n_iter=n_iter)
            if not _healthy(m):
                counts["unhealthy"] += 1
                continue
            p, info = align_user(extract_params(m), ref.params())
            if p is None:
                counts[info["status"]] += 1
                continue
            n_nonidentity += int(info["perm"] != tuple(range(n_regimes)))
            conds.append(info["cond"]), rels.append(info["rel_residual"])
            if rng is not None:
                p = permute_regimes(p, tuple(int(i) for i in rng.permutation(n_regimes)))
            aligned.append(p)
            n_long_sigma += int(np.sum(p["dur_sigma"] >= 5.0))
        if len(aligned) < min_users:
            raise PriorBuildError(f"round {r + 1}: only {len(aligned)} of {len(series)} users survived alignment (minimum {min_users}); exclusions {counts}")
        pooled = pool(aligned, median_pooled)
        by_mean = pool(aligned, ())
        change = {n: float(np.max(np.abs(pooled[n] - ref.params()[n]))) for n in PARAM_NAMES}
        report["rounds"].append({"round": r + 1, "n_pooled": len(aligned), "excluded": dict(counts), "n_nonidentity_regime_perm": n_nonidentity,
                                 "max_cond": float(max(conds)), "max_rel_residual": float(max(rels)), "max_abs_change_vs_init": change,
                                 "duration_pooling": duration_pooling, "n_user_regimes_with_dur_sigma_ge_5": n_long_sigma,
                                 "dur_mu_pooled_by_mean": by_mean["dur_mu"].tolist(), "dur_sigma_pooled_by_mean": by_mean["dur_sigma"].tolist()})
        ref = SharedPrior.from_params(pooled, n_users=len(aligned), n_excluded=counts)
    return ref, report
