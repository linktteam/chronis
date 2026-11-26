"""Mutation check for tests/test_wp9_b1_cert.py (WP9 B1). usage: PYTHONPATH=. python3 integration_report/wp9/scripts/mutation_b1.py [out.txt]
Pure mutants (cold_start_cert.py, config.py) run against the fast tests, wiring mutants (fitting.py) against the live-fit tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp8" / "scripts"))
import shutil
import mutation_runner as R

_base = shutil.ignore_patterns(".git", "__pycache__", "wp7r_evidence_skip", "wp8_evidence", "wp9_evidence", ".pytest_cache")
R.IGNORE = lambda d, names: set(_base(d, names))                    # wp7r_evidence is kept: the live tests read the prior from it

CC, CFG, FIT = "backbone/hssm/cold_start_cert.py", "backbone/hssm/config.py", "backbone/hssm/fitting.py"
PURE = [
    ("C01 too few rows is not refused", CC, "    if T < COLD_START_CERT_MIN_ROWS:", "    if T < 10:"),
    ("C02 origins are rounded down", CC, "origins = [int(math.ceil(f * T - 1e-9)) for f in fractions]", "origins = [int(f * T) for f in fractions]"),
    ("C03 the last block stops early", CC, "ends = origins[1:] + [int(T)]", "ends = origins[1:] + [int(T) - 1]"),
    ("C04 the bandwidth is n / 8", CC, "m = min(int(math.ceil(n / 4.0)) if bandwidth is None", "m = min(int(math.ceil(n / 8.0)) if bandwidth is None"),
    ("C05 the Bartlett weights are dropped", CC, "lrv += 2.0 * (1.0 - j / (m + 1.0)) * float(c[:-j] @ c[j:]) / n", "lrv += 2.0 * float(c[:-j] @ c[j:]) / n"),
    ("C06 autocovariances count once", CC, "lrv += 2.0 * (1.0 - j / (m + 1.0))", "lrv += 1.0 * (1.0 - j / (m + 1.0))"),
    ("C07 two increments are enough", CC, "    if n < 3 or not np.all(np.isfinite(d)):\n        return float(\"nan\")", "    if n < 2 or not np.all(np.isfinite(d)):\n        return float(\"nan\")"),
    ("C08 the Gaussian uses ddof 0", CC, "cov = np.atleast_2d(np.cov(X_train, rowvar=False, ddof=1))", "cov = np.atleast_2d(np.cov(X_train, rowvar=False, ddof=0))"),
    ("C09 regimes are compared by index only", CC, "best = min(float((((a - b[list(p)]) / sd) ** 2).mean()) for p in itertools.permutations(range(a.shape[0])))", "best = float((((a - b) / sd) ** 2).mean())"),
    ("C10 stability ignores the scale of the features", CC, "(((a - b[list(p)]) / sd) ** 2)", "(((a - b[list(p)]) / 1.0) ** 2)"),
    ("C11 a lower bound of exactly zero certifies", CC, "            if not lower > 0.0:", "            if not lower >= 0.0:"),
    ("C12 the margin is ignored", CC, "lower = mean - float(z) * se", "lower = mean"),
    ("C13 stability at the level is refused", CC, "elif not float(stability) <= float(tau):", "elif not float(stability) < float(tau):"),
    ("C14 non-finite increments pass", CC, "    if d is None or len(d) < 3 or not np.all(np.isfinite(d)):", "    if d is None or len(d) < 3:"),
    ("C15 a converged fit goes on to be certified", CC, "        if el.get(\"em_state\") == \"CONVERGED\":", "        if False:"),
    ("C16 other withhold reasons are overridden", CC, "        if el.get(\"other_reasons\"):", "        if False:"),
    ("C17 a degraded refit is accepted", CC, "            if getattr(m, \"degraded\", False):", "            if False:"),
    ("C18 a failed refit is skipped", CC, "                return dict(base, reasons=[R_REFIT], blocks=[list(b) for b in blocks])", "                continue"),
    ("C19 the refit sees the held-out block", CC, "                m = refit_fn(X[:o])", "                m = refit_fn(X[:e])"),
    ("C20 the Gaussian sees the held-out block", CC, "parts.append(lp - gaussian_heldout(X[:o], X[o:e]))", "parts.append(lp - gaussian_heldout(X[:e], X[o:e]))"),
    ("C21 the gain has the wrong sign", CC, "parts.append(lp - gaussian_heldout(X[:o], X[o:e]))", "parts.append(gaussian_heldout(X[:o], X[o:e]) - lp)"),
    ("C22 stability looks at the first refit", CC, "            last = m\n", "            last = last if last is not None else m\n"),
    ("C23 a fit without a prior is certifiable", CC, "if not el.get(\"used_shared_prior\") or not (el.get(\"n0\") or 0) > 0:", "if not (el.get(\"n0\") or 0) > 0:"),
    ("C24 a failed EM fit is eligible", CC, "ELIGIBLE_EM_STATES = (\"REJECTED_UPDATE\", \"MAX_ITER_REACHED\")", "ELIGIBLE_EM_STATES = (\"REJECTED_UPDATE\", \"MAX_ITER_REACHED\", \"FAILED\")"),
    ("C25 the default margin is 0.5", CFG, "COLD_START_CERT_Z = 3.0 ", "COLD_START_CERT_Z = 0.5 "),
    ("C26 the stability level is 0.5 (declared 0.25)", CFG, "COLD_START_CERT_TAU = 0.25 ", "COLD_START_CERT_TAU = 0.5 "),
    ("C27 an unexpected error certifies", CC, "return dict(base, reasons=[R_ERROR], error=", "return dict(base, state=STATE_CERTIFIED, reasons=[R_ERROR], error="),
]
WIRING = [
    ("W01 the certificate ignores the prior", FIT, "c.get(\"state\") == \"CERTIFIED_PRIOR_ANCHORED\" and self.used_shared_prior and", "c.get(\"state\") == \"CERTIFIED_PRIOR_ANCHORED\" and"),
    ("W02 the certificate counts in any EM state", FIT, "getattr(self.fit_state, \"value\", self.fit_state) in (\"REJECTED_UPDATE\", \"MAX_ITER_REACHED\"))", "True)"),
    ("W03 a certified fit stays withheld", FIT, "not ignore_non_convergence and not self.certified_prior_anchored:", "not ignore_non_convergence:"),
    ("W04 the eligibility asks with the non-convergence reason", FIT, "if underlying in _NOT_CONVERGED_STATES and not ignore_non_convergence and", "if underlying in _NOT_CONVERGED_STATES and"),
    ("W05 the certification is not attached", FIT, "            _attach_certification(dual_result, X, best_dual_model, best_k, n_initializations, max_iter, max_duration, (random_seed or 0) + best_k * 1000, shared_prior, shared_prior_n0)", "            pass"),
    ("W06 certification without a prior is accepted", FIT, "        if model_type not in (\"dual_timescale\", \"dual\", \"full_hssm\") or shared_prior is None:\n            raise ValueError(\"certify_prior_anchored", "        if False:\n            raise ValueError(\"certify_prior_anchored"),
    ("W07 the refit seed differs", FIT, "max_iter, max_duration, (random_seed or 0) + best_k * 1000, shared_prior, shared_prior_n0)\n        _log", "max_iter, max_duration, (random_seed or 0), shared_prior, shared_prior_n0)\n        _log"),
    ("W08 the certificate state string is wrong", FIT, "c.get(\"state\") == \"CERTIFIED_PRIOR_ANCHORED\"", "c.get(\"state\") == \"NOT_CERTIFIED\""),
    ("W09 an error in the certification certifies", FIT, "result.certification = dict(state=\"NOT_CERTIFIED\", reasons=[\"certification_error\"]", "result.certification = dict(state=\"CERTIFIED_PRIOR_ANCHORED\", reasons=[\"certification_error\"]"),
]
LIVE = "default_call_is_unchanged or certified_fit_drops or certificate_is_void or exception_in_the_certification or refit_is_the_production or requires_a_prior"
FAST = "not (" + LIVE.replace(" or ", " or ") + ")"

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    a = R.run(PURE, ["tests/test_wp9_b1_cert.py", "-k", FAST], out, workers=1)
    b = R.run(WIRING, ["tests/test_wp9_b1_cert.py", "-k", LIVE], (out + ".wiring") if out else None, workers=1)
    sys.exit(a or b)
