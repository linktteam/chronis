"""Mutation check for tests/test_wp8_item0_metadata.py (WP8 item 0e). usage: PYTHONPATH=. python3 integration_report/wp8/scripts/mutation_item0e.py [out.txt]
Each mutant edits ONE thing in a throw-away copy of the repo and the metadata tests must FAIL; a pattern that does not occur exactly once is reported NOT APPLIED."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mutation_runner import run

D, F, M, S = "backbone/hssm/dual_model.py", "backbone/hssm/fitting.py", "backbone/hssm/model.py", "docs/backbone/HSSM_RESULT_SCHEMA.md"
CONJ = "return bool(self._is_fitted and self.fit_state_ == FitState.CONVERGED and not self.fit_degraded_ and not any(self.duration_at_bound_.values()))"
MUTANTS = [
    ("E01 capability is the constant True again", D, '"is_production_grade_full_hssm": self.is_certified_fit_,', '"is_production_grade_full_hssm": True,'),
    ("E02 capability is the constant False", D, '"is_production_grade_full_hssm": self.is_certified_fit_,', '"is_production_grade_full_hssm": False,'),
    ("E03 unfitted model counts as certified", D, CONJ, CONJ.replace("self._is_fitted and ", "")),
    ("E04 fit state ignored", D, CONJ, CONJ.replace("self.fit_state_ == FitState.CONVERGED and ", "")),
    ("E05 degraded fit ignored", D, CONJ, CONJ.replace("not self.fit_degraded_ and ", "")),
    ("E06 duration at a wall ignored", D, CONJ, CONJ.replace("not any(self.duration_at_bound_.values())", "True")),
    ("E07 only some regimes checked for the wall (all instead of any)", D, CONJ, CONJ.replace("not any(", "not all(")),
    ("E08 MAX_ITER_REACHED accepted as converged", D, CONJ, CONJ.replace("self.fit_state_ == FitState.CONVERGED", "self.fit_state_ in (FitState.CONVERGED, FitState.MAX_ITER_REACHED)")),
    ("E09 dual optimizer_success is the constant True again", F, "optimizer_success=not best_dual_model.duration_fallback_used_,", "optimizer_success=True,"),
    ("E10 dual optimizer_success inverted", F, "optimizer_success=not best_dual_model.duration_fallback_used_,", "optimizer_success=best_dual_model.duration_fallback_used_,"),
    ("E11 dual identity says gaussian again", D, 'return "explicit_truncated_lognormal"', 'return "explicit_truncated_gaussian"'),
    ("E12 baseline identity says gaussian again", M, 'return "explicit_truncated_lognormal"', 'return "explicit_truncated_gaussian"'),
    ("E13 schema doc row names the gaussian again", S, '`"explicit_truncated_lognormal"` | `"explicit_truncated_lognormal"`', '`"explicit_truncated_gaussian"` | `"explicit_truncated_lognormal"`'),
    ("E14 dual pmf not renormalised", D, "            pdf = pdf / pdf.sum()\n            logpmf[k] = np.log(pdf)\n        return logpmf\n\n    # ---------- Kim", "            logpmf[k] = np.log(pdf)\n        return logpmf\n\n    # ---------- Kim"),
    ("E15 baseline pmf is a gaussian shape", M, "            pdf = lognorm.pdf(d, s=self.dur_sigma[k], scale=np.exp(self.dur_mu[k]))\n            pdf = np.clip(pdf, 1e-300, None)\n            pdf = pdf / pdf.sum()\n            logpmf[k] = np.log(pdf)\n        return logpmf\n\n    def _emission_loglik",
     "            pdf = np.exp(-0.5 * ((d - np.exp(self.dur_mu[k])) / self.dur_sigma[k]) ** 2)\n            pdf = np.clip(pdf, 1e-300, None)\n            pdf = pdf / pdf.sum()\n            logpmf[k] = np.log(pdf)\n        return logpmf\n\n    def _emission_loglik"),
]

if __name__ == "__main__":
    sys.exit(run(MUTANTS, ["tests/test_wp8_item0_metadata.py"], sys.argv[1] if len(sys.argv) > 1 else None, workers=2))
