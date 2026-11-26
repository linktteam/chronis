"""Mutation check for tests/test_wp9_b3_k_adequacy.py (WP9 B3). usage: PYTHONPATH=. python3 integration_report/wp9/scripts/mutation_b3.py [out.txt]
Pure mutants (k_adequacy.py, alerts.py, config.py) run against the fast tests, wiring mutants (fitting.py) against the live-fit tests."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp8" / "scripts"))
import mutation_runner as R

_base = shutil.ignore_patterns(".git", "__pycache__", "wp7r_evidence", "wp8_evidence", "wp9_evidence", ".pytest_cache")
R.IGNORE = lambda d, names: set(_base(d, names))

KA, AL, CFG, FIT = "backbone/hssm/k_adequacy.py", "backbone/hssm/alerts.py", "backbone/hssm/config.py", "backbone/hssm/fitting.py"
PURE = [
    ("B01 the separation sums over features instead of averaging", KA, "/ ((var[i] + var[j]) / 2.0)).mean()))", "/ ((var[i] + var[j]) / 2.0)).sum()))"),
    ("B02 the closest pair is never found", KA, "            if d < best:", "            if d > best:"),
    ("B03 the variance of one regime only", KA, "/ ((var[i] + var[j]) / 2.0)", "/ var[i]"),
    ("B04 the overlap level is inclusive", KA, "if math.isfinite(sep) and sep < min_separation:", "if math.isfinite(sep) and sep <= min_separation:"),
    ("B05 the BIC margin is inclusive", KA, "if runner_up is not None and runner_up[1] < bic_margin:", "if runner_up is not None and runner_up[1] <= bic_margin:"),
    ("B06 the runner-up is the farthest candidate", KA, "runner_up = min(others.items(), key=lambda kv: kv[1])", "runner_up = max(others.items(), key=lambda kv: kv[1])"),
    ("B07 the largest-candidate reason is removed", KA, "if len(cands) > 1 and int(selected_k) == cands[-1]:", "if False:"),
    ("B08 a single candidate counts as the largest", KA, "if len(cands) > 1 and int(selected_k) == cands[-1]:", "if int(selected_k) == cands[-1]:"),
    ("B09 a missing stability assessment is trusted", KA, "if fit_unstable is None or fit_unstable:", "if fit_unstable:"),
    ("B10 the best guess is not marked as shown", KA, "best_guess_shown=True,", "best_guess_shown=False,"),
    ("B11 the tag ignores the reasons", KA, "return dict(k_unsure=bool(reasons),", "return dict(k_unsure=False,"),
    ("B12 the overlap signal is removed", KA, "        reasons.append(REASON_OVERLAP)", "        pass"),
    ("B13 the restart signal is removed", KA, "        reasons.append(REASON_RESTARTS)", "        pass"),
    ("B14 the user hash is 8 digits", AL, "hexdigest()[:16]", "hexdigest()[:8]"),
    ("B15 personal fields are accepted", AL, "    if bad:\n        raise ValueError", "    if False:\n        raise ValueError"),
    ("B16 the event carries the user id", AL, "user_id_hash=user_id_hash(user_id), **fields)", "user_id_hash=str(user_id), **fields)"),
    ("B17 a failing handler breaks the caller", AL, "        except Exception:                                                  # a broken handler", "        except KeyError:                                                   # a broken handler"),
    ("B18 the alert is logged at INFO", AL, "    logger.warning(\"CHRONIS_DEV_ALERT", "    logger.info(\"CHRONIS_DEV_ALERT"),
    ("B19 the default BIC margin is 7", CFG, "K_ADEQUACY_BIC_MARGIN = 6.0", "K_ADEQUACY_BIC_MARGIN = 7.0"),
    ("B20 the default separation is 1.5", CFG, "K_ADEQUACY_MIN_SEPARATION = 1.0", "K_ADEQUACY_MIN_SEPARATION = 1.5"),
    ("B21 a result without an assessment is unsure", FIT, "return bool((self.k_adequacy or {}).get(\"k_unsure\", False))", "return bool((self.k_adequacy or {}).get(\"k_unsure\", True))"),
]
WIRING = [
    ("W01 the assessment is not attached", FIT, "    _attach_k_adequacy(baseline_result, report, tuple(candidate_ks), user_id, dataset_hash)", "    pass"),
    ("W02 no alert for an unsure fit", FIT, "    if ad[\"k_unsure\"]:\n        sig = ad.get(\"signals\", {})", "    if False:\n        sig = ad.get(\"signals\", {})"),
    ("W03 an alert for every fit", FIT, "    if ad[\"k_unsure\"]:\n        sig = ad.get(\"signals\", {})", "    if True:\n        sig = ad.get(\"signals\", {})"),
    ("W04 a failed assessment is reported as sure", FIT, "ad = dict(k_unsure=True, reasons=[\"assessment_failed\"]", "ad = dict(k_unsure=False, reasons=[\"assessment_failed\"]"),
    ("W05 the stability flag is not passed", FIT, "result.model.var, rs.get(\"fit_unstable\"), candidates=candidates)", "result.model.var, False, candidates=candidates)"),
]
FAST = "not live_fit and not selected_k_is_the_bic and not unsure_fit_raises and not clean_fit_raises and not exception_in_the_assessment"
LIVE = "live_fit or selected_k_is_the_bic or unsure_fit_raises or clean_fit_raises or exception_in_the_assessment"

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    a = R.run(PURE, ["tests/test_wp9_b3_k_adequacy.py", "-k", FAST], out, workers=1)
    b = R.run(WIRING, ["tests/test_wp9_b3_k_adequacy.py", "-k", LIVE], (out + ".wiring") if out else None, workers=1)
    sys.exit(a or b)
