"""Mutation check for tests/test_wp7r_item5_dual_honesty.py (WP7-R item 5). usage: PYTHONPATH=. python3 integration_report/wp7r/scripts/mutation_item5.py [out.txt] [--workers 2]
Each mutant edits ONE thing in backbone/hssm/fitting.py in a throw-away copy of the repo and the tests must FAIL (first failure stops the run). Exit 0 if all are killed."""
import shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
M = "backbone/hssm/fitting.py"
COL = '"collapsed": bool(eff.min() < COLLAPSE_MIN_EFFECTIVE_DAYS),\n        "source": "filtered_regime_posterior",'
MUTANTS = [
    ("F01 R1 threshold is not strict (<=)", M, COL, COL.replace("<", "<=")),
    ("F02 R1 reports the maximum as the minimum", M, '"min_effective_days": float(eff.min()),\n        "min_effective_days_threshold"', '"min_effective_days": float(eff.max()),\n        "min_effective_days_threshold"'),
    ("F03 R1 sums over regimes instead of rows", M, "eff = np.asarray(p_t_posterior, dtype=float).sum(axis=0)", "eff = np.asarray(p_t_posterior, dtype=float).sum(axis=1)"),
    ("F04 R1 collapsed always False", M, COL, COL.replace("bool(eff.min() < COLLAPSE_MIN_EFFECTIVE_DAYS)", "False")),
    ("F05 R1 result.fit_collapsed not wired", M, 'fit_collapsed=bool(dual_collapse.get("collapsed", False)),', "fit_collapsed=False,"),
    ("F06 R1 result.collapse_report not wired", M, "collapse_report=dual_collapse,", "collapse_report={},"),
    ("F07 R1 computed on all rows but the last", M, "dual_collapse = dual_collapse_report(p_t_post) ", "dual_collapse = dual_collapse_report(p_t_post[:-1]) "),
    ("F08 R2 rising test reversed", M, "logpmf[k, -1] > logpmf[k, -2]", "logpmf[k, -1] < logpmf[k, -2]"),
    ("F09 R2 rising test is >=", M, "logpmf[k, -1] > logpmf[k, -2]", "logpmf[k, -1] >= logpmf[k, -2]"),
    ("F10 R2 rising compares with the wrong step", M, "logpmf[k, -1] > logpmf[k, -2]", "logpmf[k, -1] > logpmf[k, -4]"),
    ("F11 R2 one-step lattice guard removed", M, "model.Dmax >= 2 and logpmf", "logpmf"),
    ("F12 R2 ignores the wall of the box", M, "if bool(at_bound.get(k, False)) or rising[k]]", "if rising[k]]"),
    ("F13 R2 ignores a rising pmf", M, "if bool(at_bound.get(k, False)) or rising[k]]", "if bool(at_bound.get(k, False))]"),
    ("F14 R2 degenerate always False", M, '"degenerate": bool(regimes), "Dmax"', '"degenerate": False, "Dmax"'),
    ("F15 R2 result.fit_duration_degenerate not wired", M, 'fit_duration_degenerate=bool(dual_degeneracy["degenerate"]),', "fit_duration_degenerate=False,"),
    ("F16 R2 reason prefix renamed", M, 'f"degenerate_duration: regime(s)', 'f"degenerate: regime(s)'),
    ("F17 R2 reason never added", M, "if self.fit_duration_degenerate:", "if False:"),
    ("F18 R2 reason drops the regime list", M, "{dr.get('degenerate_regimes', [])}", "{[]}"),
    ("F19 R3 retries even when the fit is not degenerate", M, 'if max_duration_retry is not None and dual_duration_degeneracy(best_dual_model)["degenerate"]:', "if max_duration_retry is not None:"),
    ("F20 R3 always adopts the retry", M, 'adopted = not dual_duration_degeneracy(retry_model)["degenerate"]', "adopted = True"),
    ("F21 R3 never adopts the retry", M, 'adopted = not dual_duration_degeneracy(retry_model)["degenerate"]', "adopted = False"),
    ("F22 R3 retry runs with the original Dmax", M, "max_duration=int(max_duration_retry), base_seed=(random_seed or 0) + best_k * 1000", "max_duration=int(max_duration), base_seed=(random_seed or 0) + best_k * 1000"),
    ("F23 R3 BIC of the retry not recorded", M, "                bic_by_k[best_k] = best_bic\n", "                pass\n"),
    ("F24 R3 metadata record not written", M, 'dual_result.convergence_metadata["dmax_retry"] = dmax_retry_info', "pass"),
    ("F25 R3 retry run-log entries not tagged", M, "full_run_log.extend(dict(e, dmax_retry=True) for e in retry_log)", "full_run_log.extend(retry_log)"),
    ("F26 R3 validation lets retry == max_duration through", M, "int(max_duration_retry) <= int(max_duration)", "int(max_duration_retry) < int(max_duration)"),
    ("F27 R3 validation drops the shared-prior check", M, 'if shared_prior is not None:\n            raise ValueError("max_duration_retry cannot', 'if False:\n            raise ValueError("max_duration_retry cannot'),
    ("F28 R3 validation drops the model-type check", M, 'if model_type not in ("dual_timescale", "dual", "full_hssm"):\n            raise ValueError("max_duration_retry applies', 'if False:\n            raise ValueError("max_duration_retry applies'),
    ("F29 R3 validation accepts a float", M, "not isinstance(max_duration_retry, (int, np.integer)) or ", ""),
    ("F30 R3 record says the wrong 'from'", M, '"from": int(max_duration),', '"from": int(max_duration_retry),'),
    ("F31 R3 record says the first fit was not degenerate", M, '"first_fit_degenerate": True,', '"first_fit_degenerate": False,'),
]


def run_one(m):
    name, rel, old, new = m
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "repo"
        shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns(".git", "__pycache__", "wp7r_evidence", ".pytest_cache"))
        f = dst / rel
        txt = f.read_text()
        if txt.count(old) != 1:
            return name, None, f"NOT APPLIED (pattern count {txt.count(old)})"
        f.write_text(txt.replace(old, new, 1))
        r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_wp7r_item5_dual_honesty.py", "-q", "-x", "-p", "no:cacheprovider"], cwd=dst,
                           env={"PYTHONPATH": str(dst), "PATH": "/usr/bin:/usr/local/bin:/bin", "HOME": td}, capture_output=True, text=True)
        return name, r.returncode != 0, "KILLED" if r.returncode != 0 else "SURVIVED"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
    if "--workers" in sys.argv:
        args = [a for a in args if a != sys.argv[sys.argv.index("--workers") + 1]]
    out = open(args[0], "w") if args else None
    killed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for name, dead, status in ex.map(run_one, MUTANTS):
            killed += bool(dead)
            line = f"{name}: {status}"
            print(line, flush=True)
            if out: out.write(line + "\n"); out.flush()
    summary = f"{killed}/{len(MUTANTS)} killed"
    print(summary)
    if out: out.write(summary + "\n")
    return 0 if killed == len(MUTANTS) else 1


if __name__ == "__main__":
    sys.exit(main())
