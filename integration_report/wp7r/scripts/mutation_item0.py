"""Mutation check for tests/test_wp7r_item0_contract_doc.py (WP7-R item 0). usage: PYTHONPATH=. python3 integration_report/wp7r/scripts/mutation_item0.py [out.txt]
Each mutant edits ONE thing (the doc, or the code the doc describes) in a throw-away copy of the repo and the doc tests must FAIL. Prints one line per mutant; exit 0 if all are killed."""
import shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOC = "docs/INTEGRATION_CONTRACT_HSSM.md"
MUTANTS = [
    ("D01 doc: drop the NUMERICAL_FAILURE row", DOC, "| `NUMERICAL_FAILURE` | The Cholesky", "| `NUMERICAL_FAILURE_X` | The Cholesky"),
    ("D02 doc: drop the RUNNING row", DOC, "| `RUNNING` |", "| `RUNNINGX` |"),
    ("D03 doc: drop the MODEL_SELECTION_CONTESTED row", DOC, "| `MODEL_SELECTION_CONTESTED` | Baseline path only", "| `MODEL_SELECTION_CONTESTEDX` | Baseline path only"),
    ("D04 doc: empty the 'what the caller does' cell of FAILED", DOC, "Show nothing. Most cases raise `FittingConvergenceError` instead of returning (section 6). |", " |"),
    ("D05 doc: drop PriorBuildError row", DOC, "| `PriorBuildError` (an `HSSMError`)", "| `PriorBuildErrx` (an `HSSMError`)"),
    ("D06 doc: drop InternalStateError row", DOC, "| `InternalStateError` |", "| `InternalStatErr` |"),
    ("D07 doc: drop the late_fallbacks prefix", DOC, "| `late_fallbacks` |", "| `late_fb` |"),
    ("D08 doc: claim the dual collapse IS withheld", DOC, "so a collapsed dual regime is NOT withheld by this rule", "so a collapsed dual regime is withheld by this rule"),
    ("D09 doc: claim degenerate duration IS a withheld reason", DOC, "is NOT a withheld reason at the WP6 head", "is a withheld reason at the WP6 head"),
    ("D10 doc: drop 'no per-session confidence'", DOC, "There is **no per-session confidence** in `HSSMResult`", "There is a per-session confidence in `HSSMResult`"),
    ("D11 doc: wrong span limit in the checklist", DOC, "(span at most `MAX_TIMESTAMP_SPAN_DAYS` = 20,000;", "(span at most `MAX_TIMESTAMP_SPAN_DAYS` = 30,000;"),
    ("D12 doc: wrong default max_iter in the checklist", DOC, "default `max_iter` 200", "default `max_iter` 100"),
    ("D13 doc: forget window_gain_at_cap in the metadata table", DOC, "| `window_gain_at_cap` |", "| `window_gain_xx` |"),
    ("D14 doc: wrong COLLAPSE threshold", DOC, "`COLLAPSE_MIN_EFFECTIVE_DAYS` = 4 effective days (column sum", "`COLLAPSE_MIN_EFFECTIVE_DAYS` = 5 effective days (column sum"),
    ("C01 code: new FitState value", "backbone/hssm/model.py", '    MODEL_SELECTION_CONTESTED = "MODEL_SELECTION_CONTESTED"\n', '    MODEL_SELECTION_CONTESTED = "MODEL_SELECTION_CONTESTED"\n    STALE = "STALE"\n'),
    ("C02 code: rename the late_fallbacks reason prefix", "backbone/hssm/fitting.py", 'reasons.append(f"late_fallbacks:', 'reasons.append(f"late_fb:'),
    ("C03 code: add a not-converged state to the tuple", "backbone/hssm/fitting.py", "_NOT_CONVERGED_STATES = (FitState.MAX_ITER_REACHED, FitState.REJECTED_UPDATE, FitState.FAILED, FitState.RUNNING)",
     "_NOT_CONVERGED_STATES = (FitState.MAX_ITER_REACHED, FitState.REJECTED_UPDATE, FitState.FAILED, FitState.RUNNING, FitState.NUMERICAL_FAILURE)"),
    ("C04 code: add a confidence field to HSSMResult", "backbone/hssm/fitting.py", "    fit_collapsed: bool = False   ", "    confidence: float = 0.0\n    fit_collapsed: bool = False   "),
    ("C05 code: timestamp span limit", "backbone/hssm/config.py", "MAX_TIMESTAMP_SPAN_DAYS = 20000", "MAX_TIMESTAMP_SPAN_DAYS = 30000"),
    ("C06 code: default max_iter", "backbone/hssm/fitting.py", "n_initializations: int = 10,\n    max_iter: int = 200,\n    random_seed: Optional[int] = None,\n    allow_fast_test_fit", "n_initializations: int = 10,\n    max_iter: int = 100,\n    random_seed: Optional[int] = None,\n    allow_fast_test_fit"),
    ("C07 code: a new exception class under the library base is not documented", "backbone/hssm/model.py", "class InsightsWithheldError(HSSMError, RuntimeError):", "class InsightsWithheldError(RuntimeError):"),
    ("C08 code: dual result now sets duration box keys differently", "backbone/hssm/fitting.py", '"first_fit": dict(best_dual_model.duration_box_refit_)', '"first_fit_x": dict(best_dual_model.duration_box_refit_)'),
    ("C09 code: collapse threshold", "backbone/hssm/config.py", "COLLAPSE_MIN_EFFECTIVE_DAYS = 4.0", "COLLAPSE_MIN_EFFECTIVE_DAYS = 3.0"),
    ("C10 code: cold-start threshold", "backbone/hssm/config.py", "min_present_sessions: int = 30", "min_present_sessions: int = 20"),
]


def main():
    out = open(sys.argv[1], "w") if len(sys.argv) > 1 else None
    killed = 0
    for name, rel, old, new in MUTANTS:
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "repo"
            shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns(".git", "__pycache__", "wp7r_evidence", ".pytest_cache"))
            f = dst / rel
            txt = f.read_text()
            if txt.count(old) != 1:
                line = f"{name}: NOT APPLIED (pattern count {txt.count(old)})"
            else:
                f.write_text(txt.replace(old, new, 1))
                r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_wp7r_item0_contract_doc.py", "-q", "-x", "-p", "no:cacheprovider"], cwd=dst,
                                   env={"PYTHONPATH": str(dst), "PATH": "/usr/bin:/usr/local/bin:/bin", "HOME": td}, capture_output=True, text=True)
                dead = r.returncode != 0
                killed += dead
                line = f"{name}: {'KILLED' if dead else 'SURVIVED'}"
        print(line, flush=True)
        if out: out.write(line + "\n"); out.flush()
    summary = f"{killed}/{len(MUTANTS)} killed"
    print(summary)
    if out: out.write(summary + "\n")
    return 0 if killed == len(MUTANTS) else 1


if __name__ == "__main__":
    sys.exit(main())
