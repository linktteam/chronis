"""WP10A A1 (finding F1): mutation control of the Python-version-independent fingerprint of `backbone/hssm/label_switching.py`. Each mutant plants ONE change in a function of the module in a throw-away
copy of the repository; the single test `test_function_bodies_are_unchanged_since_wp7r` must FAIL (KILLED). The change is a behaviour change or a behaviour-neutral rewrite: the fingerprint is a freeze of
the function bodies, so both must turn it red. usage (run it with the interpreter the suite runs on): PYTHONPATH=. python3.12 integration_report/wp10a/scripts/mutation_wp10a_a1.py [out.txt]"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp8" / "scripts"))
import mutation_runner as R

_base = shutil.ignore_patterns(".git", "__pycache__", "wp7r_evidence", "wp8_evidence", "wp9_evidence", "wp10a_evidence", ".pytest_cache")
R.IGNORE = lambda d, names: set(_base(d, names))

F = "backbone/hssm/label_switching.py"
MUT = [
    ("F01 reverse the sort order", F, "order = np.argsort(activity)", "order = np.argsort(-activity)"),
    ("F02 sort key over the wrong axis (canonicalize_labels)", F, "activity = np.linalg.norm(model.mu, axis=1)", "activity = np.linalg.norm(model.mu, axis=0)"),
    ("F03 one array is no longer permuted", F, "    model.dur_sigma = model.dur_sigma[order]\n", ""),
    ("F04 a default value changes", F, "get(int(old_k), True)", "get(int(old_k), False)"),
    ("F05 a behaviour-neutral rewrite of the audit attribute", F, "model._label_order_applied = order.tolist()", "model._label_order_applied = list(order)"),
    ("F06 the signature changes", F, "def canonicalize_labels(model) -> object:", "def canonicalize_labels(model, extra=None) -> object:"),
    ("F07 the alias no longer calls the function", F, "return canonicalize_labels(model)", "return model"),
    ("F08 wrong axis (activity_levels)", F, "return np.linalg.norm(model.mu, axis=1)", "return np.linalg.norm(model.mu, axis=0)"),
    ("F09 the return annotation changes", F, "def activity_levels(model) -> np.ndarray:", "def activity_levels(model) -> float:"),
    ("F10 a behaviour-neutral rename of a local variable", F, "order = np.argsort(activity)", "order = np.argsort(activity.copy())"),
    ("F11 the permutation of A uses only one axis", F, "model.A = model.A[np.ix_(order, order)]", "model.A = model.A[order]"),
    ("F12 a statement is added", F, "    model._label_order_applied = order.tolist()\n    return model\n\n\ndef canonicalize_regime_order", "    model._label_order_applied = order.tolist()\n    model._touched = True\n    return model\n\n\ndef canonicalize_regime_order"),
]
if __name__ == "__main__":
    sys.exit(R.run(MUT, ["tests/test_wp8_item4_label_convention.py::test_function_bodies_are_unchanged_since_wp7r"], sys.argv[1] if len(sys.argv) > 1 else None, workers=1))
