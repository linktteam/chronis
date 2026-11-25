"""Mutation check for tests/test_wp8_item2_permute_property.py (WP8 item 2). usage: PYTHONPATH=. python3 integration_report/wp8/scripts/mutation_item2.py [out.txt]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mutation_runner import run

S, LS = "backbone/hssm/shrinkage.py", "backbone/hssm/label_switching.py"
LOOP = '("dur_mu", "dur_sigma", "A_fast", "b_fast", "Q_fast", "m0", "P0", "C_obs", "d_obs", "R_obs")'
NAMES = ["dur_mu", "dur_sigma", "A_fast", "b_fast", "Q_fast", "m0", "P0", "C_obs", "d_obs", "R_obs"]
MUTANTS = [
    ("P01 pi not permuted", S, '    out["pi"] = p["pi"][idx]\n', ""),
    ("P02 A_slow permuted on rows only", S, 'out["A_slow"] = p["A_slow"][np.ix_(idx, idx)]', 'out["A_slow"] = p["A_slow"][idx]'),
]
for n in NAMES:
    kept = ", ".join(f'"{x}"' for x in NAMES if x != n)
    MUTANTS.append((f"P-{n} not permuted", S, LOOP, f"({kept},)"))
MUTANTS += [
    ("P13 inverse permutation applied", S, "idx = np.asarray(perm, dtype=int)\n    out = {n: np.array(v, copy=True) for n, v in p.items()}", "idx = np.argsort(np.asarray(perm, dtype=int))\n    out = {n: np.array(v, copy=True) for n, v in p.items()}"),
    ("P14 a regime-indexed name missing from PARAM_NAMES", S, '"C_obs", "d_obs", "R_obs")\nPRIOR_WHITELIST', '"C_obs", "d_obs")\nPRIOR_WHITELIST'),
    ("B01 baseline pi not permuted", LS, "    model.pi = model.pi[order]\n", ""),
    ("B02 baseline A permuted on rows only", LS, "model.A = model.A[np.ix_(order, order)]", "model.A = model.A[order]"),
    ("B03 baseline mu not permuted", LS, "    model.mu = model.mu[order]\n", ""),
    ("B04 baseline var not permuted", LS, "    model.var = model.var[order]\n", ""),
    ("B05 baseline dur_mu not permuted", LS, "    model.dur_mu = model.dur_mu[order]\n", ""),
    ("B06 baseline dur_sigma not permuted", LS, "    model.dur_sigma = model.dur_sigma[order]\n", ""),
    ("B07 baseline sorted descending", LS, "order = np.argsort(activity)", "order = np.argsort(-activity)"),
]

if __name__ == "__main__":
    sys.exit(run(MUTANTS, ["tests/test_wp8_item2_permute_property.py"], sys.argv[1] if len(sys.argv) > 1 else None, workers=1))
