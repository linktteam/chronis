"""Mutation check for tests/test_wp7r_item2_bic_count.py (WP7-R item 2). usage: PYTHONPATH=. python3 integration_report/wp7r/scripts/mutation_item0.py [out.txt]
Each mutant edits ONE thing (the doc, or the code the doc describes) in a throw-away copy of the repo and the doc tests must FAIL. Prints one line per mutant; exit 0 if all are killed."""
import shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
M = "backbone/hssm/model.py"
MUTANTS = [
    ("C01 default convention flipped back to legacy", M, 'n_params_convention: str = "free"', 'n_params_convention: str = "legacy"'),
    ("C02 free count equals the legacy count", M, "n_transition = K * (K - 2)", "n_transition = K * (K - 1)"),
    ("C03 free count too small", M, "n_transition = K * (K - 2)", "n_transition = K * (K - 3)"),
    ("C04 unknown convention silently accepted", M, '''        else:
            raise ValueError(f"n_params_convention must be 'legacy' or 'free', got {self.n_params_convention!r}")''', "        else:\n            n_transition = K * (K - 1)"),
    ("C05 legacy count off by one", M, "n_transition = K * (K - 1)\n        elif", "n_transition = K * (K - 1) + 1\n        elif"),
    ("C06 BIC ignores the convention", M, "return -2 * self.log_likelihood_ + self.n_params() * np.log(max(n_obs, 1))", "return -2 * self.log_likelihood_ + ((self.K - 1) + self.K * (self.K - 1) + 2 * self.K * self.F + 2 * self.K) * np.log(max(n_obs, 1))"),
    ("C07 AIC ignores the convention", M, "return -2 * self.log_likelihood_ + 2 * self.n_params()", "return -2 * self.log_likelihood_ + 2 * ((self.K - 1) + self.K * (self.K - 1) + 2 * self.K * self.F + 2 * self.K)"),
    ("C08 legacy and free swapped", M, 'if self.n_params_convention == "legacy":', 'if self.n_params_convention == "free":'),
    ("C09 variances not counted", M, "return (K - 1) + n_transition + K * F + K * F + K * 2", "return (K - 1) + n_transition + K * F + K * 2"),
    ("D01 doc says the default is legacy", "docs/INTEGRATION_CONTRACT_HSSM.md", "the default is the free count", "the default is the legacy count"),
    ("D02 doc drops the over-count size", "docs/INTEGRATION_CONTRACT_HSSM.md", "too large by **K**", "too large by **2K**"),
    ("D03 doc drops the owner decision", "docs/INTEGRATION_CONTRACT_HSSM.md", "owner decision B1", "an engineering choice"),
    ("C10 initial distribution counted as K", M, "return (K - 1) + n_transition + K * F + K * F + K * 2", "return K + n_transition + K * F + K * F + K * 2"),
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
                r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_wp7r_item2_bic_count.py", "-q", "-x", "-p", "no:cacheprovider"], cwd=dst,
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
