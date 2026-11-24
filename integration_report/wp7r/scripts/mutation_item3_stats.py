"""Mutation check for tests/test_wp7r_item3_risk_coverage.py (WP7-R item 3). usage: PYTHONPATH=. python3 integration_report/wp7r/scripts/mutation_item3_stats.py [out.txt]
Each mutant edits ONE thing in tests/utils/item3_stats.py in a throw-away copy of the repo and the tests must FAIL. Prints one line per mutant; exit 0 if all are killed."""
import shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
M = "tests/utils/item3_stats.py"
MUTANTS = [
    ("S01 judged threshold 8 -> 7", M, "MIN_PUBLISHABLE = 8 ", "MIN_PUBLISHABLE = 7 "),
    ("S02 judged threshold 8 -> 9", M, "MIN_PUBLISHABLE = 8 ", "MIN_PUBLISHABLE = 9 "),
    ("S03 judged uses >", M, 'judged=len(use) >= min_n', 'judged=len(use) > min_n'),
    ("S04 subset ignores the publishable flag", M, "pub = [s for s in seeds if publishable.get(s, False)]", "pub = list(seeds)"),
    ("S05 non-finite publishable not excluded", M, "use = [s for s in pub if _finite(scores[s]) and _finite(ref[s])]", "use = list(pub)"),
    ("S06 non-finite reference not excluded", M, "use = [s for s in pub if _finite(scores[s]) and _finite(ref[s])]", "use = [s for s in pub if _finite(scores[s])]"),
    ("S07 non-finite publishable not counted", M, "n_nonfinite = len(pub) - len(use)", "n_nonfinite = 0"),
    ("S08 paired difference sign reversed", M, "d = np.array([float(scores[s]) - float(ref[s]) for s in use])", "d = np.array([float(ref[s]) - float(scores[s]) for s in use])"),
    ("S09 margin multiplier 3 -> 2", M, "mean, se, lower = paired_margin(d, MARGIN_SE)", "mean, se, lower = paired_margin(d, 2.0)"),
    ("S10 coverage counts used not publishable", M, "coverage=(len(pub) / len(seeds)) if seeds else float(\"nan\"), judged", "coverage=(len(use) / len(seeds)) if seeds else float(\"nan\"), judged"),
    ("S11 fallback skips withheld seeds (drops them) instead of 0", M, "            d.append(0.0)\n", "            pass\n"),
    ("S12 fallback serves withheld fits", M, "if publishable.get(s, False) and _finite(scores[s]):", "if _finite(scores[s]):"),
    ("S13 fallback serves non-finite publishable as zero-score", M, "if publishable.get(s, False) and _finite(scores[s]):", "if publishable.get(s, False):"),
    ("S14 fallback keeps seeds with a non-finite reference", M, "seeds = sorted(s for s in set(scores) & set(ref) if _finite(ref[s]))", "seeds = sorted(set(scores) & set(ref))"),
    ("S15 fallback n_served counts every seed", M, "n_served += 1", "n_served += 0"),
    ("S16 survives uses >= 0", M, 'stats_by_length[L]["mean_minus_3se"] > 0)', 'stats_by_length[L]["mean_minus_3se"] >= 0)'),
    ("S17 survives lets unjudged lengths veto", M, "judged = [L for L, st in stats_by_length.items() if st[\"judged\"]]", "judged = [L for L, st in stats_by_length.items()]"),
    ("S18 survives drops the partly-judged word", M, 'return "survives" if len(judged) == len(stats_by_length) else "partly judged"', 'return "survives"'),
    ("S19 tally counts publishable fits as withheld states", M, 'by_state = Counter(str(r["fit_state"]) for r in records if r["withheld"])', 'by_state = Counter(str(r["fit_state"]) for r in records)'),
    ("S20 tally rate uses publishable", M, "withheld_rate=(n_w / n) if n else float(\"nan\")", "withheld_rate=((n - n_w) / n) if n else float(\"nan\")"),
    ("S21 prefix regex accepts colons inside text", M, 'r"(?:^|; )([a-z_]+):"', 'r"([a-z_]+):"'),
    ("S22 numerical failure not recognised", M, 'if "fit_state is NUMERICAL_FAILURE" in reason:', 'if False:'),
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
                r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_wp7r_item3_risk_coverage.py", "-q", "-x", "-p", "no:cacheprovider"], cwd=dst,
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
