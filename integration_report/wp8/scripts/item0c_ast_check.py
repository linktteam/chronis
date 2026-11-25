"""WP8 item 0c: the D12 edit is comment-only. Compares ast.dump of tests/test_dynamics_statistics.py at the base commit and in the working tree, and shows the comment diff.
usage: python3 integration_report/wp8/scripts/item0c_ast_check.py [base=27c591c]"""
import ast, difflib, subprocess, sys
base = sys.argv[1] if len(sys.argv) > 1 else "27c591c"
path = "tests/test_dynamics_statistics.py"
old = subprocess.run(["git", "show", f"{base}:{path}"], capture_output=True, text=True, check=True).stdout
new = open(path).read()
same = ast.dump(ast.parse(old)) == ast.dump(ast.parse(new))
print(f"ast.dump identical between {base} and working tree: {same}")
for l in difflib.unified_diff(old.splitlines(), new.splitlines(), "base", "now", lineterm="", n=0):
    print(l)
sys.exit(0 if same else 1)
