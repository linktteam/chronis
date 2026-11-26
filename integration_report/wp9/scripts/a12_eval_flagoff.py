"""WP9 A2 (reported, not used to choose): the evaluation curve WITHOUT the consensus flag next to the curve with it, on the stored evaluation fits (declared in section 8: candidates without the flag are reported to show what the flag contributes). usage:
  PYTHONPATH=. python3 integration_report/wp9/scripts/a12_eval_flagoff.py wp9_evidence/a12_eval_fits.jsonl OUT.txt"""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import a12_analyze as A
import wp9_fit_lib as F

recs = F.load_records(sys.argv[1])
a, b = A.split_chains(recs)
strata = A.stratum_users(a)
L = ["A2 evaluation (768 fits, weekly windows, seed A chains), consensus flag ON versus OFF, s = 0.95 / 0; REPORTED, nothing was chosen from it"]
wk = {k: [r for r in v if r["job"]["end"] in A.G_T] for k, v in a.items()}
for n in (10, 20):
    L.append(f"\nfirst {n} restarts")
    for u in (True, False):
        for h in (0.0, 2.0, 2.5, 3.0, 4.0):
            res = A.run_candidate(wk, h, 0.95 if u else 0.0, u, n)
            parts = [f"pooled {A.fmt(A.tally_rows(res))}"] + [f"{st} {A.tally_rows(res, strata[st])['served']}/{A.tally_rows(res, strata[st])['refits']} w{A.tally_rows(res, strata[st])['wrong_unflagged']}" for st in A.STRATA]
            L.append(f"  flag {'on ' if u else 'off'} h = {h}: " + " | ".join(parts))
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
