"""WP9 A1/A2 (REPORTED, not declared in advance, nothing is chosen from it): the same split of the identical-lineage cases as `a12_eval_identity_split.py`, but for the consensus flag ON (with the sharpness floor 0.95) and OFF (no
stability flag of either kind), both at the frozen gate h = 2.5, so that the owner can see what the flag buys in seed-to-seed agreement and what it costs in coverage. usage:
  PYTHONPATH=. python3 integration_report/wp9/scripts/a12_eval_flagoff_identity.py wp9_evidence/a12_eval_fits.jsonl OUT.txt"""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import a12_analyze as A
import wp9_fit_lib as F
import wp9_replay_lib as R
import lineage_study_lib as S
from backbone.hssm.lineage import AlignmentConfig

recs = F.load_records(sys.argv[1]); a, b = A.split_chains(recs); strata = A.stratum_users(a)
cfg = AlignmentConfig()
L = ["A1 seed-to-seed agreement with the consensus flag ON (sharpness floor 0.95) versus OFF (no stability flag), frozen gate h = 2.5, evaluation users 9401..9448; REPORTED, nothing was chosen from it",
     "strict case = a 14-day window where at least one seed ends ALIGNED; 'same' = both ALIGNED with >= 99% of days named identically; 'different' = both ALIGNED, names differ; 'one seed' = the other seed says 'cannot confirm'"]
for n in (10, 20):
    for u, s in ((True, 0.95), (False, 0.0)):
        for st in ("pooled",) + tuple(A.STRATA):
            users = None if st == "pooled" else strata[st]
            same = differ = one = 0
            for key_a, ca in a.items():
                uu = key_a[0]
                if users is not None and uu not in users:
                    continue
                cb = next((v for (ub, _), v in b.items() if ub == uu), None)
                ra, la, _, _ = R.replay9([r for r in ca if r["job"]["end"] in A.B_T], cfg, n=n, use_flag=u, sharpness=s)
                rb, lb, _, _ = R.replay9([r for r in cb if r["job"]["end"] in A.B_T], cfg, n=n, use_flag=u, sharpness=s)
                da = {r["end"]: (r, lr) for r, lr in zip(ra, la)}; db = {r["end"]: (r, lr) for r, lr in zip(rb, lb)}
                for end in sorted(set(da) | set(db)):
                    x, y = da.get(end), db.get(end)
                    sa = x is not None and x[0]["state"] == "ALIGNED"; sb = y is not None and y[0]["state"] == "ALIGNED"
                    if sa and sb:
                        if S.day_agreement(x[1].stable_regime_id, y[1].stable_regime_id) >= 0.99: same += 1
                        else: differ += 1
                    elif sa or sb:
                        one += 1
            tot = same + differ + one
            L.append(f"first {n} restarts, flag {'ON ' if u else 'OFF'}, {st:6s}: strict cases {tot}; same names {same} ({100*same/max(tot,1):.1f}%); both served with DIFFERENT names {differ}; one seed only {one}")
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
