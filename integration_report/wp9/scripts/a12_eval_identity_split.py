"""WP9 A1 (reported): of the strict cases of the identical-lineage criterion (a 14-day window at which at least one seed ends ALIGNED), how many are (i) served by both seeds with the same names (>= 99% of days), (ii) served by both with
DIFFERENT names, (iii) served by one seed only (the other says 'cannot confirm'). usage: PYTHONPATH=. python3 integration_report/wp9/scripts/a12_eval_identity_split.py wp9_evidence/a12_eval_fits.jsonl OUT.txt"""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import a12_analyze as A
import wp9_fit_lib as F
import wp9_replay_lib as R
import lineage_study_lib as S
from backbone.hssm.lineage import AlignmentConfig

recs = F.load_records(sys.argv[1]); a, b = A.split_chains(recs); strata = A.stratum_users(a)
cfg = AlignmentConfig()
L = ["A1 identical-lineage cases split by what happened (evaluation, frozen gate, flag on); strict case = a 14-day window where at least one seed ends ALIGNED"]
for n in (10, 20):
    tot = {}
    for st in ("pooled",) + tuple(A.STRATA):
        users = None if st == "pooled" else strata[st]
        same = differ = one = 0
        for key_a, ca in a.items():
            u = key_a[0]
            if users is not None and u not in users:
                continue
            cb = next((v for (ub, _), v in b.items() if ub == u), None)
            ra, la, _, _ = R.replay9([r for r in ca if r["job"]["end"] in A.B_T], cfg, n=n)
            rb, lb, _, _ = R.replay9([r for r in cb if r["job"]["end"] in A.B_T], cfg, n=n)
            da = {r["end"]: (r, lr) for r, lr in zip(ra, la)}; db = {r["end"]: (r, lr) for r, lr in zip(rb, lb)}
            for end in sorted(set(da) | set(db)):
                x, y = da.get(end), db.get(end)
                sa = x is not None and x[0]["state"] == "ALIGNED"; sb = y is not None and y[0]["state"] == "ALIGNED"
                if sa and sb:
                    if S.day_agreement(x[1].stable_regime_id, y[1].stable_regime_id) >= 0.99: same += 1
                    else: differ += 1
                elif sa or sb:
                    one += 1
        L.append(f"first {n} restarts, {st}: strict cases {same + differ + one}: served by both with the same names {same}; served by both with DIFFERENT names {differ}; served by one seed only {one}")
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
