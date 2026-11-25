"""Item 8 (reporting only): the ordering-ladder rung of each benchmark user's FIRST fit (window 0..90, baseline, seeds 8301..8324), whether it was verified, and whether the order equals the
planted order. Read from the committed evaluation fits through the production `step`; no new fit. usage: PYTHONPATH=. python3 integration_report/wp8/scripts/item8_rung_per_user.py wp8_evidence/item8_fits.jsonl wp8_evidence/item8_rung_per_user.txt"""
import sys
sys.path.insert(0, "integration_report/wp8/scripts")
import item8_analyze as A
import lineage_study_lib as S
from backbone.hssm.lineage import AlignmentConfig

fits, out = sys.argv[1:3]
g = A.load_groups(fits, False)
res = A.replay_all(g["B-G"], AlignmentConfig(), key=lambda j: j["user"])
streams = A.streams(g["B-G"], key=lambda j: j["user"])
lines = ["user   first-fit rung  verified  order equals planted order   (rung 0 = could not be ordered reliably; first-fit order kept, no activity claim)"]
for u in sorted(res):
    rows, lrs, _ = res[u]
    lr0 = lrs[0]
    fed = [r for r in streams[u] if S.usable(r) and r.get("accepted")]
    info = S.fit_info(fed[0])
    ok = info.recoverable and all(info.mp[raw] == stable for raw, stable in enumerate(lr0.raw_to_stable))
    lines.append(f"{u}   {lr0.activity_rung}               {str(lr0.ordering_verified):8s}  {('yes' if ok else 'NO') if lr0.ordering_verified else '(not claimed)'}")
open(out, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
