"""Plumbing smoke test for criterion 6 (K change) of item8_analyze.py: the development rehearsal fits never pick a K different from the anchor, so the branch
where K differs is exercised on FORGED records (a fourth regime added to the late split-stream fits). The numbers printed are NOT evidence of anything; the point is that
the code path runs and counts K_CHANGED steps. Run from the repo root: PYTHONPATH=. python3 integration_report/wp8/scripts/item8_crit6_smoke.py"""
import copy, json, sys, io, contextlib
sys.path.insert(0, "integration_report/wp8/scripts")
import item8_analyze as A
from identity_lib import load_records
recs = load_records("wp8_evidence/item8_rehearsal_fits.jsonl")
out = []
for r in recs:
    r = copy.deepcopy(r)
    if r["job"]["end"] >= 118 and r["job"]["kchange"] == "split":
        for k in ("z_means", "raw_means"):
            r[k].append([x + 2.0 * s for x, s in zip(r[k][0], r["scaler_std"])] if k == "raw_means" else [x + 2.0 for x in r[k][0]])
        r["exp_dur"] = list(r["exp_dur"]) + [r["exp_dur"][0]]
        r["K"] = 4
    out.append(r)
print("changed records", sum(r["K"] == 4 for r in out))
cfg = A.AlignmentConfig()
A.crit_6(out, cfg)
