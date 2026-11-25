"""POST-HOC breakdown of Item 8 criterion 3 (seed vs seed). Written AFTER the criterion 3 result (166/240 identical, NOT MET) was seen; it changes no verdict and no frozen setting.
It only splits the 74 non-identical cases by what differs: the state (ALIGNED vs not) or the stable ids when both seeds ended in the same state.
usage: PYTHONPATH=. python3 integration_report/wp8/scripts/item8_posthoc_crit3.py wp8_evidence/item8_fits.jsonl wp8_evidence/item8_posthoc_crit3.txt"""
import sys
from collections import Counter
sys.path.insert(0, "integration_report/wp8/scripts")
import numpy as np
import item8_analyze as A
import lineage_study_lib as S
from backbone.hssm.lineage import AlignmentConfig

fits, out = sys.argv[1], sys.argv[2]
g = A.load_groups(fits, False)
cfg = AlignmentConfig()
ra = A.replay_all(g["B-G"], cfg, key=lambda j: j["user"])
rb = A.replay_all(g["B-G2"], cfg, key=lambda j: j["user"])
c = Counter(); lines = []
for u in sorted(set(ra) | set(rb)):
    rows_a, lrs_a, _ = ra.get(u, ([], [], None)); rows_b, lrs_b, _ = rb.get(u, ([], [], None))
    da = {r["window"][1]: (r["state"], lr.stable_regime_id) for r, lr in zip(rows_a, lrs_a)}
    db = {r["window"][1]: (r["state"], lr.stable_regime_id) for r, lr in zip(rows_b, lrs_b)}
    for end in sorted(set(da) | set(db)):
        c["cases"] += 1
        if end not in da or end not in db:
            c["fed by only one seed"] += 1; continue
        sa, sb = da[end][0], db[end][0]
        agree = S.day_agreement(da[end][1], db[end][1])
        if sa == sb and agree >= 0.99:
            c["identical"] += 1
        elif sa != sb:
            c[f"states differ: {'/'.join(sorted((sa, sb)))}"] += 1
        else:
            c[f"same state {sa}, stable ids differ (agreement < 0.99)"] += 1
            if sa == "ALIGNED":
                lines.append(f"   both ALIGNED but stable ids differ: user {u} end {end} agreement {agree:.3f}")
txt = ["POST-HOC breakdown of criterion 3 (seed vs seed + 1000), written after the criterion 3 result was seen; changes no verdict."]
txt += [f"{k}: {v}" for k, v in sorted(c.items())] + lines
open(out, "w").write("\n".join(txt) + "\n")
print("\n".join(txt))
