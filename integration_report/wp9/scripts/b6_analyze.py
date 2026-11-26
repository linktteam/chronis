"""WP9 B6: re-certification rates and the no-relaxation check (levels of tests/PREDECLARED_CRITERIA_WP9.md section 25). usage:
  PYTHONPATH=. python3 integration_report/wp9/scripts/b6_analyze.py wp9_evidence/b6_sim.jsonl OUT.txt"""
import json, sys

recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
recs = [r for r in recs if r.get("ok")]
L = [f"B6 re-certification simulation: {len(recs)} users (seeds 9801..9812; hard 9801..9806, medium 9807..9812), production fit_hssm, candidates (2, 3, 4), 10 restarts, 100 iterations"]
w30 = [r for r in recs if r["day30"]["withheld"]]
L.append(f"fits withheld at day 30 (the first prefix with >= 30 present days): {len(w30)}/{len(recs)}"
         + (" -> TOO FEW TO READ (declared: fewer than 4)" if len(w30) < 4 else ""))
for r in w30:
    L.append(f"  user {r['user']} ({r['stratum']}): withheld for {r['day30']['reasons']}")
if w30:
    for end in ("60", "90"):
        got = [r for r in w30 if r["recert"].get(end, {}).get("state") in ("PUBLISHABLE", "PUBLISHABLE_EARLIER")]
        L.append(f"publishable by day {end} after re-certification: {len(got)}/{len(w30)}")
    eq = [e["f1_equals_recert"] for r in w30 for e in r["recert"].values() if "f1_equals_recert" in e]
    L.append(f"fresh fit on the same matrix agrees with the re-certification: {sum(eq)}/{len(eq)}")
pub30 = sum(1 for r in recs if not r["day30"]["withheld"])
L.append(f"published at day 30: {pub30}/{len(recs)}; fresh window fits (own z-scoring) published at day 60: {sum(1 for r in recs if not r['fresh']['60']['withheld'])}/{len(recs)}, at day 90: {sum(1 for r in recs if not r['fresh']['90']['withheld'])}/{len(recs)}")
reasons = {}
for r in recs:
    for k in ("60", "90"):
        for x in r["fresh"][k]["reasons"]:
            reasons[x.split(":")[0]] = reasons.get(x.split(":")[0], 0) + 1
L.append(f"withhold reasons among the fresh fits at days 60 and 90: {reasons or 'none'}")
outs = sum(r["outputs_published"] for r in recs); viol = sum(r["violations"] for r in recs)
relaxed = sum(1 for r in recs for e in r["recert"].values() if e.get("rules_relaxed"))
L.append(f"\nNO-RELAXATION CHECK (hard level): outputs with a published state {outs}; violations (publishable by the module but refused by an independent recomputation, a result handed out with a withheld state, rules_relaxed true) {viol} (+ {relaxed} with rules_relaxed true) -> "
         + ("MET (0 violations)" if viol == 0 and relaxed == 0 else "NOT MET"))
L.append("The check is vacuous for the re-certification path when no fit is withheld at day 30: it then covers the day-30 and fresh fits only.")
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
