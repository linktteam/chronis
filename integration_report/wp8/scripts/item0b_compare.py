"""WP8 item 0b: compare the K-selection study rerun with the free-count BIC default against the stored WP7-R run (legacy count).
usage: python3 integration_report/wp8/scripts/item0b_compare.py wp7r_evidence/item2_kselect.jsonl wp8_evidence/item0b_kselect_free_default.jsonl [out.txt]
Reports: datasets compared; selected K identical (count); log-likelihoods per K bit-identical (the count must not touch the EM fit); BIC difference per K equals K*ln(n_obs)
(the legacy count exceeds the free count by K); planted-K recovery under each default; every dataset whose selected K changed."""
import json, math, sys


def load(p):
    return {(r["design"], r["seed"]): r for r in map(json.loads, open(p)) if r.get("ok")}


old, new = load(sys.argv[1]), load(sys.argv[2])
keys = sorted(set(old) & set(new))
lines = [f"datasets in both runs: {len(keys)} (old {len(old)}, new {len(new)})"]
same_k = sum(old[k]["selected_k"] == new[k]["selected_k"] for k in keys)
same_ll = sum(old[k]["loglik_by_k"] == new[k]["loglik_by_k"] for k in keys)
bic_ok = 0
worst = 0.0
for k in keys:
    n = old[k]["n_obs"]; good = True
    for K, b in old[k]["bic_by_k"].items():
        d = b - new[k]["bic_by_k"][K]
        worst = max(worst, abs(d - int(K) * math.log(n)))
        good &= abs(d - int(K) * math.log(n)) < 1e-6
    bic_ok += good
lines += [f"selected K identical: {same_k} of {len(keys)}", f"log-likelihoods per K bit-identical: {same_ll} of {len(keys)}",
          f"BIC(old) - BIC(new) == K*ln(n) for every K: {bic_ok} of {len(keys)} (largest deviation {worst:.2e})"]
for d in ("easy", "hard"):
    ks = [k for k in keys if k[0] == d]
    rec_o = sum(old[k]["selected_k"] == old[k]["planted_k"] for k in ks); rec_n = sum(new[k]["selected_k"] == new[k]["planted_k"] for k in ks)
    lines.append(f"{d}: planted-K recovery legacy {rec_o}/{len(ks)}, free {rec_n}/{len(ks)}")
for k in keys:
    if old[k]["selected_k"] != new[k]["selected_k"]:
        lines.append(f"CHANGED {k}: planted {old[k]['planted_k']} legacy {old[k]['selected_k']} free {new[k]['selected_k']}")
out = "\n".join(lines)
print(out)
if len(sys.argv) > 3:
    open(sys.argv[3], "w").write(out + "\n")
