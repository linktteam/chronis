"""WP8 Item 4: per-alignment margin analysis of the DEVELOPMENT alignments (seeds 8201..8212), read from `wp8_evidence/item4_alignments.jsonl` (written by item4_analyze.py).
Answers: at which margin do alignments start to be wrong? Wrong = strict-wrong (the declared criterion: a fit that does not recover the planted regimes makes the alignment wrong) and,
next to it, permutation-wrong (both fits recoverable and the matching is not the planted one). Raw distance (the frozen one). No new fit is made here.
usage: python3 integration_report/wp8/scripts/item4_margin_report.py wp8_evidence/item4_alignments.jsonl wp8_evidence/item4_margin_analysis.txt [h]"""
import json
import sys

ali_path, out_path = sys.argv[1:3]
H = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
rows = [json.loads(l) for l in open(ali_path)]
L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    L.append(s)
    print(s)


def pct(a, b):
    return f"{a}/{b} = {100.0 * a / b:.1f}%" if b else "0/0"


def table(rr, title, edges_ratio=(0.0, 0.01, 0.03, 0.1, 0.2, 0.35, 0.5, 1.01)):
    P(f"\n== {title}: {len(rr)} alignments")
    P("  margin_ratio bucket     n    strict-correct    both-recoverable  permutation-wrong")
    for lo, hi in zip(edges_ratio[:-1], edges_ratio[1:]):
        b = [r for r in rr if lo <= r["modes"]["raw"]["margin_ratio"] < hi]
        both = [r for r in b if r["anchor_recoverable"] and r["new_recoverable"]]
        P(f"  [{lo:4.2f}, {hi:4.2f})  {len(b):6d}   {pct(sum(r['modes']['raw']['correct'] for r in b), len(b)):>18}   {len(both):6d}          {sum(not r['modes']['raw']['correct'] for r in both):4d}")
    wrong = [r for r in rr if not r["modes"]["raw"]["correct"]]
    pw = [r for r in rr if r["anchor_recoverable"] and r["new_recoverable"] and not r["modes"]["raw"]["correct"]]
    if wrong:
        P(f"  strict-wrong: {len(wrong)}; smallest margin_ratio among them {min(r['modes']['raw']['margin_ratio'] for r in wrong):.4f}; largest margin_diff among them {max(r['modes']['raw']['margin_diff'] for r in wrong):.4f}")
        P("  that is: every alignment with margin_ratio below the first number and margin_diff above the second was strict-correct; strict-wrong alignments exist at and above those values")
    if pw:
        P(f"  permutation-wrong: {len(pw)}; smallest margin_ratio {min(r['modes']['raw']['margin_ratio'] for r in pw):.4f}; largest margin_diff {max(r['modes']['raw']['margin_diff'] for r in pw):.4f}")
        for r in pw:
            m = r["modes"]["raw"]
            P(f"    user {r['user']} drift {r['drift']} {r['proto']} {r['anchor_window']}->{r['new_window']}: ratio {m['margin_ratio']:.3f} diff {m['margin_diff']:.3f} sep {r['sep_anchor']:.2f}/{r['sep_new']:.2f}")
    else:
        P("  permutation-wrong: none")


P(f"WP8 Item 4 margin analysis on development alignments (raw distance), gate h = {H}")
for cls in ("baseline", "dual", "cross"):
    rr = [r for r in rows if r["cls"] == cls]
    table(rr, f"class {cls}, all alignments")
    table([r for r in rr if r["sep_anchor"] >= H and r["sep_new"] >= H], f"class {cls}, alignments passing the gate (both fits' separation >= {H})")
P("\n== baseline by drift, alignments passing the gate: strict-wrong and margins")
for d in sorted({r["drift"] for r in rows if r["cls"] == "baseline"}):
    rr = [r for r in rows if r["cls"] == "baseline" and r["drift"] == d and r["sep_anchor"] >= H and r["sep_new"] >= H]
    mr = sorted(r["modes"]["raw"]["margin_ratio"] for r in rr)
    P(f"  drift {d}: {len(rr)} passing; strict-wrong {sum(not r['modes']['raw']['correct'] for r in rr)}; margin_ratio median {mr[len(mr) // 2]:.4f}, max {mr[-1]:.4f}" if mr else f"  drift {d}: none passing")
open(out_path, "w").write("\n".join(L) + "\n")
