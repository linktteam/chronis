"""WP9 B2 analysis (declared reading: tests/PREDECLARED_CRITERIA_WP9.md section 15). usage: PYTHONPATH=. python3 integration_report/wp9/scripts/b2_analyze.py FITS.jsonl OUT.txt [OUT.json]"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp8" / "scripts"))
import lineage_study_lib as S                                                # noqa: E402


def verdict(pub):
    """pub = {f: (publishable, n)} for f in 0, 0.5, 1.0. The declared reading."""
    a = pub[0.0][0] >= 9 and pub[0.0][1] == 12
    b = pub[1.0][0] <= 6 and pub[1.0][1] == 12
    mono = pub[0.0][0] >= pub[0.5][0] >= pub[1.0][0]
    if a and b and mono:
        return "explained by the data design"
    if a != b:
        return "partly explained by the data design"
    return "not explained by the data design"


def main(path, out_txt, out_json=None):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    L = [f"B2: {len(recs)} shrink5 fits, seeds {min(r['seed'] for r in recs)}..{max(r['seed'] for r in recs)}, 4 training segments, 200 iterations, final engine", ""]
    pub = {}
    for f in (0.0, 0.5, 1.0):
        rs = [r for r in recs if r["spread"] == f]
        k = sum(not r["withheld"] for r in rs)
        pub[f] = (k, len(rs))
        lo, hi = S.wilson(k, len(rs))
        st = Counter(r["fit_state"] for r in rs)
        reasons = Counter((r["withheld_reason"] or "").split(":")[0] for r in rs if r["withheld"])
        its = sorted(r["n_iter_run"] for r in rs)
        L.append(f"spread factor {f}: publishable {k}/{len(rs)} = {100 * k / max(len(rs), 1):.0f}% (Wilson {100 * lo:.0f}-{100 * hi:.0f}%); EM states {dict(st)}; withheld for {dict(reasons)}; iterations median {its[len(its) // 2] if its else None} (min {its[0] if its else None}, max {its[-1] if its else None}); T mean {sum(r['T_train'] for r in rs) / max(len(rs), 1):.0f}")
    chk = [r["centre_equals_item4_generator"] for r in recs if r["spread"] == 0.0]
    L += ["", f"f = 0 data equal the item 4 generator exactly in all {len(chk)} fits: {all(chk)}", f"declared reading -> {verdict(pub)}"]
    open(out_txt, "w").write("\n".join(L) + "\n")
    if out_json:
        json.dump(dict(publishable={str(k): v for k, v in pub.items()}, verdict=verdict(pub)), open(out_json, "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:])
