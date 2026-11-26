"""A1 diagnosis analysis (plan: tests/PREDECLARED_CRITERIA_WP9.md section 4). Descriptive; chooses nothing by itself.
usage: PYTHONPATH=. python3 integration_report/wp9/scripts/a1_diag_analyze.py wp9_evidence/a1_diag_fits.jsonl wp9_evidence/a1_diag_analysis.txt"""
import sys
from collections import defaultdict
sys.path.insert(0, "integration_report/wp9/scripts")
import numpy as np
import wp9_fit_lib as L

REC_ARI = 0.6                       # ARI to the planted labels at or above which a partition "recovers" the planted regimes (WP8 convention)
SAME = 0.99
NS = (4, 10, 20)


def pairs(records):
    by = defaultdict(dict)
    for r in records:
        if not r["ok"]:
            continue
        j = r["job"]
        by[(j["user"], j["end"])][j["fitseed"] != j["user"]] = r
    return {k: (v[False], v[True]) for k, v in by.items() if len(v) == 2}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.1f}% ({100 * lo:.0f}-{100 * hi:.0f})" if n else "0/0"


def main(fits, out, quiet=False):
    recs = L.load_records(fits)
    P = pairs(recs)
    lines = [f"A1 diagnosis: {len(recs)} fits ({sum(r['ok'] for r in recs)} ok), {len(P)} (window, seed-pair) cases; strata hard 9201-9204, medium 9205-9208, easy 9209-9212; windows 90 and 150; restarts = the first n of 20"]
    strata = {"hard": range(9201, 9205), "medium": range(9205, 9209), "easy": range(9209, 9213)}
    rows = {}
    for n in NS:
        for (u, end), (a, b) in sorted(P.items()):
            sa, sb = L.select_restart(a["restarts"], n), L.select_restart(b["restarts"], n)
            row = dict(user=u, end=end, n=n, failed=(sa is None or sb is None))
            if not row["failed"]:
                ari = L.partition_ari(sa["labels"], sb["labels"])
                ta, tb = (L.partition_ari(s["labels"], a["labels_true"]) for s in (sa, sb))
                ta, tb = L.partition_ari(sa["labels"], a["labels_true"]), L.partition_ari(sb["labels"], b["labels_true"])
                row.update(ari=ari, gap=abs(sa["ll"] - sb["ll"]), ta=ta, tb=tb, rec_a=ta >= REC_ARI, rec_b=tb >= REC_ARI,
                           ca=L.consensus_ari(a["restarts"], n), cb=L.consensus_ari(b["restarts"], n),
                           distinct_a=len({L.labels_to_str(L.str_to_labels(r["labels"])) for r in L.top_restarts(a["restarts"], n)}))
                if row["ari"] >= SAME:
                    row["cls"] = "same"
                elif not row["rec_a"] and not row["rec_b"]:
                    row["cls"] = "recovery failure (neither partition recovers the planted regimes)"
                elif row["gap"] > 1.0:
                    row["cls"] = "local optima (log-likelihoods differ by more than 1 nat)"
                else:
                    row["cls"] = "equal likelihood, different partition (gap <= 1 nat)"
            rows[(u, end, n)] = row
    for n in NS:
        lines += ["", f"== {n} restarts =="]
        for name, users in list(strata.items()) + [("all", range(9201, 9213))]:
            rs = [r for (u, e, nn), r in rows.items() if nn == n and u in users]
            ok = [r for r in rs if not r["failed"]]
            same = sum(1 for r in ok if r["ari"] >= SAME)
            ge9 = sum(1 for r in ok if r["ari"] >= 0.9)
            lines.append(f"{name:7s} cases {len(rs)}, fit failed in a seed {len(rs) - len(ok)}: same partition (ARI >= 0.99) {pct(same, len(ok))}; ARI >= 0.9 {pct(ge9, len(ok))}; median ARI {np.median([r['ari'] for r in ok]):.3f}, min {min(r['ari'] for r in ok):.3f}")
        ok = [r for (u, e, nn), r in rows.items() if nn == n and not r["failed"]]
        dis = [r for r in ok if r["ari"] < SAME]
        c = defaultdict(int)
        for r in dis:
            c[r["cls"]] += 1
        lines.append(f"disagreeing cases {len(dis)} of {len(ok)}: " + "; ".join(f"{k}: {v}" for k, v in sorted(c.items())))
        rec_both = sum(1 for r in ok if r["rec_a"] and r["rec_b"])
        lines.append(f"both seeds recover the planted regimes (ARI to truth >= {REC_ARI}) {pct(rec_both, len(ok))}; among those, same partition {pct(sum(1 for r in ok if r['rec_a'] and r['rec_b'] and r['ari'] >= SAME), rec_both)}")
    lines += ["", "== does agreement of the best 3 restarts inside one fit predict that the other seed gives the same partition? (20 restarts) =="]
    ok = [r for (u, e, nn), r in rows.items() if nn == 20 and not r["failed"]]
    for lvl in (0.8, 0.9, 0.95, 0.99):
        sel = [r for r in ok if r["ca"] is not None and r["cb"] is not None and r["ca"] >= lvl and r["cb"] >= lvl]
        lines.append(f"both fits' best-3 minimum ARI >= {lvl}: {len(sel)} of {len(ok)} cases kept; of those same partition {pct(sum(1 for r in sel if r['ari'] >= SAME), len(sel))}, both recover {pct(sum(1 for r in sel if r['rec_a'] and r['rec_b']), len(sel))}")
    lines += ["", "== the same, with 10 and 4 restarts (best 3 of the first n) =="]
    for n in (10, 4):
        okn = [r for (u, e, nn), r in rows.items() if nn == n and not r["failed"]]
        for lvl in (0.9, 0.99):
            sel = [r for r in okn if r["ca"] is not None and r["cb"] is not None and r["ca"] >= lvl and r["cb"] >= lvl]
            lines.append(f"n = {n}, level {lvl}: {len(sel)} of {len(okn)} kept; same partition {pct(sum(1 for r in sel if r['ari'] >= SAME), len(sel))}")
    lines += ["", "== per case, 20 restarts: user end | ARI(sel a, sel b) | ll gap | ARI to truth a, b | best-3 min ARI a, b | class =="]
    for (u, e, n), r in sorted(rows.items()):
        if n == 20 and not r["failed"]:
            f = lambda x: "  -- " if x is None else f"{x:.3f}"
            lines.append(f"{u} {e:3d} | {r['ari']:.3f} | {r['gap']:8.2f} | {r['ta']:.3f} {r['tb']:.3f} | {f(r['ca'])} {f(r['cb'])} | {r['cls']}")
    open(out, "w").write("\n".join(lines) + "\n")
    if not quiet:
        print("\n".join(lines))




# ------------------------------------------------------------------------------------------------- candidate consensus rules (descriptive; Part 2 of the pre-declaration chooses among them)
def n_same(restarts, n, sel, level=0.99):
    """Healthy (converged, non-collapsed) restarts among the first n whose partition has ARI >= level with the selected partition (the selected one counts itself)."""
    return sum(1 for r in L.top_restarts(restarts, n) if L.partition_ari(sel["labels"], r["labels"]) >= level)


def rule_pass(restarts, n, rule):
    sel = L.select_restart(restarts, n)
    if sel is None or sel["collapsed"]:
        return False
    kind, par = rule
    if kind == "top3":                                   # the best 3 healthy restarts by log-likelihood agree pairwise (min ARI >= par)
        c = L.consensus_ari(restarts, n, 3)
        return c is not None and c >= par
    if kind == "top2":
        c = L.consensus_ari(restarts, n, 2)
        return c is not None and c >= par
    if kind == "reached":                                # the selected partition is reached (ARI >= 0.99) by at least `par` healthy restarts
        return n_same(restarts, n, sel) >= par
    raise ValueError(kind)


def rules_report(records):
    P = pairs(records)
    out = ["", "== candidate rules (a fit PASSES the rule when it is not flagged); pairs = (window, two seeds) =="]
    rules = [("top3", 0.9), ("top3", 0.95), ("top3", 0.99), ("top2", 0.99), ("reached", 2), ("reached", 3), ("reached", 4)]
    for n in (10, 20):
        for rule in rules:
            fits = [(r) for (a, b) in P.values() for r in (a, b)]
            passed = [rule_pass(r["restarts"], n, rule) for r in fits]
            both = [(rule_pass(a["restarts"], n, rule) and rule_pass(b["restarts"], n, rule), a, b) for a, b in P.values()]
            kept = [(a, b) for ok, a, b in both if ok]
            same = sum(1 for a, b in kept if L.partition_ari(L.select_restart(a["restarts"], n)["labels"], L.select_restart(b["restarts"], n)["labels"]) >= SAME)
            one = [(a, b) for a, b in P.values() if rule_pass(a["restarts"], n, rule) != rule_pass(b["restarts"], n, rule)]
            by = {}
            for name, users in (("hard", range(9201, 9205)), ("medium", range(9205, 9209)), ("easy", range(9209, 9213))):
                fs = [r for r in fits if r["job"]["user"] in users]
                by[name] = sum(rule_pass(r["restarts"], n, rule) for r in fs)
                by[name + "_n"] = len(fs)
            out.append(f"n={n:2d} {rule[0]:7s} {rule[1]:<5}: fits passing {sum(passed)}/{len(passed)} (hard {by['hard']}/{by['hard_n']}, medium {by['medium']}/{by['medium_n']}, easy {by['easy']}/{by['easy_n']}); "
                       f"pairs with both passing {len(kept)}/{len(both)}: same partition {pct(same, len(kept))}; pairs where exactly one seed is flagged {len(one)}")
    return out


def per_fit_n10(records):
    P = pairs(records)
    out = ["", "== per case, 10 restarts: user end | ARI between seeds | best-3 min ARI a, b | restarts reaching the selected partition a, b | recovery (ARI to truth) a, b =="]
    for (u, e), (a, b) in sorted(P.items()):
        sa, sb = L.select_restart(a["restarts"], 10), L.select_restart(b["restarts"], 10)
        ca, cb = L.consensus_ari(a["restarts"], 10), L.consensus_ari(b["restarts"], 10)
        f = lambda x: " -- " if x is None else f"{x:.3f}"
        out.append(f"{u} {e:3d} | {L.partition_ari(sa['labels'], sb['labels']):.3f} | {f(ca)} {f(cb)} | {n_same(a['restarts'], 10, sa)} {n_same(b['restarts'], 10, sb)} | {L.partition_ari(sa['labels'], a['labels_true']):.3f} {L.partition_ari(sb['labels'], b['labels_true']):.3f}")
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], quiet=True)
    recs = L.load_records(sys.argv[1])
    extra = rules_report(recs) + per_fit_n10(recs)
    open(sys.argv[2], "a").write("\n".join(extra) + "\n")
    print("\n".join(extra))
