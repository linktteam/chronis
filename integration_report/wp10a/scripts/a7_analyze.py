"""WP10A A7 analysis (levels: tests/PREDECLARED_CRITERIA_WP10A.md Part 5, section 14). usage:
  PYTHONPATH=. python3 integration_report/wp10a/scripts/a7_analyze.py OUT.txt OUT.json W1.jsonl W2.jsonl
Every level is recomputed here from the stored fields (not from anything the runner decided)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "wp9" / "scripts"))
import b5_analyze as B9                                                                  # noqa: E402  (wilson)

TARGET, MIN_WITHHELD = 12, 8
ALTERATIONS = ("fit_collapsed", "fit_degraded", "fit_duration_degenerate", "states_pass_fallbacks", "fit_state_not_converged")
PROBES = ("n_initializations_3", "unknown_allow_fast_test_fit", "unknown_config")
OUT: list = []


def P(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)


def load(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in recs if r.get("ok")], sum(1 for r in recs if not r.get("ok"))


def used_users(recs, target: int = TARGET):
    """The declared stopping rule applied in seed order: the users up to and including the one that makes the `target`-th withheld (all users when fewer are withheld). Returns (used, extra)."""
    recs = sorted(recs, key=lambda r: r["user"])
    used, k = [], 0
    for i, r in enumerate(recs):
        used.append(r)
        k += int(bool(r["day30"]["withheld"]))
        if k >= target:
            return used, recs[i + 1:]
    return used, []


def violations(r) -> list:
    """H2, from the stored fields: published states that an independent recomputation refuses, relaxed rules, results handed out wrongly, settings probes not refused."""
    v = []
    d30 = r["day30"]
    if not d30["withheld"] and not d30["independent_publishable"]:
        v.append("day-30 state published but refused by the independent recomputation")
    for end, e in r["recert"].items():
        if e["state"] == "PUBLISHABLE_EARLIER":
            continue
        if e.get("rules_relaxed") is not False:
            v.append(f"day {end}: rules_relaxed is not False")
        if e["state"] == "PUBLISHABLE":
            if not e.get("result_returned"):
                v.append(f"day {end}: PUBLISHABLE without a result")
            if e.get("result_withheld") is not False:
                v.append(f"day {end}: PUBLISHABLE with a result that is itself withheld")
            if e.get("independent_publishable") is not True:
                v.append(f"day {end}: PUBLISHABLE but refused by the independent recomputation")
        elif e.get("result_returned"):
            v.append(f"day {end}: a result handed out with the state {e['state']}")
    if d30["withheld"]:
        probes = {p["probe"]: p for p in (r.get("probes") or [])}
        for name in PROBES:
            p = probes.get(name)
            if p is None:
                v.append(f"probe {name} missing")
            elif p["state"] != "NOT_ELIGIBLE" or p["result_returned"] or p["rules_relaxed"] is not False:
                v.append(f"probe {name} not refused ({p['state']})")
    return v


def f1_disagreements(r) -> list:
    """H3: a fresh fit on the same matrix must be withheld iff the recertify state is not PUBLISHABLE."""
    bad = []
    for end, e in r["recert"].items():
        if "f1_withheld" in e and bool(e["f1_withheld"]) != (e["state"] != "PUBLISHABLE"):
            bad.append(end)
    return bad


def checker_problems(r):
    """H4: None when the user has no published result (no control); else the list of problems (empty = control as declared)."""
    c = r.get("snapshot_control")
    if c is None:
        return None
    bad = [] if c["unaltered"] else ["the unaltered snapshot is refused"]
    for n in ALTERATIONS:
        if c["altered"].get(n) is not False:
            bad.append(f"the alteration {n} is not refused")
    return bad


def published_by(r, day: str) -> bool:
    """For a user withheld at day 30: publishable after the re-certification with the rows to `day` (60 or 90)."""
    s60 = r["recert"].get("60", {}).get("state")
    if day == "60":
        return s60 == "PUBLISHABLE"
    return s60 == "PUBLISHABLE" or r["recert"].get("90", {}).get("state") in ("PUBLISHABLE", "PUBLISHABLE_EARLIER")


def reason_kinds(reasons):
    out = {}
    for x in reasons:
        out[x.split(":")[0]] = out.get(x.split(":")[0], 0) + 1
    return out


def arm_block(name, used, extra, failed):
    w = [r for r in used if r["day30"]["withheld"]]
    p60, p90 = sum(1 for r in w if published_by(r, "60")), sum(1 for r in w if published_by(r, "90"))
    still = [x for r in w if not published_by(r, "90") for x in r["recert"].get("90", {}).get("reasons", [])]
    return dict(name=name, users=len(used), extra=len(extra), failed=failed, withheld=len(w), seeds_with_withheld=[r["user"] for r in w], p60=p60, p90=p90,
                day30_reasons=reason_kinds([x for r in w for x in r["day30"]["reasons"]]), still_withheld_reasons=reason_kinds(still),
                strata={s: (sum(1 for r in used if r["stratum"] == s), sum(1 for r in w if r["stratum"] == s)) for s in sorted({r["stratum"] for r in used})})


def levels(arms: dict) -> dict:
    """arms: {name: (used, extra, failed)}. Returns the four hard levels, recomputed from the stored records."""
    used_all = [r for (u, _, _) in arms.values() for r in u]
    n_w = sum(1 for r in used_all if r["day30"]["withheld"])
    viol = {r["user"]: violations(r) for r in used_all}
    n_viol = sum(len(v) for v in viol.values())
    f1 = [(r["user"], e) for r in used_all for e in f1_disagreements(r)]
    n_f1 = sum(1 for r in used_all for e in r["recert"].values() if "f1_withheld" in e)
    ctl = {r["user"]: checker_problems(r) for r in used_all}
    n_ctl = sum(1 for v in ctl.values() if v is not None)
    ctl_bad = {u: v for u, v in ctl.items() if v}
    return dict(H1=dict(withheld=n_w, level=MIN_WITHHELD, met=n_w >= MIN_WITHHELD),
                H2=dict(violations=n_viol, detail={u: v for u, v in viol.items() if v}, met=n_viol == 0),
                H3=dict(calls=n_f1, disagreements=f1, met=not f1 and n_f1 > 0),
                H4=dict(controls=n_ctl, problems=ctl_bad, met=n_ctl > 0 and not ctl_bad))


def report(arms: dict) -> dict:
    blocks = {n: arm_block(n, u, e, f) for n, (u, e, f) in arms.items()}
    lv = levels(arms)
    P("A7 RE-CERTIFICATION, ARMS THAT ARE WITHHELD BY DESIGN (once; production fit_hssm, generated users; levels of Part 5 section 14, not moved)")
    for n, b in blocks.items():
        P(f"\narm {n}: users used {b['users']} (extra beyond the stopping rule, not counted: {b['extra']}; failed fits excluded: {b['failed']}); withheld at day 30: {b['withheld']}; strata (users, withheld) {b['strata']}")
        P(f"  withhold reasons at day 30: {b['day30_reasons'] or 'none'}")
        if b["withheld"]:
            for day, k in (("60", b["p60"]), ("90", b["p90"])):
                lo, hi = B9.wilson(k, b["withheld"])
                P(f"  publishable after re-certification with the rows to day {day}: {k}/{b['withheld']} = {100 * k / b['withheld']:.0f}% (Wilson 95% {100 * lo:.0f}-{100 * hi:.0f}%)")
            P(f"  reasons of those still withheld at day 90: {b['still_withheld_reasons'] or 'none'}")
    tw = sum(b["withheld"] for b in blocks.values())
    t60, t90 = sum(b["p60"] for b in blocks.values()), sum(b["p90"] for b in blocks.values())
    if tw:
        P(f"\npooled over the arms: withheld {tw}; publishable by day 60 {t60}/{tw} (Wilson {100 * B9.wilson(t60, tw)[0]:.0f}-{100 * B9.wilson(t60, tw)[1]:.0f}%), by day 90 {t90}/{tw} (Wilson {100 * B9.wilson(t90, tw)[0]:.0f}-{100 * B9.wilson(t90, tw)[1]:.0f}%)")
    P("\nLEVELS")
    P(f"  H1 non-vacuous (>= {MIN_WITHHELD} users withheld at the first check): {lv['H1']['withheld']} -> " + ("MET" if lv["H1"]["met"] else "NOT MET: too few to read"))
    P(f"  H2 no relaxation (0 violations): {lv['H2']['violations']} -> " + ("MET" if lv["H2"]["met"] else f"NOT MET {lv['H2']['detail']}"))
    P(f"  H3 same procedure (a fresh fit on the same matrix agrees on every recertify call): {lv['H3']['calls'] - len(lv['H3']['disagreements'])}/{lv['H3']['calls']} -> " + ("MET" if lv["H3"]["met"] else f"NOT MET {lv['H3']['disagreements']}"))
    P(f"  H4 the checker can fail (the unaltered snapshot published, each of the 5 alterations refused): {lv['H4']['controls']} controls, problems {len(lv['H4']['problems'])} -> " + ("MET" if lv["H4"]["met"] else f"NOT MET {lv['H4']['problems']}"))
    P("\nNot shown by this arm: that more days cure the withheld fits of the dual engine or of real data (UNKNOWN); the rates above are about generated users.")
    return dict(blocks=blocks, levels=lv)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    arms = {}
    for name, path in zip(("W1", "W2"), sys.argv[3:5]):
        recs, failed = load(path)
        used, extra = used_users(recs)
        arms[name] = (used, extra, failed)
    res = report(arms)
    Path(sys.argv[1]).write_text("\n".join(OUT) + "\n")
    json.dump(res, open(sys.argv[2], "w"), indent=1, default=str)
