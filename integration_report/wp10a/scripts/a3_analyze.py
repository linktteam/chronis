"""WP10A A3 analysis (design: tests/PREDECLARED_CRITERIA_WP10A.md section 3). Replays stored fits (`a3_run.py`) through the PRODUCTION lineage at a candidate operating point (gate h, consensus flag on/off).
usage: PYTHONPATH=. python3 integration_report/wp10a/scripts/a3_analyze.py dev  FITS.jsonl OUT.txt OUT.json
       PYTHONPATH=. python3 integration_report/wp10a/scripts/a3_analyze.py eval OUT.txt OUT.json FITS1.jsonl [FITS2.jsonl ...]        (needs tests/DESIGN_FREEZE_WP10A_A3.txt; reads the frozen operating point)
       add --rehearsal to run `eval` on DEVELOPMENT fits (plumbing test; reads no evaluation seed)
Flag ON = the consensus flag of WP9 rule C plus the posterior-sharpness floor s = 0.95; flag OFF = neither (s = 0, no consensus flag)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "wp9" / "scripts"))
import a12_analyze as A                                                           # noqa: E402  (WP9 replay helpers: run_candidate, tally_rows, stratum_users, split_chains)
import wp9_fit_lib as F                                                           # noqa: E402
import wp9_replay_lib as R                                                        # noqa: E402
from wp9_replay_lib import S                                                      # noqa: E402

from backbone.hssm.lineage import AlignmentConfig                                 # noqa: E402

G_T = list(range(90, 147, 7)) + [150]
B_T = [90, 104, 118, 132, 146, 150]
FIRST_T = 90
H_CANDS = (2.0, 2.25, 2.5)
STRATA = ("hard", "medium", "easy")
SHARP_ON = 0.95                                                                   # frozen in WP9 (config.FIT_MIN_POSTERIOR_SHARPNESS)
TARGET_NAMES, TARGET_COVERAGE, TARGET_WRONG = 0.98, 0.80, 0.01                    # declared levels; never changed after a result
SAFETY_BOUND = 0.01
OUT: list = []


def P(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)


def cand_list(flag_only: bool = False) -> list:
    return [(True, h) for h in H_CANDS] if flag_only else [(u, h) for u in (True, False) for h in H_CANDS]


def sharp_of(u: bool) -> float:
    return SHARP_ON if u else 0.0


def half_of(user: int) -> str:
    return "cal" if user % 2 == 0 else "val"                                      # even calibrate, odd validate (WP9 section 8, unchanged)


# ------------------------------------------------------------------------------------------------------------------------------ statistics
def exact_upper(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided exact (Clopper-Pearson) upper confidence bound of a binomial proportion; k = 0 gives 1 - (1 - conf)^(1/n) (the rule of three)."""
    if n <= 0:
        return float("nan")
    if k >= n:
        return 1.0
    if k == 0:
        return 1.0 - (1.0 - conf) ** (1.0 / n)
    from scipy.stats import beta
    return float(beta.ppf(conf, k + 1, n - k))


def safety_demonstrated(wrong: int, n: int, bound: float = SAFETY_BOUND) -> bool:
    """The declared test (PREDECLARED_CRITERIA_WP10A section 3.4): zero wrong-and-unflagged and the one-sided exact 95% upper bound at most `bound`."""
    return n > 0 and wrong == 0 and exact_upper(0, n) <= bound


def alignments_needed(wrong: int = 0, bound: float = SAFETY_BOUND) -> int:
    n = 1
    while exact_upper(wrong, n) > bound:
        n += 1
        if n > 100000:
            return -1
    return n


# ------------------------------------------------------------------------------------------------------------------------------ replay at an operating point
def replay_chain(chain: list, u: bool, h: float, windows: list):
    cfg = AlignmentConfig(min_separation=h)
    return R.replay9([r for r in chain if r["job"]["end"] in windows], cfg, n=10, use_flag=u, sharpness=sharp_of(u))[:2]


def pair_cases(ra: list, la: list, rb: list, lb: list, names_level: float = 0.99) -> list:
    """The 14-day windows after the first fit (end > 90) at which at least one of the two seeds has an accepted fit. Per case: serve flags of both seeds and, when both serve, whether the per-day
    stable ids agree on at least `names_level` of the days."""
    da = {r["end"]: (r, lr) for r, lr in zip(ra, la)}
    db = {r["end"]: (r, lr) for r, lr in zip(rb, lb)}
    out = []
    for end in sorted(set(da) | set(db)):
        if end <= FIRST_T:
            continue
        x, y = da.get(end), db.get(end)
        sa = x is not None and x[0]["state"] == "ALIGNED"
        sb = y is not None and y[0]["state"] == "ALIGNED"
        agree = None
        if sa and sb:
            agree = bool(S.day_agreement(x[1].stable_regime_id, y[1].stable_regime_id) >= names_level)
        out.append(dict(end=end, a=sa, b=sb, agree=agree, one_missing=(x is None) != (y is None)))
    return out


def names_metric(cases: list) -> tuple:
    """(a): (agree, both-serve cases)."""
    both = [c for c in cases if c["a"] and c["b"]]
    return sum(1 for c in both if c["agree"]), len(both)


def serve_agreement(cases: list) -> dict:
    """(b): both serve / both withhold / exactly one serves; agreement = (both serve + both withhold) / all cases; and the same restricted to windows where at least one seed serves."""
    bs = sum(1 for c in cases if c["a"] and c["b"])
    bw = sum(1 for c in cases if not c["a"] and not c["b"])
    one = sum(1 for c in cases if c["a"] != c["b"])
    return dict(cases=len(cases), both_serve=bs, both_withhold=bw, one_serves=one, agree=bs + bw, agree_where_either_serves=(bs, bs + one))


def strict_identical(cases: list) -> tuple:
    """The WP9 A1-1 criterion (target 95%): cases = windows where at least one seed serves; identical = both serve with the same names."""
    st = [c for c in cases if c["a"] or c["b"]]
    return sum(1 for c in st if c["a"] and c["b"] and c["agree"]), len(st)


def first_served(rows: list, first_t: int = FIRST_T):
    """(c): the window end t* of the first later step ending ALIGNED, or None (never served by the last window). Days to first served lineage = t* - first_t."""
    for r in rows:
        if r["end"] > first_t and r["state"] == "ALIGNED":
            return r["end"]
    return None


def days_summary(tstars: list, first_t: int = FIRST_T, horizon: int = 150) -> dict:
    """tstars: window end of the first served step per user (None = never served). Shares served within 14, 28 and the full horizon (60) days after the first fit; median days among served;
    median with never-served counted as horizon - first_t + 1 (61); quartiles of that censored distribution."""
    n = len(tstars)
    cen = horizon - first_t + 1
    days = [None if t is None else t - first_t for t in tstars]
    served = [d for d in days if d is not None]
    cens = sorted(cen if d is None else d for d in days)

    def q(p):
        if not cens:
            return None
        i = min(len(cens) - 1, max(0, int(math.ceil(p * len(cens))) - 1))
        return cens[i]
    return dict(n=n, served_by_14=sum(1 for d in served if d <= 14), served_by_28=sum(1 for d in served if d <= 28), served_by_60=len(served), never=n - len(served),
                median_among_served=(sorted(served)[(len(served) - 1) // 2] if served else None), q25_censored=q(0.25), median_censored=q(0.5), q75_censored=q(0.75))


# ------------------------------------------------------------------------------------------------------------------------------ the selection rule (frozen, section 3.2)
def choose(curve: dict, cands: list):
    """curve {(u, h): {'pooled'|'cal'|'val': tally}}. WP9 rule: (1) among the candidates with zero wrong-and-unflagged on the calibration half take the highest calibration coverage (ties: larger h, then flag
    ON); (2) accept it when its validation-half wrong-and-unflagged share is at most 1% (point estimate); (3) otherwise the same pick on the pooled set; if no candidate has zero there either, the one with the
    fewest wrong-and-unflagged (then the highest pooled coverage). Returns (candidate, how)."""
    def key(c, half):
        t = curve[c][half]
        return (t["served"] / max(t["refits"], 1), c[1], 1 if c[0] else 0)
    zero_cal = [c for c in cands if curve[c]["cal"]["wrong_unflagged"] == 0]
    if zero_cal:
        c = max(zero_cal, key=lambda c: key(c, "cal"))
        v = curve[c]["val"]
        if v["alignments"] and v["wrong_unflagged"] / v["alignments"] <= 0.01:
            return c, "step 1-2: highest calibration coverage among candidates with zero wrong-and-unflagged on the calibration half; validation half wrong-and-unflagged <= 1%"
    zero_all = [c for c in cands if curve[c]["pooled"]["wrong_unflagged"] == 0]
    if zero_all:
        return max(zero_all, key=lambda c: key(c, "pooled")), "step 3: highest pooled coverage among candidates with zero wrong-and-unflagged on the pooled development set"
    return min(cands, key=lambda c: (curve[c]["pooled"]["wrong_unflagged"], -key(c, "pooled")[0])), "no candidate has zero wrong-and-unflagged: the one with the fewest"


# ------------------------------------------------------------------------------------------------------------------------------ one candidate over a set of chains
def evaluate_candidate(a: dict, b: dict, u: bool, h: float, strata: dict) -> dict:
    """Everything the declared metrics need for one operating point: coverage tallies (weekly seed-A chains), pair cases (both seeds on the 14-day windows) and first-served windows, all per user."""
    wk = {k: [r for r in v if r["job"]["end"] in G_T] for k, v in a.items()}
    res = A.run_candidate(wk, h, sharp_of(u), u, 10)
    per_user = {}
    for (user, _), (rows, _) in res.items():
        per_user.setdefault(user, {})["tstar_A"] = first_served(rows)
    by_user_b = {k[0]: v for k, v in b.items()}
    for (user, _), ca in a.items():
        cb = by_user_b.get(user)
        if cb is None:
            continue
        ra, la = replay_chain(ca, u, h, B_T)
        rb, lb = replay_chain(cb, u, h, B_T)
        per_user.setdefault(user, {})["cases"] = pair_cases(ra, la, rb, lb)
        per_user[user]["tstar_B"] = first_served(rb)
    return dict(res=res, per_user=per_user)


def summarize(ev: dict, users, tally_users=None) -> dict:
    """Pooled summary of one evaluate_candidate result over `users` (None = all)."""
    pu = {u: v for u, v in ev["per_user"].items() if users is None or u in users}
    cases = [c for v in pu.values() for c in v.get("cases", [])]
    t = A.tally_rows(ev["res"], users)
    ka, na = names_metric(cases)
    ks, ns = strict_identical(cases)
    return dict(tally=t, names=(ka, na), serve=serve_agreement(cases), strict=(ks, ns), days_A=days_summary([v.get("tstar_A") for v in pu.values()]),
                days_B=days_summary([v.get("tstar_B") for v in pu.values() if "cases" in v]), users=len(pu))


def fmt_t(t: dict) -> str:
    return f"coverage {S.rate(t['served'], t['refits'])}, wrong-and-unflagged {t['wrong_unflagged']}/{t['alignments']}"


def fmt_days(d: dict) -> str:
    return (f"n={d['n']}: served by day 14 {d['served_by_14']}, by day 28 {d['served_by_28']}, by day 60 {d['served_by_60']}, never {d['never']}; median among served {d['median_among_served']}; "
            f"censored (never = 61) quartiles {d['q25_censored']} / {d['median_censored']} / {d['q75_censored']}")


def report_block(title: str, ev: dict, strata: dict, users_all=None, with_verdicts: bool = False):
    P(f"\n== {title} ==")
    for name, users in [("pooled", users_all)] + [(st, strata[st] if users_all is None else strata[st] & users_all) for st in STRATA]:
        s = summarize(ev, users)
        t = s["tally"]
        P(f"-- {name}: users {s['users']}, accepted refits {t['refits']}, alignments {t['alignments']}")
        P(f"   {fmt_t(t)}")
        if with_verdicts:
            P(f"   coverage target >= 80%: {S.verdict(t['served'], t['refits'], TARGET_COVERAGE, '>=')}")
            P(f"   wrong-and-unflagged target <= 1%: {S.verdict(t['wrong_unflagged'], t['alignments'], TARGET_WRONG, '<=')}")
        ka, na = s["names"]
        P(f"   (a) names agree when both seeds serve: {S.rate(ka, na)}" + (f"  -> {S.verdict(ka, na, TARGET_NAMES, '>=')} (target >= 98%)" if with_verdicts else ""))
        sv = s["serve"]
        P(f"   (b) serve/withhold agreement: {S.rate(sv['agree'], sv['cases'])}; both serve {sv['both_serve']}, both withhold {sv['both_withhold']}, exactly one serves {sv['one_serves']}; where at least one serves: {S.rate(*sv['agree_where_either_serves'])}")
        P(f"   (c) days to first served lineage, seed A weekly: {fmt_days(s['days_A'])}")
        P(f"       seed B 14-day: {fmt_days(s['days_B'])}")
        ks, ns = s["strict"]
        P(f"   old A1-1 criterion (strict identical lineage among served, target 95%): {S.rate(ks, ns)}" + (f"  -> {S.verdict(ks, ns, 0.95, '>=')}" if with_verdicts else ""))


# ------------------------------------------------------------------------------------------------------------------------------ development
def dev(records, out_json):
    a, b = A.split_chains(records)
    strata = {st: set(A.stratum_users(a).get(st, ())) for st in STRATA}
    P(f"DEVELOPMENT (A3): {len(records)} fits; users {len(a)} (seed A) / {len(b)} (seed B); per stratum { {k: len(v) for k, v in strata.items()} }")
    curve, evs = {}, {}
    cal_users = {u for (u, _) in a if half_of(u) == "cal"}
    val_users = {u for (u, _) in a if half_of(u) == "val"}
    for (u, h) in cand_list():
        ev = evaluate_candidate(a, b, u, h, strata)
        evs[(u, h)] = ev
        curve[(u, h)] = dict(pooled=A.tally_rows(ev["res"]), cal=A.tally_rows(ev["res"], cal_users), val=A.tally_rows(ev["res"], val_users), **{st: A.tally_rows(ev["res"], strata[st]) for st in STRATA})
        row = curve[(u, h)]
        P(f"flag {'ON ' if u else 'OFF'} h = {h:<4}: pooled {fmt_t(row['pooled'])} | cal {fmt_t(row['cal'])} | val {fmt_t(row['val'])} | " +
          " | ".join(f"{st} {row[st]['served']}/{row[st]['refits']} w{row[st]['wrong_unflagged']}" for st in STRATA))
    on = cand_list(flag_only=True)
    ch_on, how_on = choose(curve, on)
    ch_all, how_all = choose(curve, cand_list())
    P(f"\nCHOSEN OPERATING POINT (frozen rule, over the three flag-ON candidates): flag ON, h = {ch_on[1]}. {how_on}")
    P(f"Winner of the same rule over all six candidates (reported; replayed on the evaluation fits as a secondary line if it is flag OFF): flag {'ON' if ch_all[0] else 'OFF'}, h = {ch_all[1]}. {how_all}")
    for (u, h) in cand_list():
        report_block(f"development readout, flag {'ON' if u else 'OFF'}, h = {h}", evs[(u, h)], strata)
    json.dump(dict(chosen=dict(h=ch_on[1], u=True, s=SHARP_ON, how=how_on), winner_all=dict(h=ch_all[1], u=ch_all[0], s=sharp_of(ch_all[0]), how=how_all),
                   curve={f"u={int(k[0])},h={k[1]}": v for k, v in curve.items()},
                   readout={f"u={int(k[0])},h={k[1]}": {name: summarize(evs[k], users) for name, users in [("pooled", None)] + [(st, strata[st]) for st in STRATA]} for k in evs}),
              open(out_json, "w"), indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o))


# ------------------------------------------------------------------------------------------------------------------------------ evaluation
def evaluate(records, out_json, gate: dict, rehearsal: bool):
    a, b = A.split_chains(records)
    strata = {st: set(A.stratum_users(a).get(st, ())) for st in STRATA}
    ev_users = {u for (u, _) in a if u < 10150} if not rehearsal else {u for (u, _) in a}
    P(f"EVALUATION (A3): {len(records)} fits; users {len(a)}; per stratum { {k: len(v) for k, v in strata.items()} }; frozen operating point {gate}")
    ev = evaluate_candidate(a, b, gate["u"], gate["h"], strata)
    report_block("POOLED EVALUATION POPULATION (EVAL + SAFE-EXTRA users; headline for the safety claim and for (a) to (c))", ev, strata, None, with_verdicts=True)
    if not rehearsal:
        report_block("EVAL users only (10101..10148; the 48-user readout, reported beside the pooled one)", ev, strata, ev_users, with_verdicts=True)
    P("\n== SAFETY DEMONSTRATION (seed-A weekly chains, 10 restarts, the frozen operating point; unit = gate-eligible alignment) ==")
    P(f"declared test: zero wrong-and-unflagged and the one-sided exact 95% upper bound at most 1% (needs n >= {alignments_needed(0)}; with one wrong n >= {alignments_needed(1)})")
    safety = {}
    for name, users in [("pooled", None)] + [(st, strata[st]) for st in STRATA]:
        t = A.tally_rows(ev["res"], users)
        n, w = t["alignments"], t["wrong_unflagged"]
        lo, hi = S.wilson(w, n) if n else (float("nan"), float("nan"))
        safety[name] = dict(n=n, wrong=w, exact_upper=exact_upper(w, n), wilson_upper=hi, demonstrated=safety_demonstrated(w, n))
        P(f"   {name:7s}: wrong-and-unflagged {w}/{n}; one-sided exact 95% upper bound {100 * exact_upper(w, n):.2f}%; two-sided Wilson upper {100 * hi:.2f}%; 1% bound demonstrated: {'YES' if safety_demonstrated(w, n) else 'NO'}")
    # secondary lines on the same fits (no verdicts): the current default (h = 2.5, flag ON) and the flag-OFF line at the chosen h
    secondary = {}
    for (u, h, label) in ((True, 2.5, "current default (h = 2.5, flag ON)"), (False, gate["h"], f"flag OFF at the chosen h = {gate['h']}"), (False, 2.5, "flag OFF at h = 2.5")):
        e2 = evaluate_candidate(a, b, u, h, strata)
        report_block(f"SECONDARY (reported, no verdict): {label}", e2, strata, None, with_verdicts=False)
        secondary[label] = {name: summarize(e2, users) for name, users in [("pooled", None)] + [(st, strata[st]) for st in STRATA]}
    P("\n== curve over the six candidates on the pooled evaluation population (REPORTED; nothing is chosen from it) ==")
    curve = {}
    for (u, h) in cand_list():
        e3 = ev if (u, h) == (gate["u"], gate["h"]) else evaluate_candidate(a, b, u, h, strata)
        curve[f"u={int(u)},h={h}"] = dict(pooled=A.tally_rows(e3["res"]), **{st: A.tally_rows(e3["res"], strata[st]) for st in STRATA})
        c = curve[f"u={int(u)},h={h}"]
        P(f"   flag {'ON ' if u else 'OFF'} h = {h:<4}: pooled {fmt_t(c['pooled'])} | " + " | ".join(f"{st} {c[st]['served']}/{c[st]['refits']} w{c[st]['wrong_unflagged']}" for st in STRATA))
    json.dump(dict(gate=gate, safety=safety, curve=curve, headline={name: summarize(ev, users) for name, users in [("pooled", None)] + [(st, strata[st]) for st in STRATA]}, secondary=secondary),
              open(out_json, "w"), indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o))


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    mode = args[0]
    if mode == "dev":
        fits, out_txt, out_json = args[1:4]
        dev(F.load_records(fits), out_json)
    else:
        out_txt, out_json, fits = args[1], args[2], args[3:]
        rehearsal = "--rehearsal" in sys.argv
        freeze = Path("tests/DESIGN_FREEZE_WP10A_A3.txt")
        if not freeze.exists() and not rehearsal:
            sys.exit("evaluation needs tests/DESIGN_FREEZE_WP10A_A3.txt")
        gate = json.loads(freeze.read_text().splitlines()[0]) if freeze.exists() else dict(h=2.5, s=SHARP_ON, u=True)
        recs = [r for f in fits for r in F.load_records(f)]
        evaluate(recs, out_json, gate, rehearsal)
    Path(out_txt).write_text("\n".join(OUT) + "\n")
