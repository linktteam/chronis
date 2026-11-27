"""WP10B B2 analysis (design and levels: tests/PREDECLARED_CRITERIA_WP10B.md Part 4, declared before any B2 fit). Paired with the S0 fits of B1.
Per variant V over the 36 development users: (i) publishable gain g (S0 non-publishable, V publishable) and loss l, exact one-sided sign test p = P(Bin(g + l, 1/2) >= g) < 0.05 and g - l >= 1; (ii) NOT HARMED: paired mean of
(held-out one-step mean of V - of S0) minus 1.645 * its standard error is >= -0.10 nats per step; class HELPED / NO EFFECT DEMONSTRATED / HARMED / HELPED BUT HARMFUL; reported: share publishable, categories, time ratio, fits stopped by the clock,
the SQUAREM trigger. usage: python3 b2_analyze.py REF_S0.jsonl B2.jsonl OUT.txt OUT.json"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

STRATA = ("hard", "medium", "easy")
VARIANTS = ("D60", "I600", "REG", "ALL")
SIGN_ALPHA = 0.05
HARM_MARGIN = -0.10
Z95 = 1.645
SQUAREM_MIN_STILL_CAPPED = 10          # more than 9 of 36
SQUAREM_MIN_TIME_RATIO = 3.0
OUT: list = []


def P(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)


def sign_test_p(g: int, l: int) -> float:
    """Exact one-sided sign test: P(Bin(g + l, 1/2) >= g); 1.0 when there is no discordant pair."""
    n = g + l
    if n <= 0:
        return 1.0
    return float(sum(math.comb(n, k) for k in range(g, n + 1)) / 2 ** n)


def paired_stats(d) -> dict:
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 2:
        return dict(n=n, mean=float("nan"), se=float("nan"), lower=float("nan"))
    m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(n))
    return dict(n=n, mean=m, se=se, lower=m - Z95 * se)


def classify(gain_demonstrated: bool, not_harmed: bool) -> str:
    if gain_demonstrated and not_harmed:
        return "HELPED"
    if gain_demonstrated:
        return "HELPED BUT HARMFUL"
    if not_harmed:
        return "NO EFFECT DEMONSTRATED"
    return "HARMED"


def read(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def by_user(recs, variant):
    return {r["job"]["user"]: r for r in recs if r["job"].get("variant", "S0") == variant}


def compare_variant(ref: dict, new: dict) -> dict:
    """ref, new: user -> record. Only users with a finished dual fit in both are paired; a variant fit that failed to run is counted separately (and as not publishable)."""
    users = sorted(set(ref) & set(new))
    ok_ref = {u: ref[u] for u in users if ref[u]["ok"]}
    failed = [u for u in users if not new[u]["ok"]]
    pair = [u for u in users if ref[u]["ok"] and new[u]["ok"]]
    pub_ref = {u: bool(ref[u]["dual"]["publishable"]) for u in ok_ref}
    pub_new = {u: bool(new[u]["dual"]["publishable"]) if new[u]["ok"] else False for u in users if ref[u]["ok"]}
    g = sum(1 for u in pub_ref if not pub_ref[u] and pub_new[u])
    l = sum(1 for u in pub_ref if pub_ref[u] and not pub_new[u])
    p = sign_test_p(g, l)
    d = [new[u]["dual"]["test_mean"] - ref[u]["dual"]["test_mean"] for u in pair if new[u]["dual"].get("test_finite") and ref[u]["dual"].get("test_finite")]
    ps = paired_stats(d)
    gain = bool(g - l >= 1 and p < SIGN_ALPHA)
    harmed_ok = bool(ps["lower"] >= HARM_MARGIN) if math.isfinite(ps["lower"]) else False
    cats = collections.Counter(c for u in pair if not new[u]["dual"]["publishable"] for c in new[u]["dual"]["categories"])
    t_ref = [ref[u]["dual"]["fit_seconds"] for u in pair]
    t_new = [new[u]["dual"]["fit_seconds"] for u in pair]
    return dict(n_users=len(users), n_paired=len(pair), failed_to_run=len(failed), pub_ref=sum(pub_ref.values()), pub_new=sum(pub_new.values()), gain=g, loss=l, sign_p=p, gain_demonstrated=gain, heldout=ps, not_harmed=harmed_ok,
                klass=classify(gain, harmed_ok), cats=dict(cats), still_iteration_cap=int(cats.get("iteration_cap", 0)), time_ratio=float(np.median(t_new) / np.median(t_ref)) if t_ref else float("nan"),
                clock_stops=sum(1 for u in pair if new[u]["dual"].get("wall_clock_stop")), ref_converged_changed=sum(1 for u in pair if ref[u]["dual"]["fit_state"] == "CONVERGED" and (new[u]["dual"]["fit_state"] != "CONVERGED")))


def squarem_trigger(i600: dict) -> dict:
    """The declared rule (Part 4, 4.4): build SQUAREM-S3 only if more than 9 of the 36 fits still end at the iteration cap after I600 AND the median fit time of I600 is at least 3 times that of S0."""
    capped, ratio = i600["still_iteration_cap"], i600["time_ratio"]
    return dict(still_iteration_cap=capped, time_ratio=ratio, build=bool(capped >= SQUAREM_MIN_STILL_CAPPED and math.isfinite(ratio) and ratio >= SQUAREM_MIN_TIME_RATIO))


def per_stratum(ref: dict, new: dict, strata_of) -> dict:
    out = {}
    for s in STRATA:
        r = {u: v for u, v in ref.items() if strata_of(u) == s}
        n = {u: v for u, v in new.items() if strata_of(u) == s}
        c = compare_variant(r, n)
        out[s] = dict(pub_ref=c["pub_ref"], pub_new=c["pub_new"], n=c["n_users"], gain=c["gain"], loss=c["loss"], heldout_mean=c["heldout"]["mean"])
    return out


def strata_of(u: int) -> str:
    return "hard" if u <= 11212 else ("medium" if u <= 11224 else "easy")


def main(ref_path, new_path, out_txt=None, out_json=None):
    ref = by_user(read(ref_path), "S0")
    recs = read(new_path)
    res = {}
    P(f"WP10B B2 (development users 11201..11236, setting S0, paired with the S0 fits of {ref_path}): {new_path}")
    P(f"levels (Part 4): HELPED = publishable gain with g - l >= 1 and exact one-sided sign test p < {SIGN_ALPHA} AND not harmed (paired held-out mean minus {Z95} se >= {HARM_MARGIN} nats per step)\n")
    for v in VARIANTS:
        new = by_user(recs, v)
        if not new:
            continue
        c = compare_variant(ref, new)
        c["strata"] = per_stratum(ref, new, strata_of)
        res[v] = c
        h = c["heldout"]
        P(f"[{v}] users {c['n_users']} (paired {c['n_paired']}, failed to run {c['failed_to_run']}); publishable S0 {c['pub_ref']} -> {v} {c['pub_new']}; gained {c['gain']}, lost {c['loss']}, exact one-sided sign p = {c['sign_p']:.4f} -> gain demonstrated: {c['gain_demonstrated']}")
        P(f"    held-out one-step mean, {v} minus S0 (paired, n {h['n']}): {h['mean']:+.4f} (se {h['se']:.4f}), one-sided 95% lower {h['lower']:+.4f} (margin {HARM_MARGIN}) -> not harmed: {c['not_harmed']}")
        P(f"    CLASS: {c['klass']}.  categories among the non-publishable: {c['cats']}; still at the iteration cap {c['still_iteration_cap']}; median fit time ratio {c['time_ratio']:.2f}; stopped by the clock {c['clock_stops']}; S0 CONVERGED fits that are no longer converged {c['ref_converged_changed']}")
        P("    by stratum (publishable S0 -> V, gained/lost, mean held-out change): " + "; ".join(f"{s} {x['pub_ref']}->{x['pub_new']} (+{x['gain']}/-{x['loss']}, {x['heldout_mean']:+.3f})" for s, x in c["strata"].items()))
        if c["clock_stops"]:
            P(f"    WARNING: {c['clock_stops']} fit(s) were stopped by the wall-clock guard: the variant's results depend on the speed of the machine")
    if "I600" in res:
        t = squarem_trigger(res["I600"])
        res["squarem_trigger"] = t
        P(f"\nSQUAREM trigger (Part 4, 4.4): still at the iteration cap after I600 {t['still_iteration_cap']} (needs >= {SQUAREM_MIN_STILL_CAPPED}), I600 median time ratio {t['time_ratio']:.2f} (needs >= {SQUAREM_MIN_TIME_RATIO}) -> build SQUAREM-S3: {t['build']}")
    if out_txt:
        Path(out_txt).write_text("\n".join(OUT) + "\n")
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    return res


if __name__ == "__main__":
    main(*sys.argv[1:5])
