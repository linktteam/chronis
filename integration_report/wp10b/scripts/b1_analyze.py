"""WP10B B1 analysis: reasons a dual fit does not publish, by stratum (design: tests/PREDECLARED_CRITERIA_WP10B.md Part 3). usage: python3 b1_analyze.py FITS.jsonl OUT.txt OUT.json
Descriptive (no pass/fail level): counts by reason category (a fit can carry several), the exclusive combinations, the duration degeneracy split, the EM-creep extrapolation, the box refit, and the held-out means (record only).
The 'predicted removal' lines are the declared predictions of each fix (section 3.4), computed from these same fits and compared with the observed removal in B2."""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "wp10b" / "scripts"))

STRATA = ("hard", "medium", "easy")
CATS = ("iteration_cap", "rejected_update", "other_non_convergence", "degenerate_duration", "collapsed_regime", "numerical", "other")
BUDGET_EXTRA = (200, 400, 800, 1600)        # further iterations on top of the 200 of setting S0 (so totals 400, 600, 1000, 1800)
OUT: list = []


def P(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)


def wilson(k: int, n: int, z: float = 1.959964):
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]" if n else "0/0"


def split_degeneracy(d: dict) -> dict:
    """Per dual record: how many regimes are degenerate, split into 'at a wall of the box' and 'pmf still rising at the last lattice step' (a regime can be both)."""
    deg = set(d.get("degenerate_regimes", []))
    wall = {k for k, v in d.get("at_bound", {}).items() if v}
    rising = {k for k, v in enumerate(d.get("pmf_rising", [])) if v}
    return dict(degenerate=len(deg), wall=len(wall & deg), rising=len(rising & deg), both=len(wall & rising & deg), wall_only=len((wall - rising) & deg), rising_only=len((rising - wall) & deg))


def summarize(recs: list) -> dict:
    out = {}
    for s in STRATA:
        R = [r for r in recs if r["job"]["stratum"] == s]
        if not R:
            continue
        ok = [r for r in R if r["ok"]]
        D = [r["dual"] for r in ok]
        non = [d for d in D if not d["publishable"]]
        cat_counts = {c: sum(1 for d in non if c in d["categories"]) for c in CATS}
        excl = collections.Counter(tuple(sorted(d["categories"])) for d in non)
        deg_regimes = sum(split_degeneracy(d)["degenerate"] for d in D)
        spl = [split_degeneracy(d) for d in D]
        cap = [d for d in non if "iteration_cap" in d["categories"] or "rejected_update" in d["categories"]]
        extra = [d["creep"]["extra_iterations"] for d in cap]
        known = [e for e in extra if e is not None]
        within = {b: sum(1 for e in extra if e is not None and e <= b) for b in BUDGET_EXTRA}
        gaps = {}
        for nm in ("gauss", "baseline"):
            v = [r["dual"]["test_mean"] - r[nm]["test_mean"] for r in ok if r.get(nm, {}).get("test_finite") and r["dual"].get("test_finite")]
            gaps[nm] = dict(n=len(v), mean=float(np.mean(v)) if v else float("nan"), se=float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan"))
        out[s] = dict(fits=len(R), failed=len(R) - len(ok), states=dict(collections.Counter(d["fit_state"] for d in D)), publishable=sum(1 for d in D if d["publishable"]), non_publishable=len(non),
                      cat_counts=cat_counts, exclusive={"+".join(k): v for k, v in excl.items()}, regimes=3 * len(D), degenerate_regimes=deg_regimes,
                      deg_wall=sum(x["wall"] for x in spl), deg_rising=sum(x["rising"] for x in spl), deg_both=sum(x["both"] for x in spl), deg_wall_only=sum(x["wall_only"] for x in spl),
                      deg_rising_only=sum(x["rising_only"] for x in spl), fits_with_degenerate=sum(1 for d in D if d["degenerate_regimes"]), n_cap=len(cap), cap_extra_known=len(known),
                      cap_extra_median=float(np.median(known)) if known else None, cap_within=within, box_refit=sum(1 for d in D if d["box_refit"]),
                      median_rate=float(np.median([d["creep"]["rate"] for d in cap if d["creep"]["rate"] is not None])) if any(d["creep"]["rate"] is not None for d in cap) else None,
                      dual_minus=gaps, wall_seconds_median=float(np.median([d["fit_seconds"] for d in D])) if D else None, optimizer_success=sum(1 for d in D if d["optimizer_success"]))
    return out


def report(res: dict, title: str):
    P(title)
    tot = dict(fits=0, pub=0, non=0)
    for s in STRATA:
        if s not in res:
            continue
        d = res[s]
        tot["fits"] += d["fits"] - d["failed"]; tot["pub"] += d["publishable"]; tot["non"] += d["non_publishable"]
        n = d["fits"] - d["failed"]
        P(f"\n[{s}] dual fits {d['fits']} (failed to run {d['failed']}); EM states {d['states']}; publishable {fmt(d['publishable'], n)}; non-publishable {d['non_publishable']}; median fit time {d['wall_seconds_median']}s")
        P("   reason categories among the non-publishable (a fit can carry several): " + ", ".join(f"{c} {d['cat_counts'][c]}" for c in CATS))
        P("   exclusive combinations: " + (", ".join(f"{k} x{v}" for k, v in sorted(d["exclusive"].items(), key=lambda kv: -kv[1])) or "none"))
        P(f"   degenerate duration: {d['degenerate_regimes']} of {d['regimes']} regimes in {d['fits_with_degenerate']} fits (at a wall {d['deg_wall']}, pmf rising at the last lattice step {d['deg_rising']}; both {d['deg_both']}, wall only {d['deg_wall_only']}, rising only {d['deg_rising_only']}); duration-box refit fired in {d['box_refit']} fits")
        P(f"   not finished by EM (iteration cap or rejected update): {d['n_cap']}; extrapolated further iterations to |increment| < 1e-4 known for {d['cap_extra_known']} (median {d['cap_extra_median']}, median creep rate {d['median_rate']}); within +{BUDGET_EXTRA[0]}: {d['cap_within'][BUDGET_EXTRA[0]]}, +{BUDGET_EXTRA[1]}: {d['cap_within'][BUDGET_EXTRA[1]]}, +{BUDGET_EXTRA[2]}: {d['cap_within'][BUDGET_EXTRA[2]]}, +{BUDGET_EXTRA[3]}: {d['cap_within'][BUDGET_EXTRA[3]]}")
        g = d["dual_minus"]
        P(f"   held-out one-step mean, dual minus single Gaussian {g['gauss']['mean']:+.3f} (se {g['gauss']['se']:.3f}, n {g['gauss']['n']}); dual minus baseline engine {g['baseline']['mean']:+.3f} (se {g['baseline']['se']:.3f}, n {g['baseline']['n']})  [record only]")
    P(f"\nALL STRATA: dual fits that ran {tot['fits']}, publishable {fmt(tot['pub'], tot['fits'])}")
    # declared predictions of what each fix removes (section 3.4), from these fits
    allr = [res[s] for s in STRATA if s in res]
    deg_fits = sum(d["fits_with_degenerate"] for d in allr)
    cap = sum(d["n_cap"] for d in allr)
    w400 = sum(d["cap_within"][BUDGET_EXTRA[1]] for d in allr)
    P(f"\nPREDICTIONS compared with B2 later: D60 targets degenerate_duration (present in {deg_fits} fits; rising-only regimes {sum(d['deg_rising_only'] for d in allr)}, wall-only {sum(d['deg_wall_only'] for d in allr)});"
      f" I600 (+400 iterations) targets the {cap} not finished by EM, of which {w400} are extrapolated to finish within +400")


def main(fits: str, out_txt: str, out_json: str):
    recs = [json.loads(l) for l in open(fits) if l.strip()]
    res = summarize(recs)
    report(res, f"WP10B B1 DIAGNOSIS (development users 11201..11236, setting S0): {fits}")
    Path(out_txt).write_text("\n".join(OUT) + "\n")
    json.dump(res, open(out_json, "w"), indent=1, default=float)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
