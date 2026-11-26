"""WP10A A6 part 2 analysis: the harmful-prior control (rules: tests/PREDECLARED_CRITERIA_WP10A.md Part 6 section 18 and Part 6b). usage:
  PYTHONPATH=. python3 integration_report/wp10a/scripts/a6_harm_analyze.py dev  WP9_B1_DEV_FITS.jsonl HARM_DEV_FITS.jsonl OUT.txt OUT.json
  PYTHONPATH=. python3 integration_report/wp10a/scripts/a6_harm_analyze.py eval HARM_EVAL_FITS.jsonl OUT.txt OUT.json
dev: for each candidate (H1, H2, H3 in this order) the validity rule (a) mean(fit - personal) <= -0.25 nats/step, (b) at least 6 of 36 fits below the single Gaussian, (c) at least 4 of those eligible for the
certificate; the first candidate verified is frozen. eval: E1 (no eligible harmful fit worse than the Gaussian is certified; at least 12 such fits), E0 (the true prior certifies at least one fit), E3 (reported)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

ORDER = ("H1", "H2", "H3")
A_MAX, B_MIN, C_MIN, N_DEV = -0.25, 6, 4, 36
REQUIRED_EVAL = 12
CERTIFIED = "CERTIFIED_PRIOR_ANCHORED"
OUT: list = []


def P(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)


def cp_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact one-sided Clopper-Pearson upper bound for a rate with k events in n (1 when all n)."""
    if n <= 0:
        return float("nan")
    return 1.0 if k >= n else float(beta.ppf(1.0 - alpha, k + 1, n - k))


def score_of(rec):
    """The held-out mean log density, nan when it is not finite (or absent)."""
    m = rec.get("mean")
    if m is None or not rec.get("finite", False) or not math.isfinite(m):
        return float("nan")
    return float(m)


def worse_than_gauss(rec, gauss) -> bool:
    """Part 6b: a non-finite score is worse than the Gaussian; otherwise strictly below it. `gauss` must be finite (else the pair is unusable and this is not asked)."""
    s = score_of(rec)
    return (not math.isfinite(s)) or s < gauss


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def baseline_from_b1(path) -> dict:
    """{(seed, length): dict(personal=score or nan, gauss=score or nan)} from the WP9 B1 file."""
    out = {}
    for r in read_jsonl(path):
        if r["arm"] in ("personal", "gauss"):
            out.setdefault((r["seed"], r["length"]), {})[r["arm"]] = score_of(r)
    return out


def gauss_from(recs) -> dict:
    return {(r["seed"], r["length"]): score_of(r) for r in recs if r["arm"] == "gauss"}


def validity(fits: list, baseline: dict) -> dict:
    """Rules (a), (b), (c) for the fits of one candidate against the WP9 personal and Gaussian scores of the same (seed, length)."""
    diffs, left_out, unusable, worse, worse_elig = [], 0, 0, 0, 0
    for r in fits:
        b = baseline.get((r["seed"], r["length"]), {})
        s = score_of(r)
        pers, g = b.get("personal", float("nan")), b.get("gauss", float("nan"))
        if math.isfinite(s) and math.isfinite(pers):
            diffs.append(s - pers)
        else:
            left_out += 1
        if not math.isfinite(g):
            unusable += 1
            continue
        if worse_than_gauss(r, g):
            worse += 1
            worse_elig += int(bool(r.get("eligible")))
    a = float(np.mean(diffs)) if diffs else float("nan")
    ok_a, ok_b, ok_c = (math.isfinite(a) and a <= A_MAX), worse >= B_MIN, worse_elig >= C_MIN
    return dict(n=len(fits), mean_diff_to_personal=a, n_in_mean=len(diffs), left_out=left_out, unusable_pairs=unusable, worse_than_gauss=worse, worse_and_eligible=worse_elig,
                rule_a=ok_a, rule_b=ok_b, rule_c=ok_c, verified=bool(ok_a and ok_b and ok_c))


def choose(results: dict):
    """The first candidate in ORDER that is verified (None when none)."""
    for c in ORDER:
        if c in results and results[c]["verified"]:
            return c
    return None


def dev(b1_fits, harm_fits, out_json):
    base = baseline_from_b1(b1_fits)
    recs = read_jsonl(harm_fits)
    res = {}
    P("A6 HARMFUL-PRIOR CONTROL, DEVELOPMENT (WP9 B1 development users 9631..9642, 3 lengths; the personal and Gaussian scores are the stored WP9 ones)")
    for c in ORDER:
        fits = [r for r in recs if r["arm"] == c]
        if not fits:
            continue
        v = validity(fits, base)
        res[c] = v
        P(f"\n{c}: {v['n']} fits (of 36); (a) mean(fit - personal) = {v['mean_diff_to_personal']:+.3f} nats/step over {v['n_in_mean']} fits ({v['left_out']} left out as non-finite) -> {'ok' if v['rule_a'] else 'NO'} (needs <= {A_MAX});"
          f" (b) fits below the single Gaussian {v['worse_than_gauss']} -> {'ok' if v['rule_b'] else 'NO'} (needs >= {B_MIN}); (c) of those eligible for the certificate {v['worse_and_eligible']} -> {'ok' if v['rule_c'] else 'NO'} (needs >= {C_MIN});"
          f" unusable pairs {v['unusable_pairs']}; VERIFIED HARMFUL: {v['verified']}")
        cert = sum(1 for r in fits if (r.get("certification") or {}).get("state") == CERTIFIED)
        elig = sum(1 for r in fits if r.get("eligible"))
        P(f"   for the record: eligible {elig}/{len(fits)}, certified {cert}, published after the certificate {sum(1 for r in fits if r.get('published_after'))}")
    chosen = choose(res)
    P(f"\nFROZEN BY THE DECLARED RULE: {chosen if chosen else 'NONE VERIFIED -> the control cannot be built; whether a wrong prior would be refused stays UNKNOWN'}")
    json.dump(dict(candidates=res, chosen=chosen), open(out_json, "w"), indent=1, default=float)
    return chosen


def evaluate(fits, out_json):
    recs = read_jsonl(fits)
    g = gauss_from(recs)
    H = [r for r in recs if r["arm"] == "H"]
    T = [r for r in recs if r["arm"] == "T"]
    usable = [r for r in H if math.isfinite(g.get((r["seed"], r["length"]), float("nan")))]
    worse = [r for r in usable if worse_than_gauss(r, g[(r["seed"], r["length"])])]
    ew = [r for r in worse if r.get("eligible")]
    k = sum(1 for r in ew if (r.get("certification") or {}).get("state") == CERTIFIED)
    n = len(ew)
    upper = cp_upper(k, n)
    t_cert = sum(1 for r in T if (r.get("certification") or {}).get("state") == CERTIFIED)
    t_elig = sum(1 for r in T if r.get("eligible"))
    pub_worse = sum(1 for r in worse if r.get("published_after"))
    other = sum(1 for r in H if not r.get("eligible") and r.get("fit_state") not in ("CONVERGED",))
    converged = sum(1 for r in H if r.get("fit_state") == "CONVERGED")
    valid = n >= REQUIRED_EVAL
    e1 = "NOT VALID (too few eligible harmful fits: the question stays UNKNOWN)" if not valid else ("MET" if k == 0 else "NOT MET")
    e0 = t_cert >= 1
    P("A6 HARMFUL-PRIOR CONTROL, EVALUATION (once; seeds 10771..10794, 3 lengths; WP9 B1 settings; fresh seeds)")
    P(f"arm H (the frozen harmful prior): {len(H)} fits ({len(usable)} with a usable Gaussian); worse than the single Gaussian on the held-out days: {len(worse)}; of those eligible for the certificate: {n} (required: at least {REQUIRED_EVAL}) -> control {'VALID' if valid else 'NOT VALID'}")
    P(f"   EM states of arm H: converged {converged}, eligible (REJECTED_UPDATE / MAX_ITER_REACHED, nothing else withholding) {sum(1 for r in H if r.get('eligible'))}, refused by another rule or failed {other}")
    P(f"E1 certified among the {n} eligible harmful fits that are worse than the Gaussian: {k} (exact one-sided 95% upper bound on the certification rate {100 * upper:.0f}%) -> {e1}")
    P(f"E0 arm T (the true prior): {len(T)} fits, eligible {t_elig}, certified {t_cert} -> {'informative (the certificate does certify some fits)' if e0 else 'NOT INFORMATIVE: the certificate certified nothing, so refusing the harmful fits says nothing'}")
    P(f"E3 (reported, no level): arm-H fits published (certified or accepted by the EM rule) although they score below the single Gaussian: {pub_worse} of {len(worse)}")
    mean_gap = float(np.mean([score_of(r) - g[(r['seed'], r['length'])] for r in usable if math.isfinite(score_of(r))])) if usable else float("nan")
    P(f"   mean(arm-H fit - single Gaussian) over the finite fits: {mean_gap:+.3f} nats/step")
    if e1 == "MET" and not e0:
        P("   E1 is met but not informative (see E0).")
    json.dump(dict(H=len(H), usable=len(usable), worse=len(worse), eligible_and_worse=n, certified=k, upper=upper, valid=valid, E1=e1, T=len(T), T_certified=t_cert, E0=e0, E3_published_and_worse=pub_worse,
                   converged=converged, other=other, mean_gap=mean_gap), open(out_json, "w"), indent=1, default=float)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "dev" and len(a) == 5:
        dev(a[1], a[2], a[4])
        Path(a[3]).write_text("\n".join(OUT) + "\n")
    elif a and a[0] == "eval" and len(a) == 4:
        evaluate(a[1], a[3])
        Path(a[2]).write_text("\n".join(OUT) + "\n")
    else:
        sys.exit(__doc__)
