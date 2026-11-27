"""WP10B C1(2) analysis: the mildly wrong prior control (rules: tests/PREDECLARED_CRITERIA_WP10B.md section 4 and Part 1b). usage:
  PYTHONPATH=. python3 integration_report/wp10b/scripts/c1_mild_analyze.py dev  DEV_FITS.jsonl OUT.txt OUT.json
  PYTHONPATH=. python3 integration_report/wp10b/scripts/c1_mild_analyze.py eval EVAL_FITS.jsonl OUT.txt OUT.json [EXT_FITS.jsonl]
dev: per candidate (M1, M2, M3) the validity rule: (b) at least 6 of the 36 fits below the same user's and length's single Gaussian, (c) at least 4 of those eligible for the certificate (rule (a) of WP10A is dropped, section 4.1).
The frozen candidate is the verified one with the smallest norm of the d_obs change. eval: E1 (certified among eligible harmful fits; at least 12 such fits), E0 (the true prior certifies some fits), E3 (reported)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

ORDER = ("M1", "M2", "M3")
B_MIN, C_MIN, N_DEV = 6, 4, 36
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
    """A non-finite score is worse than the Gaussian; otherwise strictly below it. `gauss` must be finite (else the pair is unusable and this is not asked)."""
    s = score_of(rec)
    return (not math.isfinite(s)) or s < gauss


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def gauss_from(recs) -> dict:
    return {(r["seed"], r["length"]): score_of(r) for r in recs if r["arm"] == "gauss"}


def certified(rec) -> bool:
    return (rec.get("certification") or {}).get("state") == CERTIFIED


def validity(fits: list, gauss: dict) -> dict:
    """Rules (b) and (c) for the fits of one candidate against the single-Gaussian score of the same (seed, length)."""
    unusable = worse = worse_elig = 0
    gaps = []
    for r in fits:
        g = gauss.get((r["seed"], r["length"]), float("nan"))
        if not math.isfinite(g):
            unusable += 1
            continue
        s = score_of(r)
        if math.isfinite(s):
            gaps.append(s - g)
        if worse_than_gauss(r, g):
            worse += 1
            worse_elig += int(bool(r.get("eligible")))
    ok_b, ok_c = worse >= B_MIN, worse_elig >= C_MIN
    return dict(n=len(fits), unusable_pairs=unusable, worse_than_gauss=worse, worse_and_eligible=worse_elig, mean_gap_to_gauss=float(np.mean(gaps)) if gaps else float("nan"),
                rule_b=ok_b, rule_c=ok_c, verified=bool(ok_b and ok_c))


def choose(results: dict, norms: dict):
    """Among the verified candidates the one with the smallest norm of the d_obs change (the mildest = the hardest test of the certificate); ties go to the declared order. None when none is verified."""
    ok = [c for c in ORDER if c in results and results[c]["verified"]]
    return min(ok, key=lambda c: (norms[c], ORDER.index(c))) if ok else None


def eligible_harmful_count(recs: list) -> int:
    """BLIND to the certification outcome (never reads it): arm-M fits that are eligible for the certificate and score below the single Gaussian of the same (seed, length)."""
    g = gauss_from(recs)
    n = 0
    for r in recs:
        if r["arm"] != "M":
            continue
        gg = g.get((r["seed"], r["length"]), float("nan"))
        if math.isfinite(gg) and r.get("eligible") and worse_than_gauss(r, gg):
            n += 1
    return n


def needs_extension(n_eligible_harmful: int) -> bool:
    """The pre-declared extension (seeds 11061..11080, once) is allowed only when fewer than 12 eligible harmful fits were found."""
    return n_eligible_harmful < REQUIRED_EVAL


def verdict_e1(k: int, n: int) -> str:
    """Part 1b: at least one certified eligible harmful fit is NOT MET whatever n is; none certified is MET with n >= 12 and NOT EVALUABLE with fewer."""
    if k >= 1:
        return "NOT MET"
    return "MET" if n >= REQUIRED_EVAL else "NOT EVALUABLE (too few eligible harmful fits)"


def dev(fits_path, out_json, prior_path="wp7r_evidence/prior_main_wp5.json"):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import c1_mild_run as R
    d = R.true_dict(prior_path)
    norms = {c: R.dobs_norm(d, c) for c in ORDER}
    recs = read_jsonl(fits_path)
    g = gauss_from(recs)
    res = {}
    P("C1(2) MILD HARMFUL-PRIOR CONTROL, DEVELOPMENT (seeds 11001..11012, 3 lengths; single-Gaussian scores from the same run; rule (a) of WP10A dropped, see section 4.1)")
    P(f"norm of the d_obs change: " + ", ".join(f"{c} {norms[c]:.2f}" for c in ORDER) + f" (H3, every intercept + 16: {R.h3_norm(d):.2f})")
    for c in ORDER:
        fits = [r for r in recs if r["arm"] == c]
        if not fits:
            continue
        v = validity(fits, g)
        v["norm"] = norms[c]
        res[c] = v
        P(f"\n{c}: {v['n']} fits (of {N_DEV}); (b) fits below the single Gaussian {v['worse_than_gauss']} -> {'ok' if v['rule_b'] else 'NO'} (needs >= {B_MIN}); (c) of those eligible for the certificate {v['worse_and_eligible']} -> {'ok' if v['rule_c'] else 'NO'} (needs >= {C_MIN});"
          f" unusable pairs {v['unusable_pairs']}; mean(fit - single Gaussian) {v['mean_gap_to_gauss']:+.3f} nats/step (record only); VERIFIED HARMFUL (mild): {v['verified']}")
        P(f"   for the record: eligible {sum(1 for r in fits if r.get('eligible'))}/{len(fits)}, certified {sum(1 for r in fits if certified(r))}, published after the certificate {sum(1 for r in fits if r.get('published_after'))}")
    chosen = choose(res, norms)
    P(f"\nFROZEN BY THE DECLARED RULE: {chosen if chosen else 'NONE VERIFIED -> no mild harmful control could be built in this setting; the evaluation is NOT run'}")
    json.dump(dict(candidates=res, chosen=chosen, norms=norms), open(out_json, "w"), indent=1, default=float)
    return chosen


def evaluate(fits_paths, out_json):
    recs = [r for p in fits_paths for r in read_jsonl(p)]
    g = gauss_from(recs)
    M = [r for r in recs if r["arm"] == "M"]
    T = [r for r in recs if r["arm"] == "T"]
    usable = [r for r in M if math.isfinite(g.get((r["seed"], r["length"]), float("nan")))]
    worse = [r for r in usable if worse_than_gauss(r, g[(r["seed"], r["length"])])]
    ew = [r for r in worse if r.get("eligible")]
    k = sum(1 for r in ew if certified(r))
    n = len(ew)
    assert n == eligible_harmful_count(recs), "the blind count and the evaluation count disagree"
    upper = cp_upper(k, n)
    t_cert = sum(1 for r in T if certified(r))
    t_elig = sum(1 for r in T if r.get("eligible"))
    pub_worse = sum(1 for r in worse if r.get("published_after"))
    e1 = verdict_e1(k, n)
    e0 = t_cert >= 1
    by_len = {ln: (sum(1 for r in ew if r["length"] == ln), sum(1 for r in ew if r["length"] == ln and certified(r))) for ln in sorted({r["length"] for r in M})}
    P(f"C1(2) MILD HARMFUL-PRIOR CONTROL, EVALUATION (once; seeds {sorted({r['seed'] for r in M})[0]}..{sorted({r['seed'] for r in M})[-1]}, {len({r['seed'] for r in M})} users, 3 lengths)")
    P(f"arm M (the frozen mild prior): {len(M)} fits ({len(usable)} with a usable Gaussian); worse than the single Gaussian on the held-out days: {len(worse)}; of those eligible for the certificate: {n} (required: at least {REQUIRED_EVAL})")
    P(f"E1 certified among the {n} eligible harmful fits: {k} (exact one-sided 95% upper bound on the certification rate {100 * upper:.0f}%) -> {e1}; by length (eligible harmful, certified): {by_len}")
    P(f"E0 arm T (the true prior): {len(T)} fits, eligible {t_elig}, certified {t_cert} -> {'informative' if e0 else 'NOT INFORMATIVE: the certificate certified nothing'}")
    P(f"E3 (reported): arm-M fits that score below the single Gaussian and are published after the certificate: {pub_worse} of {len(worse)}")
    json.dump(dict(M=len(M), usable=len(usable), worse=len(worse), eligible_and_worse=n, certified=k, upper=upper, E1=e1, T=len(T), T_eligible=t_elig, T_certified=t_cert, E0=e0, E3_published_and_worse=pub_worse, by_length=by_len),
              open(out_json, "w"), indent=1, default=float)
    return e1


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "dev" and len(a) == 4:
        dev(a[1], a[3])
        Path(a[2]).write_text("\n".join(OUT) + "\n")
    elif a and a[0] == "eval" and len(a) in (4, 5):
        evaluate([a[1]] + a[4:5], a[3])
        Path(a[2]).write_text("\n".join(OUT) + "\n")
    else:
        sys.exit(__doc__)
