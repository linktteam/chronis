"""WP10A A6 part 1 analysis: B4 (levels and statistics: tests/PREDECLARED_CRITERIA_WP10A.md Part 6 section 17). usage:
  PYTHONPATH=. python3 integration_report/wp10a/scripts/a6_b4_analyze.py FITS.jsonl OUT.txt OUT.json
The unit is the user (the mean of the user's held-out days); per condition the mean over published users with a user-level bootstrap 95% interval (2000 resamples, default_rng(0)); the difference to the
favourable control C0 with a bootstrap interval. A condition is read only with at least MIN_READ published users. Published = not withheld after the certificate."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

NOMINAL1, NOMINAL2 = math.erf(1.0 / math.sqrt(2.0)), math.erf(2.0 / math.sqrt(2.0))            # 0.6827, 0.9545
BAND, WORSE, MIN_READ, DRAWS = 0.10, 0.10, 6, 2000
CONDITIONS = ("C0", "C1", "C2", "C3")
NAMES = dict(C0="C0 favourable control", C1="C1 off-centre prior", C2="C2 atypical users", C3="C3 misspecified fit")
OUT: list = []


def P(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)


def load(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    ok = {}
    for r in recs:
        if r.get("ok"):
            ok[r["seed"]] = r
    return [ok[s] for s in sorted(ok)], sum(1 for r in recs if not r.get("ok"))


def is_published(r) -> bool:
    return bool(r.get("ok")) and not r["withheld"]


def user_matrix(recs, key="heldout"):
    """(n_users, 2 levels x 2 dims) = [c1_d0, c1_d1, c2_d0, c2_d1] per user, plus z2 per dim."""
    c = np.array([list(r[key]["c1"]) + list(r[key]["c2"]) for r in recs], dtype=float).reshape(len(recs), -1)
    z = np.array([list(r[key]["z2"]) for r in recs], dtype=float).reshape(len(recs), -1)
    return c, z


def boot_mean(m, rng, draws=DRAWS):
    """Bootstrap distribution of the column means of m (users resampled with replacement)."""
    n = len(m)
    idx = rng.integers(0, n, size=(draws, n))
    return m[idx].mean(axis=1)


def interval(b):
    return np.percentile(b, 2.5, axis=0), np.percentile(b, 97.5, axis=0)


def in_band(c) -> bool:
    """All four coverages (1 sd and 2 sd, both latent dimensions) within BAND of their nominal levels (point estimates)."""
    c = np.asarray(c, dtype=float)
    return bool(abs(c[0] - NOMINAL1) <= BAND and abs(c[1] - NOMINAL1) <= BAND and abs(c[2] - NOMINAL2) <= BAND and abs(c[3] - NOMINAL2) <= BAND)


def worse_than(c, c0) -> bool:
    """2-sd coverage lower than the control's by more than WORSE in either latent dimension."""
    c, c0 = np.asarray(c, dtype=float), np.asarray(c0, dtype=float)
    return bool((c0[2] - c[2]) > WORSE or (c0[3] - c[3]) > WORSE)


def misses(c) -> list:
    """Signed distance to the nominal levels: [1sd d0, 1sd d1, 2sd d0, 2sd d1]."""
    c = np.asarray(c, dtype=float)
    return [float(c[0] - NOMINAL1), float(c[1] - NOMINAL1), float(c[2] - NOMINAL2), float(c[3] - NOMINAL2)]


def cond_stats(recs, cond, published_only=True):
    rs = [r for r in recs if r["condition"] == cond]
    use = [r for r in rs if (is_published(r) if published_only else True)]
    out = dict(condition=cond, ok_users=len(rs), published=sum(1 for r in rs if is_published(r)), withheld=sum(1 for r in rs if r["withheld"]), used=len(use),
               readable=len(use) >= MIN_READ, variants={})
    for v in sorted({r["variant"] for r in rs}):
        rv = [r for r in rs if r["variant"] == v]
        out["variants"][v] = dict(users=len(rv), published=sum(1 for r in rv if is_published(r)), withheld=sum(1 for r in rv if r["withheld"]))
    if use:
        c, z = user_matrix(use)
        out["mean"] = c.mean(axis=0).tolist()
        out["z2"] = z.mean(axis=0).tolist()
        if len(use) >= 2:
            lo, hi = interval(boot_mean(c, np.random.default_rng(0)))
            out["lo"], out["hi"] = lo.tolist(), hi.tolist()
        out["matrix"] = c
    return out


def delta_vs_c0(st, st0):
    """Bootstrap interval of the difference (condition - C0) of the mean coverages; both groups resampled; None when a group has fewer than two users."""
    if "matrix" not in st or "matrix" not in st0 or len(st["matrix"]) < 2 or len(st0["matrix"]) < 2:
        return None
    rng = np.random.default_rng(0)
    d = boot_mean(st["matrix"], rng) - boot_mean(st0["matrix"], rng)
    lo, hi = interval(d)
    return dict(mean=(st["matrix"].mean(axis=0) - st0["matrix"].mean(axis=0)).tolist(), lo=lo.tolist(), hi=hi.tolist())


def fmt_cov(v):
    return f"1sd {v[0]:.2f} / {v[1]:.2f}, 2sd {v[2]:.2f} / {v[3]:.2f}"


def block(recs, published_only: bool, label: str) -> dict:
    stats = {c: cond_stats(recs, c, published_only) for c in CONDITIONS}
    P(f"\n{label} (nominal: 1 sd {NOMINAL1:.4f}, 2 sd {NOMINAL2:.4f}; each cell is dimension 0 / dimension 1; the unit is the user)")
    for c in CONDITIONS:
        s = stats[c]
        head = f"  {NAMES[c]}: users {s['ok_users']}, published {s['published']}, withheld {s['withheld']}; used {s['used']}"
        if not s["used"]:
            P(head + " -> nothing to read: no published fit" if published_only else head + " -> nothing to read")
            continue
        if not s["readable"]:
            P(head + f" -> too few published fits to read (< {MIN_READ}); for the record: " + fmt_cov(s["mean"]))
            continue
        lo = ", ".join(f"{x:.2f}" for x in s["lo"]); hi = ", ".join(f"{x:.2f}" for x in s["hi"])
        P(head + ": " + fmt_cov(s["mean"]) + f"; mean z^2 {s['z2'][0]:.1f} / {s['z2'][1]:.1f}; bootstrap 95% lower [{lo}] upper [{hi}] (order 1sd d0, 1sd d1, 2sd d0, 2sd d1)")
        m = misses(s["mean"])
        P(f"      in band (+-{BAND:.2f} of both nominal levels in both dimensions): {'YES' if in_band(s['mean']) else 'NO'}; signed distance to nominal: " + ", ".join(f"{x:+.2f}" for x in m))
    c0 = stats["C0"]
    for c in CONDITIONS[1:]:
        s = stats[c]
        if not (s["readable"] and c0["readable"]):
            P(f"  {NAMES[c]} against C0: not read (a group has fewer than {MIN_READ} published users)")
            continue
        d = delta_vs_c0(s, c0)
        P(f"  {NAMES[c]} against C0: difference of the 2-sd coverage {d['mean'][2]:+.2f} / {d['mean'][3]:+.2f} (bootstrap 95% [{d['lo'][2]:+.2f}, {d['hi'][2]:+.2f}] / [{d['lo'][3]:+.2f}, {d['hi'][3]:+.2f}]) -> "
          f"{'WORSE than C0 (by more than ' + format(WORSE, '.2f') + ')' if worse_than(s['mean'], c0['mean']) else 'not worse than C0 by more than ' + format(WORSE, '.2f')}")
    for c in CONDITIONS:
        for v, x in stats[c]["variants"].items():
            P(f"      {c}/{v}: users {x['users']}, published {x['published']}, withheld {x['withheld']}")
    return {c: {k: v for k, v in s.items() if k != "matrix"} for c, s in stats.items()}


def withhold_reasons(recs):
    out = {}
    for r in recs:
        if r["withheld"]:
            k = (r.get("withheld_reason") or "withheld").split(":")[0]
            out.setdefault(r["condition"], {})
            out[r["condition"]][k] = out[r["condition"]].get(k, 0) + 1
    return out


def main(fits, out_txt, out_json):
    recs, failed = load(fits)
    P(f"B4 ANALYSIS (WP10A A6 part 1): {len(recs)} users fitted ({failed} failed fits excluded); production dual fit with the prior (n0 = 5), certify_prior_anchored = True; generated users, one training length (about 90 days)")
    P(f"certified (CERTIFIED_PRIOR_ANCHORED) fits: {sum(1 for r in recs if r.get('certified'))}; withheld fits: {sum(1 for r in recs if r['withheld'])}; withhold reasons by condition: {withhold_reasons(recs)}")
    pub = block(recs, True, "HEADLINE, published fits only")
    allf = block(recs, False, "SECONDARY, all fits (published or not; labelled: a withheld fit has no user-facing uncertainty)")
    P("\nReading: coverage is measured against the true fast state mapped into the fitted basis (N17 affine map, in sample); N17 stays an expected failure whatever this shows; confidence_display_gate stays 'early_estimate'.")
    P("Not run: the parametric-bootstrap comparison of the design note; K one too large (the shared prior has K = 2). Real users: UNKNOWN.")
    json.dump(dict(users=len(recs), failed=failed, published=pub, all=allf), open(out_json, "w"), indent=1, default=float)
    Path(out_txt).write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
