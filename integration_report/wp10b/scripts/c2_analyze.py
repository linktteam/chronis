"""WP10B C2 analysis: the hard-stratum ceiling, oracle labels against the production fit (rules: tests/PREDECLARED_CRITERIA_WP10B.md section 5 and Part 2). usage:
  PYTHONPATH=. python3 integration_report/wp10b/scripts/c2_analyze.py dev  FITS.jsonl OUT.txt OUT.json
  PYTHONPATH=. python3 integration_report/wp10b/scripts/c2_analyze.py eval FITS.jsonl OUT.txt OUT.json
Per fit (one user, one window [0, t)): the ORACLE gate quantity = `lineage.separation_min` with the true labels of the window's days and the sample means of the true regimes (arm O); the same with the generator's
true means (arm G); the PRODUCTION gate quantity = `separation_min` of the production-selected restart (`wp9_replay_lib.prepared`, as the replay of WP9/WP10A). The gate passes at `separation_min >= H` (2.5)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integration_report" / "wp9" / "scripts"))

H = 2.5
MIN_DAYS = 3
SHARP = 0.95                                  # the product's sharpness floor (the consensus flag is ON)
BOOT, BOOT_SEED = 2000, 11999
STRATA = ("hard", "medium", "easy")
NOT_SERVABLE_LEVEL, GAP_LEVEL = 0.50, 0.20
OUT: list = []


def P(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)


# ------------------------------------------------------------------------------------------------- per-fit quantities
def oracle_row(raw, labels_true, means_t=None, K: int = 3) -> dict:
    """Arm O: true labels and the sample mean of each true regime's observed entries; arm G: true labels and the generator's per-day true means averaged over each regime's days of the window.
    A window is evaluable only if every one of the K true regimes has at least MIN_DAYS days in it and an observed value for every feature (otherwise no sample mean exists)."""
    from backbone.hssm.lineage import separation_min
    X = np.asarray(raw, dtype=float)
    lab = np.asarray(labels_true, dtype=int)
    counts = np.bincount(lab, minlength=K)[:K]
    out = dict(evaluable=False, counts=[int(c) for c in counts], sep_emp=float("nan"), sep_gen=float("nan"))
    if (counts < MIN_DAYS).any():
        return out
    means = []
    for k in range(K):
        sel = X[lab == k]
        if np.isnan(sel).all(axis=0).any():
            return out
        means.append(np.nanmean(sel, axis=0))
    out["sep_emp"] = float(separation_min(np.array(means), X, lab))
    if means_t is not None:
        gm = np.array([np.asarray(means_t, dtype=float)[lab == k].mean(axis=0) for k in range(K)])
        out["sep_gen"] = float(separation_min(gm, X, lab))
    out["evaluable"] = True
    return out


def production_row(prep) -> dict:
    """`prep` = `wp9_replay_lib.prepared(rec, None)` (None when no usable accepted restart). `sep` is `separation_min` of the selected restart with ITS labels; `flag` = FIT_UNSTABLE by the production consensus or a
    sharpness below the floor (the product setting of the replay)."""
    from backbone.hssm.lineage import separation_min
    if prep is None:
        return dict(usable=False, sep=float("nan"), flag=None)
    r8, cons = prep["r8"], prep["cons"]
    flag = bool(cons["fit_unstable"]) or (r8["max_post"] < SHARP)
    sep = float(separation_min(np.array(r8["raw_means"], dtype=float), prep["raw"], np.array(r8["labels_fit"], dtype=int)))
    return dict(usable=True, sep=sep, flag=flag)


_streams: dict = {}


def refit_rows(records: list) -> list:
    import wp9_fit_lib as F
    import wp9_replay_lib as RL
    rows = []
    for rec in records:
        j = rec["job"]
        if not rec.get("ok"):
            rows.append(dict(user=j["user"], stratum=j["stratum"], end=j["end"], oracle=dict(evaluable=False, counts=None, sep_emp=float("nan"), sep_gen=float("nan")), prod=production_row(None), failed=True))
            continue
        key = (j["user"], j["stratum"])
        if key not in _streams:
            _streams[key] = F.stream_for(j)
        st = _streams[key]
        w = st.window(j["start"], j["end"])
        rows.append(dict(user=j["user"], stratum=j["stratum"], end=j["end"], oracle=oracle_row(w["raw"], w["labels"], st.truth["means_t"][j["start"]:j["end"]]), prod=production_row(RL.prepared(rec, None)), failed=False))
    return rows


# ------------------------------------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.959964):
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def passes(sep) -> bool:
    return bool(math.isfinite(sep) and sep >= H)


def user_table(rows: list) -> list:
    """One entry per user over its EVALUABLE windows (oracle evaluable): counts n, oracle passes o, generator-mean passes g, production separation passes p, production gate passes (separation and not flagged) q."""
    by: dict = {}
    for r in rows:
        if not r["oracle"]["evaluable"]:
            continue
        u = by.setdefault(r["user"], dict(user=r["user"], n=0, o=0, g=0, p=0, q=0))
        u["n"] += 1
        u["o"] += passes(r["oracle"]["sep_emp"])
        u["g"] += passes(r["oracle"]["sep_gen"])
        u["p"] += bool(r["prod"]["usable"] and passes(r["prod"]["sep"]))
        u["q"] += bool(r["prod"]["usable"] and passes(r["prod"]["sep"]) and not r["prod"]["flag"])
    return [by[k] for k in sorted(by)]


def ratio(users: list, num: str, den: str = "n") -> float:
    d = sum(u[den] for u in users)
    return float("nan") if d == 0 else sum(u[num] for u in users) / d


def boot_users(users: list, stat, B: int = BOOT, seed: int = BOOT_SEED):
    """User-level bootstrap (resample users with replacement; every user brings all its windows): (point, lo, hi), percentile 95%."""
    point = stat(users)
    if len(users) < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, len(users), len(users))
        v = stat([users[i] for i in idx])
        if math.isfinite(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def verdict_hard(pO: float, pO_hi: float, gap: float, gap_lo: float) -> str:
    """Section 5.3 (declared in advance). pO = oracle pass fraction (point, upper end of its interval), gap = pO - pP (point, lower end of its paired interval)."""
    if pO < NOT_SERVABLE_LEVEL:
        return "NOT SERVABLE BY DESIGN (demonstrated: the upper end of the interval is below 0.50)" if pO_hi < NOT_SERVABLE_LEVEL else "NOT SERVABLE BY DESIGN (on the point estimate: the interval reaches 0.50)"
    if gap >= GAP_LEVEL and gap_lo > 0:
        return "NEEDS BETTER FITTING (WP8b candidate)"
    return "MIXED / NOT DECIDED"


def q3(x):
    x = [v for v in x if math.isfinite(v)]
    return (float("nan"),) * 3 if not x else tuple(float(np.percentile(x, p)) for p in (25, 50, 75))


def summarize(rows: list) -> dict:
    out = {}
    for s in STRATA:
        R = [r for r in rows if r["stratum"] == s]
        if not R:
            continue
        users = user_table(R)
        E = [r for r in R if r["oracle"]["evaluable"]]
        nE = len(E)
        kO = sum(passes(r["oracle"]["sep_emp"]) for r in E)
        pO = boot_users(users, lambda u: ratio(u, "o"))
        pG = boot_users(users, lambda u: ratio(u, "g"))
        pP = boot_users(users, lambda u: ratio(u, "p"))
        pQ = boot_users(users, lambda u: ratio(u, "q"))
        gap = boot_users(users, lambda u: ratio(u, "o") - ratio(u, "p"))
        gapq = boot_users(users, lambda u: ratio(u, "o") - ratio(u, "q"))
        n_users = len(users)
        ever_o = sum(1 for u in users if u["o"] > 0)
        ever_q = sum(1 for u in users if u["q"] > 0)
        prod_all = [r for r in R]
        out[s] = dict(refits=len(R), evaluable=nE, non_evaluable=len(R) - nE, failed_fits=sum(1 for r in R if r.get("failed")), users=n_users, oracle_pass=kO, wilson_naive_O=wilson(kO, nE),
                      pO=pO, pG=pG, pP_sep=pP, pP_gate=pQ, gap_sep=gap, gap_gate=gapq, users_oracle_ever=ever_o, users_production_gate_ever=ever_q,
                      sep_oracle_q=q3([r["oracle"]["sep_emp"] for r in E]), sep_gen_q=q3([r["oracle"]["sep_gen"] for r in E]), sep_prod_q=q3([r["prod"]["sep"] for r in E if r["prod"]["usable"]]),
                      prod_unusable_in_evaluable=sum(1 for r in E if not r["prod"]["usable"]), prod_usable_all=sum(1 for r in prod_all if r["prod"]["usable"]))
    return out


def fmt3(t):
    """(point, lo, hi) -> 'point [lo, hi]'."""
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def fmtq(t):
    """(Q1, median, Q3) -> 'median [Q1, Q3]'."""
    return f"{t[1]:.2f} [{t[0]:.2f}, {t[2]:.2f}]"


def report(res: dict, title: str, apply_verdict: bool):
    P(title)
    P(f"gate: separation_min >= {H}; evaluable refit = every one of the 3 true regimes has >= {MIN_DAYS} days and an observed value per feature; intervals = user-level bootstrap ({BOOT} resamples, seed {BOOT_SEED}), naive Wilson over refits for O")
    for s in STRATA:
        if s not in res:
            continue
        d = res[s]
        P(f"\n[{s}] refits {d['refits']} ({d['users']} users), evaluable {d['evaluable']}, non-evaluable {d['non_evaluable']}, failed fits {d['failed_fits']}; production unusable among evaluable {d['prod_unusable_in_evaluable']}")
        P(f"   (a) separation in gate units, median [Q1, Q3] over evaluable refits: oracle {fmtq(d['sep_oracle_q'])} | generator means {fmtq(d['sep_gen_q'])} | production (usable fits) {fmtq(d['sep_prod_q'])}")
        P(f"   (b) share of evaluable refits passing the gate: oracle {fmt3(d['pO'])} (naive Wilson {d['wilson_naive_O'][0]:.3f}..{d['wilson_naive_O'][1]:.3f}); generator means {fmt3(d['pG'])}; production, separation only {fmt3(d['pP_sep'])}; production, separation and not flagged {fmt3(d['pP_gate'])}")
        P(f"   (c) gap oracle - production (separation only) {fmt3(d['gap_sep'])}; oracle - production (separation and not flagged) {fmt3(d['gap_gate'])}")
        P(f"   (d) users whose oracle passes at least once: {d['users_oracle_ever']} of {d['users']}; users whose production fit passes the full gate at least once: {d['users_production_gate_ever']} of {d['users']}")
    if apply_verdict and "hard" in res:
        h = res["hard"]
        v = verdict_hard(h["pO"][0], h["pO"][2], h["gap_sep"][0], h["gap_sep"][1])
        P(f"\nINTERPRETATION RULE (section 5.3, hard stratum, evaluation): {v}")
        return v
    P("\n(development: no claim; the interpretation rule is applied to the evaluation set only)")
    return None


def main(mode: str, fits: str, out_txt: str, out_json: str):
    import wp9_fit_lib as F
    rows = refit_rows(F.load_records(fits))
    res = summarize(rows)
    v = report(res, f"WP10B C2 HARD-STRATUM CEILING, {'EVALUATION (read once)' if mode == 'eval' else 'DEVELOPMENT'}: {fits}", mode == "eval")
    Path(out_txt).write_text("\n".join(OUT) + "\n")
    json.dump(dict(mode=mode, summary=res, verdict=v), open(out_json, "w"), indent=1, default=float)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 4 and a[0] in ("dev", "eval"):
        main(*a)
    else:
        sys.exit(__doc__)
