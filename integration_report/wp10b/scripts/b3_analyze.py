"""WP10B B3 (and the data of the B4 memo): analysis of the stored records of `b3_run.py` by the rules of the dated B3 part of tests/PREDECLARED_CRITERIA_WP10B.md. usage:
  PYTHONPATH=. python3.12 integration_report/wp10b/scripts/b3_analyze.py S0_JSONL BREAKER_JSONL OUT.txt OUT.json [--ext S0_EXT_JSONL]
Truth of one fit (held-out test days, paired difference to the single Gaussian, HAC standard error): harmful = mean < 0, good = mean > 0 (point estimates, both conservative for the claim they test); the strict versions
(mean -/+ 1.645 se) are reported. A fit is certified iff its certificate state is the WP9 `CERTIFIED_PRIOR_ANCHORED` (the constant name is reused; the fits here have no prior). Refusal verdicts use the exact one-sided 95% bound."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
from backbone.hssm import cold_start_cert as C  # noqa: E402

N_HARM_MIN = 14                 # exact one-sided 95% bound 19.3% with zero certified (the WP10A A6 level)
N_GOOD_MIN = 12
SHARE_LEVEL = 0.50              # useful share of good eligible fits
PUBLISHED_LEVEL = 0.60          # B4: published share (converged-publishable or certified) of easy + medium users
Z_STRICT = 1.645
Z_B4 = 3.0
Z95 = 1.96


def load(path) -> list:
    return [json.loads(l) for l in open(path) if l.strip()]


def exact_upper(c: int, n: int, conf: float = 0.95) -> float | None:
    """Clopper-Pearson one-sided upper bound of the rate when c of n are certified (c = 0: 1 - (1 - conf) ** (1 / n))."""
    if n <= 0:
        return None
    return 1.0 if c >= n else float(beta.ppf(conf, c + 1, n - c))


def wilson(k: int, n: int, z: float = Z95) -> tuple:
    if n <= 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def truth(rec: dict) -> dict | None:
    d = np.asarray(rec.get("test_g_diff") or [], dtype=float)
    if len(d) < 3 or not np.all(np.isfinite(d)):
        return None
    mean, se = float(d.mean()), C.hac_se(d)
    ok = math.isfinite(se)
    return dict(mean=mean, se=se if ok else None, harmful=bool(mean < 0), good=bool(mean > 0), strict_harmful=bool(ok and mean + Z_STRICT * se < 0), strict_good=bool(ok and mean - Z_STRICT * se > 0))


def is_certified(rec: dict) -> bool:
    return (rec.get("certificate") or {}).get("state") == C.STATE_CERTIFIED


def refusal_verdict(c: int, n: int) -> str:
    if c >= 1:
        return "NOT MET"
    return "MET" if n >= N_HARM_MIN else "NOT DEMONSTRATED"


def share_verdict(k: int, n: int, level: float = SHARE_LEVEL) -> str:
    if n < N_GOOD_MIN:
        return "NOT DEMONSTRATED"
    lo, _ = wilson(k, n)
    if lo >= level:
        return "MET"
    return "MET on the point estimate" if k / n >= level else "NOT MET"


def refusal_block(recs: list, label: str) -> dict:
    """Among the fits the certificate was asked about (eligible, or forced for the breakers) whose held-out gain over the Gaussian is below 0: how many were certified."""
    rows = [(r, truth(r)) for r in recs if r.get("ok") and r.get("eligible", True)]
    h = [(r, t) for r, t in rows if t and t["harmful"]]
    c = sum(is_certified(r) for r, _ in h)
    hs = [(r, t) for r, t in h if t["strict_harmful"]]
    cs = sum(is_certified(r) for r, _ in hs)
    return dict(label=label, n_asked=len(rows), n_harmful=len(h), certified_among_harmful=c, upper95=exact_upper(c, len(h)), verdict=refusal_verdict(c, len(h)),
                n_strict_harmful=len(hs), certified_among_strict_harmful=cs, upper95_strict=exact_upper(cs, len(hs)),
                certified_all=sum(is_certified(r) for r, _ in rows), harmful_users=[r["job"].get("user", r["job"].get("seed")) for r, _ in h])


def share_block(recs: list, label: str) -> dict:
    rows = [(r, truth(r)) for r in recs if r.get("ok") and r.get("eligible", True)]
    g = [(r, t) for r, t in rows if t and t["good"]]
    k = sum(is_certified(r) for r, _ in g)
    gs = [(r, t) for r, t in rows if t and t["strict_good"]]
    ks = sum(is_certified(r) for r, _ in gs)
    return dict(label=label, n_asked=len(rows), n_good=len(g), certified_among_good=k, share=(k / len(g)) if g else None, wilson=wilson(k, len(g)), verdict=share_verdict(k, len(g)),
                n_strict_good=len(gs), certified_among_strict_good=ks, wilson_strict=wilson(ks, len(gs)))


def blind_harmful_count(recs: list) -> int:
    """The count that decides the extension: eligible fits whose held-out gain is below 0, read WITHOUT the certificate."""
    return sum(1 for r in recs if r.get("ok") and r.get("eligible") and (truth(r) or {}).get("harmful"))


def needs_extension(recs: list) -> bool:
    return blind_harmful_count(recs) < N_HARM_MIN


def paired_users(recs: list, key: str) -> dict:
    """One value per user: the mean of the stored paired test-day differences; mean over users and the mean - 3 * paired standard error (across users, ddof 1)."""
    v = np.array([float(np.mean(r[key])) for r in recs if r.get("ok") and r.get(key)], dtype=float)
    if len(v) < 3:
        return dict(n=int(len(v)), mean=None, se=None, lower3=None)
    se = float(v.std(ddof=1) / math.sqrt(len(v)))
    return dict(n=int(len(v)), mean=float(v.mean()), se=se, lower3=float(v.mean() - Z_B4 * se))


def published(rec: dict) -> bool:
    d = rec.get("dual") or {}
    return bool(rec.get("ok") and (d.get("publishable") or is_certified(rec)))


def b4_block(recs: list) -> dict:
    em = [r for r in recs if r["job"]["stratum"] in ("easy", "medium")]
    n, k = len(em), sum(published(r) for r in em)
    pub_only = [r for r in em if published(r)]
    lo, hi = wilson(k, n)
    verdict = "NOT DEMONSTRATED" if n == 0 else ("MET" if lo >= PUBLISHED_LEVEL else ("MET on the point estimate" if k / n >= PUBLISHED_LEVEL else "NOT MET"))
    bb = paired_users(em, "test_b_diff")
    beats = bool(bb["lower3"] is not None and bb["lower3"] > 0)
    return dict(n_users=n, published=k, published_share=(k / n) if n else None, wilson=(lo, hi), published_verdict=verdict, dual_minus_baseline_all=bb, beats_baseline_3se=beats, dual_minus_baseline_published=paired_users(pub_only, "test_b_diff"),
                dual_minus_gauss_all=paired_users(em, "test_g_diff"), rule_met=bool(beats and verdict in ("MET", "MET on the point estimate")), rule_met_strict=bool(beats and verdict == "MET"))


def stratum_table(recs: list) -> dict:
    out = {}
    for s in ("hard", "medium", "easy"):
        rs = [r for r in recs if r["job"]["stratum"] == s]
        out[s] = dict(n=len(rs), ok=sum(bool(r.get("ok")) for r in rs), converged_publishable=sum(bool((r.get("dual") or {}).get("publishable")) for r in rs), eligible=sum(bool(r.get("eligible")) for r in rs),
                      certified=sum(is_certified(r) for r in rs), published=sum(published(r) for r in rs), dual_minus_gauss=paired_users(rs, "test_g_diff"), dual_minus_baseline=paired_users(rs, "test_b_diff"))
    return out


def baseline_margin_block(recs: list) -> dict:
    """Reported, not a verdict: certified fits that ALSO beat the baseline refits by the margin (certificate `baseline.beats_baseline`)."""
    cert = [r for r in recs if is_certified(r)]
    both = [r for r in cert if ((r["certificate"].get("baseline") or {}).get("beats_baseline"))]
    return dict(certified=len(cert), certified_and_beats_baseline_refits=len(both))


def analyse(s0: list, breaker: list) -> dict:
    out = dict(s0_n_jobs=len(s0), s0_failed=sum(not r.get("ok") for r in s0), blind_harmful_count_s0=blind_harmful_count(s0), strata=stratum_table([r for r in s0 if r.get("ok")]),
               natural_refusal=refusal_block(s0, "natural S0 fits"), good_share_all=share_block(s0, "all strata"), good_share_easy_medium=share_block([r for r in s0 if r["job"]["stratum"] in ("easy", "medium")], "easy + medium"),
               baseline_margin=baseline_margin_block([r for r in s0 if r.get("ok")]), b4=b4_block(s0), breakers={})
    for e in ("stops", "B2", "real"):
        rs = [r for r in breaker if r["job"]["engine"] == e]
        for r in rs:
            r["eligible"] = True                                                      # the certificate was forced on every breaker fit
        if e == "real":
            out["breakers"][e] = dict(share=share_block(rs, "real engine, N18 generator, certificate forced"), n=len(rs), failed=sum(not r.get("ok") for r in rs))
        else:
            out["breakers"][e] = dict(refusal=refusal_block(rs, f"wrong engine {e}"), n=len(rs), failed=sum(not r.get("ok") for r in rs), certified_all=sum(is_certified(r) for r in rs))
    return out


def f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def render(a: dict) -> str:
    L = ["WP10B B3 (dual predictive certificate) and the data of the B4 memo (Python 3.12, repo pins; every number RUN)", ""]
    L.append(f"S0 users {a['s0_n_jobs']} (fits that failed to run {a['s0_failed']}); eligible harmful fits counted blind: {a['blind_harmful_count_s0']}")
    for s, t in a["strata"].items():
        L.append(f"  [{s}] fits {t['n']}, converged-publishable {t['converged_publishable']}, eligible {t['eligible']}, certified {t['certified']}, published {t['published']}; dual minus Gaussian {f(t['dual_minus_gauss']['mean'], 3)} (se {f(t['dual_minus_gauss']['se'], 3)}), dual minus baseline {f(t['dual_minus_baseline']['mean'], 3)} (se {f(t['dual_minus_baseline']['se'], 3)})")
    def ref(b):
        return (f"  {b['label']}: asked {b['n_asked']}, harmful {b['n_harmful']}, certified among harmful {b['certified_among_harmful']}, exact one-sided 95% upper bound {f(b['upper95'], 3)} -> {b['verdict']}; "
                f"strictly harmful {b['n_strict_harmful']}, certified {b['certified_among_strict_harmful']}, bound {f(b['upper95_strict'], 3)}; certified among all asked {b['certified_all']}")
    def sh(b):
        return (f"  {b['label']}: asked {b['n_asked']}, good {b['n_good']}, certified among good {b['certified_among_good']} = {f(b['share'])} (Wilson 95% {f(b['wilson'][0])}..{f(b['wilson'][1])}) -> {b['verdict']}; "
                f"strictly good {b['n_strict_good']}, certified {b['certified_among_strict_good']}")
    L += ["", "REFUSAL (the certificate must refuse harmful fits)", ref(a["natural_refusal"])]
    for e in ("stops", "B2"):
        L.append(ref(a["breakers"][e]["refusal"]) + f"; fits {a['breakers'][e]['n']} (failed {a['breakers'][e]['failed']})")
    L += ["", "USEFUL SHARE (good fits certified)", sh(a["good_share_all"]), sh(a["good_share_easy_medium"]), sh(a["breakers"]["real"]["share"]) + f"; fits {a['breakers']['real']['n']} (failed {a['breakers']['real']['failed']})"]
    bm = a["baseline_margin"]
    L += ["", f"Baseline margin (reported): certified {bm['certified']}, of which also ahead of the baseline refits by the margin {bm['certified_and_beats_baseline_refits']}"]
    b = a["b4"]
    L += ["", "B4 DATA (easy + medium users)", f"  published (converged-publishable or certified) {b['published']} of {b['n_users']} = {f(b['published_share'])} (Wilson {f(b['wilson'][0])}..{f(b['wilson'][1])}) vs level {PUBLISHED_LEVEL} -> {b['published_verdict']}",
          f"  dual minus baseline engine, held-out per user, all: mean {f(b['dual_minus_baseline_all']['mean'], 3)} se {f(b['dual_minus_baseline_all']['se'], 3)} lower (3 se) {f(b['dual_minus_baseline_all']['lower3'], 3)} (n {b['dual_minus_baseline_all']['n']}) -> beats by 3 se: {b['beats_baseline_3se']}",
          f"  published subset: mean {f(b['dual_minus_baseline_published']['mean'], 3)} lower (3 se) {f(b['dual_minus_baseline_published']['lower3'], 3)} (n {b['dual_minus_baseline_published']['n']}); dual minus Gaussian all: mean {f(b['dual_minus_gauss_all']['mean'], 3)} lower (3 se) {f(b['dual_minus_gauss_all']['lower3'], 3)}",
          f"  publish rule (beats baseline by 3 se AND published share at the level): {'MET' if b['rule_met_strict'] else ('MET on the point estimate' if b['rule_met'] else 'NOT MET')}"]
    return "\n".join(L)


def main() -> None:
    args = sys.argv[1:]
    ext = None
    if "--ext" in args:
        i = args.index("--ext")
        ext = args[i + 1]
        del args[i:i + 2]
    s0_path, br_path, out_txt, out_json = args
    s0 = load(s0_path) + (load(ext) if ext else [])
    a = analyse(s0, load(br_path))
    txt = render(a)
    Path(out_txt).write_text(txt + "\n")
    Path(out_json).write_text(json.dumps(a, indent=1, default=lambda o: None if o is None else str(o)))
    print(txt)


if __name__ == "__main__":
    main()
