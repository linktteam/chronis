"""WP9 B1 analysis (design: tests/PREDECLARED_CRITERIA_WP9.md section 16). usage:
  PYTHONPATH=. python3 integration_report/wp9/scripts/b1_analyze.py dev  FITS.jsonl OUT.txt [OUT.json]    # control validity and the declared rule for z
  PYTHONPATH=. python3 integration_report/wp9/scripts/b1_analyze.py eval FITS.jsonl OUT.txt [OUT.json]    # E1 to E6 at the z frozen in backbone/hssm/config.py
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp8" / "scripts"))
import lineage_study_lib as S                                                # noqa: E402

from backbone.hssm import config as C                                        # noqa: E402
from tests.utils.n18v2 import paired_stats                                   # noqa: E402

LENGTHS = (30, 60, 120)
FIT_ARMS = ("shrink5", "scrambled", "shifted", "personal")
CONTROLS = ("scrambled", "shifted", "personal")
Z_GRID = (1.0, 2.0, 3.0)
GAIN, UNSTABLE = "gain_not_demonstrated", "parameters_unstable"


def load(path):
    out = {}
    for ln in open(path):
        ln = ln.strip()
        if ln:
            d = json.loads(ln)
            out[(d["seed"], d["length"], d["arm"])] = d
    return out


def seeds_of(recs):
    return sorted({k[0] for k in recs})


def eligible(r):
    """The fit was withheld by EM for non-convergence only (nothing else), i.e. certification could decide it."""
    reason = r.get("em_reason") or ""
    return bool(r.get("withheld_em")) and reason.startswith("fit_state_not_converged") and ";" not in reason


def certified_at(r, z, tau=C.COLD_START_CERT_TAU):
    """Offline verdict at margin z from the stored statistics; False when the decision was never reached (too few rows, refit failure, other reasons...)."""
    c = r.get("certification") or {}
    if not eligible(r) or c.get("mean_gain") is None or c.get("se") is None or c.get("stability") is None:
        return False
    if set(c.get("reasons", [])) - {GAIN, UNSTABLE}:
        return False
    return bool(c["mean_gain"] - z * c["se"] > 0.0 and c["stability"] <= tau)


def rate(k, n):
    return float("nan") if n == 0 else k / n


def fmt(k, n):
    lo, hi = S.wilson(k, n)
    return f"{k}/{n} = {100 * rate(k, n):.0f}% (Wilson {100 * lo:.0f}-{100 * hi:.0f}%)" if n else f"{k}/{n}"


def dev_report(recs, out_txt, out_json):
    seeds = seeds_of(recs)
    L = [f"B1 development: {len(seeds)} seeds {seeds[0]}..{seeds[-1]}, lengths {LENGTHS}", ""]
    valid = {}
    for name in ("scrambled", "shifted"):
        d = [recs[(s, ln, f"shared_{name}")]["mean"] - recs[(s, ln, "shared_true")]["mean"] for s in seeds for ln in LENGTHS if (s, ln, f"shared_{name}") in recs and (s, ln, "shared_true") in recs]
        d = [x for x in d if math.isfinite(x)]
        m = sum(d) / len(d) if d else float("nan")
        valid[name] = m <= -1.0
        L.append(f"control validity, shared-only score under the {name} prior minus under the true prior: mean {m:+.3f} nats/step over {len(d)} (seed, length) cells -> {'valid (>= 1 nat worse)' if valid[name] else 'NOT a valid control'}")
    L.append("")
    table = []
    for z in Z_GRID:
        row = dict(z=z)
        for arm in FIT_ARMS:
            fits = [r for (s, ln, a), r in recs.items() if a == arm]
            el = [r for r in fits if eligible(r)]
            row[arm] = (sum(certified_at(r, z) for r in el), len(el))
        ctrl_k = sum(row[a][0] for a in CONTROLS); ctrl_n = sum(row[a][1] for a in CONTROLS)
        row["controls"] = (ctrl_k, ctrl_n)
        table.append(row)
        L.append(f"z = {z}: shrink5 certified {fmt(*row['shrink5'])}; controls pooled {fmt(ctrl_k, ctrl_n)} (scrambled {row['scrambled'][0]}/{row['scrambled'][1]}, shifted {row['shifted'][0]}/{row['shifted'][1]}, personal {row['personal'][0]}/{row['personal'][1]})")
    chosen, met = None, False
    for row in table:
        sk, sn = row["shrink5"]; ck, cn = row["controls"]
        if cn > 0 and rate(ck, cn) <= 0.10 and rate(ck, cn) <= 0.5 * rate(sk, sn):
            chosen, met = row["z"], True
            break
    if chosen is None:
        chosen = 3.0
    L += ["", f"declared rule: smallest z in {Z_GRID} with controls <= 10% and <= half the shrink5 share -> z = {chosen} ({'rule met' if met else 'NO z met the rule: z = 3 taken, shortfall reported'})"]
    cons = [(s, ln) for (s, ln, a), r in recs.items() if a == "shrink5" and eligible(r) and (r.get("certification") or {}).get("mean_gain") is not None
            and (certified_at(r, C.COLD_START_CERT_Z) != (r["certification"]["state"] == "CERTIFIED_PRIOR_ANCHORED"))]
    L.append(f"consistency of the offline verdict at the config z ({C.COLD_START_CERT_Z}) with the production state in shrink5 fits: mismatches {cons}")
    for ln in LENGTHS:
        fits = [r for (s, l2, a), r in recs.items() if a == "shrink5" and l2 == ln]
        L.append(f"  length {ln}: shrink5 EM-publishable {sum(not r['withheld_em'] for r in fits)}/{len(fits)}, eligible {sum(eligible(r) for r in fits)}, decision reached {sum((r.get('certification') or {}).get('mean_gain') is not None for r in fits)}, "
                 f"certified at z={chosen} {sum(certified_at(r, chosen) for r in fits)}; too few rows {sum('too_few_rows' in ((r.get('certification') or {}).get('reasons') or []) for r in fits)}")
    open(out_txt, "w").write("\n".join(L) + "\n")
    if out_json:
        json.dump(dict(chosen_z=chosen, rule_met=met, valid=valid, table=table), open(out_json, "w"), indent=1)
    print("\n".join(L))


def eval_report(recs, out_txt, out_json):
    z = C.COLD_START_CERT_Z
    seeds = seeds_of(recs)
    L = [f"B1 EVALUATION: seeds {seeds[0]}..{seeds[-1]} ({len(seeds)} users), lengths {LENGTHS}, certification margin z = {z}, stability level {C.COLD_START_CERT_TAU}", ""]
    ok = lambda r: r is not None and r.get("finite") and math.isfinite(r.get("mean", float("nan")))
    res = {}
    survive3, survive_cert = [], []
    for ln in LENGTHS:
        sh = {s: recs[(s, ln, "shrink5")] for s in seeds if (s, ln, "shrink5") in recs}
        g = {s: recs[(s, ln, "gauss")]["mean"] for s in seeds if (s, ln, "gauss") in recs}
        n = len(sh)
        em_pub = [s for s, r in sh.items() if not r["withheld_em"]]
        elig = [s for s, r in sh.items() if eligible(r)]
        cert = [s for s, r in sh.items() if (r.get("certification") or {}).get("state") == "CERTIFIED_PRIOR_ANCHORED"]
        pub = [s for s, r in sh.items() if r.get("published_after")]
        L.append(f"== length {ln}: {n} shrink5 fits")
        L.append(f"E1 EM-publishable {len(em_pub)}/{n}; eligible for certification {len(elig)}; certified {len(cert)} ({len(cert)}/{n} of all, {len(cert)}/{len(elig)} of eligible); publishable after {len(pub)}/{n}")
        reasons = {}
        for s, r in sh.items():
            for x in (r.get("certification") or {}).get("reasons", []):
                reasons[x] = reasons.get(x, 0) + 1
        L.append(f"   reasons recorded (a fit can have several): {reasons}")
        def gain(seed_list):
            return paired_stats({s: sh[s]["mean"] for s in seed_list if ok(sh[s])}, {s: g[s] for s in seed_list})
        gc, gp = gain(cert), gain(pub)
        def line(tag, st):
            judged = st["n"] >= 8
            return (f"{tag}: n = {st['n']}, gain {st['mean']:+.3f} (se {st['se']:.3f}), mean - 3 se {st['mean_minus_3se']:+.3f}, favouring {st['n_favour']}/{st['n']}" if st["n"] >= 2 else f"{tag}: n = {st['n']}") + ("" if judged else " (NOT judged: n < 8)")
        L.append("E2 " + line("certified fits", gc)); L.append("E3 " + line("publishable-after fits", gp))
        survive_cert.append(gc["n"] >= 8 and gc["mean_minus_3se"] > 0); survive3.append(gp["n"] >= 8 and gp["mean_minus_3se"] > 0)
        served = {s: (sh[s]["mean"] if sh[s].get("published_after") and ok(sh[s]) else g[s]) for s in seeds if s in g and s in sh}
        rc = paired_stats(served, {s: g[s] for s in served})
        L.append(f"E4 risk-coverage: coverage {len(pub)}/{n} = {100 * len(pub) / max(n, 1):.0f}%, gain on all {rc['n']} users {rc['mean']:+.3f} (se {rc['se']:.3f}), mean - 3 se {rc['mean_minus_3se']:+.3f}")
        res[ln] = dict(n=n, em_publishable=len(em_pub), eligible=len(elig), certified=len(cert), published_after=len(pub), cert_gain=gc, pub_gain=gp, risk_coverage=rc)
        for arm in CONTROLS:
            ar = {s: recs[(s, ln, arm)] for s in seeds if (s, ln, arm) in recs}
            el = [s for s, r in ar.items() if eligible(r)]
            ce = [s for s, r in ar.items() if (r.get("certification") or {}).get("state") == "CERTIFIED_PRIOR_ANCHORED"]
            L.append(f"E5 {arm}: eligible {len(el)}, certified {len(ce)}/{len(ar)}" + (f"; held-out gain of certified fits: " + ", ".join(f"{ar[s]['mean'] - g[s]:+.2f}" for s in ce) if ce else ""))
            res[ln][arm] = dict(eligible=len(el), certified=len(ce), n=len(ar))
        L.append("")
    allsh = [r for (s, ln, a), r in recs.items() if a == "shrink5"]
    k_s = sum((r.get("certification") or {}).get("state") == "CERTIFIED_PRIOR_ANCHORED" for r in allsh); n_se = sum(eligible(r) for r in allsh)
    L.append(f"E5 pooled over lengths: shrink5 certified {fmt(k_s, n_se)} of eligible")
    ctrl_ok = True
    for arm in CONTROLS:
        ar = [r for (s, ln, a), r in recs.items() if a == arm]
        k = sum((r.get("certification") or {}).get("state") == "CERTIFIED_PRIOR_ANCHORED" for r in ar); n = sum(eligible(r) for r in ar)
        met = n > 0 and rate(k, n) <= 0.10 and rate(k, n) <= 0.5 * rate(k_s, n_se)
        ctrl_ok &= met
        L.append(f"   {arm}: certified {fmt(k, n)} of eligible -> {'MET (<= 10% and <= half of shrink5)' if met else 'NOT MET'}" if n else f"   {arm}: no eligible fit -> NOT JUDGED")
    # E6 precision of the certificate over all arms
    cert_fits = [(s, ln, a) for (s, ln, a), r in recs.items() if a in FIT_ARMS and (r.get("certification") or {}).get("state") == "CERTIFIED_PRIOR_ANCHORED" and r.get("finite")]
    pos = sum(recs[k]["mean"] - recs[(k[0], k[1], "gauss")]["mean"] > 0 for k in cert_fits)
    L += ["", f"E6 certificate precision (all arms): certified fits with a positive held-out gain over the single Gaussian {fmt(pos, len(cert_fits))}"]
    sent = "survives" if all(survive3) else "does not survive"
    L += ["", f"The cold-start gain on publishable-after fits {sent} (pre-declared sentence: every length judged with n >= 8 and every mean - 3 se > 0: {survive3}); on certified fits only: {survive_cert}",
          f"Net deployment gain (risk-coverage) shown: {all(res[ln]['risk_coverage']['mean_minus_3se'] > 0 for ln in LENGTHS)}; controls criterion: {'MET' if ctrl_ok else 'NOT MET'}"]
    open(out_txt, "w").write("\n".join(L) + "\n")
    if out_json:
        json.dump(dict(z=z, per_length=res, controls_met=bool(ctrl_ok), survives=all(survive3)), open(out_json, "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    mode, fits, out = sys.argv[1], sys.argv[2], sys.argv[3]
    js = sys.argv[4] if len(sys.argv) > 4 else None
    (dev_report if mode == "dev" else eval_report)(load(fits), out, js)
