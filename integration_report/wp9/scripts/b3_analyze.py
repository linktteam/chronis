"""WP9 B3 analysis (design: tests/PREDECLARED_CRITERIA_WP9.md Part 4). usage:
  PYTHONPATH=. python3 integration_report/wp9/scripts/b3_analyze.py dev  FITS.jsonl OUT.txt [OUT.json]   # applies the declared grid rule, prints the chosen thresholds
  PYTHONPATH=. python3 integration_report/wp9/scripts/b3_analyze.py eval FITS.jsonl OUT.txt [OUT.json]   # the declared levels at the thresholds frozen in backbone/hssm/config.py
The assessment is the PRODUCTION function `backbone.hssm.k_adequacy.assess_k_adequacy`, applied offline to the stored (selected K, BIC by K, regime means and variances, A1 flag)."""
import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp8" / "scripts"))
import lineage_study_lib as S                                                # noqa: E402

from backbone.hssm import config as C                                        # noqa: E402
from backbone.hssm.k_adequacy import assess_k_adequacy                       # noqa: E402

D0 = (0.5, 0.75, 1.0, 1.5, 2.0)
DELTA = (0, 3, 6, 10, 20)
CANDIDATES = (2, 3, 4, 5)


def load(path):
    out = []
    for ln in open(path):
        ln = ln.strip()
        if ln:
            d = json.loads(ln)
            if d.get("ok"):
                out.append(d)
    return out


def unsure(rec, d0, delta, sig3):
    bic = {int(k): v for k, v in rec["bic_by_k"].items()}
    return assess_k_adequacy(rec["selected_k"], bic, rec["mu"], rec["var"], rec.get("fit_unstable"), min_overlap_sd=d0, bic_margin=float(delta), candidates=CANDIDATES,
                             flag_largest=sig3)


def metrics(recs, d0, delta, sig3):
    flag = [(r, unsure(r, d0, delta, sig3)["k_unsure"]) for r in recs]
    wrong = [(r, f) for r, f in flag if r["selected_k"] != r["planted_k"]]
    right = [(r, f) for r, f in flag if r["selected_k"] == r["planted_k"]]
    easy_r = [(r, f) for r, f in right if r["design"] == "easy"]
    hard_r = [(r, f) for r, f in right if r["design"] == "hard"]
    n = lambda xs: sum(1 for _, f in xs if f)
    return dict(d0=d0, delta=delta, sig3=sig3, n=len(flag), n_wrong=len(wrong), n_right=len(right), det=n(wrong), fa_all=n(right), fa_easy=n(easy_r), n_easy_right=len(easy_r), fa_hard=n(hard_r),
                n_hard_right=len(hard_r), tagged=n(flag))


def rate(k, n):
    return float("nan") if n == 0 else k / n


def pick(recs):
    """The declared development rule."""
    grid = [metrics(recs, d, dl, s3) for d, dl, s3 in product(D0, DELTA, (True, False))]
    ok = [g for g in grid if rate(g["fa_easy"], g["n_easy_right"]) <= 0.25]
    cap_met = bool(ok)
    pool = ok if ok else sorted(grid, key=lambda g: rate(g["fa_easy"], g["n_easy_right"]))[:1]
    key = (lambda g: (-rate(g["det"], g["n_wrong"]), rate(g["fa_all"], g["n_right"]), g["d0"], g["delta"], g["sig3"]))
    if not ok:
        return grid, pool[0], False
    return grid, sorted(pool, key=key)[0], cap_met


def fmt_ci(k, n):
    lo, hi = S.wilson(k, n)
    return f"{k}/{n} = {100 * rate(k, n):.1f}% (Wilson {100 * lo:.1f}-{100 * hi:.1f}%)"


def report_dev(recs, out_txt, out_json):
    grid, ch, cap_met = pick(recs)
    L = [f"B3 development, {len(recs)} fits (seeds 9531..9554, both designs)", ""]
    L.append(f"BIC-selected K correct: {sum(r['selected_k'] == r['planted_k'] for r in recs)}/{len(recs)}; easy {sum(r['selected_k'] == r['planted_k'] for r in recs if r['design'] == 'easy')}/"
             f"{sum(r['design'] == 'easy' for r in recs)}, hard {sum(r['selected_k'] == r['planted_k'] for r in recs if r['design'] == 'hard')}/{sum(r['design'] == 'hard' for r in recs)}")
    L.append("")
    L.append("d0    delta sig3 | detection(all wrong) | false alarm easy-right | false alarm all-right | tagged")
    for g in sorted(grid, key=lambda g: (g["d0"], g["delta"], not g["sig3"])):
        L.append(f"{g['d0']:<5} {g['delta']:<5} {str(g['sig3']):<5}| {g['det']:>3}/{g['n_wrong']:<3} = {100 * rate(g['det'], g['n_wrong']):5.1f}%    | {g['fa_easy']:>3}/{g['n_easy_right']:<3} = "
                 f"{100 * rate(g['fa_easy'], g['n_easy_right']):5.1f}%       | {g['fa_all']:>3}/{g['n_right']:<3} = {100 * rate(g['fa_all'], g['n_right']):5.1f}%      | {g['tagged']}")
    L += ["", f"25% easy false-alarm cap met by at least one grid point: {cap_met}", f"CHOSEN (declared rule): d0 = {ch['d0']}, delta = {ch['delta']}, signal (3) {'on' if ch['sig3'] else 'off'}",
          f"  detection {fmt_ci(ch['det'], ch['n_wrong'])}; false alarm easy {fmt_ci(ch['fa_easy'], ch['n_easy_right'])}; false alarm all right-K {fmt_ci(ch['fa_all'], ch['n_right'])}"]
    open(out_txt, "w").write("\n".join(L) + "\n")
    if out_json:
        json.dump(dict(chosen=ch, cap_met=cap_met, grid=grid), open(out_json, "w"), indent=1)
    print("\n".join(L[-4:]))


def report_eval(recs, out_txt, out_json):
    d0, delta, sig3 = C.K_ADEQUACY_MIN_SEPARATION, C.K_ADEQUACY_BIC_MARGIN, C.K_ADEQUACY_FLAG_LARGEST
    mism = [r["seed"] for r in recs if r.get("k_adequacy") is not None and bool(r["k_adequacy"]["k_unsure"]) != unsure(r, d0, delta, sig3)["k_unsure"]]
    m = metrics(recs, d0, delta, sig3)
    rows = [(r, unsure(r, d0, delta, sig3)) for r in recs]
    L = [f"B3 EVALUATION, {len(recs)} fits (seeds 9501..9524, both designs) at frozen thresholds d0 = {d0}, delta = {delta}, signal (3) {'on' if sig3 else 'off'}", ""]
    L.append(f"offline recomputation equals the production field `k_adequacy.k_unsure` in all stored fits: {not mism} (mismatches: {mism})")
    ok_k = lambda rs: sum(r["selected_k"] == r["planted_k"] for r, _ in rs)
    L.append(f"BIC-selected K correct: {ok_k(rows)}/{len(rows)}; easy {ok_k([x for x in rows if x[0]['design'] == 'easy'])}/{sum(x[0]['design'] == 'easy' for x in rows)}; hard {ok_k([x for x in rows if x[0]['design'] == 'hard'])}/{sum(x[0]['design'] == 'hard' for x in rows)}")
    det = fmt_ci(m["det"], m["n_wrong"])
    lo_d = S.wilson(m["det"], m["n_wrong"])[0]
    lo_f, hi_f = S.wilson(m["fa_easy"], m["n_easy_right"])
    a_pt, a_lb = rate(m["det"], m["n_wrong"]) >= 0.60, lo_d >= 0.60
    b_pt, b_ub = rate(m["fa_easy"], m["n_easy_right"]) <= 0.25, hi_f <= 0.25
    word = lambda pt, bound: "MET" if (pt and bound) else ("MET on the point estimate, NOT DEMONSTRATED" if pt else "NOT MET")
    L += ["", f"(a) detection of wrong-K fits >= 60%: {det} -> {word(a_pt, a_lb)}",
          f"(b) false alarm on right-K fits of the easy design <= 25%: {fmt_ci(m['fa_easy'], m['n_easy_right'])} -> {word(b_pt, b_ub)}",
          f"(c) hard design, reported without a level: false alarm on right-K fits {fmt_ci(m['fa_hard'], m['n_hard_right'])}; detection on wrong-K fits of the hard design "
          f"{fmt_ci(sum(f['k_unsure'] for r, f in rows if r['design'] == 'hard' and r['selected_k'] != r['planted_k']), sum(1 for r, f in rows if r['design'] == 'hard' and r['selected_k'] != r['planted_k']))}",
          "", "tagged share by design and planted K:"]
    for dz in ("easy", "hard"):
        for k in (2, 3, 4):
            sub = [(r, f) for r, f in rows if r["design"] == dz and r["planted_k"] == k]
            L.append(f"  {dz:<5} planted K={k}: tagged {sum(f['k_unsure'] for _, f in sub)}/{len(sub)}; BIC right {sum(r['selected_k'] == r['planted_k'] for r, _ in sub)}/{len(sub)}")
    reasons = {}
    for r, f in rows:
        for x in f["reasons"]:
            reasons[x] = reasons.get(x, 0) + 1
    L += ["", f"reasons that fired (a fit can have several): {reasons}"]
    blind = [(r["seed"], r["design"], r["planted_k"], r["selected_k"]) for r, f in rows if r["selected_k"] != r["planted_k"] and not f["k_unsure"]]
    L.append(f"wrong-K fits with NO signal (the declared blind spot): {len(blind)} of {m['n_wrong']}: {blind}")
    open(out_txt, "w").write("\n".join(L) + "\n")
    if out_json:
        json.dump(dict(metrics=m, a=dict(point=a_pt, lb=a_lb), b=dict(point=b_pt, ub=b_ub), mismatches=mism, blind=blind, reasons=reasons), open(out_json, "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    mode, fits, out = sys.argv[1], sys.argv[2], sys.argv[3]
    js = sys.argv[4] if len(sys.argv) > 4 else None
    recs = load(fits)
    (report_dev if mode == "dev" else report_eval)(recs, out, js)
