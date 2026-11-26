"""WP10A A4 analysis (design: tests/PREDECLARED_CRITERIA_WP10A.md Part 2, sections 5-7). Replays stored fits through the PRODUCTION lineage twice: (R) the production matching (raw means through the anchor's
stored scaler) and (Z) the fault/control: each fit in its own z-space (`patch_zspace`). Both at h = 2.5, flag ON (consensus flag + sharpness floor 0.95, as defined for A3).
usage: PYTHONPATH=. python3 integration_report/wp10a/scripts/a4_analyze.py dev  A4_FITS.jsonl A3_DEV_FITS.jsonl OUT.txt OUT.json
       PYTHONPATH=. python3 integration_report/wp10a/scripts/a4_analyze.py eval OUT.txt OUT.json FITS.jsonl [FITS2.jsonl ...]     (needs tests/DESIGN_FREEZE_WP10A_A4.txt; the design is read from it)
       add --rehearsal to run `eval` on DEVELOPMENT fits (plumbing test; reads no evaluation seed)
Criteria (frozen in Part 2): R works = >= 40 alignments, correct among served >= 95%, wrong-and-unflagged <= 1% of alignments. The control FAILS (new criterion) when `zr` >= 3 and `zr` > `rz`, where
zr = alignments on which Z served wrong and R served right, rz = R served wrong and Z served right (paired on the same fits). The old WP9 criterion (Z correct among served < 95% or Z wrong-and-unflagged > 1%)
is computed and shown beside it. A design qualifies when R works and the control fails. Best = largest zr (ties in the order D, E, F); no qualifying design = A4 closed, nothing evaluated."""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "wp9" / "scripts"))
import a12_analyze as A                                                           # noqa: E402
import wp9_fit_lib as F                                                           # noqa: E402
import wp9_replay_lib as R                                                        # noqa: E402
from wp9_replay_lib import S                                                      # noqa: E402

from backbone.hssm import lineage as L                                            # noqa: E402
from backbone.hssm.lineage import AlignmentConfig                                 # noqa: E402

H = 2.5
SHARP = 0.95
MIN_ALIGNMENTS = 40
CORRECT_LEVEL = 0.95
WU_LEVEL = 0.01
ZR_MIN = 3
ORDER = ("D", "E", "F")
FREEZE = Path("tests/DESIGN_FREEZE_WP10A_A4.txt")
OUT: list = []


def P(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)


@contextlib.contextmanager
def patch_zspace():
    """The control/fault: every summary uses its OWN scaler (mean, sd) instead of the anchor's stored one (the same replacement as the WP8/WP9 control)."""
    old = L._vectors
    L._vectors = lambda summary, scale, duration_weight: (summary.raw_means - summary.scaler_mean) / np.where(summary.scaler_std > 0, summary.scaler_std, 1.0)
    try:
        yield
    finally:
        L._vectors = old


# ------------------------------------------------------------------------------------------------------------------------------ chains per design
def chains_d(recs: list, variant: str) -> dict:
    """{(user,): [records]} for design D: the shared windows plus the windows of `variant` ('D' = planted change, 'none'), in window order."""
    out: dict = {}
    for r in recs:
        g = r["job"].get("group")
        if g == "A4D-shared" or g == f"A4D-{variant}":
            out.setdefault((r["job"]["user"],), []).append(r)
    return {k: sorted(v, key=lambda r: r["job"]["end"]) for k, v in out.items()}


def chains_e(recs: list) -> dict:
    """{(user, L, t): [anchor, short window]}: every short window is aligned with the long anchor [0, 90) as its own one-pair chain."""
    anchors = {r["job"]["user"]: r for r in recs if r["job"].get("group") == "A4E-anchor"}
    out: dict = {}
    for r in recs:
        g = r["job"].get("group") or ""
        if g.startswith("A4E-L") and r["job"]["user"] in anchors:
            out[(r["job"]["user"], int(g[len("A4E-L"):]), r["job"]["end"])] = [anchors[r["job"]["user"]], r]
    return dict(sorted(out.items()))


def chains_f(recs: list) -> dict:
    """{(user,): [records]} for design F: the production growing windows of the A3 seed-A chains (all window ends of the file)."""
    a, _ = A.split_chains(recs)
    return {(u,): v for (u, _), v in sorted(a.items())}


# ------------------------------------------------------------------------------------------------------------------------------ replay and metrics
def replay_all(chains: dict, zspace: bool, tag: str) -> list:
    rows = []
    for key, ch in chains.items():
        ctx = patch_zspace() if zspace else contextlib.nullcontext()
        with ctx:
            r, _, _, _ = R.replay9(ch, AlignmentConfig(min_separation=H), n=10, use_flag=True, sharpness=SHARP, lineage_id=f"a4-{tag}-{key}")
        for row in r:
            row["pair_key"] = (key, row["end"])
        rows += r
    return rows


def metrics(rows: list) -> dict:
    t = S.tally(rows)
    k, n = t["served_correct"], t["served"]
    return dict(alignments=t["alignments"], served=n, served_correct=k, wrong_unflagged=t["wrong_unflagged"], correct_rate=(k / n if n else float("nan")),
                wu_rate=(t["wrong_unflagged"] / t["alignments"] if t["alignments"] else float("nan")), coverage=(n / t["alignments"] if t["alignments"] else float("nan")))


def r_works(m: dict) -> bool:
    return m["alignments"] >= MIN_ALIGNMENTS and m["served"] > 0 and m["correct_rate"] >= CORRECT_LEVEL and m["wu_rate"] <= WU_LEVEL


def z_old_fails(m: dict) -> bool:
    """The WP9 criterion, reported beside the new one."""
    return m["alignments"] > 0 and ((m["served"] > 0 and m["correct_rate"] < CORRECT_LEVEL) or m["wu_rate"] > WU_LEVEL)


def paired(rr: list, rz: list) -> dict:
    a = {r["pair_key"]: r for r in rr if r.get("correct") is not None}
    b = {r["pair_key"]: r for r in rz if r.get("correct") is not None}
    wrong = lambda r: bool(r["served"]) and not r["correct"]
    right = lambda r: bool(r["served"]) and bool(r["correct"])
    both = sorted(set(a) & set(b), key=repr)
    return dict(common=len(both), zr=sum(wrong(b[k]) and right(a[k]) for k in both), rz=sum(wrong(a[k]) and right(b[k]) for k in both), z_wrong=sum(wrong(b[k]) for k in both), r_wrong=sum(wrong(a[k]) for k in both),
                served_differs=sum(bool(a[k]["served"]) != bool(b[k]["served"]) for k in both))


def control_fails(p: dict) -> bool:
    """The new, named criterion: Z wrong where R is right on at least 3 alignments, and more often than the reverse."""
    return p["zr"] >= ZR_MIN and p["zr"] > p["rz"]


def qualifies(mr: dict, p: dict) -> bool:
    return bool(r_works(mr) and control_fails(p))


def best_design(results: dict):
    """Largest zr among the qualifying designs (ties in the order D, E, F); None when no design qualifies."""
    q = [d for d in ORDER if d in results and results[d]["qualifies"]]
    if not q:
        return None
    return max(q, key=lambda d: (results[d]["paired"]["zr"], -ORDER.index(d)))


def withheld(chains: dict) -> tuple:
    """(fits, fits the publish rule or the usability check withholds): counted over the distinct records of the chains."""
    seen = {}
    for ch in chains.values():
        for r in ch:
            seen[id(r)] = r
    n = len(seen)
    w = sum(1 for r in seen.values() if R.prepared(r, 10) is None)
    return n, w


def line(tag: str, m: dict) -> str:
    return (f"{tag}: alignments {m['alignments']}, served {m['served']} (coverage {100 * m['coverage']:.0f}%), correct among served {m['served_correct']}/{m['served']} = {100 * m['correct_rate']:.1f}%, "
            f"wrong-and-unflagged {m['wrong_unflagged']}/{m['alignments']} = {100 * m['wu_rate']:.1f}%")


def study(name: str, chains: dict) -> dict:
    rr, rz = replay_all(chains, False, name), replay_all(chains, True, name)
    mr, mz, p = metrics(rr), metrics(rz), paired(rr, rz)
    n_fits, n_wh = withheld(chains)
    res = dict(R=mr, Z=mz, paired=p, r_works=r_works(mr), control_fails=control_fails(p), z_old_fails=z_old_fails(mz), qualifies=qualifies(mr, p), fits=n_fits, withheld=n_wh)
    strata = sorted({r.get("stratum") for r in rr if r.get("stratum")})
    if name == "F" and strata:
        res["by_stratum"] = {}
        for st in strata:
            a_, b_ = [r for r in rr if r.get("stratum") == st], [r for r in rz if r.get("stratum") == st]
            res["by_stratum"][st] = dict(R=metrics(a_), Z=metrics(b_), paired=paired(a_, b_))
    P(f"design {name}: {n_fits} fits ({n_wh} withheld by the publish/usability rule), {len(chains)} chains")
    P("  " + line("R (stored scaler)", mr))
    P("  " + line("Z (own z-space)  ", mz))
    P(f"  paired on {p['common']} common alignments: Z served wrong and R served right (zr) = {p['zr']}, R served wrong and Z served right (rz) = {p['rz']}; Z wrong {p['z_wrong']}, R wrong {p['r_wrong']}; serve decision differs on {p['served_differs']}")
    P(f"  R works (>= {MIN_ALIGNMENTS} alignments, >= 95% correct, <= 1% wrong-unflagged): {res['r_works']}; control fails by the NEW criterion (zr >= {ZR_MIN} and zr > rz): {res['control_fails']}; old WP9 criterion: {res['z_old_fails']}; QUALIFIES: {res['qualifies']}")
    for st, v in res.get("by_stratum", {}).items():
        P(f"    stratum {st}: R wrong-unflagged {v['R']['wrong_unflagged']}/{v['R']['alignments']}, Z wrong-unflagged {v['Z']['wrong_unflagged']}/{v['Z']['alignments']}, zr {v['paired']['zr']}, rz {v['paired']['rz']}")
    P("")
    return res


# ------------------------------------------------------------------------------------------------------------------------------ development and evaluation
def dev(a4_fits: str, a3_fits: str, out_json: str):
    recs4 = F.load_records(a4_fits)
    P(f"A4 DEVELOPMENT: {len(recs4)} A4 fits; h = {H}, flag ON (consensus + sharpness {SHARP}); AlignmentConfig min_separation {AlignmentConfig().min_separation} is the production default\n")
    res = {}
    if any(r["job"].get("group", "").startswith("A4D") for r in recs4):
        res["D"] = study("D", chains_d(recs4, "D"))
        res["D_none"] = study("D_none (the same users without the planted change; reported only)", chains_d(recs4, "none"))
    if any(r["job"].get("group", "").startswith("A4E") for r in recs4):
        res["E"] = study("E", chains_e(recs4))
    recs3 = F.load_records(a3_fits)
    res["F"] = study("F", chains_f(recs3))
    best = best_design(res)
    P("declared rule: qualifying design with the largest zr (ties in the order D, E, F): " + (best if best else "NONE QUALIFIES -> A4 is closed as 'the protective effect of the stored scaler cannot be demonstrated'; no evaluation is run"))
    res["best"] = best
    json.dump(res, open(out_json, "w"), indent=1, default=float)


def evaluate(out_json: str, fits: list, rehearsal: bool):
    if not FREEZE.exists():
        sys.exit("the evaluation needs tests/DESIGN_FREEZE_WP10A_A4.txt")
    design = json.loads(FREEZE.read_text().splitlines()[0])["design"]
    recs = [r for f in fits for r in F.load_records(f)]
    P(f"A4 EVALUATION (once): frozen design {design}; {len(recs)} fits{' (REHEARSAL on development fits)' if rehearsal else ''}\n")
    res = {}
    if design == "D":
        res["D"] = study("D", chains_d(recs, "D"))
        res["D_none"] = study("D_none (the same users without the planted change)", chains_d(recs, "none"))
    elif design == "E":
        res["E"] = study("E", chains_e(recs))
    elif design == "F":
        res["F"] = study("F", chains_f(recs))
    d = res[design]
    P(f"EVALUATION design {design}: R works: {d['r_works']}; the control {'FAILS AS REQUIRED' if d['control_fails'] else 'is INERT (reported as inert, not as passed)'}; qualifies on the evaluation users: {d['qualifies']}")
    res["design"] = design
    json.dump(res, open(out_json, "w"), indent=1, default=float)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args and args[0] == "dev":
        dev(args[1], args[2], args[4])
        Path(args[3]).write_text("\n".join(OUT) + "\n")
    elif args and args[0] == "eval":
        evaluate(args[2], args[3:], "--rehearsal" in sys.argv)
        Path(args[1]).write_text("\n".join(OUT) + "\n")
    else:
        sys.exit(__doc__)
