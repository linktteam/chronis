"""WP9 A4 analysis (design: tests/PREDECLARED_CRITERIA_WP9.md section 18). usage:
  PYTHONPATH=. python3 integration_report/wp9/scripts/a4_analyze.py dev|eval FITS.jsonl OUT.txt [OUT.json] [--design A]
Replays the stored fits through the production lineage twice: (R) the production matching (stored scaler), (Z) the WP8 control (a), each fit in its own z-space."""
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wp9_fit_lib as W                                                   # noqa: E402
import wp9_replay_lib as RL                                               # noqa: E402

from backbone.hssm import lineage as L                                    # noqa: E402
from backbone.hssm.lineage import AlignmentConfig                         # noqa: E402

S = RL.S
MIN_ALIGNMENTS = 40


@contextlib.contextmanager
def patch_zspace():
    old = L._vectors
    L._vectors = lambda summary, scale, duration_weight: (summary.raw_means - summary.scaler_mean) / np.where(summary.scaler_std > 0, summary.scaler_std, 1.0)
    try:
        yield
    finally:
        L._vectors = old


def chains(recs, design):
    """{user: [records]} for one design (the shared windows plus the design's own), in window order."""
    out = {}
    for r in recs:
        g = r["job"]["group"]
        if g == "A4-shared" or g == f"A4-{design}":
            out.setdefault(r["job"]["user"], []).append(r)
    return {u: sorted(v, key=lambda r: r["job"]["end"]) for u, v in out.items()}


def run_method(recs, design, zspace, cfg):
    rows = []
    for u, ch in sorted(chains(recs, design).items()):
        ctx = patch_zspace() if zspace else contextlib.nullcontext()
        with ctx:
            r, _, _, _ = RL.replay9(ch, cfg, n=10, use_flag=True, lineage_id=f"a4-{design}-{u}")
        rows += r
    return rows


def metrics(rows):
    t = S.tally(rows)
    k, n = t["served_correct"], t["served"]
    return dict(alignments=t["alignments"], served=t["served"], served_correct=k, wrong_unflagged=t["wrong_unflagged"], correct_rate=(k / n if n else float("nan")), wu_rate=(t["wrong_unflagged"] / t["alignments"] if t["alignments"] else float("nan")),
                coverage=(t["served"] / t["alignments"] if t["alignments"] else float("nan")))


def raw_works(m):
    return m["alignments"] >= MIN_ALIGNMENTS and m["served"] > 0 and m["correct_rate"] >= 0.95 and m["wu_rate"] <= 0.01


def z_fails(m):
    return m["alignments"] > 0 and ((m["served"] > 0 and m["correct_rate"] < 0.95) or m["wu_rate"] > 0.01)


def line(tag, m):
    return (f"{tag}: alignments {m['alignments']}, served {m['served']} (coverage {100 * m['coverage']:.0f}%), correct among served {m['served_correct']}/{m['served']} = "
            f"{100 * m['correct_rate']:.1f}%, wrong-and-unflagged {m['wrong_unflagged']}/{m['alignments']} = {100 * m['wu_rate']:.1f}%")


def paired(rr, rz):
    key = lambda r: (r["user"], r["end"])
    a = {key(r): r for r in rr if r.get("correct") is not None}
    b = {key(r): r for r in rz if r.get("correct") is not None}
    wrong = lambda r: r["served"] and not r["correct"]
    right = lambda r: r["served"] and r["correct"]
    both = set(a) & set(b)
    return dict(common=len(both), z_wrong_r_right=sum(wrong(b[k]) and right(a[k]) for k in both), r_wrong_z_right=sum(wrong(a[k]) and right(b[k]) for k in both),
                z_wrong=sum(wrong(b[k]) for k in both), r_wrong=sum(wrong(a[k]) for k in both))


def main(mode, fits, out_txt, out_json=None, design=None):
    recs = W.load_records(fits)
    cfg = AlignmentConfig()
    res, L_ = {}, [f"A4 {mode}: {len(recs)} fits; AlignmentConfig defaults min_separation {cfg.min_separation}", ""]
    designs = [d for d in ("A", "B", "C") if any(r["job"]["group"] == f"A4-{d}" for r in recs)] if mode == "dev" else [design]
    designs = designs + (["none"] if any(r["job"]["group"] == "A4-none" for r in recs) else [])
    for d in designs:
        rr, rz = run_method(recs, d, False, cfg), run_method(recs, d, True, cfg)
        mr, mz = metrics(rr), metrics(rz)
        p = paired(rr, rz)
        res[d] = dict(R=mr, Z=mz, paired=p, qualifies=bool(raw_works(mr) and z_fails(mz)))
        L_ += [f"design {d}", "  " + line("R (stored scaler)", mr), "  " + line("Z (own z-space)  ", mz), f"  paired on {p['common']} common alignments: Z wrong and R right {p['z_wrong_r_right']}, R wrong and Z right {p['r_wrong_z_right']}",
               f"  R works (>= {MIN_ALIGNMENTS} alignments, >= 95% correct, <= 1% wrong-unflagged): {raw_works(mr)}; Z fails as required: {z_fails(mz)}"]
        if mode == "dev" and d != "none":
            L_.append(f"  QUALIFIES: {res[d]['qualifies']}")
        L_.append("")
    if mode == "dev":
        chosen = next((d for d in ("A", "B", "C") if d in res and res[d]["qualifies"]), None)
        L_.append(f"declared rule: first qualifying design in order A, B, C -> {chosen if chosen else 'NONE QUALIFIES: the control stays inert on every design tried; the evaluation is run on design A'}")
        res["chosen"] = chosen
    else:
        d = design
        m = res[d]
        L_.append(f"EVALUATION design {d}: stream works for the stored scaler: {raw_works(m['R'])}; the z-space control {'FAILS as required' if z_fails(m['Z']) else 'is INERT (reported as inert, not as passed)'}")
    Path(out_txt).write_text("\n".join(L_) + "\n")
    if out_json:
        json.dump(res, open(out_json, "w"), indent=1, default=float)
    print("\n".join(L_))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    des = sys.argv[sys.argv.index("--design") + 1] if "--design" in sys.argv else None
    if des in args:
        args.remove(des)
    main(args[0], args[1], args[2], args[3] if len(args) > 3 else None, des)
