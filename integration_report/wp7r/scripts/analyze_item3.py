"""Analysis of the item 3 study with the pre-declared statistics (tests/utils/item3_stats.py). usage: PYTHONPATH=. python3 analyze_item3.py A.jsonl [B.jsonl] > out.txt"""
import json, sys, math
from collections import defaultdict
import numpy as np
from tests.utils import item3_stats as S
from tests.utils.n18v2 import paired_stats

LENGTHS = (30, 60, 120)


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def by(rows):
    d = defaultdict(dict)                       # d[(arm, length)][seed] = record
    for r in rows:
        d[(r["arm"], r["length"])][r["seed"]] = r
    return d


def sc(recs):
    return {s: (r["mean"] if r.get("finite", True) else float("nan")) for s, r in recs.items()}


def fmt(st):
    return f"mean {st['mean']:+.3f} se {st['se']:.3f} mean-3se {st['mean_minus_3se']:+.3f} fav {st.get('n_favour', 0)}/{st.get('n_used', st.get('n_seeds'))}"


def report(rows, stage, arms_fit):
    d = by(rows)
    seeds = sorted({r["seed"] for r in rows})
    print(f"\n================ STAGE {stage}  (max_iter {rows[0]['max_iter']}, {len(seeds)} seeds {seeds[0]}..{seeds[-1]}) ================")
    for arm in arms_fit:
        print(f"\n--- 1. withheld tally: {arm}")
        for L in LENGTHS:
            recs = list(d[(arm, L)].values())
            t = S.withheld_tally(recs)
            iters = [r.get("n_iter_run") for r in recs]
            crit = defaultdict(int)
            for r in recs:
                crit[str(r.get("criterion"))] += 1
            print(f"  L={L:3d}: n={t['n']} withheld={t['n_withheld']} ({t['withheld_rate']:.0%}) publishable={t['n_publishable']}  all states {t['fit_states_all']}  reasons {t['by_reason_prefix']}  "
                  f"iterations run median {int(np.median(iters))} (min {min(iters)}, max {max(iters)})  criterion {dict(crit)}  exceptions {sum(1 for r in recs if r.get('error'))}  degraded {sum(1 for r in recs if r.get('fit_degraded'))}")
    gauss = {L: sc(d[("gauss", L)]) for L in LENGTHS} if ("gauss", 30) in d else None
    shared = {L: sc(d[("shared", L)]) for L in LENGTHS} if ("shared", 30) in d else None
    ref_g = gauss
    for arm in arms_fit:
        pub = {L: {s: not r["withheld"] for s, r in d[(arm, L)].items()} for L in LENGTHS}
        scores = {L: sc(d[(arm, L)]) for L in LENGTHS}
        print(f"\n--- 2. publishable-fit statistic: {arm} minus gauss (judged needs n >= {S.MIN_PUBLISHABLE})")
        stl = {}
        for L in LENGTHS:
            st = S.subset_stats(scores[L], ref_g[L], pub[L])
            stl[L] = st
            print(f"  L={L:3d}: publishable {st['n_publishable']}/{st['n_seeds']} used {st['n_used']} nonfinite {st['n_nonfinite_publishable']} judged={st['judged']}  "
                  f"arm {st['mean_arm']:+.3f} gauss {st['mean_ref']:+.3f}  {fmt(st)}")
        print(f"  VERDICT (survives on publishable fits): {S.survives(stl)}")
        print(f"--- 3. risk-coverage (withheld = no prediction -> single Gaussian): {arm} minus gauss over all seeds")
        pos = True
        for L in LENGTHS:
            fb = S.fallback_stats(scores[L], ref_g[L], pub[L])
            pos &= fb["mean_minus_3se"] > 0
            print(f"  L={L:3d}: coverage {fb['coverage']:.0%} ({fb['n_served']}/{fb['n_seeds']})  mean {fb['mean']:+.3f} se {fb['se']:.3f} mean-3se {fb['mean_minus_3se']:+.3f}")
        print(f"  NET DEPLOYMENT GAIN: {'positive' if pos else 'not shown'}")
        print(f"--- 4. WP5 statistic (ALL fits, withheld included, not a publishable-fit statement): {arm} minus gauss")
        for L in LENGTHS:
            st = paired_stats(scores[L], ref_g[L])
            print(f"  L={L:3d}: n {st['n']} dropped {st['n_dropped']}  mean {st['mean']:+.3f} se {st['se']:.3f} mean-3se {st['mean_minus_3se']:+.3f} fav {st['n_favour']}/{st['n']}")
        if shared:
            print(f"--- 2b. {arm} minus shared on publishable fits")
            for L in LENGTHS:
                st = S.subset_stats(scores[L], shared[L], pub[L])
                print(f"  L={L:3d}: used {st['n_used']} judged={st['judged']}  {fmt(st)}")
        print(f"--- 5. eligibility: seeds with >= 30 training rows (production cold-start gate)")
        for L in LENGTHS:
            el = {s for s, r in d[(arm, L)].items() if r["T_train"] >= 30}
            both = {s for s in el if pub[L][s]}
            print(f"  L={L:3d}: eligible {len(el)}/{len(pub[L])}, eligible and publishable {len(both)}   T_train range {min(r['T_train'] for r in d[(arm, L)].values())}..{max(r['T_train'] for r in d[(arm, L)].values())}")
    if "personal" in arms_fit and "shrink5" in arms_fit:
        print("\n--- 2c. shrink5 minus personal on the seeds where BOTH are publishable")
        for L in LENGTHS:
            both = {s: (not d[("shrink5", L)][s]["withheld"]) and (not d[("personal", L)][s]["withheld"]) for s in d[("shrink5", L)]}
            st = S.subset_stats(sc(d[("shrink5", L)]), sc(d[("personal", L)]), both)
            print(f"  L={L:3d}: n both publishable {st['n_publishable']}  judged={st['judged']}  {fmt(st) if st['n_used'] >= 2 else 'no statistic'}")
    print("\n--- context: mean held-out score per arm over ALL seeds (finite only), per length")
    for arm in list(dict.fromkeys(["gauss", "shared"] + list(arms_fit))):
        if (arm, 30) not in d:
            continue
        print(f"  {arm:8s} " + "  ".join(f"L={L}: {np.nanmean([x for x in sc(d[(arm, L)]).values() if math.isfinite(x)]):+8.3f} (n finite {sum(1 for x in sc(d[(arm, L)]).values() if math.isfinite(x))})" for L in LENGTHS))


def main():
    a = load(sys.argv[1])
    report(a, "A", ("personal", "shrink5"))
    if len(sys.argv) > 2:
        b = load(sys.argv[2])
        # gauss and shared do not depend on the iteration cap: reuse them from stage A
        b_full = b + [r for r in a if r["arm"] in ("gauss", "shared")]
        report(b_full, "B", ("personal", "shrink5"))


if __name__ == "__main__":
    main()
