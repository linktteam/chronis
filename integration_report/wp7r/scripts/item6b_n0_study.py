"""WP7-R item 6b: n0 = 1 and 2 against n0 = 5. Design and decision rule: tests/PREDECLARED_CRITERIA_WP7R.md section 6 (committed before this script ran).

  PYTHONPATH=. python3 item6b_n0_study.py run  --seeds 6001:6012 --arms gauss 1 2 5 --out DEV.jsonl [--jobs 2]     (resumable; one JSON line per seed, length, arm)
  PYTHONPATH=. python3 item6b_n0_study.py dev  --file DEV.jsonl                                                       (the pre-declared rule: does n0 = 1 or 2 strictly beat 5?)
  PYTHONPATH=. python3 item6b_n0_study.py eval --file EVAL.jsonl --w W                                                (7501..7524: pre-declared word and the withheld counts)

WP5 design kept exactly: class path ShrinkageDualHSSM, 20 EM iterations (N18_ITER), population users (generate_population_user, 2/4/8 segments), score = tests.utils.n18v2.score_model,
all fits counted. "withheld" here is the class-path proxy of the public rule: fit_state != CONVERGED or fit_degraded (the public path would also add the collapsed-regime and
duration-degeneracy rules; those are not applied here and the results note says so).
"""
import argparse, json, math, os, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.getcwd())
from backbone.hssm.shrinkage import SharedPrior, ShrinkageDualHSSM       # noqa: E402
from data.synthetic_population import PopulationSpec, generate_population_user  # noqa: E402
from tests.utils import n18v2 as N                                      # noqa: E402

PRIOR_PATH = "wp7r_evidence/prior_main_wp5.json"
REF_WP5 = {30: 0.10458304433333632, 60: 0.5685922816813438, 120: 0.8857480823607287}   # stored WP5 lower bounds for n0 = 5 (reproducibility check only; not used in the rule)
CANDIDATES = (1.0, 2.0)


def run_arm(arm, user, prior):
    t0 = time.time()
    rec = dict(arm=arm)
    try:
        if arm == "gauss":
            rec.update(N.gauss_scores(user))
        else:
            m = ShrinkageDualHSSM(n_regimes=N.K, n_features=N.F, latent_dim=N.L, max_duration=N.MAX_DURATION, seed=user.seed, shared_prior=prior, n0=float(arm)).fit(user.X_train, n_iter=N.N_ITER)
            rec.update(N.score_model(m, user)); rec.update(N.fit_info(m)); rec["used_shared_prior"] = bool(m.used_shared_prior)
    except Exception as exc:                                              # recorded, never swallowed: the analysis counts errors
        rec.update(mean=float("nan"), finite=False, error=f"{type(exc).__name__}: {exc}"[:300])
    rec["wall_s"] = round(time.time() - t0, 2)
    return rec


def run_seed(job):
    seed, arms = job
    warnings.filterwarnings("ignore")
    prior = SharedPrior.from_dict(json.load(open(PRIOR_PATH)))
    out = []
    for length in N.LENGTHS:
        user = generate_population_user(seed, N.SEGMENTS[length], 2, PopulationSpec())
        for arm in arms:
            rec = run_arm(arm, user, prior)
            rec.update(seed=seed, length=length, T_train=int(len(user.X_train)), T_test=int(len(user.X_test)))
            out.append(rec)
    return seed, out


def read(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def cmd_run(a):
    lo, hi = a.seeds.split(":")
    done = {(r["seed"], r["arm"], r["length"]) for r in read(a.out)}
    jobs = []
    for seed in range(int(lo), int(hi) + 1):
        todo = [x for x in a.arms if not all((seed, x, L) in done for L in N.LENGTHS)]
        if todo:
            jobs.append((seed, todo))
    print(f"{len(jobs)} seeds to run, {a.jobs} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex, open(a.out, "a") as fh:
        for f in as_completed([ex.submit(run_seed, j) for j in jobs]):
            seed, recs = f.result()
            for r in recs:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"seed {seed} done ({len(recs)} records, {time.time() - t0:.0f}s elapsed)", flush=True)


def scores(rows):
    s = {}
    for r in rows:
        s.setdefault(r["arm"], {}).setdefault(r["length"], {})[r["seed"]] = r["mean"] if r.get("finite", True) else float("nan")
    return s


def fmt(st):
    return f"{st['mean']:+7.3f} se {st['se']:6.3f} mean-3se {st['mean_minus_3se']:+7.3f} favour {st['n_favour']:2d}/{st['n']:<2d} dropped {st['n_dropped']}"


def health(rows, arms):
    for arm in arms:
        for L in N.LENGTHS:
            rr = [r for r in rows if r["arm"] == arm and r["length"] == L]
            if not rr:
                continue
            wh = sum(1 for r in rr if r.get("fit_state", "CONVERGED") != "CONVERGED" or r.get("fit_degraded"))
            if arm != "gauss":
                print(f"   arm {arm:>4s} len {L:3d}: n {len(rr):2d}  errors {sum('error' in r for r in rr)}  non-finite {sum(not r.get('finite', True) for r in rr)}  "
                      f"withheld (class-path proxy: not CONVERGED or degraded) {wh}/{len(rr)}")


def cmd_dev(a):
    rows = read(a.file)
    s = scores(rows)
    lowers = {}
    print("development seeds 6001..6012: shrinkage minus single Gaussian (WP5 statistic)")
    for n0 in (1.0, 2.0, 5.0):
        arm = f"{n0:g}"
        lowers[n0] = {L: N.paired_stats(s[arm][L], s["gauss"][L]) for L in N.LENGTHS}
        for L in N.LENGTHS:
            print(f"  n0 {n0:g}  len {L:3d}: {fmt(lowers[n0][L])}")
        print(f"  n0 {n0:g}  score (min over lengths of mean - 3 se) = {min(v['mean_minus_3se'] for v in lowers[n0].values()):+.4f}")
    print("  stored WP5 n0 = 5 lower bounds (reference only): " + "  ".join(f"{L}: {REF_WP5[L]:+.4f}" for L in N.LENGTHS))
    print("  reproduced     n0 = 5 lower bounds (this head):  " + "  ".join(f"{L}: {lowers[5.0][L]['mean_minus_3se']:+.4f}" for L in N.LENGTHS))
    health(rows, ["1", "2", "5"])
    score = {n0: min(v["mean_minus_3se"] for v in lowers[n0].values()) for n0 in lowers}
    beat = [w for w in CANDIDATES if math.isfinite(score[w]) and math.isfinite(score[5.0]) and score[w] > score[5.0]]
    print()
    if not beat:
        print("RULE 6b.1: neither n0 = 1 nor n0 = 2 has a STRICTLY larger score than n0 = 5. Nothing further is run; the default n0 = 5 stands.")
    else:
        w = max(beat, key=lambda x: (score[x], -x))
        print(f"RULE 6b.1: n0 = {w:g} has a strictly larger score than n0 = 5 ({score[w]:+.4f} vs {score[5.0]:+.4f}). Run the evaluation once on 7501..7524 with arms gauss 5 {w:g}.")
        print(f"WINNER {w:g}")


def cmd_eval(a):
    rows = read(a.file)
    s = scores(rows)
    w = f"{a.w:g}"
    print(f"evaluation seeds 7501..7524: n0 = {w} against n0 = 5 (shrinkage minus single Gaussian; then w minus 5)")
    better = True
    for L in N.LENGTHS:
        v5, vw = N.paired_stats(s["5"][L], s["gauss"][L]), N.paired_stats(s[w][L], s["gauss"][L])
        d = N.paired_stats(s[w][L], s["5"][L])
        print(f"  len {L:3d}  n0 5 - gauss: {fmt(v5)}")
        print(f"          n0 {w} - gauss: {fmt(vw)}")
        print(f"          n0 {w} - n0 5 : {fmt(d)}")
        better &= bool(math.isfinite(d["mean_minus_3se"]) and d["mean_minus_3se"] > 0)
    health(rows, ["5", w])
    print(f"\nWORD (6b.2): n0 = {w} is {'better' if better else 'not better'} than 5 on the evaluation seeds"
          + ("" if better else " (the paired difference w minus 5 does not have mean - 3 se > 0 at every length)"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--seeds", required=True); p.add_argument("--arms", nargs="+", required=True); p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=2); p.set_defaults(f=cmd_run)
    p = sub.add_parser("dev"); p.add_argument("--file", required=True); p.set_defaults(f=cmd_dev)
    p = sub.add_parser("eval"); p.add_argument("--file", required=True); p.add_argument("--w", type=float, required=True); p.set_defaults(f=cmd_eval)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
