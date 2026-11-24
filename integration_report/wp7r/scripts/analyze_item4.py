"""Analysis of Item 4 with the words and the recommendation rule of tests/PREDECLARED_CRITERIA_WP7R.md section 4.
usage: PYTHONPATH=. python3 analyze_item4.py COVERAGE.jsonl BOOTSTRAP.jsonl COST.json"""
import json, math, sys
import numpy as np

N2, N1 = math.erf(2 / math.sqrt(2)), math.erf(1 / math.sqrt(2))
B2, B1 = (N2 - 0.04, N2 + 0.04), (N1 - 0.04, N1 + 0.04)
MIN_PUB = 8


def inband(c2, c1):
    return B2[0] <= c2 <= B2[1] and B1[0] <= c1 <= B1[1]


def seed_level(a):
    return float(np.mean(a["c2"])), float(np.mean(a["c1"]))


def mean_se(x):
    x = np.asarray(x, float)
    return float(x.mean()), (float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan"))


cov = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
boot = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
cost = json.load(open(sys.argv[3]))
cov.sort(key=lambda r: r["seed"]); boot.sort(key=lambda r: r["seed"])
print(f"band +-2: [{B2[0]:.4f}, {B2[1]:.4f}]   band +-1: [{B1[0]:.4f}, {B1[1]:.4f}]\n")

print("=== 4.2 coverage (12 seeds 9101..9112), seed-level mean over the two latent dimensions ===")
arms = ["T0", "shrink5_200", "shrink5_20", "personal_200", "personal_20"]
res = {}
for arm in arms:
    rows = [(r["seed"], r["arms"][arm]) for r in cov if r["arms"][arm].get("c2") is not None]
    err = [r["seed"] for r in cov if r["arms"][arm].get("error")]
    allv = [(s, *seed_level(a), a.get("withheld", False), a.get("fit_state")) for s, a in rows]
    pub = [v for v in allv if not v[3]]
    f = lambda v, i: mean_se([x[i] for x in v]) if v else (float("nan"), float("nan"))
    n_w = sum(1 for v in allv if v[3])
    c2a, c2s = f(allv, 1); c1a, _ = f(allv, 2)
    print(f"{arm:13s} fits {len(allv)} errors {err} withheld {n_w}  ALL: c2 {c2a:.3f} (se {c2s:.3f}) c1 {c1a:.3f}  seeds with c2<0.80: {sum(1 for v in allv if v[1] < 0.80)}", end="")
    if pub:
        c2p, c2ps = f(pub, 1); c1p, _ = f(pub, 2)
        print(f"  | PUBLISHABLE n={len(pub)}: c2 {c2p:.3f} (se {c2ps:.3f}) c1 {c1p:.3f}  in band: {inband(c2p, c1p)}")
    else:
        print("  | PUBLISHABLE n=0")
    states = {}
    for v in allv:
        states[v[4]] = states.get(v[4], 0) + 1
    print(f"              fit states {states}")
    res[arm] = dict(all=(c2a, c1a), pub=(f(pub, 1)[0], f(pub, 2)[0], len(pub)) if pub else None)
t0_ok = inband(*res["T0"]["all"])
print(f"\nharness control T0 in band: {t0_ok}" + ("" if t0_ok else "   => HARNESS INVALID, nothing below is read"))
sp = res["shrink5_200"]["pub"]
if sp is None or sp[2] < MIN_PUB:
    word42 = f"not judged (publishable shrinkage fits at 200 iterations: {0 if sp is None else sp[2]} < {MIN_PUB})"
else:
    word42 = "reaches the band" if inband(sp[0], sp[1]) else "outside the band"
print(f"4.2 WORD (publishable shrinkage-fitted fits, max_iter 200): {word42}")

print("\n=== 4.3 parametric bootstrap (seeds 9101..9106, personal, 20 iterations, B=30, warm start) ===")
ok = [r for r in boot if r.get("EB")]
for r in boot:
    e0, eb = seed_level(r["E0"]), (seed_level(r["EB"]) if r.get("EB") else (float("nan"),) * 2)
    print(f"seed {r['seed']}: T {r['T']} fit_state {r['fit_state']} replicates ok {r['n_ok']} failed {r['n_fail']}  E0 c2 {e0[0]:.3f} c1 {e0[1]:.3f} | EB c2 {eb[0]:.3f} c1 {eb[1]:.3f}  std ratio {r.get('ratio', float('nan')):.2f}  {r['t_rep_s']} s/replicate, {r['seconds']} s/seed")
e0c2 = [seed_level(r["E0"])[0] for r in ok]; e0c1 = [seed_level(r["E0"])[1] for r in ok]
ebc2 = [seed_level(r["EB"])[0] for r in ok]; ebc1 = [seed_level(r["EB"])[1] for r in ok]
m = lambda x: f"{mean_se(x)[0]:.3f} (se {mean_se(x)[1]:.3f})"
print(f"mean over {len(ok)} seeds: E0 c2 {m(e0c2)} c1 {m(e0c1)} | EB c2 {m(ebc2)} c1 {m(ebc1)}")
d = np.array(ebc2) - np.array(e0c2)
dm, dse = mean_se(d)
print(f"EB - E0 (c2): mean {dm:+.3f} paired se {dse:.3f}; seeds improved {int(np.sum(d > 0))}/{len(d)}; mean ratio std_boot/std {np.mean([r['ratio'] for r in ok]):.2f}")
reach = bool(ok) and inband(mean_se(ebc2)[0], mean_se(ebc1)[0])
improves = int(np.sum(d > 0)) >= 5
word43 = "reaches the band" if reach else ("improves but does not reach" if improves else "does not help")
print(f"4.3 WORD: {word43}")
t_rep = float(np.mean([r["t_rep_s"] for r in boot])); t_fit = float(np.mean([r["t_fit_s"] for r in boot]))
print(f"time: one 20-iteration fit {t_fit:.1f} s, one replicate (simulate + warm refit + smooth) {t_rep:.1f} s, B=30 bootstrap {30 * t_rep:.0f} s per fit")

print("\n=== 4.4 cost ===")
for k, v in cost.items():
    print(k, v)

print("\n=== 4.5 recommendation (rule fixed in advance) ===")
boot_cost_s = 30 * t_rep
if reach and boot_cost_s <= 900:
    rec = "an offline bootstrap job (to be evaluated on fresh seeds before any integration)"
elif word42 == "reaches the band":
    rec = "shrinkage-only uncertainty (favourable-case caveat)"
elif word43 == "improves but does not reach":
    rec = "continue (bootstrap helps, does not reach the band)"
else:
    rec = "keep numeric confidence hidden"
print("RECOMMENDATION:", rec)
