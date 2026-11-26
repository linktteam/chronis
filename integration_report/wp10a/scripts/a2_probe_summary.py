"""WP10A A2: summary of the probe drive (`a2_probe.py --drive OUT.json`): per pair of configurations, how many fit fields differ BITWISE (the `config` label itself is not a fit field), the largest absolute and relative
difference over the stored float values (scalars and the first values of every array), and whether any discrete outcome differs. usage: python3 a2_probe_summary.py IN.json OUT.txt"""
import json
import sys
from itertools import combinations

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from a2_probe import flat  # noqa: E402


def fitfields(r):
    return dict(flat({k: v for k, v in r.items() if k not in ("blas", "env", "config")}))


def floats(fields):
    """The stored float values: scalars (key ends with `.f`) and the first values of every float array (key contains `.head.`), parsed from their hex representation."""
    out = {}
    for k, v in fields.items():
        if (k.endswith(".f") or ".head." in k) and isinstance(v, str) and v.startswith(("0x", "-0x")):
            out[k] = float.fromhex(v)
    return out


def main(inp, out):
    d = json.load(open(inp))
    runs = d["runs"]
    name = lambda r: f"{r['config']['kernel']}/t{r['config']['threads']}" + (f"#{r['config']['rep']}" if r["config"]["rep"] else "")
    F = {name(r): fitfields(r) for r in runs}
    L = []
    kernels = []
    for r in runs:
        if r["config"]["kernel"] not in kernels:
            kernels.append(r["config"]["kernel"])
    L.append("BLAS library as reported by threadpoolctl (arch, threads) per configuration:")
    for r in runs:
        L.append(f"  {name(r):18s} {[(b['arch'], b['threads']) for b in r['blas']][:1]}")
    L.append("\nWITHIN a kernel, across thread settings (1, 2, unset):")
    for k in kernels:
        ns = [name(r) for r in runs if r["config"]["kernel"] == k]
        ref = F[ns[0]]
        res = []
        for n in ns[1:]:
            diff = [x for x in sorted(set(ref) | set(F[n])) if ref.get(x) != F[n].get(x)]
            res.append(f"{n}: {'bit-identical' if not diff else str(len(diff)) + ' fields differ'}")
        L.append(f"  {k:12s} reference {ns[0]}: " + "; ".join(res))
    L.append("\nBETWEEN kernels (threads = 1 each), fit fields that differ bitwise / largest absolute and relative difference of the stored floats / discrete outcomes that differ:")
    one = {k: next(name(r) for r in runs if r["config"]["kernel"] == k and r["config"]["threads"] == "1" and not r["config"]["rep"]) for k in kernels}
    for a, b in combinations(kernels, 2):
        fa, fb = F[one[a]], F[one[b]]
        diff = [x for x in sorted(set(fa) | set(fb)) if fa.get(x) != fb.get(x)]
        xa, xb = floats(fa), floats(fb)
        ab = [(abs(xa[x] - xb[x]), abs(xa[x] - xb[x]) / max(abs(xa[x]), abs(xb[x]), 1e-300), x) for x in xa if x in xb and xa[x] != xb[x]]
        mx = max(ab) if ab else (0.0, 0.0, "-")
        mr = max(ab, key=lambda t: t[1]) if ab else (0.0, 0.0, "-")
        disc = [x for x in diff if x.startswith("outcomes.")]
        L.append(f"  {a:11s} vs {b:11s}: {len(diff):3d} of {len(set(fa) | set(fb))} fit fields differ; stored floats that differ: {len(ab)}; max abs diff {mx[0]:.3g} (at {mx[2]}); max rel diff {mr[1]:.3g}; discrete outcomes that differ: {disc if disc else 'none'}")
    L.append("\nDiscrete outcomes (identical in every configuration unless listed above):")
    L.append("  " + json.dumps(runs[0]["outcomes"]))
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
