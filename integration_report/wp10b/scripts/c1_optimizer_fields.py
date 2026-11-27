"""WP10B C1(1): which sub-field of `optimizer_success_seed1/2` differs across CPU kernels? (design: tests/PREDECLARED_CRITERIA_WP10B.md Part 1, section C1(1)).
Read-only analysis of the stored WP10A probe data (`wp10a_evidence/a2_probe_allkernels.json`, 18 kernel settings, threads = 1, produced by `integration_report/wp10a/scripts/a2_probe.py`); no fit is run and no
random number is drawn. In that data `optimizer_success_seedN` is the LABEL of one probe fit (the dual-engine fit behind the test of the optimiser flag, seeds 1 and 2); it is not a single field. Each probe record holds
(a) three flag fields: `optimizer_success`, `duration_optimizer_success`, `duration_fallback_used`, and (b) a fingerprint of every attribute of the fitted model: arrays as a hash, a shape and the first values in hex,
scalars as hex or text. The WP10A summary compared all of these keys bitwise, so the keys under that label that differ are model attributes. This script separates them by kind.
usage: python3 c1_optimizer_fields.py [IN.json] [OUT.txt]"""
from __future__ import annotations

import collections
import json
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "integration_report" / "wp10a" / "scripts"))
from a2_probe_summary import fitfields  # noqa: E402

DEFAULT_IN = ROOT / "wp10a_evidence" / "a2_probe_allkernels.json"
FLAGS = ("optimizer_success", "duration_optimizer_success", "duration_fallback_used")


def is_float_key(k: str, v) -> bool:
    """A stored float: a scalar (key ends with `.f`) or one of the first values of a float array (key contains `.head.`), in hex."""
    return (k.endswith(".f") or ".head." in k) and isinstance(v, str) and v.startswith(("0x", "-0x"))


def kind_of(sub: str, v) -> str:
    """Classify a sub-field key (the part after `<label>.`): flag, array fingerprint (hash/shape/dtype), float value, or other discrete value (counters, integers, booleans, text)."""
    if sub in FLAGS:
        return "flag"
    if sub.endswith((".sha", ".shape.0", ".shape.1", ".shape.2", ".shape.3", ".dtype")):
        return "array_hash_or_shape"
    if is_float_key(sub, v):
        return "float"
    return "discrete"


def analyse(runs: list) -> tuple[dict, list]:
    kernels: list = []
    for r in runs:
        if r["config"]["kernel"] not in kernels:
            kernels.append(r["config"]["kernel"])
    one = {k: next(r for r in runs if r["config"]["kernel"] == k and r["config"]["threads"] == "1" and not r["config"]["rep"]) for k in kernels}
    F = {k: fitfields(one[k]) for k in kernels}
    out: dict = {"kernels": kernels, "labels": {}}
    for seed in (1, 2):
        lab = f"optimizer_success_seed{seed}"
        keys = sorted(k for k in set().union(*[set(f) for f in F.values()]) if k.startswith(lab + "."))
        rows = []
        for k in keys:
            sub = k[len(lab) + 1:]
            vals = [F[x].get(k) for x in kernels]
            distinct = {str(v) for v in vals}
            kind = kind_of(sub, vals[0])
            attr = sub.split(".")[1] if sub.startswith("model.") else sub
            rows.append(dict(key=sub, attr=attr, kind=kind, n_distinct=len(distinct), differs=len(distinct) > 1))
        by_kind = collections.Counter(r["kind"] for r in rows)
        diff_by_kind = collections.Counter(r["kind"] for r in rows if r["differs"])
        flag_values = {f: sorted({str(F[x].get(f"{lab}.{f}")) for x in kernels}) for f in FLAGS}
        diff_attrs = sorted({r["attr"] for r in rows if r["differs"]})
        discrete_rows = [r for r in rows if r["kind"] == "discrete"]
        # largest relative difference of the stored floats over every kernel pair, per differing attribute
        rel: dict = {}
        for a, b in combinations(kernels, 2):
            for r in rows:
                if r["kind"] != "float" or not r["differs"]:
                    continue
                k = f"{lab}.{r['key']}"
                va, vb = F[a].get(k), F[b].get(k)
                if not (isinstance(va, str) and isinstance(vb, str)):
                    continue
                xa, xb = float.fromhex(va), float.fromhex(vb)
                d = abs(xa - xb) / max(abs(xa), abs(xb), 1e-300)
                rel[r["attr"]] = max(rel.get(r["attr"], 0.0), d)
        out["labels"][lab] = dict(n_subfields=len(rows), by_kind=dict(by_kind), differing_by_kind=dict(diff_by_kind), flag_values=flag_values, differing_model_attributes=diff_attrs,
                                  discrete_subfields=len(discrete_rows), discrete_subfields_that_differ=[r["key"] for r in discrete_rows if r["differs"]],
                                  discrete_examples=[r["key"] for r in discrete_rows if any(s in r["key"] for s in ("_em_iteration", "_n_observations", "Dmax", "fit_state"))],
                                  max_rel_diff_of_differing_floats=rel)
    return out, kernels


def render(res: dict) -> str:
    L = [f"WP10B C1(1): sub-fields of optimizer_success_seed1/2 across {len(res['kernels'])} OpenBLAS kernel settings (threads = 1), from the stored WP10A probe data (READ of a stored file; no new fit)."]
    for lab, d in res["labels"].items():
        L.append(f"\n{lab}: {d['n_subfields']} sub-fields compared bitwise; by kind {d['by_kind']}; the ones that differ across kernels, by kind: {d['differing_by_kind']}")
        L.append("  the three flag fields over all kernels: " + "; ".join(f"{f} = {v}" for f, v in d["flag_values"].items()))
        L.append(f"  discrete sub-fields (iteration counter, integer sizes, booleans, text): {d['discrete_subfields']}; of which differ across kernels: {d['discrete_subfields_that_differ'] or 'none'}  (includes {d['discrete_examples']})")
        L.append(f"  model attributes with at least one differing float: {d['differing_model_attributes']}")
        L.append("  largest relative difference of the stored floats over all kernel pairs, per attribute: " + ", ".join(f"{a} {v:.2e}" for a, v in sorted(d["max_rel_diff_of_differing_floats"].items())))
    return "\n".join(L) + "\n"


def main(inp: str | None = None, out: str | None = None) -> None:
    runs = json.load(open(inp or DEFAULT_IN))["runs"]
    res, _ = analyse(runs)
    txt = render(res)
    print(txt)
    if out:
        Path(out).write_text(txt)
        Path(out).with_suffix(".json").write_text(json.dumps(res, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(*(sys.argv[1:3]))
