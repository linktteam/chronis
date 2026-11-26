"""WP10A A6 part 2, POST HOC diagnostic of the harmful-prior development fits (written after the dev readout of H1 and H2 was seen; it changes no criterion and freezes nothing).
Why rule (a) (mean of fit - personal over 36 fits <= -0.25) failed: the WP9 personal fits (no prior) are catastrophic on some 30-day streams, so the mean is carried by a few huge positive differences.
usage: PYTHONPATH=. python3 integration_report/wp10a/scripts/a6_harm_posthoc.py WP9_B1_DEV_FITS.jsonl HARM_DEV_FITS.jsonl OUT.txt"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("wp10a_a6_harm_analyze_posthoc", HERE / "a6_harm_analyze.py")
HA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(HA)
OUT: list = []


def P(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)


def per_length(fits, base):
    """{length: dict(n, mean, median, below_personal, below_gauss, personal_below_gauss)} of fit - personal, finite pairs only."""
    out = {}
    for ln in (30, 60, 120):
        rs = [r for r in fits if r["length"] == ln]
        d, bp, bg, pbg = [], 0, 0, 0
        for r in rs:
            b = base[(r["seed"], r["length"])]
            s = HA.score_of(r)
            if math.isfinite(s) and math.isfinite(b["personal"]):
                d.append(s - b["personal"])
                bp += int(s < b["personal"])
            if math.isfinite(b["gauss"]):
                bg += int(HA.worse_than_gauss(r, b["gauss"]))
                pbg += int(math.isfinite(b["personal"]) and b["personal"] < b["gauss"])
        out[ln] = dict(n=len(rs), mean=float(np.mean(d)) if d else float("nan"), median=float(np.median(d)) if d else float("nan"), below_personal=bp, below_gauss=bg, personal_below_gauss=pbg)
    return out


def main(b1, harm, out):
    base = HA.baseline_from_b1(b1)
    recs = HA.read_jsonl(harm)
    P("A6 HARMFUL-PRIOR CONTROL, POST HOC DIAGNOSTIC (development users 9631..9642; written after the H1 and H2 readouts were seen; no criterion is changed by it)")
    P("fit - personal is in nats per step of the held-out days; 'personal below gauss' counts the pairs where the WP9 personal fit itself scored below the single Gaussian")
    for c in HA.ORDER:
        fits = [r for r in recs if r["arm"] == c]
        if not fits:
            continue
        P(f"\n{c} ({len(fits)} fits)")
        for ln, x in per_length(fits, base).items():
            P(f"  length {ln:>3}: n {x['n']:>2}; fit - personal: mean {x['mean']:+8.3f}, median {x['median']:+7.3f}; fit below personal {x['below_personal']:>2}; fit below the single Gaussian {x['below_gauss']:>2}; personal below the Gaussian {x['personal_below_gauss']:>2}")
    Path(out).write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
