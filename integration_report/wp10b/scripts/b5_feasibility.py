"""WP10B B5: is a dual-to-dual / cross-engine lineage threshold calibration feasible in this package? (rule: tests/PREDECLARED_CRITERIA_WP10B.md Part 10, section 16, committed before any number here.)
Pure arithmetic on numbers that already exist (no fit is run):
  n_required       : smallest n with zero events whose exact one-sided 95% upper bound is at most 1% (A3 level) -> 299
  chains_needed    : chains per stratum for n alignments, under two counts: every window pair of a chain (9 per chain of 10 windows) or only pairs of fits the production policy publishes (9 x publishable rate)
  cost_hours       : fits x seconds per fit / workers, in hours
  evaluate         : conditions (a)+(b) the cost of n >= 299 alignments per stratum, under the more favourable count, is <= 6 h of two-core wall clock (evaluation fits only), (c) >= 20 development alignments pass the gate for each pair type AND a dual stability signal exists
usage: PYTHONPATH=. python3 integration_report/wp10b/scripts/b5_feasibility.py B1_FITS.jsonl B2_FITS.jsonl OUT.txt [VARIANT=I600]
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ALIGNMENTS_PER_CHAIN = 9           # 10 windows per chain (A3: t = 90, 97, ..., 146, 150) -> 9 alignments
FITS_PER_CHAIN = 10
BOUND, LEVEL = 0.01, 0.95
BUDGET_HOURS = 6.0                 # criteria 16.2(a)
WORKERS = 2
MIN_PASSING_DEV = 20               # the WP8 calibration rule: fewer than 20 passing development alignments = UNKNOWN
WP8_DEV_PASSING = {"dual_to_dual": (4, 216), "cross_engine": (16, 480)}   # READ: backbone/hssm/lineage.py, comment above AlignmentConfig
STRATA = ("hard", "medium", "easy")


def exact_upper_zero(n: int, level: float = LEVEL) -> float:
    """Exact one-sided upper bound on a rate with zero events in n trials: 1 - (1 - level)^(1/n)."""
    if n <= 0:
        return 1.0
    return 1.0 - (1.0 - level) ** (1.0 / n)


def n_required(bound: float = BOUND, level: float = LEVEL) -> int:
    """Smallest n for which zero events give an exact one-sided upper bound of at most `bound`."""
    n = max(1, int(math.log(1.0 - level) / math.log(1.0 - bound)) - 2)
    while exact_upper_zero(n, level) > bound:
        n += 1
    return n


def chains_needed(n: int, rate: float = 1.0, per_chain: int = ALIGNMENTS_PER_CHAIN) -> float:
    """Chains needed to have n alignments when a share `rate` of the window pairs counts (rate 1 = every pair). inf when the rate is 0."""
    return math.inf if rate <= 0 else math.ceil(n / (per_chain * rate))


def cost_hours(fits: float, seconds_per_fit: float, workers: int = WORKERS) -> float:
    return fits * seconds_per_fit / workers / 3600.0


def achievable_per_stratum(budget_hours: float, seconds_per_fit: float, workers: int = WORKERS) -> int:
    """Alignments per stratum that the budget buys when every window pair of a chain counts and the three strata share it equally (whole chains only)."""
    fits = budget_hours * 3600.0 * workers / seconds_per_fit
    return int(math.floor(fits / FITS_PER_CHAIN / len(STRATA))) * ALIGNMENTS_PER_CHAIN


def evaluate(rates: dict, seconds_per_fit: float, passing: dict, dual_signal: bool, budget: float = BUDGET_HOURS) -> dict:
    """rates: publishable share per stratum. Returns per-stratum fits and hours under both counts and the three conditions (criteria 16.2)."""
    n = n_required()
    out = {"n_required": n, "strata": {}}
    total_all = total_pub = 0.0
    for s in STRATA:
        c_all = chains_needed(n, 1.0)
        c_pub = chains_needed(n, rates[s])
        out["strata"][s] = dict(chains_all=c_all, fits_all=c_all * FITS_PER_CHAIN, chains_published=c_pub, fits_published=c_pub * FITS_PER_CHAIN)
        total_all += c_all * FITS_PER_CHAIN
        total_pub += c_pub * FITS_PER_CHAIN
    best = min(total_all, total_pub)                                    # the more favourable count
    out["evaluation_fits_all_pairs"] = total_all
    out["evaluation_fits_published_pairs"] = total_pub
    out["hours_all_pairs"] = cost_hours(total_all, seconds_per_fit)
    out["hours_published_pairs"] = cost_hours(total_pub, seconds_per_fit)
    out["hours_favourable"] = cost_hours(best, seconds_per_fit)
    out["a_cost_ok"] = out["hours_favourable"] <= budget            # (a) and (b) of the criteria are one number: the cost of n >= 299 alignments per stratum (evaluation fits only: development fits would add to it)
    out["c_passing_ok"] = all(k >= MIN_PASSING_DEV for k, _ in passing.values()) and bool(dual_signal)
    out["run_b5"] = bool(out["a_cost_ok"] and out["c_passing_ok"])
    out["verdict"] = "run the A3-style calibration" if out["run_b5"] else "NOT DEMONSTRATED (infeasible in this package)"
    return out


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def publishable_by_stratum(records: list, variant: str | None = None) -> tuple[dict, float]:
    """(share publishable per stratum, median fit seconds) of the finished dual fits (optionally one variant)."""
    cnt = {s: [0, 0] for s in STRATA}
    secs = []
    for r in records:
        if not r.get("ok"):
            continue
        if variant is not None and r["job"].get("variant") != variant:
            continue
        s = r["job"]["stratum"]
        cnt[s][1] += 1
        cnt[s][0] += int(bool(r["dual"]["publishable"]))
        secs.append(float(r["dual"]["fit_seconds"]))
    return {s: (cnt[s][0] / cnt[s][1] if cnt[s][1] else 0.0) for s in STRATA}, (statistics.median(secs) if secs else float("nan"))


def code_facts() -> dict:
    """What the production code says about the two pair types and the dual stability signal (RUN against the working tree)."""
    from backbone.hssm import lineage

    cfg = lineage.AlignmentConfig()

    class _Dual:                                    # a dual result: not a baseline model
        is_baseline_model = False

    return dict(dual_engine_calibrated=bool(cfg.dual_engine.calibrated), cross_engine_calibrated=bool(cfg.cross_engine.calibrated),
                dual_stability_flag=tuple(lineage.fit_stability_flag(_Dual())))


def main(argv: list) -> int:
    b1, b2, out_path = argv[1], argv[2], argv[3]
    variant = argv[4] if len(argv) > 4 else "I600"
    lines: list = []

    def P(*a):
        s = " ".join(str(x) for x in a)
        lines.append(s)
        print(s, flush=True)

    s0_rates, s0_sec = publishable_by_stratum(read_jsonl(b1))
    v_rates, v_sec = publishable_by_stratum(read_jsonl(b2), variant)
    facts = code_facts()
    dual_signal = facts["dual_stability_flag"] != (False, None)
    P(f"WP10B B5 FEASIBILITY (criteria Part 10, section 16); inputs: development fits only ({b1}, {b2}); no fit is run here")
    P(f"n required for the A3 level (zero events, one-sided exact 95% bound <= 1%): {n_required()}  (bound at 298: {exact_upper_zero(298):.4%}, at 299: {exact_upper_zero(299):.4%})")
    P(f"code (RUN): dual_engine.calibrated={facts['dual_engine_calibrated']}, cross_engine.calibrated={facts['cross_engine_calibrated']}, fit_stability_flag(dual result)={facts['dual_stability_flag']} -> a dual stability signal exists: {dual_signal}")
    P(f"WP8 development alignments passing the gate (READ, lineage.py): dual to dual {WP8_DEV_PASSING['dual_to_dual'][0]} of {WP8_DEV_PASSING['dual_to_dual'][1]}, cross-engine {WP8_DEV_PASSING['cross_engine'][0]} of {WP8_DEV_PASSING['cross_engine'][1]} (the rule needs >= {MIN_PASSING_DEV})")
    for name, rates, sec in (("S0 (B1)", s0_rates, s0_sec), (f"{variant} (B2, the frozen configuration)", v_rates, v_sec)):
        ev = evaluate(rates, sec, WP8_DEV_PASSING, dual_signal)
        P(f"\n[{name}] publishable share per stratum: " + ", ".join(f"{s} {rates[s]:.2f}" for s in STRATA) + f"; median dual fit {sec:.0f} s")
        for s in STRATA:
            d = ev["strata"][s]
            P(f"   {s}: every pair counted: {d['chains_all']:.0f} chains = {d['fits_all']:.0f} fits; only published pairs counted: " + (f"{d['chains_published']:.0f} chains = {d['fits_published']:.0f} fits" if math.isfinite(d["chains_published"]) else "impossible (publishable share 0)"))
        P(f"   evaluation fits, three strata: every pair {ev['evaluation_fits_all_pairs']:.0f} ({ev['hours_all_pairs']:.1f} h on {WORKERS} workers); published pairs only "
          + (f"{ev['evaluation_fits_published_pairs']:.0f} ({ev['hours_published_pairs']:.1f} h)" if math.isfinite(ev["evaluation_fits_published_pairs"]) else "impossible"))
        P(f"   (a)+(b) favourable cost {ev['hours_favourable']:.1f} h against the {BUDGET_HOURS:.0f} h budget: {'ok' if ev['a_cost_ok'] else 'FAILS'}; (c) >= {MIN_PASSING_DEV} passing development alignments per pair type and a dual stability signal: {'ok' if ev['c_passing_ok'] else 'FAILS'}")
        n_ach = achievable_per_stratum(BUDGET_HOURS, sec)
        P(f"   achievable in the {BUDGET_HOURS:.0f} h budget (every pair counted, whole chains, evaluation fits only): {n_ach} alignments per stratum; with zero wrong the exact one-sided 95% bound would be {exact_upper_zero(n_ach):.2%} (the level asks for <= {BOUND:.0%})")
        P(f"   VERDICT by the declared rule: {ev['verdict']}")
    Path(out_path).write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
