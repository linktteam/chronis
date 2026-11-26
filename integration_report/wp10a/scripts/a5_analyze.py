"""WP10A A5 analysis (design: tests/PREDECLARED_CRITERIA_WP10A.md Part 4, sections 8-11). usage:
  PYTHONPATH=. python3 integration_report/wp10a/scripts/a5_analyze.py dev  FITS.jsonl OUT.txt OUT.json
  PYTHONPATH=. python3 integration_report/wp10a/scripts/a5_analyze.py eval FITS.jsonl OUT.txt OUT.json [--rehearsal]      (needs tests/DESIGN_FREEZE_WP10A_A5.txt: first line {"monitor": "MA"|"MB"|"MC"})
Levels (WP9, not moved): false alarm on the none streams <= 10%; detection of the regime shift >= 70%; of the sensor change >= 70%. Alarmed = any event on any scoreable day of the none stream; detected = an
event on or after day 135 of a change stream. Verdicts as in WP9 `b5_analyze.verdict` (Wilson 95%)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "wp9" / "scripts"))
import a5_monitors as M                                                              # noqa: E402
import b5_analyze as B9                                                              # noqa: E402  (WP9: wilson, verdict)

CHANGE_DAY, END_DAY, TRAIN_END = 135, 179, 90
LEVEL_FA, LEVEL_DET = 0.10, 0.70
ORDER = ("MA", "MB", "MC")
FREEZE = Path("tests/DESIGN_FREEZE_WP10A_A5.txt")
OUT: list = []


def P(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)


def series_of(rec, stream):
    """(held-out scores, held-out posterior, last training posterior row, training scores, training posterior) of one stream of one user; days without any observation are NaN scores."""
    st = rec["streams"][stream]
    inc, post, miss = np.array(st["inc"]), np.array(st["post"]), np.array(st["miss"])
    s = inc.copy()
    s[miss] = np.nan
    return s[TRAIN_END:], post[TRAIN_END:], post[TRAIN_END - 1], s[:TRAIN_END], post[:TRAIN_END]


def run_user(rec, cand, stream):
    sc, post, prev, tr, trp = series_of(rec, stream)
    res = M.run_candidate(cand, sc, post, tr, trp, prev)
    return dict(status=res["status"], event_days=[TRAIN_END + e["day"] for e in res["events"]])


def tally(per_user: list, stream: str) -> dict:
    """per_user: [dict(status, event_days)] for the usable users of one stream."""
    users = [r for r in per_user if r["status"] == "ok"]
    n = len(users)
    if stream == "none":
        k = sum(1 for r in users if r["event_days"])
        return dict(n=n, k=k, rate=(k / n if n else float("nan")))
    pre = sum(1 for r in users if any(d < CHANGE_DAY for d in r["event_days"]))
    post = [r for r in users if any(CHANGE_DAY <= d <= END_DAY for d in r["event_days"])]
    lat = sorted(min(d for d in r["event_days"] if CHANGE_DAY <= d <= END_DAY) - CHANGE_DAY for r in post)
    return dict(n=n, k=len(post), rate=(len(post) / n if n else float("nan")), pre_change_events=pre, latency_median=(lat[len(lat) // 2] if lat else None))


def shortfall(fa: float, det_shift: float, det_sensor: float) -> float:
    return max(0.0, fa - LEVEL_FA) + max(0.0, LEVEL_DET - det_shift) + max(0.0, LEVEL_DET - det_sensor)


def eligible(fa: float, det_shift: float, det_sensor: float) -> bool:
    return fa <= LEVEL_FA and det_shift >= LEVEL_DET and det_sensor >= LEVEL_DET


def choose(summary: dict):
    """The frozen development rule: among the candidates whose point estimates meet all three levels take the lowest false-alarm rate (ties: higher regime-shift detection, then the order MA, MB, MC); if
    none is eligible take the smallest total shortfall (ties: the order). `summary`: {candidate: dict(fa, det_shift, det_sensor)}. Returns (candidate, how)."""
    cands = [c for c in ORDER if c in summary]
    el = [c for c in cands if eligible(summary[c]["fa"], summary[c]["det_shift"], summary[c]["det_sensor"])]
    if el:
        best = min(el, key=lambda c: (summary[c]["fa"], -summary[c]["det_shift"], ORDER.index(c)))
        return best, "eligible (all three levels met on the point estimates); lowest false-alarm rate"
    best = min(cands, key=lambda c: (shortfall(summary[c]["fa"], summary[c]["det_shift"], summary[c]["det_sensor"]), ORDER.index(c)))
    return best, "NONE ELIGIBLE: the smallest total shortfall"


def evaluate_set(recs: list, cands: list) -> dict:
    out = {}
    for c in cands:
        out[c] = {}
        for stream in ("none", "regime_shift", "sensor_change"):
            out[c][stream] = tally([run_user(r, c, stream) for r in recs], stream)
    return out


def verdict_line(c: str, t: dict) -> str:
    f = t["none"]; s = t["regime_shift"]; n = t["sensor_change"]
    wf, ws, wn = B9.wilson(f["k"], f["n"]), B9.wilson(s["k"], s["n"]), B9.wilson(n["k"], n["n"])
    return (f"{c}: false alarm {f['k']}/{f['n']} = {100 * f['rate']:.1f}% (Wilson {100 * wf[0]:.1f}-{100 * wf[1]:.1f}%) [{B9.verdict(f['k'], f['n'], LEVEL_FA, '<=')}]; "
            f"regime shift {s['k']}/{s['n']} = {100 * s['rate']:.1f}% (Wilson {100 * ws[0]:.1f}-{100 * ws[1]:.1f}%, median latency {s['latency_median']}, events before day {CHANGE_DAY}: {s['pre_change_events']}) [{B9.verdict(s['k'], s['n'], LEVEL_DET, '>=')}]; "
            f"sensor change {n['k']}/{n['n']} = {100 * n['rate']:.1f}% (Wilson {100 * wn[0]:.1f}-{100 * wn[1]:.1f}%, median latency {n['latency_median']}, events before day {CHANGE_DAY}: {n['pre_change_events']}) [{B9.verdict(n['k'], n['n'], LEVEL_DET, '>=')}]")


def load(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in recs if r.get("ok")], sum(1 for r in recs if not r.get("ok"))


def dev(fits, out_json):
    recs, failed = load(fits)
    P(f"A5 DEVELOPMENT: {len(recs)} users fitted ({failed} failed fits excluded); fits withheld by the publish rule (kept): {sum(1 for r in recs if r['withheld'])}")
    ev = evaluate_set(recs, ["M0"] + list(ORDER))
    for c in ["M0"] + list(ORDER):
        P("  " + verdict_line(c, ev[c]))
    summ = {c: dict(fa=ev[c]["none"]["rate"], det_shift=ev[c]["regime_shift"]["rate"], det_sensor=ev[c]["sensor_change"]["rate"]) for c in ORDER}
    best, how = choose(summ)
    P(f"\nCHOSEN BY THE FROZEN RULE: {best} ({how}); shortfalls: " + ", ".join(f"{c} {shortfall(summ[c]['fa'], summ[c]['det_shift'], summ[c]['det_sensor']):.3f}" for c in ORDER))
    json.dump(dict(chosen=best, how=how, summary=summ, tallies=ev), open(out_json, "w"), indent=1, default=float)


def evaluate(fits, out_json, rehearsal):
    if not FREEZE.exists():
        sys.exit("the evaluation needs tests/DESIGN_FREEZE_WP10A_A5.txt")
    chosen = json.loads(FREEZE.read_text().splitlines()[0])["monitor"]
    if chosen not in ORDER:
        sys.exit(f"unknown monitor {chosen!r} in the freeze file")
    recs, failed = load(fits)
    if not rehearsal and any(r["user"] < 10451 for r in recs):
        sys.exit("the evaluation file contains development users")
    P(f"A5 EVALUATION (once){' (REHEARSAL on development fits)' if rehearsal else ''}: frozen monitor {chosen}; {len(recs)} users fitted ({failed} failed fits excluded); withheld by the publish rule (kept): {sum(1 for r in recs if r['withheld'])}")
    ev = evaluate_set(recs, ["M0", chosen])
    P("  " + verdict_line(chosen, ev[chosen]))
    P("  " + verdict_line("M0 (the production monitor of WP9, same users, for comparison)", ev["M0"]))
    t = ev[chosen]
    P(f"\nSensor change, WP9 result 12/12 = 100%: now {t['sensor_change']['k']}/{t['sensor_change']['n']} ({'no regression' if t['sensor_change']['rate'] >= LEVEL_DET else 'BELOW THE LEVEL'}).")
    ok = eligible(t["none"]["rate"], t["regime_shift"]["rate"], t["sensor_change"]["rate"])
    P(f"All three levels met on the point estimates: {ok}. " + ("The chosen monitor may be ported as an additive function of monitoring.py." if ok else "Not all levels are met: the production primitives stay as in WP9; no monitor is used as an alarm."))
    json.dump(dict(monitor=chosen, tallies=ev, all_levels_met_on_point_estimates=ok), open(out_json, "w"), indent=1, default=float)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args and args[0] == "dev" and len(args) == 4:
        dev(args[1], args[3])
        Path(args[2]).write_text("\n".join(OUT) + "\n")
    elif args and args[0] == "eval" and len(args) == 4:
        evaluate(args[1], args[3], "--rehearsal" in sys.argv)
        Path(args[2]).write_text("\n".join(OUT) + "\n")
    else:
        sys.exit(__doc__)
