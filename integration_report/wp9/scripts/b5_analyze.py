"""WP9 B5: detection and false-alarm rates of the monitor simulation (levels of tests/PREDECLARED_CRITERIA_WP9.md section 24). usage:
  PYTHONPATH=. python3 integration_report/wp9/scripts/b5_analyze.py wp9_evidence/b5_sim.jsonl OUT.txt [OUT.json]"""
import json, math, sys

CHANGE_DAY, END_DAY = 135, 179
TARGET_DETECTION, TARGET_FALSE_ALARM = 0.70, 0.10


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def verdict(k, n, target, op):
    p = k / n if n else float("nan"); lo, hi = wilson(k, n)
    if op == ">=":
        if lo >= target:
            return "MET"
        return "MET on the point estimate, NOT DEMONSTRATED" if p >= target else "NOT MET"
    if hi <= target:
        return "MET"
    return "MET on the point estimate, NOT DEMONSTRATED" if p <= target else "NOT MET"


def tally(recs, stream, series):
    users = [r for r in recs if r.get("ok") and r["streams"][stream][series]["status"] == "ok"]
    n = len(users)
    pre = sum(1 for r in users if any(d < CHANGE_DAY for d in r["streams"][stream][series]["event_days"]))
    post = [r for r in users if any(CHANGE_DAY <= d <= END_DAY for d in r["streams"][stream][series]["event_days"])]
    lat = sorted(min(d for d in r["streams"][stream][series]["event_days"] if CHANGE_DAY <= d <= END_DAY) - CHANGE_DAY for r in post)
    anyev = sum(1 for r in users if r["streams"][stream][series]["event_days"])
    return dict(n=n, pre_change_events=pre, detected=len(post), any_event=anyev, latency_median=(lat[len(lat) // 2] if lat else None), latencies=lat)


def main():
    recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    recs = [r for r in recs if r.get("ok")]
    L, out = [], {}
    L.append(f"B5 monitor simulation: {len(recs)} users (seeds 9901..9912); fitted K = 3 on days 0..89, change at day {CHANGE_DAY}, scored days 90..179")
    L.append(f"fits withheld by the publish rule (information only, the monitor does not need a published fit): {sum(1 for r in recs if r['withheld'])}/{len(recs)}")
    for series, label in (("S1", "S1 raw score vs trailing window (primary, levels apply)"), ("S2", "S2 gain over the single Gaussian (reported, no level)")):
        L.append(f"\n== {label}")
        t0 = tally(recs, "none", series)
        L.append(f"false alarm on the none streams: {t0['any_event']}/{t0['n']} = {100 * t0['any_event'] / max(t0['n'], 1):.1f}% (Wilson {wilson(t0['any_event'], t0['n'])[0]*100:.1f}-{wilson(t0['any_event'], t0['n'])[1]*100:.1f}%)"
                 + (f"   level <= 10%: {verdict(t0['any_event'], t0['n'], TARGET_FALSE_ALARM, '<=')}" if series == "S1" else ""))
        out[series] = dict(none=t0)
        for stream in ("regime_shift", "sensor_change"):
            t = tally(recs, stream, series)
            out[series][stream] = t
            lo, hi = wilson(t["detected"], t["n"])
            L.append(f"{stream}: detected {t['detected']}/{t['n']} = {100 * t['detected'] / max(t['n'], 1):.1f}% (Wilson {lo*100:.1f}-{hi*100:.1f}%); events before day {CHANGE_DAY}: {t['pre_change_events']}/{t['n']}; "
                     f"latency median {t['latency_median']} days (all {t['latencies']})" + (f"   level >= 70%: {verdict(t['detected'], t['n'], TARGET_DETECTION, '>=')}" if series == "S1" else ""))
    L.append("\nmean score before / after the change (S1), per stream, over the users:")
    for stream in ("none", "regime_shift", "sensor_change"):
        b = [r["streams"][stream]["S1"]["mean_before"] for r in recs]; a = [r["streams"][stream]["S1"]["mean_after"] for r in recs]
        L.append(f"  {stream}: before {sum(b)/len(b):.2f}, after {sum(a)/len(a):.2f} nats per day")
    open(sys.argv[2], "w").write("\n".join(L) + "\n")
    if len(sys.argv) > 3:
        json.dump(out, open(sys.argv[3], "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
