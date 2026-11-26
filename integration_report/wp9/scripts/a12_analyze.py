"""WP9 A1 / A2 analysis (design: tests/PREDECLARED_CRITERIA_WP9.md sections 6 to 9 and part 3). Replays stored fits (`a12_run.py`) through the PRODUCTION lineage.
usage: PYTHONPATH=. python3 integration_report/wp9/scripts/a12_analyze.py dev  wp9_evidence/a12_dev_fits.jsonl  OUT.txt OUT.json
       PYTHONPATH=. python3 integration_report/wp9/scripts/a12_analyze.py eval wp9_evidence/a12_eval_fits.jsonl OUT.txt OUT.json     (needs tests/DESIGN_FREEZE_WP9_A12.txt)
       add --rehearsal to run `eval` on the DEVELOPMENT fits (plumbing test; reads no evaluation seed)
`dev` prints the full coverage / wrong-and-unflagged curve over the declared candidate grid and applies the frozen selection rule (section 8). `eval` applies the frozen candidate once."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import wp9_fit_lib as F
import wp9_replay_lib as R
from wp9_replay_lib import S

from backbone.hssm.lineage import AlignmentConfig

G_T = list(range(90, 147, 7)) + [150]
B_T = [90, 104, 118, 132, 146, 150]
H_GRID = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
S_GRID = (0.0, 0.85, 0.90, 0.95)
STRATA = ("hard", "medium", "easy")
OUT: list = []


def P(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)


def half_of(user: int) -> str:
    return "cal" if user % 2 == 0 else "val"                          # frozen in section 8: even user numbers calibrate, odd validate


# ------------------------------------------------------------------------------------------------------------------------------ one candidate over many chains
def run_candidate(chains: dict, h: float, s: float, u: bool, n: int):
    """Replay every chain under the candidate gate. Returns {user: (rows, lrs)}."""
    cfg = AlignmentConfig(min_separation=h)
    return {k: R.replay9(v, cfg, n=n, use_flag=u, sharpness=s)[:2] for k, v in chains.items()}


def tally_rows(res: dict, users=None) -> dict:
    """WP8 tally over the chains of `users` (all when None) plus the number of accepted refits (steps after the first fit of a chain)."""
    rows = [r for (usr, _), (rs, _) in res.items() if users is None or usr in users for r in rs]
    t = S.tally(rows)
    chains = sum(1 for (usr, _), (rs, _) in res.items() if (users is None or usr in users) and rs)
    t["refits"] = t["steps"] - chains
    return t


def stratum_users(chains: dict) -> dict:
    out = defaultdict(set)
    for (u, _), v in chains.items():
        out[v[0]["job"]["stratum"]].add(u)
    return out


def fmt(t: dict) -> str:
    cov = 100 * t["served"] / t["refits"] if t["refits"] else float("nan")
    return f"coverage {t['served']}/{t['refits']} = {cov:.1f}%, wrong-and-unflagged {t['wrong_unflagged']}/{t['alignments']}"


# ------------------------------------------------------------------------------------------------------------------------------ A1 criteria (identical lineage across two seeds)
def identical_lineage(chains_a: dict, chains_b: dict, cfg: AlignmentConfig, n: int, users=None) -> dict:
    """Both chains are replayed on the 14-day windows. Strict case: a window at which at least one seed ends ALIGNED; identical when both end ALIGNED and the per-day stable ids agree on >= 99%
    of days. WP8 definition: a window fed by at least one seed; identical = both fed, same state, agreement >= 99%. Also counts ALIGNED steps that involve a flagged fit or anchor."""
    strict = [0, 0]; wp8 = [0, 0]; one_flagged = 0; viol = 0; aligned_total = 0; first_same = [0, 0]
    for key_a, ca in chains_a.items():
        u = key_a[0]
        if users is not None and u not in users:
            continue
        cb = next((v for (ub, _), v in chains_b.items() if ub == u), None)
        if cb is None:
            continue
        ra, la, _, _ = R.replay9([r for r in ca if r["job"]["end"] in B_T], cfg, n=n)
        rb, lb, _, _ = R.replay9([r for r in cb if r["job"]["end"] in B_T], cfg, n=n)
        for rows, lrs in ((ra, la), (rb, lb)):
            for r, lr in zip(rows, lrs):
                if r["state"] == "ALIGNED":
                    aligned_total += 1
                    viol += bool(lr.fit_unstable or lr.anchor_fit_unstable)
        da = {r["end"]: (r, lr) for r, lr in zip(ra, la)}
        db = {r["end"]: (r, lr) for r, lr in zip(rb, lb)}
        for end in sorted(set(da) | set(db)):
            a, b = da.get(end), db.get(end)
            agree = a is not None and b is not None and S.day_agreement(a[1].stable_regime_id, b[1].stable_regime_id) >= 0.99
            wp8[0] += bool(a is not None and b is not None and a[0]["state"] == b[0]["state"] and agree); wp8[1] += 1
            sa = a is not None and a[0]["state"] == "ALIGNED"; sb = b is not None and b[0]["state"] == "ALIGNED"
            if sa or sb:
                strict[1] += 1; strict[0] += bool(sa and sb and agree)
            if a is not None and b is not None and a[0]["flag"] != b[0]["flag"]:
                one_flagged += 1
            if end == 90 and a is not None and b is not None and not a[0]["flag"] and not b[0]["flag"]:
                first_same[1] += 1; first_same[0] += bool(agree)
    return dict(strict_identical=strict[0], strict_cases=strict[1], wp8_identical=wp8[0], wp8_cases=wp8[1], exactly_one_seed_flagged=one_flagged, aligned_steps=aligned_total,
                aligned_with_flagged_fit_or_anchor=viol, first_fits_agree=first_same[0], first_fits_both_unflagged=first_same[1])


def fit_level_agreement(chains_a: dict, chains_b: dict, n: int, users=None) -> dict:
    """Windows of the 14-day schedule: the share of windows where both fits are unflagged and the selected partitions are the same (ARI >= 0.99)."""
    both = same = pairs = flagged_either = 0
    for (u, _), ca in chains_a.items():
        if users is not None and u not in users:
            continue
        cb = next((v for (ub, _), v in chains_b.items() if ub == u), None)
        if cb is None:
            continue
        db = {r["job"]["end"]: r for r in cb}
        for ra in ca:
            e = ra["job"]["end"]
            if e not in B_T or e not in db:
                continue
            sa, sb = F.select_restart(ra["restarts"], n), F.select_restart(db[e]["restarts"], n)
            if sa is None or sb is None:
                continue
            pairs += 1
            fa, fb = R.consensus_of(ra, n)["fit_unstable"], R.consensus_of(db[e], n)["fit_unstable"]
            flagged_either += bool(fa or fb)
            if not fa and not fb:
                both += 1; same += bool(F.partition_ari(sa["labels"], sb["labels"]) >= 0.99)
    return dict(pairs=pairs, both_unflagged=both, same_partition=same, flagged_either=flagged_either)


def split_chains(records: list):
    a = R.chains_by_user([r for r in records if r["job"]["fitseed"] == r["job"]["user"]])
    b = R.chains_by_user([r for r in records if r["job"]["fitseed"] != r["job"]["user"]])
    return a, b


def verdict_line(name, k, n, thr, kind):
    return f"{name}: {S.rate(k, n)} -> {S.verdict(k, n, thr, kind)}"


# ------------------------------------------------------------------------------------------------------------------------------ development
def dev(records, out_json):
    a, b = split_chains(records)
    strata = stratum_users(a)
    P(f"DEVELOPMENT: {len(records)} fits, {len(a)} users with seed A chains, {len(b)} with seed B; users per stratum { {k: len(v) for k, v in strata.items()} }")
    wk = {k: [r for r in v if r["job"]["end"] in G_T] for k, v in a.items()}
    curve = {}
    P("\n== curve: every candidate (u = consensus flag applied, h = separation gate, s = posterior sharpness floor); 10 restarts ==")
    for u in (True, False):
        for h in H_GRID:
            for s in S_GRID:
                res = run_candidate(wk, h, s, u, 10)
                row = dict(pooled=tally_rows(res), cal=tally_rows(res, {x for x in {k[0] for k in res} if half_of(x) == "cal"}), val=tally_rows(res, {x for x in {k[0] for k in res} if half_of(x) == "val"}))
                for st in STRATA:
                    row[st] = tally_rows(res, strata[st])
                curve[(u, h, s)] = row
                P(f"u={'on ' if u else 'off'} h={h:<3} s={s:<4} pooled {fmt(row['pooled'])} | cal {fmt(row['cal'])} | val {fmt(row['val'])} | " + " | ".join(f"{st} {row[st]['served']}/{row[st]['refits']} w{row[st]['wrong_unflagged']}" for st in STRATA))
    n_elig = curve[(True, 0.0, 0.0)]["pooled"]["alignments"]
    P(f"\ngate-eligible alignments (accepted fits with the anchor's K): {n_elig} (required >= 400: {'yes' if n_elig >= 400 else 'NO'})")
    # ---- selection rule of section 8
    cands = [(h, s) for h in H_GRID for s in S_GRID]
    key = lambda c, half: (curve[(True,) + c][half]["served"] / max(curve[(True,) + c][half]["refits"], 1), c[0], c[1])
    zero_cal = [c for c in cands if curve[(True,) + c]["cal"]["wrong_unflagged"] == 0]
    chosen, how = None, ""
    if zero_cal:
        c = max(zero_cal, key=lambda c: key(c, "cal"))
        v = curve[(True,) + c]["val"]
        if v["alignments"] and v["wrong_unflagged"] / v["alignments"] <= 0.01:
            chosen, how = c, "step 1-2: highest calibration coverage among candidates with zero wrong-and-unflagged on the calibration half; validation half wrong-and-unflagged <= 1%"
        else:
            P(f"step 2 failed for {c}: validation {v['wrong_unflagged']}/{v['alignments']}")
    if chosen is None:
        zero_all = [c for c in cands if curve[(True,) + c]["pooled"]["wrong_unflagged"] == 0]
        if zero_all:
            chosen = max(zero_all, key=lambda c: key(c, "pooled")); how = "step 3: highest pooled coverage among candidates with zero wrong-and-unflagged on the pooled development set"
        else:
            chosen = min(cands, key=lambda c: (curve[(True,) + c]["pooled"]["wrong_unflagged"], -key(c, "pooled")[0])); how = "no candidate has zero wrong-and-unflagged: the one with the fewest"
    ch = curve[(True,) + chosen]
    P(f"\nCHOSEN (frozen rule of section 8): h = {chosen[0]}, s = {chosen[1]}, consensus flag on. {how}")
    P(f"   pooled {fmt(ch['pooled'])}; calibration {fmt(ch['cal'])}; validation {fmt(ch['val'])}")
    for st in STRATA:
        P(f"   {st:6s} {fmt(ch[st])}")
    cur = curve[(True, 4.0, 0.0)]
    P(f"   for comparison, the WP8 default (h = 4.0, s = 0, flag on): pooled {fmt(cur['pooled'])}; WP8 default without the flag: {fmt(curve[(False, 4.0, 0.0)]['pooled'])}")
    # ---- second look at rule C on the 14-day chains of the development users (not used to tune anything)
    P("\n== rule C on development data, 14-day chains, seed A (fit seed = user) against seed B (fit seed = user + 1000), 10 restarts ==")
    for lab, cfg in (("production default gate h = 4.0", AlignmentConfig()), (f"chosen gate h = {chosen[0]}", AlignmentConfig(min_separation=chosen[0]))):
        il = identical_lineage(a, b, cfg, 10)
        P(f"{lab}: identical lineage among served (strict) {S.rate(il['strict_identical'], il['strict_cases'])}; WP8 definition {S.rate(il['wp8_identical'], il['wp8_cases'])}; windows where exactly one seed is flagged {il['exactly_one_seed_flagged']}; "
          f"ALIGNED steps involving a flagged fit or anchor {il['aligned_with_flagged_fit_or_anchor']} of {il['aligned_steps']}")
    fl = fit_level_agreement(a, b, 10)
    P(f"fit level: windows {fl['pairs']}; both seeds unflagged {fl['both_unflagged']}; same partition among those {S.rate(fl['same_partition'], fl['both_unflagged'])}; flagged in at least one seed {fl['flagged_either']}")
    for st in STRATA:
        il = identical_lineage(a, b, AlignmentConfig(min_separation=chosen[0]), 10, strata[st])
        fl = fit_level_agreement(a, b, 10, strata[st])
        P(f"   {st:6s} identical lineage among served {S.rate(il['strict_identical'], il['strict_cases'])}; fit level same partition among unflagged pairs {S.rate(fl['same_partition'], fl['both_unflagged'])} of {fl['pairs']} windows")
    json.dump(dict(chosen=dict(h=chosen[0], s=chosen[1], u=True, how=how), eligible_alignments=n_elig,
                   curve={f"u={int(k[0])},h={k[1]},s={k[2]}": v for k, v in curve.items()}), open(out_json, "w"), indent=1)


# ------------------------------------------------------------------------------------------------------------------------------ evaluation
def evaluate(records, out_json, gate: dict, ns=(10, 20)):
    a, b = split_chains(records)
    strata = stratum_users(a)
    res_all = {}
    P(f"EVALUATION: {len(records)} fits; users per stratum { {k: len(v) for k, v in strata.items()} }; frozen candidate {gate}")
    for n in ns:
        label = "PRIMARY" if n == 10 else "SECONDARY"
        P(f"\n===== {label}: first {n} restarts =====")
        cfg = AlignmentConfig(min_separation=gate["h"])
        wk = {k: [r for r in v if r["job"]["end"] in G_T] for k, v in a.items()}
        res = run_candidate(wk, gate["h"], gate["s"], gate["u"], n)
        out = {}
        for name, users in [("pooled", None)] + [(st, strata[st]) for st in STRATA]:
            t = tally_rows(res, users)
            il = identical_lineage(a, b, cfg, n, users)
            fl = fit_level_agreement(a, b, n, users)
            out[name] = dict(tally=t, ident=il, fit_level=fl)
            P(f"-- {name}: accepted refits {t['refits']}, alignments {t['alignments']}")
            P("   " + verdict_line("A1 criterion 1 identical lineage among served (strict; target >= 95%)", il["strict_identical"], il["strict_cases"], 0.95, ">="))
            P(f"   WP8 definition (windows fed by >= 1 seed): {S.rate(il['wp8_identical'], il['wp8_cases'])}; windows with exactly one seed flagged {il['exactly_one_seed_flagged']}; first fits (both unflagged) with the same names {il['first_fits_agree']}/{il['first_fits_both_unflagged']}")
            P(f"   A1 criterion 2 ALIGNED steps involving a flagged fit or anchor: {il['aligned_with_flagged_fit_or_anchor']} of {il['aligned_steps']} (target 0)")
            P(f"   A1 criterion 3 served share (accepted refits ending ALIGNED; reported): {S.rate(t['served'], t['refits'])}; fit level: both seeds unflagged {fl['both_unflagged']}/{fl['pairs']} windows, same partition {S.rate(fl['same_partition'], fl['both_unflagged'])}")
            P("   " + verdict_line("A2 coverage (target >= 80%)", t["served"], t["refits"], 0.80, ">="))
            P("   " + verdict_line("A2 wrong-and-unflagged among all alignments (target <= 1%)", t["wrong_unflagged"], t["alignments"], 0.01, "<="))
            P(f"   correct among served: {S.rate(t['served_correct'], t['served'])}; flagged and would have been right {t['flagged_would_be_right']}, wrong {t['flagged_would_be_wrong']}")
        json.dump(out, open(out_json.replace(".json", f"_n{n}.json"), "w"), indent=1)
        # the curve of the evaluation set, reported (never used to choose anything)
        P(f"\n-- coverage / wrong-and-unflagged curve on the evaluation set ({label.lower()}; REPORTED, not used to choose; consensus flag on, s = {gate['s']}) --")
        for h in H_GRID:
            r2 = run_candidate(wk, h, gate["s"], True, n)
            t = tally_rows(r2)
            P(f"   h = {h:<3}: {fmt(t)}")


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    mode, fits, out_txt, out_json = args[:4]
    recs = F.load_records(fits)
    if mode == "dev":
        dev(recs, out_json)
    else:
        freeze = Path("tests/DESIGN_FREEZE_WP9_A12.txt")
        if not freeze.exists() and "--rehearsal" not in sys.argv:
            sys.exit("evaluation needs tests/DESIGN_FREEZE_WP9_A12.txt")
        gate = json.loads(freeze.read_text().splitlines()[0]) if freeze.exists() else dict(h=4.0, s=0.0, u=True)
        evaluate(recs, out_json, gate, ns=(10,) if "--rehearsal" in sys.argv else (10, 20))
    open(out_txt, "w").write("\n".join(OUT) + "\n")
