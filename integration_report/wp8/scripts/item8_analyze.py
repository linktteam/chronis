"""WP8 Item 8 analysis: the criteria of tests/PREDECLARED_CRITERIA_WP8.md section 5, evaluated on the fits of item8_run.py through the PRODUCTION lineage (frozen defaults), with the negative controls.
usage: PYTHONPATH=. python3 integration_report/wp8/scripts/item8_analyze.py wp8_evidence/item8_fits.jsonl wp8_evidence/item8_analysis.txt wp8_evidence/item8_results.json
       add --rehearsal to read the DEVELOPMENT fits (wp8_evidence/item4_dev_fits.jsonl, item6_dev_fits.jsonl) in place of the evaluation groups, to test this script's plumbing without touching an evaluation seed.
A criterion is met on the point estimate; when the 95% Wilson interval crosses the threshold the line says NOT DEMONSTRATED. A dual criterion is evaluated only if the dual runs give at least 20 accepted
alignments, otherwise it is UNKNOWN with the count, and the dual lineage is also measured on every fit that did not fail numerically (labelled secondary)."""
from __future__ import annotations

import contextlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lineage_study_lib as S                                                                  # noqa: E402
from identity_lib import load_records                                                          # noqa: E402

from backbone.hssm import lineage as L                                                         # noqa: E402
from backbone.hssm.lineage import AlignmentConfig, ChannelConfig, PairThreshold, make_summary, may_claim_same_as_before, name_fit, step  # noqa: E402
from data.synthetic_identity import BASE, REGISTRY, SCALE                                      # noqa: E402

MIN_DUAL_ALIGNMENTS = 20
OUT: list = []
RES: dict = {}


def P(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)


verdict = S.verdict


def record(key, **kw):
    RES[key] = kw


def nothing_served(t, what):
    """A rate of wrong-and-unflagged alignments is 0 by construction when nothing is ever served; say so next to it so that it is not read as evidence of correct matching."""
    if t["alignments"] > 0 and t["served"] == 0:
        P(f"   NOTE ({what}): nothing was served as ALIGNED (coverage 0%); a wrong-and-unflagged rate of 0 is true by construction and is NOT evidence that the matching is right.")
        return True
    return False


# --------------------------------------------------------------------------------------------------------------------------------------------- loading
def load_groups(path: str, rehearsal: bool) -> dict:
    """Records by group. In rehearsal the DEVELOPMENT fits stand in for the groups they resemble (plumbing test only; no evaluation seed is read)."""
    g = defaultdict(dict)
    def add(name, r):
        g[name][json.dumps(r["job"], sort_keys=True)] = r
    if rehearsal:
        for p in ("wp8_evidence/item4_dev_fits.jsonl", "wp8_evidence/item6_dev_fits.jsonl", "wp8_evidence/item3_dev_fits.jsonl"):
            if Path(p).exists():
                for r in load_records(p):
                    j = r["job"]
                    if p.endswith("item3_dev_fits.jsonl"):
                        if j["engine"] == "baseline" and j["start"] == 0 and j.get("drift", 0.0) == 0.0 and j.get("kchange") is None:
                            add("B-G", r)
                        continue
                    if j["group"] == "baseline-G" and j["drift"] == 0.0 and j.get("kchange") is None:
                        add("B-G", r)
                        if j["user"] in (8201, 8202, 8203, 8204):
                            add("X-B", r)
                        if j["user"] in (8201, 8202, 8203, 8204):
                            add("B-cad", r)
                    elif j["group"] == "baseline-R" and j["drift"] in (1.0, 2.5):
                        add("B-R1" if j["drift"] == 1.0 else "B-R25", r)
                    elif j["group"] == "dual-G":
                        add("D-G", r)
                        if j["drift"] == 0.0 and j["user"] in (8201, 8202, 8203, 8204):
                            add("X-D", r)
                    elif j["group"] == "dual-R" and j["drift"] in (1.0, 2.5):
                        add("D-R1" if j["drift"] == 1.0 else "D-R25", r)
        if Path("wp8_evidence/item8_rehearsal_fits.jsonl").exists():
            for r in load_records("wp8_evidence/item8_rehearsal_fits.jsonl"):          # K-change streams of development users 8201, 8202 (criterion 6 plumbing)
                add("B-K", r)
    else:
        for r in load_records(path):
            add(r["job"]["group"], r)
    return {k: list(v.values()) for k, v in g.items()}


def streams(recs, key=lambda j: (j["user"], j.get("drift", 0.0), j.get("kchange"), j["fitseed"])):
    """Records grouped into streams (user, drift, kchange, fit seed), each sorted by window end (then start)."""
    out = defaultdict(list)
    for r in recs:
        out[key(r["job"])].append(r)
    return {k: sorted(v, key=lambda r: (r["job"]["end"], r["job"]["start"])) for k, v in sorted(out.items(), key=lambda kv: str(kv[0]))}


def replay_all(recs, cfg, accepted_only=True, patch=None, key=None):
    kw = {} if key is None else dict(key=key)
    res = {}
    for k, rs in streams(recs, **kw).items():
        res[k] = S.replay(rs, cfg, accepted_only=accepted_only, patch=patch)
    return res


def pool_rows(res):
    return [r for rows, _, _ in res.values() for r in rows]


def accepted_counts(recs):
    n = len(recs)
    usable = sum(S.usable(r) for r in recs)
    acc = sum(S.usable(r) and bool(r.get("accepted")) for r in recs)
    return n, usable, acc


# --------------------------------------------------------------------------------------------------------------------------------------------- criteria
def crit_1_2(name, recs, cfg, engine):
    """Criterion 1 (>= 95% correct among ALIGNED, <= 1% wrong-and-unflagged among all alignments) and 2 (>= 80% of accepted refits end ALIGNED)."""
    n, usable, acc = accepted_counts(recs)
    P(f"-- {name}: {n} fits, {usable} did not fail numerically, {acc} accepted (publish rule)")
    res = replay_all(recs, cfg)
    t = S.tally(pool_rows(res))
    refits = t["steps"] - len(res)                                                       # every accepted fit but the first of each stream
    P(f"   accepted alignments {t['alignments']} ({refits} accepted refits); served {S.rate(t['served'], t['alignments'])}")
    ok_dual = engine == "baseline" or t["alignments"] >= MIN_DUAL_ALIGNMENTS
    out = dict(counts=t, refits=refits, accepted=acc, fits=n, evaluated=ok_dual)
    served_none = nothing_served(t, name)
    if ok_dual:
        P(f"   criterion 1a correct among served (>= 95%): {S.rate(t['served_correct'], t['served'])} -> {verdict(t['served_correct'], t['served'], 0.95, '>=')}")
        P(f"   criterion 1b wrong-and-unflagged among all alignments (<= 1%): {S.rate(t['wrong_unflagged'], t['alignments'])} -> {verdict(t['wrong_unflagged'], t['alignments'], 0.01, '<=')}")
        P(f"   criterion 2 coverage, accepted refits ending ALIGNED (>= 80%): {S.rate(t['served'], refits)} -> {verdict(t['served'], refits, 0.80, '>=')}")
        out.update(c1a=verdict(t["served_correct"], t["served"], 0.95, ">="), c1b=verdict(t["wrong_unflagged"], t["alignments"], 0.01, "<="), c2=verdict(t["served"], refits, 0.80, ">="), nothing_served=served_none)
    else:
        P(f"   criteria 1 and 2 for the {engine} engine: UNKNOWN (only {t['alignments']} accepted alignments; the declared minimum is {MIN_DUAL_ALIGNMENTS})")
        out.update(c1a="UNKNOWN", c1b="UNKNOWN", c2="UNKNOWN")
    P(f"   conditional on both fits recovering the planted regimes: {S.rate(t['cond_correct'], t['cond_alignments'])} correct; flagged steps that would have been wrong {t['flagged_would_be_wrong']}, right {t['flagged_would_be_right']}")
    if engine == "dual":
        sec = replay_all(recs, cfg, accepted_only=False)
        ts = S.tally(pool_rows(sec))
        P(f"   SECONDARY (every fit that did not fail numerically, including fits the publish rule withholds; not the criterion): alignments {ts['alignments']}, served {S.rate(ts['served'], ts['alignments'])}, "
          f"served and correct {S.rate(ts['served_correct'], ts['served'])}, wrong-and-unflagged {S.rate(ts['wrong_unflagged'], ts['alignments'])}")
        out["secondary"] = ts
    record(name, **out)
    return res


def crit_L(name, recs, cfg, evaluated=True):
    """Ladder: among first fits (window end 90) whose ordering is declared verified, >= 90% match the planted order."""
    res = replay_all(recs, cfg)
    ok = ver = n = 0
    rungs = defaultdict(int)
    for k, rs in streams(recs).items():
        rows, lrs, _ = res[k]
        fed = [r for r in rs if S.usable(r) and r.get("accepted")]
        if not fed or fed[0]["job"]["end"] != 90:
            continue
        n += 1
        lr, info = lrs[0], S.fit_info(fed[0])
        rungs[lr.activity_rung] += 1
        if lr.ordering_verified:
            ver += 1
            order_ok = info.recoverable and all(info.mp[raw] == stable for raw, stable in enumerate(lr.raw_to_stable))
            ok += bool(order_ok)
    P(f"-- criterion L ({name}): first fits {n}; ordering verified {S.rate(ver, n)}; unverified {n - ver}; rung counts {dict(sorted(rungs.items()))}")
    v = verdict(ok, ver, 0.90, ">=") if evaluated else f"UNKNOWN (dual rule: fewer than {MIN_DUAL_ALIGNMENTS} accepted alignments; the counts above are reported, not scored)"
    P(f"   verified orderings that match the planted order (>= 90%): {S.rate(ok, ver)} -> {v}")
    record(f"L:{name}", n=n, verified=ver, correct=ok, rungs=dict(rungs), verdict=v)


def crit_3(name, recs_a, recs_b, cfg):
    """Same windows, different fit seed: the lineage is identical (same state, per-day stable ids agree on >= 99% of days), 100% of cases."""
    ra, rb = replay_all(recs_a, cfg, key=lambda j: j["user"]), replay_all(recs_b, cfg, key=lambda j: j["user"])
    cases = ident = 0
    users_identical = users = 0
    for u in sorted(set(ra) | set(rb)):
        rows_a, lrs_a, _ = ra.get(u, ([], [], None)); rows_b, lrs_b, _ = rb.get(u, ([], [], None))
        da = {r["window"][1]: (r["state"], lr.stable_regime_id) for r, lr in zip(rows_a, lrs_a)}
        db = {r["window"][1]: (r["state"], lr.stable_regime_id) for r, lr in zip(rows_b, lrs_b)}
        same = True
        for end in sorted(set(da) | set(db)):
            cases += 1
            good = end in da and end in db and da[end][0] == db[end][0] and S.day_agreement(da[end][1], db[end][1]) >= 0.99
            ident += good
            same &= good
        users += 1
        users_identical += same
    P(f"-- criterion 3 ({name}): windows fed by at least one of the two seeds {cases}; identical (same state, stable ids agree on >= 99% of days) {S.rate(ident, cases)} -> {verdict(ident, cases, 1.0, '>=')} "
      f"(the criterion is 100%); users whose whole lineage is identical {users_identical}/{users}")
    record(f"3:{name}", cases=cases, identical=ident, users=users, users_identical=users_identical, verdict=verdict(ident, cases, 1.0, ">="))


def crit_4(name, recs, cfg, res=None, evaluated=True):
    """Window growth: stable ids of the first 90 days from each later ALIGNED fit agree with the first fit's on >= 95% of days, in >= 90% of the ALIGNED fits; raw ids reported next to it."""
    res = res or replay_all(recs, cfg)
    n = good = 0
    raw_agree, st_agree = [], []
    for k, rs in streams(recs).items():
        rows, lrs, _ = res[k]
        fed = [r for r in rs if S.usable(r) and r.get("accepted")]
        if len(fed) < 2 or fed[0]["job"]["end"] != 90:
            continue
        first_ids = np.asarray(lrs[0].stable_regime_id)[:90]
        first_raw = np.asarray(fed[0]["labels_fit"])[:90]
        for i in range(1, len(fed)):
            raw_agree.append(S.day_agreement(first_raw, np.asarray(fed[i]["labels_fit"])[:90]))
            if rows[i]["state"] == "ALIGNED":
                a = S.day_agreement(first_ids, np.asarray(lrs[i].stable_regime_id)[:90])
                st_agree.append(a); n += 1; good += a >= 0.95
    ma = float(np.mean(raw_agree)) if raw_agree else float("nan")
    v4 = verdict(good, n, 0.90, ">=") if evaluated else f"UNKNOWN (dual rule: fewer than {MIN_DUAL_ALIGNMENTS} accepted alignments)"
    P(f"-- criterion 4 ({name}): ALIGNED later fits {n}; stable ids agree on >= 95% of the first 90 days in {S.rate(good, n)} (>= 90%) -> {v4}; "
      f"mean stable agreement {np.mean(st_agree) if st_agree else float('nan'):.3f}; raw-id agreement of the same stream's later fits (not relabelled): mean {ma:.3f}, share >= 95% {S.rate(sum(a >= 0.95 for a in raw_agree), len(raw_agree))}")
    record(f"4:{name}", n=n, good=good, verdict=v4, raw_mean=ma)
    return n, good


def crit_5(tag, recs, cfg, floor, engine, thr):
    n, usable, acc = accepted_counts(recs)
    res = replay_all(recs, cfg)
    t = S.tally(pool_rows(res))
    refits = t["steps"] - len(res)
    P(f"-- criterion 5 ({tag}, {engine}): {n} fits, {acc} accepted; accepted alignments {t['alignments']}")
    if engine == "dual" and t["alignments"] < MIN_DUAL_ALIGNMENTS:
        P(f"   UNKNOWN ({t['alignments']} accepted alignments, minimum {MIN_DUAL_ALIGNMENTS})")
        sec = S.tally(pool_rows(replay_all(recs, cfg, accepted_only=False)))
        P(f"   SECONDARY (all fits that did not fail numerically): alignments {sec['alignments']}, served {S.rate(sec['served'], sec['alignments'])}, correct among served {S.rate(sec['served_correct'], sec['served'])}")
        record(f"5:{tag}:{engine}", status="UNKNOWN", alignments=t["alignments"], secondary=sec)
        return
    nothing_served(t, f"criterion 5 {tag} {engine}")
    v = verdict(t["served_correct"], t["served"], thr, ">=")
    P(f"   correct among served (>= {int(thr * 100)}%): {S.rate(t['served_correct'], t['served'])} -> {v}")
    P(f"   coverage (served among accepted refits; floor {int(floor * 100)}% reported for 1 sd): {S.rate(t['served'], refits)}"
      + (f" -> {verdict(t['served'], refits, floor, '>=')}" if floor else " (reported, no floor)"))
    record(f"5:{tag}:{engine}", status=v, counts=t, refits=refits)


def _truth(job):
    from identity_lib import stream_for
    return stream_for(dict(job, drift=job.get("drift", 0.0)))


def crit_6(recs, cfg):
    """K change: K_CHANGED and no 'same as before' in 100% of refits whose fitted K differs from the anchor's; candidate precision and the share of changes the K selection sees are REPORTED."""
    res = replay_all(recs, cfg, key=lambda j: (j["user"], j.get("kchange"), j["fitseed"]))
    changed = ok = 0
    cand_total = cand_correct = 0
    sees_final = streams_n = 0
    post = [0, 0, 0]                                       # post-change fits (window end >= 118): fitted K == K after, K == 3, other
    for (user, kc, seed), (rows, lrs, _) in res.items():
        for r, lr in zip(rows, lrs):
            if "K_anchor" in r and r["K"] != r["K_anchor"]:
                changed += 1
                ok += bool(r["state"] == "K_CHANGED" and not may_claim_same_as_before(lr))
                tr = _truth(dict(user=user, kchange=kc, start=0, end=150)).truth["user"]
                for c in r.get("candidates", []):
                    cand_total += 1
                    if kc == "split" and c["kind"] == "split":
                        cand_correct += c["planted"] == [tr["split_parent"]]
                    elif kc == "merge" and c["kind"] == "merge":
                        cand_correct += c["planted"] == sorted(tr["merge_pair"])
        after = 4 if kc == "split" else 2
        fits = [r for r in streams(recs, key=lambda j: (j["user"], j.get("kchange"), j["fitseed"]))[(user, kc, seed)] if S.usable(r)]
        late = [r for r in fits if r["job"]["end"] >= 118]
        for r in late:
            post[0 if r["K"] == after else 1 if r["K"] == 3 else 2] += 1
        if fits and fits[-1]["job"]["end"] == 150:
            streams_n += 1; sees_final += fits[-1]["K"] == after
    P(f"-- criterion 6 (K change, baseline, candidate K (2, 3, 4), streams {len(res)}): refits whose fitted K differs from the anchor's {changed}; K_CHANGED and no 'same as before' {S.rate(ok, changed)} (must be 100%) -> {verdict(ok, changed, 1.0, '>=')}")
    P(f"   REPORTED, not pass/fail: split/merge candidate precision {S.rate(cand_correct, cand_total)} (expected >= 80%); K selection sees the change: final fit has the changed K in {S.rate(sees_final, streams_n)} of the streams; "
      f"post-change fits (end >= 118): K after the change {post[0]}, still 3 {post[1]}, other {post[2]}")
    record("6", changed=changed, ok=ok, verdict=verdict(ok, changed, 1.0, ">="), cand_total=cand_total, cand_correct=cand_correct, sees_final=sees_final, streams=streams_n, post=post)


def crit_7(xb, xd, cfg):
    """Engine switch: baseline and dual fits of the same window, both directions, through the production lineage (cross-engine thresholds)."""
    B = {(r["job"]["user"], r["job"]["end"]): r for r in xb}
    D = {(r["job"]["user"], r["job"]["end"]): r for r in xd}
    out = {}
    for label, acc_only in (("criterion", True), ("SECONDARY (all fits that did not fail numerically)", False)):
        rows = []
        for key in sorted(set(B) & set(D)):
            for first, second in ((B[key], D[key]), (D[key], B[key])):
                rws, _, _ = S.replay([first, second], cfg, accepted_only=acc_only)
                rows += rws
        t = S.tally(rows)
        out[label] = t
        if acc_only:
            nothing_served(t, "criterion 7, accepted cross-engine alignments")
            if t["alignments"] >= MIN_DUAL_ALIGNMENTS:
                P(f"-- criterion 7 (engine switch, both directions, accepted fits): alignments {t['alignments']}, served {S.rate(t['served'], t['alignments'])}; correct among served (>= 95%) "
                  f"{S.rate(t['served_correct'], t['served'])} -> {verdict(t['served_correct'], t['served'], 0.95, '>=')}; wrong-and-unflagged (<= 1%) {S.rate(t['wrong_unflagged'], t['alignments'])} -> {verdict(t['wrong_unflagged'], t['alignments'], 0.01, '<=')}")
            else:
                P(f"-- criterion 7 (engine switch): UNKNOWN (only {t['alignments']} accepted cross-engine alignments, minimum {MIN_DUAL_ALIGNMENTS})")
        else:
            P(f"   {label}: alignments {t['alignments']}, served {S.rate(t['served'], t['alignments'])}, correct among served {S.rate(t['served_correct'], t['served'])}, wrong-and-unflagged {S.rate(t['wrong_unflagged'], t['alignments'])}; "
              f"flagged would have been wrong {t['flagged_would_be_wrong']}, right {t['flagged_would_be_right']}")
    record("7", **{k: v for k, v in out.items()})


def crit_8(bg, cad, cfg):
    """Cadence: daily (8301..8312 only), weekly and every 14 days: each meets criterion 1; the stable ids at day 150 of the schedules agree on >= 95% of days."""
    recs = {}
    for r in bg + cad:
        recs[(r["job"]["user"], r["job"]["end"])] = r
    users = sorted({u for u, _ in recs})
    rows_by, ids150, used = defaultdict(list), defaultdict(dict), defaultdict(list)
    for u in users:
        mine = [r for (uu, t), r in recs.items() if uu == u]
        for i in (1, 7, 14):
            try:
                fits = S.fits_for_days(mine, S.schedule_days(i, 90, 150))
            except KeyError:
                continue
            rows, lrs, _ = S.replay(fits, cfg)
            if not lrs or rows[-1]["window"][1] != 150:
                continue
            rows_by[i] += rows; ids150[i][u] = lrs[-1].stable_regime_id; used[i].append(u)
    names = {1: "daily", 7: "weekly", 14: "every 14 days"}
    sched = {}
    for i in (1, 7, 14):
        t = S.tally(rows_by[i])
        sched[i] = t
        nothing_served(t, f"criterion 8 {names[i]}")
        P(f"-- criterion 8 ({names[i]}, {len(used[i])} users): alignments {t['alignments']}; correct among served (>= 95%) {S.rate(t['served_correct'], t['served'])} -> {verdict(t['served_correct'], t['served'], 0.95, '>=')}; "
          f"wrong-and-unflagged (<= 1%) {S.rate(t['wrong_unflagged'], t['alignments'])} -> {verdict(t['wrong_unflagged'], t['alignments'], 0.01, '<=')}; served {S.rate(t['served'], t['steps'] - len(used[i]))}")
    pooled = {}
    for a, b in ((1, 7), (1, 14), (7, 14)):
        us = sorted(set(ids150[a]) & set(ids150[b]))
        if us:
            ag = [S.day_agreement(ids150[a][u], ids150[b][u]) for u in us]
            pooled[(a, b)] = (float(np.mean(ag)), min(ag), sum(x >= 0.95 for x in ag), len(us))
            P(f"   day-150 stable-id agreement {names[a]} vs {names[b]}: pooled {np.mean(ag):.3f}, worst user {min(ag):.3f}, users at or above 0.95: {sum(x >= 0.95 for x in ag)}/{len(us)} (criterion >= 95% of days)")
    worst = min((v[0] for v in pooled.values()), default=float("nan"))
    verdict_ag = "UNKNOWN" if not pooled else ("MET" if worst >= 0.95 else "NOT MET")
    P(f"   criterion 8, day-150 agreement of the three schedules (pooled days, worst pair {worst:.3f}): {verdict_ag}")
    record("8", schedules={names[i]: sched[i] for i in sched}, pooled={f"{names[a]}|{names[b]}": v for (a, b), v in pooled.items()}, verdict_agreement=verdict_ag, users={names[i]: len(used[i]) for i in used})


# --------------------------------------------------------------------------------------------------------------------------------------------- negative controls
@contextlib.contextmanager
def patch_zspace():
    """(a) the matching distance without the stored scaler: every fit compared in its own z-space."""
    old = L._vectors
    L._vectors = lambda summary, scale, duration_weight: (summary.raw_means - summary.scaler_mean) / np.where(summary.scaler_std > 0, summary.scaler_std, 1.0)
    try:
        yield
    finally:
        L._vectors = old


def patch_random_perm(seed=0):
    """(b) the assignment replaced by a random permutation (the margins stay those of the real solution)."""
    @contextlib.contextmanager
    def ctx():
        old = L.alignment_from_cost
        rng = np.random.default_rng(seed)
        def rnd(cost):
            al = dict(old(cost))
            al["perm"] = tuple(int(x) for x in rng.permutation(cost.shape[0]))
            return al
        L.alignment_from_cost = rnd
        try:
            yield
        finally:
            L.alignment_from_cost = old
    return ctx


def control_fails(recs, cfg, patch, label):
    res = replay_all(recs, cfg, patch=patch)
    t = S.tally(pool_rows(res))
    refits = t["steps"] - len(res)
    v1a, v1b, v2 = verdict(t["served_correct"], t["served"], 0.95, ">="), verdict(t["wrong_unflagged"], t["alignments"], 0.01, "<="), verdict(t["served"], refits, 0.80, ">=")
    fails = "NOT MET" in (v1a, v1b)                          # declared in the Item 8 run plan: a control fails through CORRECTNESS (1a or 1b); coverage alone cannot discriminate (the frozen design itself misses 80%)
    cov_only = (not fails) and v2 == "NOT MET"
    P(f"   {label}: served {S.rate(t['served'], t['alignments'])}; correct among served {S.rate(t['served_correct'], t['served'])} -> {v1a}; wrong-and-unflagged {S.rate(t['wrong_unflagged'], t['alignments'])} -> {v1b}; "
      f"coverage {S.rate(t['served'], refits)} -> {v2}; control "
      f"{'FAILS as required (correctness)' if fails else ('misses only the coverage criterion, which the frozen design also misses: NOT DISCRIMINATING, reported as inert' if cov_only else 'DID NOT FAIL (inert on this data)')}")
    return fails, t


def control_c(recs, cfg):
    """(c) the ordering re-sorted at every refit: each fit gets the names of its own ladder instead of inheriting them by alignment; criterion 4 must fail."""
    n = good = 0
    for k, rs in streams(recs).items():
        fed = [r for r in rs if S.usable(r) and r.get("accepted")]
        if len(fed) < 2 or fed[0]["job"]["end"] != 90:
            continue
        ids = [np.asarray(name_fit(S.summary_for(r), REGISTRY, cfg.ladder)[0])[np.asarray(r["labels_fit"])] for r in fed]
        for i in range(1, len(fed)):
            n += 1
            good += S.day_agreement(ids[0][:90], ids[i][:90]) >= 0.95
    v = verdict(good, n, 0.90, ">=")
    P(f"   (c) ordering re-sorted at every refit: later fits {n}; stable ids agree on >= 95% of the first 90 days in {S.rate(good, n)} (>= 90%) -> {v}; control {'FAILS as required' if v == 'NOT MET' else 'DID NOT FAIL (inert on this data)'}")
    return v == "NOT MET", dict(n=n, good=good, verdict=v)


def _synthetic_pair(rng, physical_decisive: bool, K=3, F=7, C=6, T=240):
    """An anchor fit and a relabelled refit with two channels; one channel separates the regimes, the other carries no information."""
    if physical_decisive:
        means = BASE + SCALE * rng.normal(0, 2.5, (K, F))
        ctx = np.tile(np.ones(C) / C, (K, 1))
    else:
        means = BASE + SCALE * 0.03 * rng.normal(size=(K, F))
        ctx = np.full((K, C), 0.05)
        for k in range(K):
            ctx[k, k] = 1.0 - 0.05 * (C - 1)
    ctx = ctx / ctx.sum(axis=1, keepdims=True)
    labels = rng.permutation(np.repeat(np.arange(K), T // K))
    def fit(m, c, perm=None):
        raw = m[labels] + 0.35 * SCALE * rng.standard_normal((len(labels), F))
        mm = m if perm is None else m[perm]
        cc = c if perm is None else c[perm]
        lab = labels if perm is None else np.argsort(perm)[labels]
        return make_summary("baseline", mm + 0.03 * SCALE * rng.standard_normal(mm.shape), raw.mean(axis=0), raw.std(axis=0), lab, raw, context_profiles=cc)
    perm = rng.permutation(K)
    return fit(means, ctx), fit(means, ctx, perm), perm


def control_d(n_trials=100, seed=0):
    """(d) a distance that ignores one channel, on synthetic data where the other channel is decisive, both directions. The gate is off (min_separation 0) to isolate the distance."""
    systems = {"physical only (w2 = 0)": (1.0, 0.0), "context only (w1 = 0)": (0.0, 1.0), "both channels (w1 = w2 = 1)": (1.0, 1.0)}
    out, rng = {}, np.random.default_rng(seed)
    for direction, phys in (("physical channel decisive", True), ("context channel decisive", False)):
        per = {}
        pairs = [_synthetic_pair(rng, phys) for _ in range(n_trials)]
        for sname, (w1, w2) in systems.items():
            cfg = AlignmentConfig(min_separation=0.0, channels=ChannelConfig(w1=w1, w2=w2, context_threshold=PairThreshold("ratio", 0.0, 0.5)))
            served = correct = wrong = 0
            for a, n, perm in pairs:
                _, st = step(None, a, cfg, REGISTRY, "ctl")
                rec, _ = step(st, n, cfg, REGISTRY)
                if rec.continuity_state == "ALIGNED":
                    served += 1
                    r2s_anchor = st.anchor.raw_to_stable
                    ok = all(rec.raw_to_stable[j] == r2s_anchor[int(perm[j])] for j in range(len(perm)))
                    correct += ok; wrong += (not ok)
            v1a, v1b, v2 = verdict(correct, served, 0.95, ">="), verdict(wrong, n_trials, 0.01, "<="), verdict(served, n_trials, 0.80, ">=")
            out[(direction, sname)] = dict(served=served, correct=correct, wrong=wrong, v1a=v1a, v1b=v1b, v2=v2, fails="NOT MET" in (v1a, v1b, v2))
            per[sname] = (served, wrong)
            P(f"   (d) {direction:27s} | {sname:28s}: served {served}/{n_trials}, wrong-and-unflagged {wrong}/{n_trials}, correct among served {correct}/{served} -> criterion 1a {v1a}, 1b {v1b}, coverage {v2}")
        # (d) is judged against the both-channel system on the same data (run plan, item 5): a single-channel distance FAILS iff criterion 1a, 1b or 2 is NOT MET and it is clearly worse than the
        # both-channel system: serves at least 20 fewer of the trials, or has at least 5 more wrong-and-unflagged ones.
        sb, wb = per["both channels (w1 = w2 = 1)"]
        for sname in ("physical only (w2 = 0)", "context only (w1 = 0)"):
            sc, wc = per[sname]
            o = out[(direction, sname)]
            worse = (sb - sc >= 20) or (wc - wb >= 5)
            o["worse_than_both"] = bool(worse); o["fails"] = bool(o["fails"] and worse)
            P(f"   (d) {direction:27s} | {sname:28s}: versus both channels (served {sb}, wrong {wb}): {'clearly worse' if worse else 'not clearly worse'}; control "
              f"{'FAILS as required' if o['fails'] else 'DID NOT FAIL (inert in this direction)'}")
    req = [("context channel decisive", "physical only (w2 = 0)"), ("physical channel decisive", "context only (w1 = 0)")]
    both = all(out[k]["fails"] for k in req)
    P(f"   (d) required: the physical-only distance fails where the context decides, and the context-only distance fails where the physical decides: "
      f"{'both occurred, control FAILS as required' if both else 'NOT both occurred: control DID NOT FAIL in at least one required direction (inert)'}")
    out["required_both_failed"] = both
    return out


# --------------------------------------------------------------------------------------------------------------------------------------------- main
def main(fits_path, out_path, json_path, rehearsal=False):
    G = load_groups(fits_path, rehearsal)
    cfg = AlignmentConfig()
    P("WP8 Item 8 analysis" + (" (REHEARSAL on development fits; no evaluation seed read)" if rehearsal else ""))
    P("production defaults:", cfg)
    P("expert review status of the matching distance:", L.EXPERT_REVIEW_STATUS)
    P("groups present:", {k: len(v) for k, v in sorted(G.items())})
    P("")
    P("== Criteria 1 and 2, L, 3, 4 (protocol G, drift 0)")
    if "B-G" in G:
        crit_1_2("baseline B-G", G["B-G"], cfg, "baseline"); crit_L("baseline", G["B-G"], cfg); crit_4("baseline", G["B-G"], cfg)
    if "B-G" in G and "B-G2" in G:
        crit_3("baseline, seed vs seed + 1000", G["B-G"], G["B-G2"], cfg)
    if "D-G" in G:
        crit_1_2("dual D-G", G["D-G"], cfg, "dual")
        dual_ok = bool(RES["dual D-G"]["evaluated"])
        crit_L("dual", G["D-G"], cfg, evaluated=dual_ok); crit_4("dual", G["D-G"], cfg, evaluated=dual_ok)
    P("")
    P("== Criterion 5 (drift, protocol R)")
    for tag, thr, floor in (("R1", 0.95, 0.80), ("R25", 0.90, 0.0)):
        if f"B-{tag}" in G:
            crit_5(f"{'1 sd' if tag == 'R1' else '2.5 sd'}", G[f"B-{tag}"], cfg, floor, "baseline", thr)
        if f"D-{tag}" in G:
            crit_5(f"{'1 sd' if tag == 'R1' else '2.5 sd'}", G[f"D-{tag}"], cfg, floor, "dual", thr)
    P("")
    if "B-K" in G:
        P("== Criterion 6 (K change)"); crit_6(G["B-K"], cfg); P("")
    if "X-B" in G and "X-D" in G:
        P("== Criterion 7 (engine switch)"); crit_7(G["X-B"], G["X-D"], cfg); P("")
    if "B-G" in G:
        P("== Criterion 8 (cadence)"); crit_8(G["B-G"], G.get("B-cad", []), cfg); P("")
    P("== Criterion 9 (negative controls; each must fail)")
    ctl = {}
    if "B-G" in G:
        P("   on protocol G (baseline, drift 0):")
        ctl["a_G"] = control_fails(G["B-G"], cfg, patch_zspace, "(a) z-space, no stored scaler")[0]
        ctl["b_G"] = control_fails(G["B-G"], cfg, patch_random_perm(0), "(b) random permutation")[0]
        ctl["c"] = control_c(G["B-G"], cfg)[0]
    for tag in ("R1", "R25"):
        if f"B-{tag}" in G:
            P(f"   on protocol R ({tag}):")
            ctl[f"a_{tag}"] = control_fails(G[f"B-{tag}"], cfg, patch_zspace, "(a) z-space, no stored scaler")[0]
            ctl[f"b_{tag}"] = control_fails(G[f"B-{tag}"], cfg, patch_random_perm(0), "(b) random permutation")[0]
    d = control_d()
    ctl["d"] = {f"{k[0]}|{k[1]}": v for k, v in d.items() if isinstance(k, tuple)}
    ctl["d_required_both_failed"] = d.get("required_both_failed")
    record("controls", **ctl)
    Path(out_path).write_text("\n".join(OUT) + "\n")
    Path(json_path).write_text(json.dumps(RES, indent=1, default=str))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0], args[1], args[2], rehearsal="--rehearsal" in sys.argv)
