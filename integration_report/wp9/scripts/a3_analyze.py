"""WP9 A3 analysis (design: tests/PREDECLARED_CRITERIA_WP9.md Part 7). usage:
  PYTHONPATH=. python3 integration_report/wp9/scripts/a3_analyze.py dev  FITS.jsonl OUT.txt [OUT.json]   # the declared delta rule; prints the two chosen thresholds
  PYTHONPATH=. python3 integration_report/wp9/scripts/a3_analyze.py eval FITS.jsonl OUT.txt [OUT.json]   # the declared levels at the thresholds frozen in backbone/hssm/config.py
The signal, the decision and the candidates are the PRODUCTION functions (`backbone.hssm.k_change.assess_k_change`, applied offline to the stored recent-window fits); the lineage records are
produced by the production `lineage.step`."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "wp8" / "scripts"))
import lineage_study_lib as S                                                # noqa: E402
import wp9_fit_lib as W                                                      # noqa: E402
from identity_lib import overlap_map                                         # noqa: E402

from backbone.hssm import config as C                                        # noqa: E402
from backbone.hssm import k_change as KC                                     # noqa: E402
from backbone.hssm.lineage import AlignmentConfig, K_CHANGED, make_summary, may_claim_same_as_before, step   # noqa: E402
from data.synthetic_identity import REGISTRY                                 # noqa: E402

GRID = (0, 3, 6, 10, 20, 40)
REFIT_ENDS = (118, 134, 150)
RECENT = 45
FA_CAP = 0.10
LEVEL_DETECT, LEVEL_PRECISION = 0.50, 0.80


# ------------------------------------------------------------------------------------------------------------------------------ loading
def load(path):
    out = []
    for ln in open(path):
        ln = ln.strip()
        if ln:
            out.append(json.loads(ln))
    return out


def index(recs):
    idx = {}
    for r in recs:
        j = r["job"]
        K = j["K"] if j["kind"] != "ksel" else None
        idx[(j["kind"], j["user"], j["stream"], j["end"], K)] = r
    return idx


def usable(rec):
    return bool(rec and rec.get("ok") and rec.get("accepted") and rec.get("labels_fit") is not None)


def build_streams(idx):
    """One dict per (user, stream) whose anchor fit exists and was accepted; the others are listed as unusable."""
    users = sorted({k[1] for k in idx}); out, skipped = [], []
    for u in users:
        for s in ("split", "merge", "none"):
            a = idx.get(("anchor", u, s, 90, 3))
            if not usable(a):
                skipped.append((u, s)); continue
            st = W.stream_for(dict(user=u, kchange=None if s == "none" else s, stratum="wp8", drift=0.0))
            raw0 = st.X_obs[0:90]
            labels = W.str_to_labels(a["labels_fit"]); truth = W.str_to_labels(a["labels_true"])
            mp, recoverable = overlap_map(labels.tolist(), truth.tolist(), 3, 4)
            out.append(dict(user=u, stream=s, st=st, anchor=a, anchor_means=np.asarray(a["raw_means"], dtype=float), anchor_labels=labels, mp=mp, recoverable=recoverable, raw0=raw0,
                            anchor_var=KC.anchor_within_variances(raw0, labels, 3), truth=st.truth["user"], idx=idx))
    return out, skipped


def stored_fit_fn(S_, t):
    def fit(recent, K):
        r = S_["idx"].get(("refit", S_["user"], S_["stream"], t, K))
        if not r or not r.get("ok"):
            return None
        return dict(bic=r["bic"], labels=W.str_to_labels(r["labels_fit"]), raw_means=np.asarray(r["raw_means"], dtype=float), accepted=bool(r["accepted"]))
    return fit


def assess(S_, t, delta_up, delta_down):
    """The PRODUCTION assessment of one refit of one stream."""
    recent = S_["st"].X_obs[t - RECENT:t]
    return KC.assess_k_change(S_["anchor_means"], S_["anchor_var"], recent, fit_fn=stored_fit_fn(S_, t), k_anchor=3, recent_days=RECENT, delta_up=delta_up, delta_down=delta_down)


# ------------------------------------------------------------------------------------------------------------------------------ pure scoring helpers
def split_correct(cands, mp, parent):
    """A signal's candidates are right when a split candidate names the anchor regime that overlaps the planted parent."""
    return any(c["kind"] == "split" and 0 <= int(c["old_raw_regime"]) < len(mp) and mp[int(c["old_raw_regime"])] == int(parent) for c in cands)


def merge_correct(cands, mp, pair):
    return any(c["kind"] == "merge" and sorted(mp[int(i)] for i in c["old_raw_regimes"]) == sorted(int(x) for x in pair) for c in cands)


def streams_with(per_t, direction):
    """True when at least one refit carries a suspected change of that direction."""
    return any(a["suspected"] and a["direction"] == direction for a in per_t)


def counts(streams, du, dd):
    """Per stream kind: the number of streams and of streams with at least one `up` / `down` signal at the refits of the stored assessments (decisions made at (du, dd))."""
    out = {s: dict(n=0, up=0, down=0) for s in ("split", "merge", "none")}
    for S_ in streams:
        per_t = [assess(S_, t, du, dd) for t in REFIT_ENDS]
        o = out[S_["stream"]]
        o["n"] += 1; o["up"] += streams_with(per_t, "up"); o["down"] += streams_with(per_t, "down")
    return out


def pick_delta(fa_by_delta, grid=GRID, cap=FA_CAP):
    """The declared development rule for one direction: the smallest delta whose false-alarm rate on the no-change streams is at most `cap`; if none, the largest and the shortfall is reported.
    `fa_by_delta` maps delta -> (false-alarm streams, no-change streams). Returns (delta, cap_met)."""
    for d in grid:
        k, n = fa_by_delta[d]
        if n > 0 and k / n <= cap:
            return d, True
    return grid[-1], False


def fmt(k, n):
    return S.rate(k, n)


# ------------------------------------------------------------------------------------------------------------------------------ development
def report_dev(recs, out_txt, out_json):
    idx = index(recs); streams, skipped = build_streams(idx)
    table = {d: counts(streams, float(d), float(d)) for d in GRID}
    fa_up = {d: (table[d]["none"]["up"], table[d]["none"]["n"]) for d in GRID}
    fa_dn = {d: (table[d]["none"]["down"], table[d]["none"]["n"]) for d in GRID}
    d_up, ok_up = pick_delta(fa_up); d_dn, ok_dn = pick_delta(fa_dn)
    L = [f"A3 development, users 9221..9232: {len(streams)} usable streams (unusable anchors: {skipped})", "",
         "delta | up: split streams detected | no-change false alarm | merge streams (up, reported) || down: merge streams detected | no-change false alarm | split streams (down, reported)"]
    for d in GRID:
        t = table[d]
        L.append(f"{d:>5} | up {t['split']['up']:>2}/{t['split']['n']:<2}  FA {t['none']['up']:>2}/{t['none']['n']:<2}  merge-up {t['merge']['up']:>2}/{t['merge']['n']:<2} || "
                 f"down {t['merge']['down']:>2}/{t['merge']['n']:<2}  FA {t['none']['down']:>2}/{t['none']['n']:<2}  split-down {t['split']['down']:>2}/{t['split']['n']:<2}")
    L += ["", f"CHOSEN (declared rule): delta_up = {d_up} (10% false-alarm cap met: {ok_up}), delta_down = {d_dn} (cap met: {ok_dn})",
          f"  at the chosen deltas: split detection {fmt(table[d_up]['split']['up'], table[d_up]['split']['n'])}; up false alarm {fmt(*fa_up[d_up])}; merge detection {fmt(table[d_dn]['merge']['down'], table[d_dn]['merge']['n'])}; "
          f"down false alarm {fmt(*fa_dn[d_dn])}"]
    open(out_txt, "w").write("\n".join(L) + "\n")
    if out_json:
        json.dump(dict(delta_up=d_up, delta_down=d_dn, cap_met_up=ok_up, cap_met_down=ok_dn, table={str(d): table[d] for d in GRID}, unusable=skipped), open(out_json, "w"), indent=1)
    print("\n".join(L[-3:]))


# ------------------------------------------------------------------------------------------------------------------------------ evaluation
def summary_of(S_, t, K, rec):
    raw = S_["st"].X_obs[t - RECENT:t]
    return make_summary("baseline", np.asarray(rec["raw_means"], dtype=float), np.asarray(rec["scaler_mean"], dtype=float), np.asarray(rec["scaler_std"], dtype=float),
                        W.str_to_labels(rec["labels_fit"]), raw)


def lineage_checks(streams, du, dd):
    """Every refit of every stream through the production lineage. (i) the K = a fit with the K-change record attached; (ii) every refit at K != the anchor's (K_CHANGED). Counts the records that
    are `k_change_suspected` or K_CHANGED and how many of them could be served as 'same as before' (must be 0). Also returns the old nearest-mean rule's candidates for the `up` / `down` signals."""
    cfg = AlignmentConfig()
    n_susp = n_kch = bad = 0
    old_rule = []                                                                 # (stream, t, direction, candidates in planted terms) for every signal
    for S_ in streams:
        anchor_summary = make_summary("baseline", S_["anchor_means"], np.asarray(S_["anchor"]["scaler_mean"]), np.asarray(S_["anchor"]["scaler_std"]), S_["anchor_labels"], S_["raw0"])
        _, state0 = step(None, anchor_summary, cfg, REGISTRY, lineage_id=f"a3-{S_['user']}-{S_['stream']}")
        r2s = list(state0.anchor.raw_to_stable)
        for t in REFIT_ENDS:
            kc = assess(S_, t, du, dd)
            same = S_["idx"].get(("refit", S_["user"], S_["stream"], t, 3))
            if usable(same):
                s3 = summary_of(S_, t, 3, same)
                s3 = make_summary("baseline", s3.raw_means, s3.scaler_mean, s3.scaler_std, s3.labels, s3.raw_window, k_change=kc)
                rec, _ = step(state0, s3, cfg, REGISTRY)
                if kc["suspected"]:
                    n_susp += 1; bad += bool(may_claim_same_as_before(rec)) or rec.continuity_state != "CONTINUITY_UNVERIFIED" or "k_change_suspected" not in rec.reasons
            for K in (2, 4):
                r = S_["idx"].get(("refit", S_["user"], S_["stream"], t, K))
                if not usable(r):
                    continue
                rec, _ = step(state0, summary_of(S_, t, K, r))
                n_kch += 1; bad += bool(may_claim_same_as_before(rec)) or rec.continuity_state != K_CHANGED
                if kc["suspected"] and kc["direction"] == ("up" if K == 4 else "down"):
                    planted = []
                    for c in rec.split_merge_candidates:
                        if c["kind"] == "split":
                            planted.append(dict(kind="split", planted=[int(S_["mp"][r2s.index(c["old_stable_id"])])]))
                        else:
                            planted.append(dict(kind="merge", planted=sorted(int(S_["mp"][r2s.index(i)]) for i in c["old_stable_ids"])))
                    old_rule.append((S_["stream"], t, kc["direction"], planted, S_["truth"]))
    return n_susp, n_kch, bad, old_rule


def report_eval(recs, out_txt, out_json):
    du, dd = float(C.K_CHANGE_DELTA_UP), float(C.K_CHANGE_DELTA_DOWN)
    freeze = Path("tests/DESIGN_FREEZE_WP9_A3.txt")
    frz = json.loads(freeze.read_text().splitlines()[0]) if freeze.exists() else None
    idx = index(recs); streams, skipped = build_streams(idx)
    cnt = counts(streams, du, dd)
    L = [f"A3 EVALUATION, users 9241..9252: {len(streams)} usable streams (unusable anchors: {skipped}); thresholds delta_up = {du}, delta_down = {dd}; freeze file {frz}", ""]
    if frz is None or float(frz["delta_up"]) != du or float(frz["delta_down"]) != dd:
        L.append("!! the thresholds in config.py do not equal the freeze file: this evaluation is NOT valid")
    sp, mg, nn = cnt["split"], cnt["merge"], cnt["none"]
    det = fmt(sp["up"], sp["n"])
    L += [f"(a) split detection >= {LEVEL_DETECT:.0%} (a stream counts when an `up` signal occurs at some refit): {det} -> {S.verdict(sp['up'], sp['n'], LEVEL_DETECT, '>=')}",
          f"(b) false alarm <= {FA_CAP:.0%} per direction on the no-change streams: up {fmt(nn['up'], nn['n'])} -> {S.verdict(nn['up'], nn['n'], FA_CAP, '<=')}; "
          f"down {fmt(nn['down'], nn['n'])} -> {S.verdict(nn['down'], nn['n'], FA_CAP, '<=')}"]
    # candidates
    sig_split = [(S_, t, assess(S_, t, du, dd)) for S_ in streams if S_["stream"] == "split" for t in REFIT_ENDS]
    sig_split = [(S_, t, a) for S_, t, a in sig_split if a["suspected"] and a["direction"] == "up"]
    ok_c = sum(split_correct(a["candidates"], S_["mp"], S_["truth"]["split_parent"]) for S_, t, a in sig_split)
    sig_merge = [(S_, t, assess(S_, t, du, dd)) for S_ in streams if S_["stream"] == "merge" for t in REFIT_ENDS]
    sig_merge = [(S_, t, a) for S_, t, a in sig_merge if a["suspected"] and a["direction"] == "down"]
    ok_m = sum(merge_correct(a["candidates"], S_["mp"], S_["truth"]["merge_pair"]) for S_, t, a in sig_merge)
    L.append(f"(c) split-candidate precision >= {LEVEL_PRECISION:.0%} among the `up` signals of the split streams (per signal): {fmt(ok_c, len(sig_split))} -> {S.verdict(ok_c, len(sig_split), LEVEL_PRECISION, '>=')}")
    # lineage and the old rule
    n_susp, n_kch, bad, old_rule = lineage_checks(streams, du, dd)
    L.append(f"(d) records that are k_change_suspected ({n_susp}) or K_CHANGED ({n_kch}) and could be served as 'same as before' or lack the reason: {bad} (must be 0) -> "
             f"{'MET' if bad == 0 and (n_susp + n_kch) > 0 else ('UNKNOWN (nothing to count)' if (n_susp + n_kch) == 0 else 'NOT MET')}")
    old_split = [(p, tr) for kind, t, d, p, tr in old_rule if kind == "split" and d == "up"]
    old_ok = sum(any(c["kind"] == "split" and c["planted"] == [int(tr["split_parent"])] for c in p) for p, tr in old_split)
    old_merge = [(p, tr) for kind, t, d, p, tr in old_rule if kind == "merge" and d == "down"]
    old_mok = sum(any(c["kind"] == "merge" and c["planted"] == sorted(int(x) for x in tr["merge_pair"]) for c in p) for p, tr in old_merge)
    # K selection (WP8 definition)
    ks = {s: [idx.get(("ksel", S_["user"], s, 150, None)) for S_ in streams if S_["stream"] == s] for s in ("split", "merge")}
    see = {s: sum(1 for r in v if r and r.get("ok") and r["k_selected"] == (4 if s == "split" else 2)) for s, v in ks.items()}
    nsee = {s: sum(1 for r in v if r and r.get("ok")) for s, v in ks.items()}
    L += ["", "REPORTED without a level:",
          f"  merge detection (a `down` signal at some refit of a merge stream): {fmt(mg['down'], mg['n'])}; false alarm on the no-change streams, `down`: {fmt(nn['down'], nn['n'])}",
          f"  merge-candidate precision among the `down` signals of the merge streams: {fmt(ok_m, len(sig_merge))}",
          f"  the OLD nearest-mean candidates on the same signals: split (K' = 4 refit, `up` signals of the split streams) {fmt(old_ok, len(old_split))}; merge (K' = 2 refit) {fmt(old_mok, len(old_merge))}",
          f"  WP8 comparison, K selection (2, 3, 4) on the final window [0, 150) sees the change: split streams {fmt(see['split'], nsee['split'])}, merge streams {fmt(see['merge'], nsee['merge'])} (WP8: 3/24 = 12.5%; candidate precision 3/38 = 7.9%)",
          f"  the wrong-direction signals: `down` in split streams {fmt(sp['down'], sp['n'])}; `up` in merge streams {fmt(mg['up'], mg['n'])}"]
    open(out_txt, "w").write("\n".join(L) + "\n")
    if out_json:
        json.dump(dict(delta_up=du, delta_down=dd, counts=cnt, split_candidates=[ok_c, len(sig_split)], merge_candidates=[ok_m, len(sig_merge)], records=dict(suspected=n_susp, k_changed=n_kch, bad=bad),
                       old_rule=dict(split=[old_ok, len(old_split)], merge=[old_mok, len(old_merge)]), k_selection=dict(see=see, n=nsee), unusable=skipped), open(out_json, "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    mode, fits, out = sys.argv[1], sys.argv[2], sys.argv[3]
    js = sys.argv[4] if len(sys.argv) > 4 else None
    recs = load(fits)
    (report_dev if mode == "dev" else report_eval)(recs, out, js)
