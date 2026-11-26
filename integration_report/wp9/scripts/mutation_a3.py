"""Mutation check for tests/test_wp9_a3_k_change.py (WP9 A3). usage: PYTHONPATH=. python3 integration_report/wp9/scripts/mutation_a3.py [out.txt]
Pure mutants (k_change.py, the lineage wiring, config) run against the fast tests; the windowed-fit mutants run against the live-fit tests."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp8" / "scripts"))
import mutation_runner as R

_base = shutil.ignore_patterns(".git", "__pycache__", "wp7r_evidence", "wp8_evidence", "wp9_evidence", ".pytest_cache")
R.IGNORE = lambda d, names: set(_base(d, names))

K, LIN, CFG = "backbone/hssm/k_change.py", "backbone/hssm/lineage.py", "backbone/hssm/config.py"
PURE = [
    ("A01 the anchor variance uses ddof 0", K, "out[k, f] = float(v.var(ddof=1))", "out[k, f] = float(v.var(ddof=0))"),
    ("A02 a small regime keeps its own variance", K, "        if len(Xk) < MIN_ROWS_FOR_OWN_VARIANCE:\n            continue", "        if False:\n            continue"),
    ("A03 the own-variance row minimum is 2", K, "MIN_ROWS_FOR_OWN_VARIANCE = 3", "MIN_ROWS_FOR_OWN_VARIANCE = 2"),
    ("A04 the density ignores the variance term", K, "(-0.5 * (np.log(variances[k, obs]) + (X[t, obs] - means[k, obs]) ** 2 / variances[k, obs])).sum()", "(-0.5 * ((X[t, obs] - means[k, obs]) ** 2 / variances[k, obs])).sum()"),
    ("A05 the row goes to the least likely regime", K, "out[t] = int(ll.argmax())", "out[t] = int(ll.argmin())"),
    ("A06 a row with no observation is assigned", K, "        if not obs.any():\n            continue", "        if False:\n            continue"),
    ("A07 unassigned rows count in the shares", K, "rows = (np.asarray(new_labels) == j) & (parent_of_row >= 0)", "rows = (np.asarray(new_labels) == j)"),
    ("A08 the dominant-parent level is exclusive", K, "dominant[j] == i and shares[j, i] >= DOMINANT_PARENT_SHARE]", "dominant[j] == i and shares[j, i] > DOMINANT_PARENT_SHARE]"),
    ("A09 one child is enough for a split", K, "            if len(kids) >= 2:", "            if len(kids) >= 1:"),
    ("A10 the merge level is exclusive (first share)", K, "shares[j, order[0]] >= MERGE_MIN_SHARE and shares[j, order[1]] >= MERGE_MIN_SHARE", "shares[j, order[0]] >= MERGE_MIN_SHARE and shares[j, order[1]] > MERGE_MIN_SHARE"),
    ("A11 the merge pair is not the two largest", K, "sorted(int(x) for x in order[:2])", "sorted(int(x) for x in order[1:3])"),
    ("A12 a tie goes against the anchor K", K, "key=lambda k: (got[k], k != int(k_anchor)))", "key=lambda k: (got[k], k == int(k_anchor)))"),
    ("A13 the margin level is exclusive", K, "    if margin >= need:", "    if margin > need:"),
    ("A14 the two deltas are swapped", K, "need = float(delta_up if direction == \"up\" else delta_down)", "need = float(delta_down if direction == \"up\" else delta_up)"),
    ("A15 a non-finite BIC is accepted", K, "if b is not None and math.isfinite(float(b))}", "if b is not None}"),
    ("A16 a non-adjacent K can be a signal", K, "    if best not in (int(k_anchor) - 1, int(k_anchor) + 1):", "    if False:"),
    ("A17 the whole window is used as the recent window", K, "recent = np.asarray(raw_window, dtype=float)[-int(recent_days):]", "recent = np.asarray(raw_window, dtype=float)"),
    ("A18 a fit that is not accepted is trusted", K, "ok = f is not None and f.get(\"accepted\", False) and", "ok = f is not None and True and"),
    ("A19 an exception in a fit is not contained", K, "            except Exception:                                                               # noqa: BLE001  an unfinished fit is unavailable", "            except KeyError:                                                                # noqa: BLE001  an unfinished fit is unavailable"),
    ("A20 an exception in the assessment escapes", K, "    except Exception as exc:                                                                # noqa: BLE001  fail closed: no claim", "    except KeyError as exc:                                                                 # noqa: BLE001  fail closed: no claim"),
    ("A21 K = 1 is fitted", K, "            if K < 2:", "            if K < 1:"),
    ("A22 the reason of an anchor without a window is changed", K, "return dict(rec, k_best=int(s.K), reason=\"anchor_has_no_window\")", "return dict(rec, k_best=int(s.K), reason=\"anchor_has_no_window_\")"),
    ("A23 the anchor labels are ignored", K, "anchor_within_variances(s.raw_window, s.labels, s.K)", "anchor_within_variances(s.raw_window, np.zeros_like(s.labels), s.K)"),
    ("A24 the anchor means are negated", K, "return assess_k_change(s.raw_means, variances,", "return assess_k_change(-s.raw_means, variances,"),
    ("L01 a suspected change adds no reason", LIN, "        reasons.append(\"k_change_suspected\")", "        pass"),
    ("L02 a truthy value counts as suspected", LIN, "return bool(isinstance(kc, dict) and kc.get(\"suspected\") is True)", "return bool(isinstance(kc, dict) and kc.get(\"suspected\"))"),
    ("L03 the stable id is the raw id", LIN, "c[\"old_stable_regime\"] = int(r2s[int(c[\"old_raw_regime\"])])", "c[\"old_stable_regime\"] = int(c[\"old_raw_regime\"])"),
    ("L04 the merge stable ids are the raw ids", LIN, "c[\"old_stable_regimes\"] = sorted(int(r2s[int(i)]) for i in c[\"old_raw_regimes\"])", "c[\"old_stable_regimes\"] = sorted(int(i) for i in c[\"old_raw_regimes\"])"),
    ("L05 the record does not carry the signal", LIN, "anchor_fit_unstable=False if anchor is None else bool(anchor.summary.fit_unstable), k_change=_k_change_record(summary, anchor))", "anchor_fit_unstable=False if anchor is None else bool(anchor.summary.fit_unstable), k_change=None)"),
    ("L06 a malformed record breaks the lineage", LIN, "    if not isinstance(summary.k_change, dict):                           # None or malformed: nothing known\n        return None", "    if summary.k_change is None:\n        return None"),
    ("L07 the summary drops the record", LIN, "                      None if k_change is None else dict(k_change))", "                      None)"),
    ("C01 the recent window is 40 days", CFG, "K_CHANGE_RECENT_DAYS = 45", "K_CHANGE_RECENT_DAYS = 40"),
    ("C02 the dominant-parent share is 0.6", K, "DOMINANT_PARENT_SHARE = 0.5 ", "DOMINANT_PARENT_SHARE = 0.6 "),
    ("C03 the merge share is 0.3", K, "MERGE_MIN_SHARE = 0.25 ", "MERGE_MIN_SHARE = 0.3 "),
]
LIVE = [
    ("F01 the raw means are left in z units", K, "raw_means=mean + std * np.asarray(res.model.mu, dtype=float)", "raw_means=np.asarray(res.model.mu, dtype=float)"),
    ("F02 the fit never says it is not accepted", K, "accepted=not res.insights_withheld)", "accepted=True)"),
    ("F03 the BIC of another K is returned", K, "bic=float(res.bic_by_k[int(K)])", "bic=float(min(res.bic_by_k.values()))"),
    ("F04 the fast test switch is always on", K, "allow_fast_test_fit=bool(allow_fast_test_fit))", "allow_fast_test_fit=True)"),
]
FAST = "not fit_recent_window"
LIVE_T = "fit_recent_window or production_fit_needs"

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    a = R.run(PURE, ["tests/test_wp9_a3_k_change.py", "-k", FAST], out, workers=1)
    b = R.run(LIVE, ["tests/test_wp9_a3_k_change.py", "-k", LIVE_T], (out + ".live") if out else None, workers=1)
    sys.exit(a or b)
