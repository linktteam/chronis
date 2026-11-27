"""Mutation check for tests/test_wp8_item5_channels.py (WP8 item 5). usage: PYTHONPATH=. python3 integration_report/wp8/scripts/mutation_item5.py [out.txt]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mutation_runner import run

LN, DOC = "backbone/hssm/lineage.py", "docs/INTEGRATION_CONTRACT_HSSM.md"
MUTANTS = [
    ("C01 Hellinger without the square root", LN, "0.5 * ((np.sqrt(P)[:, None, :] - np.sqrt(Q)[None, :, :]) ** 2).sum(axis=2)", "0.5 * ((P[:, None, :] - Q[None, :, :]) ** 2).sum(axis=2)"),
    ("C02 Hellinger without the one half", LN, "return 0.5 * ((np.sqrt(P)", "return 1.0 * ((np.sqrt(P)"),
    ("C03 combined cost without the scale of channel 1", LN, "ch.w1 * D1 / s1 + ch.w2 * D2 / s2", "ch.w1 * D1 + ch.w2 * D2 / s2"),
    ("C04 combined cost without the scale of channel 2", LN, "ch.w1 * D1 / s1 + ch.w2 * D2 / s2", "ch.w1 * D1 / s1 + ch.w2 * D2"),
    ("C05 channel weights swapped", LN, "ch.w1 * D1 / s1 + ch.w2 * D2 / s2", "ch.w2 * D1 / s1 + ch.w1 * D2 / s2"),
    ("C06 the contextual channel is always on", LN, "    if ch.w2 == 0:\n        return alignment_from_cost(D1)", "    if False:\n        return alignment_from_cost(D1)"),
    ("C07 disagreement needs only one confident channel", LN, "disagree=bool(conf1 and conf2 and al1[\"perm\"] != al2[\"perm\"])", "disagree=bool((conf1 or conf2) and al1[\"perm\"] != al2[\"perm\"])"),
    ("C08 disagreement ignores the assignments", LN, "disagree=bool(conf1 and conf2 and al1[\"perm\"] != al2[\"perm\"])", "disagree=bool(conf1 and conf2)"),
    ("C09 disagreement never flagged", LN, "if detail is not None and detail.get(\"disagree\"):", "if False:"),
    ("C10 context channel judged by the physical threshold", LN, "conf1, conf2 = margin_ok(th, al1), margin_ok(ch.context_threshold, al2)", "conf1, conf2 = margin_ok(th, al1), margin_ok(th, al2)"),
    ("C11 anchor profiles not put in stable-id order", LN, "a_ctx_s = a_ctx[list(anchor.stable_to_raw)]", "a_ctx_s = a_ctx"),
    ("C12 a missing profile is not reported", LN, 'dict(context="missing: channel 1 alone was used", weights=(ch.w1, ch.w2))', "None"),
    ("C13 mismatched categories accepted", LN, "if a_ctx.shape[1] != n_ctx.shape[1]:", "if False:"),
    ("C14 profiles need not sum to 1", LN, "if not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):", "if False:"),
    ("C15 negative profiles accepted", LN, "if not np.isfinite(P).all() or (P < 0).any():", "if not np.isfinite(P).all():"),
    ("C16 daily aggregation not renormalised", LN, "out[k] = m / m.sum()", "out[k] = m"),
    ("C17 days without descriptor not skipped", LN, "rows = D[(lab == k) & ~np.isnan(D).any(axis=1)]", "rows = D[(lab == k)]"),
    ("C18 an empty regime accepted", LN, "if len(rows) == 0:", "if False:"),
    ("C19 weights both zero accepted", LN, "(self.w1 == 0 and self.w2 == 0)", "False"),
    ("C20 registry units dropped from the regime table", LN, "unit=f.unit, source=f.source", "unit='', source=f.source"),
    ("C21 the scale of a channel with identical regimes not guarded", LN, "return v if v > 0 else 1.0", "return v"),
    ("C22 K changed is judged by the contextual channel", LN, "    if summary.K != anchor.summary.K:\n        # one-to-one", "    if summary.K != anchor.summary.K and summary.context_profiles is None:\n        # one-to-one"),
    ("C23 contract loses the consent statement", DOC, "from a consented, aggregated, non-identifying pipeline", "from any pipeline"),
    ("C25 an uncalibrated class is not flagged as such", LN, '        if not th.calibrated:\n            reasons.append("threshold_not_calibrated")\n        elif al is not None:', '        if False:\n            reasons.append("threshold_not_calibrated")\n        elif al is not None:'),
    ("C24 the channel info is dropped by NEW_NAMES", LN, "channels_used=record.channels_used, channel_detail=record.channel_detail)", "channels_used=('physical',), channel_detail=None)"),
]

if __name__ == "__main__":
    sys.exit(run(MUTANTS, ["tests/test_wp8_item5_channels.py"], sys.argv[1] if len(sys.argv) > 1 else None, workers=1))
