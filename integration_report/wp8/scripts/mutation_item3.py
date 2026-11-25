"""Mutation check for tests/test_wp8_item3_ordering.py (WP8 item 3). usage: PYTHONPATH=. python3 integration_report/wp8/scripts/mutation_item3.py [out.txt]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mutation_runner import run

O = "backbone/hssm/ordering.py"
W = "w[f] = coverage[f] * snr / (1.0 + snr)"
MUTANTS = [
    ("O01 coverage not in the weight", O, W, "w[f] = snr / (1.0 + snr)"),
    ("O02 SNR not in the weight", O, W, "w[f] = coverage[f]"),
    ("O03 SNR weight not saturating", O, W, "w[f] = coverage[f] * snr"),
    ("O04 registry direction ignored", O, "sign[f] = spec.direction", "sign[f] = 1.0"),
    ("O05 order descending", O, 'return tuple(int(i) for i in np.argsort(score, kind="stable"))', 'return tuple(int(i) for i in np.argsort(-score, kind="stable"))'),
    ("O06 rung 2 uses every feature", O, '2: ("movement", "time_moving")', "2: None"),
    ("O07 rung 3 also uses time moving", O, '3: ("movement",)}', '3: ("movement", "time_moving")}'),
    ("O08 gap condition removed", O, " and gaps.min() >= config.min_gap", ""),
    ("O09 frequency condition removed", O, " and freq >= config.min_modal_freq", ""),
    ("O10 modal order need not equal the point order", O, "modal_order == order and ", ""),
    ("O11 agreement check removed", O, "if any(rungs[r].order != rungs[top].order for r in reliable[1:]):", "if False:"),
    ("O12 lowest reliable rung preferred", O, "top = reliable[0]", "top = reliable[-1]"),
    ("O13 unverified order not first-fit", O, "first_fit = tuple(range(K))", "first_fit = tuple(reversed(range(K)))"),
    ("O14 bootstrap always the same block", O, "pick = rng.integers(0, n_blocks, n_blocks)", "pick = np.zeros(n_blocks, dtype=int)"),
    ("O15 bootstrap labels not resampled with the days", O, "_rung_weights_scores(raw[idx], labels[idx], K, registry, rung)", "_rung_weights_scores(raw[idx], labels, K, registry, rung)"),
    ("O16 scores in raw units, not z-scores", O, "z = (means - mean_f) / sd_f", "z = (means - mean_f)"),
    ("O17 weights not normalised", O, "w = w / w.sum()", "w = w"),
    ("O18 a regime without observations still usable", O, "usable = (n[:, f] >= 1).all() and ", "usable = "),
    ("O19 modal frequency reported as 1", O, "return order, c / cfg.n_bootstrap", "return order, 1.0"),
    ("O20 only the first gap checked", O, "gaps = np.diff(np.sort(score))", "gaps = np.diff(np.sort(score))[:1]"),
    ("O21 bootstrap not seeded", O, "rng = np.random.default_rng(config.seed)", "rng = np.random.default_rng()"),
    ("O22 min_modal_freq validation removed", O, "if not (0.5 < self.min_modal_freq <= 1.0):", "if False:"),
    ("O23 label range not validated", O, "if K < 1 or labels.min() < 0 or labels.max() >= K:", "if K < 1:"),
    ("O24 within variance divided by T", O, "ss / np.maximum(n_f, 1)", "ss / T"),
    ("O25 grand mean unweighted", O, "grand = np.nansum(n_f * means[:, f]) / n_f.sum()", "grand = np.nanmean(means[:, f])"),
    ("O26 between variance unweighted", O, "between = np.nansum(n_f * (means[:, f] - grand) ** 2) / n_f.sum()", "between = np.nanmean((means[:, f] - grand) ** 2)"),
    ("O28 frozen min_gap default changed", O, "min_gap: float = 0.7 ", "min_gap: float = 0.5 "),
    ("O29 frozen block length changed", O, "block_days: int = 7\n", "block_days: int = 5\n"),
    ("O27 blocks overlap (stride 1)", O, "starts = np.arange(n_blocks) * cfg.block_days", "starts = np.arange(n_blocks)"),
]

if __name__ == "__main__":
    sys.exit(run(MUTANTS, ["tests/test_wp8_item3_ordering.py"], sys.argv[1] if len(sys.argv) > 1 else None, workers=1))
