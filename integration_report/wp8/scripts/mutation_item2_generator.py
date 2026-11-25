"""Mutation check for tests/test_wp8_item2_generator.py (WP8 item 2). usage: PYTHONPATH=. python3 integration_report/wp8/scripts/mutation_item2_generator.py [out.txt]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mutation_runner import run

G, R = "data/synthetic_identity.py", "backbone/hssm/feature_registry.py"
MUTANTS = [
    ("G01 activity order reversed", G, "alpha = np.linspace(-1.0, 1.0, K) * sep", "alpha = np.linspace(1.0, -1.0, K) * sep"),
    ("G02 registry direction ignored", G, "direction * loading * alpha[:, None] + offsets", "loading * alpha[:, None] + offsets"),
    ("G03 regime offsets dropped", G, "direction * loading * alpha[:, None] + offsets", "direction * loading * alpha[:, None]"),
    ("G04 noise persistence 0.9", G, "e[i] = spec.ar * e[i - 1] + c * eps[i]", "e[i] = 0.9 * e[i - 1] + c * eps[i]"),
    ("G05 noise not rescaled to the marginal sd", G, "c = np.sqrt(1.0 - spec.ar ** 2)", "c = 1.0"),
    ("G06 coverage ignored", G, 'present = rng.random((n_days, F)) < u["coverage"]', "present = rng.random((n_days, F)) < 1.0"),
    ("G07 whole missing days dropped", G, "    X_obs[day_missing] = np.nan\n", ""),
    ("G08 missing features not filled with 0", G, "        z = np.where(np.isnan(z), 0.0, z)\n", ""),
    ("G09 sample sd instead of population sd", G, "        std = np.nanstd(raw, axis=0)\n", "        std = np.nanstd(raw, axis=0, ddof=1)\n"),
    ("G10 drift twice as large", G, "            means_t[idx] += frac[idx] * drift_sd * wsd * vec\n        if k_change", "            means_t[idx] += 2 * frac[idx] * drift_sd * wsd * vec\n        if k_change"),
    ("G11 drift starts at day 0", G, "frac = (t / max(n_days - 1, 1))[:, None]", "frac = ((t + 1) / max(n_days - 1, 1))[:, None]"),
    ("G12 split offset 1 sd", G, '+ 2.0 * wsd * u["split_dir"]', '+ 1.0 * wsd * u["split_dir"]'),
    ("G13 split before the change day", G, "if s >= change_day and labels[s] == parent and fl:", "if labels[s] == parent and fl:"),
    ("G14 merge before the change day", G, "later_b = (labels == b) & (t >= change_day)", "later_b = (labels == b)"),
    ("G15 merge into the wrong regime", G, 'means_t[later_b] = u["means"][a]', 'means_t[later_b] = u["means"][b]'),
    ("G16 dwell minimum 1", G, "spec.dwell_min, spec.dwell_max))", "1, spec.dwell_max))"),
    ("G17 regime may repeat", G, "cur = int(rng.choice([k for k in range(K) if k != cur]))", "cur = int(rng.choice(range(K)))"),
    ("G18 pitch gets a direction", G, 'FeatureSpec("pitch", "Hz", "voice", "other", 0)', 'FeatureSpec("pitch", "Hz", "voice", "other", 1)'),
    ("G19 a feature without unit", G, 'FeatureSpec("mean_hr", "bpm",', 'FeatureSpec("mean_hr", "x",'),
    ("R01 source not validated", R, "if self.source not in SOURCES:", "if False:"),
    ("R02 role not validated", R, "if self.role not in ROLES:", "if False:"),
    ("R03 direction not validated", R, "if self.direction not in (-1, 0, 1):", "if False:"),
    ("R04 duplicate names accepted", R, "if len(set(names)) != len(names):", "if False:"),
    ("R05 width not checked", R, "if n_features is not None and len(reg) != n_features:", "if False:"),
    ("R06 empty name accepted", R, "if not self.name or not self.unit:", "if not self.unit:"),
    ("G20 split takes days of every regime", G, "if s >= change_day and labels[s] == parent and fl:", "if s >= change_day and fl:"),
]

if __name__ == "__main__":
    sys.exit(run(MUTANTS, ["tests/test_wp8_item2_generator.py"], sys.argv[1] if len(sys.argv) > 1 else None, workers=2))
