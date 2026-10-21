"""Reference battery generator (section 4.5).

Runs the tiny exact oracle against production on 10 tiny configurations and records the
OBSERVED discrepancies to tests/oracle/reference_battery.json. Tolerances in the tests are
then checked against this file. Re-run it deliberately, not to make a failing test pass.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.oracle.tiny_exact_oracle import run_tiny_exact_oracle
from tests.utils.model_factory import make_tiny_model, make_tiny_data

CONFIGS = [
    # (K, F, L, Dmax, T, seed, nan_prob). Oracle limits: T<=6, K<=2, Dmax<=3.
    # K=1 needs Dmax >= T (zero-diagonal A_slow and no other regime to move to).
    (1, 1, 1, 3, 3, 0, 0.0),
    (1, 2, 1, 3, 3, 1, 0.0),
    (1, 1, 2, 3, 2, 2, 0.0),
    (2, 1, 1, 2, 2, 3, 0.0),
    (2, 1, 1, 2, 3, 4, 0.0),
    (2, 2, 1, 2, 4, 5, 0.0),
    (2, 1, 1, 3, 4, 6, 0.0),
    (2, 1, 2, 2, 4, 7, 0.0),
    (2, 2, 1, 2, 5, 88, 0.1),  # with missing data
    (2, 1, 1, 2, 6, 9, 0.0),
]


def run_battery():
    results = []
    for (K, F, L, Dmax, T, seed, nan_prob) in CONFIGS:
        model = make_tiny_model(K=K, F=F, L=L, Dmax=Dmax, seed=seed)
        X = make_tiny_data(model, T, seed=seed + 100, nan_prob=nan_prob)
        fwd = model._forward_pass(X)
        bwd = model._discrete_backward(fwd)
        sm = model._switching_smoother(fwd, bwd)
        oracle = run_tiny_exact_oracle(model, X)
        results.append(dict(
            config=dict(K=K, F=F, L=L, Dmax=Dmax, T=T, seed=seed, nan_prob=nan_prob),
            has_nan=bool(np.isnan(X).any()),
            gamma_sum_err=float(np.max(np.abs(bwd.regime_posterior_smoothed.sum(axis=1) - 1.0))),
            ll_gap_abs=float(abs(fwd.log_likelihood - oracle.log_likelihood)),
            filt_gap=float(np.max(np.abs(fwd.regime_post_filtered - oracle.filtered_regime_posterior))),
            smooth_gap=float(np.max(np.abs(bwd.regime_posterior_smoothed - oracle.smoothed_regime_posterior))),
            xi_gap=float(np.max(np.abs(bwd.xi_t - oracle.pairwise_transition_marginal))) if T > 1 else 0.0,
            m_gap=float(np.max(np.abs(sm.smoothed_m - oracle.smoothed_m))),
            P_gap=float(np.max(np.abs(sm.smoothed_P - oracle.smoothed_P))),
        ))
    return results


def main():
    results = run_battery()
    out_path = pathlib.Path(__file__).parent / "reference_battery.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    hdr = f"{'config':<42} {'nan':>3} {'gam_sum':>8} {'ll_gap':>8} {'filt':>7} {'smooth':>7} {'xi':>7} {'m_gap':>7} {'P_gap':>7}"
    print(hdr)
    for r in results:
        c = r["config"]
        cfg = f"K={c['K']} F={c['F']} L={c['L']} D={c['Dmax']} T={c['T']} s={c['seed']}"
        print(f"{cfg:<42} {str(r['has_nan'])[0]:>3} {r['gamma_sum_err']:8.1e} {r['ll_gap_abs']:8.1e} "
              f"{r['filt_gap']:7.4f} {r['smooth_gap']:7.4f} {r['xi_gap']:7.4f} {r['m_gap']:7.4f} {r['P_gap']:7.4f}")

    short = [r for r in results if r["config"]["T"] <= 2]
    longer = [r for r in results if r["config"]["T"] > 2]
    print("\nT<=2 rows (%d): max ll_gap=%.2e  max filt=%.2e  max smooth=%.2e  max xi=%.2e"
          % (len(short), max(r["ll_gap_abs"] for r in short), max(r["filt_gap"] for r in short),
             max(r["smooth_gap"] for r in short), max(r["xi_gap"] for r in short)))
    print("T<=2 rows continuous: max m_gap=%.4f  max P_gap=%.4f"
          % (max(r["m_gap"] for r in short), max(r["P_gap"] for r in short)))
    print("T>=3 rows (%d): max ll_gap=%.4f mean=%.4f | max filt=%.4f | max smooth=%.4f | max m_gap=%.4f | max P_gap=%.4f"
          % (len(longer), max(r["ll_gap_abs"] for r in longer), np.mean([r["ll_gap_abs"] for r in longer]),
             max(r["filt_gap"] for r in longer), max(r["smooth_gap"] for r in longer),
             max(r["m_gap"] for r in longer), max(r["P_gap"] for r in longer)))
    print("all rows: max gamma_sum_err=%.2e" % max(r["gamma_sum_err"] for r in results))


if __name__ == "__main__":
    main()
