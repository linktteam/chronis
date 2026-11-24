"""Item 1, check (F): the shipped path (fit(n_iter=20), duration-box refit ON) versus the 20-iteration checkpoint of the pure-EM trajectory, on the 8 N18 seeds.
usage: PYTHONPATH=. python3 integration_report/wp7r/scripts/item1_faithfulness.py [jsonl] > out"""
import json, sys, warnings
import numpy as np
from scipy.special import logsumexp
warnings.simplefilter("ignore")
from backbone.hssm.dual_model import DualTimescaleHSSM
from tests.utils.heldout_prediction import continued_data, engine_scores
path = sys.argv[1] if len(sys.argv) > 1 else "wp7r_evidence/item1_iter.jsonl"
recs = {d["seed"]: d for d in map(json.loads, open(path)) if d.get("ok") and d["group"] == "N18"}
same = 0; rows = []
for seed in (202, 404, 505, 606, 707, 808, 909, 1010):
    d = continued_data(seed); X_full = d.X_full; T = len(d.X_train)
    m = DualTimescaleHSSM(n_regimes=2, n_features=4, latent_dim=2, max_duration=25, seed=seed)
    m.fit(d.X_train, n_iter=20)
    fwd = m._forward_pass(X_full); lse = logsumexp(np.asarray(fwd.alpha).reshape(len(X_full), -1), axis=1); lp = np.diff(np.concatenate([[0.0], lse]))[T:]
    shipped = float(np.mean(lp)); ck = recs[seed]["checkpoints"]["20"]["heldout"]; refit = m.duration_box_refit_ is not None
    es = float(np.mean(engine_scores("real", seed)["lp_test"])) if seed != 202 else None       # N18 uses 25 iterations for seed 202
    ident = shipped == ck; same += ident
    rows.append((seed, ident, refit))
    print(f"seed {seed}: shipped(fit n_iter=20, box refit on) {shipped:.6f}  checkpoint20 {ck:.6f}  bit-identical {ident}  refit fired {refit}  engine_scores equals shipped: {None if es is None else es == shipped}")
print(f"{same} of 8 bit-identical; differing seeds: {[s for s, i, r in rows if not i]}; of those, refit fired: {[s for s, i, r in rows if not i and r]}")
