import numpy as np, warnings; warnings.filterwarnings("ignore")
from backbone.hssm.fitting import fit_hssm
from backbone.hssm.model import GaussianHSMM
rng = np.random.default_rng(1)
K, F, T = 3, 7, 75
means = rng.normal(0, 1.0, (K, F)); means *= 2.5 / np.abs(means).max()
path, cur = [], 0
while len(path) < T:
    path += [cur] * int(rng.integers(9, 23)); cur = (cur + int(rng.integers(1, K))) % K
path = np.array(path[:T]); X = means[path] + rng.normal(0, 0.5, (T, F))
X = (X - X.mean(0)) / X.std(0); days = np.arange(T, dtype=float)
states = []
for s in range(3007, 3017):
    m = GaussianHSMM(n_regimes=K, n_features=F, max_duration=45, seed=s); m.fit(X, n_iter=60, verbose=False, timestamps=days)
    states.append(str(m.fit_state_).split(".")[-1][:4])
print(states)
r = fit_hssm(matrix=X, candidate_ks=(3,), n_initializations=10, max_iter=60, random_seed=7, timestamps=days, user_id="u", dataset_hash="h")
print(r.fit_state, r.insights_withheld)
try:
    fit_hssm(matrix=X, candidate_ks=(3,), n_initializations=2, max_iter=10, random_seed=7, timestamps=days * 86400.0, user_id="u", dataset_hash="h")
    print("seconds-as-days: NO ERROR")
except ValueError as e:
    print("seconds-as-days ValueError:", str(e)[:120])
