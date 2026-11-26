"""Mutation check for A6 part 1 (B4): the design, the generator copy, the prior edits, the coverage measure, the per-user flow (`a6_b4_run.py`) and the analysis (`a6_b4_analyze.py`) against
tests/test_wp10a_a6_b4.py. usage: PYTHONPATH=. python3 integration_report/wp10a/scripts/mutation_wp10a_a6_b4.py [out.txt]"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp8" / "scripts"))
import mutation_runner as R

_base = shutil.ignore_patterns(".git", "__pycache__", "wp7r_evidence", "wp8_evidence", "wp9_evidence", "wp10a_evidence", ".pytest_cache")
R.IGNORE = lambda d, names: set(_base(d, names))

RN = "integration_report/wp10a/scripts/a6_b4_run.py"
AN = "integration_report/wp10a/scripts/a6_b4_analyze.py"
MUT = [
    ("B01 eight users per condition", RN, "PER_CONDITION = 16", "PER_CONDITION = 8"),
    ("B02 the seeds start at 10700", RN, "SEEDS = list(range(10701, 10765))", "SEEDS = list(range(10700, 10764))"),
    ("B03 four users at one sd", RN, 'variant="off_centre_1sd" if j < 8 else "off_centre_2sd", prior_shift_sd=1.0 if j < 8 else 2.0)', 'variant="off_centre_1sd" if j < 4 else "off_centre_2sd", prior_shift_sd=1.0 if j < 4 else 2.0)'),
    ("B04 the far prior is three sds off", RN, "prior_shift_sd=1.0 if j < 8 else 2.0)", "prior_shift_sd=1.0 if j < 8 else 3.0)"),
    ("B05 atypical users have spread 2", RN, 'job.update(variant="atypical_3x", spread=3.0)', 'job.update(variant="atypical_3x", spread=2.0)'),
    ("B06 the dropped feature is the first", RN, 'job["drop_feature"] = F - 1', 'job["drop_feature"] = 0'),
    ("B07 Dmax is 12 in the short variant", RN, 'job["dmax"] = 8', 'job["dmax"] = 12'),
    ("B08 Dmax is 25 in the long variant", RN, 'job["dmax"] = 30', 'job["dmax"] = 25'),
    ("B09 the K-too-small users have two regimes", RN, 'job["true_K"] = 3', 'job["true_K"] = 2'),
    ("B10 one population sd is 0.5", RN, "SD_INTERCEPT = 0.6", "SD_INTERCEPT = 0.5"),
    ("B11 the prior is shifted the other way", RN, 'p["d_obs"] = np.asarray(p["d_obs"], dtype=float) + shift_sd * SD_INTERCEPT', 'p["d_obs"] = np.asarray(p["d_obs"], dtype=float) - shift_sd * SD_INTERCEPT'),
    ("B12 the prior keeps the dropped feature only", RN, "keep = [i for i in range(prior.F) if i != f]", "keep = [i for i in range(prior.F) if i == f]"),
    ("B13 the loadings keep the dropped feature", RN, 'for k in ("C_obs", "d_obs", "R_obs"):', 'for k in ("d_obs", "R_obs"):'),
    ("B14 two sd is three sd", RN, "c2=np.mean(np.abs(z) <= 2.0, axis=0).tolist()", "c2=np.mean(np.abs(z) <= 3.0, axis=0).tolist()"),
    ("B15 one sd is one and a half", RN, "c1=np.mean(np.abs(z) <= 1.0, axis=0).tolist()", "c1=np.mean(np.abs(z) <= 1.5, axis=0).tolist()"),
    ("B16 no affine map", RN, "    ref = per_regime_affine_prediction(true_m, m_hat, true_p, min_points=3 * (m_hat.shape[1] + 1))", "    ref = true_m"),
    ("B17 the rows are not selected", RN, "    z = ((m_hat - ref) / std)[rows]", "    z = ((m_hat - ref) / std)"),
    ("B18 the prior weight is 1", RN, "shared_prior=prior, shared_prior_n0=N0, certify_prior_anchored=True)", "shared_prior=prior, shared_prior_n0=1.0, certify_prior_anchored=True)"),
    ("B19 the certificate is off", RN, "shared_prior=prior, shared_prior_n0=N0, certify_prior_anchored=True)", "shared_prior=prior, shared_prior_n0=N0, certify_prior_anchored=False)"),
    ("B20 K = 3 is fitted", RN, "candidate_ks=(K,), random_seed=seed", "candidate_ks=(3,), random_seed=seed"),
    ("B21 the fit seed is shifted", RN, "candidate_ks=(K,), random_seed=seed,", "candidate_ks=(K,), random_seed=seed + 1,"),
    ("B22 the held-out rows are the training rows", RN, 'rec["heldout"] = coverage_on(m_hat, std, true_m, true_p, slice(t_tr, None))', 'rec["heldout"] = coverage_on(m_hat, std, true_m, true_p, slice(0, t_tr))'),
    ("B23 the matrix keeps the dropped feature", RN, "X_tr, X_te = X_tr[:, keep], X_te[:, keep]", "pass"),
    ("B24 the prior is not shifted", RN, '            if job["prior_shift_sd"]:', "            if False:"),
    ("B25 the true K is not used for the users", RN, 'spec = PopulationSpec(n_regimes=job["true_K"])', "spec = PopulationSpec()"),
    ("B26 the spread is not used for the users", RN, 'spread_factor=job["spread"])', "spread_factor=1.0)"),
    ("B27 the withheld flag is inverted", RN, "rec.update(withheld=bool(res.insights_withheld),", "rec.update(withheld=not bool(res.insights_withheld),"),
    ("B28 the certified flag is always False", RN, 'certified=bool(getattr(res, "certified_prior_anchored", False)),', "certified=False,"),
    ("B29 four training segments", RN, "N_TRAIN_SEG, N_TEST_SEG = 6, 2", "N_TRAIN_SEG, N_TEST_SEG = 4, 2"),
    ("B30 finished users are run again", RN, "    todo = [s for s in seeds if s not in done]", "    todo = list(seeds)"),
    ("B31 the limit skips the first users", RN, "        todo = todo[:a.limit]", "        todo = todo[a.limit:]"),
    ("B32 the draws of the generator copy differ", RN, 'dur = int(np.clip(rng.lognormal(mean=np.log(truth["mean_dwell"]), sigma=0.3), 8, 30))', 'dur = int(np.clip(rng.lognormal(mean=np.log(truth["mean_dwell"]), sigma=0.31), 8, 30))'),
    ("B33 the first latent state has another scale", RN, "m_curr = rng.normal(scale=0.2, size=Ll)", "m_curr = rng.normal(scale=0.3, size=Ll)"),
    ("B34 the test states are not those of the observations", RN, "        m_te[t] = m_curr\n    X_te", "        m_te[t] = m_curr * 0.0\n    X_te"),
    ("A01 the one-sd level is 1.5 sd", AN, "NOMINAL1, NOMINAL2 = math.erf(1.0 / math.sqrt(2.0)), math.erf(2.0 / math.sqrt(2.0))", "NOMINAL1, NOMINAL2 = math.erf(1.5 / math.sqrt(2.0)), math.erf(2.0 / math.sqrt(2.0))"),
    ("A02 the band is 0.15", AN, "BAND, WORSE, MIN_READ, DRAWS = 0.10, 0.10, 6, 2000", "BAND, WORSE, MIN_READ, DRAWS = 0.15, 0.10, 6, 2000"),
    ("A03 worse means 0.15 lower", AN, "BAND, WORSE, MIN_READ, DRAWS = 0.10, 0.10, 6, 2000", "BAND, WORSE, MIN_READ, DRAWS = 0.10, 0.15, 6, 2000"),
    ("A04 five users are enough to read", AN, "BAND, WORSE, MIN_READ, DRAWS = 0.10, 0.10, 6, 2000", "BAND, WORSE, MIN_READ, DRAWS = 0.10, 0.10, 5, 2000"),
    ("A05 1000 draws", AN, "BAND, WORSE, MIN_READ, DRAWS = 0.10, 0.10, 6, 2000", "BAND, WORSE, MIN_READ, DRAWS = 0.10, 0.10, 6, 1000"),
    ("A06 a failed user counts as published", AN, 'return bool(r.get("ok")) and not r["withheld"]', 'return not r["withheld"]'),
    ("A07 the first record of a seed counts", AN, '            ok[r["seed"]] = r', '            ok.setdefault(r["seed"], r)'),
    ("A08 the two-sd columns come first", AN, 'c = np.array([list(r[key]["c1"]) + list(r[key]["c2"]) for r in recs], dtype=float).reshape(len(recs), -1)', 'c = np.array([list(r[key]["c2"]) + list(r[key]["c1"]) for r in recs], dtype=float).reshape(len(recs), -1)'),
    ("A09 the band needs one dimension only", AN, "return bool(abs(c[0] - NOMINAL1) <= BAND and abs(c[1] - NOMINAL1) <= BAND and abs(c[2] - NOMINAL2) <= BAND and abs(c[3] - NOMINAL2) <= BAND)", "return bool(abs(c[0] - NOMINAL1) <= BAND and abs(c[2] - NOMINAL2) <= BAND)"),
    ("A10 worse needs both dimensions", AN, "return bool((c0[2] - c[2]) > WORSE or (c0[3] - c[3]) > WORSE)", "return bool((c0[2] - c[2]) > WORSE and (c0[3] - c[3]) > WORSE)"),
    ("A11 worse looks at one sd", AN, "return bool((c0[2] - c[2]) > WORSE or (c0[3] - c[3]) > WORSE)", "return bool((c0[0] - c[0]) > WORSE or (c0[1] - c[1]) > WORSE)"),
    ("A12 the signed distance is reversed", AN, "return [float(c[0] - NOMINAL1), float(c[1] - NOMINAL1), float(c[2] - NOMINAL2), float(c[3] - NOMINAL2)]", "return [float(NOMINAL1 - c[0]), float(NOMINAL1 - c[1]), float(NOMINAL2 - c[2]), float(NOMINAL2 - c[3])]"),
    ("A13 a condition is readable with at least seven users", AN, "readable=len(use) >= MIN_READ,", "readable=len(use) > MIN_READ,"),
    ("A14 the headline uses all fits", AN, "use = [r for r in rs if (is_published(r) if published_only else True)]", "use = [r for r in rs if True]"),
    ("A15 the bootstrap seed is 1", AN, '            lo, hi = interval(boot_mean(c, np.random.default_rng(0)))', '            lo, hi = interval(boot_mean(c, np.random.default_rng(1)))'),
    ("A16 the difference is C0 minus the condition", AN, 'mean=(st["matrix"].mean(axis=0) - st0["matrix"].mean(axis=0)).tolist()', 'mean=(st0["matrix"].mean(axis=0) - st["matrix"].mean(axis=0)).tolist()'),
    ("A17 the bootstrap difference is a sum", AN, 'd = boot_mean(st["matrix"], rng) - boot_mean(st0["matrix"], rng)', 'd = boot_mean(st["matrix"], rng) + boot_mean(st0["matrix"], rng)'),
    ("A18 the percentile interval is 5 to 95", AN, "return np.percentile(b, 2.5, axis=0), np.percentile(b, 97.5, axis=0)", "return np.percentile(b, 5.0, axis=0), np.percentile(b, 95.0, axis=0)"),
    ("A19 the headline block is the all-fits block", AN, 'pub = block(recs, True, "HEADLINE, published fits only")', 'pub = block(recs, False, "HEADLINE, published fits only")'),
    ("A20 the withhold reason is the text after the colon", AN, 'k = (r.get("withheld_reason") or "withheld").split(":")[0]', 'k = (r.get("withheld_reason") or "withheld").split(":")[-1]'),
    ("A21 the comparison ignores the readability of C0", AN, "        if not (s[\"readable\"] and c0[\"readable\"]):", "        if not s[\"readable\"]:"),
    ("A22 the worse verdict is inverted", AN, "{'WORSE than C0 (by more than ' + format(WORSE, '.2f') + ')' if worse_than(s['mean'], c0['mean'])", "{'WORSE than C0 (by more than ' + format(WORSE, '.2f') + ')' if not worse_than(s['mean'], c0['mean'])"),
]
if __name__ == "__main__":
    sys.exit(R.run(MUT, ["tests/test_wp10a_a6_b4.py", "-q", "-x"], sys.argv[1] if len(sys.argv) > 1 else None, workers=1))
