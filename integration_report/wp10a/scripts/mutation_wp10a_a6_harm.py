"""Mutation check for A6 part 2 (the harmful-prior control): the study settings, the candidate priors, the eligibility and the driver (`a6_harm_run.py`) and the validity rule, the choice and the
evaluation levels (`a6_harm_analyze.py`) against tests/test_wp10a_a6_harm.py. usage: PYTHONPATH=. python3 integration_report/wp10a/scripts/mutation_wp10a_a6_harm.py [out.txt]"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wp8" / "scripts"))
import mutation_runner as R

_base = shutil.ignore_patterns(".git", "__pycache__", "wp7r_evidence", "wp8_evidence", "wp9_evidence", "wp10a_evidence", ".pytest_cache")
R.IGNORE = lambda d, names: set(_base(d, names))

RN = "integration_report/wp10a/scripts/a6_harm_run.py"
AN = "integration_report/wp10a/scripts/a6_harm_analyze.py"
MUT = [
    ("H01 H1 is 3", RN, 'CANDIDATES = {"H1": 4.0, "H2": 8.0, "H3": 16.0}', 'CANDIDATES = {"H1": 3.0, "H2": 8.0, "H3": 16.0}'),
    ("H02 H2 is 6", RN, 'CANDIDATES = {"H1": 4.0, "H2": 8.0, "H3": 16.0}', 'CANDIDATES = {"H1": 4.0, "H2": 6.0, "H3": 16.0}'),
    ("H03 H3 is 12", RN, 'CANDIDATES = {"H1": 4.0, "H2": 8.0, "H3": 16.0}', 'CANDIDATES = {"H1": 4.0, "H2": 8.0, "H3": 12.0}'),
    ("H04 the development seeds start at 9632", RN, '"dev": list(range(9631, 9643))', '"dev": list(range(9632, 9644))'),
    ("H05 the evaluation seeds start at 10772", RN, '"eval": list(range(10771, 10795))', '"eval": list(range(10772, 10796))'),
    ("H06 100 iterations", RN, "MAX_ITER = 200", "MAX_ITER = 100"),
    ("H07 n0 is 3", RN, "K, F, L, DMAX, N0 = 2, 4, 2, 25, 5.0", "K, F, L, DMAX, N0 = 2, 4, 2, 25, 3.0"),
    ("H08 two restarts", RN, "n_initializations=1,", "n_initializations=2,"),
    ("H09 no fast-test switch", RN, "allow_fast_test_fit=True,", "allow_fast_test_fit=False,"),
    ("H10 the fit seed is not shifted", RN, "random_seed=seed - 2000", "random_seed=seed"),
    ("H11 K = 3 is fitted", RN, "candidate_ks=(K,), n_initializations=1", "candidate_ks=(3,), n_initializations=1"),
    ("H12 the certificate is off", RN, "shared_prior_n0=N0, certify_prior_anchored=True)", "shared_prior_n0=N0, certify_prior_anchored=False)"),
    ("H13 eligibility ignores the other reasons", RN, "eligible=bool(em_state in ELIGIBLE_EM_STATES and not other)", "eligible=bool(em_state in ELIGIBLE_EM_STATES)"),
    ("H14 MAX_ITER_REACHED is not eligible", RN, 'ELIGIBLE_EM_STATES = ("REJECTED_UPDATE", "MAX_ITER_REACHED")', 'ELIGIBLE_EM_STATES = ("REJECTED_UPDATE",)'),
    ("H15 the EM verdict is read with the certificate on", RN, "        res.certification = {}\n        em_reason = res.withheld_reason", "        em_reason = res.withheld_reason"),
    ("H16 published_after is inverted", RN, "published_after=res.withheld_reason is None,", "published_after=res.withheld_reason is not None,"),
    ("H17 the prior is shifted the other way", RN, 'd["d_obs"] = (np.asarray(d["d_obs"], dtype=float) + float(shift)).tolist()', 'd["d_obs"] = (np.asarray(d["d_obs"], dtype=float) - float(shift)).tolist()'),
    ("H18 the evaluation has no true-prior arm", RN, 'return ["H", "T"]', 'return ["H"]'),
    ("H19 an arm without its Gaussian is complete", RN, 'need = {"gauss"} | set(arms)', "need = set(arms)"),
    ("H20 the evaluation runs without the freeze file", RN, "        if not FREEZE.exists():", "        if False:"),
    ("H21 an unknown candidate is accepted", RN, "        if cand not in CANDIDATES:", "        if False:"),
    ("H22 the Gaussian is not scored", RN, 'g = dict(arm="gauss", **N.gauss_scores(user)); g.update(base); out.append(g)', "pass"),
    ("H23 the true prior arm gets the harmful prior", RN, 'priors[a] = true_prior(prior_path) if a == "T" else harmful_prior(prior_path, shifts[a])', 'priors[a] = harmful_prior(prior_path, shifts.get(a, 0.0))'),
    ("G01 the candidate order is reversed", AN, 'ORDER = ("H1", "H2", "H3")', 'ORDER = ("H3", "H2", "H1")'),
    ("G02 rule (a) needs -0.2", AN, "A_MAX, B_MIN, C_MIN, N_DEV = -0.25, 6, 4, 36", "A_MAX, B_MIN, C_MIN, N_DEV = -0.20, 6, 4, 36"),
    ("G03 rule (b) needs 5", AN, "A_MAX, B_MIN, C_MIN, N_DEV = -0.25, 6, 4, 36", "A_MAX, B_MIN, C_MIN, N_DEV = -0.25, 5, 4, 36"),
    ("G04 rule (c) needs 3", AN, "A_MAX, B_MIN, C_MIN, N_DEV = -0.25, 6, 4, 36", "A_MAX, B_MIN, C_MIN, N_DEV = -0.25, 6, 3, 36"),
    ("G05 the evaluation needs 11 fits", AN, "REQUIRED_EVAL = 12", "REQUIRED_EVAL = 11"),
    ("G06 the bound is undefined when all are certified", AN, "    return 1.0 if k >= n else float(beta.ppf(1.0 - alpha, k + 1, n - k))", "    return 1.0 if k > n else float(beta.ppf(1.0 - alpha, k + 1, n - k))"),
    ("G07 the bound is the lower tail", AN, "float(beta.ppf(1.0 - alpha, k + 1, n - k))", "float(beta.ppf(alpha, k + 1, n - k))"),
    ("G08 a score not flagged finite is used", AN, 'if m is None or not rec.get("finite", False) or not math.isfinite(m):', "if m is None or not math.isfinite(m):"),
    ("G09 a tie is worse", AN, "return (not math.isfinite(s)) or s < gauss", "return (not math.isfinite(s)) or s <= gauss"),
    ("G10 a non-finite score is not worse", AN, "return (not math.isfinite(s)) or s < gauss", "return math.isfinite(s) and s < gauss"),
    ("G11 the difference is personal minus fit", AN, "diffs.append(s - pers)", "diffs.append(pers - s)"),
    ("G12 a pair is used when either score is finite", AN, "if math.isfinite(s) and math.isfinite(pers):", "if math.isfinite(s) or math.isfinite(pers):"),
    ("G13 every worse fit counts as eligible", AN, 'worse_elig += int(bool(r.get("eligible")))', "worse_elig += 1"),
    ("G14 rule (a) is inverted", AN, "ok_a, ok_b, ok_c = (math.isfinite(a) and a <= A_MAX), worse >= B_MIN, worse_elig >= C_MIN", "ok_a, ok_b, ok_c = (math.isfinite(a) and a >= A_MAX), worse >= B_MIN, worse_elig >= C_MIN"),
    ("G15 rule (b) is strict", AN, "ok_a, ok_b, ok_c = (math.isfinite(a) and a <= A_MAX), worse >= B_MIN, worse_elig >= C_MIN", "ok_a, ok_b, ok_c = (math.isfinite(a) and a <= A_MAX), worse > B_MIN, worse_elig >= C_MIN"),
    ("G16 rule (c) is strict", AN, "ok_a, ok_b, ok_c = (math.isfinite(a) and a <= A_MAX), worse >= B_MIN, worse_elig >= C_MIN", "ok_a, ok_b, ok_c = (math.isfinite(a) and a <= A_MAX), worse >= B_MIN, worse_elig > C_MIN"),
    ("G17 one rule is enough", AN, "verified=bool(ok_a and ok_b and ok_c)", "verified=bool(ok_a or ok_b or ok_c)"),
    ("G18 the last verified candidate is chosen", AN, "    for c in ORDER:\n        if c in results and results[c][\"verified\"]:", "    for c in reversed(ORDER):\n        if c in results and results[c][\"verified\"]:"),
    ("G19 every worse fit counts as eligible in E1", AN, 'ew = [r for r in worse if r.get("eligible")]', "ew = worse"),
    ("G20 a certified fit is one that is not certified", AN, 'k = sum(1 for r in ew if (r.get("certification") or {}).get("state") == CERTIFIED)', 'k = sum(1 for r in ew if (r.get("certification") or {}).get("state") != CERTIFIED)'),
    ("G21 the control is valid with more than 12 fits", AN, "valid = n >= REQUIRED_EVAL", "valid = n > REQUIRED_EVAL"),
    ("G22 E1 is met with one certified fit", AN, '("MET" if k == 0 else "NOT MET")', '("MET" if k <= 1 else "NOT MET")'),
    ("G23 E0 needs two certified fits", AN, "e0 = t_cert >= 1", "e0 = t_cert >= 2"),
    ("G24 E3 counts the withheld fits", AN, 'pub_worse = sum(1 for r in worse if r.get("published_after"))', 'pub_worse = sum(1 for r in worse if not r.get("published_after"))'),
    ("G25 the Gaussian need not be usable", AN, "usable = [r for r in H if math.isfinite(g.get((r[\"seed\"], r[\"length\"]), float(\"nan\")))]", "usable = [r for r in H]"),
    ("G26 converged fits are those that did not converge", AN, 'converged = sum(1 for r in H if r.get("fit_state") == "CONVERGED")', 'converged = sum(1 for r in H if r.get("fit_state") != "CONVERGED")'),
    ("G27 the other-rule count includes converged fits", AN, 'other = sum(1 for r in H if not r.get("eligible") and r.get("fit_state") not in ("CONVERGED",))', 'other = sum(1 for r in H if not r.get("eligible"))'),
    ("G28 the baseline reads the shrink5 arm as personal", AN, 'if r["arm"] in ("personal", "gauss"):', 'if r["arm"] in ("personal", "gauss", "shrink5"):'),
]
if __name__ == "__main__":
    sys.exit(R.run(MUT, ["tests/test_wp10a_a6_harm.py", "-q", "-x"], sys.argv[1] if len(sys.argv) > 1 else None, workers=1))
