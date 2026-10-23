# Pre-declared criteria: work package 4 (housekeeping, dependencies, hold-back gaps, N18 waiver)

Written and committed BEFORE any new WP4 test or script was run (brief section 0, rule 4). Criteria are not edited after results are seen. If one turns
out wrong it is reported as a finding and a NEW dated section is added at the end; the original text stays.
Branch `ledger/wp4`, created from `ledger/wp3` (head `0661388`). Labels: RUN / READ / REPORTED / UNKNOWN.

## 0. What I had already seen before writing this file

* From WP3 (RUN, see `integration_report/wp3/WP3_REPORT.md`): full suite 305 collected, 300 passed, 2 failed (N18 C1, C3), 3 xfailed; N18 numbers C1 mean -0.7087 se 0.4920 (3 of 8 seeds),
  C3 mean 2.2200 se 0.7444 mean-3se -0.0131; sensitivity V1 (24 new seeds) C3 passes (t 5.20) and C1 does not (-0.266, t -1.21).
* READ in this session, before writing this file (no new test, no import scan, no PyPI probe, no fit has been run in WP4):
  * `backbone/hssm/gating.py`: `fit_hssm_gated` calls `fit_hssm(...)` WITHOUT `model_type`, and `fit_hssm`'s default is `"baseline"`, so today it always fits the baseline GaussianHSMM, which has no health flags.
    A gate on `insights_withheld` therefore cannot fire through `fit_hssm_gated` as it stands; it can fire for a dual model only if the dual model is requested.
  * `backbone/attractors/validation.py:35` is the only caller (`model, run_log = fit_hssm_gated(...)`).
  * `fit_dual_with_random_restarts`: usable restarts are CONVERGED / MAX_ITER_REACHED / REJECTED_UPDATE; selection is `ll > best_ll` (strict, so the first maximum wins); the run log has 6 keys per restart.
  * `fit_hssm` dual branch: `estimate_states(X)` is called on the chosen model BEFORE the `HSSMResult` is built, so firings inside that pass are not in `fit_cholesky_fallbacks` (which counts `fit()` only).
  * `HSSMResult`: the aliases `p_t`, `regime_posterior`, `m_t` return the raw arrays; `insights_withheld` / `withheld_reason` are computed properties.
  * `.gitignore` ends with the pattern `*.txt` (no final newline). The repo has no README, no pytest.ini / tox.ini / setup.cfg / pyproject, no `.python-version`, no `scripts/` directory. `tests/__init__.py` exists, `data/` has no `__init__.py`.
  * Installed here: Python 3.13.16, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, scikit-learn 1.9.1, joblib 1.6.0, threadpoolctl 3.7.0, narwhals 2.26.0, pytest 9.1.1. PyPI was unreachable in WP2 and WP3 (403 / DNS); I have not re-probed it in WP4.

## 1. Item 0a, `.gitignore`

* The bare `*.txt` is replaced by patterns for generated files only: `*.log`, `*.out`, `*.tmp`, `scratch/`. Explicit negations `!requirements*.txt` and `!*.cdx.json` are added after them as a guard
  (they do nothing today; they protect against a later broad pattern). The file gets a final newline. No other existing pattern is removed.
* Interpretation of the brief (stated so it can be corrected): the point of the change is that a NEW evidence or requirements text file must not be silently left out of commits, so `wp4_evidence/x.txt` is expected to be **not ignored**; logs stay ignored.
* Criteria (RUN, `git check-ignore -v --no-index`, which is needed because the default skips tracked files):
  * BEFORE the change (negative control showing the check discriminates): `requirements.txt` IS reported as ignored by `*.txt`.
  * AFTER: `requirements.txt`, `requirements-optional.txt`, `wp4_evidence/x.txt`, `sbom/chronis-ml.cdx.json` are NOT matched (no output, exit 1); `wp4_evidence/x.log` IS matched by `*.log` (exit 0; positive control).
  * Adding a new file `wp4_evidence/x.txt` in the worktree shows as untracked (`??`) in `git status --porcelain`, then it is removed again.

## 2. Item 0b and 0c, comment-only edits

* 0b: the two lines "Status: accepted by the owner ... [OWNER: date]. Re-check trigger: [OWNER: ...]" in the D12 notice are replaced by exactly
  `Status: accepted as the official rule. No acceptance date and no re-check trigger are recorded (owner decision); see the ledger open items.` (with the existing `# - ` comment prefix). Nothing else in the notice changes.
* 0c: a comment block above `test_smoother_uses_smoothed_not_filtered_posterior` states: it is the only test that detects B1; no exact-oracle comparison does; two oracle designs failed on the real engine (0.497 maximum
  posterior gap at T=5); this control needs further development. The assertion is not changed.
* Criteria (RUN): for both files `ast.dump(ast.parse(before)) == ast.dump(ast.parse(after))`; the changed lines are all comment lines; no `[OWNER` text remains in `tests/test_dynamics_statistics.py`; the 0c comment contains the four statements (grep).
  Reason: a comment change must not be able to change behaviour; the AST ignores comments, so equality proves it.

## 3. Item 0d, dependencies and environment

* `.python-version` contains `3.12`. Neither a README nor an ENV note exists, so I create a root `README.md` with only a title and a two-line "Environment" section (target Python 3.12; `python -m venv`, `pip install -r requirements.txt`, `python scripts/check_env.py`, `pytest`). This is a deviation from "whichever exists" and is reported as such.
* Import scan (`scripts/scan_imports.py`, RUN): parses every `.py` under `backbone/`, `data/`, `tests/` with `ast`, classifies each top-level import as standard library (`sys.stdlib_module_names`), first-party (`backbone`, `data`, `tests`) or third party, and maps import names to distribution names by a table in the script
  (`sklearn` -> `scikit-learn`, `dateutil` -> `python-dateutil`, ...). Criterion: after the change, every third-party distribution imported by `backbone/`, `data/`, `tests/` is pinned in `requirements.txt`; the script exits non-zero otherwise. Imports inside functions and `try` blocks count.
* Pins added: libraries imported but missing, plus `joblib`, `threadpoolctl`, `narwhals` (scikit-learn 1.9.1 requirements, READ from its metadata). Version rule: PyPI cannot be queried, so the version is the one installed in this session (RUN, `importlib.metadata`), because that is the only version I have run the tests under;
  these pins are NOT tested together with the other pins (numpy 2.2.6, scipy 1.15.3, pandas 2.3.3): that is N20 and stays UNKNOWN unless the fresh-venv run below succeeds. Existing pins are not changed (including `exceptiongroup`, `tomli`, `colorama`, which are only needed on older Python or Windows and are harmless on 3.12).
* `requirements-optional.txt`: the SBOM components that are not imported by `backbone/`, `data/` or `tests/` (mlflow, pip-audit, openai-whisper, pyannote.audio, opensmile, hdbscan, bertopic, bocd, unless the scan shows one is imported), each with a comment naming the module that needs it (from the scan; "not imported anywhere in this repo" if none),
  at the version the SBOM states (REPORTED, not installed or tested here). `requirements.txt` never lists them.
* SBOM reconciliation (script check, RUN): after the change every `requirements.txt` pin is an SBOM component with the same version, every SBOM component is in `requirements.txt` or `requirements-optional.txt` with the same version, JSON stays valid, component order of existing entries is kept, existing licence text is kept.
  New component licences come from installed package metadata (RUN) or are omitted and listed if unavailable. `boundary`: I search the repo for a definition; if none, existing values stay, new components copy the observed convention (runtime library true, dev tool false) and I say the meaning is unknown.
* `pytest.ini` (none exists): exactly `[pytest]`, `testpaths = tests`, `norecursedirs = tests/diagnostics`. Criterion (RUN): plain `pytest --collect-only -q` from the repo root, run so that the current directory is NOT on `sys.path` (to imitate the `pytest` console script), collects exactly the same number of tests as `runpytest.sh tests/` with 0 collection errors; no test from `tests/diagnostics` is collected.
* `scripts/check_env.py` (RUN, three runs): (1) real `requirements.txt` in this environment: prints the Python version and every pinned library with installed version, exits 1 and names the mismatches (expected here: the pins are not what is installed); (2) a temporary requirements file generated from the installed versions: exits 0; (3) the same file with one version changed: exits 1 and names that package.
  A Python version different from `.python-version` is printed as a warning and does not fail the check (reason: the owner's target is 3.12 but a newer interpreter is not an error in itself).
* Fresh venv: first probe PyPI (`curl` through the proxy). If reachable: `uv venv --python 3.12`, install `requirements.txt`, run `scripts/check_env.py`, run plain `pytest -q` from the repo root. If not reachable: NOT RUN, no claim; `scripts/n20_run_under_pins.sh` is rewritten to do exactly these steps (`bash -n` only).

## 4. Item 1, hold-back gaps

Design (changes against the brief are stated in the report):
* a. `InsightsWithheldError` (subclass of `HSSMError` and `RuntimeError`, attributes `reason`, `result`) in `backbone/hssm/model.py`, exported from `backbone.hssm`. `fit_hssm_gated_result(...)` has the arguments of `fit_hssm_gated` plus optional `model_type="baseline"` and `latent_dim=2`
  (reason: the old function always fits the baseline model, so a dual model could not be requested through it) and returns the full `HSSMResult`; it never raises for withheld insights. `fit_hssm_gated` keeps its signature and return value, calls it with the defaults and raises `InsightsWithheldError` when `insights_withheld`.
* b. Restart selection: among usable restarts, a restart with `fit_degraded_` False is always preferred; the best healthy one is chosen by the same `ll > best_ll` rule; if no healthy restart exists, the best degraded one is chosen by the same rule (the result is then degraded and withheld). Run log unchanged.
* c. Late firings: `HSSMResult` records `fallback_count_at_build` (the model's cumulative count after the `estimate_states` pass inside `fit_hssm`) and `states_pass_fallbacks` (firings inside that pass; change against the brief: those firings happen after the fit finished but before the result was built, so a count taken at build time alone would miss them and the published outputs would be unprotected).
  `late_fallbacks` (computed) = current model count minus `fallback_count_at_build`, 0 if not recorded or no model. The rule becomes: withheld iff `fit_degraded` or `fit_state == NUMERICAL_FAILURE` or `states_pass_fallbacks > 0` or `late_fallbacks > 0`. `insights_withheld` stays a computed property.
* d. The aliases `p_t`, `regime_posterior`, `m_t` return `None` when withheld; raw fields unchanged.
* One internal log: build-time withholding keeps the single WARNING of WP3; a result that becomes withheld later logs one WARNING the first time a withheld result is read through `publishable_outputs()` or an alias (private once-flag, not a dataclass field; nothing user-facing).
* e. `docs/backbone/HSSM_RESULT_SCHEMA.md` lists `fit_degraded`, `fit_cholesky_fallbacks`, `fit_exceptions`, `states_pass_fallbacks`, `fallback_count_at_build`, `insights_withheld`, `withheld_reason`, `late_fallbacks`, `publishable_outputs()` and the alias rule.
* Not done, flagged: the choice of K by BIC across candidate Ks also ignores degradation (only restarts within one K are covered).

Tests (`tests/test_hssm_holdback_closure.py`), each with a negative control, and the mutation set (`integration_report/wp4/scripts/mutation_wp4.py`, at least 10 mutants; a mutant is caught if a test that passes unmutated fails on it):
* gated call on a healthy baseline fit equals a verbatim copy of the old `fit_hssm_gated` (parameter arrays `np.array_equal`, run log equal); `fit_hssm_gated_result` returns the same model;
* gated call on a forced degraded dual fit raises `InsightsWithheldError` carrying the reason and the result; the same fit through `fit_hssm_gated_result` returns a withheld result; healthy fit does not raise;
* selection: two restarts where the degraded one has the HIGHER log-likelihood -> the healthy one is chosen; healthy-only -> the maximum is chosen with first-wins ties as before; all degraded -> the highest log-likelihood is chosen and the result is withheld;
* aliases return `None` when withheld and the raw arrays when healthy; raw fields still readable when withheld;
* late firing: a healthy result whose model fires the fallback during a later `estimate_states` becomes withheld (reason names it, accessor `None`, one log record); firing inside the `estimate_states` pass of `fit_hssm` is withheld at build time; an `estimate_states` call without firing changes nothing.

Bit-identity on healthy fits (rule 5, RUN, `np.array_equal(..., equal_nan=True)`): the reference tree is `git archive 0661388`; the changed tree is the WP4 worktree after Item 1. Both run `integration_report/wp4/scripts/ab_identity_wp4.py`:
(A) the 36 oracle configs (reference battery 10, lossless K<=4 13, K=3 concentrated 6, K=3 lossy 7): forward log-likelihood, alpha, filtered regime posterior, smoothed regime posterior, smoothed mean and covariance, plus the dynamics statistics for the 13 K=3 configs;
(B) the 8 N18 seeds (202, 404, 505, 606, 707, 808, 909, 1010), training series only: `fit_dual_with_random_restarts` (3 restarts, 8 iterations, `bypass_init_gate`): all 12 parameter arrays, log-likelihood history, fit state, run log, `estimate_states` outputs of the selected model; `fit_hssm(model_type="dual_timescale")` outputs present in both trees; `fit_hssm_gated` model parameters and run log.
Criterion: 0 differences among all compared items; the count is reported. Reason: no numeric change on healthy fits is approved; exact equality is the only test that can show it.
Timing: all new WP4 tests together under 180 s on an idle machine (same budget as WP3).

## 5. Item 2, the two red N18 tests

* `C1` and `C3` of `test_real_engine_beats_opponent_on_held_out_data` get `pytest.mark.xfail(strict=True, reason=...)` through `pytest.param`; the reason contains the numbers (C1: mean -0.7087, se 0.4920, 3 of 8 seeds; C3: mean 2.2200, se 0.7444, mean-3se -0.0131) and the owner's sentence
  "C1 is a cold-start finding, addressed in WP5; C3 is a near-miss that clears with 24 seeds (see WP3 sensitivity V1)". The assertion, margin, seeds, data and opponents are unchanged (`git diff` shows only the parametrization marks).
* Criterion (RUN): the full suite run once at the end gives 0 failed; xfailed = 5 (the 3 from WP2 and these 2); passed = collected - 5. No new test is added in this item, so there is no mutation set for it; strictness is shown by the `2 xfailed` line (an unexpected pass would be reported as a failure).
* Note for the report: "cold-start" is the owner's reading; the WP3 evidence is consistent with it (level with a single Gaussian at about 60 training steps, ahead in 12 of 12 seeds at about 120) but no cause was tested.

## 6. Stop conditions in force

The repo contradicts the brief for an item (stop that item, report, continue); a change that would make a healthy-fit output differ; a test that would need an existing assertion changed; a wish to tune a threshold; a result that looks too good.
