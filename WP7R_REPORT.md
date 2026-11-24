# WP7-R report (restart): Items 0 to 6 on `ledger/wp7r`

Labels: RUN (executed here), READ (read in code or documents), REPORTED (from the owner's files, not re-run), UNKNOWN. Every item has a pre-declaration in `tests/PREDECLARED_CRITERIA_WP7R.md` written before its scored runs, and a results document `tests/WP7R_ITEM*_RESULTS.md`. Nothing was pushed to `main`, merged or force-pushed.

## 1. Backup and restore
* Repo `linktteam/chronis` (public, owner-approved), remote `backup`. Branch **`ledger/wp7r`**, the last pushed hash is the tip of `ledger/wp7r` at the time of export (printed in the chat message and in `wp7r_item6/HEAD.txt`; `git ls-remote backup refs/heads/ledger/wp7r` shows it). Engine and test code last changed at `0e32c09`; everything after it is documents and evidence. `ledger/wp6` stays at `c95cb5d`.
* Exports (each: patch from `c95cb5d`, tree tarball, bundle of `ledger/wp6` + `ledger/wp7r`, evidence zip, `SHA256SUMS.txt`; each restore test = SHA256 check, bundle verify, patch applies to a clean `git archive c95cb5d`, base + patch equals the tarball tree, the contract smoke script exits 0): `wp7r_item0` (head bc6eccb), `wp7r_item1` (db5f788), `wp7r_item2` (9ac0e6e), all RESTORE TEST PASSED when exported. **Items 3 to 6 are interleaved on one branch (Item 3 was re-read after Item 5, Item 4 ran on the Item 5 engine), so they are exported once as `wp7r_item6`; its restore-test result is in `wp7r_item6/RESTORE_TEST.txt` and in the chat message (a report cannot contain the result of a test of the archive that contains it).** `wp7r_item0/1/2` were re-checked after the work of Items 3 to 6 with `scripts/restore_check.sh`: all PASS.

## 2. Item 1: does more EM make the dual model predict worse? (RUN, 28 seeds; 9 for Dmax 60)
Verdict word (pre-declared): **no consistent trend**. Pooled slope of the held-out score on log2(iterations) -0.026 nats/step per doubling (se 0.021, t = -1.23), negative in 15 of 28 seeds. N18 design (8 seeds) declines (-0.112 per doubling, 6 of 8 negative); D-pop (20 seeds) is flat (+0.008). Item 5c (the dual publish gate) was therefore **not built**.

| held-out score, mean (n) | 20 it | 40 it | 80 it | 160 it | 20 to 160: paired difference (se) |
|---|---|---|---|---|---|
| N18 (8) | -5.740 | -5.959 | -6.009 | -6.095 | -0.356 (0.205), 2 of 8 better |
| D-pop (20) | -3.971 | -3.976 | -3.962 | -3.950 | +0.021 (0.025), 11 of 20 better |
| pooled (28) | -4.477 | -4.542 | -4.546 | -4.563 | -0.086 (0.067), 13 of 28 better |

Training log-likelihood rose in 28 of 28 seeds at every step; no step is beyond 2 paired se. Salvage-note facts: EM creeps (rate 0.9923 per iteration; median 863 iterations to |dll| < 1e-4, note said 0.995 and 1,330) **RE-VERIFIED with different numbers**; "segments longer than Dmax go with degenerate durations" **RE-VERIFIED** (Fisher p = 0.0009, odds ratio 29.8); "Dmax = 60 cured 8 of 9" **RE-VERIFIED** by the declared box rule (caveats: not a general cure, does not make a fit publishable); "99% of the remaining gain is in the observation block" **NOT RE-VERIFIED** (not attempted).

## 3. Item 2: K selection and the BIC count (baseline model; RUN, 24 seeds x 2 designs)
* **BIC selection rate by planted K** (served K = planted K): easy design 20/24 overall (8/8, 7/8, 5/8 for K = 2, 3, 4; 20 of 20 when every planted regime occurs in the series, a generator flaw found post hoc); hard design 8/24 (8/8, 0/8, 0/8). BIC never over-selected; it under-selects when regimes overlap and the wrong-K fit is **served without a flag**.
* **Collapsed candidates:** served collapsed fits 0; a collapsed candidate chosen over a healthy one 0; the demotion rule was **never triggered** (smallest minimum 4.04 effective days), so this study does not exercise it.
* **P1-01:** the BIC parameter count is wrong by K (code counts `K(K-1)` transitions, `K(K-2)` are free); it moves the selected K in **0 of 48** datasets; an opt-in `GaussianHSMM.n_params_convention` (`"legacy"` default, `"free"`) was added, 13 mutants killed, defaults bit-identical.
* **4-day sensitivity:** 3, 4 and 6 days give identical selections and withholding (0 of 48 differ); the rule is not exercised on well-populated regimes, so the data cannot rank the thresholds. Rule unchanged.

## 4. Item 3: shrinkage (cold start) under fail-closed (RUN, 24 users x 3 lengths)
**The WP5 cold-start gain does not survive on publishable fits, and under the WP5 design nothing is publishable.** Headline = the FINAL engine (the study was re-statused after Item 5; all 40 refits reproduced exactly, 18 are newly withheld).

| `shrink5`, 200 iterations | length 30 | length 60 | length 120 |
|---|---|---|---|
| withheld (original engine -> final engine) | 42% -> 67% | 46% -> 62% | 54% -> 79% |
| publishable fits | 8 | 9 | 5 |
| held-out gain over the single Gaussian on publishable fits (mean, se) | +1.111 (0.239) | +0.626 (0.302) | +1.284 (0.122) |
| `mean - 3 se` | +0.393 | -0.280 | +0.918 (not judged, 5 < 8) |
| risk-coverage `mean - 3 se` (coverage) | -0.029 (33%) | -0.143 (38%) | -0.066 (21%) |

* Stage A (the WP5 design, 20 iterations): **100% withheld** for `shrink5` and `personal`. `personal` is withheld in 100% of the stage B fits at every length (0 of 24 publishable). Held-out score on publishable fits: `shrink5` beats the shared-only model on every publishable seed (+3.85, +5.35, +6.43).
* Words: **"does not survive"**; net deployment gain **"not shown"** at every length. Post hoc: the withheld fits predict better than the publishable ones. The cold-start feature is **not reported as working**.

## 5. Item 4: N17, `m_t_std` for shrinkage-fitted fits (RUN; nothing integrated, xfail stays)
Harness control (true parameters) in band (`c2` 0.944, `c1` 0.700; bands 0.9145 to 0.9945 and 0.6427 to 0.7227).
* **Coverage of publishable shrinkage-fitted fits (12 seeds, 200 iterations): 11 publishable, `c2` 0.934, `c1` 0.681: "reaches the band"** (a favourable case: the prior is centred on the generator; one publishable fit, seed 9102, has `c2` 0.685; the coverage is the same at 20 iterations, so it comes from the prior). Personal fits: `c2` 0.733 (20 it) and 0.589 (200 it), none publishable.
* **Bootstrap (6 seeds, B = 30):** `c2` 0.730 -> 0.844 (+0.114, paired se 0.006, 5 of 5 seeds that ran improved; seed 9106 returned nothing because a regime had 8 steps, below the 9-step minimum of the frame removal): **"improves but does not reach"**.
* **Cost (measured):** P = 50: Hessian 128 s + Jacobian 13 s; bootstrap B = 30 = 87 s per fit. P = 204: Hessian about 3,550 s (EXTRAPOLATED), bootstrap B = 30 about 173 s (extrapolated).
* **Recommendation (the pre-declared rule): "shrinkage-only uncertainty", with the favourable-case caveat** (fresh seeds and an off-centre prior are needed before any integration; numeric confidence stays hidden).

## 6. Item 5: dual honesty rules (RUN)
* **Built (dual model only; numeric outputs unchanged):** R1 `collapsed_regime` (filtered-posterior effective days, strictly below 4.0); R2 `degenerate_duration` (duration at a wall of the box, or a pmf still rising at the last lattice step); R3 opt-in `fit_hssm(max_duration_retry=...)`. Why: a dual fit with an empty regime or a truncated duration law was served as healthy. Contract doc 9.2, 9.3, 9.9 updated; the Item 0 doc test that asserted the old "NOT withheld" behaviour was deliberately changed.
* **Bit-identity against `c95cb5d`:** plain dual path 12 of 12 (1 previously healthy, unchanged; 11 already withheld; 0 unexplained); shared-prior path 12 of 12 (6 previously healthy, unchanged; 1 previously publishable newly withheld by R2; 5 already withheld; 0 unexplained). R2 flags 4 of 12 fits in each battery.
* **Tests:** 21; **mutants 31 of 31 killed**; Item 0 doc mutants 28 of 28 killed (one, D14, had silently stopped applying and was repaired).
* **Share of dual fits passing the gate:** not applicable, the Item 5c gate was not built.

## 6a / 6b. The 14 older tests and n0
* **6a (READ + RUN):** none of the 14 tests WP6 adjusted lost or weakened an assertion on a healthy fit (`tests/WP7R_ITEM6A_AUDIT.md`: one removed assertion, replaced by a strict superset). They pass in the second full run below.
* **6b (RUN):** on 6001..6012 neither n0 = 1 (score -0.391) nor n0 = 2 (+0.092) has a strictly larger score than n0 = 5 (+0.106) under the WP5 rule, so the evaluation seeds 7501..7524 were **not used** and the default `n0` stays 5.0. The WP5 n0 = 5 numbers reproduce to 0.002 but not bit for bit (not investigated).

## 7. Criteria and designs changed, what was looked at, what was not verified
* **Dated additions (no criterion edited after its results):** 2026-11-24 Item 5.5 (a second identity battery, added because the first had one healthy fit); 2026-11-24 Item 3.9 (re-status of the Item 3 publishable fits, added when I found that Item 3 had run before R1/R2 existed). The Item 0 doc assertions about dual collapse changed with Item 5 (deliberate). Item 4's bootstrap failure on seed 9106 is a pre-declared minimum, not a change.
* **Looked at before declaring:** recorded in each section header of `tests/PREDECLARED_CRITERIA_WP7R.md`. Development-only seeds touched: 7601 to 7603, 7610 to 7639; evaluation seeds 7501..7524 and 7701..7724 were never used.
* **Not verified:** the shrinkage coverage on fresh seeds or an off-centre prior (Item 4); whether R2 is too strict on short series; the observation-block share of the EM creep (Item 1); the collapse-demotion path under real collapsed candidates (Item 2); `personal` with 10 restarts (Item 3 used one); the cause of the 0.002 n0 = 5 reproduction gap; the real-data behaviour of anything. Item 4's cost at P = 204 is extrapolated. One generator family and K = 2 (baseline: F = 7) throughout.
* **Open owner decisions (non-blocking):** (a) whether the default BIC count should switch from `"legacy"` to `"free"` (moves 0 of 48 selections here); (b) whether prior-anchored fits (which predict best but stop after 2 or 3 EM iterations) need a different certification, and whether R2 should be relaxed for them; (c) Item 4's recommendation (shrinkage-only uncertainty) needs the fresh-seed evaluation before any integration.
* **Full suite (RUN, `wp7r_evidence/full_suite_wp7r.txt`): 550 passed, 5 xfailed in 1,345 s** at `0e32c09` (WP6 reference: 510 passed, 5 xfailed in 1,610 s contended; the 5 xfails are unchanged, the N17 cells among them). The FIRST full run (22 min, at `fd87c9c`) had **2 failures caused by Item 5**, fixed and mutation-checked, no assertion weakened: `test_118_dual_model_handoff_to_attractor_detector` (R2 withheld its 15-iteration fit; the WP6 helper `tests/utils/iteration_limit_as_converged.py` now also clears the R1/R2 flags inside its context) and `test_wp5_integration.py::test_new_arguments_are_last_optional_and_default_to_none` (`fit_hssm` has a new last argument `max_duration_retry`; the test now expects the tail `shared_prior`, `shared_prior_n0`, `max_duration_retry`). These two are older tests adjusted by this package, in the sense of the 6a audit.

## 8. Artifacts
`/mnt/user-data/outputs/wp7r_item6/` (patch `wp7r_item6.patch`, `wp7r_tree.tar.gz`, `wp7r.bundle`, `wp7r_evidence_item6.zip`, `SHA256SUMS.txt`); earlier: `wp7r_item0/`, `wp7r_item1/`, `wp7r_item2/`. The SHA256 list of each export is in its `SHA256SUMS.txt` and is repeated in the chat message.
