# WP8 Item 0: restore, housekeeping, small owner decisions (results)

Labels: RUN = executed in this container and the output is in `wp8_evidence/`; READ = read in code or docs; REPORTED = taken from an earlier package; UNKNOWN = not established.
Base: `27c591c` (`ledger/wp7r`, WP7-R final). Branch: `ledger/wp8`. Seeds used in this item: none new; the 48 K-selection datasets are the WP7-R ones (seeds 7801..7824, easy and hard design), re-run, not re-scored against a new criterion.

## 0a. Start-up and first export
Branch `ledger/wp8` created from `27c591c`. Export and restore scripts are `integration_report/wp8/scripts/export_item.sh` and `restore_check.sh` (copies of the WP7-R scripts with base `27c591c` and WP8 names). The restore result of this export is in `WP8_REPORT.md` section 1 and in `integration_report/RESUME.md`. Public-repo hygiene scan: done before this item; the owner decided to continue (commit trailers `Co-Authored-By` and `Claude-Session` stay; this overrides brief rule 5 for this branch; the public history stays as is). One finding stays open for the owner: a third-party email address appears in 6 upstream commits that were already public (see `WP8_REPORT.md` section 8).

## 0b. B1: BIC default "free" (RUN)
- Change: `GaussianHSMM.n_params_convention = "free"` (class attribute); `"legacy"` stays selectable and gives the previous numbers bit for bit (test). Contract doc 9.8 and `CHANGELOG.md` updated.
- Selections on the 48 WP7-R K-selection datasets, rerun with the new default through the public `fit_hssm` path (`wp8_evidence/item0b_kselect_free_default.jsonl`, comparison `wp8_evidence/item0b_compare.txt`, script `integration_report/wp8/scripts/item0b_compare.py`):
  - selected K identical to the stored WP7-R run: **48 of 48**;
  - log-likelihoods per K bit-identical: **48 of 48** (the count does not touch the EM fit);
  - BIC(legacy) minus BIC(free) equals `K ln(n)` for every K in every dataset: 48 of 48 (largest deviation 1.3e-13);
  - planted-K recovery: easy 20/24 under both defaults, hard 8/24 under both defaults.
- Tests: `tests/test_wp7r_item2_bic_count.py` (6 tests; one renamed because the default flipped: `test_default_count_is_the_free_count_and_legacy_stays_selectable_bit_for_bit`). Mutants: **13 of 13 killed** (`wp8_evidence/mutation_item0b_bic.txt`, script `integration_report/wp7r/scripts/mutation_item2_count.py`, edited to flip back to the legacy default).
- Reading of the result: the switch is more correct and did not change a single choice on these datasets. It can change a choice in principle (penalty smaller by `K ln n`, 4.3 per K step at n = 75); the smallest BIC gap on the WP7-R data was 4.8 (hard design), so the effect on real data is UNKNOWN.

## 0c. C4: D12 comment (RUN)
`tests/test_dynamics_statistics.py` status line now reads "accepted as the official rule; no acceptance date recorded; Re-check trigger: before the first pilot." `ast.dump` identical before and after (`wp8_evidence/item0c_ast_identity.txt`, script `integration_report/wp8/scripts/item0c_ast_check.py`).

## 0d. C7: requirements (RUN)
Moved from `requirements.txt` to `requirements-optional.txt`, with the module that needs each noted in the file: statsmodels, patsy, mypy, mypy_extensions, pathspec, librt, ast_serialize, typing_extensions. Import scan: "OK: every third-party import is pinned" (`item0d_scan_imports.txt`); SBOM check OK (`item0d_check_sbom.txt`).
Collection: 555 tests before and 555 after the requirements edit alone (RUN, measured before the new tests were added). The final tree collects 576: the 555 plus 21 new metadata tests (0e), where one older id was renamed in 0b (`item0d_collection_before_after.txt`).
**`scripts/check_env.py` does not pass in this container (UNVERIFIED, not forced):** 9 of 19 pinned packages differ from the installed ones (it was 12 of 27 before the split) because this container has Python 3.13 and scipy 1.18.1 against the pinned 3.12 and 1.15.3, and PyPI is unreachable (DNS), so the pins cannot be installed here. The brief asked to prove it passes; that proof needs a matching Python 3.12 environment with network access. Evidence: `wp8_evidence/item0d_check_env.txt`.

## 0e. Metadata honesty (READ first, then RUN)
grep of the tree found all three hard-coded claims PRESENT. Fixed to report what happened:
1. `DualTimescaleHSSM.model_capability["is_production_grade_full_hssm"]` was `True` always. Now `is_certified_fit_`: fitted, `fit_state_ == CONVERGED`, no Cholesky fallback, no duration parameter at a wall of the duration box. Unfitted: `False`. (Model-level part of the publish rule only; the full decision stays `HSSMResult.insights_withheld`.)
2. Dual path `optimizer_success=True` always. Now `not duration_fallback_used_` (same expression the baseline path already used).
3. `duration_model_identity` said `"explicit_truncated_gaussian"` in both models; the pmf is a truncated, renormalised discrete log-normal (a test builds both pmfs and shows the Gaussian one differs). Now `"explicit_truncated_lognormal"`.
Older tests/docs that pinned the old values were updated (`test_baseline_identity`, `test_canonical_api`, `test_dual_model_scaffold`, `docs/backbone/HSSM_RESULT_SCHEMA.md` row). No fitted number, publish decision or served output changed (the three fields are metadata).
Tests: `tests/test_wp8_item0_metadata.py` (21), mutants **15 of 15 killed** after one survivor (E03, "unfitted model counts as certified") was killed by adding a test (`wp8_evidence/mutation_item0e.txt`).
Note found while testing: with the default tolerance every dual fit on the 5-segment test series ends `MAX_ITER_REACHED`, so the "True" test case uses a loose tolerance (`tol=1.0`) to reach a genuinely `CONVERGED` fit. This is a test-construction fact, not a change to the convergence rule.

## Bit-identity of raw outputs after Item 0 (RUN)
The WP7-R identity battery (`integration_report/wp7r/scripts/item5_identity_battery.py`, dual model through the public path, seeds 5301..5312, one restart, max_iter 200) was run on a clean `git archive 27c591c` and on this tree: numeric outputs (m_t, m_t_std, p_t, p_t posterior, duration parameters, BIC, fit_state, Dmax) bit-identical in **12 of 12**; status unchanged in 12 of 12 (1 publishable, 11 withheld); NOT EXPLAINED 0 (`wp8_evidence/item0_identity_battery_dual_compare.txt`). Baseline path: the 48 K-selection datasets show bit-identical log-likelihoods (section 0b); only the BIC and `n_params` values change, by design.
Note for later items: in this battery only 1 of 12 dual fits on 12-segment population users is publishable (the others end MAX_ITER_REACHED and are withheld), and one fit takes 60 to 150 s. Both facts limit how many accepted dual fits the WP8 benchmark can contain; they are budgeted for in `tests/PREDECLARED_CRITERIA_WP8.md`.

## Full suite (RUN)
`PYTHONPATH=. python3 -m pytest tests -q`: **571 passed, 5 xfailed** in 1273 s (576 collected = 555 at `27c591c` + 21 new, with one renamed id); WP7-R reference 550 passed, 5 xfailed. `wp8_evidence/full_suite_wp8_item0.txt`.
