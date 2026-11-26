# Changelog

Entries are for behaviour that a consumer of `backbone/hssm/` can observe. Newest first. Module paths and field names under `backbone/hssm/` are stable; new fields are additive only.

## WP9 (branch `ledger/wp9`, from `ledger/wp8` 7887118)

Everything below is additive. Raw outputs of fits that do not use an opt-in option are bit-identical to WP8 (counts in the WP9 report). Design and numbers: `tests/PREDECLARED_CRITERIA_WP9.md`, the per-item results documents, `docs/INTEGRATION_CONTRACT_HSSM.md` section 9.11.

- **A1 FIT_UNSTABLE.** New module `backbone/hssm/fit_stability.py`; six new keys in `HSSMResult.restart_summary` (`consensus_m`, `consensus_tau`, `n_healthy_restarts`, `consensus_min_ari`, `fit_unstable`, `fit_unstable_reason`); `FitSummary.fit_unstable` / `fit_unstable_reason`, `LineageRecord.fit_unstable` / `anchor_fit_unstable`; lineage reasons `fit_unstable` and `anchor_fit_unstable` (never ALIGNED). The selected model and every published number are unchanged.
- **A3 K change.** New module `backbone/hssm/k_change.py`; `FitSummary.k_change`, `LineageRecord.k_change`, lineage reason `k_change_suspected` (state `CONTINUITY_UNVERIFIED`, old names kept, never "same as before"); `summarize_fit(..., k_change=None)` and `make_summary(..., k_change=None)` gain an optional argument; config `K_CHANGE_RECENT_DAYS`, `K_CHANGE_DELTA_UP`, `K_CHANGE_DELTA_DOWN`.
- **B3 "unsure number of phases".** New modules `backbone/hssm/k_adequacy.py` and `backbone/hssm/alerts.py`; `HSSMResult.k_adequacy` and the property `k_unsure` (baseline; `{}` for the dual engine); one developer alert per unsure fit (hashed user id); config `K_ADEQUACY_*`. Which K is selected does not change.
- **B1 cold-start certification (opt-in).** New module `backbone/hssm/cold_start_cert.py`; `fit_hssm(..., certify_prior_anchored=False)`; `HSSMResult.certification`, property `certified_prior_anchored`; state `CERTIFIED_PRIOR_ANCHORED`; `HSSMResult.withheld_reason` is now built from the new `_withheld_reasons()` (same text, same order; a certified prior-anchored fit drops only `fit_state_not_converged`); config `COLD_START_CERT_*`.
- **A2 gate and sharpness floor.** `AlignmentConfig.min_separation` default 4.0 -> **2.5** (observable: more lineage steps can be ALIGNED; the development data show 0 of 540 wrong-and-unflagged); new `config.FIT_MIN_POSTERIOR_SHARPNESS = 0.95` and `fit_stability.posterior_sharpness` / `sharpness_unstable`: a baseline fit with a mean maximum posterior below it is FIT_UNSTABLE with reason `low_posterior_sharpness` (it flagged 3 of 960 development fits). No fitted number changes.
- **B5 monitoring primitives.** New module `backbone/hssm/monitoring.py` (`predictability_monitor`, `alert_on_drop`, `confidence_display_gate`, `baseline_predictive_increments`); `config.N17_PASSED = False`, `CONFIDENCE_DISPLAY_THRESHOLD = 0.85`, `MONITOR_*`. Nothing calls them from the fit path.
- **B6 re-certification.** New module `backbone/hssm/recertify.py` (`recertify`, `recertify_due`, `recertification_sweep`); the publish rule is the existing one and is never relaxed. Nothing calls it from the fit path.
- **Study tooling (no behaviour change):** `data/synthetic_identity.generate_identity_stream(..., feature_scale_change=None)` (A4 negative control; default streams bit-identical, 8 hashes pinned in tests).
- **Test changes to older packages:** `tests/test_wp8_item4_engines.py` uses a 120-day anchor (the 90-day fit of seed 8791 is FIT_UNSTABLE under A1); `tests/test_wp8_item4_freeze.py` pins the A2 gate default 2.5 (WP8: 4.0) and its gate-boundary test moved to 2.5; `tests/test_wp9_b3_k_adequacy.py` tests the mechanics at explicit thresholds and pins the frozen defaults separately; `tests/test_wp7r_item0_contract_doc.py` reads `_withheld_reasons` as well as `withheld_reason` (B1 moved the reason list); `tests/test_wp5_integration.py::test_new_arguments_are_last_optional_and_default_to_none` lists `certify_prior_anchored` as the new last argument of `fit_hssm` (default False, off), the earlier three unchanged.

## WP8 (branch `ledger/wp8`)

### Item 0b: BIC parameter-count default is now "free" (owner decision B1)
- **What changed:** `GaussianHSMM.n_params_convention` defaults to `"free"` (was `"legacy"`). The free count is `(K-1) + K(K-2) + 2KF + 2K`; the legacy count was `(K-1) + K(K-1) + 2KF + 2K`, which counts the zero transition diagonal and so over-counts by `K` parameters (BIC penalty too large by `K ln n`).
- **Observable effect:** `n_params()`, `bic()`, `aic()` of a baseline model, and the BIC values stored in `HSSMResult` / K-selection logs, are smaller by `K` parameters (BIC by `K ln n`). EM fits are untouched: log-likelihoods are bit-identical.
- **Selection effect (measured):** on the 48 WP7-R K-selection datasets the selected K is identical in 48 of 48 (`tests/WP8_ITEM0_RESULTS.md`, `wp8_evidence/item0b_compare.txt`).
- **Old behaviour:** still selectable, `model.n_params_convention = "legacy"` (class attribute; bit-for-bit the previous numbers, test-covered).

### Item 0e: result metadata reports what happened (no constants)
- `DualTimescaleHSSM.model_capability["is_production_grade_full_hssm"]` was `True` for every dual model. It is now `True` only for a fitted model whose fit state is `CONVERGED`, with no Cholesky fallback and no duration parameter at a wall of the duration box (`is_certified_fit_`). An unfitted dual model reports `False`.
- The dual path of `fit_hssm` stored `optimizer_success=True` always. It now stores `not duration_fallback_used_`, as the baseline path already did.
- `duration_model_identity` of both models said `"explicit_truncated_gaussian"`; the code fits a truncated, renormalised discrete log-normal. Both now say `"explicit_truncated_lognormal"`.
- These are metadata only: no fitted number, publish decision (`insights_withheld`) or served output changed.

### Item 0c / 0d
- Comment-only change to the D12 status line above the K=3 test (`ast.dump` identical). Requirements split: packages nothing imports moved from `requirements.txt` to `requirements-optional.txt` with the module that needs each (owner decision C7).
