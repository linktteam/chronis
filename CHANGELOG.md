# Changelog

Entries are for behaviour that a consumer of `backbone/hssm/` can observe. Newest first. Module paths and field names under `backbone/hssm/` are stable; new fields are additive only.

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
