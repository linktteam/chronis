"""WP7-R item 0: the contract document (docs/INTEGRATION_CONTRACT_HSSM.md, section 9) says what the code does.

Five doc tests. Each compares the document with something READ from the code or RUN through the smoke script, so a change in either that is not mirrored in the
other fails here. Mutation check: integration_report/wp7r/scripts/mutation_item0.py (every test is killed by at least one mutant).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import re
from pathlib import Path

import pytest

from backbone.hssm import config as hssm_config
from backbone.hssm import fitting as hssm_fitting
from backbone.hssm import gating, model as hssm_model, shrinkage

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "INTEGRATION_CONTRACT_HSSM.md").read_text()


def _section(title_prefix: str) -> str:
    """Text of the section whose heading starts with `title_prefix` (up to the next heading of the same or a higher level)."""
    m = re.search(rf"^(#+) {re.escape(title_prefix)}.*$", DOC, flags=re.M)
    assert m, f"heading {title_prefix!r} missing from the contract doc"
    level = len(m.group(1))
    rest = DOC[m.end():]
    nxt = re.search(rf"^#{{1,{level}}} ", rest, flags=re.M)
    return rest[: nxt.start()] if nxt else rest


@pytest.fixture(scope="module")
def smoke():
    spec = importlib.util.spec_from_file_location("hssm_contract_smoke", ROOT / "scripts" / "hssm_contract_smoke.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run()


def test_every_fitstate_value_has_a_row_in_the_table():
    table = _section("9.1 Every `FitState` value")
    rows = {m.group(1) for m in re.finditer(r"^\| `([A-Z_]+)` \|", table, flags=re.M)}
    assert rows == {s.value for s in hssm_model.FitState}, (rows, [s.value for s in hssm_model.FitState])
    # every row says what the caller does and whether the fit is withheld: three non-empty cells after the name
    for line in table.splitlines():
        if re.match(r"^\| `[A-Z_]+` \|", line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            assert len(cells) >= 4 and all(cells[1:4]), line


def test_every_exception_class_is_in_the_exceptions_table_and_the_hierarchy_is_as_stated():
    table = _section("6. Exceptions a caller can see")
    classes = [hssm_model.NotFittedError, hssm_model.FittingConvergenceError, hssm_model.InsightsWithheldError, hssm_model.InternalStateError,
               gating.ColdStartError]
    for cls in classes:
        assert f"`{cls.__name__}`" in table, cls.__name__
    assert "`ValueError`" in table and "`MemoryError`" in table
    # the shrinkage prior builder has its own exception; it is not raised by fit_hssm, but a caller that builds a prior sees it
    assert issubclass(shrinkage.PriorBuildError, hssm_model.HSSMError)
    assert "PriorBuildError" in _section("9.3 Metadata") + _section("9.2 Withheld semantics") + _section("6. Exceptions a caller can see")
    # every library exception derives from HSSMError except ColdStartError (a plain Exception); both facts are used by callers that catch broadly
    assert all(issubclass(c, hssm_model.HSSMError) for c in classes if c is not gating.ColdStartError)
    assert issubclass(gating.ColdStartError, Exception) and not issubclass(gating.ColdStartError, hssm_model.HSSMError)


def test_withheld_semantics_in_the_doc_match_every_reason_prefix_of_the_code():
    sec = _section("9.2 Withheld semantics")
    src = inspect.getsource(hssm_fitting.HSSMResult.withheld_reason.fget)
    prefixes = set(re.findall(r'reasons\.append\(f?"([a-z_]+):', src))
    assert {"fit_state_not_converged", "collapsed_regime", "degenerate_duration", "fit_degraded", "states_pass", "late_fallbacks"} <= prefixes, prefixes
    assert "fit_state is NUMERICAL_FAILURE" in src
    for pre in sorted(prefixes) + ["fit_state is NUMERICAL_FAILURE"]:
        assert f"`{pre}`" in sec, pre
    # the not-converged states the doc lists in the first row are exactly the code's tuple (no more, no fewer)
    row = next(line for line in sec.splitlines() if line.startswith("| `fit_state_not_converged`"))
    listed = set(re.findall(r"`([A-Z_]+)`", row))
    assert listed == {st.value for st in hssm_fitting._NOT_CONVERGED_STATES}, (listed, hssm_fitting._NOT_CONVERGED_STATES)
    assert hssm_config.COLLAPSE_MIN_EFFECTIVE_DAYS == 4.0 and "`COLLAPSE_MIN_EFFECTIVE_DAYS` = 4" in sec
    # WP7-R item 5 (deliberate change of this assertion, which used to say "NOT withheld by this rule"): a dual collapse and a degenerate duration ARE withheld,
    # and the doc says what the dual collapse is computed from and what "still rising at the lattice end" means
    assert "NOT withheld by this rule" not in sec and "NOT a withheld reason" not in sec
    assert "FILTERED" in sec and "`pmf[Dmax] > pmf[Dmax-1]`" in sec and "`DUAL_DURATION_BOUND_TOL` = 0.001" in sec
    assert hssm_config.DUAL_DURATION_BOUND_TOL == 0.001


def test_metadata_fields_named_in_the_doc_are_the_ones_the_smoke_run_prints(smoke):
    sec = _section("9.3 Metadata")
    first_col = set()
    for line in sec.splitlines():
        if re.match(r"^\| `", line):
            first_col |= set(re.findall(r"`([A-Za-z0-9_]+)`", line.strip().strip("|").split("|")[0]))
    for name in ("convergence_criterion", "window_gain_at_cap", "duration_box", "duration_at_bound", "collapse_report", "restart_summary", "guard_repairs",
                 "fit_duration_degenerate", "duration_degenerate_report", "dmax_retry",
                 "bic_by_k", "used_shared_prior", "shared_prior_n0", "fit_cholesky_fallbacks", "fit_exceptions"):
        assert name in first_col, name                      # each has its own row in the metadata table
    assert "DURATION_BOX_REFIT" in sec
    # RUN: the dual result carries the duration box keys and the per-regime bound flags; the baseline result carries the collapse and restart keys
    assert set(smoke["dual_short"]["duration_box_keys"]) == {"applied", "first_fit", "n_applications", "refit"}
    assert set(smoke["dual_short"]["duration_at_bound"]) == {0, 1}
    assert smoke["baseline"]["duration_at_bound"] == {} and set(smoke["dual_short"]["collapse_keys"]) == {"collapsed", "effective_days", "min_effective_days", "min_effective_days_threshold", "source"} and smoke["dual_short"]["restart_summary_keys"] == []
    assert {"collapsed", "effective_days", "min_effective_days"} <= set(smoke["baseline"]["collapse_keys"])
    assert set(smoke["dual_short"]["duration_degenerate_keys"]) == {"at_bound", "pmf_rising_at_end", "degenerate_regimes", "degenerate", "Dmax"} and smoke["baseline"]["duration_degenerate_keys"] == []
    assert {"n_restarts", "n_converged", "ari_median"} <= set(smoke["baseline"]["restart_summary_keys"])
    # RUN: the dual at 15 iterations is withheld as the doc says, and publishable_outputs() is None
    d = smoke["dual_short"]
    assert d["insights_withheld"] and d["publishable_outputs_is_none"] and str(d["withheld_reason"]).startswith("fit_state_not_converged")


def test_no_per_session_confidence_and_the_input_conventions_in_the_doc_match_the_code(smoke):
    sec = _section("9.4 What `HSSMResult` does not contain")
    assert "no per-session confidence" in sec
    names = {f.name for f in dataclasses.fields(hssm_fitting.HSSMResult)}
    props = {n for n in dir(hssm_fitting.HSSMResult) if not n.startswith("_")}
    assert not ({"confidence", "session_confidence", "p_t_confidence"} & (names | props))
    assert not smoke["baseline"]["has_session_confidence_field"] and not smoke["dual_short"]["has_session_confidence_field"]
    # timestamps are calendar days with the 20000-day guard, and the checklist says so with the code's own value
    assert hssm_config.MAX_TIMESTAMP_SPAN_DAYS == 20000
    checklist = _section("9.7 Integrator checklist")
    assert "`MAX_TIMESTAMP_SPAN_DAYS` = 20,000" in checklist and "calendar DAYS" in checklist
    assert smoke["errors"]["seconds_as_days"].startswith("ValueError") and smoke["errors"]["cold_start"] == "ColdStartError"
    # defaults quoted in the doc: max_iter 200, at least 10 restarts, 30 present sessions
    sig = inspect.signature(hssm_fitting.fit_hssm)
    assert sig.parameters["max_iter"].default == 200 and sig.parameters["n_initializations"].default == 10
    assert hssm_config.DEFAULT_COLD_START_CONFIG.min_present_sessions == 30
    assert "default `max_iter` 200" in checklist and "at least 30 present sessions" in checklist and "at least 10 restarts" in checklist
