"""WP10B A0: the mutation runner's classification of a mutant (`integration_report/wp10b/scripts/mutation_runner10b.py`): caught only when a NAMED test fails by its own assertion; a crash, an error, a missing or a skipped
named test is not caught; a pass is a survivor. Synthetic junit reports; the runner itself is exercised end to end by the mutation sets (`mutation_wp10b_a0.py` and later), whose first version showed two survivors that were then traced to a test and to redundant code."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("wp10b_mut_runner_under_test", ROOT / "integration_report" / "wp10b" / "scripts" / "mutation_runner10b.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


def _q(msg):
    return escape(msg, {'"': "&quot;"})


def junit(*cases):
    body = ""
    for name, kind, msg in cases:
        inner = {"pass": "", "fail": '<failure message="%s">trace</failure>' % _q(msg), "error": '<error message="%s">trace</error>' % _q(msg), "skip": '<skipped message="s"/>'}[kind]
        body += f'<testcase classname="tests.test_x" name="{name}">{inner}</testcase>'
    return f'<?xml version="1.0"?><testsuites><testsuite>{body}</testsuite></testsuites>'


NAMED = ["tests/test_x.py::test_rule"]


@pytest.mark.parametrize("msg", ["assert 1 == 2", "AssertionError: wrong count", "Failed: DID NOT RAISE <class 'ValueError'>"])
def test_the_tests_own_assertion_failing_is_caught(msg):
    assert M.classify(junit(("test_rule", "fail", msg)), NAMED)[0] == "CAUGHT"


@pytest.mark.parametrize("msg", ["TypeError: bad operand", "NameError: name 'x' is not defined", "ImportError: cannot import", "SyntaxError: invalid syntax", "ZeroDivisionError: division by zero"])
def test_a_crash_of_the_planted_code_is_not_caught(msg):
    assert M.classify(junit(("test_rule", "fail", msg)), NAMED)[0] == "CRASH"


def test_an_error_is_not_caught():
    assert M.classify(junit(("test_rule", "error", "collection error")), NAMED)[0] == "CRASH"


def test_a_pass_is_a_survivor():
    assert M.classify(junit(("test_rule", "pass", "")), NAMED)[0] == "SURVIVED"


def test_a_named_test_that_did_not_run_is_not_caught():
    assert M.classify(junit(("some_other_test", "fail", "assert False")), NAMED)[0] == "NOT RUN"


def test_a_skipped_named_test_is_not_caught():
    assert M.classify(junit(("test_rule", "skip", "")), NAMED)[0] == "NOT RUN"


def test_another_test_failing_does_not_count_for_this_mutant():
    """Only the tests named for the mutant decide; an unrelated failing test in the report is ignored."""
    status, _ = M.classify(junit(("test_rule", "pass", ""), ("test_unrelated", "fail", "assert False")), NAMED)
    assert status == "SURVIVED"


def test_parametrised_cases_of_a_named_test_count():
    assert M.classify(junit(("test_rule[a]", "pass", ""), ("test_rule[b]", "fail", "assert False")), NAMED)[0] == "CAUGHT"


def test_a_crash_in_one_case_and_an_assertion_in_another_is_caught_by_the_assertion():
    assert M.classify(junit(("test_rule[a]", "fail", "TypeError: x"), ("test_rule[b]", "fail", "assert False")), NAMED)[0] == "CAUGHT"
