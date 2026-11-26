"""WP8 Item 4: the baseline label convention is documented honestly (it is not an activity ordering) and its behaviour did not change (owner decision A4, brief Item 4).
Behaviour is pinned two ways: a numeric check against an inline reference, and a fingerprint of each function without its docstring, computed from the file at 27c591c.

WP10A A1 (finding F1): the original fingerprint was the sha256 of `ast.dump(...)`, whose text differs between Python minor versions (3.13 omits empty and None fields that 3.12 prints), so the pins
failed on 3.12 although no function had changed. The fingerprint is now computed from a canonical form of the AST that does not depend on the formatting of `ast.dump` (`_canon`: node class,
fields in their declared order, None and empty-list fields left out, constants by type and repr); the pins were recomputed from the file at 27c591c and are identical on 3.11, 3.12 and 3.13
(RUN). It protects the same thing as before: the function bodies, arguments, decorators and return annotations are unchanged since WP7-R (docstrings, comments, blank lines and parentheses are
not part of the fingerprint). The old `ast.dump` pins are kept in a test that runs on Python 3.13 only. The fingerprint still fails on a planted change: `test_the_fingerprint_fails_on_every_planted_change`
(in the suite) and `integration_report/wp10a/scripts/mutation_wp10a_a1.py` (mutants planted in a throw-away copy of the repository)."""
from __future__ import annotations

import ast
import hashlib
import inspect
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from backbone.hssm import label_switching as ls

# canonical fingerprints (WP10A A1), computed from the file at 27c591c; equal on Python 3.11, 3.12 and 3.13 and equal for the current file
PINNED = {"canonicalize_labels": "5240d304f550fad1", "canonicalize_regime_order": "a033c02e70976bed", "activity_levels": "1cb73374b1647aba"}
# the WP8 pins: sha256 of ast.dump of the same functions, valid for the ast.dump format of Python 3.13 only
PINNED_LEGACY_AST_DUMP_PY313 = {"canonicalize_labels": "78c5ab74b05af982", "canonicalize_regime_order": "c11dd3be1bd0c373", "activity_levels": "f521af2aaf83ea9d"}   # 27c591c


def _canon(node):
    """Canonical, Python-version-independent form of an AST: the same tree gives the same value whatever the `ast.dump` format of the running interpreter."""
    if isinstance(node, ast.Constant):
        return ("Constant", type(node.value).__name__, repr(node.value))
    if isinstance(node, ast.AST):
        items = []
        for f in node._fields:
            v = getattr(node, f, None)
            if v is None or (isinstance(v, list) and not v):
                continue
            items.append((f, _canon(v)))
        return (type(node).__name__, tuple(items))
    if isinstance(node, list):
        return tuple(_canon(x) for x in node)
    return repr(node)


def _without_docstring(fn_node: ast.FunctionDef):
    body = fn_node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    return body


def _digest(fn_node: ast.FunctionDef) -> str:
    payload = repr((fn_node.name, _canon(fn_node.args), _canon(_without_docstring(fn_node)), _canon(fn_node.decorator_list), _canon(fn_node.returns)))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _legacy_digest(fn_node: ast.FunctionDef) -> str:
    clone = ast.FunctionDef(name=fn_node.name, args=fn_node.args, body=_without_docstring(fn_node), decorator_list=fn_node.decorator_list, returns=fn_node.returns, type_comment=None, type_params=[])
    return hashlib.sha256(ast.dump(clone).encode()).hexdigest()[:16]


def _digests_of(source: str) -> dict:
    return {n.name: _digest(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}


def _model(seed, K=4, F=5):
    r = np.random.default_rng(seed)
    A = r.random((K, K)); A /= A.sum(axis=1, keepdims=True)
    return SimpleNamespace(pi=r.dirichlet(np.ones(K)), A=A, mu=r.standard_normal((K, F)), var=r.random((K, F)) + 0.1, dur_mu=r.random(K), dur_sigma=r.random(K),
                           duration_optimizer_success_={k: bool(k % 2) for k in range(K)}, duration_optimizer_messages_={k: f"m{k}" for k in range(K)})


def test_function_bodies_are_unchanged_since_wp7r():
    assert _digests_of(inspect.getsource(ls)) == PINNED


@pytest.mark.skipif(sys.version_info[:2] != (3, 13), reason="the WP8 pins are sha256 of ast.dump text, which is valid for the ast.dump format of Python 3.13 only (WP10A A1, finding F1)")
def test_function_bodies_are_unchanged_since_wp7r_legacy_ast_dump_pins():
    tree = ast.parse(inspect.getsource(ls))
    assert {n.name: _legacy_digest(n) for n in tree.body if isinstance(n, ast.FunctionDef)} == PINNED_LEGACY_AST_DUMP_PY313


PLANTED = [                                                                    # (function, text that occurs once in the module source, replacement, what it plants)
    ("canonicalize_labels", "order = np.argsort(activity)", "order = np.argsort(-activity)", "reverse the sort order (a behaviour change)"),
    ("canonicalize_labels", "activity = np.linalg.norm(model.mu, axis=1)", "activity = np.linalg.norm(model.mu, axis=0)", "sort key over the wrong axis (a behaviour change)"),
    ("canonicalize_labels", "    model.dur_sigma = model.dur_sigma[order]\n", "", "one array is no longer permuted (a behaviour change)"),
    ("canonicalize_labels", "get(int(old_k), True)", "get(int(old_k), False)", "a default value changes"),
    ("canonicalize_labels", "model._label_order_applied = order.tolist()", "model._label_order_applied = list(order)", "the audit attribute changes type (a behaviour-neutral rewrite)"),
    ("canonicalize_labels", "def canonicalize_labels(model) -> object:", "def canonicalize_labels(model, extra=None) -> object:", "the signature changes"),
    ("canonicalize_regime_order", "return canonicalize_labels(model)", "return model", "the alias no longer calls the function"),
    ("activity_levels", "return np.linalg.norm(model.mu, axis=1)", "return np.linalg.norm(model.mu, axis=0)", "wrong axis"),
    ("activity_levels", "def activity_levels(model) -> np.ndarray:", "def activity_levels(model) -> float:", "the return annotation changes"),
]


@pytest.mark.parametrize("fn,old,new,what", PLANTED, ids=[w for *_, w in PLANTED])
def test_the_fingerprint_fails_on_every_planted_change(fn, old, new, what):
    src = inspect.getsource(ls)
    assert src.count(old) == 1, "the planted text must occur exactly once"
    got = _digests_of(src.replace(old, new, 1))
    assert got[fn] != PINNED[fn], what
    assert {k: v for k, v in got.items() if k != fn} == {k: v for k, v in PINNED.items() if k != fn}      # only the changed function moves


def test_the_fingerprint_ignores_comments_docstrings_blank_lines_and_parentheses():
    src = inspect.getsource(ls)
    doc = '"""Legacy alias matching foundation\'s old label switching interface."""'
    assert "    order = np.argsort(activity)\n" in src and doc in src
    src = src.replace("    order = np.argsort(activity)\n", "    # a comment\n\n    order = (np.argsort(activity))\n", 1)
    src = src.replace(doc, '"""A different docstring."""', 1)
    src = re.sub(r"\n\n\n", "\n\n\n\n", src)
    assert _digests_of(src) == PINNED


def test_the_fingerprint_is_not_the_text_of_ast_dump():
    """The canonical form must not depend on how `ast.dump` prints a tree: a tree built by hand with every optional field spelled out hashes like the parsed one."""
    fn = next(n for n in ast.parse(inspect.getsource(ls)).body if isinstance(n, ast.FunctionDef) and n.name == "activity_levels")
    spelled = ast.FunctionDef(name=fn.name, args=fn.args, body=fn.body, decorator_list=[], returns=fn.returns, type_comment=None, type_params=[])
    assert _digest(spelled) == _digest(fn) == PINNED["activity_levels"]


@pytest.mark.parametrize("seed", range(12))
def test_canonicalize_labels_sorts_by_l2_norm_and_permutes_every_array_alike(seed):
    m = _model(seed); ref = SimpleNamespace(**{k: (v.copy() if hasattr(v, "copy") else v) for k, v in vars(m).items()})
    order = np.argsort(np.linalg.norm(ref.mu, axis=1))
    out = ls.canonicalize_labels(m)
    assert out is m and m._label_order_applied == order.tolist()
    np.testing.assert_array_equal(m.mu, ref.mu[order]); np.testing.assert_array_equal(m.var, ref.var[order])
    np.testing.assert_array_equal(m.pi, ref.pi[order]); np.testing.assert_array_equal(m.A, ref.A[np.ix_(order, order)])
    np.testing.assert_array_equal(m.dur_mu, ref.dur_mu[order]); np.testing.assert_array_equal(m.dur_sigma, ref.dur_sigma[order])
    assert m.duration_optimizer_success_ == {new: ref.duration_optimizer_success_[int(old)] for new, old in enumerate(order)}
    assert m.duration_optimizer_messages_ == {new: ref.duration_optimizer_messages_[int(old)] for new, old in enumerate(order)}
    assert list(ls.activity_levels(m)) == sorted(np.linalg.norm(ref.mu, axis=1))


def test_the_constant_keeps_its_value_and_has_an_honest_name_next_to_the_old_one():
    assert ls.ACTIVITY_LEVEL_CONVENTION == "mean_L2_norm_of_regime_mean_vector"
    assert ls.LABEL_ORDER_CONVENTION == ls.ACTIVITY_LEVEL_CONVENTION


def test_the_documentation_no_longer_claims_an_activity_ordering():
    for doc in (ls.__doc__, ls.canonicalize_labels.__doc__, ls.activity_levels.__doc__):
        low = " ".join(doc.lower().split())
        assert "not an activity" in low, doc
    mod = " ".join(ls.__doc__.lower().split())
    for claim in ("regime 0 = lowest", "sort regimes by mean behavioral activity", "never to be changed"):
        assert claim not in mod
    assert "7.7%" in ls.__doc__ and "lineage.py" in ls.__doc__ and "ordering.py" in ls.__doc__
