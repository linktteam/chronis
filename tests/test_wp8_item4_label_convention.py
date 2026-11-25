"""WP8 Item 4: the baseline label convention is documented honestly (it is not an activity ordering) and its behaviour did not change (owner decision A4, brief Item 4).
Behaviour is pinned two ways: a numeric check against an inline reference, and the sha256 of the AST of each function without its docstring, computed from the file at 27c591c."""
from __future__ import annotations

import ast
import hashlib
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from backbone.hssm import label_switching as ls

PINNED = {"canonicalize_labels": "78c5ab74b05af982", "canonicalize_regime_order": "c11dd3be1bd0c373", "activity_levels": "f521af2aaf83ea9d"}   # 27c591c


def _digest(fn_node: ast.FunctionDef) -> str:
    body = fn_node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    clone = ast.FunctionDef(name=fn_node.name, args=fn_node.args, body=body, decorator_list=fn_node.decorator_list, returns=fn_node.returns, type_comment=None, type_params=[])
    return hashlib.sha256(ast.dump(clone).encode()).hexdigest()[:16]


def _model(seed, K=4, F=5):
    r = np.random.default_rng(seed)
    A = r.random((K, K)); A /= A.sum(axis=1, keepdims=True)
    return SimpleNamespace(pi=r.dirichlet(np.ones(K)), A=A, mu=r.standard_normal((K, F)), var=r.random((K, F)) + 0.1, dur_mu=r.random(K), dur_sigma=r.random(K),
                           duration_optimizer_success_={k: bool(k % 2) for k in range(K)}, duration_optimizer_messages_={k: f"m{k}" for k in range(K)})


def test_function_bodies_are_unchanged_since_wp7r():
    tree = ast.parse(inspect.getsource(ls))
    got = {n.name: _digest(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert got == PINNED


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
