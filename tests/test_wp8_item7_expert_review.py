"""WP8 Item 7 (owner decision A7): every place that defines or uses the matching distance, a ladder weight or an alignment threshold is tagged `EXPERT_REVIEW(identity-distance)`,
`docs/EXPERT_REVIEW_REGISTER.md` is generated from the tags, a stale register fails this test, and every lineage record says `expert_review_status = "PENDING"`.
Mutation check: integration_report/wp8/scripts/mutation_item7.py."""
from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from backbone.hssm import lineage as L
from backbone.hssm.lineage import ALIGNED, CONTINUITY_UNVERIFIED, K_CHANGED, NEW_LINEAGE, AlignmentConfig, make_summary, step
from backbone.hssm.ordering import LadderConfig
from data.synthetic_identity import REGISTRY, SCALE, IdentitySpec, draw_user

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("expert_review_register", ROOT / "integration_report" / "wp8" / "scripts" / "expert_review_register.py")
R = importlib.util.module_from_spec(_spec)
sys.modules["expert_review_register"] = R
_spec.loader.exec_module(R)

# the sites that must stay tagged (file, qualified definition). If one of these is untagged the register may be regenerated and still pass the stale-register test,
# so they are pinned here as well. The list is the minimum; the AST scan below catches any other function that uses a watched name without a tag.
REQUIRED = [
    ("backbone/hssm/lineage.py", "<module>"), ("backbone/hssm/lineage.py", "PairThreshold"), ("backbone/hssm/lineage.py", "ChannelConfig"),
    ("backbone/hssm/lineage.py", "AlignmentConfig"), ("backbone/hssm/lineage.py", "LineageRecord"), ("backbone/hssm/lineage.py", "separation_min"),
    ("backbone/hssm/lineage.py", "hellinger2_matrix"), ("backbone/hssm/lineage.py", "_vectors"), ("backbone/hssm/lineage.py", "alignment"),
    ("backbone/hssm/lineage.py", "alignment_from_cost"), ("backbone/hssm/lineage.py", "margin_ok"), ("backbone/hssm/lineage.py", "_combined_alignment"),
    ("backbone/hssm/lineage.py", "step"), ("backbone/hssm/lineage.py", "name_fit"), ("backbone/hssm/lineage.py", "split_merge_candidates"),
    ("backbone/hssm/ordering.py", "<module>"), ("backbone/hssm/ordering.py", "RUNG_ROLES"), ("backbone/hssm/ordering.py", "LadderConfig"),
    ("backbone/hssm/ordering.py", "_rung_weights_scores"), ("backbone/hssm/ordering.py", "order_regimes"), ("backbone/hssm/refit_policy.py", "<module>"),
]


# ------------------------------------------------------------------------------------------------------------------------------------------------ the register
def test_the_committed_register_equals_what_the_code_generates():
    path = ROOT / "docs" / "EXPERT_REVIEW_REGISTER.md"
    assert path.exists(), "docs/EXPERT_REVIEW_REGISTER.md is missing: run integration_report/wp8/scripts/expert_review_register.py --write"
    assert path.read_text() == R.render(), "the register is stale: run integration_report/wp8/scripts/expert_review_register.py --write and commit it"


def test_the_register_lists_every_tagged_site_once_and_only_existing_sites():
    text = (ROOT / "docs" / "EXPERT_REVIEW_REGISTER.md").read_text()
    sites = R.all_sites()
    assert f"## Tagged sites ({len(sites)})" in text
    rows = [ln for ln in text.splitlines() if ln.startswith("| ") and ln.split("|")[1].strip().isdigit()]
    assert len(rows) == len(sites) and len(sites) >= len(REQUIRED)
    listed = set()
    for ln in rows:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        file, sym = cells[1].strip("`"), cells[2].strip("`")
        listed.add((file, "<module>" if sym == "module" else sym))
        assert (ROOT / file).exists(), f"the register lists a file that does not exist: {file}"
    found = {(s["file"], s["symbol"]) for s in sites}
    assert listed == found, "the register and the code disagree on which sites are tagged"


def test_a_site_that_vanished_or_appeared_changes_the_generated_register(tmp_path):
    """The stale-register test is only as good as the generator: removing a tag, adding a tag, or editing a tag's text must each change the output (negative controls)."""
    src = textwrap.dedent('''
        """Module text.

        EXPERT_REVIEW(identity-distance): the module defines a threshold."""

        # EXPERT_REVIEW(identity-distance): the constant below is a threshold.
        LIMIT = 3

        def f(x):
            """EXPERT_REVIEW(identity-distance): f applies the threshold."""
            return x < LIMIT
        ''')
    pkg = tmp_path / "mod.py"; pkg.write_text(src)
    base = R.find_sites(pkg, tmp_path)
    assert sorted((s["symbol"], s["kind"]) for s in base) == [("<module>", "docstring"), ("LIMIT", "comment"), ("f", "docstring")]
    for changed in (src.replace("# EXPERT_REVIEW(identity-distance): the constant below is a threshold.\n", ""),
                    src.replace('"""EXPERT_REVIEW(identity-distance): f applies the threshold."""', '"""f applies the threshold."""'),
                    src.replace("f applies the threshold", "f applies another threshold"),
                    src + "\n# EXPERT_REVIEW(identity-distance): g is new.\ndef g():\n    return 1\n"):
        pkg.write_text(changed)
        assert R.find_sites(pkg, tmp_path) != base


def test_a_docstring_that_only_mentions_the_tag_is_not_a_site_and_a_dangling_comment_is_refused(tmp_path):
    p = tmp_path / "m.py"
    p.write_text('def f():\n    """This function is not marked: see EXPERT_REVIEW(identity-distance) for what a mark means."""\n    return 1\n')
    assert R.find_sites(p, tmp_path) == []
    p.write_text("x = 1\n# EXPERT_REVIEW(identity-distance): nothing follows this comment\n")
    with pytest.raises(ValueError, match="not followed by"):
        R.find_sites(p, tmp_path)


def test_a_comment_belongs_to_the_definition_below_it_not_the_one_above_and_methods_are_qualified(tmp_path):
    p = tmp_path / "m.py"
    p.write_text(textwrap.dedent('''
        def above():
            return 1

        class A:
            # EXPERT_REVIEW(identity-distance): this method validates a threshold.
            def check(self):
                return 2

        class B:
            def check(self):
                return 3

        # EXPERT_REVIEW(identity-distance): decorated class.
        @staticmethod
        class C:
            pass
        '''))
    got = [(s["symbol"], s["kind"]) for s in R.find_sites(p, tmp_path)]
    assert sorted(got) == [("A.check", "comment"), ("C", "comment")]


def test_two_tags_on_one_definition_are_both_kept(tmp_path):
    p = tmp_path / "m.py"
    p.write_text('# EXPERT_REVIEW(identity-distance): first.\ndef f():\n    """EXPERT_REVIEW(identity-distance): second."""\n    return 1\n')
    assert sorted(s["kind"] for s in R.find_sites(p, tmp_path)) == ["comment", "docstring"]


# ------------------------------------------------------------------------------------------------------------------------------------ every use is tagged
@pytest.mark.parametrize("site", REQUIRED, ids=[f"{f.split('/')[-1]}::{s}" for f, s in REQUIRED])
def test_required_sites_stay_tagged(site):
    found = {(s["file"], s["symbol"]) for s in R.all_sites()}
    assert site in found


@pytest.mark.parametrize("path", sorted((ROOT / "backbone" / "hssm").glob("*.py")), ids=lambda p: p.name)
def test_no_function_uses_the_distance_a_ladder_weight_or_a_threshold_without_the_tag(path):
    assert R.untagged_uses(path) == []


def test_the_use_scan_finds_an_untagged_use_by_name_and_by_attribute_and_accepts_a_tagged_one(tmp_path):
    p = tmp_path / "m.py"
    p.write_text(textwrap.dedent('''
        def by_name(a, b):
            return margin_ok(a, b)

        def by_attribute(cfg):
            return cfg.min_separation

        class K:
            def method(self, cfg):
                return cfg.cross_engine

        # EXPERT_REVIEW(identity-distance): tagged use.
        def tagged(cfg):
            return cfg.min_separation

        def unrelated(x):
            return x + 1
        '''))
    assert [q for q, _ in R.untagged_uses(p)] == ["K.method", "by_attribute", "by_name"]


def test_the_watched_names_exist_in_the_package_so_that_a_rename_cannot_silence_the_scan():
    names = set()
    import ast
    for path in (ROOT / "backbone" / "hssm").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
    missing = sorted(n for n in R.WATCHED if n not in names)
    assert missing == [], f"watched names that no longer exist in backbone/hssm (rename them in WATCHED too): {missing}"


# --------------------------------------------------------------------------------------------------------------------------------------- status on records
CFG = AlignmentConfig(min_separation=1.0, ladder=LadderConfig(n_bootstrap=40, min_gap=0.5))


def _summary(means, seed=0, T=240, noise=0.35):
    rng = np.random.default_rng(seed)
    K, F = means.shape
    labels = rng.permutation(np.repeat(np.arange(K), T // K)[:T])
    window = means[labels] + noise * SCALE * rng.standard_normal((len(labels), F))
    return make_summary("baseline", means, window.mean(axis=0), window.std(axis=0), labels, window)


def _means(seed, K=3, sep=3.0):
    return draw_user(seed, IdentitySpec(n_regimes=K, sep_lo=sep, sep_hi=sep, offset_sd=0.3))["means"]


def test_the_status_constant_is_pending_and_the_register_says_so():
    assert L.EXPERT_REVIEW_STATUS == "PENDING"
    assert "**Status: PENDING.**" in (ROOT / "docs" / "EXPERT_REVIEW_REGISTER.md").read_text()


def test_every_kind_of_record_says_pending_and_so_does_the_result_field():
    m3 = _means(8791)
    first, st = step(None, _summary(m3, 1), CFG, REGISTRY, lineage_id="u:7")
    aligned, st = step(st, _summary(m3, 2), CFG, REGISTRY)
    k_changed, st2 = step(st, _summary(_means(8792, K=2), 3), CFG, REGISTRY)
    unverified, _ = step(st, _summary(m3, 4), AlignmentConfig(min_separation=1e6, ladder=CFG.ladder), REGISTRY)
    assert [r.continuity_state for r in (first, aligned, k_changed, unverified)] == [NEW_LINEAGE, ALIGNED, K_CHANGED, CONTINUITY_UNVERIFIED]
    for rec in (first, aligned, k_changed, unverified):
        assert rec.expert_review_status == "PENDING"
        d = rec.as_dict()
        assert d["expert_review_status"] == "PENDING"
        json.dumps(d)                                              # the field survives serialisation

    class Result:                                                  # the additive HSSMResult field carries it too
        lineage = None
    res = L.annotate_result(Result(), aligned)
    assert res.lineage["expert_review_status"] == "PENDING"
    assert "expert_review_status" in L.RECORD_COLUMNS


def test_a_record_cannot_be_built_as_reviewed_by_the_default_path():
    """The status is a field default taken from the constant: nothing in the step path writes another value."""
    import dataclasses
    f = {x.name: x for x in dataclasses.fields(L.LineageRecord)}["expert_review_status"]
    assert f.default == "PENDING"


# ---------------------------------------------------------------------------------------------------------------------------------------- script and contract
def test_check_mode_fails_when_the_register_is_stale_and_passes_when_it_is_current(tmp_path, monkeypatch, capsys):
    stale = tmp_path / "REGISTER.md"
    monkeypatch.setattr(R, "REGISTER", stale)
    assert R.main(["--check"]) == 1                                # missing
    stale.write_text(R.render() + "\nextra line\n")
    assert R.main(["--check"]) == 1                                # stale
    assert R.main(["--write"]) == 0 and stale.read_text() == R.render()
    assert R.main(["--check"]) == 0                                # current


def test_the_register_prints_the_numbers_under_review_from_the_code():
    text = (ROOT / "docs" / "EXPERT_REVIEW_REGISTER.md").read_text()
    cfg = AlignmentConfig()
    assert f"min_separation={cfg.min_separation!r}" in text
    assert f"min_gap={LadderConfig().min_gap!r}" in text and f"min_modal_freq={LadderConfig().min_modal_freq!r}" in text
    assert f"tau={cfg.same_engine.tau!r}" in text and f"rho={cfg.cross_engine.rho!r}" in text
    assert "refit_interval_days=7" in text and "bounds 1..28 days" in text


def test_the_contract_names_the_register_and_the_pending_status():
    doc = (ROOT / "docs" / "INTEGRATION_CONTRACT_HSSM.md").read_text()
    assert "docs/EXPERT_REVIEW_REGISTER.md" in doc and 'expert_review_status = "PENDING"' in doc
    assert "EXPERT_REVIEW(identity-distance)" in doc


def test_every_threshold_of_the_configuration_classes_is_a_watched_name():
    """A new class of pair threshold (the Item 4 freeze added `dual_engine`) or a new channel threshold must not escape the tag scan: every PairThreshold field of the configurations is in WATCHED."""
    from dataclasses import fields
    from backbone.hssm.lineage import ChannelConfig, PairThreshold
    found = [(cls.__name__, f.name) for cls in (AlignmentConfig, ChannelConfig) for f in fields(cls) if isinstance(getattr(cls(), f.name), PairThreshold)]
    assert {n for _, n in found} >= {"same_engine", "dual_engine", "cross_engine", "context_threshold"}
    missing = [(c, n) for c, n in found if n not in R.WATCHED]
    assert not missing, f"threshold fields that the untagged-use scan does not watch: {missing}"
