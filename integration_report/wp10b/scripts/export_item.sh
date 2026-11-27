#!/bin/bash
# WP10B export + restore test. usage: bash integration_report/wp10b/scripts/export_item.sh N   (run from the repo root, on branch ledger/wp10b, everything committed; background jobs dirty the tree, so run it from a clean `git worktree add --detach` of HEAD)
# writes /mnt/user-data/outputs/wp10b_itemN/{wp10b_itemN.patch, wp10b_tree.tar.gz, wp10b.bundle, wp10b_evidence_itemN.zip, SHA256SUMS.txt} and runs the restore test.
set -u
PY=${PY:-/home/claude/work/venv312/bin/python}   # Python 3.12 with the repo pins (WP10B rule 7)
exec > >(tee /tmp/wp10b_export_log.$$) 2>&1
N="${1:?item number}"; BASE=95dd125
OUT=/mnt/user-data/outputs/wp10b_item$N; mkdir -p "$OUT"; rm -f "$OUT"/*
[ -z "$(git status --porcelain)" ] || { echo "REFUSED: uncommitted changes"; git status --short | head -5; exit 2; }
HEAD=$(git rev-parse --short HEAD)
git diff --binary $BASE..HEAD > "$OUT/wp10b_item$N.patch"
git archive --format=tar.gz --prefix=chronis_hssm_wp10b/ HEAD > "$OUT/wp10b_tree.tar.gz"
git bundle create "$OUT/wp10b.bundle" ledger/wp7r ledger/wp8 ledger/wp9 ledger/wp10a ledger/wp10b 2>&1 | tail -n 1
( cd wp10b_evidence && zip -q -r "$OUT/wp10b_evidence_item$N.zip" . -x '*.pyc' ) || echo "zip failed"
( cd "$OUT" && sha256sum wp10b_item$N.patch wp10b_tree.tar.gz wp10b.bundle wp10b_evidence_item$N.zip > SHA256SUMS.txt )
echo "exported head $HEAD:"; ls -l "$OUT" | tail -n +2
# ---- restore test (fresh folder; never the working copy) ----
T=$(mktemp -d /tmp/wp10b_restore.XXXXXX); FAIL=0
( cd "$OUT" && sha256sum -c SHA256SUMS.txt >/dev/null 2>&1 ) && echo "restore: SHA256 ok" || { echo "restore: SHA256 FAILED"; FAIL=1; }
mkdir "$T/tree" && tar xzf "$OUT/wp10b_tree.tar.gz" -C "$T/tree"
git bundle verify "$OUT/wp10b.bundle" >/dev/null 2>&1 && echo "restore: bundle verifies" || { echo "restore: bundle verify FAILED (a bundle of ledger/wp10b needs the wp7r history it contains)"; FAIL=1; }
mkdir "$T/base" && git archive $BASE | tar x -C "$T/base" && ( cd "$T/base" && git init -q . 2>/dev/null; git apply --check "$OUT/wp10b_item$N.patch" ) \
  && echo "restore: patch applies to a clean export of $BASE" || { echo "restore: patch FAILED to apply"; FAIL=1; }
( cd "$T/base" && git apply "$OUT/wp10b_item$N.patch" 2>/dev/null && rm -rf .git && diff -r -q . "$T/tree/chronis_hssm_wp10b" >/dev/null && echo "restore: base + patch == tarball tree" ) || { echo "restore: base+patch differs from tarball tree"; FAIL=1; }
( cd "$T/tree/chronis_hssm_wp10b" && PYTHONPATH=. timeout 100 ${PY:-/home/claude/work/venv312/bin/python} scripts/hssm_contract_smoke.py 2>&1 | grep -E "^(PASS|FAIL)" | sort | uniq -c ) 
( cd "$T/tree/chronis_hssm_wp10b" && PYTHONPATH=. timeout 100 ${PY:-/home/claude/work/venv312/bin/python} scripts/hssm_contract_smoke.py >/dev/null 2>&1 ) && echo "restore: smoke script exit 0" || { echo "restore: smoke script FAILED"; FAIL=1; }
[ $FAIL = 0 ] && echo "RESTORE TEST PASSED (item $N, head $HEAD)" || echo "RESTORE TEST FAILED (item $N)"
rm -rf "$T"

# keep the restore-test log (WP10B brief rule 1)
sleep 1; cp /tmp/wp10b_export_log.$$ "$OUT/restore_log_item$N.txt"; ( cd "$OUT" && sha256sum restore_log_item$N.txt > RESTORE_LOG.sha256 )
