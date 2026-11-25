#!/bin/bash
# WP8 export + restore test. usage: bash integration_report/wp8/scripts/export_item.sh N   (run from the repo root, on branch ledger/wp8, everything committed)
# writes /mnt/user-data/outputs/wp8_itemN/{wp8_itemN.patch, wp8_tree.tar.gz, wp8.bundle, wp8_evidence_itemN.zip, SHA256SUMS.txt} and runs the restore test.
set -u
N="${1:?item number}"; BASE=27c591c
OUT=/mnt/user-data/outputs/wp8_item$N; mkdir -p "$OUT"; rm -f "$OUT"/*
[ -z "$(git status --porcelain)" ] || { echo "REFUSED: uncommitted changes"; git status --short | head -5; exit 2; }
HEAD=$(git rev-parse --short HEAD)
git diff --binary $BASE..HEAD > "$OUT/wp8_item$N.patch"
git archive --format=tar.gz --prefix=chronis_hssm_wp8/ HEAD > "$OUT/wp8_tree.tar.gz"
git bundle create "$OUT/wp8.bundle" ledger/wp7r ledger/wp8 2>&1 | tail -n 1
( cd wp8_evidence && zip -q -r "$OUT/wp8_evidence_item$N.zip" . -x '*.pyc' ) || echo "zip failed"
( cd "$OUT" && sha256sum wp8_item$N.patch wp8_tree.tar.gz wp8.bundle wp8_evidence_item$N.zip > SHA256SUMS.txt )
echo "exported head $HEAD:"; ls -l "$OUT" | tail -n +2
# ---- restore test (fresh folder; never the working copy) ----
T=$(mktemp -d /tmp/wp8_restore.XXXXXX); FAIL=0
( cd "$OUT" && sha256sum -c SHA256SUMS.txt >/dev/null 2>&1 ) && echo "restore: SHA256 ok" || { echo "restore: SHA256 FAILED"; FAIL=1; }
mkdir "$T/tree" && tar xzf "$OUT/wp8_tree.tar.gz" -C "$T/tree"
git bundle verify "$OUT/wp8.bundle" >/dev/null 2>&1 && echo "restore: bundle verifies" || { echo "restore: bundle verify FAILED (a bundle of ledger/wp8 needs the wp7r history it contains)"; FAIL=1; }
mkdir "$T/base" && git archive $BASE | tar x -C "$T/base" && ( cd "$T/base" && git init -q . 2>/dev/null; git apply --check "$OUT/wp8_item$N.patch" ) \
  && echo "restore: patch applies to a clean export of $BASE" || { echo "restore: patch FAILED to apply"; FAIL=1; }
( cd "$T/base" && git apply "$OUT/wp8_item$N.patch" 2>/dev/null && rm -rf .git && diff -r -q . "$T/tree/chronis_hssm_wp8" >/dev/null && echo "restore: base + patch == tarball tree" ) || { echo "restore: base+patch differs from tarball tree"; FAIL=1; }
( cd "$T/tree/chronis_hssm_wp8" && PYTHONPATH=. timeout 100 python3 scripts/hssm_contract_smoke.py 2>&1 | grep -E "^(PASS|FAIL)" | sort | uniq -c ) 
( cd "$T/tree/chronis_hssm_wp8" && PYTHONPATH=. timeout 100 python3 scripts/hssm_contract_smoke.py >/dev/null 2>&1 ) && echo "restore: smoke script exit 0" || { echo "restore: smoke script FAILED"; FAIL=1; }
[ $FAIL = 0 ] && echo "RESTORE TEST PASSED (item $N, head $HEAD)" || echo "RESTORE TEST FAILED (item $N)"
rm -rf "$T"
