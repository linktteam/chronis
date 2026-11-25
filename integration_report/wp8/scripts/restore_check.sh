#!/bin/bash
# WP7-R: re-run the restore test on an ALREADY EXPORTED folder (read-only on it). usage: bash restore_check.sh N   (run from the repo root; N = the export number)
set -u
N="${1:?item number}"; BASE=27c591c; OUT=/mnt/user-data/outputs/wp8_item$N; FAIL=0
T=$(mktemp -d /tmp/wp8_restore_chk.XXXXXX)
( cd "$OUT" && sha256sum -c SHA256SUMS.txt >/dev/null 2>&1 ) && echo "item $N restore: SHA256 ok" || { echo "item $N restore: SHA256 FAILED"; FAIL=1; }
git bundle verify "$OUT/wp8.bundle" >/dev/null 2>&1 && echo "item $N restore: bundle verifies" || { echo "item $N restore: bundle verify FAILED"; FAIL=1; }
mkdir "$T/tree" && tar xzf "$OUT/wp8_tree.tar.gz" -C "$T/tree"
mkdir "$T/base" && git archive $BASE | tar x -C "$T/base"
( cd "$T/base" && git init -q . 2>/dev/null; git apply --check "$OUT/wp8_item$N.patch" ) && echo "item $N restore: patch applies to a clean $BASE" || { echo "item $N restore: patch FAILED"; FAIL=1; }
( cd "$T/base" && git apply "$OUT/wp8_item$N.patch" 2>/dev/null && rm -rf .git && diff -r -q . "$T/tree/chronis_hssm_wp8" >/dev/null ) && echo "item $N restore: base + patch == tarball tree" || { echo "item $N restore: base+patch differs from tarball tree"; FAIL=1; }
( cd "$T/tree/chronis_hssm_wp8" && PYTHONPATH=. timeout 100 python3 scripts/hssm_contract_smoke.py >/dev/null 2>&1 ) && echo "item $N restore: smoke script exit 0" || { echo "item $N restore: smoke script FAILED"; FAIL=1; }
[ $FAIL = 0 ] && echo "item $N RESTORE TEST PASSED" || echo "item $N RESTORE TEST FAILED"
rm -rf "$T"
