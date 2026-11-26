#!/bin/bash
# usage: bash integration_report/wp10a/scripts/export_retry.sh N : commits an interim snapshot of the running evidence files (explicit paths, only when modified) and then exports item N
N="${1:?item number}"
for f in wp10a_evidence/a3_dev_fits.jsonl wp10a_evidence/a3_eval_fits.jsonl wp10a_evidence/a3_safe_fits.jsonl; do
  [ -f "$f" ] && ! git diff --quiet -- "$f" 2>/dev/null && git add "$f" || true
  [ -f "$f" ] && [ -n "$(git ls-files --others --exclude-standard -- "$f")" ] && git add "$f" || true
done
if ! git diff --cached --quiet; then git commit -q -m "WP10A: interim snapshot of the running A3 fits (not final; the final files replace it)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"; fi
bash integration_report/wp10a/scripts/export_item.sh "$N"
