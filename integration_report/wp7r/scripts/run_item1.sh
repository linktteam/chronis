#!/bin/bash
# resumable: seeds already in the jsonl are skipped
cd /home/claude/work/hssm
export PYTHONPATH=.
python3 -u integration_report/wp7r/scripts/item1_iter_study.py --group N18 --seeds 202,404,505,606,707,808,909,1010 --out wp7r_evidence/item1_iter.jsonl --workers 2
python3 -u integration_report/wp7r/scripts/item1_iter_study.py --group DPOP --seeds 5201:5221 --out wp7r_evidence/item1_iter.jsonl --workers 2
