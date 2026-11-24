#!/bin/bash
cd /home/claude/work/hssm; export PYTHONPATH=.
python3 -u integration_report/wp7r/scripts/item1_iter_study.py --group N18 --seeds 202 --out wp7r_evidence/item1_s3_dmax60.jsonl --workers 1 --dmax 60
python3 -u integration_report/wp7r/scripts/item1_iter_study.py --group DPOP --seeds 5203,5204,5207,5208,5211,5212,5213,5215 --out wp7r_evidence/item1_s3_dmax60.jsonl --workers 2 --dmax 60
