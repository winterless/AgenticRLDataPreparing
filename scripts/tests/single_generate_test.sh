#!/bin/bash
set -euo pipefail

python scripts/data_preprocess/generate_toucan.py \
-i Toucan-1.5M/Toucan-1.5M/Kimi-K2/train-00000-of-00040.parquet \
--sample-size 1 \
--seed 23 \
-o data/demo/toucan_raw.jsonl

python scripts/analysis/pretty_toucan.py \
-i data/demo/toucan_raw.jsonl \
-n 1 > data/demo/toucan.txt

python scripts/data_preprocess/obfuscate_jsonl.py \
-i data/demo/toucan_raw.jsonl \
-o data/demo/toucan.jsonl \
--alias stats/function_alias.json

python scripts/build_has/build_has_api_script.py \
-i data/demo/toucan.jsonl \
-s stats/function_stats.json \
-o data/demo/toucan_api_available.jsonl \
--mode available \
--negatives 12

python scripts/build_has/build_has_api_script.py \
-i data/demo/toucan.jsonl \
-s stats/function_stats.json \
-o data/demo/toucan_api_params.jsonl \
--mode params \
--negatives 5

python scripts/build_has/build_has_api_script.py \
-i data/demo/toucan.jsonl \
-s stats/function_stats.json \
-o data/demo/toucan_api_param_values.jsonl \
--mode param_values \
--negatives 5 \
--param-pool stats/param_pool.json

python scripts/data_postprocess/assemble_toucan.py \
-i data/demo \
-m data/demo
