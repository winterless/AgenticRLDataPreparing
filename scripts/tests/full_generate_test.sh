#!/bin/bash
set -euo pipefail

python scripts/data_preprocess/generate_toucan.py \
-i Toucan-1.5M/Toucan-1.5M --workers 32

python scripts/analysis/function_stats.py \
-i Toucan-1.5M/Toucan-1.5M \
-o stats/function_stats_raw.csv \
--meta-output stats/function_stats_raw.json \
--alias-output stats/function_alias.json \
--workers 32

python scripts/data_preprocess/obfuscate_jsonl.py \
-i Toucan-1.5M/Toucan-1.5M \
-o data/Toucan-1.5M-obf \
--alias stats/function_alias.json \
--workers 32

python scripts/analysis/function_stats.py \
-i data/Toucan-1.5M-obf \
-o stats/function_stats.csv \
--meta-output stats/function_stats.json \
--workers 32

python scripts/data_preprocess/build_param_pool.py \
-i data/Toucan-1.5M-obf \
-s stats/function_stats.json \
-o stats/param_pool.json \
--workers 32

python scripts/build_has/batch_generate.py \
-i Toucan-1.5M/Toucan-1.5M \
-o data/Toucan-1.5M-generate \
-s stats/function_stats.json \
--workers 32 \
--param-pool stats/param_pool.json

python scripts/data_postprocess/assemble_toucan.py -i data/Toucan-1.5M-obf -m data/Toucan-1.5M-generate --workers 16