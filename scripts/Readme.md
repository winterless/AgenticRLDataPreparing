## Script layout

- `data_preprocess/`: parquet ➜ jsonl conversion / cleaning / parameter池构建（`generate_toucan.py`, `clean_toucan.py`, `build_param_pool.py`）
- `analysis/`: readability tools & statistics (`pretty_toucan.py`, `function_stats.py`)
- `build_has/`: HAS generation + orchestration (`build_has_api_script.py`, `batch_generate.py`)

```
# generate jsonl file
python scripts/data_preprocess/generate_toucan.py -i Toucan-1.5M/Toucan-1.5M/Kimi-K2/train-00000-of-00040.parquet --sample-size 10 -o data/toucan.jsonl
# generate all jsonl in dir
python scripts/data_preprocess/generate_toucan.py -i Toucan-1.5M/Toucan-1.5M
# make jsonl file readable
python scripts/analysis/pretty_toucan.py -i data/toucan.jsonl -n 1 > data/toucan1.txt

# summarize function usage
python scripts/analysis/function_stats.py -i Toucan-1.5M/Toucan-1.5M -o stats/function_stats.csv --meta-output stats/function_stats.json --workers 32
# build parameter pool for param_values mode
python scripts/data_preprocess/build_param_pool.py -i Toucan-1.5M/Toucan-1.5M -s stats/function_stats.json -o stats/param_pool.json --workers 16
# generate api options (available/params/param_values)
python scripts/build_has/build_has_api_script.py -i data/toucan_1000.jsonl -s stats/function_stats.json -o data/has_api_available.jsonl --mode available --negatives 9 --max-samples 200
python scripts/build_has/build_has_api_script.py -i data/toucan_1000.jsonl -s stats/function_stats.json -o data/has_api_param_values.jsonl --mode param_values --negatives 5 --max-samples 200 --param-pool stats/param_pool.json
# batch generate options (script-based param_values uses param_pool)
python scripts/build_has/batch_generate.py -i Toucan-1.5M/Toucan-1.5M -o data/Toucan-1.5M-generate -s stats/function_stats.json --workers 32 --param-pool stats/param_pool.json
# batch generate prompt-based param_values (串行执行)
python scripts/build_has/batch_generate.py -i Toucan-1.5M/Toucan-1.5M -o data/has_prompt_batch -s stats/function_stats.json --prompt-mode --prompt-limit 10 --prompt-temperature 0.4 --prompt-max-tokens 512
# prompt-based question_param_values (Toucan-driven)
python scripts/build_has/build_has_api_prompt.py -i data/toucan_1000.jsonl -s stats/function_stats.json -o data/has_prompt_toucan.jsonl --limit 200 --temperature 0.4 --max-tokens 512
```