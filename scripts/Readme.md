## Script layout

- `data_preprocess/`: parquet ➜ jsonl conversion and cleaning (`generate_toucan.py`, `clean_toucan.py`)
- `analysis/`: readability tools & statistics (`pretty_toucan.py`, `function_stats.py`)
- `build_has/`: HAS generation + orchestration (`build_has_api.py`, `batch_generate.py`)

```
# generate jsonl file
python scripts/data_preprocess/generate_toucan.py -i Toucan-1.5M/Toucan-1.5M/Kimi-K2/train-00000-of-00040.parquet --sample-size 10 -o data/toucan.jsonl
# generate all jsonl in dir
python scripts/data_preprocess/generate_toucan.py -i Toucan-1.5M/Toucan-1.5M
# make jsonl file readable
python scripts/analysis/pretty_toucan.py -i data/toucan.jsonl -n 1 > data/toucan1.txt

# summarize function usage
python scripts/analysis/function_stats.py -i Toucan-1.5M/Toucan-1.5M -o stats/function_stats.csv --meta-output stats/function_stats.json --workers 32
# generate api options (random/available/params/param_values)
python scripts/build_has/build_has_api.py -i data/toucan_1000.jsonl -s stats/function_stats.json -o data/has_api_random.jsonl --mode param_values --negatives 5 --max-samples 200
# batch generate options
python scripts/build_has/batch_generate.py -i Toucan-1.5M/Toucan-1.5M -o data/Toucan-1.5M-generate -s stats/function_stats.json --workers 8
```