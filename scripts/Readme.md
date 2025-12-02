```
# generate jsonl file
python scripts/generate_toucan.py -i Toucan-1.5M/Toucan-1.5M/Kimi-K2/train-00000-of-00040.parquet --sample-size 10 -o data/toucan.jsonl
# generate all jsonl in dir
python scripts/generate_toucan.py -i Toucan-1.5M/Toucan-1.5M
# make jsonl file readable
python scripts/pretty_toucan.py -i data/toucan.jsonl -n 1 > data/toucan1.txt

# summarize funciton
python scripts/function_stats.py -i Toucan-1.5M/Toucan-1.5M -o stats/function_stats.csv --meta-output function_stats.json --workers 32
# generate api options
python scripts/build_has_api.py -i data/toucan_1000.jsonl -s stats/function_meta.json -o data/has_api_random.jsonl --mode random --negatives 5 --max-samples 200
# batch generate options
python scripts/batch_generate.py -i Toucan-1.5M/Toucan-1.5M -o data/Toucan-1.5M-generate -s stats/function_stats.json --workers 8
```