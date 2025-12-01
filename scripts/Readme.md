```
# generate jsonl file
python scripts/generate_toucan.py -i Toucan-1.5M/Toucan-1.5M/Kimi-K2/train-00000-of-00040.parquet --sample-size 10 -o data/toucan.jsonl
# generate all jsonl in dir
python scripts/generate_toucan.py -i Toucan-1.5M/Toucan-1.5M
# make jsonl file readable
python scripts/pretty_toucan.py -i data/toucan.jsonl -n 5 > data/toucan5.txt

# summarize funciton
python scripts/function_stats.py -i Toucan-1.5M/Toucan-1.5M -o stats/function_stats.csv --workers 8 --top 200
```