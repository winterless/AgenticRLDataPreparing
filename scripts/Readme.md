```
# generate jsonl file
python generate_toucan.py -i Toucan-1.5M/Toucan-1.5M/Kimi-K2/train-00000-of-00040.parquet -l 10 -o toucan.jsonl
# make jsonl file readable
python pretty_toucan.py -i toucan.jsonl -n 5 > toucan5.txt
```