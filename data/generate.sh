toucan_name="toucan2"
python scripts/data_preprocess/generate_toucan.py -i Toucan-1.5M/Toucan-1.5M/Kimi-K2/train-00000-of-00040.parquet --sample-size 1 --seed 23 -o data/${toucan_name}.jsonl
python scripts/analysis/pretty_toucan.py -i data/${toucan_name}.jsonl -n 1 > data/${toucan_name}.txt
python scripts/build_has/build_has_api_script.py -i data/${toucan_name}.jsonl -s stats/function_stats.json -o data/${toucan_name}_api_available.jsonl --mode available --negatives 12
python scripts/build_has/build_has_api_script.py -i data/${toucan_name}.jsonl -s stats/function_stats.json -o data/${toucan_name}_api_params.jsonl --mode params --negatives 5
python scripts/build_has/build_has_api_script.py -i data/${toucan_name}.jsonl -s stats/function_stats.json -o data/${toucan_name}_api_param_values.jsonl --mode param_values --negatives 5 --param-pool stats/param_pool.json