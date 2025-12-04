toucan_name="toucan2"
raw_dir="Toucan-1.5M/Toucan-1.5M"
obf_dir="data/Toucan-1.5M-obf"

# 1) 采样一个 demo jsonl（仅用于快速查看）
python scripts/data_preprocess/generate_toucan.py -i ${raw_dir}/Kimi-K2/train-00000-of-00040.parquet --sample-size 1 --seed 23 -o data/${toucan_name}.jsonl

# 2) 使用全量 Toucan-1.5M 构建统计 + alias map（function_stats_raw.* 仅作为中间产物）
python scripts/analysis/function_stats.py -i ${raw_dir} -o stats/function_stats_raw.csv --meta-output stats/function_stats_raw.json --alias-output stats/function_alias.json --workers 32

# 3) 将 demo jsonl 以及全量 Toucan-1.5M 都转换成混淆版
python scripts/data_preprocess/obfuscate_jsonl.py -i data/${toucan_name}.jsonl -o data/${toucan_name}_obf.jsonl --alias stats/function_alias.json
python scripts/data_preprocess/obfuscate_jsonl.py -i "${raw_dir}" -o "${obf_dir}" --alias stats/function_alias.json --workers 32

# 4) 基于混淆后的 Toucan-1.5M 重新统计 function_stats / param_pool
python scripts/analysis/function_stats.py -i "${obf_dir}" -o stats/function_stats.csv --meta-output stats/function_stats.json --workers 32
python scripts/data_preprocess/build_param_pool.py -i "${obf_dir}" -s stats/function_stats.json -o stats/param_pool.json --workers 32

# 5) 输出可读文本（同时展示 alias + 原名，便于人工检查）
python scripts/analysis/pretty_toucan.py -i data/${toucan_name}_obf.jsonl -n 1 --alias-map stats/function_alias.json > data/${toucan_name}.txt

# 6) 构造 HAS 题库（直接使用混淆后的 demo jsonl）
python scripts/build_has/build_has_api_script.py -i data/${toucan_name}_obf.jsonl -s stats/function_stats.json -o data/${toucan_name}_api_available.jsonl --mode available --negatives 12
python scripts/build_has/build_has_api_script.py -i data/${toucan_name}_obf.jsonl -s stats/function_stats.json -o data/${toucan_name}_api_params.jsonl --mode params --negatives 5
python scripts/build_has/build_has_api_script.py -i data/${toucan_name}_obf.jsonl -s stats/function_stats.json -o data/${toucan_name}_api_param_values.jsonl --mode param_values --negatives 5 --param-pool stats/param_pool.json