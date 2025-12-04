toucan_name="toucan"
raw_dir="Toucan-1.5M/Toucan-1.5M"
obf_dir="data/Toucan-1.5M-obf"
regenerate=false

sample_raw="data/${toucan_name}_raw.jsonl"
sample_obf="data/${toucan_name}.jsonl"
alias_map="stats/function_alias.json"
raw_stats_csv="stats/function_stats_raw.csv"
raw_stats_json="stats/function_stats_raw.json"
obf_stats_csv="stats/function_stats.csv"
obf_stats_json="stats/function_stats.json"
param_pool="stats/param_pool.json"

# 1) 采样一个 demo jsonl（仅用于快速查看）
python scripts/data_preprocess/generate_toucan.py \
  -i "${raw_dir}/Kimi-K2/train-00000-of-00040.parquet" \
  --sample-size 1 \
  --seed 23 \
  -o "${sample_raw}"

# 2) （按需）使用全量 Toucan-1.5M 构建统计 + alias map（function_stats_raw.* 为中间产物）
if [[ "${regenerate}" == true || ! -f "${alias_map}" || ! -f "${raw_stats_csv}" || ! -f "${raw_stats_json}" ]]; then
  python scripts/analysis/function_stats.py \
    -i "${raw_dir}" \
    -o "${raw_stats_csv}" \
    --meta-output "${raw_stats_json}" \
    --alias-output "${alias_map}" \
    --workers 32
else
  echo "[SKIP] reuse raw stats + alias map (${alias_map})"
fi

# 3) 将 demo jsonl 以及全量 Toucan-1.5M 转换成混淆版
python scripts/data_preprocess/obfuscate_jsonl.py \
  -i "${sample_raw}" \
  -o "${sample_obf}" \
  --alias "${alias_map}"

if [[ "${regenerate}" == true || ! -d "${obf_dir}" ]]; then
  python scripts/data_preprocess/obfuscate_jsonl.py \
    -i "${raw_dir}" \
    -o "${obf_dir}" \
    --alias "${alias_map}" \
    --workers 32
else
  echo "[SKIP] reuse obfuscated directory ${obf_dir}"
fi

# 4) （按需）基于混淆后的 Toucan-1.5M 重新统计 function_stats / param_pool
if [[ "${regenerate}" == true || ! -f "${obf_stats_json}" || ! -f "${param_pool}" ]]; then
  python scripts/analysis/function_stats.py \
    -i "${obf_dir}" \
    -o "${obf_stats_csv}" \
    --meta-output "${obf_stats_json}" \
    --workers 32
  python scripts/data_preprocess/build_param_pool.py \
    -i "${obf_dir}" \
    -s "${obf_stats_json}" \
    -o "${param_pool}" \
    --workers 32
else
  echo "[SKIP] reuse stats + param_pool (${obf_stats_json}, ${param_pool})"
fi

# 5) 输出可读文本（同时展示 alias + 原名，便于人工检查）
python scripts/analysis/pretty_toucan.py \
  -i "${sample_obf}" \
  -n 1 \
  --alias-map "${alias_map}" \
  > "data/${toucan_name}.txt"

# 6) 构造 HAS 题库（直接使用混淆后的 demo jsonl）
python scripts/build_has/build_has_api_script.py \
  -i "${sample_obf}" \
  -s "${obf_stats_json}" \
  -o "data/${toucan_name}_api_available.jsonl" \
  --mode available \
  --negatives 12

python scripts/build_has/build_has_api_script.py \
  -i "${sample_obf}" \
  -s "${obf_stats_json}" \
  -o "data/${toucan_name}_api_params.jsonl" \
  --mode params \
  --negatives 5

python scripts/build_has/build_has_api_script.py \
  -i "${sample_obf}" \
  -s "${obf_stats_json}" \
  -o "data/${toucan_name}_api_param_values.jsonl" \
  --mode param_values \
  --negatives 5 \
  --param-pool "${param_pool}"