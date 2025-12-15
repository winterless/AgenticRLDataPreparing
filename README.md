## 快速开始（推荐）

- **同步测试脚本（以 README 为真源）**：

```bash
python scripts/tests/generate_test_scripts.py
```

- **跑全量（不依赖 LLM prompt）**：

```bash
./scripts/tests/full_generate_test.sh
```

- **跑单文件 Demo（不依赖 LLM prompt）**：

```bash
./scripts/tests/single_generate_test.sh
```

- **覆盖默认变量**（任意脚本前加环境变量即可）：
  - 示例：`WORKERS=64 RAW_ROOT=/path/to/data ./scripts/tests/full_generate_test.sh`

---

## 流程概览（原始轨迹 ➜ 混淆 ➜ MCQ ➜ 组装）

- **生成/准备轨迹**：parquet → jsonl（`generate_toucan.py`）
- **统计与 alias**：抽取函数签名/元数据、生成 alias（`function_stats.py --alias-output`）
- **混淆**：用 alias 重写工具/函数名（`obfuscate_jsonl.py`）
- **基于混淆数据重建统计 + 参数池**（`function_stats.py` + `build_param_pool.py`）
- **生成 MCQ**（脚本模式：`build_has_api_script.py`；全量批处理：`batch_generate.py`）
- **组装对话+MCQ**（`assemble_toucan.py`）

> `data/demo/` 内是最小可跑样例：所有 “single” 默认输出到 `${DEMO_DIR}`。

---

## 命令清单（README 为真源，自动生成 `scripts/tests/*_generate_test.sh`）

下面这段 `bash` 代码块会被 `scripts/tests/generate_test_scripts.py` 解析：
- `# vars`：会写入生成的 `.sh` 顶部（默认值可被环境变量覆盖）
- `# full` / `# single`：各自进入对应脚本
- `# online`：会被跳过（只做文档示例）
- `# test`：只做说明，不进入脚本

<details>
<summary>展开：用于自动生成测试脚本的命令（# vars / # full / # single）</summary>

```bash
# vars
WORKERS="${WORKERS:-32}"

# full pipeline roots
RAW_ROOT="${RAW_ROOT:-Toucan-1.5M/Toucan-1.5M}"
OBF_ROOT="${OBF_ROOT:-data/Toucan-1.5M-obf}"
GENERATE_ROOT="${GENERATE_ROOT:-data/Toucan-1.5M-generate}"

# stats outputs
STATS_DIR="${STATS_DIR:-stats}"
STATS_RAW_CSV="${STATS_RAW_CSV:-${STATS_DIR}/function_stats_raw.csv}"
STATS_RAW_JSON="${STATS_RAW_JSON:-${STATS_DIR}/function_stats_raw.json}"
ALIAS_JSON="${ALIAS_JSON:-${STATS_DIR}/function_alias.json}"
STATS_CSV="${STATS_CSV:-${STATS_DIR}/function_stats.csv}"
STATS_JSON="${STATS_JSON:-${STATS_DIR}/function_stats.json}"
PARAM_POOL="${PARAM_POOL:-${STATS_DIR}/param_pool.json}"

# demo (single) paths
DEMO_DIR="${DEMO_DIR:-data/demo}"
DEMO_PARQUET="${DEMO_PARQUET:-Toucan-1.5M/Toucan-1.5M/Kimi-K2/train-00000-of-00040.parquet}"
DEMO_RAW_JSONL="${DEMO_RAW_JSONL:-${DEMO_DIR}/toucan_raw.jsonl}"
DEMO_PRETTY_TXT="${DEMO_PRETTY_TXT:-${DEMO_DIR}/toucan.txt}"
DEMO_JSONL="${DEMO_JSONL:-${DEMO_DIR}/toucan.jsonl}"
DEMO_API_AVAILABLE="${DEMO_API_AVAILABLE:-${DEMO_DIR}/toucan_api_available.jsonl}"
DEMO_API_PARAMS="${DEMO_API_PARAMS:-${DEMO_DIR}/toucan_api_params.jsonl}"
DEMO_API_PARAM_VALUES="${DEMO_API_PARAM_VALUES:-${DEMO_DIR}/toucan_api_param_values.jsonl}"

# ---------- full: end-to-end (no prompt) ----------
# full
python scripts/data_preprocess/generate_toucan.py \
  -i "${RAW_ROOT}" --drop-non-utf8 --workers "${WORKERS}"

# full
python scripts/analysis/function_stats.py \
  -i "${RAW_ROOT}" \
  -o "${STATS_RAW_CSV}" \
  --meta-output "${STATS_RAW_JSON}" \
  --alias-output "${ALIAS_JSON}" \
  --workers "${WORKERS}"

# full
python scripts/data_preprocess/obfuscate_jsonl.py \
  -i "${RAW_ROOT}" \
  -o "${OBF_ROOT}" \
  --alias "${ALIAS_JSON}" \
  --workers "${WORKERS}"

# full
python scripts/analysis/function_stats.py \
  -i "${OBF_ROOT}" \
  -o "${STATS_CSV}" \
  --meta-output "${STATS_JSON}" \
  --workers "${WORKERS}"

# full
python scripts/data_preprocess/build_param_pool.py \
  -i "${OBF_ROOT}" \
  -s "${STATS_JSON}" \
  -o "${PARAM_POOL}" \
  --workers "${WORKERS}"

# full
python scripts/build_has/batch_generate.py \
  -i "${OBF_ROOT}" \
  -o "${GENERATE_ROOT}" \
  -s "${STATS_JSON}" \
  --workers "${WORKERS}" \
  --param-pool "${PARAM_POOL}"

# full
python scripts/data_postprocess/assemble_toucan.py \
  -i "${OBF_ROOT}" \
  -m "${GENERATE_ROOT}" \
  --workers "${WORKERS}"

# ---------- single: demo (no prompt) ----------
# single
python scripts/data_preprocess/generate_toucan.py \
  -i "${DEMO_PARQUET}" \
  --sample-size 1 \
  --seed 23 \
  --drop-non-utf8 \
  -o "${DEMO_RAW_JSONL}"

# single
python scripts/analysis/pretty_toucan.py \
  -i "${DEMO_RAW_JSONL}" \
  -n 1 > "${DEMO_PRETTY_TXT}"

# single
python scripts/data_preprocess/obfuscate_jsonl.py \
  -i "${DEMO_RAW_JSONL}" \
  -o "${DEMO_JSONL}" \
  --alias "${ALIAS_JSON}"

# single
python scripts/build_has/build_has_api_script.py \
  -i "${DEMO_JSONL}" \
  -s "${STATS_JSON}" \
  -o "${DEMO_API_AVAILABLE}" \
  --mode available \
  --negatives 12

# single
python scripts/build_has/build_has_api_script.py \
  -i "${DEMO_JSONL}" \
  -s "${STATS_JSON}" \
  -o "${DEMO_API_PARAMS}" \
  --mode params \
  --negatives 5

# single
python scripts/build_has/build_has_api_script.py \
  -i "${DEMO_JSONL}" \
  -s "${STATS_JSON}" \
  -o "${DEMO_API_PARAM_VALUES}" \
  --mode param_values \
  --negatives 5 \
  --param-pool "${PARAM_POOL}"

# single
python scripts/data_postprocess/assemble_toucan.py \
  -i "${DEMO_DIR}" \
  -m "${DEMO_DIR}"

# ---------- optional / docs-only ----------
# test
python scripts/data_postprocess/assemble_toucan.py \
  -i "${OBF_ROOT}" \
  -m "${GENERATE_ROOT}" \
  --workers "${WORKERS}" \
  --passthrough-only

# online
python scripts/build_has/batch_generate.py \
  -i "${OBF_ROOT}" \
  -o data/has_prompt_batch \
  -s "${STATS_JSON}" \
  --prompt-mode \
  --prompt-limit 10 \
  --prompt-temperature 0.4 \
  --prompt-max-tokens 512

# online
python scripts/build_has/build_has_api_prompt.py \
  -i "${DEMO_JSONL}" \
  -s "${STATS_JSON}" \
  -o data/demo/has_prompt_toucan.jsonl \
  --temperature 0.4 \
  --max-tokens 512
```

</details>

---

## 关键脚本与常用参数（速查）

- **`scripts/data_preprocess/generate_toucan.py`**：parquet → jsonl
  - **`-i/--input`**：parquet 文件或目录
  - **`-o/--output`**：输出 jsonl（不指定时按脚本默认路径）
  - **`--workers`**：并行度
  - **`--sample-size` / `--seed`**：抽样（demo 用）
  - **`--drop-non-utf8`**：过滤不可编码内容

- **`scripts/analysis/function_stats.py`**：统计函数/工具元数据，可生成 alias
  - **`-i`**：输入 jsonl 文件或目录
  - **`-o`**：输出 csv
  - **`--meta-output`**：输出 json（schema/meta）
  - **`--alias-output`**：输出 alias map（从 raw 构建时用）
  - **`--workers`**：并行度

- **`scripts/data_preprocess/obfuscate_jsonl.py`**：按 alias 混淆 jsonl
  - **`-i` / `-o`**：输入/输出（文件或目录）
  - **`--alias`**：alias map 路径
  - **`--workers`**：并行度（目录模式）

- **`scripts/data_preprocess/build_param_pool.py`**：构建参数值池（供 param_values 负样本）
  - **`-i`**：混淆后的 jsonl（文件或目录）
  - **`-s`**：`function_stats.json`
  - **`-o`**：`param_pool.json`
  - **`--workers`**：并行度

- **`scripts/build_has/build_has_api_script.py`**：生成单文件 MCQ（demo）
  - **`--mode`**：`available | params | param_values`
  - **`--negatives`**：负例数量
  - **`--param-pool`**：`param_values` 模式需要

- **`scripts/build_has/batch_generate.py`**：全量批处理生成 MCQ（无 prompt 或 prompt）
  - **`--workers`**：并行度（脚本模式）
  - **`--param-pool`**：参数池
  - **`--prompt-mode`**：启用 prompt 生成（串行更适合小规模）
  - **`--prompt-limit` / `--prompt-temperature` / `--prompt-max-tokens`**：prompt 控制

- **`scripts/data_postprocess/assemble_toucan.py`**：将轨迹 + MCQ 拼装为训练格式
  - **`-i/--conv-root`**：对话根目录
  - **`-m/--mcq-root`**：MCQ 根目录
  - **`--workers`**：并行度
  - **`--passthrough-only`**：不注入 MCQ，仅做 UTF-8 严格写出（消融/对比）
  - **`--no-text-output`**：只输出 jsonl，不写 txt
  - **`--show-function-name`**：MCQ 题头展示函数名（默认隐藏）




---

## HAS-API 题型策略（脚本模式细节）

生成逻辑集中在 `scripts/build_has/build_has_api_script.py`：

1. **available 模式**
   - 仅在 `assistant` 且包含 `function_call` 的消息上出题。
   - 先使用 `available_tools` 中的真实候选，再补充同 family / 语义相近 alias 作为干扰项。
   - 输出格式为 `alias_name`，不含描述，防止模型通过原名记忆。

2. **params 模式**
   - 根据 `function_stats.json` 中的 JSON Schema 解析必填字段。
   - 正确选项是所有 required 参数集合；干扰项包括缺失必填、加入额外 optional、或单字段组合。

3. **param_values 模式**
   - 真值来自实际 `function_call.arguments`（字符串先 `json.loads`）。
   - 负例来源：
     - `ParamPool`（函数/参数/类型三级聚类）抽取真实历史值。
     - `_mutate_with_pool` 支持一次扰动 1~2 个字段。
     - `_drop_argument` 针对 required / 任意字段生成缺失参数的负例。
   - `ParamPool.sample()` 优先从相同函数/参数的历史值采样，并以较小概率注入跨函数干扰项。

```text
参数值生成思路
1. 解析真实参数：correct_option = _format_arg_values(_parse_arguments(fc))
2. 扰动策略（按类型推断）：
   - bool：取反
   - int/float：±1/2/5
   - enum string：挑其他枚举值；否则附加少量后缀
   - 其余：若无法生成则跳过
3. ParamPool：按 (函数, 参数) → 单参数 → 类型 三层聚类，并记录去重值
4. 组装题目：variations[:num_neg] + correct_option，随机打乱
```

以上策略保证：
- 负样本具备语义相关性与多样性，避免模型靠“公共特征”偷懒。
- 所有 jsonl 输出仅包含 alias 名称；`pretty_toucan.py --alias-map` 可在人工检查时恢复原名。

---

## 数据拼装（轨迹 + MCQ）

`scripts/data_postprocess/assemble_toucan.py` 会扫描 `-i/--conv-root` 目录下的对话 jsonl，并在 `-m/--mcq-root` 里查找同前缀的 `_api_{available,params,param_values}.jsonl`。一旦发现完整组合，就会写出 `<prefix>_mcq_assembled.jsonl` + `<prefix>_mcq_assembled.txt`（路径位于 MCQ 目录）。`data/demo/` 已内置 `toucan`

> JSONL/TXT 始终包含正确答案（便于模型学习与人工校验，`Answer: the answer is ...`）。如不需要文本副本可使用 `--no-text-output`。

运行方式：
- **推荐**：直接运行 README 顶部的 `./scripts/tests/full_generate_test.sh` / `./scripts/tests/single_generate_test.sh`
- **消融/过滤**（不注入 MCQ）：运行 `assemble_toucan.py --passthrough-only`（见上方“命令清单”的 optional 区块）

<details>
<summary>展开：拼装规则（详细）</summary>

1. 开头写 `Question:`、`工具清单：`，需要注入这个轨迹对应MCP文件中的假工具，工具乱序。
2. `System tool declare:` 来自原始 `messages` 中 system/tool_declare，需要注入这个轨迹对应MCP文件中的假工具，工具乱序。
3. 展开 user/assistant/function 消息；在 assistant 准备 `function_call` 时，插入与该 `message_index` 对应的 MCQ（顺序：available → params → param_values）。
注意是每个assistant的function_call按照function，params，parma_values的顺序插入mcq
4. MCQ 题头格式：

```
[MCQ:param_values|function=func_xxx|msg=4]
问：……
选项：A.… B.… C.… D.…
```

5. MCQ 区块后接 function_call 参数与 function 响应，再附 `[[原文回答]]`。
6. 轨迹结束后补 `Target tools:`、`Question quality assessment:`、`Response quality assessment:`、`Metadata:`。

该流程确保模型既能看到完整对话，又能学习多项选择题，不暴露答案。更多细节、字段含义及 concat 规划请参考 `scripts/ARCHITECTURE.md`。
</details>

P.S 清洗数据脚本
```
# test
python scripts/data_preprocess/clean_utf8_dir.py -i /path/to/src_dir -o /path/to/dst_dir --workers 8
```

## 测试
- **测试脚本生成（自动同步 README 标签）**
  ```
  # generate tests based on README.md python commands
  # test
  python scripts/tests/generate_test_scripts.py
  ```

- **全量数据构建（不依赖 LLM prompt）**
  ```
  ./scripts/tests/full_generate_test.sh
  ```

- **单文件 Demo 构建（不依赖 LLM prompt）**
  ```
  ./scripts/tests/single_generate_test.sh
  ```