## HAS 数据构造流程（概览）

1. **原始数据准备**
   - 使用 `generate_toucan.py` 抽取 parquet → jsonl：  
     `python scripts/data_preprocess/generate_toucan.py -i Toucan-1.5M/Toucan-1.5M --workers 32`
   - 数据清洗（分类，扩展节点，过滤回话，基于阈值等清洗）：暂未进行

2. **轨迹数据混淆**
   - 对轨迹中的函数名进行混淆，避免模型通过名称推导

3. **增强候选选项生成**
   - 基于函数名选择，函数签名选择，函数参数值选择三项
   - 在原始轨迹中，插入对应的选择题，结尾衔接答案

4. **CPT续训**

5. **候选打分与排序（候选—）**
   - 通过 GPT / Gemini 评分，保留解释，并以 `state + actions + score` 结构存储。


---

## 脚本目录与职责

| 目录 | 作用 | 代表脚本 |
| --- | --- | --- |
| `data_preprocess/` | parquet ➜ jsonl、构建参数池、批量混淆 | `generate_toucan.py`, `obfuscate_jsonl.py`, `build_param_pool.py` |
| `analysis/` | 统计与可视化，构建 alias map、生成可读文本 | `function_stats.py`, `pretty_toucan.py` |
| `build_has/` | HAS-API 题目生成（脚本/Prompt）及批处理 | `build_has_api_script.py`, `build_has_api_prompt.py`, `batch_generate.py` |

> `data/generate.sh` 封装了下述流程，可通过 `regenerate=true` 控制是否重跑全量统计与混淆。

---

## 原始轨迹 ➜ 混淆数据 ➜ HAS-API 题库
(单文件demo示例详见data/generate.sh)

### 1. 采样一份可控的原始 jsonl（首次或需要重建时）

```bash
python scripts/data_preprocess/generate_toucan.py \
 -i Toucan-1.5M/Toucan-1.5M --workers 32
```

### 2. 构建 function_stats + alias map（首次或需要重建时）

```bash
python scripts/analysis/function_stats.py \
  -i Toucan-1.5M/Toucan-1.5M \
  -o stats/function_stats_raw.csv \
  --meta-output stats/function_stats_raw.json \
  --alias-output stats/function_alias.json \
  --workers 32
```

### 3. 使用 alias map 混淆全量数据（首次或需要重建时）

```bash
python scripts/data_preprocess/obfuscate_jsonl.py \
  -i Toucan-1.5M/Toucan-1.5M \
  -o data/Toucan-1.5M-obf \
  --alias stats/function_alias.json \
  --workers 32
```

### 4. 基于混淆数据重建统计与参数池（首次或需要重建时）

```bash
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
```

### 5. 生成 HAS-API 选择题（脚本模式）

```bash
# available 模式（整合 random 逻辑、函数名仅输出 alias）
python scripts/build_has/build_has_api_script.py \
  -i data/toucan.jsonl \
  -s stats/function_stats.json \
  -o data/toucan_api_available.jsonl \
  --mode available \
  --negatives 12

# params 模式（必填参数判断）
python scripts/build_has/build_has_api_script.py \
  -i data/toucan.jsonl \
  -s stats/function_stats.json \
  -o data/toucan_api_params.jsonl \
  --mode params \
  --negatives 5

# param_values 模式（真实参数池 + 干扰项）
python scripts/build_has/build_has_api_script.py \
  -i data/toucan.jsonl \
  -s stats/function_stats.json \
  -o data/toucan_api_param_values.jsonl \
  --mode param_values \
  --negatives 5 \
  --param-pool stats/param_pool.json
```

### 6. 批量生成 / Prompt 生成

```bash
# 批量脚本模式（支持 available/params/param_values 并行生成）
python scripts/build_has/batch_generate.py \
  -i Toucan-1.5M/Toucan-1.5M \
  -o data/Toucan-1.5M-generate \
  -s stats/function_stats.json \
  --workers 32 \
  --param-pool stats/param_pool.json

# 批量 prompt 模式（串行执行，适合小规模产出）
python scripts/build_has/batch_generate.py \
  -i Toucan-1.5M/Toucan-1.5M \
  -o data/has_prompt_batch \
  -s stats/function_stats.json \
  --prompt-mode \
  --prompt-limit 10 \
  --prompt-temperature 0.4 \
  --prompt-max-tokens 512

# 单文件 prompt 生成（调试 / 小样本）
python scripts/build_has/build_has_api_prompt.py \
  -i data/toucan_1000.jsonl \
  -s stats/function_stats.json \
  -o data/has_prompt_toucan.jsonl \
  --limit 200 \
  --temperature 0.4 \
  --max-tokens 512
```

---

## 脚本分层（详细介绍）

### 1. 数据进入层（`data_preprocess/`）

| 脚本 | 主要作用 | 如何泛化 |
| --- | --- | --- |
| `generate_toucan.py` | 使用 `pyarrow` 流式读取 parquet 并写出 jsonl，支持列裁剪、行数上限、水库采样，避免整块加载。 | 将 `-i/--input` 指向任意 parquet；若 schema 不同可通过 `--columns` 指定导出字段，输出路径可自定义或沿用默认。 |
| `obfuscate_jsonl.py` | 根据 alias map 重写 jsonl 中的 `available_tools`、`messages.function_call`、`target_tools`、`metadata` 等字段，只保留混淆名（支持目录并行处理）。 | 结构字段名称不一致时，可调整脚本里的访问逻辑或在运行前自定义转换。 |
> 若需要在统计之外独立生成 alias map，可继续使用 `build_function_alias.py`（输入仍为 `function_stats_raw.json`）；但日常流程推荐直接通过 `function_stats.py --alias-output` 一并生成。生成后的统计/参数池若需混淆，直接基于混淆版 jsonl 重新跑 `function_stats.py` 与 `build_param_pool.py`，无需额外脚本。

### 2. 结构理解与统计（`analysis/`）

| 脚本 | 主要作用 | 如何泛化 |
| --- | --- | --- |
| `pretty_toucan.py` | 将少量 jsonl 记录转成带注释的 YAML/文本，便于人工检查对话、工具声明、函数调用参数。 | 如果你的工具声明不包含 `im_middle` 这类自定义标记，可替换解析函数；脚本本身已兼容字符串或字典形式。 |
| `function_stats.py` | 扫描 jsonl，统计函数/工具出现频次，输出 `function_stats.csv`、`function_meta.json`，并可通过 `--alias-output` 同步生成 alias map。 | `-i` 可指向任意目录或文件；按需启用 `--alias-output`（以及 `--alias-existing`）即可一并产出混淆映射。 |

### 3. HAS 题目构造（`build_has/`）

| 脚本 | 主要作用 | 如何泛化 |
| --- | --- | --- |
| `build_has_api_script.py` | 逐条遍历函数调用，基于 `available/params/param_values` 等策略生成选择题。输入 jsonl 已经过混淆，因此脚本输出天然只含 alias。 | 任何包含 `messages[*].function_call` 的数据集都可直接使用；若字段名不同，重写 `_parse_arguments` 或对应题目构造函数。 |
| `batch_generate.py` | 批量驱动器：可并行运行 `build_has_api_script.py`、可选 pretty 打印、复制原始文件等。 | 调整默认路径或通过 CLI 覆盖，即可用于其它数据目录的批量处理。 |
| `build_has_api_prompt.py` | 调用 LLM（vLLM/OpenAI API）回放对话，自动合成 `question_param_values` 题目，并在落盘前校验 JSON。 | 只要 jsonl 中包含 `messages` 与 `function_call.arguments`，即可直接使用；如需适配其它模型，修改 prompt 构造和 `--model` 参数即可。 |



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

## 数据拼装（轨迹 + MCQ）doing

在训练阶段会把“原始轨迹 + MCQ”串成一条文本样本（详见 `scripts/ARCHITECTURE.md` 第 5 节）。关键规则：

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