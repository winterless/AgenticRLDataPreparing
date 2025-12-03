## HAS 数据生成架构

`scripts/` 目录提供了一套从“智能体对话 + 工具调用”数据生成 HAS-API 选择题的工具链。虽然示例集中在 Toucan 语料，但所有脚本都只依赖通用输入（parquet/jsonl 源文件、OpenAI 风格 `messages`、函数元数据），因此可以轻松迁移到其它数据集。

### 1. 数据进入层（`data_preprocess/`）

| 脚本 | 主要作用 | 如何泛化 |
| --- | --- | --- |
| `generate_toucan.py` | 使用 `pyarrow` 流式读取 parquet 并写出 jsonl，支持列裁剪、行数上限、水库采样，避免整块加载。 | 将 `-i/--input` 指向任意 parquet；若 schema 不同可通过 `--columns` 指定导出字段，输出路径可自定义或沿用默认。 |
| `clean_toucan.py` | 按题目/回复评分、工具使用率等阈值过滤 jsonl，得到更干净的语料。 | 根据目标数据的质量字段调整 `--min-*` 参数或改写过滤逻辑，只要输出仍是相同结构的 jsonl，后续步骤无需改动。 |

### 2. 结构理解与统计（`analysis/`）

| 脚本 | 主要作用 | 如何泛化 |
| --- | --- | --- |
| `pretty_toucan.py` | 将少量 jsonl 记录转成带注释的 YAML/文本，便于人工检查对话、工具声明、函数调用参数。 | 如果你的工具声明不包含 `im_middle` 这类自定义标记，可替换解析函数；脚本本身已兼容字符串或字典形式。 |
| `function_stats.py` | 扫描一个或多个 jsonl，统计函数/工具出现频次，输出 `function_stats.csv` 和 `function_meta.json`。 | `-i` 可指向任意目录或文件；若函数信息存放在其它字段，可改写 `extract_functions` 即可。 |

### 3. HAS 题目构造（`build_has/`）

| 脚本 | 主要作用 | 如何泛化 |
| --- | --- | --- |
| `build_has_api_script.py` | 逐条遍历函数调用，基于 `random/available/params/param_values` 等策略生成选择题，需要 `function_meta.json` 提供参数 schema。 | 任何包含 `messages[*].function_call` 的数据集都可直接使用；若字段名不同，重写 `_parse_arguments` 或对应题目构造函数。 |
| `batch_generate.py` | 批量驱动器：可并行运行 `build_has_api_script.py`、可选 pretty 打印、复制原始文件等。 | 调整默认路径或通过 CLI 覆盖，即可用于其它数据目录的批量处理。 |
| `build_has_api_prompt.py` | 调用 LLM（vLLM/OpenAI API）回放对话，自动合成 `question_param_values` 题目，并在落盘前校验 JSON。 | 只要 jsonl 中包含 `messages` 与 `function_call.arguments`，即可直接使用；若字段/模型不同，修改 prompt 构造和 `--model` 参数即可。 |

### 4. 端到端流程

1. **转换**：使用 `generate_toucan.py` 将 parquet 转为 jsonl，可选抽样。
2. **清洗**：通过 `clean_toucan.py` 过滤低质量记录。
3. **洞察**：用 `pretty_toucan.py` 做人工抽检，`function_stats.py` 产出统计与 `function_meta.json`。
4. **生成**：选择确定性方式（`build_has_api_script.py` / `batch_generate.py`）或提示词方式（`build_has_api_prompt.py`）输出 HAS 数据。

所有阶段通过 jsonl 与 JSON 元信息衔接，因此换用其它数据集时，只需替换入口的转换/清洗逻辑，题目生成部分可以原封不动复用。


