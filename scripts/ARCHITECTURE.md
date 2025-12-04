## HAS 数据生成架构

`scripts/` 目录提供了一套从“智能体对话 + 工具调用”数据生成 HAS-API 选择题的工具链。虽然示例集中在 Toucan 语料，但所有脚本都只依赖通用输入（parquet/jsonl 源文件、OpenAI 风格 `messages`、函数元数据），因此可以轻松迁移到其它数据集。

### 1. 数据进入层（`data_preprocess/`）

| 脚本 | 主要作用 | 如何泛化 |
| --- | --- | --- |
| `generate_toucan.py` | 使用 `pyarrow` 流式读取 parquet 并写出 jsonl，支持列裁剪、行数上限、水库采样，避免整块加载。 | 将 `-i/--input` 指向任意 parquet；若 schema 不同可通过 `--columns` 指定导出字段，输出路径可自定义或沿用默认。 |
| `clean_toucan.py` | 按题目/回复评分、工具使用率等阈值过滤 jsonl，得到更干净的语料。 | 根据目标数据的质量字段调整 `--min-*` 参数或改写过滤逻辑，只要输出保持 Toucan 结构即可。 |
| `obfuscate_jsonl.py` | 根据 alias map 重写 jsonl 中的 `available_tools`、`messages.function_call`、`target_tools`、`metadata` 等字段，只保留混淆名。 | 结构字段名称不一致时，可调整脚本里的访问逻辑或在运行前自定义转换。 |
| `obfuscate_function_stats.py`（可选） | 对已有 `function_stats.json/.csv`、`function_meta.json` 进行就地混淆，兼容历史产物。 | 传入 alias map 即可重写；如已基于混淆 jsonl 重新统计，可跳过。 |
| `obfuscate_param_pool.py`（可选） | 将 `param_pool.json` 中的函数 key 替换为 alias，确保下游只看到混淆名。 | 默认覆盖原文件，可通过 `--output` 写入新路径。 |

> 若需要在统计之外独立生成 alias map，可继续使用 `build_function_alias.py`（输入仍为 `function_stats_raw.json`）；但日常流程推荐直接通过 `function_stats.py --alias-output` 一并生成。

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
| `build_has_api_prompt.py` | 调用 LLM（vLLM/OpenAI API）回放对话，自动合成 `question_param_values` 题目，并在落盘前校验 JSON。 | 只要 jsonl 中包含 `messages` 与 `function_call.arguments`，即可直接使用；若字段/模型不同，修改 prompt 构造和 `--model` 参数即可。 |

### 4. 端到端流程

1. **转换**：使用 `generate_toucan.py` 将 parquet 转为 jsonl，可选抽样。
2. **清洗**：通过 `clean_toucan.py` 过滤低质量记录。
3. **洞察 + 混淆**：先对原始 jsonl 跑 `function_stats.py --alias-output stats/function_alias.json` 获取 `function_stats_raw.*` 与 alias map，再用 `obfuscate_jsonl.py` 批量生成混淆版数据，并基于混淆 jsonl 重新跑一次 `function_stats.py` 与 `build_param_pool.py`。
4. **生成**：选择确定性方式（`build_has_api_script.py` / `batch_generate.py`）或提示词方式（`build_has_api_prompt.py`）输出 HAS 数据。

所有阶段通过 jsonl 与 JSON 元信息衔接，因此换用其它数据集时，只需替换入口的转换/清洗逻辑，题目生成部分可以原封不动复用。

### 5. 训练样本拼装（轨迹 + MCQ）

为了直接喂给模型训练，我们将“原始轨迹 + 派生选择题”合成为一条连续文本样本（不强制 JSON 结构），按照真实对话顺序逐步展开，规则如下：

1. **开头信息**  
   - 输出 `Question:` 段落（可附 `subset_name` 等轻量标签），紧接着给出用户原始提问。  
   - `Available tools` 段重命名为 `工具清单：`，直接列出现有工具（保持原始顺序即可），**不再额外插入假工具**，只在需要时去重并压缩描述。

2. **Tool declare 与系统上下文**  
   - 写出 `System tool declare:`，内容来自原始 `messages` 中的 system/tool_declare 区块。  
   - `api_available` 已内建额外的干扰工具，如需让模型在“工具清单/tool declare”阶段也看到这些假候选，可选择性补充 1~2 条描述，避免上下文过长。若需要在人工排查时查看「混淆名 + 原名」，可运行 `pretty_toucan.py --alias-map stats/function_alias.json`。

3. **Messages 顺序展开**  
   - 依次输出 user / assistant / function 消息。  
   - 当 assistant 准备执行 `function_call` 时，**在公布 call 参数之前**插入全部与该 `message_index` 相关的 MCQ，顺序固定为 `api_available → api_params → api_param_values`，缺失则跳过。
   - 每道题使用简短的结构化提示，例如：
     ```
     [MCQ:param_values|function=ipma-weather-data-server-get_weather_forecast|msg=4]
     问：……
     选项：A.… B.… C.… D.…
     ```
     不在文本中泄露正确答案；答案仅在监督标签侧使用。
   - 题干/选项出现频率较高时要合并、去重或截断长串，从而保持上下文紧凑。

4. **函数结果与原文答案**  
   - MCQ 区块结束后，继续写 assistant 的 function_call 参数。  
   - 输出对应的 `function` 角色结果，并用 `[[原文回答]]`（或等效标签）贴回原始 assistant 解释性文字，确保模型仍能看到完整因果链。

5. **收尾信息**  
   - 所有 `messages` 播放完毕后，追加 `Target tools:`（逗号分隔）、`Question quality assessment:`、`Response quality assessment:`，保留核心评分与理由，长文本可裁剪。  
   - 最后补上 `Metadata:`，写入 `prompt_id / mcp_servers / generation_params` 等关键信息，方便追踪来源。

通过该拼装流程，单条样本即可同时提供：

- 原始对话上下文（Question → Tool declare → Messages → Function 回执）；
- 基于同一调用生成的多套 MCQ（含结构化提示，但无显式答案）；
- 质量与元数据标签（供训练/评估使用）。

这样既能让模型在真实语境中学习“何时/如何调用工具”，又能在函数调用前后插入多种题型，实现统一训练语料。


