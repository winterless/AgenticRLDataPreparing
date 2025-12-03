## HAS 数据构造流程

1. **原始数据准备**
    1.1 生成需要处理的全量数据  
    ```python scripts/data_preprocess/generate_toucan.py -i Toucan-1.5M/Toucan-1.5M --workers 8```
    1.2 采集 Toucan-1.5M 等多工具轨迹数据，保留部分失败样本用于恢复分支。
    1.3 基于 question/response 质量指标做清洗。
    ```python scripts/data_preprocess/clean_toucan.py```

2. **任务分层与标签**
    - 对轨迹按工具集合、任务类型打标签。
    ```
    - 识别 reasoning / action / exception 等节点角色。

3. **节点筛选**
    - 选择可扩展节点（规划、异常处理、总结）。
    - 过滤上下文不足或已结束的节点。

4. **候选生成（多分支）**
    - 针对节点设计 prompt，让 LLM 生成 2~3 个候选 `<think>+<tool_call>`。
    - 检查 JSON 合法性、上下文一致性。

5. **候选打分与排序**
    - 用 GPT/Gemini 打分，保留解释；将 state + actions + score 结构化存储。

6. **数据存档**
    - 用 JSONL/YAML 落盘，记录 uuid、message_idx、评估分数、模型信息等。

7. **质量复检**
    - 自动校验 `<tool_call>` JSON、标签成对等；人工抽查。

8. **与 CPT 流程联动**
    - 复用 CPT 的清洗、标签、验证脚本；HAS 数据作为分支层加入。

9. **小规模验证**
    - 选部分多候选节点喂入 RL/IFT，观察 loss/奖励，确认增益。


## HAS-API 脚本数据构建

4. **候选生成（多分支）**
    - 仅在assistant (function_call: *)行生效
    - 选项1：从stats/function_stats.json中抽取5个函数，与正确函数混在一起，构成选择题（屏蔽可选选项）
    <!-- - （废弃）选项2：从stats/function_stats.json中抽取5个和正确函数聚类函数，与正确函数混在一起，构成选择题（屏蔽可选选项和之前的函数名） -->
    - 选项2：从原有轨迹候选数据中，列出所有available_tools选项，让模型选择
    - 选项3：在正确函数选项下，从正确函数的参数中挑选正确的参数组合
    - 选项4：在正确的函数参数组合下，挑选正确的参数值

```
参数值生成思路
解析真实参数
对每条 function_call 读取 arguments，若是字符串则 json.loads，得到真实 dict 作为正确答案。
引用：question_param_values 先 _parse_arguments(fc) 并生成 correct_option。
  args = _parse_arguments(fc)
  ...
  correct_option = _format_arg_values(args)
针对类型的扰动策略（来自 _mutate_value）
  if isinstance(value, bool):
      return not value
  if isinstance(value, (int, float)):
      delta = random.choice([-5, -2, -1, 1, 2, 5])
      return value + delta
  if isinstance(value, str):
      enums = prop.get("enum")
      ...
      suffix = random.choice(["_alt","_backup","_test","_v2"])
布尔值：直接取反。
数值：从 ±1/2/5 中随机加减。
字符串：优先换成同 enum 中的其他值，否则附加随机后缀。
其他类型或无法变动时返回 None，会跳过本次尝试。
形成多套干扰选项
  while len(variations) < num_neg and attempts < max_attempts:
      target = random.choice(key_candidates)
      mutated = dict(args); mutated[target] = mutated_value
      option = _format_arg_values(mutated)
      variations.add(option)
随机挑一个参数键，按上面的策略生成新值。
与正确答案不同才加入集合，直到凑够 num_neg 或达到尝试上限。
整题结构
  options = list(variations)[:num_neg]
  options.append(correct_option)
  random.shuffle(options)
  return {
      "question": f"For the call to {func_name}, which parameter values are correct?",
      "answer_type": "single_choice",
  }
生成若干扰动选项+1 个正确选项，打乱顺序；题干说明“哪个参数取值正确”。
整体流程都在 scripts/build_has/build_has_api_script.py 的 question_param_values、_mutate_value、_format_arg_values 中。
```