## HAS 数据构造流程

1. **原始数据准备**
    1.1 生成需要处理的全量数据  
    ```python scripts/generate_toucan.py -i Toucan-1.5M/Toucan-1.5M --workers 8```
    1.2 采集 Toucan-1.5M 等多工具轨迹数据，保留部分失败样本用于恢复分支。
    1.3 基于 question/response 质量指标做清洗。
    ```clean_toucan.py```

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
