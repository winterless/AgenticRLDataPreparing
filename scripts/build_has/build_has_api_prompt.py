

#!/usr/bin/env python3
"""
Prompt-based generator for HAS-API `question_param_values` style data derived
from Toucan conversation logs. Each output line mirrors
`build_has_api_script.py --mode param_values`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    _openai = importlib.import_module("openai")
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("Please install the 'openai' package: pip install openai") from exc

OpenAI = _openai.OpenAI
OpenAIError = _openai.OpenAIError

from utils import (
    format_arg_values,
    iter_function_calls,
    load_jsonl,
    load_meta,
    parse_arguments,
)


DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = (
    "/home/unlimitediw/workspace/TYDeepResearch/AgenticRLModelTraining/model/"
    "Qwen3-32B-AWQ/Qwen3-32B-AWQ"
)
SYSTEM_PROMPT = """You write multiple-choice questions that verify whether a tool call
used the correct parameter values. Always respond with valid JSON.
"""


def calc_sign(payload: dict) -> tuple[int, str]:
    """为 data 构造签名字符串，预留鉴权逻辑（当前返回空签名）。"""
    # 为data构造签名字符串
    timestamp = int(time.time())
    sign = ""
    return timestamp, sign


@dataclass
class GenerationTask:
    function_name: str
    schema: dict
    arguments: dict
    context: str
    record_uuid: str | None = None
    message_index: int | None = None


@dataclass
class PromptLimits:
    context_chars: int
    schema_chars: int
    args_chars: int

    def copy(self) -> "PromptLimits":
        return PromptLimits(self.context_chars, self.schema_chars, self.args_chars)

    def shrink(self, factor: float = 0.7) -> "PromptLimits":
        return PromptLimits(
            max(int(self.context_chars * factor), 200),
            max(int(self.schema_chars * factor), 400),
            max(int(self.args_chars * factor), 200),
        )


@dataclass
class FailureTracker:
    """记录失败统计和样例，便于排查问题函数。"""

    stats: dict[str, int]
    samples: dict[str, list[dict]]

    def __init__(self) -> None:
        self.stats = defaultdict(int)
        self.samples = defaultdict(list)

    def record(self, reason: str, task: GenerationTask, detail: str = "") -> None:
        self.stats[reason] += 1
        bucket = self.samples[reason]
        if len(bucket) < 5:
            bucket.append(
                {
                    "function_name": task.function_name,
                    "record_uuid": task.record_uuid,
                    "message_index": task.message_index,
                    "detail": detail[:200],
                }
            )

    def report(self) -> None:
        if not self.stats:
            return
        print("[WARN] Failure summary:", file=sys.stderr)
        for reason, count in self.stats.items():
            print(f"  - {reason}: {count}", file=sys.stderr)
            for sample in self.samples[reason]:
                fn = sample["function_name"]
                detail = sample.get("detail") or ""
                print(
                    f"      sample -> {fn} @idx={sample['message_index']} detail={detail}",
                    file=sys.stderr,
                )


def truncate_text(text: str, limit: int) -> str:
    """通用字符截断工具，超过上限时保留前缀并追加省略号。"""
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit - 3
    return text[: head if head > 0 else 0] + "..."


def summarize_schema(schema: dict, limit: int) -> str:
    """将函数参数 schema 概括成简洁的多行字符串，并截断长度。"""
    props = (schema or {}).get("properties") or {}
    required = set((schema or {}).get("required") or [])
    lines: list[str] = []
    for name, prop in props.items():
        type_info = prop.get("type") if isinstance(prop, dict) else None
        desc = prop.get("description") if isinstance(prop, dict) else None
        line = f"- {name}"
        if type_info:
            line += f" ({type_info})"
        if name in required:
            line += " [required]"
        if desc:
            clean_desc = " ".join(str(desc).split())
            if clean_desc:
                line += f": {clean_desc}"
        lines.append(line)
        summary = "\n".join(lines)
        if len(summary) > limit:
            break
    if not lines:
        summary = json.dumps(schema, ensure_ascii=False, indent=2)
    return truncate_text(summary, limit)


def normalize_option_text(option: str) -> str:
    """规范化 options 字符串，便于与 canonical 做宽松对比。"""
    if not isinstance(option, str):
        return ""
    text = option.strip()
    text = " ".join(text.split())
    text = re.sub(r"\s*=\s*", "=", text)
    text = re.sub(r"\s*;\s*", "; ", text)
    return text


def list_available_models(client: OpenAI) -> list[str]:
    """调用 /v1/models 列出当前可用的模型 id，用于 404 回退。"""
    try:
        resp = client.models.list()
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[WARN] Failed to list models from endpoint: {exc}", file=sys.stderr)
        return []
    models = getattr(resp, "data", []) or []
    names: list[str] = []
    for item in models:
        model_id = getattr(item, "id", None)
        if model_id:
            names.append(model_id)
    return names


def toucan_tasks(input_path: Path, meta: dict[str, dict], limit: int | None) -> Iterable[GenerationTask]:
    """遍历 Toucan jsonl，抽取函数调用上下文，包装成 GenerationTask。"""
    produced = 0
    for record in load_jsonl(input_path):
        record_uuid = record.get("uuid")
        for msg_idx, fc in iter_function_calls(record):
            func = fc["name"]
            args = parse_arguments(fc)
            if not args:
                continue
            schema = meta.get(func, {}).get("function") or meta.get(func) or {}
            if not schema:
                continue
            context = record.get("messages", "")
            if isinstance(context, list):
                # keep last user message for brevity
                user_msgs = [m.get("content") for m in context if m.get("role") == "user"]
                context = user_msgs[-1] if user_msgs else ""
            yield GenerationTask(
                function_name=func,
                schema=schema,
                arguments=args,
                context=context or "",
                record_uuid=record_uuid,
                message_index=msg_idx,
            )
            produced += 1
            if limit and produced >= limit:
                return


def build_prompt(task: GenerationTask, limits: PromptLimits) -> str:
    """根据任务信息构建提示词，包含 schema/参数/上下文摘要。"""
    schema_text = summarize_schema(task.schema, limits.schema_chars)
    args_text = truncate_text(json.dumps(task.arguments, ensure_ascii=False, indent=2), limits.args_chars)
    canonical = format_arg_values(task.arguments)
    context = truncate_text(task.context.strip() or "Conversation context omitted.", limits.context_chars)
    return (
        "Goal: create a HAS-API multiple-choice question that checks whether the agent\n"
        "used the correct parameters when calling a tool.\n\n"
        f"Function name: {task.function_name}\n"
        f"Context snippet: {context}\n"
        f"Function schema (JSON):\n{schema_text}\n\n"
        f"Correct arguments JSON:\n{args_text}\n\n"
        "You must output JSON with exactly three keys: question (string), options\n"
        "(array of 4-5 short strings), and answer (string).\n"
        f"The canonical correct option string is:\n{canonical}\n"
        "Include that exact string verbatim as one of the options and set answer to the\n"
        "same string. For distractors, change one or two parameter values while keeping\n"
        "the same format (key=value; key2=value2 ...). Respond with JSON only."
    )


def extract_json_block(text: str) -> str | None:
    """从模型输出中提取 JSON 代码块，容忍未标注语言或裸花括号。"""
    text = text.strip()
    if not text:
        return None
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        return code_block.group(1)
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)
    return None


def call_llm(client: OpenAI, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    """统一的 LLM 调用入口，附带签名占位与系统提示设置。"""
    # 构造请求参数
    # 生成签名
    calc_sign({"prompt": prompt})
    # 构造请求头
    # 发送请求并获得响应
    # 处理响应数据
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def validate_payload(payload: dict, canonical: str) -> tuple[bool, str]:
    """校验模型返回的 JSON，确保字段齐全且答案匹配 canonical。"""
    if not isinstance(payload, dict):
        return False, "payload_not_dict"
    question = payload.get("question")
    options = payload.get("options")
    answer = payload.get("answer")
    if not question or not isinstance(question, str):
        return False, "invalid_question"
    if not options or not isinstance(options, list):
        return False, "invalid_options"
    normalized_canonical = normalize_option_text(canonical)
    normalized_options = [normalize_option_text(opt) for opt in options]
    if normalized_canonical not in normalized_options:
        return False, "canonical_missing"
    if normalize_option_text(answer) != normalized_canonical:
        return False, "answer_mismatch"
    return True, ""


def write_entry(sink, task: GenerationTask, payload: dict) -> None:
    """将合格的题目写入 jsonl，一条一行。"""
    entry = {
        "mode": "prompt_toucan_param_values",
        "question": payload["question"],
        "options": payload["options"],
        "answer": payload["answer"],
        "function_name": task.function_name,
        "record_uuid": task.record_uuid,
        "message_index": task.message_index,
    }
    json.dump(entry, sink, ensure_ascii=False)
    sink.write("\n")


def parse_args() -> argparse.Namespace:
    """解析命令行参数，统一入口，方便在其它数据集复用。"""
    parser = argparse.ArgumentParser(description="Prompt-based HAS param_values generator for Toucan data.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output jsonl path.")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Toucan jsonl input file.")
    parser.add_argument("-s", "--stats", type=Path, required=True, help="function_meta stats JSON.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of samples.")
    parser.add_argument("--temperature", type=float, default=0.4, help="LLM sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens for completion.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per sample on bad outputs.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between calls.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible endpoint.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name/path served by vLLM.")
    parser.add_argument("--api-key", default="EMPTY", help="API key for the endpoint.")
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=800,
        help="Truncate conversation snippet to at most N characters (default: 800).",
    )
    parser.add_argument(
        "--max-schema-chars",
        type=int,
        default=1600,
        help="Truncate serialized schema summary to at most N characters (default: 1600).",
    )
    parser.add_argument(
        "--max-args-chars",
        type=int,
        default=400,
        help="Truncate canonical arguments JSON to at most N characters (default: 400).",
    )
    return parser.parse_args()


def main() -> None:
    """主流程：加载配置、遍历任务、调用 LLM 生成并写入 jsonl。"""
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    if not args.stats.exists():
        raise SystemExit(f"Stats file not found: {args.stats}")
    meta = load_meta(args.stats)

    tasks_iter = toucan_tasks(args.input, meta, args.limit)

    base_limits = PromptLimits(
        context_chars=args.max_context_chars,
        schema_chars=args.max_schema_chars,
        args_chars=args.max_args_chars,
    )

    current_model = args.model
    produced = 0
    failure_tracker = FailureTracker()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as sink:
        for task in tasks_iter:
            limits = base_limits.copy()
            canonical = format_arg_values(task.arguments)
            success = False
            for attempt in range(1, args.retries + 1):
                prompt = build_prompt(task, limits)
                try:
                    content = call_llm(client, current_model, prompt, args.temperature, args.max_tokens)
                except OpenAIError as exc:
                    err_text = getattr(exc, "message", str(exc))
                    print(f"[WARN] LLM call failed (attempt {attempt}): {err_text}", file=sys.stderr)
                    lowered = (err_text or "").lower()
                    status_code = getattr(exc, "status_code", None)
                    if (
                        status_code == 404
                        or "model" in lowered and "does not exist" in lowered
                        or "not found" in lowered
                    ):
                        available = list_available_models(client)
                        if available:
                            if current_model not in available:
                                fallback = available[0]
                                print(
                                    f"[WARN] Model '{current_model}' unavailable. "
                                    f"Falling back to '{fallback}'. Available models: {available}",
                                    file=sys.stderr,
                                )
                                current_model = fallback
                            else:
                                print(
                                    f"[WARN] Requested model '{current_model}' exists but endpoint still returned 404.",
                                    file=sys.stderr,
                                )
                        else:
                            print(
                                "[WARN] Unable to fetch available models; please ensure the vLLM server exposes the "
                                "desired `--served-model-name`.",
                                file=sys.stderr,
                            )
                        failure_tracker.record("model_not_found", task, err_text)
                        time.sleep(args.sleep)
                        continue
                    if "maximum context length" in lowered or "context length" in lowered:
                        limits = limits.shrink()
                        print(
                            f"[WARN] Prompt truncated further to avoid token overflow "
                            f"(context={limits.context_chars}, schema={limits.schema_chars}, args={limits.args_chars}).",
                            file=sys.stderr,
                        )
                        failure_tracker.record("context_length", task, err_text)
                    time.sleep(args.sleep)
                    continue
                payload_str = extract_json_block(content)
                if not payload_str:
                    print(f"[WARN] No JSON detected for {task.function_name} (attempt {attempt}).", file=sys.stderr)
                    failure_tracker.record("no_json", task, content or "")
                    time.sleep(args.sleep)
                    continue
                try:
                    payload = json.loads(payload_str)
                except json.JSONDecodeError as exc:
                    print(f"[WARN] JSON parse error for {task.function_name}: {exc}", file=sys.stderr)
                    failure_tracker.record("json_parse_error", task, payload_str)
                    time.sleep(args.sleep)
                    continue
                ok, reason = validate_payload(payload, canonical)
                if not ok:
                    print(
                        f"[WARN] Invalid payload for {task.function_name} (attempt {attempt}). reason={reason}",
                        file=sys.stderr,
                    )
                    failure_tracker.record(reason or "invalid_payload", task, json.dumps(payload, ensure_ascii=False)[:200])
                    time.sleep(args.sleep)
                    continue
                write_entry(sink, task, payload)
                produced += 1
                success = True
                break
            if not success:
                print(f"[ERROR] Exhausted retries for {task.function_name}, skipping.", file=sys.stderr)
                failure_tracker.record("exhausted_retries", task)
            if args.limit and produced >= args.limit:
                break
            time.sleep(args.sleep)

    print(f"[INFO] Generated {produced} prompt-based entries from Toucan data.")
    failure_tracker.report()


if __name__ == "__main__":
    main()


def build_prompt(task: GenerationTask, limits: PromptLimits) -> str:
    schema_text = summarize_schema(task.schema, limits.schema_chars)
    args_text = truncate_text(json.dumps(task.arguments, ensure_ascii=False, indent=2), limits.args_chars)
    canonical = format_arg_values(task.arguments)
    context = truncate_text(task.context.strip() or "Conversation context omitted.", limits.context_chars)
    return (
        "Goal: create a HAS-API multiple-choice question that checks whether the agent\n"
        "used the correct parameters when calling a tool.\n\n"
        f"Function name: {task.function_name}\n"
        f"Context snippet: {context}\n"
        f"Function schema (JSON):\n{schema_text}\n\n"
        f"Correct arguments JSON:\n{args_text}\n\n"
        "You must output JSON with exactly three keys: question (string), options\n"
        "(array of 4-5 short strings), and answer (string).\n"
        f"The canonical correct option string is:\n{canonical}\n"
        "Include that exact string verbatim as one of the options and set answer to the\n"
        "same string. For distractors, change one or two parameter values while keeping\n"
        "the same format (key=value; key2=value2 ...). Respond with JSON only."
    )