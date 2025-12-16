#!/usr/bin/env python3
"""
Assemble obfuscated Toucan trajectories with HAS-API MCQs into training-ready text.

The script reads the canonical `toucan*.jsonl` artifacts under `data/`, injects the
available/params/param_values MCQs at each function-call turn, and writes a JSONL
file whose `text` field contains the concatenated prompt described in README.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import string
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from textwrap import indent

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.utils.has_utils import load_jsonl, parse_arguments  # noqa: E402

MODE_ORDER = ("available", "params", "param_values")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NO_MCQ_TAG_DEFAULT = "[NO_MCQ]"


@dataclass(frozen=True)
class BatchJob:
    prefix: str
    conversation: Path
    available: Path | None
    params: Path | None
    param_values: Path | None
    output: Path
    text_output: Path | None
    skip_mcq: bool
    missing_mcq_dump: Path | None = None


def _derive_no_mcq_path(path: Path) -> Path:
    name = path.name
    if name.endswith("_mcq_assembled.jsonl"):
        return path.with_name(name.replace("_mcq_assembled.jsonl", "_no_mcq_assembled.jsonl"))
    if name.endswith("_mcq_assembled.txt"):
        return path.with_name(name.replace("_mcq_assembled.txt", "_no_mcq_assembled.txt"))
    return path.with_name(path.stem + "_no_mcq" + path.suffix)


def _mcq_keep_for_uuid(uuid: str, p: float, seed: int) -> bool:
    """Deterministic per-record Bernoulli(p) based on uuid+seed."""
    if p >= 1.0:
        return True
    if p <= 0.0:
        return False
    # Use a stable hash to avoid randomness differences across processes.
    h = hashlib.md5(f"{seed}:{uuid}".encode("utf-8")).hexdigest()
    # Take 32 bits for a stable uniform in [0,1).
    x = int(h[:8], 16) / 2**32
    return x < p


def _iter_alias_candidate_names(record: dict, per_record_mcq: dict) -> set[str]:
    """Collect all function/tool names that should be aliased for this record."""
    names: set[str] = set()

    def add(val):
        if isinstance(val, str) and val:
            names.add(val)

    # available_tools
    raw_tools = parse_json_like(record.get("available_tools"))
    tools = raw_tools if isinstance(raw_tools, list) else []
    for tool in tools:
        func = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(func, dict):
            add(func.get("name"))

    # messages: function_call.name, role=function name, tool_calls
    messages = ensure_messages(record)
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role in {"function", "tool"}:
            add(msg.get("name"))
        fc = msg.get("function_call")
        if isinstance(fc, dict):
            add(fc.get("name"))
        tc = parse_json_like(msg.get("tool_calls"))
        if isinstance(tc, list):
            for entry in tc:
                func = entry.get("function") if isinstance(entry, dict) else None
                if isinstance(func, dict):
                    add(func.get("name"))

    # MCQs: function_name + options (strings or dicts with name)
    for msg_mcqs in per_record_mcq.values():
        if not isinstance(msg_mcqs, dict):
            continue
        for mode_entries in msg_mcqs.values():
            if not isinstance(mode_entries, list):
                continue
            for entry in mode_entries:
                if not isinstance(entry, dict):
                    continue
                add(entry.get("function_name"))
                options = entry.get("options", [])
                if isinstance(options, list):
                    for opt in options:
                        if isinstance(opt, str):
                            add(opt)
                        elif isinstance(opt, dict):
                            add(opt.get("name"))

    # target_tools
    target = parse_json_like(record.get("target_tools"))
    if isinstance(target, list):
        for t in target:
            add(t)
    elif isinstance(target, str):
        add(target)

    return names


def _rand_alias_token(prefix: str, length: int, rng: random.Random, used: set[str]) -> str:
    alphabet = string.ascii_lowercase + string.digits
    while True:
        token = prefix + "".join(rng.choices(alphabet, k=length))
        if token not in used:
            used.add(token)
            return token


def _build_random_alias_map(names: set[str], uuid: str, seed: int) -> dict[str, str]:
    """
    Build a per-record random alias map (deterministic by uuid+seed).
    This is intended to hide real tool names inside Available tools / MCQ / function calls.
    """
    rng = random.Random(f"{seed}:{uuid}")
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for name in sorted(names):
        if not name:
            continue
        mapping[name] = _rand_alias_token("func_", 8, rng, used)
    return mapping


def _apply_alias_to_option(option, alias_map: dict[str, str]) -> object:
    if not alias_map:
        return option
    if isinstance(option, str):
        return alias_map.get(option, option)
    if isinstance(option, dict) and option.get("name"):
        new_opt = dict(option)
        new_opt["name"] = alias_map.get(new_opt["name"], new_opt["name"])
        return new_opt
    return option


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitch Toucan trajectories with MCQs.")
    parser.add_argument(
        "-i",
        "--conv-root",
        type=Path,
        default=REPO_ROOT / "data",
        help="Root directory containing raw conversation json/jsonl files (default: data/).",
    )
    parser.add_argument(
        "-m",
        "--mcq-root",
        type=Path,
        default=None,
        help="Root directory containing *_api_*.jsonl MCQ files (default: same as --conv-root).",
    )
    parser.add_argument(
        "-s",
        "--stats",
        type=Path,
        default=None,
        help=(
            "Path to function_stats.json. If provided, MCQ distractor functions "
            "will be looked up and their descriptions/schemas added to Available tools."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Max worker threads when assembling multiple files (default: 4).",
    )
    parser.add_argument(
        "--no-text-output",
        action="store_true",
        help="Skip emitting the pretty text companion files.",
    )
    parser.add_argument(
        "--show-function-name",
        action="store_true",
        help="Include the target function name in MCQ headers (default hides it).",
    )
    parser.add_argument(
        "--task-select-tag",
        type=str,
        default="",
        help=(
            "If set (non-empty), emit this tag line immediately before each injected MCQ block. "
            "Use this as a gating signal, e.g. '<TASK=SELECT>'."
        ),
    )
    parser.add_argument(
        "--task-select-end-tag",
        type=str,
        nargs="?",
        const="",
        default=None,
        help=(
            "Emit this tag line immediately after each injected MCQ block. "
            "If omitted, it will be auto-derived from --task-select-tag (e.g. '<TASK=SELECT>' -> '</TASK=SELECT>'). "
            "If provided with no value, it disables emitting the end tag."
        ),
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip emitting 'Metadata:' section in assembled text (default: skip).",
    )
    parser.add_argument(
        "--skip-question-quality",
        action="store_true",
        help="Skip emitting 'Question quality assessment:' section (default: skip).",
    )
    parser.add_argument(
        "--skip-response-quality",
        action="store_true",
        help="Skip emitting 'Response quality assessment:' section (default: skip).",
    )
    parser.add_argument(
        "--answer-redact",
        choices=["none", "redact", "drop"],
        default="none",
        help=(
            "Control whether to reveal MCQ answers in assembled text. "
            "'none' keeps the original Answer line; 'redact' keeps the line but hides the value; "
            "'drop' removes the Answer line."
        ),
    )
    parser.add_argument(
        "--random-alias-per-record",
        action="store_true",
        help=(
            "If set, generate a per-record random alias map and apply it consistently to "
            "Available tools / MCQ options / function_call names in the assembled text."
        ),
    )
    parser.add_argument(
        "--random-alias-seed",
        type=int,
        default=0,
        help="Seed used for deterministic --random-alias-per-record alias generation (default: 0).",
    )
    parser.add_argument(
        "--mcq-tag",
        type=str,
        default="",
        help=(
            "If set (non-empty), emit this tag line immediately before each injected MCQ block "
            "to isolate MCQ style (e.g. '[MCQ]')."
        ),
    )
    parser.add_argument(
        "--emit-no-mcq-tag",
        action="store_true",
        help=(
            "If set, emit a shard marker tag for records that are emitted without any MCQ blocks "
            "(e.g. due to --mcq-subsample or --passthrough-only)."
        ),
    )
    parser.add_argument(
        "--mcq-subsample",
        type=float,
        default=1.0,
        help=(
            "Subsample MCQ injection probability per record (0..1). "
            "When <1, some records will be emitted without MCQ blocks (deterministically by uuid)."
        ),
    )
    parser.add_argument(
        "--mcq-subsample-seed",
        type=int,
        default=0,
        help="Seed used for deterministic --mcq-subsample decisions (default: 0).",
    )
    parser.add_argument(
        "--split-shards",
        action="store_true",
        help=(
            "If set, split outputs into two shards per conversation file: "
            "*_mcq_assembled.* (records that kept MCQs) and *_no_mcq_assembled.* (records without MCQs)."
        ),
    )
    parser.add_argument(
        "--passthrough-only",
        action="store_true",
        help=(
            "Do not stitch MCQs; just emit UTF-8 cleaned records/text. "
            "Useful for ablations or scaling-law runs without MCQ augmentation."
        ),
    )
    parser.add_argument(
        "--keep-missing-mcq",
        action="store_true",
        help=(
            "Keep conversations even if MCQ files are missing; emit them without MCQs instead of skipping."
        ),
    )
    parser.add_argument(
        "--missing-mcq-dump-dir",
        type=Path,
        default=None,
        help=(
            "If set, pretty-prints conversations that lacked MCQs into this directory "
            "(one txt per conversation) for later inspection. Only used when MCQs are missing."
        ),
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return list(load_jsonl(path))
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON structure in {path}")
def build_mcq_index(paths: Iterable[Path | None]):
    index: dict[str, dict[int, dict[str, list[dict]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    total = 0
    for path in paths:
        if not path:
            continue
        for item in load_jsonl(path):
            uuid = item.get("record_uuid")
            msg_idx = item.get("message_index")
            mode = (item.get("mode") or "").lower()
            if uuid is None or msg_idx is None or mode not in MODE_ORDER:
                continue
            index[str(uuid)][int(msg_idx)][mode].append(item)
            total += 1
    return index, total


def ensure_messages(record: dict) -> list[dict]:
    messages = record.get("messages")
    if isinstance(messages, list):
        return messages
    if isinstance(messages, str):
        try:
            decoded = json.loads(messages)
            return decoded if isinstance(decoded, list) else []
        except json.JSONDecodeError:
            return []
    return []


def parse_json_like(value):
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def format_system_tool_declare(content: str) -> str:
    if "<|im_system|>" not in content:
        return content.strip()
    start_mid = content.find("<|im_middle|>")
    end_tag = "<|im_end|>"
    end_idx = content.find(end_tag, start_mid)
    if start_mid == -1 or end_idx == -1:
        return content.strip()
    raw = content[start_mid + len("<|im_middle|>") : end_idx].strip()
    try:
        decoded = bytes(raw, "utf-8").decode("unicode_escape")
    except Exception:
        decoded = raw
    start_list = decoded.find("[")
    end_list = decoded.rfind("]")
    if start_list == -1 or end_list == -1:
        return decoded.strip()
    payload = decoded[start_list : end_list + 1]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return decoded.strip()
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True).strip()


def format_json_block(payload) -> str:
    if isinstance(payload, (dict, list)):
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    return str(payload)


def dump_yaml(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return trimmed
        try:
            parsed = json.loads(trimmed)
        except json.JSONDecodeError:
            return trimmed
    elif isinstance(value, (dict, list)):
        parsed = value
    else:
        return str(value)
    return yaml.safe_dump(parsed, sort_keys=False, allow_unicode=True).strip()




def _collect_mcq_function_names(mcqs: dict) -> set[str]:
    """Extract all function names from MCQ options."""
    names: set[str] = set()
    for msg_mcqs in mcqs.values():
        for mode_entries in msg_mcqs.values():
            for entry in mode_entries:
                options = entry.get("options", [])
                if isinstance(options, list):
                    for opt in options:
                        # Options can be strings (function names) or dicts
                        if isinstance(opt, str):
                            names.add(opt)
                        elif isinstance(opt, dict) and opt.get("name"):
                            names.add(opt["name"])
    return names


def _build_extra_tools_from_stats(
    func_names: set[str],
    existing_names: set[str],
    stats_meta: dict[str, dict],
) -> list[dict]:
    """Build tool definitions for MCQ distractor functions not in available_tools."""
    extra: list[dict] = []
    for name in func_names:
        if name in existing_names:
            continue
        meta = stats_meta.get(name)
        if not meta:
            continue
        # Build tool structure matching available_tools format
        tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": meta.get("description", ""),
                "parameters": meta.get("parameters", {}),
            }
        }
        extra.append(tool)
    return extra


def format_tools(record: dict, extra_tools: list[dict] | None = None) -> list[str]:
    raw = parse_json_like(record.get("available_tools"))
    tools = raw if isinstance(raw, list) else []
    # Merge extra_tools (MCQ distractors) into the list
    if extra_tools:
        tools = list(tools) + extra_tools
    lines: list[str] = []
    for entry in tools:
        function = entry.get("function") if isinstance(entry, dict) else None
        if not isinstance(function, dict):
            continue
        name = function.get("name", "unknown")
        block: list[str] = [f"- {name}"]
        desc = (function.get("description") or "").strip()
        if desc:
            block.append(indent(desc, "    "))
        params = function.get("parameters")
        if params:
            block.append(indent("Parameters:", "    "))
            block.append(indent(format_json_block(params), "      "))
        lines.append(indent("\n".join(block), "  "))
    return lines


def format_options(options) -> str:
    rendered = []
    opts = options if isinstance(options, list) else [options]
    for idx, option in enumerate(opts):
        prefix = LETTERS[idx] if idx < len(LETTERS) else f"Option{idx+1}"
        rendered.append(f"{prefix}. {option}")
    return "\n".join(rendered)


def format_mcq_block(
    entry: dict,
    answer_redact: str,
    show_function_name: bool,
    task_select_tag: str,
    task_select_end_tag: str,
    mcq_tag: str,
    alias_map: dict[str, str] | None = None,
) -> str:
    header_bits = [f"[MCQ:{entry.get('mode')}"]
    header_bits.append(f"|msg={entry.get('message_index')}")
    if show_function_name and entry.get("function_name"):
        fn = entry["function_name"]
        if alias_map:
            fn = alias_map.get(fn, fn)
        header_bits.append(f"|function={fn}")
    header_bits.append("]")
    header = "".join(header_bits)
    lines: list[str] = []
    if mcq_tag:
        lines.append(mcq_tag)
    if task_select_tag:
        lines.append(task_select_tag)
    options = entry.get("options", [])
    if alias_map and isinstance(options, list):
        options = [_apply_alias_to_option(opt, alias_map) for opt in options]
    lines.extend(
        [
            header,
            "Question:",
            indent(entry.get("question", ""), "  "),
            "Options:",
            indent(format_options(options), "  "),
        ]
    )
    if entry.get("answer"):
        if answer_redact == "none":
            lines.append(f"Answer: the answer is {entry['answer']}")
        elif answer_redact == "redact":
            lines.append("Answer: the answer is [ANSWER_REDACTED]")
        elif answer_redact == "drop":
            pass
    if task_select_end_tag:
        lines.append(task_select_end_tag)
    return "\n".join(lines)


def assemble_record(
    record: dict,
    mcqs,
    answer_redact: str,
    show_function_name: bool,
    task_select_tag: str,
    task_select_end_tag: str,
    mcq_tag: str,
    emit_no_mcq_tag: bool,
    record_has_mcq: bool,
    random_alias_per_record: bool,
    random_alias_seed: int,
    skip_metadata: bool,
    skip_question_quality: bool,
    skip_response_quality: bool,
    stats_meta: dict[str, dict] | None = None,
) -> str:
    parts: list[str] = []
    uuid = record.get("uuid") or record.get("record_uuid") or "unknown"
    question = record.get("question", "").strip()
    parts.append(f"=== Record | uuid={uuid} ===")
    if emit_no_mcq_tag and not record_has_mcq:
        parts.append(NO_MCQ_TAG_DEFAULT)
    if question:
        parts.append("Question:")
        parts.append(indent(question, "  "))
    # Collect all function names from MCQ options
    per_record_mcq = mcqs.get(str(uuid), {})
    alias_map: dict[str, str] | None = None
    if random_alias_per_record:
        alias_names = _iter_alias_candidate_names(record, per_record_mcq)
        alias_map = _build_random_alias_map(alias_names, str(uuid), random_alias_seed)
    mcq_func_names = _collect_mcq_function_names(per_record_mcq) if stats_meta else set()
    
    # Build extra tools for MCQ distractors
    extra_tools = None
    if stats_meta and mcq_func_names:
        raw_available = parse_json_like(record.get("available_tools"))
        available_tools = raw_available if isinstance(raw_available, list) else []
        existing_names = set()
        for tool in available_tools:
            func = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(func, dict) and func.get("name"):
                existing_names.add(func["name"])
        extra_tools = _build_extra_tools_from_stats(mcq_func_names, existing_names, stats_meta)
    
    tool_lines = format_tools(record, extra_tools)
    if tool_lines:
        parts.append("Available tools:")
        if alias_map:
            aliased_lines: list[str] = []
            # `format_tools` returns text; do a simple safe replace only on "- name" prefixes.
            # Keep this scoped to avoid accidentally changing descriptions.
            for line in tool_lines:
                stripped = line.lstrip()
                if stripped.startswith("- "):
                    # expected: "  - <name>"
                    prefix_len = len(line) - len(stripped)
                    prefix = line[:prefix_len]
                    after = stripped[2:]
                    name = after.splitlines()[0].strip()
                    aliased = alias_map.get(name, name)
                    aliased_lines.append(prefix + "- " + after.replace(name, aliased, 1))
                else:
                    aliased_lines.append(line)
            parts.extend(aliased_lines)
        else:
            parts.extend(tool_lines)
    messages = ensure_messages(record)
    parts.append("Messages:")

    awaiting_answer = False

    for idx, message in enumerate(messages):
        role = message.get("role", "unknown")
        function_call = message.get("function_call")
        if function_call:
            injections = per_record_mcq.get(idx, {})
            for mode in MODE_ORDER:
                for entry in injections.get(mode, []):
                    parts.append(
                        indent(
                            format_mcq_block(
                                entry,
                                answer_redact,
                                show_function_name,
                                task_select_tag,
                                task_select_end_tag,
                                mcq_tag,
                                alias_map,
                            ),
                            "    ",
                        )
                    )
            name = function_call.get("name", "unknown")
            if alias_map:
                name = alias_map.get(name, name)
            args = parse_arguments(function_call) or function_call.get("arguments")
            parts.append(
                indent(f"assistant (function_call: {name}) @msg={idx}:", "  ")
            )
            parts.append(indent(format_json_block(args), "    "))
            awaiting_answer = True
            continue

        if role == "function":
            func_name = message.get("name", "function")
            if alias_map:
                func_name = alias_map.get(func_name, func_name)
            payload = parse_json_like(message.get("content"))
            parts.append(indent(f"function[{func_name}] @msg={idx}:", "  "))
            parts.append(indent(format_json_block(payload), "    "))
            continue

        content = message.get("content", "")
        if awaiting_answer and role == "assistant" and content:
            parts.append(indent("[[原文回答]]", "  "))
            awaiting_answer = False
        header = role
        if role == "system" and isinstance(content, str) and "<|im_system|>" in content:
            header = f"{role} (tool_declare)"
        parts.append(indent(f"{header}:", "  "))
        if isinstance(content, str):
            rendered = (
                format_system_tool_declare(content)
                if role == "system"
                else content.strip()
            )
            parts.append(indent(rendered, "    "))
        else:
            parts.append(indent(format_json_block(content), "    "))

    if record.get("target_tools"):
        targets = record["target_tools"]
        if alias_map and isinstance(targets, list):
            targets = [alias_map.get(t, t) if isinstance(t, str) else t for t in targets]
        elif alias_map and isinstance(targets, str):
            targets = alias_map.get(targets, targets)
        rendered = (
            yaml.safe_dump(targets, allow_unicode=True, sort_keys=False).strip()
            if isinstance(targets, (dict, list))
            else str(targets)
        )
        parts.append("Target tools:")
        parts.append(indent(rendered, "  "))
    if record.get("question_quality_assessment") and not skip_question_quality:
        parsed = parse_json_like(record["question_quality_assessment"])
        parts.append("Question quality assessment:")
        parts.append(indent(format_json_block(parsed), "  "))
    if record.get("response_quality_assessment") and not skip_response_quality:
        parsed = parse_json_like(record["response_quality_assessment"])
        parts.append("Response quality assessment:")
        parts.append(indent(format_json_block(parsed), "  "))
    if record.get("metadata") and not skip_metadata:
        parsed = parse_json_like(record["metadata"])
        parts.append("Metadata:")
        parts.append(indent(format_json_block(parsed), "  "))

    return "\n".join(parts).strip() + "\n"


def assemble_to_outputs(
    conversation_path: Path,
    available_path: Path | None,
    params_path: Path | None,
    param_values_path: Path | None,
    output_path: Path,
    text_path: Path | None,
    show_function_name: bool,
    skip_mcq: bool,
    task_select_tag: str,
    task_select_end_tag: str,
    answer_redact: str,
    mcq_tag: str,
    emit_no_mcq_tag: bool,
    random_alias_per_record: bool,
    random_alias_seed: int,
    mcq_subsample: float,
    mcq_subsample_seed: int,
    split_shards: bool,
    skip_metadata: bool,
    skip_question_quality: bool,
    skip_response_quality: bool,
    stats_meta: dict[str, dict] | None = None,
    missing_mcq_dump: Path | None = None,
) -> tuple[int, int, int]:
    if not conversation_path.exists():
        raise FileNotFoundError(f"Conversation file missing: {conversation_path}")
    if not skip_mcq:
        for label, path in [
            ("available", available_path),
            ("params", params_path),
            ("param_values", param_values_path),
        ]:
            if not path or not path.exists():
                raise FileNotFoundError(f"{label} MCQ missing: {path}")

    records = load_records(conversation_path)
    if skip_mcq:
        mcq_index, mcq_total = {}, 0
    else:
        mcq_index, mcq_total = build_mcq_index([available_path, params_path, param_values_path])
    to_emit = records

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_no_mcq_path = _derive_no_mcq_path(output_path) if split_shards else None
    text_no_mcq_path = _derive_no_mcq_path(text_path) if (split_shards and text_path) else None

    text_fh = text_path.open("w", encoding="utf-8") if text_path else None
    text_no_fh = text_no_mcq_path.open("w", encoding="utf-8") if text_no_mcq_path else None
    missing_fh = None
    if missing_mcq_dump:
        missing_mcq_dump.parent.mkdir(parents=True, exist_ok=True)
        missing_fh = missing_mcq_dump.open("w", encoding="utf-8")
    kept = 0
    kept_mcq = 0
    kept_no_mcq = 0
    dropped_jsonl = 0
    dropped_text = 0
    dropped_missing = 0
    try:
        with output_path.open("w", encoding="utf-8") as fh_mcq, (
            output_no_mcq_path.open("w", encoding="utf-8") if output_no_mcq_path else nullcontext()
        ) as fh_no:
            for record in to_emit:
                uuid = record.get("uuid") or record.get("record_uuid") or "unknown"
                keep_mcq = (not skip_mcq) and _mcq_keep_for_uuid(
                    str(uuid), mcq_subsample, mcq_subsample_seed
                )
                base_text = assemble_record(
                    record,
                    mcq_index if keep_mcq else {},
                    answer_redact,
                    show_function_name,
                    task_select_tag,
                    task_select_end_tag,
                    mcq_tag,
                    emit_no_mcq_tag,
                    record_has_mcq=keep_mcq,
                    random_alias_per_record=random_alias_per_record,
                    random_alias_seed=random_alias_seed,
                    skip_metadata,
                    skip_question_quality,
                    skip_response_quality,
                    stats_meta,
                )
                payload = {
                    "uuid": record.get("uuid") or record.get("record_uuid"),
                    "text": base_text,
                }
                try:
                    target_fh = fh_mcq if (not split_shards or keep_mcq) else fh_no
                    if target_fh is None:
                        target_fh = fh_mcq
                    target_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    kept += 1
                    if keep_mcq:
                        kept_mcq += 1
                    else:
                        kept_no_mcq += 1
                except UnicodeEncodeError:
                    dropped_jsonl += 1
                    print(
                        f"[WARN] Dropped non-UTF-8 jsonl line (uuid={payload['uuid']}, file={conversation_path})",
                        file=sys.stderr,
                    )
                    continue

                if text_fh:
                    pretty_text = base_text
                    try:
                        target_text_fh = text_fh if (not split_shards or keep_mcq) else text_no_fh
                        if target_text_fh is None:
                            target_text_fh = text_fh
                        target_text_fh.write(pretty_text)
                        if not pretty_text.endswith("\n"):
                            target_text_fh.write("\n")
                        target_text_fh.write("\n")
                    except UnicodeEncodeError:
                        dropped_text += 1
                        print(
                            f"[WARN] Dropped non-UTF-8 text line (uuid={payload['uuid']}, file={conversation_path})",
                            file=sys.stderr,
                        )
                        continue
                if missing_fh:
                    try:
                        missing_fh.write(base_text)
                        if not base_text.endswith("\n"):
                            missing_fh.write("\n")
                        missing_fh.write("\n")
                    except UnicodeEncodeError:
                        dropped_missing += 1
                        print(
                            f"[WARN] Dropped non-UTF-8 missing-MCQ text line (uuid={payload['uuid']}, file={conversation_path})",
                            file=sys.stderr,
                        )

    finally:
        if text_fh:
            text_fh.close()
        if text_no_fh:
            text_no_fh.close()
        if missing_fh:
            missing_fh.close()

    if dropped_jsonl or dropped_text or dropped_missing:
        print(
            f"[INFO] {conversation_path.name}: kept {kept}, "
            f"dropped_jsonl={dropped_jsonl}, dropped_text={dropped_text}, dropped_missing={dropped_missing}",
            file=sys.stderr,
        )

    return kept, kept_mcq, mcq_total


def _auto_derive_end_tag(task_select_tag: str) -> str:
    tag = (task_select_tag or "").strip()
    if not tag:
        return ""
    # Common case: "<TASK=SELECT>" -> "</TASK=SELECT>"
    if tag.startswith("<") and tag.endswith(">") and not tag.startswith("</"):
        inner = tag[1:-1].strip()
        if inner:
            return f"</{inner}>"
    # Fallback: no reliable derivation.
    return ""

def discover_batch_jobs(
    conv_root: Path,
    mcq_root: Path,
    include_text: bool,
    require_mcq: bool,
    global_skip_mcq: bool,
    keep_missing_mcq: bool,
    missing_mcq_dump_dir: Path | None,
) -> tuple[list[BatchJob], list[str]]:
    jobs: list[BatchJob] = []
    warnings: list[str] = []
    conversation_files = sorted(
        p
        for p in conv_root.rglob("*.json*")
        if p.is_file()
        and p.suffix in {".json", ".jsonl"}
        and not p.name.endswith("_mcq_assembled.jsonl")
        and "_api_" not in p.stem
    )
    for conv_path in conversation_files:
        try:
            rel = conv_path.relative_to(conv_root)
        except ValueError:
            # Should not happen, but skip just in case.
            continue
        prefix = conv_path.stem
        mcq_dir = (mcq_root / rel.parent).resolve()
        available = mcq_dir / f"{prefix}_api_available.jsonl"
        params = mcq_dir / f"{prefix}_api_params.jsonl"
        param_values = mcq_dir / f"{prefix}_api_param_values.jsonl"
        missing = [path for path in (available, params, param_values) if not path.exists()]
        skip_mcq = global_skip_mcq
        missing_dump = None
        if missing:
            if require_mcq and not keep_missing_mcq:
                warnings.append(
                    f"[WARN] Skip '{prefix}' (relative {rel}) because missing: "
                    + ", ".join(str(m) for m in missing)
                )
                continue
            warnings.append(
                f"[WARN] Missing MCQs for '{prefix}' (relative {rel}); assembling without MCQs."
            )
            available = params = param_values = None
            skip_mcq = True
            if missing_mcq_dump_dir:
                missing_dump = (missing_mcq_dump_dir / rel).with_suffix(".txt")
        text_output = (mcq_dir / f"{prefix}_mcq_assembled.txt") if include_text else None
        jobs.append(
            BatchJob(
                prefix=prefix,
                conversation=conv_path,
                available=available,
                params=params,
                param_values=param_values,
                output=mcq_dir / f"{prefix}_mcq_assembled.jsonl",
                text_output=text_output,
                skip_mcq=skip_mcq,
                missing_mcq_dump=missing_dump,
            )
        )


    return jobs, warnings


def run_batch(conv_root: Path, mcq_root: Path, args: argparse.Namespace) -> None:
    if not conv_root.exists() or not conv_root.is_dir():
        raise SystemExit(f"Conversation root not found: {conv_root}")
    if not mcq_root.exists() or not mcq_root.is_dir():
        raise SystemExit(f"MCQ root not found: {mcq_root}")

    # Load function stats for MCQ distractor descriptions
    stats_meta: dict[str, dict] | None = None
    if args.stats and args.stats.exists():
        try:
            stats_meta = json.loads(args.stats.read_text(encoding="utf-8"))
            print(f"[INFO] Loaded {len(stats_meta)} function profiles from {args.stats}")
        except Exception as exc:
            print(f"[WARN] Failed to load stats from {args.stats}: {exc}")
            stats_meta = None

    # Resolve end-tag behavior:
    # - None: auto-derive from start tag (if any)
    # - "": explicitly disable end tag
    # - otherwise: use provided string
    task_select_end_tag = (
        _auto_derive_end_tag(args.task_select_tag)
        if args.task_select_end_tag is None
        else args.task_select_end_tag
    )

    jobs, warnings = discover_batch_jobs(
        conv_root,
        mcq_root,
        not args.no_text_output,
        require_mcq=not args.passthrough_only and not args.keep_missing_mcq,
        global_skip_mcq=args.passthrough_only,
        keep_missing_mcq=args.keep_missing_mcq,
        missing_mcq_dump_dir=args.missing_mcq_dump_dir,
    )
    for msg in warnings:
        print(msg)
    if not jobs:
        print("[INFO] No valid conversation/MCQ combinations discovered; nothing to do.")
        return

    max_workers = max(1, args.workers or 1)
    print(f"[INFO] Launching batch assembly for {len(jobs)} files (workers={max_workers}).")
    successes = 0
    failures = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                assemble_to_outputs,
                job.conversation,
                job.available,
                job.params,
                job.param_values,
                job.output,
                job.text_output,
                args.show_function_name,
                job.skip_mcq,
                args.task_select_tag,
                task_select_end_tag,
                args.answer_redact,
                args.mcq_tag,
                args.emit_no_mcq_tag,
                args.random_alias_per_record,
                args.random_alias_seed,
                args.mcq_subsample,
                args.mcq_subsample_seed,
                args.split_shards,
                args.skip_metadata,
                args.skip_question_quality,
                args.skip_response_quality,
                stats_meta,
                job.missing_mcq_dump,
            ): job
            for job in jobs
        }
        for future in as_completed(future_map):
            job = future_map[future]
            try:
                count, kept_mcq, mcq_total = future.result()
            except Exception as exc:
                failures += 1
                print(f"[ERROR] Failed to assemble '{job.prefix}' ({job.conversation}): {exc}")
            else:
                successes += 1
                text_msg = f" + {job.text_output}" if job.text_output else ""
                print(
                    f"[INFO] Built {job.output}{text_msg} | records={count}, mcq_records={kept_mcq}, mcq_entries={mcq_total}"
                )

    print(
        f"[INFO] Batch complete. Successful: {successes}. Failed: {failures}. Total: {len(jobs)}."
    )

def main():
    args = parse_args()
    conv_root = args.conv_root
    mcq_root = args.mcq_root or conv_root
    run_batch(conv_root, mcq_root, args)


if __name__ == "__main__":
    main()

