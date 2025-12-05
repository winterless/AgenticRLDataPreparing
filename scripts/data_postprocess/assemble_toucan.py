#!/usr/bin/env python3
"""
Assemble obfuscated Toucan trajectories with HAS-API MCQs into training-ready text.

The script reads the canonical `toucan*.jsonl` artifacts under `data/`, injects the
available/params/param_values MCQs at each function-call turn, and writes a JSONL
file whose `text` field contains the concatenated prompt described in README.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from textwrap import indent

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.utils.has_utils import load_jsonl, parse_arguments  # noqa: E402
from scripts.data_postprocess.render_toucan_text import (  # noqa: E402
    convert_jsonl_to_txt,
)

MODE_ORDER = ("available", "params", "param_values")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitch Toucan trajectories with MCQs.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Directory containing toucan*.json(l) files (default: data/).",
    )
    parser.add_argument(
        "--conversation",
        type=Path,
        default=None,
        help="Path to the obfuscated trajectory json/jsonl (default: data/toucan.jsonl).",
    )
    parser.add_argument(
        "--mcq-available",
        type=Path,
        default=None,
        help="Path to available-mode MCQ jsonl (default: data/toucan_api_available.jsonl).",
    )
    parser.add_argument(
        "--mcq-params",
        type=Path,
        default=None,
        help="Path to params-mode MCQ jsonl (default: data/toucan_api_params.jsonl).",
    )
    parser.add_argument(
        "--mcq-param-values",
        type=Path,
        default=None,
        help="Path to param_values-mode MCQ jsonl (default: data/toucan_api_param_values.jsonl).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "toucan_mcq_assembled.jsonl",
        help="Output path for assembled JSONL (default: data/toucan_mcq_assembled.jsonl).",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=None,
        help="Optional pretty text output path (default: same as --output but .txt).",
    )
    parser.add_argument(
        "--no-text-output",
        action="store_true",
        help="Skip emitting the pretty text companion file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of records to materialize.",
    )
    parser.add_argument(
        "--reveal-answers",
        action="store_true",
        help="If set, append correct answer text after each MCQ block.",
    )
    parser.add_argument(
        "--show-function-name",
        action="store_true",
        help="Include the target function name in MCQ headers (default hides it).",
    )
    return parser.parse_args()


def pick_default(data_dir: Path, stem: str) -> Path | None:
    for suffix in (".jsonl", ".json"):
        candidate = data_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


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


def format_tools(record: dict) -> list[str]:
    raw = parse_json_like(record.get("available_tools"))
    tools = raw if isinstance(raw, list) else []
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


def format_mcq_block(entry: dict, reveal_answer: bool, show_function_name: bool) -> str:
    header_bits = [f"[MCQ:{entry.get('mode')}"]
    header_bits.append(f"|msg={entry.get('message_index')}")
    if show_function_name and entry.get("function_name"):
        header_bits.append(f"|function={entry['function_name']}")
    header_bits.append("]")
    header = "".join(header_bits)
    lines = [
        header,
        "Question:",
        indent(entry.get("question", ""), "  "),
        "Options:",
        indent(format_options(entry.get("options", [])), "  "),
    ]
    if reveal_answer and entry.get("answer"):
        lines.append(f"Answer: the answer is {entry['answer']}")
    return "\n".join(lines)


def assemble_record(record: dict, mcqs, reveal_answer: bool, show_function_name: bool) -> str:
    parts: list[str] = []
    uuid = record.get("uuid") or record.get("record_uuid") or "unknown"
    question = record.get("question", "").strip()
    parts.append(f"=== Record | uuid={uuid} ===")
    if question:
        parts.append("Question:")
        parts.append(indent(question, "  "))
    tool_lines = format_tools(record)
    if tool_lines:
        parts.append("Available tools:")
        parts.extend(tool_lines)
    messages = ensure_messages(record)
    parts.append("Messages:")

    awaiting_answer = False
    per_record_mcq = mcqs.get(str(uuid), {})

    for idx, message in enumerate(messages):
        role = message.get("role", "unknown")
        function_call = message.get("function_call")
        if function_call:
            injections = per_record_mcq.get(idx, {})
            for mode in MODE_ORDER:
                for entry in injections.get(mode, []):
                    parts.append(
                        indent(
                            format_mcq_block(entry, reveal_answer, show_function_name),
                            "    ",
                        )
                    )
            name = function_call.get("name", "unknown")
            args = parse_arguments(function_call) or function_call.get("arguments")
            parts.append(
                indent(f"assistant (function_call: {name}) @msg={idx}:", "  ")
            )
            parts.append(indent(format_json_block(args), "    "))
            awaiting_answer = True
            continue

        if role == "function":
            func_name = message.get("name", "function")
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
        rendered = (
            yaml.safe_dump(targets, allow_unicode=True, sort_keys=False).strip()
            if isinstance(targets, (dict, list))
            else str(targets)
        )
        parts.append("Target tools:")
        parts.append(indent(rendered, "  "))
    if record.get("question_quality_assessment"):
        parsed = parse_json_like(record["question_quality_assessment"])
        parts.append("Question quality assessment:")
        parts.append(indent(format_json_block(parsed), "  "))
    if record.get("response_quality_assessment"):
        parsed = parse_json_like(record["response_quality_assessment"])
        parts.append("Response quality assessment:")
        parts.append(indent(format_json_block(parsed), "  "))
    if record.get("metadata"):
        parsed = parse_json_like(record["metadata"])
        parts.append("Metadata:")
        parts.append(indent(format_json_block(parsed), "  "))

    return "\n".join(parts).strip() + "\n"


def main():
    args = parse_args()
    conversation_path = args.conversation or pick_default(args.data_dir, "toucan")
    available_path = args.mcq_available or pick_default(args.data_dir, "toucan_api_available")
    params_path = args.mcq_params or pick_default(args.data_dir, "toucan_api_params")
    param_values_path = args.mcq_param_values or pick_default(
        args.data_dir, "toucan_api_param_values"
    )

    if not conversation_path or not conversation_path.exists():
        raise SystemExit("Conversation json/jsonl not found; specify --conversation.")
    for label, path in [
        ("available", available_path),
        ("params", params_path),
        ("param_values", param_values_path),
    ]:
        if not path or not path.exists():
            raise SystemExit(f"{label} MCQ file missing; pass --mcq-{label.replace('_', '-')}.")

    records = load_records(conversation_path)
    mcq_index, mcq_total = build_mcq_index([available_path, params_path, param_values_path])

    to_emit = records[: args.limit] if args.limit else records
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text_path = None
    if not args.no_text_output:
        if args.text_output:
            text_path = args.text_output
        elif args.output.suffix:
            text_path = args.output.with_suffix(".txt")
        else:
            text_path = args.output.with_name(args.output.name + ".txt")
        text_path.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as fh:
        for record in to_emit:
            text = assemble_record(
                record, mcq_index, args.reveal_answers, args.show_function_name
            )
            payload = {
                "uuid": record.get("uuid") or record.get("record_uuid"),
                "text": text,
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    if text_path:
        convert_jsonl_to_txt(args.output, text_path, args.limit)

    text_msg = f" and {text_path}" if text_path else ""
    print(
        f"[INFO] Wrote {len(to_emit)} assembled samples to {args.output}{text_msg}. "
        f"MCQ entries consumed: {mcq_total}."
    )


if __name__ == "__main__":
    main()

