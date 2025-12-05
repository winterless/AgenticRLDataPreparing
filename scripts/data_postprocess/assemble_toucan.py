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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from textwrap import indent

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.data_postprocess.render_toucan_text import convert_jsonl_to_txt  # noqa: E402
from scripts.utils.has_utils import load_jsonl, parse_arguments  # noqa: E402

MODE_ORDER = ("available", "params", "param_values")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class BatchJob:
    prefix: str
    conversation: Path
    available: Path
    params: Path
    param_values: Path
    output: Path
    text_output: Path | None


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


def assemble_to_outputs(
    conversation_path: Path,
    available_path: Path,
    params_path: Path,
    param_values_path: Path,
    output_path: Path,
    text_path: Path | None,
    reveal_answers: bool,
    text_reveal_answers: bool,
    show_function_name: bool,
) -> tuple[int, int]:
    if not conversation_path.exists():
        raise FileNotFoundError(f"Conversation file missing: {conversation_path}")
    for label, path in [
        ("available", available_path),
        ("params", params_path),
        ("param_values", param_values_path),
    ]:
        if not path or not path.exists():
            raise FileNotFoundError(f"{label} MCQ missing: {path}")

    records = load_records(conversation_path)
    mcq_index, mcq_total = build_mcq_index([available_path, params_path, param_values_path])
    to_emit = records

    output_path.parent.mkdir(parents=True, exist_ok=True)
    text_fh = text_path.open("w", encoding="utf-8") if text_path else None
    try:
        with output_path.open("w", encoding="utf-8") as fh:
            for record in to_emit:
                base_text = assemble_record(
                    record, mcq_index, reveal_answers, show_function_name
                )
                payload = {
                    "uuid": record.get("uuid") or record.get("record_uuid"),
                    "text": base_text,
                }
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

                if text_fh:
                    pretty_text = (
                        base_text
                        if text_reveal_answers == reveal_answers
                        else assemble_record(
                            record, mcq_index, text_reveal_answers, show_function_name
                        )
                    )
                    text_fh.write(pretty_text)
                    if not pretty_text.endswith("\n"):
                        text_fh.write("\n")
                    text_fh.write("\n")
    finally:
        if text_fh:
            text_fh.close()

    return len(to_emit), mcq_total


def discover_batch_jobs(
    conv_root: Path, mcq_root: Path, include_text: bool
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
        if missing:
            warnings.append(
                f"[WARN] Skip '{prefix}' (relative {rel}) because missing: "
                + ", ".join(str(m) for m in missing)
            )
            continue
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
            )
        )
    return jobs, warnings


def run_batch(conv_root: Path, mcq_root: Path, args: argparse.Namespace) -> None:
    if not conv_root.exists() or not conv_root.is_dir():
        raise SystemExit(f"Conversation root not found: {conv_root}")
    if not mcq_root.exists() or not mcq_root.is_dir():
        raise SystemExit(f"MCQ root not found: {mcq_root}")

    jobs, warnings = discover_batch_jobs(conv_root, mcq_root, not args.no_text_output)
    for msg in warnings:
        print(msg)
    if not jobs:
        print("[INFO] No valid conversation/MCQ combinations discovered; nothing to do.")
        return

    max_workers = max(1, args.workers or 1)
    print(f"[INFO] Launching batch assembly for {len(jobs)} files (workers={max_workers}).")
    successes = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                assemble_to_outputs,
                job.conversation,
                job.available,
                job.params,
                job.param_values,
                job.output,
                job.text_output,
                args.reveal_answers,
                True,
                args.show_function_name,
            ): job
            for job in jobs
        }
        for future in as_completed(future_map):
            job = future_map[future]
            try:
                count, mcq_total = future.result()
            except Exception as exc:
                failures += 1
                print(f"[ERROR] Failed to assemble '{job.prefix}' ({job.conversation}): {exc}")
            else:
                successes += 1
                text_msg = f" + {job.text_output}" if job.text_output else ""
                print(
                    f"[INFO] Built {job.output}{text_msg} | records={count}, mcq_entries={mcq_total}"
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

