#!/usr/bin/env python3
"""
Make Toucan jsonl file human readable

Example:
    python scripts/analysis/pretty_toucan.py -i data/demo/toucan.jsonl -n 1 > data/demo/toucan.txt
"""

import argparse
import json
from pathlib import Path
from textwrap import indent

import yaml

alias_reverse: dict[str, str] | None = None


def set_alias_map(path: Path | None):
    global alias_reverse
    if not path:
        alias_reverse = None
        return
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Alias map must be a JSON object")
    alias_reverse = {alias: original for original, alias in data.items()}


def show_name(name: str | None) -> str:
    if not name:
        return "unknown"
    if alias_reverse and name in alias_reverse:
        original = alias_reverse[name]
        if original != name:
            return f"{name} ({original})"
    return name


def parse_tool_declare(content: str) -> str:
    start_mid = content.find("<|im_middle|>")
    end_tag = "<|im_end|>"
    end_idx = content.find(end_tag, start_mid)
    if start_mid == -1 or end_idx == -1:
        return content.strip()
    raw = content[start_mid + len("<|im_middle|>") : end_idx].strip()
    raw_unescaped = bytes(raw, "utf-8").decode("unicode_escape")
    start_list = raw_unescaped.find("[")
    end_list = raw_unescaped.rfind("]")
    if start_list == -1 or end_list == -1:
        return raw_unescaped.strip()
    payload = raw_unescaped[start_list : end_list + 1]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        fixed = raw_unescaped.replace('[{","type', '[{"type')
        try:
            data = json.loads(fixed[fixed.find("[") : fixed.rfind("]") + 1])
        except json.JSONDecodeError:
            return raw_unescaped.strip()
    yaml_text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    return yaml_text.strip()


def dump_json_like(value):
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value.strip()
        return yaml.safe_dump(parsed, sort_keys=False, allow_unicode=True).strip()
    elif isinstance(value, (dict, list)):
        return yaml.safe_dump(value, sort_keys=False, allow_unicode=True).strip()
    return str(value)


def format_message(entry: dict) -> str:
    role = entry.get("role", "unknown")
    if entry.get("function_call"):
        fc = entry["function_call"]
        try:
            args_obj = json.loads(fc["arguments"])
            args = json.dumps(args_obj, indent=2, ensure_ascii=False)
        except Exception:
            args = fc["arguments"]
        name = show_name(fc.get("name"))
        return f"{role} (function_call: {name}):\n{indent(args, '  ')}"
    elif entry.get("content"):
        content = entry["content"].strip()
        if role == "system" and "<|im_system|>" in content:
            yaml_block = parse_tool_declare(content)
            return f"{role} (tool_declare):\n{indent(yaml_block, '  ')}"
        return f"{role}:\n{indent(content, '  ')}"
    else:
        return f"{role}: (empty)"

def pretty_print_record(record: dict, index: int) -> str:
    header = f"=== Record {index} | uuid={record.get('uuid')} ==="
    question = (record.get("question") or "").strip()
    available_tools = record.get("available_tools")
    try:
        tools = json.loads(available_tools) if isinstance(available_tools, str) else available_tools
    except Exception:
        tools = available_tools

    lines = [header]
    if record.get("subset_name"):
        lines.append(f"Subset: {record['subset_name']}")
    if question:
        lines.append(f"Question:\n{indent(question, '  ')}")
    if tools:
        lines.append("Available tools:")
        if isinstance(tools, list):
            for tool in tools:
                func = tool.get("function", {})
                name = show_name(func.get("name"))
                desc = func.get("description", "")
                lines.append(f"  - {name}: {desc}")
        else:
            lines.append(indent(json.dumps(tools, indent=2, ensure_ascii=False), '  '))

    messages_raw = record.get("messages")
    if isinstance(messages_raw, str):
        messages = json.loads(messages_raw)
    else:
        messages = messages_raw or []
    lines.append("Messages:")
    for msg in messages:
        lines.append(indent(format_message(msg), '  '))

    if record.get("target_tools"):
        lines.append("Target tools:")
        targets = record["target_tools"]
        if isinstance(targets, list):
            formatted = [show_name(t) for t in targets]
            lines.append(indent(str(formatted), '  '))
        else:
            lines.append(indent(str(targets), '  '))

    q_quality = dump_json_like(record.get("question_quality_assessment"))
    if q_quality:
        lines.append("Question quality assessment:")
        lines.append(indent(q_quality, '  '))

    r_quality = dump_json_like(record.get("response_quality_assessment"))
    if r_quality:
        lines.append("Response quality assessment:")
        lines.append(indent(r_quality, '  '))

    metadata_raw = record.get("metadata")
    if metadata_raw:
        lines.append("Metadata:")
        try:
            metadata_obj = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
            meta_yaml = yaml.safe_dump(metadata_obj, sort_keys=False, allow_unicode=True).strip()
            lines.append(indent(meta_yaml, '  '))
        except Exception:
            lines.append(indent(str(metadata_raw), '  '))

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Pretty print out_limit jsonl records")
    parser.add_argument(
        "-i",
        "--input",
        default="out_limit1.jsonl",
        help="Path to jsonl file",
    )
    parser.add_argument(
        "-n",
        "--num-records",
        type=int,
        default=None,
        help="Print only the first N records",
    )
    parser.add_argument(
        "--alias-map",
        type=Path,
        default=None,
        help="Optional function_alias.json to display raw names alongside aliases.",
    )
    args = parser.parse_args()

    set_alias_map(args.alias_map)

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    with path.open() as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = pretty_print_record(record, idx)
            print(text)
            print()
            if args.num_records is not None and idx >= args.num_records:
                break

if __name__ == "__main__":
    main()
