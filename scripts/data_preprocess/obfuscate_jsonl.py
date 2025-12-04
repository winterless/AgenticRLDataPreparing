#!/usr/bin/env python3
"""
Rewrite jsonl data so that all function names are replaced by aliases.

Example:
    python scripts/data_preprocess/obfuscate_jsonl.py \
        -i data/toucan2.jsonl \
        -o data/toucan2_obf.jsonl \
        --alias stats/function_alias.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.utils.function_alias import load_alias_map, apply_alias

_WORKER_ALIAS: dict[str, str] | None = None


def obfuscate_file(src: Path, dst: Path, alias_map: dict[str, str]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fh_in, dst.open("w", encoding="utf-8") as fh_out:
        for line in fh_in:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            new_record = mask_record(record, alias_map)
            json.dump(new_record, fh_out, ensure_ascii=False)
            fh_out.write("\n")


def worker_init(alias_path: str):
    global _WORKER_ALIAS
    _WORKER_ALIAS = load_alias_map(Path(alias_path))


def worker_task(task: tuple[str, str, str]) -> str:
    global _WORKER_ALIAS
    src_str, dst_str, alias_path = task
    if _WORKER_ALIAS is None:
        _WORKER_ALIAS = load_alias_map(Path(alias_path))
    obfuscate_file(Path(src_str), Path(dst_str), _WORKER_ALIAS)
    return dst_str


def parse_json_field(value):
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed, True
        except json.JSONDecodeError:
            return value, False
    return value, False


def dump_json_field(value, was_string: bool):
    if was_string:
        return json.dumps(value, ensure_ascii=False)
    return value


def mask_available(record: dict, alias_map: dict[str, str]) -> None:
    tools, was_string = parse_json_field(record.get("available_tools"))
    if isinstance(tools, list):
        for tool in tools:
            func = tool.get("function")
            if isinstance(func, dict) and func.get("name"):
                func["name"] = apply_alias(func["name"], alias_map)
    record["available_tools"] = dump_json_field(tools, was_string)


def mask_messages(record: dict, alias_map: dict[str, str]) -> None:
    raw_messages = record.get("messages")
    messages, was_string = parse_json_field(raw_messages)
    if isinstance(messages, list):
        for msg in messages:
            if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                msg["content"] = obfuscate_tool_declare(msg["content"], alias_map)
            fc = msg.get("function_call")
            if isinstance(fc, dict) and fc.get("name"):
                fc["name"] = apply_alias(fc["name"], alias_map)
    record["messages"] = dump_json_field(messages, was_string)


def mask_target_tools(record: dict, alias_map: dict[str, str]) -> None:
    target = record.get("target_tools")
    target_obj, was_string = parse_json_field(target)
    if isinstance(target_obj, list):
        record["target_tools"] = dump_json_field(
            [apply_alias(name, alias_map) for name in target_obj], was_string
        )
    elif isinstance(target_obj, str):
        record["target_tools"] = apply_alias(target_obj, alias_map)


def mask_metadata(record: dict, alias_map: dict[str, str]) -> None:
    metadata = record.get("metadata")
    meta_obj, was_string = parse_json_field(metadata)
    if isinstance(meta_obj, dict):
        servers = meta_obj.get("mcp_servers") or []
        if isinstance(servers, list):
            for server in servers:
                resp = server.get("remote_server_response") or {}
                tools = resp.get("tools") or []
                if isinstance(tools, list):
                    for tool in tools:
                        if isinstance(tool, dict) and tool.get("name"):
                            tool["name"] = apply_alias(tool["name"], alias_map)
    record["metadata"] = dump_json_field(meta_obj, was_string)


def mask_record(record: dict, alias_map: dict[str, str]) -> dict:
    mask_available(record, alias_map)
    mask_messages(record, alias_map)
    mask_target_tools(record, alias_map)
    mask_metadata(record, alias_map)
    if record.get("function_name"):
        record["function_name"] = apply_alias(record["function_name"], alias_map)
    return record


def obfuscate_tool_declare(content: str, alias_map: dict[str, str]) -> str:
    marker = "<|im_middle|>"
    end_tag = "<|im_end|>"
    start = content.find(marker)
    if start == -1:
        return content
    start_payload = start + len(marker)
    end_payload = content.find(end_tag, start_payload)
    if end_payload == -1:
        return content
    raw_payload = content[start_payload:end_payload]
    try:
        data = json.loads(raw_payload)
    except json.JSONDecodeError:
        return content
    if isinstance(data, list):
        changed = False
        for tool in data:
            func = tool.get("function")
            if isinstance(func, dict) and func.get("name"):
                new_name = apply_alias(func["name"], alias_map)
                if new_name != func["name"]:
                    func["name"] = new_name
                    changed = True
        if changed:
            encoded = json.dumps(data, ensure_ascii=False)
            return content[:start_payload] + encoded + content[end_payload:]
    return content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply function aliases to jsonl data.")
    parser.add_argument(
        "-i", "--input", type=Path, required=True, help="Source jsonl file or directory."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Destination jsonl file or directory.",
    )
    parser.add_argument("--alias", type=Path, required=True, help="function_alias.json path.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes when input is a directory (default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alias_map = load_alias_map(args.alias)

    if args.input.is_file():
        if args.output.is_dir():
            dst = args.output / args.input.name
        else:
            dst = args.output
        dst.parent.mkdir(parents=True, exist_ok=True)
        obfuscate_file(args.input, dst, alias_map)
        print(f"[INFO] Wrote obfuscated jsonl to {dst}")
        return

    if not args.input.exists():
        raise SystemExit(f"Input directory not found: {args.input}")

    src_files = sorted(args.input.rglob("*.jsonl"))
    if not src_files:
        raise SystemExit(f"No jsonl files found under {args.input}")

    if args.workers <= 1:
        for src in src_files:
            rel = src.relative_to(args.input)
            dst = args.output / rel
            obfuscate_file(src, dst, alias_map)
            print(f"[INFO] Obfuscated {src} -> {dst}")
        return

    print(f"[INFO] Obfuscating {len(src_files)} files with {args.workers} workers.")
    tasks = []
    for src in src_files:
        rel = src.relative_to(args.input)
        dst = args.output / rel
        tasks.append((str(src), str(dst), str(args.alias)))

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=worker_init,
        initargs=(str(args.alias),),
    ) as executor:
        future_map = {executor.submit(worker_task, task): task for task in tasks}
        for future in as_completed(future_map):
            src_str, dst_str, _ = future_map[future]
            try:
                future.result()
                print(f"[INFO] Obfuscated {src_str} -> {dst_str}")
            except Exception as exc:
                print(f"[WARN] Failed to obfuscate {src_str}: {exc}")


if __name__ == "__main__":
    main()

