#!/usr/bin/env python3
"""
Rewrite jsonl data so that all function names are replaced by aliases.

Example:
    python scripts/data_preprocess/obfuscate_jsonl.py \
        -i data/demo/toucan_raw.jsonl \
        -o data/demo/toucan.jsonl \
        --alias stats/function_alias.json
"""

from __future__ import annotations

import argparse
import json
import random
import string
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.utils.function_alias import apply_alias, load_alias_map

_WORKER_ALIAS: dict[str, str] | None = None  # used only in global-scope mode


@dataclass
class ObfuscateConfig:
    alias_scope: str  # "record" | "global"
    alias_prefix: str
    alias_length: int
    alias_seed_per_record: bool
    emit_alias_map: Path | None
    skip_tools_block: bool
    alias_path: Path


def _rand_alias_token(prefix: str, length: int, rng: random.Random, used: set[str]) -> str:
    alphabet = string.ascii_lowercase + string.digits
    while True:
        token = prefix + "".join(rng.choices(alphabet, k=length))
        if token not in used:
            used.add(token)
            return token


def build_record_alias_map(
    func_names: set[str],
    *,
    prefix: str,
    length: int,
    seed: str | None,
) -> dict[str, str]:
    rng = random.Random(seed)
    used: set[str] = set()
    return {name: _rand_alias_token(prefix, length, rng, used) for name in sorted(func_names)}


def collect_function_names(record: dict, include_tools_block: bool = True) -> set[str]:
    names: set[str] = set()

    def add_name(val):
        if isinstance(val, str) and val:
            names.add(val)

    tools, _ = parse_json_field(record.get("available_tools"))
    if isinstance(tools, list):
        for tool in tools:
            func = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(func, dict):
                add_name(func.get("name"))

    messages, _ = parse_json_field(record.get("messages"))
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") in {"function", "tool"}:
                add_name(msg.get("name"))
            fc = msg.get("function_call")
            if isinstance(fc, dict):
                add_name(fc.get("name"))
            tool_calls = msg.get("tool_calls")
            tc_obj, _tc_was = parse_json_field(tool_calls)
            if isinstance(tc_obj, list):
                for tc in tc_obj:
                    func = tc.get("function") if isinstance(tc, dict) else None
                    if isinstance(func, dict):
                        add_name(func.get("name"))
            if include_tools_block and msg.get("role") == "system" and isinstance(msg.get("content"), str):
                names.update(_collect_tools_block_names(msg["content"]))

    target, _ = parse_json_field(record.get("target_tools"))
    if isinstance(target, list):
        for t in target:
            add_name(t)
    elif isinstance(target, str):
        add_name(target)

    metadata, _ = parse_json_field(record.get("metadata"))
    if isinstance(metadata, dict):
        servers = metadata.get("mcp_servers") or []
        if isinstance(servers, list):
            for server in servers:
                if not isinstance(server, dict):
                    continue
                resp = server.get("remote_server_response") or {}
                tools_resp = resp.get("tools") or []
                if isinstance(tools_resp, list):
                    for tool in tools_resp:
                        if isinstance(tool, dict):
                            add_name(tool.get("name"))

    add_name(record.get("function_name"))
    return names


def obfuscate_file(src: Path, dst: Path, cfg: ObfuscateConfig, global_alias: dict[str, str] | None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    alias_out_fh = None
    alias_out_path = None
    if cfg.emit_alias_map:
        alias_out_path = cfg.emit_alias_map / f"{src.name}.alias_map.jsonl"
        alias_out_path.parent.mkdir(parents=True, exist_ok=True)
        alias_out_fh = alias_out_path.open("w", encoding="utf-8")

    with src.open("r", encoding="utf-8") as fh_in, dst.open("w", encoding="utf-8") as fh_out:
        for idx, line in enumerate(fh_in):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if cfg.alias_scope == "global":
                alias_map = global_alias or {}
            else:
                seed = None
                if cfg.alias_seed_per_record:
                    seed = str(record.get("uuid") or record.get("id") or record.get("record_id") or "")
                func_names = collect_function_names(record, include_tools_block=not cfg.skip_tools_block)
                alias_map = build_record_alias_map(
                    func_names,
                    prefix=cfg.alias_prefix,
                    length=cfg.alias_length,
                    seed=seed if seed else None,
                )

            new_record = mask_record(record, alias_map, rewrite_tools_block=not cfg.skip_tools_block)
            json.dump(new_record, fh_out, ensure_ascii=False)
            fh_out.write("\n")

            if alias_out_fh is not None and cfg.alias_scope == "record":
                alias_payload = {
                    "uuid": record.get("uuid"),
                    "record_id": record.get("record_id"),
                    "line_index": idx,
                    "alias_map": alias_map,
                }
                json.dump(alias_payload, alias_out_fh, ensure_ascii=False)
                alias_out_fh.write("\n")

    if alias_out_fh is not None:
        alias_out_fh.close()
        print(f"[INFO] Wrote alias map log to {alias_out_path}")


def worker_init(alias_path: str):
    global _WORKER_ALIAS
    _WORKER_ALIAS = load_alias_map(Path(alias_path))


def worker_task(task: tuple[str, str, ObfuscateConfig]) -> str:
    global _WORKER_ALIAS
    src_str, dst_str, cfg = task
    if cfg.alias_scope == "global":
        if _WORKER_ALIAS is None:
            _WORKER_ALIAS = load_alias_map(cfg.alias_path)
        obfuscate_file(Path(src_str), Path(dst_str), cfg, _WORKER_ALIAS)
    else:
        obfuscate_file(Path(src_str), Path(dst_str), cfg, None)
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


def mask_messages(record: dict, alias_map: dict[str, str], rewrite_tools_block: bool = True) -> None:
    raw_messages = record.get("messages")
    messages, was_string = parse_json_field(raw_messages)
    if isinstance(messages, list):
        for msg in messages:
            role = msg.get("role")
            if role in {"function", "tool"} and msg.get("name"):
                msg["name"] = apply_alias(msg["name"], alias_map)
            if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                content = msg["content"]
                content = obfuscate_tool_declare(content, alias_map)
                if rewrite_tools_block:
                    content = obfuscate_tools_listing(content, alias_map)
                msg["content"] = content
            fc = msg.get("function_call")
            if isinstance(fc, dict) and fc.get("name"):
                fc["name"] = apply_alias(fc["name"], alias_map)
            tool_calls = msg.get("tool_calls")
            tc_obj, tc_was_string = parse_json_field(tool_calls)
            if isinstance(tc_obj, list):
                for tc in tc_obj:
                    func = tc.get("function")
                    if isinstance(func, dict) and func.get("name"):
                        func["name"] = apply_alias(func["name"], alias_map)
            if tool_calls is not None:
                msg["tool_calls"] = dump_json_field(tc_obj, tc_was_string)
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


def _replace_text_alias(value, original: str | None, alias: str | None):
    """Replace occurrences of the original function name inside MCQ-style text."""
    if not original or not alias or original == alias:
        return value
    if isinstance(value, str):
        return value.replace(original, alias)
    if isinstance(value, list):
        return [_replace_text_alias(v, original, alias) for v in value]
    return value


def mask_record(record: dict, alias_map: dict[str, str], rewrite_tools_block: bool = True) -> dict:
    mask_available(record, alias_map)
    mask_messages(record, alias_map, rewrite_tools_block=rewrite_tools_block)
    mask_target_tools(record, alias_map)
    mask_metadata(record, alias_map)
    original_fn = record.get("function_name")
    aliased_fn = apply_alias(original_fn, alias_map) if original_fn else None
    if original_fn:
        record["function_name"] = aliased_fn
        # Also obfuscate textual mentions in MCQ-style fields so names never leak.
        for key in ("question", "answer"):
            if key in record:
                record[key] = _replace_text_alias(record[key], original_fn, aliased_fn)
        if "options" in record:
            record["options"] = _replace_text_alias(
                record.get("options"), original_fn, aliased_fn
            )
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


TOOLS_BLOCK_START = "<tools>"
TOOLS_BLOCK_END = "</tools>"


def _collect_tools_block_names(content: str) -> set[str]:
    names: set[str] = set()
    search_pos = 0
    while True:
        start = content.find(TOOLS_BLOCK_START, search_pos)
        if start == -1:
            break
        block_start = start + len(TOOLS_BLOCK_START)
        end = content.find(TOOLS_BLOCK_END, block_start)
        if end == -1:
            break
        block = content[block_start:end]
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            func = obj.get("function") if isinstance(obj, dict) else None
            if isinstance(func, dict) and func.get("name"):
                names.add(func["name"])
        search_pos = end + len(TOOLS_BLOCK_END)
    return names



def obfuscate_tools_listing(content: str, alias_map: dict[str, str]) -> str:
    """Replace raw function names that appear inside <tools>...</tools> sections."""
    search_pos = 0
    while True:
        start = content.find(TOOLS_BLOCK_START, search_pos)
        if start == -1:
            break
        block_start = start + len(TOOLS_BLOCK_START)
        end = content.find(TOOLS_BLOCK_END, block_start)
        if end == -1:
            break
        block = content[block_start:end]
        rewritten = _rewrite_tools_block(block, alias_map)
        content = content[:block_start] + rewritten + content[end:]
        search_pos = block_start + len(rewritten) + len(TOOLS_BLOCK_END)
    return content


def _rewrite_tools_block(block: str, alias_map: dict[str, str]) -> str:
    lines = block.splitlines()
    rewritten: list[str] = []
    for line in lines:
        rewritten.append(_rewrite_tool_line(line, alias_map))
    return "\n".join(rewritten)


def _rewrite_tool_line(line: str, alias_map: dict[str, str]) -> str:
    stripped = line.strip()
    if not stripped:
        return line
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return line
    func = obj.get("function")
    if isinstance(func, dict) and func.get("name"):
        func["name"] = apply_alias(func["name"], alias_map)
    prefix_len = len(line) - len(line.lstrip())
    prefix = line[:prefix_len]
    return prefix + json.dumps(obj, ensure_ascii=False)


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
        "--alias-scope",
        choices=["record", "global"],
        default="global",
        help="Use per-record random aliases (record) or a global map (global). Default: global.",
    )
    parser.add_argument(
        "--alias-prefix",
        type=str,
        default="func_",
        help="Prefix for generated aliases (record scope).",
    )
    parser.add_argument(
        "--alias-length",
        type=int,
        default=8,
        help="Length of random suffix for aliases (record scope).",
    )
    parser.add_argument(
        "--alias-seed-per-record",
        action="store_true",
        help="Derive random seed from record uuid/id for deterministic reruns (record scope).",
    )
    parser.add_argument(
        "--emit-alias-map",
        type=Path,
        default=None,
        help="If set, write per-record alias maps as jsonl into this directory (record scope).",
    )
    parser.add_argument(
        "--skip-tools-block",
        action="store_true",
        help="Do not rewrite <tools>...</tools> blocks inside system prompts.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes when input is a directory (default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.alias_scope == "global":
        alias_map = load_alias_map(args.alias)
    else:
        alias_map = None

    cfg = ObfuscateConfig(
        alias_scope=args.alias_scope,
        alias_prefix=args.alias_prefix,
        alias_length=args.alias_length,
        alias_seed_per_record=args.alias_seed_per_record,
        emit_alias_map=args.emit_alias_map,
        skip_tools_block=args.skip_tools_block,
        alias_path=args.alias,
    )

    if args.input.is_file():
        if args.output.is_dir():
            dst = args.output / args.input.name
        else:
            dst = args.output
        dst.parent.mkdir(parents=True, exist_ok=True)
        obfuscate_file(args.input, dst, cfg, alias_map)
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
            obfuscate_file(src, dst, cfg, alias_map)
            print(f"[INFO] Obfuscated {src} -> {dst}")
        return

    print(f"[INFO] Obfuscating {len(src_files)} files with {args.workers} workers.")
    tasks = []
    for src in src_files:
        rel = src.relative_to(args.input)
        dst = args.output / rel
        tasks.append((str(src), str(dst), cfg))

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
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

