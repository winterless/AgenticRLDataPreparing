#!/usr/bin/env python3
"""
Aggregate tool/function usage statistics from Toucan jsonl files.

Example:
    python scripts/analysis/function_stats.py -i Toucan-1.5M/Toucan-1.5M -o stats/function_stats.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def iter_jsonl_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.jsonl"))


def extract_functions(record: dict):
    funcs: list[str] = []
    tool_meta: dict[str, dict] = {}
    available = record.get("available_tools")
    try:
        tools = json.loads(available) if isinstance(available, str) else available
    except (TypeError, json.JSONDecodeError):
        tools = None
    if isinstance(tools, list):
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name")
            if name:
                funcs.append(name)
                tool_meta.setdefault(name, func)
    messages = record.get("messages")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            messages = None
    if isinstance(messages, list):
        for msg in messages:
            fc = msg.get("function_call")
            if fc and fc.get("name"):
                funcs.append(fc["name"])
    metadata = record.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = None
    if isinstance(metadata, dict):
        server_info = metadata.get("mcp_servers") or []
        for server in server_info:
            resp = server.get("remote_server_response", {})
            for tool in resp.get("tools", []):
                name = tool.get("name")
                if name:
                    funcs.append(name)
                    tool_meta.setdefault(name, tool)
    return funcs, tool_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate function usage stats.")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Root directory or jsonl file under Toucan-1.5M.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("function_stats.csv"),
        help="CSV file to save aggregated counts (default: function_stats.csv).",
    )
    parser.add_argument(
        "--meta-output",
        type=Path,
        default=Path("function_meta.json"),
        help="JSON file to store function metadata (default: function_meta.json).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Only keep top-N most frequent functions.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing files.",
    )
    return parser.parse_args()


def process_file(file: Path) -> tuple[Counter[str], dict[str, dict], int]:
    local_counter: Counter[str] = Counter()
    local_meta: dict[str, dict] = {}
    local_total = 0
    with file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            local_total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            funcs, meta = extract_functions(record)
            for name in funcs:
                local_counter[name] += 1
            for name, info in meta.items():
                local_meta.setdefault(name, info)
    return local_counter, local_meta, local_total


def main() -> None:
    args = parse_args()
    files = iter_jsonl_files(args.input)
    if not files:
        raise SystemExit(f"No jsonl files found under {args.input}")

    counter: Counter[str] = Counter()
    total_records = 0
    meta_store: dict[str, dict] = {}

    workers = max(1, args.workers)
    print(f"[INFO] Found {len(files)} jsonl files. Processing with {workers} worker(s).")
    if workers == 1 or len(files) == 1:
        for file in files:
            print(f"[INFO] Processing {file}")
            local_counter, local_meta, local_total = process_file(file)
            counter.update(local_counter)
            total_records += local_total
            for name, info in local_meta.items():
                meta_store.setdefault(name, info)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_file, file): file for file in files}
            for future in as_completed(futures):
                file = futures[future]
                try:
                    local_counter, local_meta, local_total = future.result()
                except Exception as exc:
                    print(f"[WARN] Failed processing {file}: {exc}")
                    continue
                print(f"[INFO] Finished {file} (records: {local_total})")
                counter.update(local_counter)
                total_records += local_total
                for name, info in local_meta.items():
                    meta_store.setdefault(name, info)

    items = counter.most_common(args.top) if args.top else counter.most_common()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["function_name", "count"])
        for name, count in items:
            writer.writerow([name, count])
    with args.meta_output.open("w", encoding="utf-8") as metaj:
        json.dump(meta_store, metaj, ensure_ascii=False, indent=2)

    print(f"Processed {total_records} records from {len(files)} files.")
    print(
        f"Unique functions: {len(counter)}. Count CSV: {args.output}. Metadata JSON: {args.meta_output}."
    )


if __name__ == "__main__":
    main()

