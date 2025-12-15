#!/usr/bin/env python3
"""
Streaming aggregation of tool/function usage statistics from Toucan jsonl files.

Outputs:
- CSV (function_name,count)
- JSON (function metadata: name/description/parameters)
- Optional alias map (raw -> alias) if requested

Design goals:
- Stream/merge per file; do not load all records into memory.
- Be tolerant of stringified JSON in fields (messages, available_tools, metadata).
- Support alias-map backmapping so record-scope aliases can be mapped to canonical names.
- Optional dual-source mode: collect schema from canonical/raw data, counts from obfuscated data.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.utils.function_alias import build_alias_map, load_alias_map, save_alias_map


def iter_jsonl_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.jsonl"))


def _json_maybe_load(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (dict, list)):
        return obj
    if isinstance(obj, str):
        obj = obj.strip()
        if not obj:
            return None
        try:
            return json.loads(obj)
        except Exception:
            return None
    return None


def _unalias(name: str | None, alias_rev: Dict[str, str] | None) -> str | None:
    if not name:
        return name
    if not alias_rev:
        return name
    return alias_rev.get(name, name)


def _extract_available(
    record: Dict[str, Any], meta_store: Dict[str, Dict[str, Any]], alias_rev: Dict[str, str] | None = None
) -> None:
    tools = _json_maybe_load(record.get("available_tools"))
    if not isinstance(tools, list):
        return
    for tool in tools:
        func = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(func, dict):
            continue
        name = _unalias(func.get("name"), alias_rev)
        if not name:
            continue
        if name not in meta_store:
            meta_store[name] = {
                "name": name,
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            }


def _extract_metadata(
    record: Dict[str, Any], meta_store: Dict[str, Dict[str, Any]], alias_rev: Dict[str, str] | None = None
) -> None:
    metadata = _json_maybe_load(record.get("metadata"))
    if not isinstance(metadata, dict):
        return
    servers = metadata.get("mcp_servers")
    if not isinstance(servers, list):
        return
    for server in servers:
        if not isinstance(server, dict):
            continue
        resp = server.get("remote_server_response")
        if not isinstance(resp, dict):
            continue
        tools = resp.get("tools")
        if not isinstance(tools, list):
            continue
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = _unalias(tool.get("name"), alias_rev)
            if not name:
                continue
            if name not in meta_store:
                meta_store[name] = {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                }


def _count_messages(record: Dict[str, Any], counter: Counter, alias_rev: Dict[str, str] | None = None) -> None:
    messages = _json_maybe_load(record.get("messages"))
    if not isinstance(messages, list):
        return
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        fc = msg.get("function_call")
        if isinstance(fc, dict):
            name = _unalias(fc.get("name"), alias_rev)
            if name:
                counter[name] += 1
        if msg.get("role") == "function":
            name = _unalias(msg.get("name"), alias_rev)
            if name:
                counter[name] += 1


def extract_functions(record: dict) -> tuple[list[str], dict[str, dict]]:
    # Deprecated helper (kept for API compatibility if used elsewhere).
    counter: Counter = Counter()
    meta: Dict[str, Dict[str, Any]] = {}
    _extract_available(record, meta)
    _extract_metadata(record, meta)
    _count_messages(record, counter)
    funcs = []
    for name, cnt in counter.items():
        funcs.extend([name] * cnt)
    return funcs, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate function usage stats.")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Root directory or jsonl file under Toucan-1.5M (obfuscated or raw).",
    )
    parser.add_argument(
        "--canonical-meta-input",
        type=Path,
        default=None,
        help="Optional raw/global-scope source to collect canonical schema/meta (no alias map).",
    )
    parser.add_argument(
        "--canonical-workers",
        type=int,
        default=1,
        help="Workers for canonical-meta pass (default: 1).",
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
        help="Number of parallel workers for processing obfuscated files (default: 1).",
    )
    parser.add_argument(
        "--alias-output",
        type=Path,
        default=None,
        help="Optional path to write function_alias.json (built from collected names).",
    )
    parser.add_argument(
        "--alias-existing",
        type=Path,
        default=None,
        help="Optional existing alias map to extend when writing --alias-output.",
    )
    parser.add_argument(
        "--alias-map-dir",
        type=Path,
        default=None,
        help="Directory containing per-file alias map logs (<src>.alias_map.jsonl) from obfuscate_jsonl.py.",
    )
    parser.add_argument(
        "--alias-map-suffix",
        type=str,
        default=".alias_map.jsonl",
        help="Suffix appended to source jsonl filename inside --alias-map-dir (default: .alias_map.jsonl).",
    )
    parser.add_argument(
        "--skip-schema",
        action="store_true",
        help="Skip available_tools/metadata schema extraction on obfuscated pass (keeps counts only, lower memory).",
    )
    return parser.parse_args()


def _next_alias_rev(alias_iter) -> Dict[str, str] | None:
    if alias_iter is None:
        return None
    try:
        line = next(alias_iter)
    except StopIteration:
        return None
    line = line.strip()
    if not line:
        return {}
    try:
        payload = json.loads(line)
    except Exception:
        return {}
    amap = payload.get("alias_map") if isinstance(payload, dict) else None
    if not isinstance(amap, dict):
        return {}
    rev = {v: k for k, v in amap.items() if isinstance(k, str) and isinstance(v, str)}
    return rev


def _alias_iter_for(file: Path, alias_dir: Path | None, alias_suffix: str):
    if not alias_dir:
        return None, None
    alias_path = alias_dir / f"{file.name}{alias_suffix}"
    if not alias_path.exists():
        return None, alias_path
    fh = alias_path.open("r", encoding="utf-8")

    def _iter():
        for line in fh:
            yield line
        fh.close()

    return _iter(), alias_path


def process_file(
    file: Path,
    alias_dir: Path | None,
    alias_suffix: str,
    skip_schema: bool,
) -> tuple[Counter[str], dict[str, dict], int, Path | None]:
    local_counter: Counter[str] = Counter()
    local_meta: dict[str, dict] = {}
    total = 0

    alias_iter, alias_path = _alias_iter_for(file, alias_dir, alias_suffix)

    with file.open("r", encoding="utf-8") as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except Exception:
                continue
            alias_rev = _next_alias_rev(alias_iter)
            if not skip_schema:
                _extract_available(record, local_meta, alias_rev)
                _extract_metadata(record, local_meta, alias_rev)
            _count_messages(record, local_counter, alias_rev)

    return local_counter, local_meta, total, alias_path


def _run_pass(
    files: list[Path],
    *,
    alias_dir: Path | None,
    alias_suffix: str,
    skip_schema: bool,
    workers: int,
    label: str,
) -> tuple[Counter[str], dict[str, dict], int]:
    counter: Counter[str] = Counter()
    meta_store: dict[str, dict] = {}
    total_records = 0

    workers = max(1, workers)
    print(f"[INFO] [{label}] {len(files)} file(s) with {workers} worker(s).")

    if workers == 1 or len(files) == 1:
        for file in files:
            print(f"[INFO] [{label}] Processing {file}")
            local_counter, local_meta, local_total, alias_path = process_file(
                file, alias_dir, alias_suffix, skip_schema
            )
            if alias_path and not alias_path.exists():
                print(f"[WARN] [{label}] Alias map not found for {file}: {alias_path}")
            counter.update(local_counter)
            for name, info in local_meta.items():
                if name not in meta_store:
                    meta_store[name] = info
            total_records += local_total
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_file, file, alias_dir, alias_suffix, skip_schema): file
                for file in files
            }
            for future in as_completed(futures):
                file = futures[future]
                try:
                    local_counter, local_meta, local_total, alias_path = future.result()
                except Exception as exc:
                    print(f"[WARN] [{label}] Failed processing {file}: {exc}")
                    continue
                if alias_path and not alias_path.exists():
                    print(f"[WARN] [{label}] Alias map not found for {file}: {alias_path}")
                print(f"[INFO] [{label}] Finished {file} (records: {local_total})")
                counter.update(local_counter)
                for name, info in local_meta.items():
                    if name not in meta_store:
                        meta_store[name] = info
                total_records += local_total

    print(f"[INFO] [{label}] Processed {total_records} records.")
    return counter, meta_store, total_records


def main() -> None:
    args = parse_args()

    # Canonical/meta pass (schema only, no alias map, no skip_schema)
    canonical_meta: dict[str, dict] = {}
    if args.canonical_meta_input:
        canonical_files = iter_jsonl_files(args.canonical_meta_input)
        if not canonical_files:
            raise SystemExit(f"No jsonl files found under {args.canonical_meta_input}")
        _, canonical_meta, _ = _run_pass(
            canonical_files,
            alias_dir=None,
            alias_suffix=args.alias_map_suffix,
            skip_schema=False,
            workers=args.canonical_workers,
            label="canonical",
        )

    # Obfuscated/primary pass (counts + optional schema via alias maps)
    files = iter_jsonl_files(args.input)
    if not files:
        raise SystemExit(f"No jsonl files found under {args.input}")

    counter: Counter[str]
    obf_meta: dict[str, dict]
    counter, obf_meta, total_records = _run_pass(
        files,
        alias_dir=args.alias_map_dir,
        alias_suffix=args.alias_map_suffix,
        skip_schema=args.skip_schema,
        workers=args.workers,
        label="obf",
    )

    # Merge meta: canonical first (authoritative), then fill missing from obf pass
    meta_store: dict[str, dict] = {}
    if canonical_meta:
        meta_store.update(canonical_meta)
    for name, info in obf_meta.items():
        if name not in meta_store:
            meta_store[name] = info

    items = counter.most_common(args.top) if args.top else counter.most_common()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["function_name", "count"])
        for name, count in items:
            writer.writerow([name, count])
    args.meta_output.parent.mkdir(parents=True, exist_ok=True)
    with args.meta_output.open("w", encoding="utf-8") as metaj:
        json.dump(meta_store, metaj, ensure_ascii=False, indent=2)

    print(f"Processed {total_records} records from {len(files)} files (obf pass).")
    print(
        f"Unique functions: {len(counter)}. Count CSV: {args.output}. Metadata JSON: {args.meta_output}."
    )

    if args.alias_output:
        existing = {}
        if args.alias_existing:
            try:
                existing = load_alias_map(args.alias_existing)
            except FileNotFoundError:
                print(f"[WARN] Existing alias map not found: {args.alias_existing}")
            except ValueError as exc:
                print(f"[WARN] Failed to load existing alias map: {exc}")
        alias_map = build_alias_map(counter.keys(), existing)
        save_alias_map(alias_map, args.alias_output)
        print(f"[INFO] Alias map written to {args.alias_output}")


if __name__ == "__main__":
    main()
