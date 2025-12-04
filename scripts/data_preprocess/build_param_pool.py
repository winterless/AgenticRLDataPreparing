#!/usr/bin/env python3
"""
Aggregate parameter values & simple clusters from Toucan-style trajectories.

Usage example:
    python scripts/data_preprocess/build_param_pool.py \
        -i data/Toucan-1.5M/Toucan-1.5M \
        -s stats/function_stats.json \
        -o stats/param_pool.json
"""

from __future__ import annotations

import argparse
import json
import sys
import concurrent.futures
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


if hasattr(sys, "set_int_max_str_digits"):
    try:
        sys.set_int_max_str_digits(0)
    except Exception:
        pass


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))

from utils.has_utils import (
    infer_param_type,
    iter_function_calls,
    load_jsonl,
    load_meta,
    parse_arguments,
)


WORKER_META: dict | None = None
WORKER_MAX_VALUES: int = 0


def _worker_init(meta_path: str, max_values: int):
    global WORKER_META, WORKER_MAX_VALUES
    WORKER_META = load_meta(Path(meta_path))
    WORKER_MAX_VALUES = max_values


def _worker_process(path_str: str):
    if WORKER_META is None:
        raise RuntimeError("Worker meta not initialized.")
    builder = PoolBuilder(WORKER_META, WORKER_MAX_VALUES)
    stats = {"records": 0, "function_calls": 0, "arguments": 0}
    path = Path(path_str)
    for record in load_jsonl(path):
        stats["records"] += 1
        for _, fc in iter_function_calls(record):
            stats["function_calls"] += 1
            parsed = parse_arguments(fc)
            if not parsed:
                continue
            stats["arguments"] += len(parsed)
            builder.register(fc["name"], parsed)
    snapshot = {
        "functions": builder.functions,
        "params": builder.params,
        "types": builder.types,
    }
    return path_str, snapshot, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reusable parameter pools from tool-call trajectories."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input jsonl file or directory that contains jsonl files.",
    )
    parser.add_argument(
        "-s",
        "--stats",
        type=Path,
        required=True,
        help="function_stats.json (provides schema/type information).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("stats/param_pool.json"),
        help="Destination path for the aggregated pool (default: stats/param_pool.json).",
    )
    parser.add_argument(
        "--max-values",
        type=int,
        default=120,
        help="Maximum unique values to keep per cluster (default: 120).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Number of worker processes (default: min(8, cpu_count)).",
    )
    return parser.parse_args()


def discover_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*.jsonl") if p.is_file())


def classify_string(value: str) -> str:
    lower = value.lower()
    if lower.startswith(("http://", "https://")):
        return "string:url"
    if lower.startswith(("file://", "s3://")) or "/" in value or "\\" in value:
        return "string:path_or_uri"
    if len(value.split()) >= 6:
        return "string:long_text"
    if value.isdigit():
        return "string:numeric"
    if any(ch.isdigit() for ch in value) and any(ch.isalpha() for ch in value):
        return "string:alnum"
    return "string:general"


def cluster_label(param_type: str, value, schema: dict | None) -> str:
    schema = schema or {}
    if schema.get("enum"):
        return "enum"
    if param_type in {"integer", "number"}:
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, int):
            magnitude = len(str(abs(value))) if value != 0 else 1
            if value < 0:
                sign_bucket = "number:negative"
            elif value == 0:
                sign_bucket = "number:zero"
            else:
                sign_bucket = "number:positive"
            if magnitude <= 1:
                mag_bucket = "single_digit"
            elif magnitude <= 2:
                mag_bucket = "two_digits"
            elif magnitude <= 4:
                mag_bucket = "four_digits"
            elif magnitude <= 8:
                mag_bucket = "eight_digits"
            else:
                mag_bucket = "huge"
            return f"{sign_bucket}:{mag_bucket}"
        if isinstance(value, float):
            magnitude = abs(value)
            if magnitude == 0:
                return "number:zero"
            if magnitude < 1:
                return "number:fraction"
            if magnitude < 10:
                return "number:small"
            if magnitude < 100:
                return "number:medium"
            if magnitude < 1000:
                return "number:large"
            return "number:huge"
        return "number:other"
    if param_type == "boolean":
        return "boolean"
    if param_type == "array":
        item_type = (schema.get("items") or {}).get("type", "any")
        return f"array:{item_type}"
    if param_type == "object":
        return "object"
    if param_type == "string":
        if isinstance(value, str):
            return classify_string(value)
        return "string:non_str"
    return "unknown"


def ensure_func_param_entry(container: dict, func_name: str, param_name: str, p_type: str, required: bool):
    func_entry = container.setdefault(func_name, {"params": {}})
    param_entry = func_entry["params"].setdefault(
        param_name,
        {"type": p_type, "required": required, "observed": 0, "clusters": {}},
    )
    if p_type and not param_entry.get("type"):
        param_entry["type"] = p_type
    if required:
        param_entry["required"] = True
    return param_entry


def ensure_param_entry(container: dict, param_name: str, p_type: str):
    entry = container.setdefault(
        param_name, {"type": p_type, "observed": 0, "clusters": {}}
    )
    if p_type and not entry.get("type"):
        entry["type"] = p_type
    return entry


def ensure_type_entry(container: dict, p_type: str):
    entry = container.setdefault(p_type, {"clusters": {}, "observed": 0})
    return entry


def add_cluster_value(entry: dict, cluster_key: str, value, max_values: int):
    clusters = entry.setdefault("clusters", {})
    bucket = clusters.setdefault(cluster_key, {"count": 0, "values": [], "_seen": set()})
    bucket["count"] += 1
    if len(bucket["values"]) >= max_values:
        return
    key = json.dumps(value, sort_keys=True, ensure_ascii=False)
    if key in bucket["_seen"]:
        return
    bucket["_seen"].add(key)
    bucket["values"].append(value)


def scrub_clusters(entry: dict):
    for bucket in (entry.get("clusters") or {}).values():
        bucket.pop("_seen", None)


class PoolBuilder:
    def __init__(self, meta: dict, max_values: int):
        self.meta = meta
        self.max_values = max_values
        self.functions: dict[str, dict] = {}
        self.params: dict[str, dict] = {}
        self.types: dict[str, dict] = {}

    def register(self, func_name: str, arguments: dict):
        info = self.meta.get(func_name, {})
        params = (
            (info.get("function") or {}).get("parameters")
            or info.get("parameters")
            or info.get("input_schema")
            or {}
        )
        properties = params.get("properties") or {}
        required = set(params.get("required") or [])

        for param_name, value in arguments.items():
            schema = properties.get(param_name) or {}
            p_type = infer_param_type(schema, value, default="unknown")
            cluster = cluster_label(p_type, value, schema)

            func_entry = ensure_func_param_entry(
                self.functions, func_name, param_name, p_type, param_name in required
            )
            func_entry["observed"] += 1
            add_cluster_value(func_entry, cluster, value, self.max_values)

            param_entry = ensure_param_entry(self.params, param_name, p_type)
            param_entry["observed"] += 1
            add_cluster_value(param_entry, cluster, value, self.max_values)

            if p_type and p_type != "unknown":
                type_entry = ensure_type_entry(self.types, p_type)
                type_entry["observed"] += 1
                add_cluster_value(type_entry, cluster, value, self.max_values)

    def as_dict(self, meta_summary: dict) -> dict:
        for func in self.functions.values():
            for param in func["params"].values():
                scrub_clusters(param)
        for param in self.params.values():
            scrub_clusters(param)
        for entry in self.types.values():
            scrub_clusters(entry)
        return {
            "functions": self.functions,
            "params": self.params,
            "types": self.types,
            "meta": meta_summary,
        }

    def merge_snapshot(self, snapshot: dict):
        for func_name, func_info in (snapshot.get("functions") or {}).items():
            for param_name, param_info in (func_info.get("params") or {}).items():
                merged = ensure_func_param_entry(
                    self.functions,
                    func_name,
                    param_name,
                    param_info.get("type"),
                    param_info.get("required", False),
                )
                merged["observed"] += param_info.get("observed", 0)
                if param_info.get("required"):
                    merged["required"] = True
                for cluster_key, bucket in (param_info.get("clusters") or {}).items():
                    for value in bucket.get("values") or []:
                        add_cluster_value(merged, cluster_key, value, self.max_values)

        for param_name, param_info in (snapshot.get("params") or {}).items():
            merged = ensure_param_entry(self.params, param_name, param_info.get("type"))
            merged["observed"] += param_info.get("observed", 0)
            for cluster_key, bucket in (param_info.get("clusters") or {}).items():
                for value in bucket.get("values") or []:
                    add_cluster_value(merged, cluster_key, value, self.max_values)

        for type_name, type_info in (snapshot.get("types") or {}).items():
            merged = ensure_type_entry(self.types, type_name)
            merged["observed"] += type_info.get("observed", 0)
            for cluster_key, bucket in (type_info.get("clusters") or {}).items():
                for value in bucket.get("values") or []:
                    add_cluster_value(merged, cluster_key, value, self.max_values)


def main() -> None:
    args = parse_args()
    files = discover_files(args.input)
    if not files:
        raise SystemExit(f"No jsonl files found under: {args.input}")
    meta = load_meta(args.stats)
    builder = PoolBuilder(meta, max_values=args.max_values)

    total_records = 0
    total_calls = 0
    total_arguments = 0

    print(f"[INFO] Building param pool from {len(files)} files using {args.workers} workers.")

    if args.workers <= 1:
        _worker_init(str(args.stats), args.max_values)
        for path in files:
            _, snapshot, stats = _worker_process(str(path))
            builder.merge_snapshot(snapshot)
            total_records += stats["records"]
            total_calls += stats["function_calls"]
            total_arguments += stats["arguments"]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(str(args.stats), args.max_values),
        ) as executor:
            futures = {
                executor.submit(_worker_process, str(path)): path for path in files
            }
            for future in concurrent.futures.as_completed(futures):
                _, snapshot, stats = future.result()
                builder.merge_snapshot(snapshot)
                total_records += stats["records"]
                total_calls += stats["function_calls"]
                total_arguments += stats["arguments"]

    summary = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_path": str(args.input),
        "files_count": len(files),
        "total_records": total_records,
        "total_function_calls": total_calls,
        "total_arguments": total_arguments,
        "stats_path": str(args.stats),
        "max_values_per_cluster": args.max_values,
    }

    output = builder.as_dict(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)

    print(
        f"[INFO] Param pool saved to {args.output} "
        f"(functions={len(output['functions'])}, params={len(output['params'])})"
    )


if __name__ == "__main__":
    main()

