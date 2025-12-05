#!/usr/bin/env python3
"""Shared helpers for HAS generation scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_jsonl(path: Path) -> Iterable[dict]:
    """Yield json objects from a jsonl file, skipping malformed lines."""
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_meta(stats_path: Path) -> dict[str, dict]:
    """Load function_meta JSON containing schema info."""
    with stats_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def iter_function_calls(record: dict):
    """Iterate over (message_index, function_call) pairs inside a record."""
    messages = record.get("messages")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return
    if not isinstance(messages, list):
        return
    for idx, msg in enumerate(messages):
        fc = msg.get("function_call")
        if fc and isinstance(fc, dict) and fc.get("name"):
            yield idx, fc


def parse_arguments(function_call: dict) -> dict | None:
    """Parse the arguments field from a function_call."""
    args = function_call.get("arguments")
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def format_arg_values(args: dict) -> str:
    """Canonicalize arguments dict into sorted key=value; key2=value2 string."""
    pairs = []
    for key in sorted(args.keys()):
        pairs.append(f"{key}={json.dumps(args[key], ensure_ascii=False)}")
    return "; ".join(pairs)


def infer_param_type(schema: dict | None, value, default: str | None = None) -> str | None:
    """Best-effort inference of parameter type, falling back to runtime value."""
    schema = schema or {}
    declared = schema.get("type")
    if isinstance(declared, list):
        chosen = None
        for candidate in declared:
            if isinstance(candidate, str) and candidate != "null":
                chosen = candidate
                break
        if chosen is None and declared:
            chosen = declared[0]
        declared = chosen
    if declared and not isinstance(declared, str):
        declared = None
    if declared:
        return declared
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return default








