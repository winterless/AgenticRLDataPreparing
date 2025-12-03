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


