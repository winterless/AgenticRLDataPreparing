#!/usr/bin/env python3
"""
Simple cleaner for Toucan-style jsonl data.

Example:
    python clean_toucan.py -i data/toucan.jsonl -o data/toucan_clean.jsonl \
        --min-question-score 3.5 --min-response-score 3.0 --min-tool-usage 0.2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_float(value: str | float | None, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_json_field(field: str | dict | None) -> dict:
    if field is None:
        return {}
    if isinstance(field, dict):
        return field
    try:
        return json.loads(field)
    except (TypeError, json.JSONDecodeError):
        return {}


def record_passes_filters(record: dict, args: argparse.Namespace) -> bool:
    q_quality = _load_json_field(record.get("question_quality_assessment"))
    r_quality = _load_json_field(record.get("response_quality_assessment"))

    if (
        _parse_float(q_quality.get("overall_score")) < args.min_question_score
        or _parse_float(r_quality.get("overall_score")) < args.min_response_score
    ):
        return False

    desired_pct = _parse_float(r_quality.get("desired_tools_used_percentage"))
    if desired_pct < args.min_tool_usage:
        return False

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter Toucan jsonl by quality metrics.")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Source jsonl file.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Filtered jsonl path.")
    parser.add_argument(
        "--min-question-score",
        type=float,
        default=3.0,
        help="Minimum question overall_score required (default: 3.0).",
    )
    parser.add_argument(
        "--min-response-score",
        type=float,
        default=3.0,
        help="Minimum response overall_score required (default: 3.0).",
    )
    parser.add_argument(
        "--min-tool-usage",
        type=float,
        default=0.0,
        help="Minimum desired_tools_used_percentage required (default: 0.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    kept = 0
    total = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open("r", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as sink:
        for line in source:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record_passes_filters(record, args):
                json.dump(record, sink, ensure_ascii=False)
                sink.write("\n")
                kept += 1

    print(f"Kept {kept}/{total} records ({(kept/total*100 if total else 0):.1f}%).")


if __name__ == "__main__":
    main()


