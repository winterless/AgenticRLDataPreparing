#!/usr/bin/env python3
"""
Generate a deterministic function alias map based on function_stats.json.

Example:
    python scripts/data_preprocess/build_function_alias.py \
        --stats stats/function_stats.json \
        --output stats/function_alias.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.utils.function_alias import build_alias_map, load_alias_map, save_alias_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build function alias map from stats JSON.")
    parser.add_argument(
        "--stats",
        type=Path,
        required=True,
        help="Path to function_stats.json (keys must be real function names).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stats/function_alias.json"),
        help="Where to store the alias map (default: stats/function_alias.json).",
    )
    parser.add_argument(
        "--existing",
        type=Path,
        default=None,
        help="Optional existing alias map to reuse / extend for stability.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.stats.exists():
        raise SystemExit(f"Stats JSON not found: {args.stats}")

    with args.stats.open("r", encoding="utf-8") as fh:
        stats = json.load(fh)
    if not isinstance(stats, dict):
        raise SystemExit(f"Stats file must contain a JSON object: {args.stats}")
    names = list(stats.keys())
    if args.existing and args.existing.exists():
        existing = load_alias_map(args.existing)
    else:
        existing = {}
    alias_map = build_alias_map(names, existing)
    save_alias_map(alias_map, args.output)
    print(f"[INFO] Generated alias map with {len(alias_map)} entries -> {args.output}")


if __name__ == "__main__":
    main()

