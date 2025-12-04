#!/usr/bin/env python3
"""
Rewrite param_pool.json so that function names are obfuscated.

Example:
    python scripts/data_preprocess/obfuscate_param_pool.py \
        --alias stats/function_alias.json \
        --input stats/param_pool.json \
        --output stats/param_pool_obf.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.utils.function_alias import load_alias_map, apply_alias


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Obfuscate function names inside param_pool.json.")
    parser.add_argument("--alias", type=Path, required=True, help="Path to function_alias.json.")
    parser.add_argument("--input", type=Path, required=True, help="Source param_pool.json.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination param_pool.json (defaults to overwriting the input).",
    )
    return parser.parse_args()


def rename_keys(data: dict[str, dict], alias_map: dict[str, str]) -> dict[str, dict]:
    renamed: dict[str, dict] = {}
    for key, value in (data or {}).items():
        renamed[apply_alias(key, alias_map)] = value
    return renamed


def main() -> None:
    args = parse_args()
    alias_map = load_alias_map(args.alias)
    if not args.input.exists():
        raise SystemExit(f"param_pool JSON not found: {args.input}")
    with args.input.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    data["functions"] = rename_keys(data.get("functions") or {}, alias_map)
    data["meta"] = data.get("meta") or {}
    output_path = args.output or args.input
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote obfuscated param pool to {output_path}")


if __name__ == "__main__":
    main()

