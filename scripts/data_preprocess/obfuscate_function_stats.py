#!/usr/bin/env python3
"""
Rewrite function_stats (CSV/JSON) using a precomputed alias map.

Example:
    python scripts/data_preprocess/obfuscate_function_stats.py \
        --alias stats/function_alias.json \
        --stats-json stats/function_stats.json \
        --stats-csv stats/function_stats.csv \
        --meta-json stats/function_meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.utils.function_alias import load_alias_map, apply_alias


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Obfuscate function_stats artifacts.")
    parser.add_argument("--alias", type=Path, required=True, help="Path to function_alias.json.")
    parser.add_argument("--stats-json", type=Path, required=True, help="function_stats.json path.")
    parser.add_argument("--stats-csv", type=Path, required=True, help="function_stats.csv path.")
    parser.add_argument(
        "--meta-json",
        type=Path,
        default=None,
        help="Optional function_meta.json path (will be rewritten in place).",
    )
    return parser.parse_args()


def rewrite_stats_json(path: Path, alias_map: dict[str, str]) -> None:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    new_data = {apply_alias(name, alias_map): value for name, value in data.items()}
    with path.open("w", encoding="utf-8") as fh:
        json.dump(new_data, fh, ensure_ascii=False, indent=2)
    print(f"[INFO] Rewrote {path}")


def rewrite_stats_csv(path: Path, alias_map: dict[str, str]) -> None:
    with path.open("r", encoding="utf-8") as src:
        reader = list(csv.reader(src))
    header, *rows = reader
    rewritten = [header]
    for row in rows:
        if not row:
            continue
        row[0] = apply_alias(row[0], alias_map)
        rewritten.append(row)
    with path.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.writer(dst)
        writer.writerows(rewritten)
    print(f"[INFO] Rewrote {path}")


def main() -> None:
    args = parse_args()
    alias_map = load_alias_map(args.alias)
    rewrite_stats_json(args.stats_json, alias_map)
    rewrite_stats_csv(args.stats_csv, alias_map)
    if args.meta_json:
        rewrite_stats_json(args.meta_json, alias_map)


if __name__ == "__main__":
    main()

