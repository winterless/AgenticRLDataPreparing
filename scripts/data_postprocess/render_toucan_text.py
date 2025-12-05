#!/usr/bin/env python3
"""
Render assembled Toucan JSONL samples into a pretty text file.

This script can be used directly via `-i/--input` and `-o/--output`,
and expose helper functions that other modules (e.g. assemble_toucan.py)
can import to emit the same pretty text file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Toucan MCQ JSONL (with `text` fields) into pretty txt."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the JSONL file produced by assemble_toucan.py.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Destination txt path (default: replace suffix of input with .txt).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of records to render.",
    )
    return parser.parse_args()


def _iter_texts(input_path: Path, limit: int | None = None) -> Iterator[str]:
    processed = 0
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL line in {input_path}: {exc}") from exc
            text = payload.get("text")
            if text is None:
                raise ValueError(
                    "Input JSONL is missing the `text` field required for rendering."
                )
            yield text
            processed += 1
            if limit and processed >= limit:
                break


def write_texts_to_file(texts: Iterable[str], output_path: Path) -> int:
    """
    Write an iterable of pre-formatted text blocks into `output_path`,
    inserting blank lines between records to preserve readability.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for text in texts:
            fh.write(text)
            if not text.endswith("\n"):
                fh.write("\n")
            fh.write("\n")
            count += 1
    return count


def convert_jsonl_to_txt(
    input_path: Path, output_path: Path, limit: int | None = None
) -> int:
    """
    Stream the `text` fields from `input_path` JSONL into `output_path`.
    Returns the number of rendered records.
    """
    return write_texts_to_file(_iter_texts(input_path, limit=limit), output_path)


def main() -> None:
    args = parse_args()
    output = (
        args.output
        if args.output
        else args.input.with_suffix(".txt")
        if args.input.suffix
        else args.input.parent / (args.input.name + ".txt")
    )
    count = convert_jsonl_to_txt(args.input, output, args.limit)
    print(f"[INFO] Rendered {count} records from {args.input} into {output}.")


if __name__ == "__main__":
    main()


