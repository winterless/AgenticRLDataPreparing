#!/usr/bin/env python3
"""
CLI helper that converts a Parquet file into JSON Lines (jsonl) format.

Example:
    python generate.py --input data/sample.parquet --output out.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "pyarrow is required. Install it via `pip install pyarrow` and retry."
    ) from exc


def _iter_records(
    parquet_file: pq.ParquetFile, batch_size: int, columns: list[str] | None
) -> Iterable[dict]:
    """Yield row dictionaries by streaming through the parquet file."""
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        # Converting to_pylist keeps memory usage bounded by `batch_size`.
        for record in batch.to_pylist():
            yield record


def convert(
    input_path: Path,
    output_path: Path,
    batch_size: int,
    columns: list[str] | None,
    limit: int | None,
) -> None:
    """Stream parquet rows into jsonl to avoid loading the whole file."""
    parquet_file = pq.ParquetFile(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    with output_path.open("w", encoding="utf-8") as sink:
        for row in _iter_records(parquet_file, batch_size=batch_size, columns=columns):
            if limit is not None and emitted >= limit:
                break
            json.dump(row, sink, ensure_ascii=False)
            sink.write("\n")
            emitted += 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a Parquet file into JSON Lines format.")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Path to the source parquet file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path for the generated jsonl file (defaults to replacing the input suffix with .jsonl).",
    )
    parser.add_argument(
        "-c",
        "--columns",
        nargs="+",
        help="Optional subset of columns to export.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=2048,
        help="Number of rows processed per batch to balance speed and memory (default: 2048).",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        help="Optional cap on the number of rows written (e.g. 500).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    output = args.output or args.input.with_suffix(".jsonl")
    convert(
        args.input,
        output,
        batch_size=args.batch_size,
        columns=args.columns,
        limit=args.limit,
    )

    print(f"Wrote {output}")  # stdout message for quick confirmation
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

