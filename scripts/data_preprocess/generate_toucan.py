#!/usr/bin/env python3
"""
CLI helper that converts a Parquet file into JSON Lines (jsonl) format.

Example:
    python scripts/data_preprocess/generate_toucan.py \
        -i Toucan-1.5M/Toucan-1.5M/Kimi-K2/train-00000-of-00040.parquet \
        -o data/demo/toucan.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _dump_row(row: dict, sink, drop_non_utf8: bool) -> bool:
    """
    Serialize and write one row. Returns True if written; False if dropped.
    """
    try:
        payload = json.dumps(row, ensure_ascii=False)
        if drop_non_utf8:
            payload.encode("utf-8")
        sink.write(payload)
        sink.write("\n")
        return True
    except UnicodeEncodeError:
        return False


def convert(
    input_path: Path,
    output_path: Path,
    batch_size: int,
    columns: list[str] | None,
    limit: int | None,
    sample_size: int | None,
    seed: int | None,
    drop_non_utf8: bool,
) -> tuple[Path, int, int]:
    """
    Stream parquet rows into jsonl to avoid loading the whole file.

    Returns (output_path, kept_lines, dropped_non_utf8_lines).
    """
    parquet_file = pq.ParquetFile(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = dropped = 0

    if sample_size:
        reservoir: list[dict] = []
        if seed is not None:
            random.seed(seed)
        
        seen = 0
        cap = sample_size
        with output_path.open("w", encoding="utf-8") as sink:
            for row in _iter_records(parquet_file, batch_size=batch_size, columns=columns):
                if limit is not None and seen >= limit:
                    break
                seen += 1
                if len(reservoir) < cap:
                    reservoir.append(row)
                else:
                    j = random.randint(1, seen)
                    if j <= cap:
                        reservoir[j - 1] = row
            for row in reservoir:
                if _dump_row(row, sink, drop_non_utf8):
                    kept += 1
                else:
                    dropped += 1
    else:
        emitted = 0
        with output_path.open("w", encoding="utf-8") as sink:
            for row in _iter_records(parquet_file, batch_size=batch_size, columns=columns):
                if limit is not None and emitted >= limit:
                    break
                if _dump_row(row, sink, drop_non_utf8):
                    emitted += 1
                    kept += 1
                else:
                    dropped += 1
    return output_path, kept, dropped


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
    parser.add_argument(
        "-s",
        "--sample-size",
        type=int,
        help="Randomly sample this many rows from the (optionally limited) dataset before writing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed used for sampling.",
    )
    parser.add_argument(
        "--drop-non-utf8",
        action="store_true",
        help="Skip lines that cannot be encoded as UTF-8 (helps downstream Windows tooling).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers when processing directories.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    if args.input.is_dir():
        output_base = args.output
        if output_base:
            output_base.mkdir(parents=True, exist_ok=True)
        files = sorted(args.input.rglob("*.parquet"))
        if not files:
            raise SystemExit(f"No parquet files found in directory: {args.input}")

        workers = max(1, args.workers)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {}
            for file in files:
                if output_base:
                    output = output_base / (file.stem + ".jsonl")
                else:
                    output = file.with_suffix(".jsonl")
                future = executor.submit(
                    convert,
                    file,
                    output,
                    args.batch_size,
                    args.columns,
                    args.limit,
                    args.sample_size,
                    args.seed,
                    args.drop_non_utf8,
                )
                future_map[future] = output
            for future in as_completed(future_map):
                out_path, kept, dropped = future.result()
                extra = f", dropped {dropped} non-UTF-8 lines" if dropped else ""
                print(f"Wrote {out_path} (kept {kept} lines{extra})")
    else:
        output = args.output or args.input.with_suffix(".jsonl")
        _, kept, dropped = convert(
            args.input,
            output,
            batch_size=args.batch_size,
            columns=args.columns,
            limit=args.limit,
            sample_size=args.sample_size,
            seed=args.seed,
            drop_non_utf8=args.drop_non_utf8,
        )
        extra = f", dropped {dropped} non-UTF-8 lines" if dropped else ""
        print(f"Wrote {output} (kept {kept} lines{extra})")  # stdout message for quick confirmation
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

