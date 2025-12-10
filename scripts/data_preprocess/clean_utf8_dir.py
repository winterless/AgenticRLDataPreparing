#!/usr/bin/env python3
"""
Remove non-UTF-8 lines from every file under a directory.

Usage:
    python scripts/data_preprocess/clean_utf8_dir.py \
        -i /path/to/src_dir \
        -o /path/to/dst_dir
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _clean_file(src: Path, dst: Path) -> tuple[int, int]:
    """
    Copy `src` to `dst`, dropping any line that cannot be decoded as UTF-8.

    Returns:
        kept, dropped counts.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    with src.open("rb") as fh_in, dst.open("w", encoding="utf-8", newline="") as fh_out:
        for raw_line in fh_in:
            try:
                text_line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                dropped += 1
                continue
            fh_out.write(text_line)
            kept += 1
    return kept, dropped


def _task(args: tuple[str, str]) -> tuple[str, int, int]:
    src_str, dst_str = args
    kept, dropped = _clean_file(Path(src_str), Path(dst_str))
    return src_str, kept, dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drop non-UTF-8 lines from all files in a directory."
    )
    parser.add_argument(
        "-i", "--input-dir", type=Path, required=True, help="Source directory to scan."
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, required=True, help="Destination directory."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    files = [p for p in sorted(args.input_dir.rglob("*")) if p.is_file()]
    if not files:
        raise SystemExit(f"No files found under {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_kept = total_dropped = 0
    workers = max(1, args.workers)

    if workers == 1:
        for src in files:
            rel = src.relative_to(args.input_dir)
            dst = args.output_dir / rel
            kept, dropped = _clean_file(src, dst)
            total_kept += kept
            total_dropped += dropped
            if dropped:
                print(f"[WARN] Dropped {dropped} lines in {src}")
    else:
        tasks = []
        for src in files:
            rel = src.relative_to(args.input_dir)
            dst = args.output_dir / rel
            tasks.append((str(src), str(dst)))

        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_task, task): task for task in tasks}
            for future in as_completed(future_map):
                src_str, kept, dropped = future.result()
                total_kept += kept
                total_dropped += dropped
                if dropped:
                    print(f"[WARN] Dropped {dropped} lines in {src_str}")

    print(
        f"[INFO] Completed. Kept {total_kept} lines, dropped {total_dropped} non-UTF-8 lines."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

