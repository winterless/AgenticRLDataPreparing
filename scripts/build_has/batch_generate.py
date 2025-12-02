#!/usr/bin/env python3
"""Batch convert Toucan jsonl files into HAS-ready artifacts in parallel."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = BASE_DIR / "Toucan-1.5M" / "Toucan-1.5M"
DEFAULT_OUTPUT = BASE_DIR / "data" / "Toucan-1.5M-generate"
DEFAULT_STATS = BASE_DIR / "stats" / "function_stats.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pretty_toucan + HAS-API generation for all jsonl files in a directory."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Directory containing source jsonl files (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Directory to store generated artifacts (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "-s",
        "--stats",
        type=Path,
        default=DEFAULT_STATS,
        help=f"function_meta JSON produced by function_stats.py (default: {DEFAULT_STATS}).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["random", "available", "params", "param_values"],
        help="HAS-API modes to run for each jsonl file.",
    )
    parser.add_argument(
        "--negatives",
        type=int,
        default=5,
        help="Desired number of negative options for random/params modes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed forwarded to build_has_api.py.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optionally limit the number of files processed (useful for smoke tests).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Forwarded to build_has_api.py to cap HAS-API outputs per file.",
    )
    parser.add_argument(
        "--copy-input",
        action="store_true",
        help="Copy each source jsonl into the output directory for reference.",
    )
    parser.add_argument(
        "--pretty-records",
        type=int,
        default=0,
        help="Pretty-print at most N records per file (0 disables pretty output).",
    )
    return parser.parse_args()


@dataclass
class JobConfig:
    output_dir: Path
    stats_path: Path
    pretty_script: Path
    has_script: Path
    modes: list[str]
    negatives: int
    seed: int
    copy_input: bool
    max_samples: int | None
    pretty_records: int


def run_command(cmd: list[str], log_prefix: str, stdout=None) -> None:
    cmd_str = " ".join(cmd)
    print(f"[{log_prefix}] RUN {cmd_str}")
    subprocess.run(cmd, check=True, stdout=stdout)


def process_file(jsonl_path: Path, rel_path: Path, cfg: JobConfig) -> tuple[str, bool, str | None]:
    log_prefix = rel_path.as_posix()
    try:
        dest_dir = cfg.output_dir / rel_path.parent
        dest_dir.mkdir(parents=True, exist_ok=True)

        if cfg.copy_input:
            dest_jsonl = dest_dir / jsonl_path.name
            shutil.copy2(jsonl_path, dest_jsonl)

        if cfg.pretty_records != 0:
            pretty_output = dest_dir / f"{jsonl_path.stem}.txt"
            cmd = [sys.executable, str(cfg.pretty_script), "-i", str(jsonl_path)]
            if cfg.pretty_records > 0:
                cmd.extend(["-n", str(cfg.pretty_records)])
            with pretty_output.open("w", encoding="utf-8") as pretty_f:
                run_command(cmd, log_prefix, stdout=pretty_f)

        for mode in cfg.modes:
            api_output = dest_dir / f"{jsonl_path.stem}_api_{mode}.jsonl"
            cmd = [
                sys.executable,
                str(cfg.has_script),
                "-i",
                str(jsonl_path),
                "-s",
                str(cfg.stats_path),
                "-o",
                str(api_output),
                "--mode",
                mode,
                "--negatives",
                str(cfg.negatives),
                "--seed",
                str(cfg.seed),
            ]
            if cfg.max_samples:
                cmd.extend(["--max-samples", str(cfg.max_samples)])
            run_command(cmd, log_prefix)

        return (log_prefix, True, None)
    except subprocess.CalledProcessError as exc:
        return (log_prefix, False, f"Command failed: {exc}")
    except Exception as exc:
        return (log_prefix, False, str(exc))


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")
    if not args.stats.exists():
        raise SystemExit(f"Stats file not found: {args.stats}")

    jsonl_files = sorted(args.input_dir.rglob("*.jsonl"))
    if not jsonl_files:
        print(f"[WARN] No jsonl files found under {args.input_dir}")
        return
    if args.max_files:
        jsonl_files = jsonl_files[: args.max_files]

    cfg = JobConfig(
        output_dir=args.output_dir,
        stats_path=args.stats,
        pretty_script=BASE_DIR / "scripts" / "analysis" / "pretty_toucan.py",
        has_script=BASE_DIR / "scripts" / "build_has" / "build_has_api.py",
        modes=args.modes,
        negatives=args.negatives,
        seed=args.seed,
        copy_input=args.copy_input,
        max_samples=args.max_samples,
        pretty_records=args.pretty_records,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Found {len(jsonl_files)} jsonl files. Launching {args.workers} workers.")

    tasks = []
    for file_path in jsonl_files:
        rel_path = file_path.relative_to(args.input_dir)
        tasks.append((file_path, rel_path))

    success = 0
    failures: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {
            executor.submit(process_file, file_path, rel_path, cfg): rel_path.as_posix()
            for file_path, rel_path in tasks
        }
        for future in as_completed(future_to_task):
            rel = future_to_task[future]
            ok = False
            err = None
            try:
                _, ok, err = future.result()
            except Exception as exc:
                ok = False
                err = str(exc)
            if ok:
                success += 1
                print(f"[DONE] {rel}")
            else:
                failures.append((rel, err or "unknown error"))
                print(f"[FAIL] {rel}: {err}")

    print(f"[SUMMARY] Completed {success}/{len(tasks)} files.")
    if failures:
        print("[SUMMARY] Failures:")
        print(json.dumps(failures, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

