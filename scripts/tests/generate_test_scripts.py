#!/usr/bin/env python3
"""
Extract tagged python commands from README and materialize shell test scripts.

Rules:
  - Inside fenced ```bash blocks, lines like `# full` / `# single` / `# online`
    act as decorators for the next python command.
  - Inside fenced ```bash blocks, `# vars` collects shell variable defaults
    for generated scripts (applies to both full and single).
  - Commands starting with `python ` (or `python3 `) inherit the most recent tag.
    The tag is consumed after one command (i.e., it applies to the *next* command only).
  - `# online` commands are skipped; `# full` goes to full_generate_test.sh;
    `# single` goes to single_generate_test.sh.
  - `# test` comments are treated as documentation only and clear any pending tag.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable

# NOTE: match directive lines only (avoid treating comments like "# full pipeline" as tags)
TAG_PATTERN = re.compile(r"^#\s*(?P<tag>[A-Za-z][A-Za-z0-9_]*)\s*$")
SCRIPT_PATH = Path(__file__).resolve()


def find_git_root(start: Path) -> Path:
    """Return nearest ancestor containing a .git directory."""
    for candidate in [start] + list(start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


GIT_ROOT = find_git_root(Path.cwd())
DEFAULT_PROJECT = SCRIPT_PATH.parents[2]
ALT_PROJECT = GIT_ROOT / "AgenticRLDataPreparing"
if (ALT_PROJECT / "README.md").exists():
    PROJECT_ROOT = ALT_PROJECT
else:
    PROJECT_ROOT = DEFAULT_PROJECT


def detect_tag(line: str) -> str | None:
    """Return normalized tag name if the line declares one."""
    match = TAG_PATTERN.match(line.strip())
    if not match:
        return None
    token = match.group("tag").lower()
    if token in {"test"}:
        return "test"
    if token in {"vars", "var", "env"}:
        return "vars"
    if token in {"online"}:
        return "online"
    if token in {"full"}:
        return "full"
    if token in {"single", "signle"}:
        return "single"
    return None


def is_python_command(line: str) -> bool:
    """
    True if the line starts a python invocation.

    Supports common wrappers like:
      - `python ...`
      - `python3 ...`
      - `VAR=1 python ...`
      - `stdbuf -oL -eL python ...`
    """
    stripped = line.strip()
    if not stripped:
        return False
    tokens = stripped.split()
    if not tokens:
        return False

    i = 0
    # Skip leading env assignments: KEY=VALUE
    while i < len(tokens) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", tokens[i]):
        i += 1
    if i >= len(tokens):
        return False

    if tokens[i] in {"python", "python3"}:
        return True

    # Allow `stdbuf ... python ...`
    if tokens[i] == "stdbuf":
        i += 1
        while i < len(tokens) and tokens[i] not in {"python", "python3"}:
            i += 1
        return i < len(tokens) and tokens[i] in {"python", "python3"}

    return False


def _dedupe_vars(lines: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            continue
        m = re.match(r"^(?:export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=", stripped)
        key = m.group("key") if m else None
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(stripped)
    return out


def collect_commands(readme: Path) -> dict[str, list[str]]:
    inside_bash = False
    current_tag: str | None = None
    commands: dict[str, list[str]] = {"full": [], "single": [], "vars": []}

    lines = readme.read_text(encoding="utf-8").splitlines()
    i = 0
    total = len(lines)
    while i < total:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```bash"):
            inside_bash = True
            current_tag = None
            i += 1
            continue
        if stripped.startswith("```"):
            inside_bash = False
            current_tag = None
            i += 1
            continue
        if not inside_bash:
            i += 1
            continue

        tag = detect_tag(stripped)
        if tag:
            if tag == "test":
                current_tag = None
                i += 1
                continue
            current_tag = tag
            i += 1
            continue

        if current_tag == "vars":
            block = []
            while i < total:
                nxt = lines[i]
                nxt_strip = nxt.strip()
                if nxt_strip.startswith("```"):
                    inside_bash = False
                    current_tag = None
                    i += 1
                    break
                # Stop if a new directive starts (e.g. # full / # single / # vars)
                if detect_tag(nxt_strip):
                    break
                # Keep comment lines; keep indentation minimal by stripping.
                if nxt_strip:
                    block.append(nxt_strip)
                i += 1
            commands["vars"].extend(block)
            # Tag is a decorator for the next block only.
            current_tag = None
            continue

        if is_python_command(stripped) and current_tag in {"full", "single"}:
            target_tag = current_tag
            block = [stripped]
            i += 1
            while i < total:
                nxt = lines[i]
                nxt_strip = nxt.strip()
                if not nxt_strip:
                    break
                if nxt_strip.startswith("```"):
                    inside_bash = False
                    current_tag = None
                    i += 1
                    break
                if nxt_strip.startswith("#"):
                    break
                block.append(nxt_strip)
                i += 1
            commands[target_tag].append("\n".join(block))
            # Tag is a decorator for the next command only.
            current_tag = None
            # Skip blank/comment separator
            i += 1
            continue

        i += 1

    # Dedupe vars by var name, keep first occurrence.
    commands["vars"] = _dedupe_vars(commands["vars"])
    return commands


def write_shell_script(path: Path, commands: Iterable[str], vars_lines: Iterable[str]) -> None:
    lines = ["#!/bin/bash", "set -euo pipefail", ""]
    vars_written = False
    for v in vars_lines:
        vars_written = True
        lines.append(v)
    if vars_written:
        lines.append("")
    written = False
    for cmd in commands:
        written = True
        lines.append(cmd)
        lines.append("")
    if not written:
        lines.append("# No commands detected for this mode.")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines).rstrip() + "\n"
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate shell tests from README tags.")
    parser.add_argument(
        "--readme",
        type=Path,
        default=PROJECT_ROOT / "README.md",
        help="Path to README file (default: project README).",
    )
    parser.add_argument(
        "--full-output",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "tests" / "full_generate_test.sh",
        help="Output shell script for # full commands (default: scripts/tests/full_generate_test.sh).",
    )
    parser.add_argument(
        "--single-output",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "tests" / "single_generate_test.sh",
        help="Output shell script for # single commands (default: scripts/tests/single_generate_test.sh).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.readme.exists():
        raise SystemExit(f"README not found: {args.readme}")
    commands = collect_commands(args.readme)
    write_shell_script(args.full_output, commands["full"], commands["vars"])
    write_shell_script(args.single_output, commands["single"], commands["vars"])
    print(
        f"[INFO] Generated {len(commands['full'])} full and "
        f"{len(commands['single'])} single commands."
    )


if __name__ == "__main__":
    main()

