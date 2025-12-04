#!/usr/bin/env python3
"""Deterministic, human-readable aliases for function/tool names."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

SYLLABLES = [
    "ba",
    "be",
    "bi",
    "bo",
    "bu",
    "ca",
    "ce",
    "ci",
    "co",
    "cu",
    "da",
    "de",
    "di",
    "do",
    "du",
    "fa",
    "fe",
    "fi",
    "fo",
    "fu",
    "ga",
    "ge",
    "gi",
    "go",
    "gu",
    "la",
    "le",
    "li",
    "lo",
    "lu",
    "na",
    "ne",
    "ni",
    "no",
    "nu",
    "ra",
    "re",
    "ri",
    "ro",
    "ru",
    "sa",
    "se",
    "si",
    "so",
    "su",
    "ta",
    "te",
    "ti",
    "to",
    "tu",
    "va",
    "ve",
    "vi",
    "vo",
    "vu",
]

ALIAS_PREFIX = "func_"


def _alias_from_digest(name: str, salt: str = "", attempt: int = 0) -> str:
    """Map a function name to a pronounceable alias using SHA1 + syllables."""
    seed = f"{salt}|{name}|{attempt}".encode("utf-8")
    digest = hashlib.sha1(seed).hexdigest()
    parts: list[str] = []
    for i in range(0, 12, 2):
        chunk = digest[i : i + 2]
        if not chunk:
            break
        idx = int(chunk, 16) % len(SYLLABLES)
        parts.append(SYLLABLES[idx])
    # keep aliases compact: 3 syllables + last 3 hex chars as suffix
    alias_body = "".join(parts[:3])
    suffix = digest[-3:]
    return f"{ALIAS_PREFIX}{alias_body}{suffix}"


def build_alias_map(
    names: Iterable[str],
    existing: dict[str, str] | None = None,
    *,
    salt: str = "",
) -> dict[str, str]:
    """Create/extend an alias map, preserving existing assignments."""
    mapping = dict(existing or {})
    used_aliases = {alias for alias in mapping.values() if isinstance(alias, str)}
    for name in sorted(set(names)):
        if not isinstance(name, str) or not name:
            continue
        if name in mapping:
            continue
        attempt = 0
        while True:
            alias = _alias_from_digest(name, salt=salt, attempt=attempt)
            if alias not in used_aliases:
                mapping[name] = alias
                used_aliases.add(alias)
                break
            attempt += 1
    return mapping


def load_alias_map(path: Path) -> dict[str, str]:
    """Load a JSON mapping of original name -> alias."""
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Alias map must be a dict, got {type(data)}")
    mapping: dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        mapping[key] = value
    return mapping


def save_alias_map(mapping: dict[str, str], path: Path) -> None:
    """Persist alias map as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(mapping, fh, ensure_ascii=False, indent=2)


def invert_alias_map(mapping: dict[str, str]) -> dict[str, str]:
    """Return alias -> original map (last writer wins on duplicates)."""
    return {alias: name for name, alias in mapping.items()}


def apply_alias(name: str | None, mapping: dict[str, str]) -> str | None:
    """Look up alias; fall back to original when unseen."""
    if name is None:
        return None
    return mapping.get(name, name)


__all__ = [
    "SYLLABLES",
    "build_alias_map",
    "load_alias_map",
    "save_alias_map",
    "invert_alias_map",
    "apply_alias",
]

