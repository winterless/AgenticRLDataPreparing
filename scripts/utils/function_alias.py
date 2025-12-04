#!/usr/bin/env python3
"""Helpers for generating and applying deterministic function aliases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

SYLLABLES = [
    "ka",
    "ke",
    "ki",
    "ko",
    "ku",
    "la",
    "le",
    "li",
    "lo",
    "lu",
    "ma",
    "me",
    "mi",
    "mo",
    "mu",
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


def _alias_from_digest(name: str, salt: str = "") -> str:
    digest = hashlib.sha1((name + salt).encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "big")
    base = len(SYLLABLES)
    parts: list[str] = []
    for _ in range(4):
        parts.append(SYLLABLES[idx % base])
        idx //= base
    suffix = hashlib.sha1((name + salt + "_alias").encode("utf-8")).hexdigest()[:2]
    return f"func_{''.join(parts)}{suffix}"


def build_alias_map(names: Iterable[str], existing: dict[str, str] | None = None) -> dict[str, str]:
    alias_map = dict(existing or {})
    used = set(alias_map.values())
    for name in sorted({n for n in names if n}):
        if name in alias_map:
            continue
        alias = _alias_from_digest(name)
        salt = 1
        while alias in used:
            alias = _alias_from_digest(name, str(salt))
            salt += 1
        alias_map[name] = alias
        used.add(alias)
    return alias_map


def load_alias_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Alias file must contain a JSON object: {path}")
    return {str(k): str(v) for k, v in data.items()}


def save_alias_map(mapping: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(mapping, fh, ensure_ascii=False, indent=2)


def invert_alias_map(mapping: dict[str, str]) -> dict[str, str]:
    return {alias: original for original, alias in mapping.items()}


def apply_alias(name: str | None, mapping: dict[str, str]) -> str | None:
    if not name:
        return name
    return mapping.get(name, name)

