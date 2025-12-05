#!/usr/bin/env python3
"""
Generate HAS-API style multiple-choice data without using LLMs.

Example:
    python scripts/build_has/build_has_api_script.py \
        -i data/demo/toucan.jsonl \
        -s stats/function_stats.json \
        -o data/demo/toucan_api_available.jsonl \
        --mode available
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.append(str(SCRIPTS_ROOT))

from utils.has_utils import (
    format_arg_values,
    infer_param_type,
    iter_function_calls,
    load_jsonl,
    load_meta,
    parse_arguments,
)


if hasattr(sys, "set_int_max_str_digits"):
    try:
        sys.set_int_max_str_digits(0)  # disable limit for large JSON ints
    except Exception:
        pass


def parse_available_tools(record: dict) -> list[str]:
    tools = record.get("available_tools")
    if tools is None:
        return []
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            return []
    names = []
    for tool in tools or []:
        func = tool.get("function") or {}
        if func.get("name"):
            names.append(func["name"])
    return names


def _function_family(name: str) -> str:
    """Return a coarse family key based on dashed function name segments."""
    if not name:
        return ""
    parts = name.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3])
    if len(parts) == 2:
        return "-".join(parts)
    # fall back to underscore grouping
    under = name.split("_")
    if len(under) >= 2:
        return "_".join(under[:2])
    return name


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) if text else set()


def _build_function_profiles(meta: dict[str, dict]) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    for name, info in meta.items():
        func_block = info.get("function") or {}
        description = (
            func_block.get("description")
            or info.get("description")
            or "No description provided."
        )
        tokens = _tokenize(description) | _tokenize(name.replace("-", " "))
        profiles[name] = {
            "family": _function_family(name),
            "tokens": tokens,
            "description": description.strip(),
        }
    return profiles


def _format_option(func_name: str, profiles: dict[str, dict] | None) -> str:
    return func_name


def _select_tool_distractors(
    func_name: str,
    all_funcs: list[str],
    max_count: int,
    profiles: dict[str, dict] | None = None,
    exclude: set[str] | None = None,
) -> list[str]:
    if max_count <= 0:
        return []
    exclude = set(exclude or set())
    pool = [f for f in all_funcs if f != func_name and f not in exclude]
    if not pool:
        return []

    profile = (profiles or {}).get(func_name) or {}
    family_key = profile.get("family") or _function_family(func_name)

    family_pool = [f for f in pool if ((profiles or {}).get(f) or {}).get("family") == family_key]
    random.shuffle(family_pool)
    family_target = min(len(family_pool), max(1, int(round(max_count * 0.3))))
    family_sample = family_pool[:family_target]

    def semantic_score(candidate: str) -> int:
        cand_profile = (profiles or {}).get(candidate) or {}
        cand_tokens = cand_profile.get("tokens") or set()
        target_tokens = profile.get("tokens") or set()
        return len(target_tokens & cand_tokens)

    remaining_pool = [f for f in pool if f not in family_sample]
    semantic_candidates = sorted(
        remaining_pool,
        key=lambda fn: (semantic_score(fn), fn),
        reverse=True,
    )
    semantic_target = max(0, max_count - len(family_sample) - 1)
    semantic_sample = semantic_candidates[:semantic_target]

    picked = list(dict.fromkeys(family_sample + semantic_sample))
    needed = max_count - len(picked)
    if needed > 0:
        leftover = [f for f in pool if f not in picked]
        random.shuffle(leftover)
        picked.extend(leftover[:needed])

    return picked[:max_count]


def question_available(
    func_name: str,
    available: list[str],
    all_funcs: list[str],
    num_neg: int,
    profiles: dict[str, dict] | None = None,
) -> dict | None:
    if not available:
        return None
    deduped: list[str] = []
    seen: set[str] = set()
    for name in available:
        if not isinstance(name, str):
            continue
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)

    if func_name not in seen:
        return None

    max_neg = max(1, num_neg)
    base_negatives = [name for name in deduped if name != func_name]
    if len(base_negatives) > max_neg:
        base_negatives = random.sample(base_negatives, max_neg)

    needed = max_neg - len(base_negatives)
    if needed > 0:
        exclude = set(deduped)
        extra = _select_tool_distractors(
            func_name,
            all_funcs,
            needed,
            profiles=profiles,
            exclude=exclude,
        )
        base_negatives.extend(extra)

    options = [_format_option(name, profiles) for name in base_negatives]
    correct_option = _format_option(func_name, profiles)
    options.append(correct_option)
    random.shuffle(options)
    if len(options) < 2:
        return None
    return {
        "question": "Given the available tools (with decoys), which one should the agent call next?",
        "options": options,
        "answer": correct_option,
    }


def _format_params(params: Iterable[str]) -> str:
    ordered = sorted(params)
    if not ordered:
        return ""
    if len(ordered) == 1:
        return ordered[0]
    return ", ".join(ordered)


def question_params(func_name: str, meta: dict[str, dict], num_neg: int) -> dict | None:
    info = meta.get(func_name, {})
    params = ((info.get("function") or {}).get("parameters") or info.get("parameters") or {})
    required = params.get("required") or []
    properties = params.get("properties") or {}
    required_set = {p for p in required if p in properties}
    if not required_set:
        return None
    correct_option = _format_params(required_set)

    other_params = [p for p in properties.keys() if p not in required_set]
    candidate_sets: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    # combos missing one required param
    if len(required_set) > 1:
        for missing in required_set:
            combo = tuple(sorted(required_set - {missing}))
            if combo and combo not in seen:
                seen.add(combo)
                candidate_sets.append(combo)

    # combos including extra optional params
    for extra in other_params:
        combo = tuple(sorted(required_set | {extra}))
        if combo not in seen:
            seen.add(combo)
            candidate_sets.append(combo)

    if not candidate_sets:
        # fall back to single-parameter distractors if possible
        for param in properties.keys():
            combo = (param,)
            if combo != tuple(sorted(required_set)) and combo not in seen:
                seen.add(combo)
                candidate_sets.append(combo)
        if not candidate_sets:
            return None

    k = min(num_neg, len(candidate_sets))
    sampled = random.sample(candidate_sets, k)
    negs = [_format_params(combo) for combo in sampled]
    options = negs + [correct_option]
    random.shuffle(options)
    return {
        "question": f"When calling {func_name}, which parameters must be provided? (Select all that apply)",
        "options": options,
        "answer": correct_option,
        "answer_type": "multi_select",
    }


class ParamPool:
    """Helper to sample realistic parameter values from a pre-built pool."""

    def __init__(self, data: dict | None, global_mix_prob: float = 0.1):
        data = data or {}
        self.functions = data.get("functions") or {}
        self.params = data.get("params") or {}
        self.types = data.get("types") or {}
        self.global_mix_prob = min(max(global_mix_prob, 0.0), 1.0)

    @property
    def enabled(self) -> bool:
        return bool(self.functions or self.params or self.types)

    def sample(self, func_name: str, param_name: str, param_type: str | None, original):
        """Return a value different from original, prioritizing same function/parameter history."""
        func_entry = (self.functions.get(func_name) or {}).get("params", {}).get(param_name)
        param_entry = self.params.get(param_name)
        type_entry = self.types.get(param_type) if param_type else None

        prefer_global = random.random() < self.global_mix_prob
        search_order: list[dict | None] = []
        if not prefer_global:
            search_order.append(func_entry)
        search_order.extend([param_entry, type_entry])
        if prefer_global:
            search_order.append(func_entry)

        for entry in search_order:
            candidate = self._pick_alternative(entry, original)
            if candidate is not None:
                return candidate
        return None

    @staticmethod
    def _canonical(value) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return str(value)

    def _pick_alternative(self, entry: dict | None, original):
        if not entry:
            return None
        original_key = self._canonical(original)
        clusters = list((entry.get("clusters") or {}).values())
        random.shuffle(clusters)
        for cluster in clusters:
            values = cluster.get("values") or []
            if not values:
                continue
            pool = list(values)
            random.shuffle(pool)
            for value in pool:
                if self._canonical(value) != original_key:
                    return value
        return None


def load_param_pool(path: Path | None) -> ParamPool:
    if not path:
        return ParamPool(None)
    if not path.exists():
        print(f"[WARN] Param pool file not found: {path}", file=sys.stderr)
        return ParamPool(None)
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"[WARN] Failed to parse param pool JSON: {exc}", file=sys.stderr)
        return ParamPool(None)
    return ParamPool(data)


def _drop_argument(args: dict, candidate_fields: list[str]) -> dict | None:
    """Return a copy missing one of the candidate fields."""
    if not args or not candidate_fields:
        return None
    field = random.choice(candidate_fields)
    mutated = dict(args)
    mutated.pop(field, None)
    if mutated == args or not mutated:
        return None
    return mutated


def _mutate_with_pool(func_name: str, args: dict, properties: dict, pool: ParamPool) -> dict | None:
    if not pool or not pool.enabled or not args:
        return None
    mutated = dict(args)
    fields = list(args.keys())
    max_fields = min(2, len(fields))
    k = random.randint(1, max_fields)
    changed = False
    for field in random.sample(fields, k):
        prop = properties.get(field) or {}
        schema_type = infer_param_type(prop, args[field])
        replacement = pool.sample(func_name, field, schema_type, args[field])
        if replacement is None:
            continue
        mutated[field] = replacement
        changed = True
    return mutated if changed else None


def question_param_values(
    func_name: str,
    fc: dict,
    meta: dict[str, dict],
    num_neg: int,
    pool: ParamPool | None = None,
) -> dict | None:
    args = parse_arguments(fc)
    if not args:
        return None
    if not pool or not pool.enabled:
        return None

    info = meta.get(func_name, {})
    params = ((info.get("function") or {}).get("parameters") or info.get("parameters") or {})
    properties = params.get("properties") or {}

    correct_option = format_arg_values(args)
    variations: set[str] = set()
    attempts = 0
    max_attempts = num_neg * 8
    required_fields = [p for p in params.get("required") or [] if p in args]

    while len(variations) < num_neg and attempts < max_attempts:
        attempts += 1
        strategies = ["pool", "pool", "drop_required", "drop_any"]
        strategy = random.choice(strategies)
        if strategy == "pool":
            mutated = _mutate_with_pool(func_name, args, properties, pool)
        elif strategy == "drop_required" and required_fields:
            mutated = _drop_argument(args, required_fields)
        elif strategy == "drop_any":
            mutated = _drop_argument(args, list(args.keys()))
        else:
            mutated = None
        if not mutated:
            continue
        option = format_arg_values(mutated)
        if option and option != correct_option:
            variations.add(option)

    if not variations:
        return None

    options = list(variations)
    if len(options) > num_neg:
        options = random.sample(options, num_neg)
    options.append(correct_option)
    random.shuffle(options)

    return {
        "question": f"For the call to {func_name}, which parameter values are correct?",
        "options": options,
        "answer": correct_option,
        "answer_type": "single_choice",
    }


QUESTION_BUILDERS = {
    "available": question_available,
    "params": question_params,
    "param_values": question_param_values,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HAS-API MCQ data without LLM.")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Source jsonl data.")
    parser.add_argument("-s", "--stats", type=Path, required=True, help="function_meta JSON.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output jsonl.")
    parser.add_argument(
        "--mode",
        choices=list(QUESTION_BUILDERS.keys()),
        required=True,
        help="Strategy for generating options.",
    )
    parser.add_argument(
        "--negatives",
        type=int,
        default=9,
        help="Number of negative options (up to 9 for available/params modes).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit on outputs.")
    parser.add_argument(
        "--param-pool",
        type=Path,
        default=Path("stats/param_pool.json"),
        help="Parameter pool JSON produced by build_param_pool.py (param_values mode).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    meta = load_meta(args.stats)
    function_profiles = _build_function_profiles(meta)
    all_functions = list(meta.keys())
    builder = QUESTION_BUILDERS[args.mode]
    produced = 0
    param_pool = load_param_pool(args.param_pool) if args.mode == "param_values" else ParamPool(None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as sink:
        for record in load_jsonl(args.input):
            available = parse_available_tools(record)
            for msg_idx, fc in iter_function_calls(record):
                func_name = fc["name"]
                if args.mode == "available":
                    result = builder(
                        func_name,
                        available,
                        all_functions,
                        args.negatives,
                        function_profiles,
                    )
                elif args.mode == "params":
                    result = builder(func_name, meta, args.negatives)
                elif args.mode == "param_values":
                    result = builder(func_name, fc, meta, args.negatives, param_pool)
                else:
                    result = None

                if not result:
                    continue

                entry = {
                    "mode": args.mode,
                    "question": result["question"],
                    "options": result["options"],
                    "answer": result["answer"],
                    "function_name": func_name,
                    "record_uuid": record.get("uuid"),
                    "message_index": msg_idx,
                }
                json.dump(entry, sink, ensure_ascii=False)
                sink.write("\n")
                produced += 1
                if args.max_samples and produced >= args.max_samples:
                    break
            if args.max_samples and produced >= args.max_samples:
                break

    print(f"[INFO] Generated {produced} HAS-API entries using mode={args.mode}.")


if __name__ == "__main__":
    main()

