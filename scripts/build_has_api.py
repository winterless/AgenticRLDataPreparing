#!/usr/bin/env python3
"""
Generate HAS-API style multiple-choice data without using LLMs.

Example:
    python scripts/build_has_api.py \
        -i data/toucan_1000.jsonl \
        --stats stats/function_meta.json \
        -o data/has_api_random.jsonl \
        --mode random
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable


if hasattr(sys, "set_int_max_str_digits"):
    try:
        sys.set_int_max_str_digits(0)  # disable limit for large JSON ints
    except Exception:
        pass


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_meta(stats_path: Path) -> dict[str, dict]:
    with stats_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


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


def iter_function_calls(record: dict):
    messages = record.get("messages")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return
    if not isinstance(messages, list):
        return
    for idx, msg in enumerate(messages):
        fc = msg.get("function_call")
        if fc and fc.get("name"):
            yield idx, fc


def question_random(func_name: str, all_funcs: list[str], num_neg: int) -> dict | None:
    pool = [f for f in all_funcs if f != func_name]
    if not pool:
        return None
    k = min(num_neg, len(pool))
    negs = random.sample(pool, k)
    options = negs + [func_name]
    random.shuffle(options)
    return {
        "question": "Which tool should the agent call next?",
        "options": options,
        "answer": func_name,
    }


def question_available(func_name: str, available: list[str]) -> dict | None:
    # use available_tools list as options if includes correct
    if func_name not in available or len(available) < 2:
        return None
    options = list(dict.fromkeys(available))  # deduplicate but preserve order
    return {
        "question": "Select the proper tool from the available tool list.",
        "options": options,
        "answer": func_name,
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


def _format_arg_values(args: dict) -> str:
    pairs = []
    for key in sorted(args.keys()):
        pairs.append(f"{key}={json.dumps(args[key], ensure_ascii=False)}")
    return "; ".join(pairs)


def _mutate_value(value, prop: dict | None):
    prop = prop or {}
    if isinstance(value, bool):
        return not value
    if isinstance(value, float):
        float_deltas = [-2.0, -1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0, 2.0]
        delta = random.choice(float_deltas)
        candidate = value + delta
        if candidate != value:
            return round(candidate, 6)
    if isinstance(value, int):
        delta = random.choice([-5, -2, -1, 1, 2, 5])
        candidate = value + delta
        if candidate != value:
            return candidate
    if isinstance(value, str):
        enums = prop.get("enum")
        if enums:
            choices = [e for e in enums if e != value]
            if choices:
                return random.choice(choices)
        # fallback: append suffix or swap case
        suffixes = ["_alt", "_backup", "_test", "_v2"]
        suffix = random.choice(suffixes)
        candidate = value + suffix
        if candidate != value:
            return candidate
    return None


def _parse_arguments(fc: dict) -> dict | None:
    args = fc.get("arguments")
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed
            return None
        except json.JSONDecodeError:
            return None
    return None


def question_param_values(func_name: str, fc: dict, meta: dict[str, dict], num_neg: int) -> dict | None:
    args = _parse_arguments(fc)
    if not args:
        return None

    info = meta.get(func_name, {})
    params = ((info.get("function") or {}).get("parameters") or info.get("parameters") or {})
    properties = params.get("properties") or {}

    correct_option = _format_arg_values(args)
    variations: set[str] = set()
    attempts = 0
    max_attempts = num_neg * 5

    while len(variations) < num_neg and attempts < max_attempts:
        attempts += 1
        key_candidates = list(args.keys())
        if not key_candidates:
            break
        target = random.choice(key_candidates)
        mutated_value = _mutate_value(args[target], properties.get(target))
        if mutated_value is None:
            continue
        mutated = dict(args)
        mutated[target] = mutated_value
        option = _format_arg_values(mutated)
        if option != correct_option:
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
    "random": question_random,
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
    parser.add_argument("--negatives", type=int, default=5, help="Number of negative options.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit on outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    meta = load_meta(args.stats)
    all_functions = list(meta.keys())
    builder = QUESTION_BUILDERS[args.mode]
    produced = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as sink:
        for record in load_jsonl(args.input):
            available = parse_available_tools(record)
            for msg_idx, fc in iter_function_calls(record):
                func_name = fc["name"]
                if args.mode == "random":
                    result = builder(func_name, all_functions, args.negatives)
                elif args.mode == "available":
                    result = builder(func_name, available)
                elif args.mode == "params":
                    result = builder(func_name, meta, args.negatives)
                elif args.mode == "param_values":
                    result = builder(func_name, fc, meta, args.negatives)
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

