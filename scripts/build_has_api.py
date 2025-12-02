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
from pathlib import Path
from typing import Iterable


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


def split_prefix(name: str, levels: int = 2) -> str:
    parts = name.split("-")
    return "-".join(parts[:levels]) if len(parts) >= levels else name


def build_cluster_map(functions: Iterable[str]) -> dict[str, list[str]]:
    clusters: dict[str, list[str]] = {}
    for name in functions:
        prefix = split_prefix(name)
        clusters.setdefault(prefix, []).append(name)
    return clusters


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


def question_cluster(func_name: str, clusters: dict[str, list[str]], num_neg: int) -> dict | None:
    prefix = split_prefix(func_name)
    candidates = [f for f in clusters.get(prefix, []) if f != func_name]
    if not candidates:
        return None
    k = min(num_neg, len(candidates))
    negs = random.sample(candidates, k)
    options = negs + [func_name]
    random.shuffle(options)
    return {
        "question": f"Among similar APIs ({prefix}), which tool should be called?",
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


QUESTION_BUILDERS = {
    "random": question_random,
    "cluster": question_cluster,
    "available": question_available,
    "params": question_params,
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
    cluster_map = build_cluster_map(all_functions)

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
                elif args.mode == "cluster":
                    result = builder(func_name, cluster_map, args.negatives)
                elif args.mode == "available":
                    result = builder(func_name, available)
                elif args.mode == "params":
                    result = builder(func_name, meta, args.negatives)
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

