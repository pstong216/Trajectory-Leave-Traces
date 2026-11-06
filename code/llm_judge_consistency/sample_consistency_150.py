#!/usr/bin/env python3
"""
Stratified random sample of 150 task keys from the existing Gemini
annotation results, proportional to each source's size, for an
LLM-judge self-consistency check (rerun the judge twice on the same
150 items and measure agreement).

Sampling is seeded for reproducibility. Output manifest:
  result_consistency/sample_150.json
"""

import json
import random
from pathlib import Path

SEED = 42
TARGET_N = 150

SCRIPT_DIR = Path(__file__).parent
RESULT_DIR = SCRIPT_DIR / "result"
OUT_DIR = SCRIPT_DIR / "result_consistency"

SOURCES = [
    "webarena_claude_gemini.json",
    "webarena_claude_m_gemini.json",
    "webarena_gpt_5_2_gemini.json",
    "webarena_gpt_5_2_m_gemini.json",
    "webarena_gptnano_gemini.json",
    "webarena_gptnano_m_gemini.json",
    "webarena_kimi_gemini.json",
    "webarena_kimi_m_gemini.json",
    "webarena_qwen30b_gemini.json",
    "webarena_qwen30b_m_gemini.json",
]


def largest_remainder_alloc(sizes: dict, total: int) -> dict:
    """Proportional integer allocation across sources summing exactly to `total`."""
    grand = sum(sizes.values())
    raw = {k: v * total / grand for k, v in sizes.items()}
    floor = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(floor.values())
    order = sorted(raw, key=lambda k: raw[k] - floor[k], reverse=True)
    for k in order[:remainder]:
        floor[k] += 1
    return floor


def main():
    rng = random.Random(SEED)

    source_keys = {}
    for fname in SOURCES:
        path = RESULT_DIR / fname
        with open(path) as f:
            data = json.load(f)
        keys = sorted(data["items"].keys())
        source_keys[fname] = keys

    sizes = {k: len(v) for k, v in source_keys.items()}
    alloc = largest_remainder_alloc(sizes, TARGET_N)

    manifest = {"seed": SEED, "target_n": TARGET_N, "by_source": {}, "items": []}

    for fname in SOURCES:
        keys = source_keys[fname]
        n = alloc[fname]
        sampled = rng.sample(keys, n) if n <= len(keys) else keys
        manifest["by_source"][fname] = {
            "total_available": len(keys),
            "n_sampled": len(sampled),
        }
        for task_key in sampled:
            dataset, model, task_id = task_key.split("::")
            manifest["items"].append({
                "taskKey": task_key,
                "dataset": dataset,
                "model": model,
                "taskId": task_id,
                "source_file": fname,
            })

    manifest["items"].sort(key=lambda x: x["taskKey"])

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "sample_150.json"
    out_path.write_text(json.dumps(manifest, indent=2))

    print(f"Total pool     : {sum(sizes.values())} items across {len(SOURCES)} sources")
    print(f"Sampled        : {len(manifest['items'])} items (target {TARGET_N})")
    print()
    for fname in SOURCES:
        info = manifest["by_source"][fname]
        print(f"  {fname:38s} {info['n_sampled']:3d} / {info['total_available']}")
    print()
    print(f"Manifest written to: {out_path}")


if __name__ == "__main__":
    main()
