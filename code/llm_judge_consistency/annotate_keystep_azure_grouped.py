#!/usr/bin/env python3
"""
Same self-consistency check as annotate_keystep_azure.py (--pass_tag pass1/2/3),
but grouped by TASK instead of by pass: for each task, fire all N passes back
to back before moving to the next task. This keeps the identical call1/call2
prompt prefixes close together in time so Azure's prompt cache can actually
hit, instead of spacing repeats of the same prompt 150 items apart.

Usage:
  export AZURE_OPENAI_API_KEY="..."
  export AZURE_OPENAI_ENDPOINT="https://<your-resource>.services.ai.azure.com/api/projects/<your-project>"

  python annotate_keystep_azure_grouped.py --deployment gpt-5.2 \
      --pass_tags gpt52_pass1,gpt52_pass2,gpt52_pass3
"""

import argparse
import json
import time
from pathlib import Path

from annotate_keystep_azure import (
    load_or_fresh, save_annotation, run_invariant_analysis, run_judge,
)
from annotate_keystep_gemini import SOURCE_CONFIG, load_webarena_trajectory

import os

SCRIPT_DIR = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deployment", required=True)
    ap.add_argument("--pass_tags", required=True, help="comma-separated, e.g. gpt52_pass1,gpt52_pass2,gpt52_pass3")
    ap.add_argument("--manifest", default=str(SCRIPT_DIR / "result_consistency" / "sample_150.json"))
    ap.add_argument("--base_dir", default=str(SCRIPT_DIR / "data"))
    ap.add_argument("--temperature", default="0.0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not endpoint or not api_key:
        raise RuntimeError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.")
    temperature = None if args.temperature.lower() == "none" else float(args.temperature)

    pass_tags = args.pass_tags.split(",")
    base_dir = Path(args.base_dir).expanduser().resolve()
    with open(args.manifest) as f:
        manifest = json.load(f)
    items = manifest["items"]
    if args.limit:
        items = items[:args.limit]

    # per-pass_tag: {source_key: annot_data}, opened lazily
    annot_state = {pt: {} for pt in pass_tags}
    log_fh = {}
    for pt in pass_tags:
        out_dir = SCRIPT_DIR / "result_consistency" / pt
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "logs").mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_fh[pt] = open(out_dir / "logs" / f"azure_{pt}_{ts}.jsonl", "w", buffering=1)

    print(f"Deployment : {args.deployment}")
    print(f"Pass tags  : {pass_tags}  (grouped by task for cache locality)")
    print(f"Manifest   : {args.manifest}  ({len(items)} items)")
    print()

    success = {pt: 0 for pt in pass_tags}
    skipped = {pt: 0 for pt in pass_tags}
    errors = {pt: 0 for pt in pass_tags}

    for it in items:
        model = it["model"]
        task_id = it["taskId"]
        task_key = it["taskKey"]
        source_key = f"webarena/{model}"
        dataset, _model, results_rel, _sample_rel, _rj, chat_dir_rel = SOURCE_CONFIG[source_key]
        results_dir = base_dir / results_rel
        chat_dir = base_dir / chat_dir_rel if chat_dir_rel else None

        traj = load_webarena_trajectory(task_id, model, results_dir, chat_dir)
        if traj is None or not traj["steps"]:
            print(f"[{model}] task_id={task_id}  SKIP (no trajectory)")
            for pt in pass_tags:
                skipped[pt] += 1
            continue
        n_steps = len(traj["steps"])

        print(f"[{model}] task_id={task_id}  ({n_steps} steps)")

        for pt in pass_tags:
            out_dir = SCRIPT_DIR / "result_consistency" / pt
            annot_path = out_dir / f"{dataset}_{model}_azure.json"
            if source_key not in annot_state[pt]:
                annot_state[pt][source_key] = (annot_path, load_or_fresh(annot_path, dataset, model))
            _, annot_data = annot_state[pt][source_key]

            if task_key in annot_data.get("items", {}):
                print(f"  [{pt}] SKIP (already done)")
                skipped[pt] += 1
                continue

            print(f"  [{pt}] call1...", end="", flush=True)
            step_analyses, u1 = run_invariant_analysis(traj, args.deployment, endpoint, api_key,
                                                        args.delay, temperature)
            log_fh[pt].write(json.dumps({"event": "call1", "task_key": task_key, "success": step_analyses is not None, **u1}) + "\n")
            if step_analyses is None:
                print("  ERROR (call1 failed)")
                errors[pt] += 1
                continue
            time.sleep(args.delay)

            print("call2...", end="", flush=True)
            result, u2 = run_judge(traj, step_analyses, n_steps, args.deployment, endpoint, api_key,
                                   args.delay, temperature)
            log_fh[pt].write(json.dumps({"event": "call2", "task_key": task_key, "success": result is not None, **u2}) + "\n")
            if result is None:
                print("  ERROR (call2 failed)")
                errors[pt] += 1
                continue

            save_annotation(annot_data, annot_path, task_id, dataset, model, result,
                            step_analyses, annotator_tag=f"azure_{args.deployment}_{pt}")
            print(f"  keyStep={result['keyStep']}  cause={result['failureCause']}")
            success[pt] += 1
            time.sleep(args.delay)

    for pt in pass_tags:
        log_fh[pt].close()
    print()
    for pt in pass_tags:
        print(f"[{pt}] success={success[pt]} skipped={skipped[pt]} errors={errors[pt]}")


if __name__ == "__main__":
    main()
