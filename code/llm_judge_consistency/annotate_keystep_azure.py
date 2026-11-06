#!/usr/bin/env python3
"""
LLM-as-Judge annotation — Azure OpenAI backend, restricted to a fixed
sample manifest (for the Gemini-vs-rerun / self-consistency check).

Reuses the exact same two-call pipeline, taxonomy, and trajectory
loaders as annotate_keystep_gemini.py — only the API call layer differs
(Azure OpenAI chat completions instead of Gemini).

Only processes task keys listed in --manifest (default:
result_consistency/sample_150.json), so it never touches the full
1030-item corpus.

Run the SAME command twice with two different --pass_tag values
(e.g. pass1 / pass2) to get two independent judge runs on the same
150 items for a self-consistency check.

Usage:
  export AZURE_OPENAI_API_KEY="..."
  export AZURE_OPENAI_ENDPOINT="https://<your-resource>.services.ai.azure.com/api/projects/<your-project>"

  python annotate_keystep_azure.py --deployment gpt-5.4-nano --pass_tag pass1
  python annotate_keystep_azure.py --deployment gpt-5.4-nano --pass_tag pass2
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from annotate_keystep_gemini import (
    FAILURE_CAUSES,
    INVARIANT_SYSTEM_INSTRUCTION,
    JUDGE_SYSTEM_INSTRUCTION,
    SOURCE_CONFIG,
    build_judge_prompt,
    load_webarena_trajectory,
)

SCRIPT_DIR = Path(__file__).parent


# ── Plain-text invariant prompt (webarena has no images, unlike mind2web) ──────

def build_invariant_text(traj: dict) -> str:
    steps = traj["steps"]
    n = len(steps)
    lines = [
        f"Task:          {traj['task']}",
        f"Final outcome: {traj['final_result']}",
        f"Total steps:   {n}  (0-indexed: 0 to {n - 1})",
        "",
    ]
    for s in steps:
        k = s["step"]
        lines.append(f"--- Step {k} ---")
        if s.get("url"):
            lines.append(f"URL: {s['url']}")
        obs = s.get("observation_text", "")
        if obs:
            lines.append(f"Accessibility tree:\n{obs}")
        lines.append(f"Action taken:    {s['action']}")
        lines.append(f"Agent reasoning: {s['reasoning_baseline'][:600]}")
        if s.get("reasoning_sample0"):
            lines.append(f"Sampled reasoning: {s['reasoning_sample0'][:400]}")
        lines.append("")
    return "\n".join(lines)


# ── Azure OpenAI caller ──────────────────────────────────────────────────────

def call_azure(messages: list, deployment: str, endpoint: str, api_key: str,
                temperature: float | None = 0.0,
                retries: int = 8, delay: float = 5.0) -> tuple[dict | None, dict]:
    # Azure AI Foundry project endpoint — unified v1 route, no api-version needed.
    url = f"{endpoint.rstrip('/')}/openai/v1/chat/completions"
    usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
             "latency_s": 0.0, "attempts": 0}
    body = {
        "model": deployment,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 8000,
    }
    if temperature is not None:
        body["temperature"] = temperature

    for attempt in range(1, retries + 1):
        usage["attempts"] = attempt
        t0 = time.perf_counter()
        try:
            r = requests.post(url, headers={"api-key": api_key, "Content-Type": "application/json"},
                               json=body, timeout=120)
            usage["latency_s"] = round(time.perf_counter() - t0, 3)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", delay * attempt))
                print(f"    [attempt {attempt}/{retries}] 429 rate-limited — waiting {wait:.0f}s...", flush=True)
                if attempt < retries:
                    time.sleep(wait)
                continue
            r.raise_for_status()
            resp = r.json()
            um = resp.get("usage", {})
            usage["input_tokens"] = um.get("prompt_tokens", 0)
            usage["output_tokens"] = um.get("completion_tokens", 0)
            usage["cached_tokens"] = um.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            text = resp["choices"][0]["message"]["content"].strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text), usage
        except Exception as e:
            usage["latency_s"] = round(time.perf_counter() - t0, 3)
            wait = delay * attempt
            print(f"    [attempt {attempt}/{retries}] Error: {e}", flush=True)
            if attempt < retries:
                time.sleep(wait)

    return None, usage


def run_invariant_analysis(traj: dict, deployment: str, endpoint: str, api_key: str,
                            delay: float, temperature: float | None) -> tuple[list | None, dict]:
    messages = [
        {"role": "system", "content": INVARIANT_SYSTEM_INSTRUCTION},
        {"role": "user", "content": build_invariant_text(traj)},
    ]
    result, usage = call_azure(messages, deployment, endpoint, api_key, temperature=temperature, delay=delay)
    if result is None:
        return None, usage
    analyses = result.get("step_analyses")
    if not isinstance(analyses, list):
        return None, usage
    return analyses, usage


def run_judge(traj: dict, step_analyses: list, n_steps: int,
              deployment: str, endpoint: str, api_key: str, delay: float,
              temperature: float | None) -> tuple[dict | None, dict]:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_INSTRUCTION},
        {"role": "user", "content": build_judge_prompt(traj, step_analyses)},
    ]
    result, usage = call_azure(messages, deployment, endpoint, api_key, temperature=temperature, delay=delay)
    if result is None:
        return None, usage
    if "keyStep" not in result or "failureCause" not in result:
        return None, usage

    fc_raw = result.get("failureCause") or ""
    if fc_raw not in FAILURE_CAUSES:
        for fc in FAILURE_CAUSES:
            if fc.lower() in fc_raw.lower():
                result["failureCause"] = fc
                break
        else:
            result["failureCause"] = fc_raw

    ks = result.get("keyStep")
    result["keyStep"] = max(0, min(int(ks), n_steps - 1)) if ks is not None else n_steps - 1
    return result, usage


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_or_fresh(annot_path: Path, dataset: str, model: str) -> dict:
    if annot_path.exists():
        with open(annot_path) as f:
            return json.load(f)
    return {
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset, "model": model,
        "sourceFiles": ["azure_annotated"], "items": {},
    }


def save_annotation(annot_data: dict, annot_path: Path, task_id: str, dataset: str,
                     model: str, result: dict, step_analyses: list, annotator_tag: str):
    task_key = f"{dataset}::{model}::{task_id}"
    annot_data["items"][task_key] = {
        "taskKey": task_key, "dataset": dataset, "model": model, "taskId": str(task_id),
        "keyStep": result.get("keyStep"),
        "failureSteps": result.get("failureSteps", []),
        "failureCause": result.get("failureCause", ""),
        "notes": result.get("notes", ""),
        "step_analyses": step_analyses,
        "annotator": annotator_tag,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }
    annot_data["updatedAt"] = datetime.now(timezone.utc).isoformat()
    annot_path.write_text(json.dumps(annot_data, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deployment", required=True, help="Azure OpenAI deployment name")
    ap.add_argument("--pass_tag", required=True, help="e.g. pass1 / pass2 — keeps independent runs separate")
    ap.add_argument("--manifest", default=str(SCRIPT_DIR / "result_consistency" / "sample_150.json"))
    ap.add_argument("--base_dir", default=str(SCRIPT_DIR / "data"))
    ap.add_argument("--api_version", default="2024-12-01-preview")
    ap.add_argument("--temperature", default="0.0",
                    help="Float, or 'none' to omit (required for reasoning-tier models "
                         "like gpt-5.6-sol that reject non-default temperature)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not endpoint or not api_key:
        raise RuntimeError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.")

    temperature = None if args.temperature.lower() == "none" else float(args.temperature)

    base_dir = Path(args.base_dir).expanduser().resolve()
    with open(args.manifest) as f:
        manifest = json.load(f)
    items = manifest["items"]
    if args.limit:
        items = items[:args.limit]

    out_dir = SCRIPT_DIR / "result_consistency" / args.pass_tag
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"azure_{args.pass_tag}_{ts}.jsonl"
    log_fh = open(log_path, "w", buffering=1)

    print(f"Deployment    : {args.deployment}")
    print(f"Pass tag      : {args.pass_tag}")
    print(f"Manifest      : {args.manifest}  ({len(items)} items)")
    print(f"Output dir    : {out_dir}")
    print()

    success, skipped, errors = 0, 0, 0

    # group by (dataset, model) so we open/reuse one annot_data + chat_dir per source
    by_source: dict[str, list] = {}
    for it in items:
        by_source.setdefault(it["model"], []).append(it)

    for model, model_items in by_source.items():
        source_key = f"webarena/{model}"
        dataset, _model, results_rel, _sample_rel, _rj, chat_dir_rel = SOURCE_CONFIG[source_key]
        results_dir = base_dir / results_rel
        chat_dir = base_dir / chat_dir_rel if chat_dir_rel else None

        annot_path = out_dir / f"{dataset}_{model}_azure.json"
        annot_data = load_or_fresh(annot_path, dataset, model)
        already_done = set(annot_data.get("items", {}).keys())

        for it in model_items:
            task_id = it["taskId"]
            task_key = it["taskKey"]
            print(f"[{model}] task_id={task_id}", end="  ", flush=True)

            if task_key in already_done:
                print("SKIP (already done)")
                skipped += 1
                continue

            traj = load_webarena_trajectory(task_id, model, results_dir, chat_dir)
            if traj is None or not traj["steps"]:
                print("SKIP (no trajectory)")
                skipped += 1
                continue

            n_steps = len(traj["steps"])
            print(f"({n_steps} steps)", end="  ", flush=True)

            if args.dry_run:
                print("[dry run]")
                continue

            print("call1...", end="", flush=True)
            step_analyses, u1 = run_invariant_analysis(traj, args.deployment, endpoint, api_key,
                                                        args.delay, temperature)
            write_log = {"event": "call1", "task_key": task_key, "success": step_analyses is not None, **u1}
            log_fh.write(json.dumps(write_log) + "\n")
            if step_analyses is None:
                print("  ERROR (call1 failed)")
                errors += 1
                continue
            time.sleep(args.delay)

            print("call2...", end="", flush=True)
            result, u2 = run_judge(traj, step_analyses, n_steps, args.deployment, endpoint, api_key,
                                   args.delay, temperature)
            write_log2 = {"event": "call2", "task_key": task_key, "success": result is not None, **u2}
            log_fh.write(json.dumps(write_log2) + "\n")
            if result is None:
                print("  ERROR (call2 failed)")
                errors += 1
                continue

            save_annotation(annot_data, annot_path, task_id, dataset, model, result,
                            step_analyses, annotator_tag=f"azure_{args.deployment}_{args.pass_tag}")
            print(f"  keyStep={result['keyStep']}  cause={result['failureCause']}")
            success += 1
            time.sleep(args.delay)

    log_fh.close()
    print()
    print(f"Done. success={success} skipped={skipped} errors={errors}")


if __name__ == "__main__":
    main()
