#!/usr/bin/env python3
"""
LLM-as-Judge annotation workflow using Gemini API.

Two-call pipeline per task (inspired by AgentRx):
  Call 1 — Dynamic invariant analysis:
    Given the full trajectory, evaluate each step: what should have happened,
    what did happen, and was any error later recovered?
  Call 2 — Judge:
    Given the task + action log + violation report from Call 1, apply the
    3-step root-cause algorithm to identify keyStep + classify failure cause.

Outputs:  <this_script_dir>/result/<dataset>_<model>_gemini.json
          <this_script_dir>/logs/<source>_<timestamp>.jsonl

Usage:
  export GEMINI_API_KEY="your_key"

  # Point --base_dir at your local copy of results_total/
  python annotate_keystep_gemini.py --source mind2web/claude --base_dir /path/to/results_total
  python annotate_keystep_gemini.py --source webarena/claude --base_dir /path/to/results_total
  python annotate_keystep_gemini.py --source mind2web/claude --base_dir /path/to/results_total --dry_run
  python annotate_keystep_gemini.py --source mind2web/claude --base_dir /path/to/results_total --limit 10
  python annotate_keystep_gemini.py --source mind2web/claude --base_dir /path/to/results_total --start_from 50

  # If results_total/ is the parent of this script's directory, --base_dir can be omitted:
  python annotate_keystep_gemini.py --source mind2web/claude
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from google import genai
from google.genai import types

# ── Failure cause taxonomy ─────────────────────────────────────────────────────

FAILURE_CAUSES = [
    # Memory
    "Memory:OverSimplification",
    "Memory:Hallucination",
    "Memory:RetrievalFailure",
    # Reflection
    "Reflection:ProgressMisassessment",
    "Reflection:OutcomeMisinterpretation",
    "Reflection:CausalMisattribution",
    "Reflection:Hallucination",
    # Planning
    "Planning:InefficientPlanning",
    "Planning:ConstraintIgnorance",
    "Planning:ImpossibleAction",
    # Action
    "Action:PlanningActionDisconnect",
    "Action:FormatError",
    "Action:ParameterError",
    # System
    "System:StepLimitExhaustion",
    "System:ToolExecutionError",
    "System:EnvironmentError",
]

# ── Call 1 system instruction: dynamic invariant analysis ──────────────────────
# Static across all tasks → benefits from Gemini prompt caching.
# Input contract: only raw trajectory data, NO UQ features.

INVARIANT_SYSTEM_INSTRUCTION = """You are a web agent trajectory analyst.

Your task: given a FAILED web agent trajectory, analyze each step to determine
whether the agent's action was correct at that point, and if not, whether the
error was later recovered.

For EACH step output one entry:
  "step"              : step index (integer, 0-based)
  "expected_action"   : one sentence — what the agent SHOULD do at this step
                        given the task goal and all prior context
  "correct"           : true if the agent's action matches the expected; false otherwise
  "error_description" : if correct=false, describe concisely what went wrong
                        (set to null if correct=true)
  "recovered"         : if correct=false, did the agent successfully get back on
                        track in a LATER step?
                        true  = yes, the agent recognized and fixed this error
                        false = no, this error persisted toward the final failure
                        null  = not applicable (correct=true)

RECOVERY definition: after this error the agent took a different approach that
made genuine progress toward the goal. An attempt to fix the error that also
fails is NOT recovery.

Output a single JSON object:
{
  "step_analyses": [
    {
      "step": <int>,
      "expected_action": <string>,
      "correct": <bool>,
      "error_description": <string or null>,
      "recovered": <bool or null>
    },
    ...   (one entry per step, in order)
  ]
}

Output ONLY valid JSON, no markdown, no extra text."""


# ── Call 2 system instruction: root-cause judge ────────────────────────────────
# Static across all tasks → benefits from Gemini prompt caching.

JUDGE_SYSTEM_INSTRUCTION = """You are an expert annotator for web agent failure analysis.
Always respond with valid JSON only.

You will be given:
  1. A trajectory summary: task goal, final outcome, and the action taken at each step.
  2. A VIOLATION REPORT from prior step-by-step analysis, listing which steps had
     errors and whether each error was later recovered by the agent.

ROOT-CAUSE DETECTION ALGORITHM (from AgentRx)
=============================================
Use this exact procedure to identify the keyStep:

  Step 1 — Scan the VIOLATION REPORT from the lowest step number upward and
            record the first violation (recovered or unrecovered).
  Step 2 — Using the ACTION LOG, verify whether this violation was truly
            resolved later: look for evidence that the agent recognized the
            error and successfully got back on track.
            If yes → Resolved. If no such evidence → Not resolved.
  Step 3 — If Resolved: continue scanning to the next violation, repeat Step 2.
            If Not resolved: this step is the keyStep — stop.

The violation report's recovered/unrecovered labels are a starting hint, but you
must verify against the ACTION LOG. A violation marked [RECOVERED] that you
judge to be unresolved should still be treated as Not resolved.

The keyStep is the EARLIEST step whose error was permanent and directly led to
the final failure. Steps with errors that were genuinely corrected do NOT
qualify, regardless of how bad they looked at the time.

FAILURE CAUSE TAXONOMY
======================
Use exactly one code from the list below (format: Module:ErrorType).

── MEMORY ──────────────────────────────────────────────────────────────────

Memory:OverSimplification
  The agent summarizes or simplifies past trajectory information too crudely,
  losing critical details needed for the current decision.
  keyStep = first step where acting on the oversimplified memory leads to an
  action whose damage cannot be undone.

Memory:Hallucination
  The agent recalls events or states that never happened, filling gaps with
  fabricated history and acting on it as if it were real.
  keyStep = first step where an action is taken based on the hallucinated
  memory, making recovery from the false assumption impossible.

Memory:RetrievalFailure
  Relevant information from earlier in the trajectory exists but is not
  retrieved when needed — causing the agent to repeat a failed approach or
  ignore known context (e.g. stuck in an action loop).
  keyStep = first step where failing to retrieve prior context leads to an
  unrecoverable decision.

── REFLECTION ───────────────────────────────────────────────────────────────

Reflection:ProgressMisassessment
  The agent incorrectly evaluates its own progress — too optimistic, too
  pessimistic, or fails to recognise that it is looping or has already
  completed (or irreversibly missed) the goal.
  keyStep = first step where the wrong progress assessment drives an action
  that makes recovery impossible.

Reflection:OutcomeMisinterpretation
  The agent executes an action but misreads the immediate result or
  environment feedback, drawing the wrong conclusion about what just happened
  (e.g. thinks a form was submitted when it was not).
  keyStep = first step where the misinterpreted outcome leads to an action
  that makes recovery impossible.

Reflection:CausalMisattribution
  The agent correctly notices a failure but blames the wrong cause, leading
  to misguided corrections that perpetuate the failure.
  keyStep = first step where acting on the wrong causal diagnosis makes
  recovery impossible.

Reflection:Hallucination
  The agent reflects on events or results that never occurred (e.g. claims a
  button was clicked when it was not, or that a page loaded when it did not).
  keyStep = first step where acting on the hallucinated observation makes
  recovery impossible.

── PLANNING ─────────────────────────────────────────────────────────────────

Planning:InefficientPlanning
  The agent's overall strategy is fundamentally wrong or unnecessarily
  convoluted; it cannot lead to task completion regardless of execution
  quality.
  keyStep = first step where the flawed plan is committed to and recovery
  becomes impossible.

Planning:ConstraintIgnorance
  The agent ignores real constraints (step budget, page structure, available
  actions) when forming its plan, pursuing an approach that was never viable.
  keyStep = first step where ignoring the constraint leads to an action that
  makes recovery impossible.

Planning:ImpossibleAction
  The agent plans a step that is physically or logically impossible given
  current preconditions (e.g. clicking a button that does not exist yet,
  submitting a form before filling required fields).
  keyStep = first step where the impossible action is attempted and the
  failure makes recovery impossible.

── ACTION ───────────────────────────────────────────────────────────────────

Action:PlanningActionDisconnect
  The plan is correct and the action type is correct, but the action
  systematically fails to execute (e.g. always targets the wrong element, or
  the environment cannot process the action at all).
  keyStep = first step where this disconnect permanently blocks progress.

Action:FormatError
  The plan is correct but the wrong action type is used
  (e.g. click instead of hover; Enter instead of Tab).
  keyStep = first step where this format mismatch becomes unrecoverable.

Action:ParameterError
  The action type is correct but the parameters are wrong
  (e.g. wrong element ID used repeatedly; malformed input value).
  keyStep = first step where the parameter error permanently blocks progress.

── SYSTEM ───────────────────────────────────────────────────────────────────

System:StepLimitExhaustion
  The agent behaves reasonably throughout but simply runs out of the 30-step
  budget before the task is complete.
  keyStep = index of the LAST step in the trajectory.

System:ToolExecutionError
  An external tool, API call, or web element misbehaves or errors in a way
  the agent cannot work around, causing downstream failure.
  keyStep = first step where the tool error occurs and makes recovery
  impossible.

System:EnvironmentError
  The web environment breaks expected rules (page crash, unexpected redirect,
  state change not caused by the agent) making the task impossible through
  no fault of the agent.
  keyStep = first step where the environment error makes recovery impossible.

OUTPUT FORMAT
=============
Output a JSON object with exactly these fields:

  "keyStep"      : integer — identified by the root-cause algorithm above.
                   For StepLimit: last step index.
                   For all others: earliest unrecovered error step.
  "failureSteps" : list of integers — ALL steps that contributed to the failure.
  "failureCause" : string — exactly one code from the taxonomy above.
  "notes"        : string — one or two sentences naming the unrecovered error and
                   explaining why earlier recovered errors do not qualify.

Output ONLY valid JSON, no markdown, no extra text."""


# ── Data loaders ───────────────────────────────────────────────────────────────
# Input contract: only raw trajectory data is loaded.
# UQ features (U_sem, U_change, U_ground, U_cval, C_S, R_T) are intentionally
# NOT loaded — annotation must use only observable agent behavior.

def load_mind2web_trajectory(task_id: int, model: str,
                              results_dir: Path) -> dict | None:
    """
    Load a mind2web trajectory.
    Returns: {task, final_result, steps: [{step, action, reasoning_baseline,
              reasoning_sample0, screenshot_path}]}
    screenshot_path points to trajectory/<k>_full_screenshot.png (may not exist).
    """
    result_path = results_dir / str(task_id) / "result.json"
    if not result_path.exists():
        return None

    with open(result_path) as f:
        r = json.load(f)

    traj_dir = results_dir / str(task_id) / "trajectory"
    steps = []
    n = len(r.get("action_history", []))
    for k in range(n):
        step = {
            "step":               k,
            "action":             r["action_history"][k],
            "reasoning_baseline": r["thoughts"][k] if k < len(r.get("thoughts", [])) else "",
            "reasoning_sample0":  "",
            "screenshot_path":    str(traj_dir / f"{k}_full_screenshot.png"),
        }
        step_samples = r.get("step_samples", [])
        if k < len(step_samples) and step_samples[k]:
            step["reasoning_sample0"] = step_samples[k][0]
        steps.append(step)

    return {
        "task":         r.get("task", ""),
        "final_result": r.get("final_result_response", ""),
        "steps":        steps,
    }


def _load_chat_obs(chat_dir: Path, task_id: str) -> tuple[dict, list]:
    """Resolve chat data from chat_dir for task_id. Two layouts:
      - chat_dir/<task_id>/chat.json  (claude/kimi layout)
      - chat_dir/<task_id>.json       (flat layout: gpt52/gptnano/qwen30b)
    Returns (chat_dict, chat_steps). Returns ({}, []) if file missing or empty.
    """
    candidates = [
        chat_dir / task_id / "chat.json",
        chat_dir / f"{task_id}.json",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            with open(p) as f:
                chat = json.load(f)
            return chat, chat.get("steps", [])
    return {}, []


def load_webarena_trajectory(task_id: str, model: str,
                              results_dir: Path, chat_dir: Path | None) -> dict | None:
    """
    Load a webarena trajectory. Three layouts:

    Layout A — results_dir/trajectory_1/chat/<id>/chat.json + trajectory_1/actions/<id>.json
      Full data: intent, observation_text, reasoning from full_prompt.

    Layout B — results_dir/actions/<id>.json  (actions-only, no chat)
      + chat_dir/<id>/chat.json if chat_dir is provided (flat layout, no trajectory_1 subpath)
      obs_text comes from chat_dir; actions/reasoning from actions file + uq_samples.

    Layout C — results_dir/actions/<id>.json only, no chat_dir
      No observation or intent available.
    """
    # ── Layout A: trajectory_1/actions + chat from chat_dir or trajectory_1/chat ─
    actions_path_a = results_dir / "trajectory_1" / "actions" / f"{task_id}.json"

    if actions_path_a.exists():
        # Prefer chat_dir if provided; fall back to results_dir/trajectory_1/chat
        if chat_dir:
            chat, chat_steps = _load_chat_obs(chat_dir, task_id)
        else:
            chat, chat_steps = _load_chat_obs(
                results_dir / "trajectory_1" / "chat", task_id)
        with open(actions_path_a) as f: actions = json.load(f)

        action_list = actions.get("actions", [])
        n = chat.get("num_steps", len(chat_steps))
        steps = []
        for k in range(n):
            cs = chat_steps[k] if k < len(chat_steps) else {}
            reasoning_baseline = ""
            for msg in reversed(cs.get("full_prompt") or []):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            c.get("text", "") for c in content if isinstance(c, dict))
                    reasoning_baseline = content
                    break
            steps.append({
                "step":               k,
                "url":                cs.get("url", ""),
                "observation_text":   cs.get("observation_text", ""),
                "action":             action_list[k] if k < len(action_list) else "",
                "reasoning_baseline": reasoning_baseline,
                "reasoning_sample0":  "",
            })
        return {
            "task":         chat.get("intent", ""),
            "final_result": action_list[-1] if action_list else "",
            "steps":        steps,
        }

    # ── Layout B/C: actions/<id>.json (top-level or under trajectory_1/) ────────
    actions_path_b = results_dir / "actions" / f"{task_id}.json"
    if not actions_path_b.exists():
        actions_path_b = results_dir / "trajectory_1" / "actions" / f"{task_id}.json"
    if not actions_path_b.exists():
        return None

    with open(actions_path_b) as f:
        actions = json.load(f)
    action_list = actions.get("actions", [])

    # Load obs_text + intent from flat chat_dir if provided
    intent = ""
    chat_steps_b: list = []
    if chat_dir:
        chat_b, chat_steps_b = _load_chat_obs(chat_dir, task_id)
        intent = chat_b.get("intent", "")

    steps = []
    for k, action in enumerate(action_list):
        reasoning_sample0 = ""
        uq_dir = results_dir / "uq_samples" / task_id
        sample_file = uq_dir / f"step_{k}_samples.json"
        if sample_file.exists():
            with open(sample_file) as f:
                sdata = json.load(f)
            samples = sdata.get("samples", [])
            if samples:
                reasoning_sample0 = samples[0].get("raw_prediction", "")

        cs = chat_steps_b[k] if k < len(chat_steps_b) else {}
        steps.append({
            "step":               k,
            "url":                cs.get("url", ""),
            "observation_text":   cs.get("observation_text", ""),
            "action":             action,
            "reasoning_baseline": "",
            "reasoning_sample0":  reasoning_sample0,
        })

    return {
        "task":         intent,
        "final_result": action_list[-1] if action_list else "",
        "steps":        steps,
    }


# ── Prompt builders ────────────────────────────────────────────────────────────

def build_invariant_contents(traj: dict, dataset: str) -> list:
    """Call 1 contents: returns a list of types.Part for the Gemini API.

    mind2web  — text per step  +  screenshot image (PNG) per step
    webarena  — text per step  +  full accessibility tree (text, no images)

    UQ features are never included. Page state comes from screenshots or
    the accessibility tree only.
    """
    steps = traj["steps"]
    n     = len(steps)

    header = (
        f"Task:          {traj['task']}\n"
        f"Final outcome: {traj['final_result']}\n"
        f"Total steps:   {n}  (0-indexed: 0 to {n-1})\n"
    )
    parts = [types.Part.from_text(text=header)]

    for s in steps:
        k = s["step"]
        lines = [f"--- Step {k} ---"]

        if dataset == "webarena":
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

        parts.append(types.Part.from_text(text="\n".join(lines)))

        # mind2web: attach the per-step screenshot
        if dataset == "mind2web":
            screenshot_path = Path(s.get("screenshot_path", ""))
            if screenshot_path.exists():
                parts.append(
                    types.Part.from_bytes(
                        data=screenshot_path.read_bytes(),
                        mime_type="image/png",
                    )
                )

    return parts


def build_judge_prompt(traj: dict, step_analyses: list) -> str:
    """Call 2 user message: compact action log + structured violation report."""
    steps = traj["steps"]
    n     = len(steps)

    action_by_step = {s["step"]: s["action"] for s in steps}

    lines = [
        f"Task:          {traj['task']}",
        f"Final outcome: {traj['final_result']}",
        f"Total steps:   {n}  (0-indexed: 0 to {n-1})",
        "",
        "ACTION LOG",
        "----------",
    ]
    for s in steps:
        lines.append(f"  Step {s['step']:2d}: {s['action']}")

    lines += ["", "VIOLATION REPORT", "----------------"]

    violations = [a for a in step_analyses if not a.get("correct", True)]
    if not violations:
        lines.append("  (no violations detected — may be StepLimit)")
    else:
        for v in violations:
            tag = "[RECOVERED]  " if v.get("recovered") else "[UNRECOVERED]"
            lines.append(f"  {tag} Step {v['step']}")
            lines.append(f"    Expected : {v['expected_action']}")
            lines.append(f"    Error    : {v['error_description']}")
            lines.append("")

    lines += [
        "",
        "Apply the root-cause algorithm and taxonomy to identify the keyStep and "
        "classify the failure cause.",
    ]

    return "\n".join(lines)


# ── Gemini caller ──────────────────────────────────────────────────────────────

_gemini_client: genai.Client | None = None

def _get_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client()
    return _gemini_client


def _rate_limit_wait(error_str: str, fallback: float = 60.0) -> float:
    """Extract the suggested retryDelay (seconds) from a 429 error string."""
    m = re.search(r"retry.*?(\d+(?:\.\d+)?)s", error_str, re.IGNORECASE)
    return float(m.group(1)) + 3.0 if m else fallback   # +3s safety buffer


def call_gemini(contents,
                system_instruction: str | None = None,
                cache_name: str | None = None,
                model_name: str = "gemini-3.5-flash",
                retries: int = 8, delay: float = 5.0) -> tuple[dict | None, dict]:
    """
    If cache_name is given, uses explicit cached content (system_instruction already
    embedded in the cache — do NOT also pass system_instruction).
    Otherwise falls back to passing system_instruction directly.

    On 429 RESOURCE_EXHAUSTED the suggested retryDelay is parsed from the error
    and used as the actual wait time (instead of a short fixed delay).

    Returns (parsed_result, usage) where usage has keys:
      input_tokens, output_tokens, cached_tokens, latency_s, attempts
    """
    client = _get_client()
    usage: dict = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
                   "latency_s": 0.0, "attempts": 0}

    cfg_kwargs: dict = {"temperature": 0.0, "response_mime_type": "application/json"}
    if cache_name:
        cfg_kwargs["cached_content"] = cache_name
    elif system_instruction:
        cfg_kwargs["system_instruction"] = system_instruction

    for attempt in range(1, retries + 1):
        usage["attempts"] = attempt
        t0 = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            usage["latency_s"] = round(time.perf_counter() - t0, 3)
            um = response.usage_metadata
            if um:
                usage["input_tokens"]  = getattr(um, "prompt_token_count",           0) or 0
                usage["output_tokens"] = getattr(um, "candidates_token_count",        0) or 0
                usage["cached_tokens"] = getattr(um, "cached_content_token_count",    0) or 0
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text), usage

        except Exception as e:
            usage["latency_s"] = round(time.perf_counter() - t0, 3)
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = _rate_limit_wait(err_str)
                print(f"    [attempt {attempt}/{retries}] 429 rate-limited — "
                      f"waiting {wait:.0f}s...", flush=True)
            else:
                wait = delay * attempt
                print(f"    [attempt {attempt}/{retries}] Error: {e}", flush=True)
            if attempt < retries:
                time.sleep(wait)

    return None, usage


def setup_caches(model_name: str) -> tuple[str | None, str | None]:
    """Create explicit Gemini prompt caches for both system instructions.

    Returns (invariant_cache_name, judge_cache_name).
    Either value is None if cache creation failed (caller falls back to inline
    system_instruction).
    """
    client = _get_client()

    def try_create(instruction: str, label: str) -> str | None:
        try:
            cache = client.caches.create(
                model=model_name,
                config=types.CreateCachedContentConfig(
                    system_instruction=instruction,
                    ttl="3600s",
                ),
            )
            print(f"  cache OK  ({label}): {cache.name}")
            return cache.name
        except Exception as e:
            print(f"  cache FAIL ({label}): {e} — using inline system_instruction")
            return None

    print("Setting up prompt caches...")
    inv_cache   = try_create(INVARIANT_SYSTEM_INSTRUCTION, "invariant")
    judge_cache = try_create(JUDGE_SYSTEM_INSTRUCTION,     "judge")
    print()
    return inv_cache, judge_cache


def delete_caches(*cache_names: str | None):
    client = _get_client()
    for name in cache_names:
        if name:
            try:
                client.caches.delete(name=name)
            except Exception:
                pass


def run_invariant_analysis(traj: dict, dataset: str,
                            model_name: str, delay: float,
                            cache_name: str | None = None) -> tuple[list | None, dict]:
    """Call 1. Returns (step_analyses, usage)."""
    contents = build_invariant_contents(traj, dataset)
    result, usage = call_gemini(
        contents,
        system_instruction=None if cache_name else INVARIANT_SYSTEM_INSTRUCTION,
        cache_name=cache_name,
        model_name=model_name,
        delay=delay,
    )
    if result is None:
        return None, usage
    analyses = result.get("step_analyses")
    if not isinstance(analyses, list):
        return None, usage
    return analyses, usage


def run_judge(traj: dict, step_analyses: list,
              n_steps: int, model_name: str, delay: float,
              cache_name: str | None = None) -> tuple[dict | None, dict]:
    """Call 2. Returns (result, usage)."""
    prompt = build_judge_prompt(traj, step_analyses)
    result, usage = call_gemini(
        prompt,
        system_instruction=None if cache_name else JUDGE_SYSTEM_INSTRUCTION,
        cache_name=cache_name,
        model_name=model_name,
        delay=delay,
    )
    if result is None:
        return None, usage

    if "keyStep" not in result or "failureCause" not in result:
        return None, usage

    # Normalise failureCause to known codes
    fc_raw = result.get("failureCause") or ""
    if fc_raw not in FAILURE_CAUSES:
        for fc in FAILURE_CAUSES:
            if fc.lower() in fc_raw.lower():
                result["failureCause"] = fc
                break
        else:
            result["failureCause"] = fc_raw

    # Clamp keyStep to valid range
    ks = result.get("keyStep")
    result["keyStep"] = max(0, min(int(ks), n_steps - 1)) if ks is not None else n_steps - 1
    return result, usage


# ── Logging ────────────────────────────────────────────────────────────────────

def open_log(log_dir: Path, source_slug: str) -> tuple[IO, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"{source_slug}_{ts}.jsonl"
    return open(path, "w", buffering=1), path  # line-buffered


def write_log(log_fh: IO, record: dict):
    log_fh.write(json.dumps(record) + "\n")


# ── Annotation I/O ─────────────────────────────────────────────────────────────

def load_or_fresh_annot_data(annot_path: Path, dataset: str, model: str) -> dict:
    if annot_path.exists():
        with open(annot_path) as f:
            data = json.load(f)
        print(f"Resuming       : {annot_path}  ({len(data.get('items', {}))} already done)")
        return data
    return {
        "updatedAt":   datetime.now(timezone.utc).isoformat(),
        "dataset":     dataset,
        "model":       model,
        "sourceFiles": ["gemini_annotated"],
        "items":       {},
    }


def save_annotation(annot_data: dict, annot_path: Path,
                    task_id: str, dataset: str, model: str,
                    result: dict, step_analyses: list):
    task_key = f"{dataset}::{model}::{task_id}"
    annot_data["items"][task_key] = {
        "taskKey":       task_key,
        "dataset":       dataset,
        "model":         model,
        "taskId":        str(task_id),
        "keyStep":       result.get("keyStep"),
        "failureSteps":  result.get("failureSteps", []),
        "failureCause":  result.get("failureCause", ""),
        "notes":         result.get("notes", ""),
        "step_analyses": step_analyses,
        "annotator":     "gemini_invariant",
        "updatedAt":     datetime.now(timezone.utc).isoformat(),
    }
    annot_data["updatedAt"] = datetime.now(timezone.utc).isoformat()
    annot_path.write_text(json.dumps(annot_data, indent=2))


# ── Task discovery ─────────────────────────────────────────────────────────────

def get_failed_task_ids_mind2web(results_dir: Path, result_json_path: Path) -> list[str]:
    """Read failed task IDs from the ground-truth result JSON (e.g. claude_result.json)."""
    if not result_json_path.exists():
        raise FileNotFoundError(f"Result JSON not found: {result_json_path}")
    with open(result_json_path) as f:
        data = json.load(f)
    failed = {
        str(t["task_id"]) for t in data.get("tasks", [])
        if t.get("status") == "failure"
    }
    available = {
        d for d in os.listdir(results_dir)
        if (results_dir / d / "result.json").exists()
    }
    return sorted(failed & available, key=lambda x: int(x))


def get_failed_task_ids_webarena(results_dir: Path) -> list[str]:
    """Supports both trajectory_1/actions/ and top-level actions/ layouts."""
    actions_dir = results_dir / "trajectory_1" / "actions"
    if not actions_dir.exists():
        actions_dir = results_dir / "actions"
    failed = []
    if not actions_dir.exists():
        return failed
    for fname in os.listdir(actions_dir):
        p = actions_dir / fname
        if not p.suffix == ".json":
            continue
        with open(p) as f:
            d = json.load(f)
        if d.get("score", 1) == 0:
            failed.append(fname.replace(".json", ""))
    return sorted(failed, key=lambda x: int(x))


# ── Source configuration ───────────────────────────────────────────────────────
# Each entry: (dataset, model, results_dir_rel, sample_dir_rel, result_json_rel, chat_dir_rel)
# All *_rel paths are relative to --base_dir.
#
# When using the bundled data/ folder, pass --base_dir data
# The data/ folder has a flat layout:  data/<source>/actions/<id>.json
#                                       data/<source>/chat/<id>.json
#
# result_json_rel : mind2web only — path to <model>_result.json (ground-truth pass/fail)
# chat_dir_rel    : webarena only — directory containing per-task chat JSON files

SOURCE_CONFIG = {
    # ── webarena verbalized ─────────────────────────────────────────────────────
    # Use with:  --source webarena/claude --base_dir data
    "webarena/claude":    ("webarena", "claude",    "claude",    None, None, "claude/chat"),
    "webarena/kimi":      ("webarena", "kimi",      "kimi",      None, None, "kimi/chat"),
    "webarena/gpt_5_2":   ("webarena", "gpt_5_2",   "gpt_5_2",   None, None, "gpt_5_2/chat"),
    "webarena/gptnano":   ("webarena", "gptnano",   "gptnano",   None, None, "gptnano/chat"),
    "webarena/qwen30b":   ("webarena", "qwen30b",   "qwen30b",   None, None, "qwen30b/chat"),
    # ── webarena multisampled ───────────────────────────────────────────────────
    "webarena/claude_m":  ("webarena", "claude_m",  "claude_m",  None, None, "claude_m/chat"),
    "webarena/kimi_m":    ("webarena", "kimi_m",    "kimi_m",    None, None, "kimi_m/chat"),
    "webarena/gpt_5_2_m": ("webarena", "gpt_5_2_m", "gpt_5_2_m", None, None, "gpt_5_2_m/chat"),
    "webarena/gptnano_m": ("webarena", "gptnano_m", "gptnano_m", None, None, "gptnano_m/chat"),
    "webarena/qwen30b_m": ("webarena", "qwen30b_m", "qwen30b_m", None, None, "qwen30b_m/chat"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",       required=True,
                    choices=list(SOURCE_CONFIG.keys()),
                    help="e.g. mind2web/claude  or  webarena/qwen30b")
    ap.add_argument("--base_dir",     default=None,
                    help="Path to results_total/ (the directory containing mind2web/ and webarena/). "
                         "Defaults to the parent of this script's directory.")
    ap.add_argument("--gemini_model", default="gemini-3.5-flash",
                    help="Gemini model name (default: gemini-3.5-flash)")
    ap.add_argument("--limit",        type=int, default=None,
                    help="Max number of tasks to annotate in this run")
    ap.add_argument("--start_from",   type=str, default=None,
                    help="Skip all tasks before this task_id (inclusive start)")
    ap.add_argument("--dry_run",      action="store_true",
                    help="Print both system instructions + prompts for first task, no API call")
    ap.add_argument("--delay",        type=float, default=2.0,
                    help="Seconds to wait between API calls (default 2.0)")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError("Set GEMINI_API_KEY environment variable.")

    # base_dir = results_total/ (where mind2web/ and webarena/ live)
    if args.base_dir:
        base_dir = Path(args.base_dir).expanduser().resolve()
    else:
        base_dir = Path(__file__).parent.parent  # annotation/ → results_total/

    dataset, model, results_rel, sample_rel, result_json_rel, chat_dir_rel = SOURCE_CONFIG[args.source]
    results_dir      = base_dir / results_rel
    sample_dir       = base_dir / sample_rel       if sample_rel      else None
    result_json_path = base_dir / result_json_rel  if result_json_rel else None
    chat_dir         = base_dir / chat_dir_rel     if chat_dir_rel    else None

    # Output goes into annotation/result/ and annotation/logs/ (next to this script)
    script_dir = Path(__file__).parent
    annot_dir  = script_dir / "result"
    log_dir    = script_dir / "logs"
    annot_dir.mkdir(parents=True, exist_ok=True)
    annot_path = annot_dir / f"{dataset}_{model}_gemini.json"

    annot_data = load_or_fresh_annot_data(annot_path, dataset, model)

    if dataset == "mind2web":
        task_ids = get_failed_task_ids_mind2web(results_dir, result_json_path)
    else:
        task_ids = get_failed_task_ids_webarena(results_dir)

    if args.start_from is not None:
        try:
            idx = task_ids.index(str(args.start_from))
            task_ids = task_ids[idx:]
        except ValueError:
            raise ValueError(f"--start_from task_id '{args.start_from}' not found in failed task list")

    if args.limit:
        task_ids = task_ids[:args.limit]

    source_slug = args.source.replace("/", "_")

    print(f"Source        : {args.source}")
    print(f"Base dir      : {base_dir}")
    print(f"Tasks         : {len(task_ids)}")
    print(f"Gemini model  : {args.gemini_model}")
    print(f"Dry run       : {args.dry_run}")
    print(f"Pipeline      : Call 1 (invariant analysis) → Call 2 (judge)")
    print()

    if args.dry_run:
        sep = "=" * 70
        print(sep)
        print("CALL 1 — INVARIANT SYSTEM INSTRUCTION (cached)")
        print(sep)
        print(INVARIANT_SYSTEM_INSTRUCTION)
        print()
        print(sep)
        print("CALL 2 — JUDGE SYSTEM INSTRUCTION (cached)")
        print(sep)
        print(JUDGE_SYSTEM_INSTRUCTION)
        print()

    success, skipped, errors = 0, 0, 0
    total_in, total_out, total_cached = 0, 0, 0

    inv_cache_name = judge_cache_name = None
    if not args.dry_run:
        inv_cache_name, judge_cache_name = setup_caches(args.gemini_model)

    log_fh, log_path = open_log(log_dir, source_slug)
    write_log(log_fh, {
        "event": "run_start",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": args.source,
        "model": args.gemini_model,
        "n_tasks": len(task_ids),
        "inv_cache": inv_cache_name,
        "judge_cache": judge_cache_name,
    })
    print(f"Log           : {log_path}")
    print()

    already_done = set(annot_data.get("items", {}).keys())

    try:
        for i, task_id in enumerate(task_ids, 1):
            task_key = f"{dataset}::{model}::{task_id}"
            print(f"[{i}/{len(task_ids)}] task_id={task_id}", end="  ", flush=True)

            if task_key in already_done:
                print("SKIP (already annotated)")
                skipped += 1
                continue

            if dataset == "mind2web":
                traj = load_mind2web_trajectory(int(task_id), model, results_dir)
            else:
                traj = load_webarena_trajectory(task_id, model, results_dir, chat_dir)

            if traj is None or not traj["steps"]:
                print("SKIP (no trajectory)")
                skipped += 1
                write_log(log_fh, {
                    "event": "skip", "task_id": str(task_id),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                continue

            n_steps = len(traj["steps"])
            print(f"({n_steps} steps)", end="  ", flush=True)

            if args.dry_run:
                print()
                print("-" * 70)
                mode = "screenshots" if dataset == "mind2web" else "accessibility tree"
                print(f"CALL 1 USER MESSAGE — invariant contents ({mode})")
                print("-" * 70)
                parts = build_invariant_contents(traj, dataset)
                n_images = sum(1 for p in parts if hasattr(p, "inline_data") and p.inline_data)
                for p in parts:
                    if hasattr(p, "inline_data") and p.inline_data:
                        print(f"  [IMAGE: {p.inline_data.mime_type}]")
                    else:
                        print(p.text if hasattr(p, "text") else str(p))
                print(f"  → {len(parts)} parts total ({n_images} images)")
                print()
                print("-" * 70)
                print("CALL 2 USER MESSAGE — judge prompt")
                print("(shown with placeholder violations since Call 1 hasn't run)")
                print("-" * 70)
                placeholder = [
                    {"step": k, "correct": False,
                     "expected_action": "<would be filled by Call 1>",
                     "error_description": "<would be filled by Call 1>",
                     "recovered": False}
                    for k in range(min(2, n_steps))
                ]
                print(build_judge_prompt(traj, placeholder))
                print()
                continue

            # ── Call 1: dynamic invariant analysis ────────────────────────────
            print("call1...", end="", flush=True)
            t_call1 = datetime.now(timezone.utc).isoformat()
            step_analyses, u1 = run_invariant_analysis(traj, dataset,
                                                        args.gemini_model, args.delay,
                                                        cache_name=inv_cache_name)
            write_log(log_fh, {
                "event": "call1", "task_id": str(task_id), "timestamp": t_call1,
                "success": step_analyses is not None,
                "input_tokens": u1["input_tokens"], "output_tokens": u1["output_tokens"],
                "cached_tokens": u1["cached_tokens"], "latency_s": u1["latency_s"],
                "attempts": u1["attempts"],
            })
            if step_analyses is None:
                print("  ERROR (invariant call failed)")
                errors += 1
                continue
            time.sleep(args.delay)

            total_in     += u1["input_tokens"]
            total_out    += u1["output_tokens"]
            total_cached += u1["cached_tokens"]

            n_violations  = sum(1 for a in step_analyses if not a.get("correct", True))
            n_unrecovered = sum(1 for a in step_analyses
                                if not a.get("correct", True) and not a.get("recovered"))
            print(f"  violations={n_violations} unrecovered={n_unrecovered}", end="  ", flush=True)

            # ── Call 2: judge ──────────────────────────────────────────────────
            print("call2...", end="", flush=True)
            t_call2 = datetime.now(timezone.utc).isoformat()
            result, u2 = run_judge(traj, step_analyses, n_steps, args.gemini_model, args.delay,
                                   cache_name=judge_cache_name)
            write_log(log_fh, {
                "event": "call2", "task_id": str(task_id), "timestamp": t_call2,
                "success": result is not None,
                "input_tokens": u2["input_tokens"], "output_tokens": u2["output_tokens"],
                "cached_tokens": u2["cached_tokens"], "latency_s": u2["latency_s"],
                "attempts": u2["attempts"],
                "keyStep": result.get("keyStep") if result else None,
                "failureCause": result.get("failureCause") if result else None,
            })
            if result is None:
                print("  ERROR (judge call failed)")
                errors += 1
                continue

            total_in     += u2["input_tokens"]
            total_out    += u2["output_tokens"]
            total_cached += u2["cached_tokens"]

            save_annotation(annot_data, annot_path, task_id, dataset, model,
                            result, step_analyses)
            tok_msg = (f"  in={u1['input_tokens']+u2['input_tokens']}"
                       f"  out={u1['output_tokens']+u2['output_tokens']}"
                       f"  cached={u1['cached_tokens']+u2['cached_tokens']}")
            print(f"  keyStep={result['keyStep']}  cause={result['failureCause']}{tok_msg}")
            success += 1

            time.sleep(args.delay)

    finally:
        write_log(log_fh, {
            "event": "run_end",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": success, "skipped": skipped, "errors": errors,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_cached_tokens": total_cached,
        })
        log_fh.close()
        delete_caches(inv_cache_name, judge_cache_name)

    print()
    print(f"Done.  success={success}  skipped={skipped}  errors={errors}")
    print(f"Tokens — input={total_in}  output={total_out}  cached={total_cached}")
    if not args.dry_run:
        print(f"Output : {annot_path}")
        print(f"Log    : {log_path}")


if __name__ == "__main__":
    main()
