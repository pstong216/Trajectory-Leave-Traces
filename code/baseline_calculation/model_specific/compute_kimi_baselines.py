"""
compute_kimi_baselines.py — Baseline UQ scorer for kimi/webarena trajectories.

Data layout (different from mind2web result.json convention):

  verbalized_dir/
    uq_samples/<task_id>/
      step_<k>_logprobs.json   → {task_id, step, samples: [{tokens, token_logprobs}]}
      step_<k>_samples.json    → {task_id, step, samples: [{raw_prediction, parsed_action}]}
    actions/<task_id>.json     → {task_id, score, actions}   (ground truth)

  entropy_dir/
    uq_samples/<task_id>/
      step_<k>_samples.json    → {task_id, step, samples: [{raw_prediction}]}  (10 samples)

Six methods:
  1. Verbalized confidence at step k
  2. Action token mean log-prob at step k
  3. Full output mean log-prob at step k
  4. Cumulative mean output log-prob (0→k)
  5. Action entropy at step k
  6. Cumulative mean action entropy (0→k)

Usage:
  python compute_kimi_baselines.py \\
    --verbalized_dir /path/to/results_kimi_verbalized/trajectory_1 \\
    --entropy_dir    /path/to/kimi \\
    --output_dir     /path/to/output \\
    [--model kimi] [--benchmark webarena]
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

BASELINE_DIR = Path(__file__).resolve().parents[1]  # release copy: model_specific/ is 1 level under baseline_calculation/, not 2
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from keystep_labels import (as_success_text, load_keystep_map,
                            load_keystep_map_entropy_source, label_step_from_keystep)


# ---------------------------------------------------------------------------
# Signal helpers (same logic as compute_baselines.py)
# ---------------------------------------------------------------------------

def parse_confidence(text: str) -> float | None:
    m = re.search(r'Confidence:\s*([\d.]+)', text)
    return float(m.group(1)) if m else None


def extract_action_text(text: str) -> str | None:
    m = re.search(r'```(.+?)```', text, re.DOTALL)
    return m.group(1).strip() if m else None


def action_token_logprob_mean(tokens: list[str], token_logprobs: list[float]) -> float | None:
    text = ''.join(tokens)
    m = re.search(r'```(.+?)```', text, re.DOTALL)
    if not m:
        return None
    action_start, action_end = m.start(1), m.end(1)
    char_pos = 0
    action_lps = []
    for tok, lp in zip(tokens, token_logprobs):
        tok_end = char_pos + len(tok)
        if tok_end > action_start and char_pos < action_end:
            action_lps.append(lp)
        char_pos = tok_end
    return -sum(action_lps) / len(action_lps) if action_lps else None


def full_output_logprob_mean(token_logprobs: list[float]) -> float | None:
    if not token_logprobs:
        return None
    return -sum(token_logprobs) / len(token_logprobs)


def action_entropy(samples: list[str]) -> float | None:
    actions = []
    for s in samples:
        a = extract_action_text(s)
        if a is not None:
            actions.append(re.sub(r'\s+', ' ', a).strip())
    if not actions:
        return None
    counts = Counter(actions)
    total = len(actions)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# ---------------------------------------------------------------------------
# Data loading (kimi-specific file layout)
# ---------------------------------------------------------------------------

def _step_index(filename: str) -> int:
    """Extract step number from 'step_<k>_logprobs.json' or 'step_<k>_samples.json'."""
    m = re.search(r'step_(\d+)_', filename)
    return int(m.group(1)) if m else -1


def load_verbalized_task(task_dir: str) -> list[dict]:
    """
    Load per-step verbalized data for one task.
    Returns list of dicts with keys: step, tokens, token_logprobs, raw_prediction.
    """
    logprob_files = {
        _step_index(f): os.path.join(task_dir, f)
        for f in os.listdir(task_dir)
        if f.endswith('_logprobs.json') and _step_index(f) >= 0
    }
    sample_files = {
        _step_index(f): os.path.join(task_dir, f)
        for f in os.listdir(task_dir)
        if f.endswith('_samples.json') and _step_index(f) >= 0
    }

    steps = []
    for k in sorted(set(logprob_files) | set(sample_files)):
        entry = {'step': k, 'tokens': [], 'token_logprobs': [], 'raw_prediction': ''}

        if k in logprob_files:
            with open(logprob_files[k]) as f:
                d = json.load(f)
            samples = d.get('samples', [])
            if samples:
                s = samples[0]
                entry['tokens'] = s.get('tokens', [])
                entry['token_logprobs'] = s.get('token_logprobs', [])

        if k in sample_files:
            with open(sample_files[k]) as f:
                d = json.load(f)
            samples = d.get('samples', [])
            if samples:
                entry['raw_prediction'] = samples[0].get('raw_prediction', '')

        steps.append(entry)

    return steps


def load_entropy_task(task_dir: str) -> list[dict]:
    """
    Load per-step multi-sample data for one task (entropy computation).
    Returns list of dicts with keys: step, raw_predictions (list of str).
    """
    sample_files = {
        _step_index(f): os.path.join(task_dir, f)
        for f in os.listdir(task_dir)
        if f.endswith('_samples.json') and _step_index(f) >= 0
    }

    steps = []
    for k in sorted(sample_files):
        with open(sample_files[k]) as f:
            d = json.load(f)
        preds = [s.get('raw_prediction', '') for s in d.get('samples', [])]
        steps.append({'step': k, 'raw_predictions': preds})

    return steps


def load_ground_truth(actions_dir: str) -> dict[str, float]:
    """Load score from actions/<task_id>.json files. Returns task_id_str → score."""
    gt = {}
    if not os.path.isdir(actions_dir):
        return gt
    for fname in os.listdir(actions_dir):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(actions_dir, fname)) as f:
            d = json.load(f)
        task_id = str(d.get('task_id', fname.replace('.json', '')))
        gt[task_id] = d.get('score', float('nan'))
    return gt


# ---------------------------------------------------------------------------
# Per-task scoring
# ---------------------------------------------------------------------------

def score_verbalized_task_steps(steps: list[dict]) -> list[dict]:
    rows = []
    m3_running = []
    for entry in steps:
        tokens = entry['tokens']
        token_logprobs = entry['token_logprobs']
        text = entry['raw_prediction']

        # Skip steps with no useful data
        if not text and not tokens:
            continue

        m1 = parse_confidence(text)
        m2 = action_token_logprob_mean(tokens, token_logprobs)
        m3 = full_output_logprob_mean(token_logprobs)

        m3_running.append(m3 if m3 is not None else float('nan'))
        valid = [v for v in m3_running if not math.isnan(v)]
        m4 = sum(valid) / len(valid) if valid else None

        rows.append({
            'step': entry['step'],
            'm1_verbalized_conf': m1,
            'm2_action_logprob_mean': m2,
            'm3_output_logprob_mean': m3,
            'm4_cumulative_output_logprob_mean': m4,
        })
    return rows


def score_entropy_task_steps(steps: list[dict]) -> list[dict]:
    rows = []
    m5_running = []
    for entry in steps:
        preds = entry['raw_predictions']
        if not preds or all(not p for p in preds):
            continue

        m5 = action_entropy(preds)

        m5_running.append(m5 if m5 is not None else float('nan'))
        valid = [v for v in m5_running if not math.isnan(v)]
        m6 = sum(valid) / len(valid) if valid else None

        rows.append({
            'step': entry['step'],
            'm5_action_entropy': m5,
            'm6_cumulative_action_entropy': m6,
        })
    return rows


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def build_per_step_df(
    verbalized_uq_dir: str,
    entropy_uq_dir: str,
    ground_truth: dict[str, float],
    model: str,
    benchmark: str,
    annotation_dir: str | None = None,
    ground_truth_entropy: dict[str, float] | None = None,
) -> pd.DataFrame:
    verb_rows: dict[tuple, dict] = {}
    source = f'{benchmark}/{model}'
    keystep_map = load_keystep_map(source, annotation_dir) if benchmark == 'webarena' else {}
    # Separate label for M5/M6 (2026-07-30, bug): see
    # load_keystep_map_entropy_source()'s docstring.
    keystep_map_entropy = load_keystep_map_entropy_source(source, annotation_dir) if benchmark == 'webarena' else {}
    # Ground truth is ALSO rollout-specific (2026-07-31, flagged):
    # verbalized_dir and entropy_dir are two independent kimi rollouts, each
    # with its own actions/<id>.json score; they disagree on 17/163 (10.4%)
    # of tasks. success_entropy (M5/M6) must use the entropy rollout's own
    # outcome, not the verbalized rollout's -- falls back to `ground_truth`
    # only if a task has no entry of its own in the entropy dict.
    ground_truth_entropy = ground_truth_entropy or {}
    if os.path.isdir(verbalized_uq_dir):
        for task_id in os.listdir(verbalized_uq_dir):
            task_dir = os.path.join(verbalized_uq_dir, task_id)
            if not os.path.isdir(task_dir):
                continue
            steps = load_verbalized_task(task_dir)
            for row in score_verbalized_task_steps(steps):
                verb_rows[(task_id, row['step'])] = row

    ent_rows: dict[tuple, dict] = {}
    if os.path.isdir(entropy_uq_dir):
        for task_id in os.listdir(entropy_uq_dir):
            task_dir = os.path.join(entropy_uq_dir, task_id)
            if not os.path.isdir(task_dir):
                continue
            steps = load_entropy_task(task_dir)
            for row in score_entropy_task_steps(steps):
                ent_rows[(task_id, row['step'])] = row

    # Restrict to the VERBALIZED rollout's own keys only (2026-07-30,
    # bug: was a union of verb_rows | ent_rows -- same fix as
    # compute_baselines.py / compute_webarena_step_baselines.py).
    all_keys = set(verb_rows)
    records = []
    for key in sorted(all_keys, key=lambda x: (int(x[0]), x[1])):
        task_id, step = key
        v = verb_rows.get(key, {})
        e = ent_rows.get(key, {})
        score = ground_truth.get(str(task_id), float('nan'))
        final_success = True if score == 1.0 else (False if score == 0.0 else None)

        score_ent = ground_truth_entropy.get(str(task_id))
        if score_ent is None:
            score_ent = score  # entropy rollout has no own ground truth for this task; fall back
        final_success_entropy = True if score_ent == 1.0 else (False if score_ent == 0.0 else None)

        records.append({
            'model': model,
            'benchmark': benchmark,
            'task_id': task_id,
            'step': step,
            'success': as_success_text(
                label_step_from_keystep(final_success, step, keystep_map.get(str(task_id)))
            ),
            'success_entropy': as_success_text(
                label_step_from_keystep(final_success_entropy, step, keystep_map_entropy.get(str(task_id)))
            ),
            'm1_verbalized_conf': v.get('m1_verbalized_conf'),
            'm2_action_logprob_mean': v.get('m2_action_logprob_mean'),
            'm3_output_logprob_mean': v.get('m3_output_logprob_mean'),
            'm4_cumulative_output_logprob_mean': v.get('m4_cumulative_output_logprob_mean'),
            'm5_action_entropy': e.get('m5_action_entropy'),
            'm6_cumulative_action_entropy': e.get('m6_cumulative_action_entropy'),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbalized_dir', required=True,
                        help='results_kimi_verbalized/trajectory_1 directory')
    parser.add_argument('--entropy_dir', required=True,
                        help='kimi/ directory (10 samples per step)')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model', default='kimi')
    parser.add_argument('--benchmark', default='webarena')
    parser.add_argument('--annotation_dir', default=None,
                        help='Optional annotation/result dir with key-step labels.')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    verbalized_uq_dir = os.path.join(args.verbalized_dir, 'uq_samples')
    entropy_uq_dir = os.path.join(args.entropy_dir, 'uq_samples')
    actions_dir = os.path.join(args.verbalized_dir, 'actions')
    actions_dir_entropy = os.path.join(args.entropy_dir, 'actions')

    print(f'Verbalized uq_samples: {verbalized_uq_dir}')
    verb_tasks = len(os.listdir(verbalized_uq_dir)) if os.path.isdir(verbalized_uq_dir) else 0
    print(f'  -> {verb_tasks} task folders')

    print(f'Entropy uq_samples: {entropy_uq_dir}')
    ent_tasks = len(os.listdir(entropy_uq_dir)) if os.path.isdir(entropy_uq_dir) else 0
    print(f'  -> {ent_tasks} task folders')

    gt = load_ground_truth(actions_dir)
    if gt:
        success = sum(1 for v in gt.values() if v == 1.0)
        print(f'Ground truth (verbalized rollout, for M1-4/M7/M11-14): {len(gt)} tasks | '
              f'{success} success / {len(gt)-success} failure')
    else:
        print('No ground truth found')

    gt_ent = load_ground_truth(actions_dir_entropy)
    if gt_ent:
        se = sum(1 for v in gt_ent.values() if v == 1.0)
        disagree = sum(1 for t in (set(gt) & set(gt_ent)) if gt[t] != gt_ent[t])
        print(f'Ground truth (entropy rollout, for M5/M6):             {len(gt_ent)} tasks | '
              f'{se} success / {len(gt_ent)-se} failure  ({disagree} disagree with verbalized rollout)')

    print('Computing per-step scores...')
    per_step_df = build_per_step_df(
        verbalized_uq_dir, entropy_uq_dir, gt, args.model, args.benchmark, args.annotation_dir,
        ground_truth_entropy=gt_ent,
    )
    print(f'  -> {len(per_step_df)} step rows across {per_step_df["task_id"].nunique()} tasks')

    step_out = os.path.join(args.output_dir, 'per_step_scores.csv')
    per_step_df.to_csv(step_out, index=False)

    print(f'\nDone.')
    print(f'  per_step_scores.csv : {len(per_step_df)} rows  -> {step_out}')

    score_cols = [c for c in per_step_df.columns if c.startswith('m') and c[1].isdigit()]
    print('\nStep-level score summary:')
    print(per_step_df[score_cols].describe().loc[['count', 'mean', 'std', 'min', 'max']].to_string())

    if 'success' in per_step_df.columns:
        print('\nSuccess rate (step-level):', per_step_df['success'].eq('success').mean().round(3))
        print('\nMean step scores by task outcome:')
        print(per_step_df.groupby('success')[score_cols].mean().to_string())


if __name__ == '__main__':
    main()
