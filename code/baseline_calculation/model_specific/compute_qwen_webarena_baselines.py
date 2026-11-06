"""
compute_qwen_webarena_baselines.py — DEPRECATED 2026-07-30, do not use.

The docstring below claimed "no verbalized/logprob data for qwen30b webarena"
-- that was false. webarena/results_qwen30b_verbalized/ has real, usable
verbalized-confidence and logprob data (confirmed: 1914/1917 step samples
have a parseable "Confidence: X" string; matching *_logprobs.json files
exist for M2-M4). Whoever wrote this script either never checked or the data
was added later and this was never updated -- either way, M1-M4/M7 for
webarena/qwen30b were silently missing from every baseline table for no
real reason.

Use compute_webarena_step_baselines.py instead (same one used for
claude/gpt_5_2/gptnano) with:
  --verbalized_dir <results_root>/webarena/results_qwen30b_verbalized
  --entropy_dir    <results_root>/webarena/qwen30b
It handles this data correctly (verified: 165 tasks, 1917 step rows, M1-M4
populated). This file is kept only for history; do not run it.

--- original (inaccurate) docstring below ---
compute_qwen_webarena_baselines.py — Step-level baseline UQ for webarena/qwen30b.

Data layout (entropy-only; no verbalized/logprob data for qwen30b webarena):
  entropy_dir/trajectory_1/uq_samples/<task_id>/step_<k>_samples.json
    → {task_id, step, samples: [{raw_prediction}]}  (10 samples)
  entropy_dir/trajectory_1/actions/<task_id>.json
    → {task_id, score}  (ground truth)

Methods 1–4 (verbalized/logprob) will be NaN; methods 5–6 (entropy) are computed.
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

from keystep_labels import as_success_text, load_keystep_map, label_step_from_keystep


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def extract_action_text(text: str) -> str | None:
    m = re.search(r'```(.+?)```', text, re.DOTALL)
    return m.group(1).strip() if m else None


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
# Data loading (step_k_samples.json format)
# ---------------------------------------------------------------------------

def _step_index(filename: str) -> int:
    m = re.search(r'step_(\d+)_', filename)
    return int(m.group(1)) if m else -1


def load_entropy_task(task_dir: str) -> list[dict]:
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
        rows.append({'step': entry['step'],
                     'm5_action_entropy': m5,
                     'm6_cumulative_action_entropy': m6})
    return rows


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_per_step_df(
    entropy_uq_dir: str,
    ground_truth: dict[str, float],
    model: str,
    benchmark: str,
    annotation_dir: str | None = None,
) -> pd.DataFrame:
    records = []
    source = f'{benchmark}/{model}'
    keystep_map = load_keystep_map(source, annotation_dir) if benchmark == 'webarena' else {}
    if not os.path.isdir(entropy_uq_dir):
        return pd.DataFrame()

    for task_id in sorted(os.listdir(entropy_uq_dir), key=lambda x: int(x) if x.isdigit() else x):
        task_dir = os.path.join(entropy_uq_dir, task_id)
        if not os.path.isdir(task_dir):
            continue
        steps = load_entropy_task(task_dir)
        score = ground_truth.get(str(task_id), float('nan'))
        final_success = True if score == 1.0 else (False if score == 0.0 else None)

        for row in score_entropy_task_steps(steps):
            records.append({
                'model': model,
                'benchmark': benchmark,
                'task_id': task_id,
                'step': row['step'],
                'success': as_success_text(
                    label_step_from_keystep(final_success, row['step'], keystep_map.get(str(task_id)))
                ),
                'm1_verbalized_conf': None,
                'm2_action_logprob_mean': None,
                'm3_output_logprob_mean': None,
                'm4_cumulative_output_logprob_mean': None,
                'm5_action_entropy': row['m5_action_entropy'],
                'm6_cumulative_action_entropy': row['m6_cumulative_action_entropy'],
            })

    return pd.DataFrame(records)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entropy_dir', required=True,
                        help='webarena/qwen30b directory (contains trajectory_1/)')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model', default='qwen30b')
    parser.add_argument('--benchmark', default='webarena')
    parser.add_argument('--annotation_dir', default=None,
                        help='Optional annotation/result dir with key-step labels.')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    entropy_uq_dir = os.path.join(args.entropy_dir, 'trajectory_1', 'uq_samples')
    actions_dir = os.path.join(args.entropy_dir, 'trajectory_1', 'actions')

    print(f'Entropy uq_samples: {entropy_uq_dir}')
    n_tasks = len(os.listdir(entropy_uq_dir)) if os.path.isdir(entropy_uq_dir) else 0
    print(f'  -> {n_tasks} task folders')

    gt = load_ground_truth(actions_dir)
    success_count = sum(1 for v in gt.values() if v == 1.0)
    failure_count = sum(1 for v in gt.values() if v == 0.0)
    print(f'Ground truth: {len(gt)} tasks | {success_count} success / {failure_count} failure')

    print('Computing per-step scores...')
    per_step_df = build_per_step_df(
        entropy_uq_dir, gt, args.model, args.benchmark, args.annotation_dir
    )
    print(f'  -> {len(per_step_df)} step rows across {per_step_df["task_id"].nunique()} tasks')

    step_out = os.path.join(args.output_dir, 'per_step_scores.csv')
    per_step_df.to_csv(step_out, index=False)
    print(f'\nDone. per_step_scores.csv -> {step_out}')

    score_cols = ['m5_action_entropy', 'm6_cumulative_action_entropy']
    print('\nStep-level score summary:')
    print(per_step_df[score_cols].describe().loc[['count', 'mean', 'std']].to_string())

    success_rate = per_step_df['success'].eq('success').mean()
    print(f'\nSuccess rate (step-level): {success_rate:.3f}')
    print('\nMean step scores by task outcome:')
    print(per_step_df.groupby('success')[score_cols].mean().to_string())


if __name__ == '__main__':
    main()
