"""
compute_webarena_step_baselines.py — Step-level baseline UQ for webarena agents.

Supports two layouts:

1. Combined layout (single directory):
  data_dir/trajectory_1/uq_samples/<task_id>/
    step_<k>_samples.json
    step_<k>_logprobs.json
  data_dir/trajectory_1/actions/<task_id>.json

2. Split layout (recommended for current webarena runs):
  verbalized_dir/.../uq_samples/<task_id>/
    step_<k>_samples.json      (1 sample with verbalized text)
    step_<k>_logprobs.json     (optional)
  entropy_dir/.../uq_samples/<task_id>/
    step_<k>_samples.json      (10 samples for entropy)
  actions_dir/<task_id>.json

Methods 1–4 are computed from the verbalized source.
Methods 5–6 are computed from the entropy source.
"""

import argparse
import json
import math
import os
import re
from collections import Counter

import pandas as pd

from keystep_labels import (as_success_text, load_keystep_map,
                            load_keystep_map_entropy_source, label_step_from_keystep)


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def parse_confidence(text: str) -> float | None:
    m = re.search(r'Confidence:\s*([\d.]+)', text)
    return float(m.group(1)) if m else None


def extract_action_text(text: str) -> str | None:
    m = re.search(r'```(.+?)```', text, re.DOTALL)
    return m.group(1).strip() if m else None


def action_token_logprob_mean(tokens, token_logprobs) -> float | None:
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


def full_output_logprob_mean(token_logprobs) -> float | None:
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
# Data loading
# ---------------------------------------------------------------------------

def _step_index(filename: str) -> int:
    m = re.search(r'step_(\d+)_', filename)
    return int(m.group(1)) if m else -1


def load_task(task_dir: str) -> list[dict]:
    """Load all steps for one task. Returns list of per-step dicts."""
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
        entry = {'step': k, 'tokens': [], 'token_logprobs': [],
                 'raw_predictions': [], 'first_prediction': ''}

        if k in logprob_files:
            with open(logprob_files[k]) as f:
                d = json.load(f)
            s = d.get('samples', [{}])[0]
            entry['tokens'] = s.get('tokens', [])
            entry['token_logprobs'] = s.get('token_logprobs', [])

        if k in sample_files:
            with open(sample_files[k]) as f:
                d = json.load(f)
            preds = [s.get('raw_prediction', '') for s in d.get('samples', [])]
            entry['raw_predictions'] = preds
            entry['first_prediction'] = preds[0] if preds else ''

        steps.append(entry)
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

def score_task(steps: list[dict]) -> list[dict]:
    rows = []
    m3_running, m5_running = [], []

    for entry in steps:
        tokens = entry['tokens']
        token_logprobs = entry['token_logprobs']
        text = entry['first_prediction']
        preds = entry['raw_predictions']

        has_logprobs = bool(tokens or token_logprobs)
        has_samples = bool(preds and any(preds))

        if not has_logprobs and not has_samples:
            continue

        # M1–M4 from logprobs / first sample
        m1 = parse_confidence(text) if text else None
        m2 = action_token_logprob_mean(tokens, token_logprobs) if has_logprobs else None
        m3 = full_output_logprob_mean(token_logprobs) if has_logprobs else None

        m3_running.append(m3 if m3 is not None else float('nan'))
        valid3 = [v for v in m3_running if not math.isnan(v)]
        m4 = sum(valid3) / len(valid3) if valid3 else None

        # M5–M6 from all samples
        m5 = action_entropy(preds) if has_samples else None
        m5_running.append(m5 if m5 is not None else float('nan'))
        valid5 = [v for v in m5_running if not math.isnan(v)]
        m6 = sum(valid5) / len(valid5) if valid5 else None

        rows.append({
            'step': entry['step'],
            'm1_verbalized_conf': m1,
            'm2_action_logprob_mean': m2,
            'm3_output_logprob_mean': m3,
            'm4_cumulative_output_logprob_mean': m4,
            'm5_action_entropy': m5,
            'm6_cumulative_action_entropy': m6,
        })
    return rows


# ---------------------------------------------------------------------------
# Assembly
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
    verb_rows: dict[tuple[str, int], dict] = {}
    ent_rows: dict[tuple[str, int], dict] = {}
    records = []
    source = f'{benchmark}/{model}'
    keystep_map = load_keystep_map(source, annotation_dir) if benchmark == 'webarena' else {}
    # Separate label for M5/M6 (2026-07-30, bug): their score
    # comes from entropy_uq_dir (the plain rollout), not verbalized_uq_dir,
    # so they need the keyStep annotated against THAT rollout -- see
    # load_keystep_map_entropy_source()'s docstring.
    keystep_map_entropy = load_keystep_map_entropy_source(source, annotation_dir) if benchmark == 'webarena' else {}

    # Ground truth (task pass/fail) is ALSO rollout-specific for webarena --
    # each independent rollout has its own actions/<task_id>.json with its
    # own score, and the two rollouts disagree on 10-25% of tasks (confirmed
    # 2026-07-31, all 5 models). `ground_truth` must be the verbalized
    # rollout's own outcome (for M1-4/M7/M11-14); `ground_truth_entropy` must
    # be the entropy rollout's own outcome (for M5/M6). Falls back to
    # `ground_truth` only if a given task has no entry in the entropy dict
    # (e.g. the entropy rollout genuinely has no separate actions/ dir).
    ground_truth_entropy = ground_truth_entropy or {}

    if os.path.isdir(verbalized_uq_dir):
        for task_id in sorted(os.listdir(verbalized_uq_dir), key=lambda x: int(x) if x.isdigit() else x):
            task_dir = os.path.join(verbalized_uq_dir, task_id)
            if not os.path.isdir(task_dir):
                continue
            for row in score_task(load_task(task_dir)):
                verb_rows[(task_id, row["step"])] = row

    if os.path.isdir(entropy_uq_dir):
        for task_id in sorted(os.listdir(entropy_uq_dir), key=lambda x: int(x) if x.isdigit() else x):
            task_dir = os.path.join(entropy_uq_dir, task_id)
            if not os.path.isdir(task_dir):
                continue
            for row in score_task(load_task(task_dir)):
                ent_rows[(task_id, row["step"])] = row

    # Restrict to the VERBALIZED rollout's own keys only (2026-07-30,
    # bug: was a union of verb_rows | ent_rows). Same reasoning
    # as compute_baselines.py's build_per_step_df -- verbalized and entropy
    # are two independent rollouts that can have different lengths; the
    # keyStep label was annotated against the verbalized trajectory
    # specifically, so entropy-only steps beyond its length must not be
    # labeled with that same cutoff.
    all_keys = sorted(set(verb_rows), key=lambda x: (int(x[0]) if str(x[0]).isdigit() else x[0], x[1]))
    for task_id, step in all_keys:
        score = ground_truth.get(str(task_id), float('nan'))
        final_success = True if score == 1.0 else (False if score == 0.0 else None)
        key_step = keystep_map.get(str(task_id))
        v = verb_rows.get((task_id, step), {})
        e = ent_rows.get((task_id, step), {})
        success = as_success_text(label_step_from_keystep(final_success, step, key_step))

        score_ent = ground_truth_entropy.get(str(task_id))
        if score_ent is None:
            score_ent = score  # entropy rollout has no own ground truth for this task; fall back
        final_success_entropy = True if score_ent == 1.0 else (False if score_ent == 0.0 else None)
        key_step_entropy = keystep_map_entropy.get(str(task_id))
        success_entropy = as_success_text(label_step_from_keystep(final_success_entropy, step, key_step_entropy))
        records.append({
            'model': model, 'benchmark': benchmark,
            'task_id': task_id, 'step': step, 'success': success,
            'success_entropy': success_entropy,
            'm1_verbalized_conf': v.get('m1_verbalized_conf'),
            'm2_action_logprob_mean': v.get('m2_action_logprob_mean'),
            'm3_output_logprob_mean': v.get('m3_output_logprob_mean'),
            'm4_cumulative_output_logprob_mean': v.get('m4_cumulative_output_logprob_mean'),
            'm5_action_entropy': e.get('m5_action_entropy'),
            'm6_cumulative_action_entropy': e.get('m6_cumulative_action_entropy'),
        })
    return pd.DataFrame(records)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None,
                        help='Combined agent dir containing trajectory_1/')
    parser.add_argument('--verbalized_dir', default=None,
                        help='Optional separate verbalized dir containing trajectory_1/ or uq_samples/')
    parser.add_argument('--entropy_dir', default=None,
                        help='Optional separate entropy dir containing trajectory_1/ or uq_samples/')
    parser.add_argument('--actions_dir', default=None,
                        help='Optional explicit actions dir')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--benchmark', default='webarena')
    parser.add_argument('--annotation_dir', default=None,
                        help='Optional annotation/result dir with key-step labels.')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    def resolve_uq_dir(base: str | None) -> str | None:
        if not base:
            return None
        if os.path.isdir(os.path.join(base, 'trajectory_1', 'uq_samples')):
            return os.path.join(base, 'trajectory_1', 'uq_samples')
        if os.path.isdir(os.path.join(base, 'uq_samples')):
            return os.path.join(base, 'uq_samples')
        return base if os.path.isdir(base) else None

    def resolve_actions_dir(base: str | None) -> str:
        if not base:
            return ""
        if os.path.isdir(os.path.join(base, 'trajectory_1', 'actions')):
            return os.path.join(base, 'trajectory_1', 'actions')
        if os.path.isdir(os.path.join(base, 'actions')):
            return os.path.join(base, 'actions')
        return ""

    if args.data_dir:
        verbalized_uq_dir = entropy_uq_dir = resolve_uq_dir(args.data_dir)
        actions_dir_verb = actions_dir_ent = (
            args.actions_dir or os.path.join(args.data_dir, 'trajectory_1', 'actions'))
    else:
        verbalized_uq_dir = resolve_uq_dir(args.verbalized_dir)
        entropy_uq_dir = resolve_uq_dir(args.entropy_dir)
        # 2026-07-31 (flagged, MANIFEST.csv caution): ground truth is
        # rollout-specific for webarena -- verbalized_dir and entropy_dir are
        # two independent rollouts, each with its own actions/<id>.json
        # score, and they disagree on 10-25% of tasks across all 5 models.
        # `success` (M1-4/M7/M11-14, verbalized-sourced) must use the
        # verbalized rollout's OWN outcome; `success_entropy` (M5/M6,
        # entropy-sourced) must use the entropy rollout's OWN outcome. An
        # explicit --actions_dir override applies to both (rare/manual use).
        if args.actions_dir:
            actions_dir_verb = actions_dir_ent = args.actions_dir
        else:
            actions_dir_verb = resolve_actions_dir(args.verbalized_dir)
            actions_dir_ent = resolve_actions_dir(args.entropy_dir)

    nv = len(os.listdir(verbalized_uq_dir)) if verbalized_uq_dir and os.path.isdir(verbalized_uq_dir) else 0
    ne = len(os.listdir(entropy_uq_dir)) if entropy_uq_dir and os.path.isdir(entropy_uq_dir) else 0
    print(f'verbalized uq_samples: {nv} tasks')
    print(f'entropy uq_samples:    {ne} tasks')

    gt = load_ground_truth(actions_dir_verb)
    success_count = sum(1 for v in gt.values() if v == 1.0)
    failure_count = sum(1 for v in gt.values() if v == 0.0)
    print(f'Ground truth (verbalized rollout, for M1-4/M7/M11-14): {len(gt)} tasks | '
          f'{success_count} success / {failure_count} failure')

    gt_ent = load_ground_truth(actions_dir_ent)
    if actions_dir_ent and actions_dir_ent != actions_dir_verb:
        se = sum(1 for v in gt_ent.values() if v == 1.0)
        fe = sum(1 for v in gt_ent.values() if v == 0.0)
        disagree = sum(1 for t in (set(gt) & set(gt_ent)) if gt[t] != gt_ent[t])
        print(f'Ground truth (entropy rollout, for M5/M6):             {len(gt_ent)} tasks | '
              f'{se} success / {fe} failure  ({disagree} disagree with verbalized rollout)')

    print('Computing per-step scores...')
    per_step_df = build_per_step_df(
        verbalized_uq_dir or "", entropy_uq_dir or "", gt, args.model, args.benchmark, args.annotation_dir,
        ground_truth_entropy=gt_ent,
    )
    print(f'  -> {len(per_step_df)} step rows across {per_step_df["task_id"].nunique()} tasks')

    out = os.path.join(args.output_dir, 'per_step_scores.csv')
    per_step_df.to_csv(out, index=False)
    print(f'Done. per_step_scores.csv -> {out}')

    score_cols = [c for c in per_step_df.columns if c.startswith('m') and c[1].isdigit()]
    print('\nStep-level score summary:')
    print(per_step_df[score_cols].describe().loc[['count', 'mean', 'std']].to_string())
    success_rate = per_step_df['success'].eq('success').mean()
    print(f'\nSuccess rate (step-level): {success_rate:.3f}')


if __name__ == '__main__':
    main()
