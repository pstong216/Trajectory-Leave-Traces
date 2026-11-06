"""
compute_baselines.py — Generic baseline UQ scorer for web agent trajectories.

Six methods:
  1. Verbalized confidence at step k             (verbalized_dir)
  2. Action token mean log-prob at step k        (verbalized_dir)
  3. Full output mean log-prob at step k         (verbalized_dir)
  4. Cumulative mean output log-prob (0→k)       (verbalized_dir)
  5. Action entropy at step k                    (entropy_dir)
  6. Cumulative mean action entropy (0→k)        (entropy_dir)

Usage:
  python compute_baselines.py \
    --verbalized_dir /path/to/verbalized_results \
    --entropy_dir    /path/to/multisample_results \
    --output_dir     /path/to/output \
    [--ground_truth  /path/to/result_summary.json] \
    [--model         qwen30b] \
    [--benchmark     mind2web]
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
# Token-level helpers (methods 1–4, verbalized_dir)
# ---------------------------------------------------------------------------

def parse_confidence(text: str) -> float | None:
    """Extract the verbalized confidence score, e.g. 'Confidence: 0.85'."""
    m = re.search(r'Confidence:\s*([\d.]+)', text)
    return float(m.group(1)) if m else None


def extract_action_text(text: str) -> str | None:
    """Return the action string inside the first ```...``` block."""
    m = re.search(r'```(.+?)```', text, re.DOTALL)
    return m.group(1).strip() if m else None


def action_token_logprob_mean(tokens: list[str], token_logprobs: list[float]) -> float | None:
    """
    Mean log-prob of tokens that fall inside the ```...``` action span.
    Reconstructs the full text, locates the action span by character offset,
    then maps back to token indices.
    """
    text = ''.join(tokens)
    m = re.search(r'```(.+?)```', text, re.DOTALL)
    if not m:
        return None

    action_start = m.start(1)
    action_end = m.end(1)

    char_pos = 0
    action_lps = []
    for tok, lp in zip(tokens, token_logprobs):
        tok_end = char_pos + len(tok)
        if tok_end > action_start and char_pos < action_end:
            action_lps.append(lp)
        char_pos = tok_end

    return -sum(action_lps) / len(action_lps) if action_lps else None


def full_output_logprob_mean(token_logprobs: list[float]) -> float | None:
    """Mean negative log-prob for a step output (-log p, positive value)."""
    if not token_logprobs:
        return None
    return -sum(token_logprobs) / len(token_logprobs)


# ---------------------------------------------------------------------------
# Entropy helper (method 5–6, entropy_dir)
# ---------------------------------------------------------------------------

def action_entropy(samples: list[str]) -> float | None:
    """
    Shannon entropy (bits) over the distribution of actions in `samples`.
    Each sample is the full step output; the action is the ```...``` content.
    Returns None if no valid actions could be extracted.
    """
    actions = []
    for s in samples:
        a = extract_action_text(s)
        if a is not None:
            # Normalise whitespace so minor formatting differences don't split bins
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
# Per-task scoring
# ---------------------------------------------------------------------------

def score_verbalized_task(result: dict) -> list[dict]:
    """
    Compute methods 1–4 for every valid step of one task.
    Returns a list of per-step dicts.
    """
    step_logprobs = result.get('step_logprobs', [])
    step_samples = result.get('step_samples', [])

    rows = []
    m3_running = []  # accumulates full-output mean logprobs for method 4

    n_steps = max(len(step_logprobs), len(step_samples))
    for k in range(n_steps):
        lp_entry = step_logprobs[k] if k < len(step_logprobs) else None
        samples_k = step_samples[k] if k < len(step_samples) else []

        sample_text = samples_k[0] if samples_k else ''
        if not sample_text:
            continue

        m1 = parse_confidence(sample_text)
        if lp_entry is None:
            m2 = None
            m3 = None
        else:
            tokens = lp_entry.get('tokens', [])
            token_logprobs = lp_entry.get('token_logprobs', [])
            m2 = action_token_logprob_mean(tokens, token_logprobs)
            m3 = full_output_logprob_mean(token_logprobs)

        m3_running.append(m3 if m3 is not None else float('nan'))
        valid = [v for v in m3_running if not math.isnan(v)]
        m4 = sum(valid) / len(valid) if valid else None

        rows.append({
            'step': k,
            'm1_verbalized_conf': m1,
            'm2_action_logprob_mean': m2,
            'm3_output_logprob_mean': m3,
            'm4_cumulative_output_logprob_mean': m4,
        })

    return rows


def score_entropy_task(result: dict) -> list[dict]:
    """
    Compute methods 5–6 for every valid step of one task.
    Returns a list of per-step dicts.
    """
    step_samples = result.get('step_samples', [])

    rows = []
    m5_running = []

    for k, samples_k in enumerate(step_samples):
        # Skip empty trailing entry
        if not samples_k or (len(samples_k) == 1 and not samples_k[0]):
            continue

        m5 = action_entropy(samples_k)

        m5_running.append(m5 if m5 is not None else float('nan'))
        valid = [v for v in m5_running if not math.isnan(v)]
        m6 = sum(valid) / len(valid) if valid else None

        rows.append({
            'step': k,
            'm5_action_entropy': m5,
            'm6_cumulative_action_entropy': m6,
        })

    return rows


# ---------------------------------------------------------------------------
# Directory loading
# ---------------------------------------------------------------------------

def load_results(directory: str) -> dict[str, dict]:
    """
    Load all result.json files from `directory`.
    Returns a dict mapping task_id (hash string) → result dict.
    """
    results = {}
    if not os.path.isdir(directory):
        return results
    for folder in os.listdir(directory):
        rpath = os.path.join(directory, folder, 'result.json')
        if not os.path.isfile(rpath):
            continue
        try:
            with open(rpath) as f:
                r = json.load(f)
            task_id = r.get('task_id') or r.get('task_index')
            if task_id is not None:
                results[str(task_id)] = r
        except (json.JSONDecodeError, KeyError):
            print(f'Warning: could not parse {rpath}')
    return results


def load_ground_truth(gt_path: str) -> dict[str, str]:
    """
    Load ground-truth success labels. Supports two formats (auto-detected):

    Format A (standard, e.g. mind2web/claude_result.json,
    mind2web/results_claude_verbalized/eval.json):
      {"tasks": [{"task_id": "0", "status": "success|failure"}, ...]}

    Format B (2026-07-31, e.g. mind2web/results_gptnano_verbalized/eval.json
    -- gptnano's verbalized rollout never got a Format-A eval.json; this is
    the raw per-task judge dump instead): JSON Lines, one task object per
    line, each with {"task_id": ..., "predicted_label": 0|1} where
    1=success, 0=failure (confirmed against evaluation_details.response's
    "Status:" field).

    task_id in ground-truth is the folder/task_index number (string).
    Returns dict: task_index_str → "success"|"failure"
    """
    if not gt_path or not os.path.isfile(gt_path):
        return {}
    with open(gt_path) as f:
        head = f.read(1)
        f.seek(0)
        first_line = f.readline()
        f.seek(0)
        # Format A is a single JSON object with a top-level "tasks" array.
        # Format B is JSON-Lines -- the first line alone parses as a
        # complete, self-contained JSON object with "predicted_label".
        try:
            first_obj = json.loads(first_line)
        except (json.JSONDecodeError, ValueError):
            first_obj = None
        if isinstance(first_obj, dict) and 'predicted_label' in first_obj:
            gt: dict[str, str] = {}
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                status = 'success' if d.get('predicted_label') == 1 else 'failure'
                gt[str(d['task_id'])] = status
            return gt
        d = json.load(f)
    return {str(t['task_id']): t['status'] for t in d.get('tasks', [])}


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def build_per_step_df(
    verbalized: dict[str, dict],
    entropy: dict[str, dict],
    model: str,
    benchmark: str,
    ground_truth: dict[str, str] | None = None,
    annotation_dir: str | None = None,
    ground_truth_verbalized: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Join verbalized and entropy per-step scores into one DataFrame."""
    source = f'{benchmark}/{model}'
    keystep_map = load_keystep_map(source, annotation_dir)
    # Separate label for M5/M6 (2026-07-30, bug): their score
    # comes from entropy_dir (the plain rollout), not verbalized_dir, so they
    # need the keyStep annotated against THAT rollout -- not keystep_map
    # above, which matches the verbalized rollout. See
    # load_keystep_map_entropy_source()'s docstring.
    keystep_map_entropy = load_keystep_map_entropy_source(source, annotation_dir)

    # Build verbalized per-step scores keyed by (task_id, step)
    verb_rows: dict[tuple, dict] = {}
    for task_id, result in verbalized.items():
        task_index = str(result.get('task_index', task_id))
        for row in score_verbalized_task(result):
            key = (task_id, task_index, row['step'])
            verb_rows[key] = row

    # Build entropy per-step scores
    ent_rows: dict[tuple, dict] = {}
    for task_id, result in entropy.items():
        task_index = str(result.get('task_index', task_id))
        for row in score_entropy_task(result):
            key = (task_id, task_index, row['step'])
            ent_rows[key] = row

    # Restrict to the VERBALIZED rollout's own (task_id, task_index, step) keys
    # only (2026-07-30, bug: was a union of verb_rows | ent_rows).
    # verbalized and entropy are two INDEPENDENT samples of the same task --
    # they frequently have different lengths (confirmed: 34-79% of mind2web
    # tasks across all 5 models have mismatched verbalized/plain step counts).
    # The keyStep label was annotated by a human/gemini annotator watching the
    # VERBALIZED trajectory specifically (that's what the "_v" file's task-id
    # match confirms). Unioning in entropy-only steps beyond the verbalized
    # rollout's length applied that same keyStep cutoff to steps from a
    # DIFFERENT, independently-sampled run the annotator never saw -- i.e.
    # baseline's own trajectory data was silently getting mixed with steps
    # that don't belong to the labeled trajectory at all. Entropy-only tail
    # steps are now dropped entirely rather than mislabeled.
    all_keys = set(verb_rows)

    # Ground truth is ALSO rollout-specific for mind2web (2026-07-31,
    # flagged, confirmed via 0-orphan + exact action_history/
    # final_result_response match): `ground_truth` (e.g. {model}_result.json)
    # is the PLAIN rollout's own judged outcome -- correct for M5/M6
    # (success_entropy) but WRONG for M1-4/M7/M11-14 (success), which read
    # from verbalized_dir. `ground_truth_verbalized` (e.g.
    # results_{model}_verbalized/eval.json) is the correct source for
    # `success`. Falls back to `ground_truth` when not provided (e.g. no
    # verbalized-specific ground truth file exists for this model yet).
    gt_success = ground_truth_verbalized if ground_truth_verbalized else ground_truth

    records = []
    for key in sorted(all_keys):
        task_id, task_index, step = key
        v = verb_rows.get(key, {})
        e = ent_rows.get(key, {})
        success = None
        if ground_truth or gt_success:
            status = gt_success.get(str(task_id), gt_success.get(str(task_index))) if gt_success else None
            final_success = True if status == 'success' else (False if status == 'failure' else None)
            # mind2web result.json's internal "task_id" is a content hash; the
            # keystep annotation files are keyed by the numeric id, which is
            # "task_index" here. task_id can never match -- use task_index only
            # (the old task_id-first fallback was dead code that masked this).
            key_step = keystep_map.get(str(task_index))
            success = as_success_text(label_step_from_keystep(final_success, step, key_step))

            status_ent = ground_truth.get(str(task_id), ground_truth.get(str(task_index))) if ground_truth else None
            final_success_ent = True if status_ent == 'success' else (False if status_ent == 'failure' else None)
            key_step_entropy = keystep_map_entropy.get(str(task_index))
            success_entropy = as_success_text(label_step_from_keystep(final_success_ent, step, key_step_entropy))
        else:
            success_entropy = None
        records.append({
            'model': model,
            'benchmark': benchmark,
            'task_id': task_id,
            'task_index': task_index,
            'step': step,
            'success': success,
            'success_entropy': success_entropy,
            'm1_verbalized_conf': v.get('m1_verbalized_conf'),
            'm2_action_logprob_mean': v.get('m2_action_logprob_mean'),
            'm3_output_logprob_mean': v.get('m3_output_logprob_mean'),
            'm4_cumulative_output_logprob_mean': v.get('m4_cumulative_output_logprob_mean'),
            'm5_action_entropy': e.get('m5_action_entropy'),
            'm6_cumulative_action_entropy': e.get('m6_cumulative_action_entropy'),
        })

    return pd.DataFrame(records)


def build_task_df(
    per_step_df: pd.DataFrame,
    ground_truth: dict[str, str],
    model: str,
    benchmark: str,
) -> pd.DataFrame:
    """
    Aggregate per-step scores to task level.
    For each task: last-step value and mean across all steps.
    """
    score_cols = [
        'm1_verbalized_conf',
        'm2_action_logprob_mean',
        'm3_output_logprob_mean',
        'm4_cumulative_output_logprob_mean',
        'm5_action_entropy',
        'm6_cumulative_action_entropy',
    ]

    records = []
    for (task_id, task_index), grp in per_step_df.groupby(['task_id', 'task_index']):
        grp = grp.sort_values('step')
        row = {
            'model': model,
            'benchmark': benchmark,
            'task_id': task_id,
            'task_index': task_index,
            'num_steps': len(grp),
        }
        if ground_truth:
            row['success'] = grp['success'].iloc[-1] if 'success' in grp.columns else ground_truth.get(str(task_index), None)

        for col in score_cols:
            valid = grp[col].dropna()
            # _last: value at the last step where this method has data
            row[f'{col}_last'] = valid.iloc[-1] if len(valid) > 0 else float('nan')
            row[f'{col}_mean'] = valid.mean() if len(valid) > 0 else float('nan')

        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Compute baseline UQ scores for web agent trajectories.')
    parser.add_argument('--verbalized_dir', required=True,
                        help='Directory containing verbalized results (one result.json per task subfolder).')
    parser.add_argument('--entropy_dir', required=True,
                        help='Directory containing multi-sample results for entropy computation.')
    parser.add_argument('--output_dir', required=True,
                        help='Directory where per_step_scores.csv and task_scores.csv will be written.')
    parser.add_argument('--ground_truth', default=None,
                        help="Optional JSON file with task success/failure labels "
                             "(the PLAIN rollout's own outcome -- used for success_entropy/M5-M6).")
    parser.add_argument('--ground_truth_verbalized', default=None,
                        help="Optional JSON file with the VERBALIZED rollout's own outcome "
                             "-- used for success/M1-4,M7,M11-14. Falls back to --ground_truth "
                             "when not given (e.g. no verbalized-specific ground truth exists yet).")
    parser.add_argument('--model', default='unknown',
                        help='Model name tag written into output (default: unknown).')
    parser.add_argument('--benchmark', default='unknown',
                        help='Benchmark name tag written into output (default: unknown).')
    parser.add_argument('--annotation_dir', default=None,
                        help='Optional annotation/result dir with key-step labels.')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Loading verbalized results from: {args.verbalized_dir}')
    verbalized = load_results(args.verbalized_dir)
    print(f'  -> {len(verbalized)} tasks loaded')

    print(f'Loading multi-sample results from: {args.entropy_dir}')
    entropy = load_results(args.entropy_dir)
    print(f'  -> {len(entropy)} tasks loaded')

    gt = load_ground_truth(args.ground_truth)
    if gt:
        print(f'Ground truth (plain rollout, for success_entropy/M5-6): {len(gt)} task labels')
    else:
        print('No ground truth provided — success column omitted')

    gt_verb = load_ground_truth(args.ground_truth_verbalized) if args.ground_truth_verbalized else {}
    if gt_verb:
        disagree = sum(1 for t in (set(gt) & set(gt_verb)) if gt[t] != gt_verb[t])
        print(f'Ground truth (verbalized rollout, for success/M1-4,M7,M11-14): {len(gt_verb)} task labels '
              f'({disagree} disagree with plain rollout)')
    elif args.ground_truth_verbalized:
        print(f'  [WARN] --ground_truth_verbalized given but empty/unreadable: {args.ground_truth_verbalized}')

    print('Computing per-step scores...')
    per_step_df = build_per_step_df(
        verbalized, entropy, args.model, args.benchmark, gt, args.annotation_dir,
        ground_truth_verbalized=gt_verb,
    )

    print('Aggregating to task level...')
    task_df = build_task_df(per_step_df, gt, args.model, args.benchmark)

    step_out = os.path.join(args.output_dir, 'per_step_scores.csv')
    task_out = os.path.join(args.output_dir, 'task_scores.csv')
    per_step_df.to_csv(step_out, index=False)
    task_df.to_csv(task_out, index=False)

    print(f'\nDone.')
    print(f'  per_step_scores.csv : {len(per_step_df)} rows  -> {step_out}')
    print(f'  task_scores.csv     : {len(task_df)} rows  -> {task_out}')

    # Quick summary stats
    print('\nTask-level score summary (means across tasks):')
    last_cols = [c for c in task_df.columns if c.endswith('_last')]
    print(task_df[last_cols].describe().loc[['mean', 'std', 'min', 'max']].to_string())


if __name__ == '__main__':
    main()
