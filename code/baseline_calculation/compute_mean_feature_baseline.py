#!/usr/bin/env python3
"""
compute_mean_feature_baseline.py — Mean-feature aggregation baseline (no ML).

Instead of training a model, this method:
  1. Z-score normalises each feature column (fit on the full dataset)
  2. Imputes NaNs with per-column median before scaling
  3. Computes the row-wise mean of normalised features as the risk score
  4. Min-max rescales the mean score to [0, 1] for metric compatibility
  5. Aggregates step-level scores to task level (last step and cross-step mean)

Three feature sets:
  macro_only  — 27 behavioral / trajectory features
  micro_only  — 18 UQ indicator features (U_sem, U_change, U_ground, U_cval, C_S, R_T)
  all         — macro + micro combined (45 features)

Metrics reported: AUROC, AUPRC, AURC, excess-AURC (e_AURC), ECE, Brier.
Evaluated at three levels: step-level, task-last (score at final step),
task-mean (mean score across all steps).

Usage:
  python compute_mean_feature_baseline.py --source mind2web/qwen30b
  python compute_mean_feature_baseline.py --source webarena/claude
  python compute_mean_feature_baseline.py --source webarena/gpt_5_2
  python compute_mean_feature_baseline.py --source webarena/qwen30b

Results are saved to:
  {source}/{uq_dir}/risk_model_results/mean_feature_baseline/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                              roc_auc_score)
from sklearn.preprocessing import StandardScaler

from keystep_labels import load_keystep_map, label_step_from_keystep

# ── Feature definitions ───────────────────────────────────────────────────────

MACRO_FEATS = [
    "step_k", "n_single_loops", "n_double_loops", "loop_length_current",
    "n_distinct_actions", "action_count_current", "max_action_count",
    "action_repeat_rate", "n_grounding_errors", "n_parse_errors",
    "consecutive_errors_now", "grounding_error_at_k", "parse_error_at_k",
    "grounding_loop_count", "n_click", "n_type", "n_scroll", "n_go_back",
    "n_distinct_elements", "action_type_entropy", "steps_since_new_action",
    "type_text_length", "reasoning_len_at_k", "reasoning_len_cumul_mean",
    "reasoning_len_cumul_max", "conf_at_k", "conf_cumul_mean",
]

MICRO_FEATS = []
for _ind in ["U_sem", "U_change", "U_ground", "U_cval", "C_S", "R_T"]:
    for _suf in ["at_k", "cumul_mean", "cumul_max"]:
        MICRO_FEATS.append(f"{_ind}_{_suf}")

FEATURE_SETS = {
    "macro_only": MACRO_FEATS,
    "micro_only": MICRO_FEATS,
    "all":        MACRO_FEATS + MICRO_FEATS,
}

# ── Metrics ───────────────────────────────────────────────────────────────────

def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    val  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        val += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return val / max(len(y_true), 1)


def aurc(y_true: np.ndarray, y_prob: np.ndarray):
    """
    Area Under the Risk-Coverage curve (selective prediction).
    Samples ordered by ascending y_prob (most confident first).
    Returns (aurc_score, aurc_baseline, excess_aurc).
    Lower aurc is better; negative e_aurc = better than random.
    """
    order    = np.argsort(y_prob)
    y_sorted = y_true[order]
    risks    = np.cumsum(y_sorted) / np.arange(1, len(y_sorted) + 1)
    score    = float(np.mean(risks))
    baseline = float(np.mean(y_true))
    return score, baseline, score - baseline


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    valid = ~np.isnan(y_prob)
    yt, yp = y_true[valid], y_prob[valid]
    if len(np.unique(yt)) < 2:
        return {}
    aurc_score, aurc_base, e_aurc = aurc(yt, yp)
    return {
        "n_samples":  int(valid.sum()),
        "auroc":      round(float(roc_auc_score(yt, yp)),           4),
        "auprc":      round(float(average_precision_score(yt, yp)), 4),
        "aurc":       round(aurc_score,                             4),
        "aurc_base":  round(aurc_base,                              4),
        "e_aurc":     round(e_aurc,                                 4),
        "aurc_gain":  round(aurc_base - aurc_score,                 4),
        "brier":      round(float(brier_score_loss(yt, yp)),        4),
        "ece":        round(float(ece(yt, yp)),                     4),
    }


# ── Mean-feature scorer ───────────────────────────────────────────────────────

def mean_feature_score(df: pd.DataFrame, feat_cols: list) -> np.ndarray:
    """
    Compute a training-free risk score for each row in `df`:
      1. Restrict to features actually present in the DataFrame.
      2. Impute NaNs with per-column median.
      3. Z-score normalise (fit on all rows).
      4. Take the row-wise mean of normalised values.
      5. Min-max rescale to [0, 1].

    The returned array has the same row order as `df`.
    """
    avail = [c for c in feat_cols if c in df.columns]
    if not avail:
        return np.full(len(df), float("nan"))

    X = df[avail].copy()
    X = X.fillna(X.median())

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scores   = X_scaled.mean(axis=1)

    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores = (scores - s_min) / (s_max - s_min)
    else:
        scores = np.zeros_like(scores)

    return scores


# ── Task-level aggregation ────────────────────────────────────────────────────

def aggregate_to_task(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """
    Reduce per-step scores to one row per task.
    Returns task_id, outcome, num_steps, <score_col>_last, <score_col>_mean.
    """
    rows = []
    for task_id, grp in df.groupby("task_id"):
        grp   = grp.sort_values("step_k")
        valid = grp[score_col].dropna()
        rows.append({
            "task_id":              task_id,
            "outcome":              int(grp["outcome"].iloc[-1]),
            "num_steps":            len(grp),
            f"{score_col}_last":    float(valid.iloc[-1]) if len(valid) > 0 else float("nan"),
            f"{score_col}_mean":    float(valid.mean())   if len(valid) > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def relabel_with_keysteps(df: pd.DataFrame, source: str, annotation_dir: str | None) -> pd.DataFrame:
    if "outcome" not in df.columns:
        return df

    key_map = load_keystep_map(source, annotation_dir)
    if not key_map:
        return df

    out = df.copy()
    step_col = "step_k" if "step_k" in out.columns else "step"

    def relabel_row(row):
        final_failure = int(row["outcome"]) == 1
        final_success = not final_failure
        success = label_step_from_keystep(final_success, int(row[step_col]), key_map.get(str(row["task_id"])))
        return 0 if success == 1 else 1

    out["outcome"] = out.apply(relabel_row, axis=1)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Mean-feature aggregation baseline — no ML model required.")
    ap.add_argument("--source",  required=True,
                    help="Relative path under results_total/, e.g. mind2web/qwen30b")
    ap.add_argument("--uq_dir",  default="uq_analysis_change_qwen",
                    help="Subdirectory containing feature_matrix.csv (default: uq_analysis_change_qwen)")
    ap.add_argument("--csv_path", default=None,
                    help="Explicit feature_matrix.csv path, overriding --source/--uq_dir "
                         "derivation. Use this for embeded_hlvm/thr_0p3 matrices -- "
                         "--source stays the plain 'benchmark/model' string either way, "
                         "since it's also used for keystep label resolution "
                         "(load_keystep_map expects exactly that format).")
    ap.add_argument("--annotation_dir", default=None,
                    help="Optional annotation/result dir with key-step labels.")
    args = ap.parse_args()

    script_dir   = Path(__file__).resolve().parent   # results_total/baseline/
    results_root = script_dir.parent                 # results_total/
    if args.csv_path is not None:
        csv_path = Path(args.csv_path)
        if not csv_path.is_absolute():
            csv_path = results_root / csv_path
    else:
        csv_path = results_root / args.source / args.uq_dir / "feature_matrix.csv"

    if not csv_path.exists():
        sys.exit(f"[ERROR] feature_matrix.csv not found at:\n  {csv_path}")

    out_dir = (results_root / args.source / args.uq_dir
               / "risk_model_results" / "mean_feature_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = relabel_with_keysteps(df, args.source, args.annotation_dir)
    print(f"\nSource        : {args.source}")
    print(f"Feature matrix: {csv_path}")
    print(f"Tasks         : {df['task_id'].nunique()}")
    print(f"Steps (total) : {len(df)}")
    print(f"Failure steps : {int(df['outcome'].sum())}  "
          f"({100 * df['outcome'].mean():.1f}%)\n")

    all_metrics    = {}
    task_score_dfs = []

    for fs_name, feat_cols in FEATURE_SETS.items():
        avail = [c for c in feat_cols if c in df.columns]
        print(f"{'='*60}")
        print(f"  Feature set : {fs_name}  ({len(avail)} of {len(feat_cols)} features present)")
        print(f"{'='*60}")

        score_col        = f"score_{fs_name}"
        df[score_col]    = mean_feature_score(df, avail)

        # ── Step-level evaluation ─────────────────────────────────────────
        step_m = evaluate(df["outcome"].values, df[score_col].values)
        print(f"  Step-level  AUROC={step_m.get('auroc', float('nan')):.4f}  "
              f"AURC={step_m.get('aurc', float('nan')):.4f}  "
              f"AURC gain={step_m.get('aurc_gain', float('nan')):+.4f}  "
              f"ECE={step_m.get('ece', float('nan')):.4f}")

        # ── Task-level aggregation & evaluation ───────────────────────────
        task_df = aggregate_to_task(df, score_col)

        tm_last = evaluate(task_df["outcome"].values,
                           task_df[f"{score_col}_last"].values)
        tm_mean = evaluate(task_df["outcome"].values,
                           task_df[f"{score_col}_mean"].values)

        print(f"  Task (last) AUROC={tm_last.get('auroc', float('nan')):.4f}  "
              f"AURC={tm_last.get('aurc', float('nan')):.4f}  "
              f"AURC gain={tm_last.get('aurc_gain', float('nan')):+.4f}  "
              f"ECE={tm_last.get('ece', float('nan')):.4f}")
        print(f"  Task (mean) AUROC={tm_mean.get('auroc', float('nan')):.4f}  "
              f"AURC={tm_mean.get('aurc', float('nan')):.4f}  "
              f"AURC gain={tm_mean.get('aurc_gain', float('nan')):+.4f}  "
              f"ECE={tm_mean.get('ece', float('nan')):.4f}\n")

        all_metrics[fs_name] = {
            "n_features":  len(avail),
            "features":    avail,
            "step_level":  step_m,
            "task_last":   tm_last,
            "task_mean":   tm_mean,
        }

        task_score_dfs.append(
            task_df[["task_id", f"{score_col}_last", f"{score_col}_mean"]]
            .rename(columns={
                f"{score_col}_last": f"{fs_name}_last",
                f"{score_col}_mean": f"{fs_name}_mean",
            })
        )

    # ── Save outputs ──────────────────────────────────────────────────────────

    # Step-level scores
    step_score_cols = ["task_id", "outcome", "step_k"] + [f"score_{fs}" for fs in FEATURE_SETS]
    step_out = df[[c for c in step_score_cols if c in df.columns]]
    step_path = out_dir / "step_scores.csv"
    step_out.to_csv(step_path, index=False)

    # Task-level scores (merge all feature sets onto a single table)
    task_base = df.groupby("task_id")["outcome"].first().reset_index()
    task_base["num_steps"] = df.groupby("task_id")["step_k"].count().values
    for tdf in task_score_dfs:
        task_base = task_base.merge(tdf, on="task_id", how="left")
    task_path = out_dir / "task_scores.csv"
    task_base.to_csv(task_path, index=False)

    # Metrics JSON
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  SUMMARY — {args.source}  (mean-feature aggregation baseline)")
    print(f"{'='*72}")
    hdr = (f"  {'Feature set':<12}  {'Agg':<6}  "
           f"{'AUROC':>7}  {'AURC':>7}  {'AURC gain':>10}  "
           f"{'ECE':>6}  {'Brier':>7}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for fs_name, res in all_metrics.items():
        for label, m in [("step",  res["step_level"]),
                         ("last",  res["task_last"]),
                         ("mean",  res["task_mean"])]:
            print(f"  {fs_name:<12}  {label:<6}  "
                  f"{m.get('auroc',     float('nan')):>7.4f}  "
                  f"{m.get('aurc',      float('nan')):>7.4f}  "
                  f"{m.get('aurc_gain', float('nan')):>+10.4f}  "
                  f"{m.get('ece',       float('nan')):>6.4f}  "
                  f"{m.get('brier',     float('nan')):>7.4f}")
        print()

    print(f"Output saved to:\n  {out_dir}")
    print(f"  {step_path.name}  ({len(step_out)} rows)")
    print(f"  {task_path.name}  ({len(task_base)} rows)")
    print(f"  {metrics_path.name}")


if __name__ == "__main__":
    main()
