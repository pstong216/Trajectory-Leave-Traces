#!/usr/bin/env python3
"""
stacked_logistic_baseline.py — Supervised stacked baselines (M13, M14).

Composes a feature vector per (task, step) from the existing baseline scalar
scores M1-M4 and M7 in each model's per_step_scores.csv, then trains two
logistic regressions. m5/m6 (action entropy) are EXCLUDED from this vector
(2026-07-30): m5/m6 are correctly sourced from entropy_dir
(the multi-sample rollout -- required to compute entropy at all), but
M1-M4/M7 come from the separate, independent verbalized rollout. Joining
them into one row's feature vector would stitch together two independent
samples of the same task as if they were one observation. m5/m6 remain valid
standalone baseline methods on their own; they just don't feed this stacked
vector.

M13 — HTC_L2 protocol (same as htc_step_baseline.py / M11):
  - task-level StratifiedKFold(5, shuffle, seed=42) on task final success
  - per fold: median imputation (train medians) + StandardScaler
  - C searched via a single 80/20 group-respecting holdout over
    {0.01, 0.1, 1.0, 10.0} (2026-07-30: previously fixed C=1.0 -- unfair vs
    M12/M14's tuned L1/L2; now uses the same best_C_for_fold() protocol as
    M11's L2 search, matching the original HTC paper's stated 80/20-holdout
    hyperparameter protocol, arxiv 2601.15778 Appendix A.2.6, rather than a
    k-fold inner CV)
  - LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000)
  - no sample weights

M14 — Logistic_L2 protocol (same as
codes/train_risk_model/train_risk_model_temp_sca.py, the classifier behind
the paper main table's Macro/Micro/M&M rows):
  - task-level StratifiedKFold(5) with NO shuffle, tasks in order of first
    appearance, stratified on task final outcome
  - per fold: median imputation (train medians, all-NaN cols -> 0)
  - sample_weight = "task_length": each step weighted 1/n_steps of its task
  - StandardScaler + GridSearchCV over C in {0.01, 0.1, 1.0, 10.0}, inner
    StratifiedGroupKFold(3, shuffle, seed=0) (2026-07-30: was plain cv=3,
    same fix as train_risk_model_temp_sca.py's _cv3()), scoring="roc_auc",
    LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
  - no post-hoc calibration (Logistic_L2 is in NO_POSTHOC_CALIB_MODELS)

M8-M10 (mean-feature scores from the HLVM feature matrix) are deliberately
excluded — the vector contains only external baseline signals.

Labels are the key-step-relabeled step-level success (1=success, 0=failure)
already present in per_step_scores.csv. Predictions are out-of-fold
P(success) — probability-native, no extra calibration or min-max.

Metrics per model: AUROC, raw AURC, AURC_ref (random), AURC_oracle,
E-AURC proper (= AURC - AURC_oracle, as in recover_aurc.py), aurc_gain,
Brier, ECE.

Usage:
  python stacked_logistic_baseline.py

Outputs:
  baseline/<benchmark>/<model>/results/stacked_step_scores.csv
  baseline/stacked_logistic_metrics.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

BASELINE_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(BASELINE_ROOT.parent / "baseline_extension"))
from compute_m2_cumulative import add_m2_cumulative  # noqa: E402 -- M15 feature

BASE_FEATURE_COLS = [
    "m1_verbalized_conf",
    "m2_action_logprob_mean",
    "m3_output_logprob_mean",
    "m4_cumulative_output_logprob_mean",
    # m5_action_entropy / m6_cumulative_action_entropy REMOVED 2026-07-30
    # (design decision): they're correctly sourced from entropy_dir (the
    # multi-sample rollout -- needed to compute entropy at all, that's not a
    # bug on their own). But M13/M14 build ONE feature vector per (task,step)
    # by joining M1-M7 together, and M1-M4/M7 come from the VERBALIZED
    # rollout -- a different, independent sample. Including m5/m6 stitched
    # two independent rollouts' data into a single row's feature vector.
    # Keep m5/m6 as their own standalone baseline methods; just don't feed
    # them into the stacked models.
    "m7_conf_cumul_mean",
    # m15_cumulative_action_logprob_mean ADDED 2026-08-02 (design decision):
    # cumulative (0->k) version of m2, same rollout (verbalized) as the rest
    # of this vector -- no cross-rollout stitching concern like m5/m6 above.
    # NOTE: also feeds M14 (run_stacked_cv_logl2 uses the same BASE_FEATURE_COLS)
    # -- M14 isn't reported anywhere downstream in this release, so this is
    # inert for it, just flagged so it isn't a surprise later.
    "m15_cumulative_action_logprob_mean",
]

N_BINS = 10


# ── Metrics ────────────────────────────────────────────────────────────────────

def ece_equal_width(probs: np.ndarray, labels: np.ndarray) -> float:
    if len(probs) == 0:
        return float("nan")
    probs = np.clip(probs, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    val = 0.0
    for i in range(N_BINS):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs <= hi if i == N_BINS - 1 else probs < hi)
        if mask.sum() == 0:
            continue
        val += mask.sum() / len(probs) * abs(probs[mask].mean() - labels[mask].mean())
    return float(val)


def aurc(conf: np.ndarray, success: np.ndarray) -> tuple[float, float]:
    """Raw AURC and random reference. conf: higher = more confident;
    success: 1=success, 0=failure. Same convention as update_baselines_with_features."""
    n = len(conf)
    if n == 0:
        return float("nan"), float("nan")
    order = np.argsort(-conf)
    errors = (1 - success[order]).astype(float)
    risks = np.cumsum(errors) / np.arange(1, n + 1)
    return float(np.mean(risks)), float(np.mean(errors))


def aurc_oracle(success: np.ndarray) -> float:
    """AURC of a perfect oracle (all successes ranked before failures).
    Same formula as codes/train_risk_model/recover_aurc.py."""
    n = len(success)
    n_f = int((1 - success).sum())
    n_s = n - n_f
    if n_f == 0:
        return 0.0
    if n_s == 0:
        return 1.0
    i_vals = np.arange(n_s + 1, n + 1, dtype=float)
    return float(np.sum((i_vals - n_s) / i_vals) / n)


def evaluate(conf: np.ndarray, success: np.ndarray) -> dict:
    valid = ~np.isnan(conf)
    c, y = conf[valid], success[valid].astype(int)
    if len(c) == 0 or len(np.unique(y)) < 2:
        return {}
    raw_aurc, aurc_ref = aurc(c, y)
    oracle = aurc_oracle(y)
    return {
        "n": int(len(c)),
        "auroc": round(float(roc_auc_score(y, c)), 4),
        "aurc": round(raw_aurc, 4),
        "aurc_ref": round(aurc_ref, 4),
        "aurc_oracle": round(oracle, 4),
        "e_aurc": round(raw_aurc - oracle, 4),
        "aurc_gain": round(aurc_ref - raw_aurc, 4),
        "brier": round(float(brier_score_loss(y, np.clip(c, 1e-7, 1 - 1e-7))), 4),
        "ece": round(ece_equal_width(c, y), 4),
    }


C_GRID = [0.01, 0.1, 1.0, 10.0]


def _single_holdout_split(X, y, groups, seed=0):
    """One group-respecting 80%/20% train/validation split (2026-07-30: matches
    the original HTC paper's stated protocol -- arxiv 2601.15778, Appendix
    A.2.6: single holdout, NOT a further k-fold inner CV. Mirrors
    htc_step_baseline.py's _single_holdout_split() exactly."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return next(iter(splitter.split(X, y, groups=groups)))


def best_C_for_fold(X_tr: np.ndarray, y_tr: np.ndarray, groups: np.ndarray) -> float:
    """L2 hyperparameter search for M13 (2026-07-30): previously
    fixed at C=1.0 while M12/M11's L1 counterpart searched alpha -- unfair
    comparison since only the L1 side got tuned. Single 80/20 group-respecting
    holdout (not 3-fold), matching the original HTC paper's protocol and
    htc_step_baseline.py's best_C_for_fold() exactly, so M11 and M13 stay on
    the same protocol per this file's own docstring claim."""
    itr, iva = _single_holdout_split(X_tr, y_tr, groups, seed=0)
    best_score, best_C = -np.inf, 1.0
    if len(np.unique(y_tr[itr])) < 2 or len(np.unique(y_tr[iva])) < 2:
        return best_C
    for C in C_GRID:
        m = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=1000, random_state=42)
        m.fit(X_tr[itr], y_tr[itr])
        p = m.predict_proba(X_tr[iva])[:, 1]
        score = (roc_auc_score(y_tr[iva], p) - brier_score_loss(y_tr[iva], p)
                  - ece_equal_width(p, y_tr[iva]))
        if score > best_score:
            best_score, best_C = score, C
    return best_C


# ── Stacked CV training ────────────────────────────────────────────────────────

def run_stacked_cv_htc(df: pd.DataFrame, feat_cols: list[str], group_col: str) -> np.ndarray:
    """M13: out-of-fold P(success), HTC_L2 protocol (mirrors compute_htc_step_scores)."""
    probs = np.full(len(df), np.nan, dtype=float)

    task_meta = (
        df.sort_values("step")
        .groupby(group_col)["success"]
        .last()
        .astype(int)
        .reset_index()
        .rename(columns={"success": "task_final_success"})
    )
    if task_meta["task_final_success"].nunique() < 2:
        return probs

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, va_idx in cv.split(
        task_meta[group_col].values,
        task_meta["task_final_success"].values,
    ):
        train_groups = set(task_meta.iloc[tr_idx][group_col])
        val_groups = set(task_meta.iloc[va_idx][group_col])
        train_mask = df[group_col].isin(train_groups).values
        val_mask = df[group_col].isin(val_groups).values
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        y_tr = df.loc[train_mask, "success"].values.astype(int)
        if len(np.unique(y_tr)) < 2:
            continue

        X_tr = df.loc[train_mask, feat_cols].astype(float)
        X_va = df.loc[val_mask, feat_cols].astype(float)
        medians = X_tr.median()
        X_tr = X_tr.fillna(medians).fillna(0.0).values
        X_va = X_va.fillna(medians).fillna(0.0).values

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)

        train_task_ids = df.loc[train_mask, group_col].values
        best_C = best_C_for_fold(X_tr, y_tr, train_task_ids)
        clf = LogisticRegression(penalty="l2", C=best_C, solver="lbfgs",
                                 max_iter=1000, random_state=42)
        clf.fit(X_tr, y_tr)
        probs[np.where(val_mask)[0]] = clf.predict_proba(X_va)[:, 1]

    return probs


def run_stacked_cv_logl2(df: pd.DataFrame, feat_cols: list[str], group_col: str) -> np.ndarray:
    """M14: out-of-fold P(success), Logistic_L2 protocol from
    train_risk_model_temp_sca.py (run_cv + fit_model)."""
    probs = np.full(len(df), np.nan, dtype=float)

    # Task order = order of first appearance, like df["task_id"].unique()
    task_ids = df[group_col].unique()
    task_outcome = (
        df.sort_values("step").groupby(group_col)["success"].last().astype(int)
        .reindex(task_ids).values
    )
    if len(np.unique(task_outcome)) < 2:
        return probs

    skf = StratifiedKFold(n_splits=5)  # no shuffle, deterministic
    for tr_idx, va_idx in skf.split(task_ids, task_outcome):
        train_groups = set(task_ids[tr_idx])
        val_groups = set(task_ids[va_idx])
        train_mask = df[group_col].isin(train_groups).values
        val_mask = df[group_col].isin(val_groups).values
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        train_df = df.loc[train_mask]
        y_tr = train_df["success"].values.astype(int)
        if len(np.unique(y_tr)) < 2:
            continue

        # safe_medians: train-fold medians, all-NaN columns fall back to 0.0
        medians = train_df[feat_cols].median().fillna(0.0)
        X_tr = train_df[feat_cols].fillna(medians).values
        X_va = df.loc[val_mask, feat_cols].fillna(medians).values

        # weighting="task_length": each task contributes equally
        step_counts = train_df.groupby(group_col)["step"].transform("count")
        sample_weight = (1.0 / step_counts).values

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_tr)
        # Grouped by task (2026-07-30, same fix as train_risk_model_temp_sca.py's
        # _cv3(): plain cv=3 let steps from the same task span the inner
        # train/val split, biasing hyperparameter selection).
        train_task_ids = train_df[group_col].values
        gs = GridSearchCV(
            LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42),
            {"C": [0.01, 0.1, 1.0, 10.0]},
            cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0),
            scoring="roc_auc", n_jobs=1,
        )
        gs.fit(Xs, y_tr, groups=train_task_ids, sample_weight=sample_weight)
        probs[np.where(val_mask)[0]] = gs.best_estimator_.predict_proba(
            scaler.transform(X_va))[:, 1]

    return probs


# ── Per-model runner ───────────────────────────────────────────────────────────

def run_one(results_dir: Path) -> list[dict]:
    pss = results_dir / "per_step_scores.csv"
    if not pss.exists():
        print(f"  [SKIP] missing {pss}")
        return []

    df = pd.read_csv(pss)
    if df["success"].dtype == object:
        df["success"] = df["success"].map({"success": 1, "failure": 0})
    df["success"] = pd.to_numeric(df["success"], errors="coerce")
    df = df.dropna(subset=["success"]).reset_index(drop=True)
    df = add_m2_cumulative(df).rename(
        columns={"m2c_cumulative_action_logprob_mean": "m15_cumulative_action_logprob_mean"})

    model = df["model"].iloc[0] if "model" in df.columns else results_dir.parent.name
    benchmark = df["benchmark"].iloc[0] if "benchmark" in df.columns else "unknown"
    group_col = "task_index" if "task_index" in df.columns else "task_id"
    df[group_col] = df[group_col].astype(str)

    feat_cols = [c for c in BASE_FEATURE_COLS
                 if c in df.columns and df[c].notna().sum() > 0]
    if len(feat_cols) < 2:
        print(f"  [SKIP] only {len(feat_cols)} usable feature columns")
        return []

    # Require at least one observed feature per row
    df = df[df[feat_cols].notna().any(axis=1)].reset_index(drop=True)

    print(f"  features ({len(feat_cols)}): {', '.join(c.split('_')[0] for c in feat_cols)}")
    print(f"  rows: {len(df)}  tasks: {df[group_col].nunique()}  "
          f"step success rate: {df['success'].mean():.3f}")

    probs_htc = run_stacked_cv_htc(df, feat_cols, group_col)
    probs_logl2 = run_stacked_cv_logl2(df, feat_cols, group_col)

    out = df[[c for c in ("task_id", "task_index", "step", "success") if c in df.columns]].copy()
    out["m13_stacked_step"] = probs_htc
    out["m14_stacked_logl2_step"] = probs_logl2
    out.to_csv(results_dir / "stacked_step_scores.csv", index=False)

    success = df["success"].values.astype(float)
    records = []
    for method, probs in [("m13_stacked_step", probs_htc),
                          ("m14_stacked_logl2_step", probs_logl2)]:
        m = evaluate(probs, success)
        if not m:
            continue
        print(f"  {method:<24}  AUROC={m['auroc']:.4f}  "
              f"E-AURC={m['e_aurc']:.4f}  Brier={m['brier']:.4f}")
        records.append({
            "model": model, "benchmark": benchmark, "method": method,
            "n_features": len(feat_cols), "features": "|".join(feat_cols), **m,
        })
    return records


def main():
    ap = argparse.ArgumentParser(description="Stacked L2 logistic baseline over M1-M7 scores.")
    ap.add_argument("--output", default=None,
                    help="Summary CSV path (default: baseline/stacked_logistic_metrics.csv)")
    ap.add_argument("--results_root", default=None,
                    help="Root containing <root>/{webarena,mind2web}/<model>/results/per_step_scores.csv "
                         "(default: this script's own directory, i.e. baseline/ in the working tree). "
                         "Override for a self-contained run against a different output tree, e.g. a "
                         "this release's reproduction run's own Stage-1/2 output directory.")
    args = ap.parse_args()

    results_root = Path(args.results_root) if args.results_root else BASELINE_ROOT
    result_dirs = sorted(results_root.glob("webarena/*/results")) + \
                  sorted(results_root.glob("mind2web/*/results"))

    all_records = []
    for rd in result_dirs:
        print(f"\n{'='*60}\n  {rd.parent.parent.name}/{rd.parent.name}\n{'='*60}")
        all_records.extend(run_one(rd))

    if not all_records:
        print("No results.")
        return

    out_path = Path(args.output) if args.output else BASELINE_ROOT / "stacked_logistic_metrics.csv"
    summary = pd.DataFrame(all_records)
    summary.to_csv(out_path, index=False)

    print(f"\n\n{'='*100}")
    print("  SUMMARY — stacked logistic baselines (M1-M7 feature vector)")
    print("  M13 = HTC_L2 protocol; M14 = Logistic_L2 protocol (paper main-table classifier)")
    print(f"{'='*100}")
    print(summary[["benchmark", "model", "method", "n", "n_features",
                   "auroc", "aurc", "e_aurc", "aurc_gain", "brier", "ece"]].to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
