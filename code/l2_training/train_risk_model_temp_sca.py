#!/usr/bin/env python3
"""
Variant of train_risk_model.py — Scheme 4: Selectable post-hoc calibration.

Calibration scheme (controlled by --calib, applies uniformly to ALL models):

  temperature (default)
    All models       → temperature scaling on the 20% task-level cal split.
      T* minimises NLL on cal-set logit scores (optimised in log-space).
      Logits: decision_function (Logistic), raw_score (LightGBM),
              log-odds of predict_proba (RF, ElasticNet, MLP).
      LSTM: task-level 80/20 split to preserve sequence integrity.

  platt
    All models       → Platt scaling (sigmoid(a·logit + b)) on the 20% cal split.
      (a, b) minimise NLL; adds a bias so the calibrator can shift the mean.
      For RF/Calibrated_RF logits are derived from log-odds of predict_proba.

  isotonic
    All models       → isotonic regression on cal-set probabilities.
      Probabilities: sigmoid(logits) for models with native logits;
                     predict_proba for RF/Calibrated_RF.
      Non-parametric and monotone; higher variance on small cal sets.

  Calibrated_RF is treated identically to RF under all schemes.

Models: Logistic Regression (L2), Ridge, Random Forest, Calibrated RF,
        ElasticNet, LightGBM, Logistic ElasticNet (sigmoid + L1+L2), MLP, LSTM

Sampling strategies:
  uniform  — sample prefixes uniformly at random per task
  balanced — sample one prefix from each of early / middle / late thirds
  fixed    — keep at most --max_prefix prefixes per task (uniform)

Class weights computed from task counts in each outer-train fold:
  w_failure = (N_f + N_s) / (2 * N_f)
  w_success = (N_f + N_s) / (2 * N_s)

Usage:
  python train_risk_model_temp_sca.py --source webarena/claude
  python train_risk_model_temp_sca.py --source mind2web/qwen30b --sampling balanced --max_prefix 5
  python train_risk_model_temp_sca.py --source mind2web/qwen30b --calib platt
  python train_risk_model_temp_sca.py --source mind2web/qwen30b --calib isotonic
  python train_risk_model_temp_sca.py --source mind2web/claude \
      --csv_path mind2web/embeded_hlvm/claude/uq_analysis_change_qwen/thr_0p3/feature_matrix.csv
"""

import argparse
import json
import random
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightgbm import LGBMClassifier, early_stopping as lgb_early_stopping, log_evaluation as lgb_log_evaluation
from scipy.optimize import minimize_scalar
from scipy.special import expit as sigmoid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, make_scorer, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _cv3() -> StratifiedGroupKFold:
    """Inner CV splitter for GridSearchCV. Groups by task_id so that steps
    from the same task never span both the inner-train and inner-val split
    (fixed 2026-07-30: was a plain row-level `cv=3`, which let hyperparameter
    search "peek" at other steps of a task it was validating on, since
    adjacent steps of the same task are highly correlated). Every call site
    must also pass `groups=<per-row task_id array>` to `.fit()`."""
    return StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)

# ── Feature sets ──────────────────────────────────────────────────────────────

MACRO_FEATS = [
    "step_k", "n_single_loops", "n_double_loops", "loop_length_current",
    "n_distinct_actions", "action_count_current", "max_action_count",
    "action_repeat_rate", "n_grounding_errors", "n_parse_errors",
    "consecutive_errors_now", "grounding_error_at_k", "parse_error_at_k",
    "grounding_loop_count", "n_click", "n_type", "n_scroll", "n_go_back",
    "n_distinct_elements", "action_type_entropy", "steps_since_new_action",
    "type_text_length",
    "thought_action_ineffective_at_k", "thought_action_ineffective_cumul_count",
    "thought_wrong_context_at_k", "thought_wrong_context_cumul_count",
    "thought_environment_error_at_k", "thought_environment_error_cumul_count",
    "reasoning_len_at_k", "reasoning_len_cumul_mean", "reasoning_len_cumul_max",
]

UQ_FEATS = []
for ind in ["U_sem", "U_change", "U_ground", "U_cval", "C_S", "R_T"]:
    for suf in ["at_k", "cumul_mean", "cumul_max"]:
        UQ_FEATS.append(f"{ind}_{suf}")

FEATURE_SETS = {
    "macro_only": MACRO_FEATS,
    "uq_only":    UQ_FEATS,
    "all":        MACRO_FEATS + UQ_FEATS,
}


# ── Keystep label remapping ──────────────────────────────────────────────────

def load_annotation(annot_path: Path) -> dict:
    """Load annotation JSON and return taskId (str) -> keyStep (int or None)."""
    with open(annot_path) as f:
        data = json.load(f)
    return {v["taskId"]: v.get("keyStep") for v in data["items"].values()}


def resolve_annotation_path(results_dir: Path, source: str,
                            annot_dir: Path | None = None,
                            annot_path: Path | None = None) -> Path:
    """Resolve the Ours/L2-pipeline keystep annotation file for one source.
    When annot_dir is not explicitly given, searches both known annotation roots:
      annotation/result/      (webarena per-model annotations)
      annotate_vis/result/    (mind2web per-model annotations)

    Each benchmark has TWO disagreeing annotation passes per model (confirmed
    2026-07-30, ~75-90% keyStep disagreement on shared tasks), split by which
    raw trajectory source they were annotated against:

      mind2web  <model>_gemini.json     (no suffix) -- plain raw trajectories
                                          -> THIS pipeline (Ours/L2)
                <model>_v_gemini.json   -- verbalized raw trajectories
                                          -> baseline/keystep_labels.py (baseline)
      webarena  <model>_gemini.json     (no suffix) -- verbalized raw trajectories
                                          -> baseline/keystep_labels.py (baseline)
                <model>_m_gemini.json   -- plain/multi-sample raw trajectories
                                          -> THIS pipeline (Ours/L2)

    i.e. the "no suffix" file is Ours/L2's file for mind2web but is the
    BASELINE's file for webarena -- the two benchmarks use the opposite
    convention. Confirmed empirically: for every model, exactly one of the
    two files has a 0-orphan task-id match against
    <benchmark>/results_<model>_verbalized/ (that one is the baseline file);
    the other matches the plain raw dir (that one is this pipeline's file).
    """
    if annot_path is not None:
        return annot_path
    stem = source.replace("/", "_")
    bench = source.split("/", 1)[0]
    if annot_dir is not None:
        search_dirs = [annot_dir]
    else:
        search_dirs = [
            results_dir / "annotation" / "result",
            results_dir / "annotate_vis" / "result",
        ]
    if bench == "webarena":
        suffixes = [f"{stem}_m_gemini.json", f"{stem}_m.json"]
    else:
        suffixes = [f"{stem}_gemini.json", f"{stem}.json"]
    for d in search_dirs:
        for suffix in suffixes:
            cand = d / suffix
            if cand.exists():
                return cand
    # fallback to first candidate in first search dir (will trigger FileNotFoundError)
    return search_dirs[0] / suffixes[0]


def apply_keystep_labels(df: pd.DataFrame, annotation: dict) -> pd.DataFrame:
    """
    Relabel steps using keystep annotations.

    - Annotated failed task:   step_k < keyStep -> label=0, >= keyStep -> label=1
    - Unannotated failed task: kept with label=outcome (all steps label=1)
    - Success task:            always kept with label=0, never dropped
    """
    df = df.copy()
    annotated_with_ks = {tid for tid, ks in annotation.items() if ks is not None}

    failed_mask = df["outcome"] == 1
    task_id_str = df["task_id"].astype(str)
    unannotated_failed = df.loc[failed_mask & ~task_id_str.isin(annotated_with_ks), "task_id"].nunique()

    df["label"] = df["outcome"].copy()
    relabeled = 0
    for task_id_str_val, key_step in annotation.items():
        if key_step is None:
            continue
        mask = df["task_id"].astype(str) == str(task_id_str_val)
        if not mask.any():
            continue
        if int(df.loc[mask, "outcome"].iloc[0]) == 0:
            continue
        df.loc[mask & (df["step_k"] < key_step), "label"] = 0
        df.loc[mask & (df["step_k"] >= key_step), "label"] = 1
        relabeled += 1

    print(f"  [keystep] relabeled={relabeled} failed tasks  "
          f"unannotated_kept_as_raw={unannotated_failed}  "
          f"total tasks={df['task_id'].nunique()}", flush=True)
    return df


def label_col(df: pd.DataFrame) -> str:
    return "label" if "label" in df.columns else "outcome"


def resolve_output_dir(results_root: Path, source: str, csv_path: Path,
                       calib: str, sampling: str, max_prefix: int,
                       inner_metric: str, ece_mode: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = (f"keystep_sampling={sampling}_prefix={max_prefix}"
               f"_inner={inner_metric}_calib={calib}_ece={ece_mode}_{timestamp}")

    try:
        rel = csv_path.resolve().relative_to(results_root.resolve())
    except ValueError:
        rel = Path(source.replace("/", "_")) / csv_path.parent.name / csv_path.name

    parts = rel.parts
    if len(parts) >= 6 and parts[1] == "embeded_hlvm":
        dataset = parts[0]
        model = parts[2]
        variant = parts[3]
        thr = parts[4]
        return results_root / "uq_results" / dataset / model / variant / thr / "train_risk_model_temp_sca" / run_tag

    return results_root / "uq_results" / source.replace("/", "_") / csv_path.parent.name / "train_risk_model_temp_sca" / run_tag

# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_prefixes(task_df: pd.DataFrame, strategy: str,
                    max_prefix: int, rng: np.random.Generator) -> pd.DataFrame:
    n = len(task_df)
    if n == 0:
        return task_df

    if strategy == "full":
        idx = np.arange(n)

    elif strategy == "uniform":
        k = min(max_prefix, n)
        idx = rng.choice(n, size=k, replace=False)

    elif strategy == "balanced":
        thirds = np.array_split(np.arange(n), 3)
        idx = np.array([rng.choice(seg) for seg in thirds if len(seg) > 0])

    elif strategy == "fixed":
        k = min(max_prefix, n)
        idx = rng.choice(n, size=k, replace=False)

    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    return task_df.iloc[sorted(idx)]


def build_train_sample(train_df: pd.DataFrame, strategy: str,
                       max_prefix: int, seed: int = 42) -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    chunks = []
    for _, tdf in train_df.groupby("task_id"):
        tdf_sorted = tdf.sort_values("step_k")
        chunks.append(sample_prefixes(tdf_sorted, strategy, max_prefix, rng))
    return pd.concat(chunks, ignore_index=True)


# ── Task-level class weights ──────────────────────────────────────────────────

def task_class_weights(train_df: pd.DataFrame):
    task_labels = train_df.groupby("task_id")["outcome"].first()
    N_f = (task_labels == 1).sum()
    N_s = (task_labels == 0).sum()
    total = N_f + N_s

    if N_f == 0 or N_s == 0:
        return pd.Series(np.ones(len(train_df)), index=train_df.index)

    w_f = total / (2.0 * N_f)
    w_s = total / (2.0 * N_s)

    weight_map = task_labels.map({1: w_f, 0: w_s})
    return train_df["task_id"].map(weight_map).values


# ── Metrics ───────────────────────────────────────────────────────────────────

def ece(y_true, y_prob, n_bins=5):
    """Adaptive ECE (equal-mass binning): each bin holds the same number of
    samples, sorted by predicted probability.  Unlike equal-width binning this
    cannot be trivially minimised by near-constant predictors that cluster in
    one region of [0, 1], making it comparable across methods with very
    different score distributions."""
    order    = np.argsort(y_prob)
    y_sorted = y_true[order]
    p_sorted = y_prob[order]
    val = 0.0
    for chunk in np.array_split(np.arange(len(y_true)), n_bins):
        if len(chunk) == 0:
            continue
        val += len(chunk) * abs(y_sorted[chunk].mean() - p_sorted[chunk].mean())
    return val / max(len(y_true), 1)


def ece_equal_width(y_true, y_prob, n_bins=10):
    """Standard ECE (equal-width binning): [0,1] divided into n_bins fixed-width
    intervals.  Each bin contributes |mean_confidence - mean_accuracy| weighted
    by its fraction of samples.  Traditional formulation; may be unreliable when
    predictions cluster in a narrow region."""
    y_prob = np.clip(y_prob, 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    val = 0.0
    n   = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi if i == n_bins - 1 else y_prob < hi)
        if mask.sum() == 0:
            continue
        val += mask.sum() / n * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(val)


def aurc(y_true, y_prob):
    """
    Area Under the Risk-Coverage curve (selective prediction).
    Samples are ordered by ascending y_prob (most confident / lowest risk first).
    Risk at coverage c = failure rate among the c-fraction selected.
    Lower is better.

    Also returns:
      aurc_baseline : AURC of a random model = overall failure rate
      e_aurc        : excess AURC = aurc - baseline  (negative = better than random)
    """
    order    = np.argsort(y_prob)
    y_sorted = y_true[order]
    risks    = np.cumsum(y_sorted) / np.arange(1, len(y_sorted) + 1)
    score    = float(np.mean(risks))
    baseline = float(np.mean(y_true))
    return score, baseline, score - baseline


def aurc_oracle(y_true: np.ndarray) -> float:
    """AURC of a perfect oracle ordering (all successes/0s ranked before
    failures/1s -- y_true here is 1=failure, this script's own convention).
    Same formula as baseline/stacked_logistic_baseline.py's aurc_oracle() and
    baseline/update_baselines_with_features.py's aurc_oracle() (2026-07-31,
    kept identical across all three files so e_aurc = raw_aurc - aurc_oracle,
    the standard "Excess-AURC" from the selective prediction literature,
    means the same thing everywhere it's reported -- always >= 0,
    lower=better/closer-to-oracle). n_f/n_s are counted directly from
    y_true's own 1=failure encoding here, equivalent to computing them from
    a 1=success array as the other two files do."""
    n = len(y_true)
    n_f = int(np.sum(y_true))
    n_s = n - n_f
    if n_f == 0:
        return 0.0
    if n_s == 0:
        return 1.0
    i_vals = np.arange(n_s + 1, n + 1, dtype=float)
    return float(np.sum((i_vals - n_s) / i_vals) / n)


def rc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Full Risk-Coverage curve (most-confident steps selected first)."""
    valid    = ~np.isnan(y_prob)
    yt, yp   = y_true[valid], y_prob[valid]
    order    = np.argsort(yp)
    n        = len(order)
    coverages = np.arange(1, n + 1) / n
    risks     = np.cumsum(yt[order]) / np.arange(1, n + 1)
    return pd.DataFrame({"coverage": coverages, "risk": risks,
                         "threshold": yp[order]})


def risk_at_coverage(y_true: np.ndarray, y_prob: np.ndarray,
                     coverages: tuple = (0.5, 0.6, 0.7, 0.8, 0.9)) -> dict:
    """Failure rate when selecting the most-confident c-fraction of steps."""
    valid  = ~np.isnan(y_prob)
    yt, yp = y_true[valid], y_prob[valid]
    order  = np.argsort(yp)
    y_sorted = yt[order]
    n = len(y_sorted)
    return {f"risk@{int(c*100)}": round(float(y_sorted[:max(1, int(np.floor(c * n)))].mean()), 4)
            for c in coverages}


def coverage_at_risk(y_true: np.ndarray, y_prob: np.ndarray,
                     thresholds: tuple = (0.05, 0.10, 0.20, 0.30)) -> dict:
    """Largest coverage achievable while keeping failure rate ≤ threshold."""
    valid  = ~np.isnan(y_prob)
    yt, yp = y_true[valid], y_prob[valid]
    order  = np.argsort(yp)
    n      = len(order)
    cum_risk = np.cumsum(yt[order]) / np.arange(1, n + 1)
    out = {}
    for t in thresholds:
        feasible = np.where(cum_risk <= t)[0]
        out[f"cov@risk{int(t*100)}"] = round(float((feasible[-1] + 1) / n) if len(feasible) else 0.0, 4)
    return out


def make_inner_scorer(metric: str):
    """
    Return a sklearn-compatible scorer for inner CV hyperparameter search.
      metric="auroc"  →  standard "roc_auc" string (higher is better)
      metric="aurc"   →  custom scorer minimising AURC (lower is better)
      metric="brier"  →  minimise Brier score (lower is better)
      metric="log_loss" → minimise log loss (lower is better)
    LightGBM early-stopping and ElasticNet (regression) always use their
    own fixed metrics regardless of this setting.
    """
    if metric == "auroc":
        return "roc_auc"
    if metric == "brier":
        return make_scorer(brier_score_loss, greater_is_better=False, needs_proba=True)
    if metric == "log_loss":
        return make_scorer(log_loss, greater_is_better=False, needs_proba=True)
    def _aurc_fn(y_true, y_prob):
        order    = np.argsort(y_prob)
        y_sorted = y_true[order]
        risks    = np.cumsum(y_sorted) / np.arange(1, len(y_sorted) + 1)
        return float(np.mean(risks)) - float(np.mean(y_true))
    return make_scorer(_aurc_fn, greater_is_better=False, needs_proba=True)


def evaluate(y_true, y_prob, ece_mode: str = "adaptive") -> dict:
    valid = ~np.isnan(y_prob)
    yt, yp = y_true[valid], y_prob[valid]
    if len(np.unique(yt)) < 2:
        return {}
    out = {
        "n_samples":   int(valid.sum()),
        "auroc":       round(float(roc_auc_score(yt, yp)),          4),
        "auprc":       round(float(average_precision_score(yt, yp)), 4),
        "aurc":        round(float(aurc(yt, yp)[2]),                  4),
        "aurc_base":   round(float(aurc(yt, yp)[1]),                  4),
        "e_aurc":      round(float(aurc(yt, yp)[0]) - aurc_oracle(yt), 4),
        "brier":       round(float(brier_score_loss(yt, yp)),        4),
        "bss":         round(float(1 - brier_score_loss(yt, yp) / (float(np.mean(yt)) * (1 - float(np.mean(yt))))), 4),
    }
    if ece_mode in ("adaptive", "both"):
        out["ece_adaptive"] = round(float(ece(yt, yp)),             4)
    if ece_mode in ("equal_width", "both"):
        out["ece_ew"]       = round(float(ece_equal_width(yt, yp)), 4)
    # backward-compat alias: "ece" always points to whichever single mode is active
    if ece_mode == "adaptive":
        out["ece"] = out["ece_adaptive"]
    elif ece_mode == "equal_width":
        out["ece"] = out["ece_ew"]
    out.update(risk_at_coverage(yt, yp, coverages=(0.5, 0.6, 0.7, 0.8, 0.9)))
    out.update(coverage_at_risk(yt, yp, thresholds=(0.05, 0.10, 0.20, 0.30)))
    return out


# ── Calibration helpers ───────────────────────────────────────────────────────

# Models that have no native logits — calibration uses predict_proba outputs
RF_MODELS = {"RF", "Calibrated_RF"}

# Per-model default calibration (used when --calib auto).
# Override with --calib temperature|platt|isotonic to apply one method globally.
MODEL_CALIB = {
    "Logistic_L2":         "temperature",  # true log-odds; 1-param rescale is robust
    "RF":                  "temperature",
    "Calibrated_RF":       "temperature",
    "ElasticNet":          "platt",        # regression output; affine Platt fit is stable
    "LightGBM":            "platt",        # overconfident trees; Platt reliable on small data
    "Logistic_ElasticNet": "temperature",  # true log-odds from decision_function
    "MLP":                 "temperature",  # Guo et al. 2017
    "LSTM":                "temperature",  # neural net; hardcoded to temp-scale path
}

NO_POSTHOC_CALIB_MODELS = {"Logistic_L2", "Logistic_ElasticNet", "HTC_L2", "HTC_L1"}

# sklearn models that use holdout temperature calibration
SKLEARN_OOF_MODELS = {"ElasticNet", "LightGBM", "MLP"}
# sequence models that use an 80/20 hold-out for calibration
SEQ_MODELS = {"LSTM", "Transformer"}


def calibration_variants(name: str, calib: str) -> list[tuple[str, str]]:
    """
    Return [(method, result_label), ...] for one base model.
    - Models in NO_POSTHOC_CALIB_MODELS: raw only, no suffix.
    - All other models: always both raw and temperature-calibrated.
    """
    if name in NO_POSTHOC_CALIB_MODELS:
        return [("none", name)]
    return [("none", f"{name}[raw]"), ("temperature", f"{name}[temperature]")]


def safe_medians(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """
    Per-column medians with a final fallback of 0.0 for columns that are all-NaN
    in the current training fold.
    """
    meds = df[cols].median()
    return meds.fillna(0.0)


def temperature_scale(logits_cal: np.ndarray, y_cal: np.ndarray,
                      logits_test: np.ndarray,
                      sample_weight: np.ndarray):
    """
    Find optimal temperature T* on the cal set by minimising weighted NLL,
    then apply to test logits.  T is optimised in log-space to keep it strictly
    positive.  sample_weight must be provided (e.g. 1/task_length so every task
    contributes equally, consistent with how the model was trained).

    Returns (calibrated_probs, T_opt).
    """
    def nll(log_t):
        T = np.exp(log_t)
        p = sigmoid(logits_cal / T)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        ll = y_cal * np.log(p) + (1 - y_cal) * np.log(1 - p)
        return -float(np.average(ll, weights=sample_weight))

    res   = minimize_scalar(nll, bounds=(-4.0, 5.0), method="bounded")
    T_opt = float(np.exp(res.x))
    return sigmoid(logits_test / T_opt), T_opt


def platt_scale(logits_cal: np.ndarray, y_cal: np.ndarray,
                logits_test: np.ndarray):
    """
    Platt scaling: fit sigmoid(a·logit + b) by minimising NLL on the cal set,
    then apply to test logits.  Two free parameters allow shifting the mean,
    unlike single-parameter temperature scaling.

    Returns (calibrated_probs, a_opt, b_opt).
    """
    from scipy.optimize import minimize as _minimize

    def nll_and_grad(params):
        a, b = params
        p = sigmoid(a * logits_cal + b)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        nll_val = -float(np.mean(y_cal * np.log(p) + (1 - y_cal) * np.log(1 - p)))
        err = p - y_cal
        grad_a = float(np.mean(err * logits_cal))
        grad_b = float(np.mean(err))
        return nll_val, np.array([grad_a, grad_b])

    res = _minimize(nll_and_grad, x0=[1.0, 0.0], method="L-BFGS-B", jac=True,
                    options={"maxiter": 1000, "ftol": 1e-10})
    a_opt, b_opt = res.x
    return sigmoid(a_opt * logits_test + b_opt), a_opt, b_opt


def isotonic_scale(probs_cal: np.ndarray, y_cal: np.ndarray,
                   probs_test: np.ndarray) -> np.ndarray:
    """
    Isotonic regression calibration on probability outputs.
    Non-parametric monotone mapping; fit on cal set, applied to test set.

    Returns calibrated_probs.
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs_cal, y_cal)
    return iso.predict(probs_test)


def get_logits(clf, X: np.ndarray, name: str) -> np.ndarray:
    """
    Extract logit-scale scores from a fitted model.

    - Logistic_L2, Logistic_ElasticNet : decision_function (true log-odds)
    - LightGBM                          : predict with raw_score=True
    - ElasticNet, MLP, others           : log-odds of predict_proba
    """
    if name in ("Logistic_L2", "Logistic_ElasticNet", "HTC_L2", "HTC_L1"):
        return clf.decision_function(X)
    if name == "LightGBM":
        return clf.predict(X, raw_score=True)
    # ElasticNet (regression wrapper) and MLP: convert probabilities to log-odds
    p = clf.predict_proba(X)[:, 1]
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


# ── Model wrappers ────────────────────────────────────────────────────────────

class ElasticNetScorer:
    def __init__(self, inner_cv=3):
        self._inner_cv = inner_cv
        self._best     = None
        self._scaler   = StandardScaler()

    def fit(self, X, y, sample_weight=None, groups=None):
        Xs = self._scaler.fit_transform(X)
        param_grid = {
            "alpha":    [0.001, 0.01, 0.1, 1.0],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
        base = ElasticNet(max_iter=5000)
        gs   = GridSearchCV(base, param_grid, cv=_cv3(),
                            scoring="neg_mean_squared_error", n_jobs=1)
        gs.fit(Xs, y, groups=groups, sample_weight=sample_weight)
        self._best = gs.best_estimator_
        return self

    def predict_proba(self, X):
        Xs = self._scaler.transform(X)
        s  = np.clip(self._best.predict(Xs), 0.0, 1.0)
        return np.column_stack([1 - s, s])


class LogisticElasticNetScorer:
    def __init__(self, inner_cv=3):
        self._inner_cv = inner_cv
        self._best     = None
        self._scaler   = StandardScaler()

    def fit(self, X, y, sample_weight=None, groups=None):
        Xs         = self._scaler.fit_transform(X)
        param_grid = {
            "C":        [0.01, 0.1, 1.0, 10.0],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
        base = LogisticRegression(
            penalty="elasticnet", solver="saga",
            max_iter=2000, random_state=42,
        )
        gs = GridSearchCV(base, param_grid, cv=_cv3(),
                          scoring="roc_auc", n_jobs=1)
        gs.fit(Xs, y, groups=groups, sample_weight=sample_weight)
        self._best = gs.best_estimator_
        return self

    def predict_proba(self, X):
        Xs = self._scaler.transform(X)
        return self._best.predict_proba(Xs)

    def decision_function(self, X):
        Xs = self._scaler.transform(X)
        return self._best.decision_function(Xs)


# ── LSTM ──────────────────────────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0.2 if num_layers > 1 else 0.0)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):                          # x: (1, T, F)
        out, _ = self.lstm(x)                      # out: (1, T, H)
        return torch.sigmoid(self.fc(out)).squeeze(-1)  # (1, T)

    def logits(self, x):                           # pre-sigmoid fc output
        out, _ = self.lstm(x)
        return self.fc(out).squeeze(-1)            # (1, T)


class LSTMScorer:
    """LSTM risk scorer — processes full step sequences per task."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 1,
                 epochs: int = 50, lr: float = 1e-3, patience: int = 10):
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.epochs      = epochs
        self.lr          = lr
        self.patience    = patience
        self._model      = None
        self._scaler     = StandardScaler()
        # MPS LSTM kernel is unstable (crashes on short sequences); force CPU.
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

    def _build_sequences(self, df: pd.DataFrame, feat_cols: list,
                          medians: pd.Series):
        task_labels = df.groupby("task_id")["outcome"].first()
        N_f   = (task_labels == 1).sum()
        N_s   = (task_labels == 0).sum()
        total = N_f + N_s
        w_f   = float(total / (2.0 * N_f)) if N_f > 0 else 1.0
        w_s   = float(total / (2.0 * N_s)) if N_s > 0 else 1.0

        seqs = []
        for task_id, tdf in df.groupby("task_id"):
            tdf = tdf.sort_values("step_k")
            X   = self._scaler.transform(tdf[feat_cols].fillna(medians).values)
            y   = tdf["outcome"].values.astype(np.float32)
            w_val = w_f if int(task_labels[task_id]) == 1 else w_s
            w   = np.full(len(y), w_val, dtype=np.float32)
            seqs.append((torch.FloatTensor(X),
                         torch.FloatTensor(y),
                         torch.FloatTensor(w)))
        return seqs

    def fit(self, train_df: pd.DataFrame, feat_cols: list,
            medians: pd.Series, seed: int = 42):
        torch.manual_seed(seed)

        self._scaler.fit(train_df[feat_cols].fillna(medians).values)
        self._model = _LSTMNet(len(feat_cols), self.hidden_size, self.num_layers).to(self._device)
        optimizer   = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        all_tasks = train_df["task_id"].unique()
        rng       = np.random.default_rng(seed)
        rng.shuffle(all_tasks)
        cut       = int(0.8 * len(all_tasks))
        fit_df    = train_df[train_df["task_id"].isin(all_tasks[:cut])]
        val_df    = train_df[train_df["task_id"].isin(all_tasks[cut:])]

        fit_seqs  = self._build_sequences(fit_df, feat_cols, medians)
        val_seqs  = self._build_sequences(val_df, feat_cols, medians)

        best_auroc, best_epoch, best_state = 0.0, 0, None

        for epoch in range(1, self.epochs + 1):
            self._model.train()
            order = np.random.default_rng(seed + epoch).permutation(len(fit_seqs))
            for i in order:
                X, y, w = fit_seqs[i]
                X, y, w = X.to(self._device), y.to(self._device), w.to(self._device)
                out  = self._model(X.unsqueeze(0)).squeeze(0)
                loss = (F.binary_cross_entropy(out, y, reduction="none") * w).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self._model.eval()
            vp, vl = [], []
            with torch.no_grad():
                for X, y, _ in val_seqs:
                    X = X.to(self._device)
                    out = self._model(X.unsqueeze(0)).squeeze(0)
                    vp.extend(out.cpu().numpy().tolist())
                    vl.extend(y.numpy().tolist())
            vp, vl = np.array(vp), np.array(vl)
            if len(np.unique(vl)) >= 2:
                val_auroc = roc_auc_score(vl, vp)
                if val_auroc > best_auroc:
                    best_auroc = val_auroc
                    best_epoch = epoch
                    best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                elif epoch - best_epoch >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        print(f"      [LSTM]         best_epoch={best_epoch}  "
              f"val AUROC={best_auroc:.4f}", flush=True)
        return self

    def predict_proba(self, test_df: pd.DataFrame, feat_cols: list,
                       medians: pd.Series) -> np.ndarray:
        self._model.eval()
        test_df = test_df.reset_index(drop=True)
        result  = np.full(len(test_df), np.nan)
        with torch.no_grad():
            for _, tdf in test_df.groupby("task_id"):
                tdf_s = tdf.sort_values("step_k")
                X     = self._scaler.transform(tdf_s[feat_cols].fillna(medians).values)
                out   = self._model(torch.FloatTensor(X).to(self._device).unsqueeze(0)).squeeze(0).cpu().numpy()
                result[tdf_s.index] = out
        return result

    def predict_logits(self, test_df: pd.DataFrame, feat_cols: list,
                       medians: pd.Series) -> np.ndarray:
        """Return pre-sigmoid fc outputs (logit scale) for temperature scaling."""
        self._model.eval()
        test_df = test_df.reset_index(drop=True)
        result  = np.full(len(test_df), np.nan)
        with torch.no_grad():
            for _, tdf in test_df.groupby("task_id"):
                tdf_s = tdf.sort_values("step_k")
                X     = self._scaler.transform(tdf_s[feat_cols].fillna(medians).values)
                out   = self._model.logits(torch.FloatTensor(X).to(self._device).unsqueeze(0)).squeeze(0).cpu().numpy()
                result[tdf_s.index] = out
        return result


# ── Transformer ───────────────────────────────────────────────────────────────

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class _TransformerNet(nn.Module):
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = _PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def _encode(self, x):
        return self.encoder(self.pos_enc(self.input_proj(x)))

    def forward(self, x):
        return torch.sigmoid(self.fc(self._encode(x))).squeeze(-1)

    def logits(self, x):
        return self.fc(self._encode(x)).squeeze(-1)


class TransformerScorer:
    """Transformer risk scorer — sequence model parallel to LSTMScorer."""

    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, epochs: int = 50, lr: float = 1e-3,
                 patience: int = 10):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self._model = None
        self._scaler = StandardScaler()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_sequences(self, df: pd.DataFrame, feat_cols: list, medians: pd.Series):
        current_label_col = label_col(df)
        task_labels = df.groupby("task_id")["outcome"].first()
        N_f = (task_labels == 1).sum()
        N_s = (task_labels == 0).sum()
        total = N_f + N_s
        w_f = float(total / (2.0 * N_f)) if N_f > 0 else 1.0
        w_s = float(total / (2.0 * N_s)) if N_s > 0 else 1.0

        seqs = []
        for task_id, tdf in df.groupby("task_id"):
            tdf = tdf.sort_values("step_k")
            X = self._scaler.transform(tdf[feat_cols].fillna(medians).values)
            y = tdf[current_label_col].values.astype(np.float32)
            w_val = w_f if int(task_labels[task_id]) == 1 else w_s
            w = np.full(len(y), w_val, dtype=np.float32)
            seqs.append((torch.FloatTensor(X), torch.FloatTensor(y), torch.FloatTensor(w)))
        return seqs

    def fit(self, train_df: pd.DataFrame, feat_cols: list, medians: pd.Series, seed: int = 42):
        torch.manual_seed(seed)
        self._scaler.fit(train_df[feat_cols].fillna(medians).values)
        self._model = _TransformerNet(
            len(feat_cols), self.d_model, self.nhead, self.num_layers, self.dropout
        ).to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        all_tasks = train_df["task_id"].unique()
        rng = np.random.default_rng(seed)
        rng.shuffle(all_tasks)
        cut = int(0.8 * len(all_tasks))
        fit_df = train_df[train_df["task_id"].isin(all_tasks[:cut])]
        val_df = train_df[train_df["task_id"].isin(all_tasks[cut:])]

        fit_seqs = self._build_sequences(fit_df, feat_cols, medians)
        val_seqs = self._build_sequences(val_df, feat_cols, medians)

        best_auroc, best_epoch, best_state = 0.0, 0, None
        for epoch in range(1, self.epochs + 1):
            self._model.train()
            order = np.random.default_rng(seed + epoch).permutation(len(fit_seqs))
            for i in order:
                X, y, w = fit_seqs[i]
                X, y, w = X.to(self._device), y.to(self._device), w.to(self._device)
                out = self._model(X.unsqueeze(0)).squeeze(0)
                loss = (F.binary_cross_entropy(out, y, reduction="none") * w).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self._model.eval()
            vp, vl = [], []
            with torch.no_grad():
                for X, y, _ in val_seqs:
                    out = self._model(X.to(self._device).unsqueeze(0)).squeeze(0)
                    vp.extend(out.cpu().numpy().tolist())
                    vl.extend(y.numpy().tolist())
            vp, vl = np.array(vp), np.array(vl)
            if len(np.unique(vl)) >= 2:
                val_auroc = roc_auc_score(vl, vp)
                if val_auroc > best_auroc:
                    best_auroc = val_auroc
                    best_epoch = epoch
                    best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                elif epoch - best_epoch >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        print(f"      [Transformer]  best_epoch={best_epoch}  val AUROC={best_auroc:.4f}", flush=True)
        return self

    def predict_proba(self, test_df: pd.DataFrame, feat_cols: list, medians: pd.Series) -> np.ndarray:
        self._model.eval()
        test_df = test_df.reset_index(drop=True)
        result = np.full(len(test_df), np.nan)
        with torch.no_grad():
            for _, tdf in test_df.groupby("task_id"):
                tdf_s = tdf.sort_values("step_k")
                X = self._scaler.transform(tdf_s[feat_cols].fillna(medians).values)
                out = self._model(torch.FloatTensor(X).to(self._device).unsqueeze(0)).squeeze(0).cpu().numpy()
                result[tdf_s.index] = out
        return result

    def predict_logits(self, test_df: pd.DataFrame, feat_cols: list, medians: pd.Series) -> np.ndarray:
        self._model.eval()
        test_df = test_df.reset_index(drop=True)
        result = np.full(len(test_df), np.nan)
        with torch.no_grad():
            for _, tdf in test_df.groupby("task_id"):
                tdf_s = tdf.sort_values("step_k")
                X = self._scaler.transform(tdf_s[feat_cols].fillna(medians).values)
                out = self._model.logits(torch.FloatTensor(X).to(self._device).unsqueeze(0)).squeeze(0).cpu().numpy()
                result[tdf_s.index] = out
        return result


# ── Model registry ────────────────────────────────────────────────────────────

RF_PARAMS  = dict(n_estimators=200, max_depth=8, min_samples_leaf=5,
                  n_jobs=1)  # random_state set per-call in fit_model_fast/fit_model
LGB_PARAMS = dict(n_estimators=500, max_depth=4, min_child_samples=10,
                  learning_rate=0.05, verbose=-1, n_jobs=1, random_state=42)

# HTC baseline alpha grid (C = 1/alpha)
ALPHA_GRID_HTC = [0.001, 0.01, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0,
                  6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 50.0]


def best_alpha_htc(X_tr_scaled: np.ndarray, y_tr: np.ndarray, groups: np.ndarray) -> float:
    """
    Select L1 alpha by 3-fold inner CV using the HTC composite score:
      AUROC - Brier - ECE_ew   (simultaneously optimises discrimination + calibration).
    X_tr_scaled must already be StandardScaler-transformed.
    Grouped by task_id (same reasoning as _cv3()) so no task's steps span the
    inner-train/inner-val split.
    """
    best_score, best_alpha = -np.inf, 1.0
    inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)
    for alpha in ALPHA_GRID_HTC:
        model = LogisticRegression(penalty="l1", C=1.0 / alpha,
                                   solver="liblinear", max_iter=1000, random_state=42)
        fold_scores = []
        for itr, iva in inner_cv.split(X_tr_scaled, y_tr, groups=groups):
            if len(np.unique(y_tr[itr])) < 2 or len(np.unique(y_tr[iva])) < 2:
                continue
            model.fit(X_tr_scaled[itr], y_tr[itr])
            p = model.predict_proba(X_tr_scaled[iva])[:, 1]
            fold_scores.append(
                roc_auc_score(y_tr[iva], p)
                - brier_score_loss(y_tr[iva], p)
                - ece_equal_width(y_tr[iva], p)
            )
        if fold_scores and float(np.mean(fold_scores)) > best_score:
            best_score, best_alpha = float(np.mean(fold_scores)), alpha
    return best_alpha


def make_models():
    return {
        "Logistic_L2": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"),
        ),
        "RF":                 RandomForestClassifier(**RF_PARAMS),
        "Calibrated_RF":      None,   # built inside fit_model (GridSearchCV like RF)
        "ElasticNet":         ElasticNetScorer(inner_cv=3),
        "LightGBM":           LGBMClassifier(**LGB_PARAMS),
        "Logistic_ElasticNet": LogisticElasticNetScorer(inner_cv=3),
        "MLP":                None,   # built inside fold (inner GridSearchCV)
        "LSTM":               None,   # built inside fold (sequence model)
        "Transformer":        None,   # built inside fold (sequence model)
    }


def fit_model(name, X_train, y_train, sample_weight, seed=42,
              inner_scorer="roc_auc", task_ids=None):
    models = make_models()

    # Calibrated_RF: same GridSearchCV tuning as RF.
    # Outer isotonic pass (same as RF) handles calibration — no internal split.
    if name in ("RF", "Calibrated_RF"):
        gs = GridSearchCV(
            RandomForestClassifier(
                n_jobs=1, random_state=seed,
            ),
            {
                "n_estimators":      [200],
                "max_depth":         [4, 8, None],
                "min_samples_leaf":  [3, 5, 10],
                "min_samples_split": [2, 5, 10],
                "max_features":      ["sqrt"],
            },
            cv=_cv3(), scoring=inner_scorer, n_jobs=1,
        )
        gs.fit(X_train, y_train, groups=task_ids, sample_weight=sample_weight)
        print(f"      [{name:<14}] best {gs.best_params_}  "
              f"inner-val AUROC={gs.best_score_:.4f}", flush=True)
        return gs.best_estimator_

    if name == "Logistic_L2":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        gs = GridSearchCV(
            LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42),
            {"C": [0.01, 0.1, 1.0, 10.0]},
            cv=_cv3(), scoring=inner_scorer, n_jobs=1,
        )
        gs.fit(Xs, y_train, groups=task_ids, sample_weight=sample_weight)
        print(f"      [Logistic_L2]  best C={gs.best_params_['C']}  "
              f"inner-val AUROC={gs.best_score_:.4f}", flush=True)
        clf = make_pipeline(scaler, gs.best_estimator_)
        clf.steps[0] = ("standardscaler", scaler)
        return clf

    if name == "HTC_L2":
        scaler = StandardScaler()
        Xs  = scaler.fit_transform(X_train)
        clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                 max_iter=2000, random_state=42)
        clf.fit(Xs, y_train, sample_weight=sample_weight)
        print(f"      [HTC_L2]       C=1.0 (fixed L2)", flush=True)
        return make_pipeline(scaler, clf)

    if name == "HTC_L1":
        scaler     = StandardScaler()
        Xs         = scaler.fit_transform(X_train)
        best_alpha = best_alpha_htc(Xs, y_train, task_ids)
        clf        = LogisticRegression(penalty="l1", C=1.0 / best_alpha,
                                        solver="liblinear", max_iter=2000, random_state=42)
        clf.fit(Xs, y_train, sample_weight=sample_weight)
        print(f"      [HTC_L1]       best_alpha={best_alpha}  C={1.0/best_alpha:.4f}", flush=True)
        return make_pipeline(scaler, clf)

    if name == "LightGBM":
        n       = len(X_train)
        val_idx = int(0.8 * n)
        rng     = np.random.default_rng(seed)
        perm    = rng.permutation(n)
        fit_i, val_i = perm[:val_idx], perm[val_idx:]

        clf = LGBMClassifier(**LGB_PARAMS)
        clf.fit(
            X_train[fit_i], y_train[fit_i],
            sample_weight=sample_weight[fit_i],
            eval_set=[(X_train[val_i], y_train[val_i])],
            callbacks=[
                lgb_early_stopping(stopping_rounds=20, verbose=False),
                lgb_log_evaluation(period=-1),
            ],
        )
        best_iter  = clf.best_iteration_
        best_score = list(clf.best_score_["valid_0"].values())[0]
        metric     = list(clf.best_score_["valid_0"].keys())[0]
        print(f"      [LightGBM]     best_iter={best_iter}  "
              f"val {metric}={best_score:.4f}", flush=True)
        return clf

    if name == "MLP":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        gs = GridSearchCV(
            MLPClassifier(max_iter=300, random_state=seed),
            {
                "hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "alpha":              [0.0001, 0.001, 0.01],
            },
            cv=_cv3(), scoring=inner_scorer, n_jobs=1,
        )
        gs.fit(Xs, y_train, groups=task_ids, sample_weight=sample_weight)
        print(f"      [MLP]          best {gs.best_params_}  "
              f"inner-val score={gs.best_score_:.4f}", flush=True)
        return make_pipeline(scaler, gs.best_estimator_)

    if name == "Logistic_ElasticNet":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        param_grid = {
            "C":        [0.01, 0.1, 1.0, 10.0],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
        base = LogisticRegression(
            penalty="elasticnet", solver="saga",
            max_iter=2000, random_state=42,
        )
        gs = GridSearchCV(base, param_grid, cv=_cv3(),
                          scoring=inner_scorer, n_jobs=1)
        gs.fit(Xs, y_train, groups=task_ids, sample_weight=sample_weight)
        print(f"      [Logistic_ElasticNet] best C={gs.best_params_['C']}  "
              f"l1_ratio={gs.best_params_['l1_ratio']}  "
              f"inner-val score={gs.best_score_:.4f}", flush=True)
        clf = LogisticElasticNetScorer(inner_cv=3)
        clf._scaler = scaler
        clf._best   = gs.best_estimator_
        return clf

    if name == "ElasticNet":
        clf = models[name]
        clf.fit(X_train, y_train, sample_weight=sample_weight, groups=task_ids)
        return clf

    clf = models[name]
    try:
        clf.fit(X_train, y_train, sample_weight=sample_weight)
    except (TypeError, ValueError):
        step_name = clf.steps[-1][0]
        clf.fit(X_train, y_train,
                **{f"{step_name}__sample_weight": sample_weight})
    return clf




# ── 5-fold CV ─────────────────────────────────────────────────────────────────

def fit_model_fast(name: str, X_train: np.ndarray, y_train: np.ndarray,
                   sample_weight: np.ndarray, seed: int = 42):
    """
    Fixed-hyperparameter fit — used for holdout calibration (80/20 split).
    The final prediction model still uses fit_model() with full GridSearchCV.
    """
    if name in ("RF", "Calibrated_RF"):
        clf = RandomForestClassifier(**RF_PARAMS, random_state=seed)
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        return clf
    if name == "Logistic_L2":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
        clf.fit(Xs, y_train, sample_weight=sample_weight)
        return make_pipeline(scaler, clf)
    if name == "HTC_L2":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(Xs, y_train, sample_weight=sample_weight)
        return make_pipeline(scaler, clf)
    if name == "HTC_L1":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        clf = LogisticRegression(penalty="l1", C=1.0, solver="liblinear", max_iter=1000, random_state=42)
        clf.fit(Xs, y_train, sample_weight=sample_weight)
        return make_pipeline(scaler, clf)
    if name == "ElasticNet":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        base = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
        base.fit(Xs, y_train, sample_weight=sample_weight)
        scorer = ElasticNetScorer.__new__(ElasticNetScorer)
        scorer._scaler = scaler
        scorer._best   = base
        return scorer
    if name == "LightGBM":
        clf = LGBMClassifier(**LGB_PARAMS)
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        return clf
    if name == "MLP":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        clf = MLPClassifier(hidden_layer_sizes=(64,), alpha=0.001,
                            max_iter=200, random_state=seed)
        clf.fit(Xs, y_train, sample_weight=sample_weight)
        return make_pipeline(scaler, clf)
    # fallback
    return fit_model(name, X_train, y_train, sample_weight, seed=seed)


def run_cv(df: pd.DataFrame, feat_cols: list, n_splits: int = 5,
           inner_scorer="roc_auc", calib: str = "auto",
           ece_mode: str = "adaptive",
           weighting: str = "task_length",
           model_names: list | None = None,
           outer_shuffle: bool = True):
    """outer_shuffle default flipped to True (2026-08-01, superseding the
    2026-07-31 note below). History: the outer 5-fold split originally
    defaulted to NO shuffle (task order = first-appearance order, same
    convention as stacked_logistic_baseline.py's M14); a 2026-07-31
    one-off experiment (shuffle=True) was run but not adopted, since it
    wasn't yet clear the comparison was even apples-to-apples -- M11/HTC's
    own production protocol (htc_step_baseline.py) has always used
    shuffle=True, so the un-shuffled default was comparing Ours against
    baselines under two DIFFERENT outer-CV protocols, not one.
    2026-08-01: re-examined empirically (code/generalization/
    prefix_time_analysis/{htc_shuffle_ablation,outer_shuffle_ablation}/) --
    (a) un-shuffled M11 scores measurably worse than shuffled M11 across 5
    reshuffle seeds, plausibly because un-shuffled CV hands whole
    contiguous task blocks to single folds (WebArena's task suite is laid
    out in per-website blocks), and (b) shuffling Ours' own outer CV does
    NOT consistently help or hurt Ours relative to HTC (moves by ~1pt,
    sign flips by benchmark) -- so this is not a case of picking whichever
    setting favors either method, it's adopting the more defensible,
    already-precedented protocol (matching M11/M13's own convention) for
    BOTH sides of every comparison. Every "Ours(L2)" number reported
    anywhere in this release after 2026-08-01 uses shuffle=True,
    random_state=42; pass outer_shuffle=False (or CLI --no_outer_shuffle)
    to reproduce the pre-2026-08-01 numbers."""
    current_label_col = label_col(df)
    task_ids      = df["task_id"].unique()
    task_outcomes = df.groupby("task_id")["outcome"].first().reindex(task_ids).values
    skf           = StratifiedKFold(n_splits=n_splits, shuffle=outer_shuffle,
                                    random_state=42 if outer_shuffle else None)

    if model_names is None:
        # Restricted to Logistic_L2 only (2026-07-30: RF/ElasticNet/
        # LightGBM/MLP/HTC_*/Logistic_ElasticNet/LSTM/Transformer all take real
        # compute time across 10 combos x 5 folds and aren't "Ours" -- pass
        # model_names explicitly to run_cv() if you need the other baselines back.
        model_names = ["Logistic_L2"]
    results       = {}
    train_results = {}
    pred_rows     = []

    for fold, (tr_task_idx, te_task_idx) in enumerate(
            skf.split(task_ids, task_outcomes), 1):
        tr_tasks = set(task_ids[tr_task_idx])
        te_tasks = set(task_ids[te_task_idx])

        train_df_fold = df[df["task_id"].isin(tr_tasks)].copy()
        test_df       = df[df["task_id"].isin(te_tasks)].copy()

        avail   = [c for c in feat_cols if c in df.columns]
        medians = safe_medians(train_df_fold, avail)

        X_train          = train_df_fold[avail].fillna(medians).fillna(0.0).values
        y_train          = train_df_fold[current_label_col].values
        train_task_ids_arr = train_df_fold["task_id"].values
        X_test           = test_df[avail].fillna(medians).fillna(0.0).values
        y_test           = test_df[current_label_col].values

        if weighting == "class":
            sample_weight = task_class_weights(train_df_fold)
        elif weighting == "task_length":
            # Each task contributes equally regardless of trajectory length:
            #   w_step = 1 / n_steps_in_task  →  sum over task = 1.0
            step_counts   = train_df_fold.groupby("task_id")["step_k"].transform("count")
            sample_weight = (1.0 / step_counts).values
        else:  # "none"
            sample_weight = np.ones(len(X_train))

        if len(np.unique(y_train)) < 2:
            print(f"    [WARN] fold {fold}: only one class in train — skipping")
            continue

        print(f"  fold {fold}: train={len(X_train)} steps  "
              f"({int(y_train.sum())} label=1 / {int((y_train == 0).sum())} label=0)  "
              f"test={len(X_test)} steps  ({int(y_test.sum())} label=1)", flush=True)

        fold_preds = {
            "task_id": test_df["task_id"].values,
            "step_k":  test_df["step_k"].values,
            "outcome": test_df["outcome"].values,
            "label":   y_test,
            "fold":    fold,
        }

        # 80/20 task-holdout masks — shared across models for non-OOF calibration
        rng_ho = np.random.default_rng(fold * 1000 + 999)
        _ho_tasks = train_df_fold["task_id"].unique()
        _ho_shuf  = rng_ho.permutation(_ho_tasks)
        _ho_cut   = int(0.8 * len(_ho_shuf))
        ho_fit_mask = np.isin(train_task_ids_arr, _ho_shuf[:_ho_cut])
        ho_cal_mask = np.isin(train_task_ids_arr, _ho_shuf[_ho_cut:])

        for name in model_names:
            try:
                calib_variants_list = calibration_variants(name, calib)
                method_outputs = {}

                # ── Logistic models: no post-hoc calibration ──────────────────
                if name in NO_POSTHOC_CALIB_MODELS:
                    clf = fit_model(name, X_train, y_train, sample_weight,
                                    seed=fold, inner_scorer=inner_scorer,
                                    task_ids=train_task_ids_arr)
                    logits_test  = get_logits(clf, X_test,  name)
                    train_prob   = sigmoid(get_logits(clf, X_train, name))
                    method_outputs[name] = sigmoid(logits_test)
                    print(f"      [{name:<22}] no calibration", flush=True)

                # ── RF: inner-OOF calibration on full training data ────────────
                elif name in RF_MODELS:
                    # Full model on ALL training data
                    clf = fit_model(name, X_train, y_train, sample_weight,
                                    seed=fold, inner_scorer=inner_scorer,
                                    task_ids=train_task_ids_arr)
                    raw_prob_test = clf.predict_proba(X_test)[:, 1]
                    train_prob    = clf.predict_proba(X_train)[:, 1]
                    method_outputs[f"{name}[raw]"] = raw_prob_test
                    print(f"      [{name}[raw]            ] raw score", flush=True)

                    # Holdout temperature calibration (80/20 task split)
                    if ho_cal_mask.sum() > 0:
                        clf_ho         = fit_model_fast(name, X_train[ho_fit_mask], y_train[ho_fit_mask],
                                                        sample_weight[ho_fit_mask], seed=fold)
                        logits_ho_test = get_logits(clf_ho, X_test, name)
                        log_ho_cal     = get_logits(clf_ho, X_train[ho_cal_mask], name)
                        y_ho  = y_train[ho_cal_mask]
                        sw_ho = sample_weight[ho_cal_mask]
                        vho   = ~np.isnan(log_ho_cal)
                        if vho.sum() > 0 and len(np.unique(y_ho[vho])) >= 2:
                            y_prob_ho, T_ho = temperature_scale(
                                log_ho_cal[vho], y_ho[vho], logits_ho_test,
                                sample_weight=sw_ho[vho])
                            print(f"      [{name}[temperature_ho]] T*={T_ho:.4f}", flush=True)
                            method_outputs[f"{name}[temperature_ho]"] = y_prob_ho
                        else:
                            method_outputs[f"{name}[temperature_ho]"] = sigmoid(logits_ho_test)
                            print(f"      [{name}[temperature_ho]] fallback", flush=True)

                # ── sklearn models: holdout temperature calibration ───────────
                elif name in SKLEARN_OOF_MODELS:
                    clf = fit_model(name, X_train, y_train, sample_weight,
                                    seed=fold, inner_scorer=inner_scorer,
                                    task_ids=train_task_ids_arr)
                    logits_test   = get_logits(clf, X_test, name)
                    raw_prob_test = sigmoid(logits_test)
                    train_prob    = sigmoid(get_logits(clf, X_train, name))
                    method_outputs[f"{name}[raw]"] = raw_prob_test
                    print(f"      [{name}[raw]            ] raw score", flush=True)

                    # Holdout temperature calibration (80/20 task split)
                    if ho_cal_mask.sum() > 0:
                        clf_ho         = fit_model_fast(name, X_train[ho_fit_mask], y_train[ho_fit_mask],
                                                        sample_weight[ho_fit_mask], seed=fold)
                        logits_ho_test = get_logits(clf_ho, X_test, name)
                        log_ho_cal     = get_logits(clf_ho, X_train[ho_cal_mask], name)
                        y_ho  = y_train[ho_cal_mask]
                        sw_ho = sample_weight[ho_cal_mask]
                        vho   = ~np.isnan(log_ho_cal)
                        if vho.sum() > 0 and len(np.unique(y_ho[vho])) >= 2:
                            y_prob_ho, T_ho = temperature_scale(
                                log_ho_cal[vho], y_ho[vho], logits_ho_test,
                                sample_weight=sw_ho[vho])
                            print(f"      [{name}[temperature_ho]] T*={T_ho:.4f}", flush=True)
                            method_outputs[f"{name}[temperature_ho]"] = y_prob_ho
                        else:
                            method_outputs[f"{name}[temperature_ho]"] = sigmoid(logits_ho_test)
                            print(f"      [{name}[temperature_ho]] fallback", flush=True)

                # ── Sequence models: 80/20 hold-out calibration ───────────────
                elif name in SEQ_MODELS:
                    all_tr_tasks = train_df_fold["task_id"].unique()
                    rng_seq = np.random.default_rng(fold * 1000 + 7)
                    shuffled = rng_seq.permutation(all_tr_tasks)
                    cut = int(0.8 * len(shuffled))
                    fit_tasks = set(shuffled[:cut])
                    cal_tasks = set(shuffled[cut:])

                    fit_df  = train_df_fold[train_df_fold["task_id"].isin(fit_tasks)]
                    cal_df  = train_df_fold[train_df_fold["task_id"].isin(cal_tasks)].reset_index(drop=True)
                    test_df_reset = test_df.reset_index(drop=True)
                    train_df_reset = train_df_fold.reset_index(drop=True)

                    SeqClass = LSTMScorer if name == "LSTM" else TransformerScorer

                    # Calibration model on 80%
                    seq_cal = SeqClass()
                    seq_cal.fit(fit_df, avail, medians, seed=fold)
                    logits_cal_seq = seq_cal.predict_logits(cal_df, avail, medians)
                    y_cal_seq = cal_df[current_label_col].values
                    valid_cal = ~np.isnan(logits_cal_seq)

                    # Full model on 100%
                    seq_full = SeqClass()
                    seq_full.fit(train_df_fold, avail, medians, seed=fold)
                    logits_test_seq  = seq_full.predict_logits(test_df_reset,  avail, medians)
                    logits_train_seq = seq_full.predict_logits(train_df_reset, avail, medians)

                    raw_prob_test = sigmoid(logits_test_seq)
                    train_prob    = sigmoid(logits_train_seq)

                    # Compute task-length weights for the cal set (same as training)
                    cal_step_counts = (cal_df.groupby("task_id")["step_k"]
                                       .transform("count").values.astype(float))
                    sw_seq_cal = 1.0 / cal_step_counts

                    method_outputs[f"{name}[raw]"] = raw_prob_test
                    if valid_cal.sum() > 0 and len(np.unique(y_cal_seq[valid_cal])) >= 2:
                        y_prob_tmp, T_opt = temperature_scale(
                            logits_cal_seq[valid_cal], y_cal_seq[valid_cal], logits_test_seq,
                            sample_weight=sw_seq_cal[valid_cal])
                        print(f"      [{name}[temperature_ho]] T*={T_opt:.4f}", flush=True)
                        method_outputs[f"{name}[temperature_ho]"] = y_prob_tmp
                    else:
                        print(f"      [{name}[temperature_ho]] fallback", flush=True)
                        method_outputs[f"{name}[temperature_ho]"] = raw_prob_test

                else:
                    continue  # safety net for unknown model names

                # ── Record metrics ─────────────────────────────────────────────
                for result_name, y_prob in method_outputs.items():
                    results.setdefault(result_name, [])
                    train_results.setdefault(result_name, [])

                    metrics = evaluate(y_test, y_prob, ece_mode=ece_mode)
                    metrics["fold"] = fold
                    results[result_name].append(metrics)

                    train_m = evaluate(y_train, train_prob, ece_mode=ece_mode)
                    train_m["fold"] = fold
                    train_results[result_name].append(train_m)

                    fold_preds[f"{result_name}_prob"] = y_prob
                    ece_str = ""
                    if ece_mode in ("adaptive", "both"):
                        ece_str += f"  ECE_adap={metrics.get('ece_adaptive', '?')}"
                    if ece_mode in ("equal_width", "both"):
                        ece_str += f"  ECE_ew={metrics.get('ece_ew', '?')}"
                    print(f"    {result_name:<22} AUROC={metrics.get('auroc', '?')}  "
                          f"AUPRC={metrics.get('auprc', '?')}  "
                          f"BSS={metrics.get('bss', '?')}"
                          f"{ece_str}", flush=True)

            except Exception as e:
                print(f"    [ERROR] {name}: {e}", flush=True)
                import traceback; traceback.print_exc()
                if name in NO_POSTHOC_CALIB_MODELS:
                    err_names = [name]
                else:
                    err_names = [f"{name}[raw]", f"{name}[temperature_ho]"]
                for result_name in err_names:
                    results.setdefault(result_name, []).append({"fold": fold, "error": str(e)})
                    train_results.setdefault(result_name, []).append({"fold": fold, "error": str(e)})
                    fold_preds[f"{result_name}_prob"] = np.full(len(y_test), np.nan)

        pred_rows.append(pd.DataFrame(fold_preds))

    pred_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    return results, train_results, pred_df


# ── Summary ───────────────────────────────────────────────────────────────────

def summarise(cv_results: dict) -> pd.DataFrame:
    rows = []
    for model, folds in cv_results.items():
        valid = [f for f in folds if "auroc" in f]
        if not valid:
            continue
        rc_metrics = (["risk@50", "risk@60", "risk@70", "risk@80", "risk@90"] +
                      ["cov@risk5", "cov@risk10", "cov@risk20", "cov@risk30"])
        for metric in (["auroc", "auprc", "aurc", "e_aurc", "brier", "bss",
                        "ece", "ece_adaptive", "ece_ew"] + rc_metrics):
            vals = [f[metric] for f in valid if metric in f]
            if not vals:
                continue
            rows.append({
                "model":   model,
                "metric":  metric,
                "mean":    round(float(np.mean(vals)), 4),
                "std":     round(float(np.std(vals)),  4),
                "n_folds": len(vals),
            })
    return pd.DataFrame(rows)


def print_session_grid(all_summaries: list[dict], metric: str = "auroc"):
    sessions = ["macro_only", "uq_only", "all"]
    pref_order = ["Logistic_L2", "RF", "Calibrated_RF",
                  "ElasticNet", "LightGBM", "Logistic_ElasticNet", "MLP", "LSTM",
                  "Transformer"]
    seen_models = sorted({row["model"] for row in all_summaries},
                         key=lambda m: (
                             pref_order.index(m.split("[", 1)[0]) if m.split("[", 1)[0] in pref_order else 999,
                             m.split("[", 1)[0],
                             m,
                         ))

    lookup = {}
    for row in all_summaries:
        if row["metric"] == metric:
            lookup[(row["model"], row["feature_set"])] = (row["mean"], row["std"])

    col_w = 18
    hdr   = f"{'Model':<22}" + "".join(f"{s:^{col_w}}" for s in sessions)
    print(f"\n  {metric.upper()} — mean ± std across 5 folds")
    print("  " + "─" * len(hdr))
    print("  " + hdr)
    print("  " + "─" * len(hdr))
    for m in seen_models:
        row_str = f"{m:<22}"
        for s in sessions:
            val = lookup.get((m, s))
            if val:
                row_str += f"{val[0]:.4f}±{val[1]:.4f}".center(col_w)
            else:
                row_str += "     —     ".center(col_w)
        print("  " + row_str)
    print("  " + "─" * len(hdr))


# ── Feature importance (full-data fit) ───────────────────────────────────────

def get_importances(df: pd.DataFrame, feat_cols: list) -> dict:
    """Logistic_L2 coefficients only (2026-07-30: RF/LightGBM/ElasticNet
    importances dropped by design -- not "Ours", not worth the compute)."""
    avail    = [c for c in feat_cols if c in df.columns]
    medians  = safe_medians(df, avail)
    X        = df[avail].fillna(medians).fillna(0.0).values
    y        = df[label_col(df)].values
    weights  = task_class_weights(df)
    task_ids = df["task_id"].values

    out = {}
    lr_pipe = fit_model("Logistic_L2", X, y, weights, task_ids=task_ids)
    out["Logistic_L2_coef"] = dict(zip(avail, lr_pipe[-1].coef_[0].round(6).tolist()))
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",     required=True,
                    help="e.g. mind2web/claude or webarena/qwen30b")
    ap.add_argument("--sampling",   default="full",
                    choices=["uniform", "balanced", "fixed", "full"])
    ap.add_argument("--max_prefix", type=int, default=5)
    ap.add_argument("--uq_dir",       default="uq_analysis_change_qwen")
    ap.add_argument("--csv_path", default=None,
                    help="Explicit feature_matrix.csv path. Use this for embeded_hlvm matrices.")
    ap.add_argument("--annot_dir",  default=None,
                    help="Directory containing source annotation files. "
                         "Defaults to <results_total>/annotate_vis/result/")
    ap.add_argument("--annot_path", default=None,
                    help="Explicit annotation JSON path.")
    ap.add_argument("--output_root", default=None,
                    help="Organized output root. Default: <results_total>/uq_results/")
    ap.add_argument("--inner_metric", default="auroc", choices=["auroc", "aurc", "brier", "log_loss"],
                    help="Metric used for inner-CV hyperparameter selection")
    ap.add_argument("--calib", default="temperature",
                    choices=["auto", "temperature", "platt", "isotonic", "both"],
                    help="Post-hoc calibration method.  "
                         "auto (default) = per-model best (see MODEL_CALIB dict); "
                         "temperature=1-param T on logits; "
                         "platt=2-param sigmoid(a·logit+b); "
                         "isotonic=non-parametric monotone regression; "
                         "both=run temperature and platt in the same execution.")
    ap.add_argument("--ece_mode", default="both",
                    choices=["adaptive", "equal_width", "both"],
                    help="ECE variant to compute.  "
                         "adaptive (default) = equal-mass binning (5 bins); "
                         "equal_width = standard equal-width binning (10 bins); "
                         "both = compute and report both variants.")
    ap.add_argument("--weighting", default="task_length",
                    choices=["none", "class", "task_length"],
                    help="Sample weighting strategy: "
                         "none = uniform weights; "
                         "class = inverse-frequency class weights (balances failure/success tasks); "
                         "task_length (default) = 1/n_steps per step so every task contributes equally.")
    ap.add_argument("--no_outer_shuffle", action="store_true",
                    help="Use the OLD un-shuffled outer 5-fold split (task order = "
                         "first-appearance order) instead of the shuffled default "
                         "(random_state=42, matching M11/M13's own protocol). Un-shuffled "
                         "was canonical before 2026-08-01; kept available for reproducing "
                         "old numbers, not recommended for new results.")
    ap.add_argument("--label_mode", default="keystep", choices=["keystep", "trajectory"],
                    help="keystep (default): the paper's key-step-aware supervision -- "
                         "pre-keyStep prefixes of a failed trajectory are relabeled 0, "
                         "keyStep-and-after are 1 (apply_keystep_labels()). "
                         "trajectory: the naive ablation this is compared against -- skip "
                         "keyStep relabeling entirely and propagate the raw trajectory-level "
                         "outcome to every prefix (label = outcome for every step, including "
                         "pre-keyStep ones on failed tasks). Added 2026-08-02 for the "
                         "key-step-supervision-ablation experiment "
                         "(code/keystep_ablation/) -- see that folder's README for the "
                         "evaluation protocol this feeds.")
    args = ap.parse_args()

    set_global_seed(42)
    results_dir = Path(__file__).parent.parent.parent
    base_dir = results_dir / args.source
    if args.csv_path is not None:
        csv_path = Path(args.csv_path)
        if not csv_path.is_absolute():
            csv_path = results_dir / csv_path
    else:
        csv_path = base_dir / args.uq_dir / "feature_matrix.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"feature_matrix.csv not found: {csv_path}\n"
                                f"Run build_feature_matrix.py first.")

    inner_scorer = make_inner_scorer(args.inner_metric)
    if args.output_root is not None:
        output_root = Path(args.output_root)
        if not output_root.is_absolute():
            output_root = results_dir / output_root
    else:
        output_root = results_dir / "uq_results"

    out_dir = resolve_output_dir(
        results_root=results_dir,
        source=args.source,
        csv_path=csv_path,
        calib=args.calib,
        sampling=args.sampling,
        max_prefix=args.max_prefix,
        inner_metric=args.inner_metric,
        ece_mode=args.ece_mode,
    )
    if args.output_root is not None:
        out_dir = output_root / out_dir.relative_to(results_dir / "uq_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "label" not in df.columns and args.label_mode == "trajectory":
        df["label"] = df["outcome"]
        print("Annotation : skipped (--label_mode trajectory -- raw outcome propagated to every prefix)")
    elif "label" not in df.columns:
        if args.annot_path is not None:
            annot_path = Path(args.annot_path)
            if not annot_path.is_absolute():
                annot_path = results_dir / annot_path
        else:
            if args.annot_dir is not None:
                annot_dir = Path(args.annot_dir)
                if not annot_dir.is_absolute():
                    annot_dir = results_dir / annot_dir
            else:
                annot_dir = None
            annot_path = resolve_annotation_path(
                results_dir=results_dir,
                source=args.source,
                annot_dir=annot_dir,
            )

        if not annot_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {annot_path}\n"
                f"Provide --annot_path or use an already aligned matrix."
            )
        annotation = load_annotation(annot_path)
        print(f"Annotation : {annot_path}  ({len(annotation)} entries)")
        df = apply_keystep_labels(df, annotation)
    else:
        print("Annotation : skipped (label column already present)")

    current_label_col = label_col(df)
    n_label1 = int(df[current_label_col].sum())
    n_label0 = int((df[current_label_col] == 0).sum())
    label_check = df[["task_id", "step_k", "outcome", current_label_col]].copy()
    label_check = label_check.rename(columns={current_label_col: "label"})
    changed_mask = label_check["outcome"] != label_check["label"]
    label_check[changed_mask].to_csv(out_dir / "relabeled_steps.csv", index=False)

    _lstm_dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device   : LSTM={_lstm_dev} (MPS skipped — unstable LSTM kernel); "
          f"sklearn/LightGBM=cpu")
    if args.calib == "auto":
        calib_summary = "auto (" + ", ".join(f"{m}={c}" for m, c in MODEL_CALIB.items()) + ")"
    else:
        calib_summary = args.calib
    print(f"Source   : {args.source}  [calibration={calib_summary}]")
    print(f"Matrix   : {csv_path}")
    print(f"Output   : {out_dir}")
    print(f"Training : full trajectories  weighting={args.weighting}")
    print(f"Tasks    : {df['task_id'].nunique()}  "
          f"Steps: {len(df)}  "
          f"Orig failure steps: {int(df['outcome'].sum())}  "
          f"Label=1 steps: {n_label1}  Label=0 steps: {n_label0}", flush=True)

    all_cv             = {}
    all_train_cv       = {}
    all_val_cv         = {}
    all_summ_rows      = []
    all_summ_dfs       = []
    all_train_summ_dfs = []

    for fs_name, feat_cols in FEATURE_SETS.items():
        avail = [c for c in feat_cols if c in df.columns]
        print(f"\n{'='*60}")
        print(f"  Session: {fs_name}  ({len(avail)} features)")
        print(f"{'='*60}", flush=True)

        cv_res, train_cv_res, pred_df = run_cv(
            df, avail,
            inner_scorer=inner_scorer, calib=args.calib,
            ece_mode=args.ece_mode,
            weighting=args.weighting,
            outer_shuffle=not args.no_outer_shuffle)

        summ_df = summarise(cv_res)
        summ_df.insert(0, "feature_set", fs_name)
        all_summ_dfs.append(summ_df)
        all_cv[fs_name] = cv_res

        train_summ_df = summarise(train_cv_res)
        train_summ_df.insert(0, "feature_set", fs_name)
        all_train_summ_dfs.append(train_summ_df)
        all_train_cv[fs_name] = train_cv_res

        if not pred_df.empty:
            pred_df.to_csv(out_dir / f"cv_predictions_{fs_name}.csv", index=False)

            # ── RC curves aggregated across folds ─────────────────────────────
            rc_rows = []
            for mname in [c.replace("_prob", "") for c in pred_df.columns if c.endswith("_prob")]:
                yp_all = pred_df[f"{mname}_prob"].values
                yt_all = pred_df["label"].values if "label" in pred_df.columns else pred_df["outcome"].values
                valid  = ~np.isnan(yp_all)
                if valid.sum() < 2 or len(np.unique(yt_all[valid])) < 2:
                    continue
                curve = rc_curve(yt_all[valid], yp_all[valid])
                curve.insert(0, "model", mname)
                rc_rows.append(curve)
            if rc_rows:
                pd.concat(rc_rows, ignore_index=True).to_csv(
                    out_dir / f"rc_curves_{fs_name}.csv", index=False)

        for _, row in summ_df.iterrows():
            all_summ_rows.append(row.to_dict())

    print(f"\n{'='*60}")
    print(f"  RESULTS GRID  —  {args.source}  [calib={args.calib}, labels={current_label_col}]")
    print(f"  full trajectories  weighting={args.weighting}")
    print(f"{'='*60}")
    ece_metrics = {"adaptive": ["ece_adaptive"],
                   "equal_width": ["ece_ew"],
                   "both": ["ece_adaptive", "ece_ew"]}
    for metric in ["auroc", "auprc", "aurc", "brier", "bss"] + ece_metrics[args.ece_mode]:
        print_session_grid(all_summ_rows, metric=metric)

    (out_dir / "cv_results.json").write_text(
        json.dumps({"test": all_cv, "train": all_train_cv}, indent=2))

    summary_df = pd.concat(all_summ_dfs, ignore_index=True)
    summary_df.to_csv(out_dir / "summary_table.csv", index=False)

    train_summary_df = pd.concat(all_train_summ_dfs, ignore_index=True)
    train_summary_df.to_csv(out_dir / "train_summary_table.csv", index=False)

    print(f"\nComputing feature importances on full data...", flush=True)
    all_feats = [c for c in FEATURE_SETS["all"] if c in df.columns]
    imp = get_importances(df, all_feats)
    (out_dir / "feature_importance.json").write_text(json.dumps(imp, indent=2))

    print(f"\nDone. Results in {out_dir}")
    print(f"  cv_results.json              (test / train per fold)")
    print(f"  summary_table.csv            (test metrics: auroc/auprc/aurc/brier/bss/ece/risk@k/cov@riskN)")
    print(f"  train_summary_table.csv      (train metrics, mean±std)")
    print(f"  cv_predictions_<fs>.csv      (per-step predicted probs)")
    print(f"  rc_curves_<fs>.csv           (full RC curve per model, aggregated across folds)")
    print(f"  feature_importance.json")


if __name__ == "__main__":
    main()
