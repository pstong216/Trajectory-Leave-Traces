#!/usr/bin/env python3
"""
update_baselines_with_features.py — Extend baseline per_step_scores.csv with
feature-based UQ methods (M7, HTC M11-M12) drawn from the HLVM feature matrix.

M8/M9/M10 (mean-feature scores) REMOVED 2026-07-30 (design decision): they are
computed with zero train/test split -- StandardScaler and the min-max rescale
are both fit on the full dataset and then evaluated on that same full
dataset, so their AUROC is not a genuine held-out generalization estimate the
way every other baseline method's is. Not a legitimate "baseline" method,
should never have been reported alongside the CV'd ones. See
compute_mean_feature_baseline.py if you need the underlying score for some
other diagnostic purpose -- it is no longer wired into this merge.

New methods added
-----------------
M7  conf_cumul_mean       Verbalized confidence averaged from step 0 → k
                          (cumulative mean of M1; higher = more confident)

M7 comes directly from the existing per_step_scores.csv (M1 must already be
populated there by the Stage-1 script).

Usage
-----
  # Single model:
  python update_baselines_with_features.py \\
      --baseline_dir  baseline/webarena/claude/results \\
      --source        webarena/claude \\
      --uq_dir        uq_analysis_change_qwen

  # All models at once (edit MODEL_CONFIGS below and run with --all):
  python update_baselines_with_features.py --all
"""

import argparse
import math
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit as sigmoid
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler

# htc_step_baseline.py sits next to this file in the working tree (baseline/),
# but one directory over in code/{baseline_calculation,htc}/ --
# check both so this script runs unmodified from either layout.
_here = Path(__file__).resolve().parent
for _candidate in (_here, _here.parent / "htc"):
    if (_candidate / "htc_step_baseline.py").is_file() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from htc_step_baseline import compute_htc_step_scores

warnings.filterwarnings("ignore")

# ── Model configuration ────────────────────────────────────────────────────────
# Each entry: (baseline_dir_relative, source_relative, uq_dir, conf_dir_relative_or_None)
# conf_dir: optional path to a verbalized uq_samples dir whose step_{k}_samples.json
#           files are used to load/overwrite M1 (verbalized confidence at step k).
#           If None, M1 is taken from the existing per_step_scores.csv.
# Paths are relative to results_total/

RESULTS_ROOT = Path(__file__).resolve().parent.parent   # results_total/

MODEL_CONFIGS = [
    ("baseline/webarena/claude/results",        "webarena/claude",    "uq_analysis_change_qwen",
     "webarena/results_claude_verbalized"),
    ("baseline/webarena/gpt_5_2/results",       "webarena/gpt_5_2",   "uq_analysis_change_qwen",
     "webarena/results_gpt_5_2_verbalized/uq_samples"),
    ("baseline/webarena/gptnano/results",       "webarena/gptnano",   "uq_analysis_change_qwen",
     "webarena/results_gptnano_verbalized/uq_samples"),
    ("baseline/webarena/kimi/results",          "webarena/kimi",      "uq_analysis_change_qwen", None),
    ("baseline/webarena/qwen30b/results",       "webarena/qwen30b",   "uq_analysis_change",
     "webarena/results_qwen30b_verbalized/uq_samples"),
    ("baseline/mind2web/qwen30b/results", "mind2web/qwen30b", "uq_analysis_change", None),
    ("baseline/mind2web/kimi/results",          "mind2web/kimi",      "uq_analysis_change_qwen", None),
    ("baseline/mind2web/gpt_5_2/results",       "mind2web/gpt_5_2",   "uq_analysis_change_qwen", None),
    ("baseline/mind2web/gptnano/results",       "mind2web/results_gptnano_verbalized",
     "uq_analysis_change_qwen", None),
    ("baseline/mind2web/claude/results",        "mind2web/claude",
     "uq_analysis_change_qwen", "mind2web/results_claude_verbalized"),
]

# ── Metric helpers ─────────────────────────────────────────────────────────────

N_BINS  = 10
N_FOLDS = 5

NEW_SCORE_COLS = [
    "m7_conf_cumul_mean",
]

HTC_SCORE_COLS = [
    "m11_htc_full_step",
    "m12_htc_reduced_step",
]

# Higher score = more confident (True) or more uncertain/risky (False → flip)
HIGHER_IS_CONFIDENT = {
    "m7_conf_cumul_mean":  True,
    "m11_htc_full_step":   True,
    "m12_htc_reduced_step": True,
}


def _is_prob_native_method(col: str) -> bool:
    return col in HTC_SCORE_COLS


def _clip_probs(probs) -> np.ndarray:
    return np.clip(np.asarray(probs, dtype=float), 1e-7, 1 - 1e-7)


def _prob_to_logit(probs) -> np.ndarray:
    p = _clip_probs(probs)
    return np.log(p / (1.0 - p))


def ece_random_ref(success_rate: float) -> float:
    bin_mids = np.linspace(1 / (2 * N_BINS), 1 - 1 / (2 * N_BINS), N_BINS)
    return float(np.mean(np.abs(bin_mids - success_rate)))


def brier_ref_val(success_rate: float) -> float:
    return float(success_rate * (1.0 - success_rate))


def bss(brier: float, ref: float) -> float:
    return float('nan') if ref == 0 else float((ref - brier) / ref)


def ece_equal_width(probs, labels) -> float:
    if len(probs) == 0:
        return float('nan')
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


def aurc(scores, labels):
    n = len(scores)
    if n == 0:
        return float('nan'), float('nan')
    order  = np.argsort(-scores)
    errors = (1 - labels[order]).astype(float)
    risks  = np.cumsum(errors) / np.arange(1, n + 1)
    return float(np.mean(risks)), float(np.mean(errors))


def aurc_oracle(labels):
    """AURC of a perfect oracle (all successes ranked before failures).
    labels: 1=success, 0=failure. Same formula as stacked_logistic_baseline.py's
    aurc_oracle() / codes/train_risk_model/recover_aurc.py -- kept identical so
    e_aurc (= aurc - aurc_oracle, the standard "Excess-AURC" from the selective
    prediction literature, always >= 0, lower=better/closer-to-oracle) means the
    same thing across every method reported in this file, M13/M14, and Ours."""
    n = len(labels)
    n_f = int((1 - labels).sum())
    n_s = n - n_f
    if n_f == 0:
        return 0.0
    if n_s == 0:
        return 1.0
    i_vals = np.arange(n_s + 1, n + 1, dtype=float)
    return float(np.sum((i_vals - n_s) / i_vals) / n)


def calibrate_platt(scores, labels, groups):
    probs = np.full_like(scores, np.nan, dtype=float)
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    X = scores.reshape(-1, 1)
    for tr, va in skf.split(X, labels, groups):
        if len(np.unique(labels[tr])) < 2:
            continue
        clf = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf.fit(X[tr], labels[tr])
        probs[va] = clf.predict_proba(X[va])[:, 1]
    return probs


def calibrate_isotonic(scores, labels, groups):
    probs = np.full_like(scores, np.nan, dtype=float)
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for tr, va in skf.split(scores.reshape(-1, 1), labels, groups):
        if len(np.unique(labels[tr])) < 2:
            continue
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(scores[tr], labels[tr])
        probs[va] = ir.predict(scores[va])
    return np.clip(probs, 0.0, 1.0)


def calibrate_temperature(scores, labels, groups):
    def nll(T, z, y):
        p = np.clip(sigmoid(z / T), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    probs = np.full_like(scores, np.nan, dtype=float)
    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for tr, va in skf.split(scores.reshape(-1, 1), labels, groups):
        if len(np.unique(labels[tr])) < 2:
            continue
        res  = minimize_scalar(nll, args=(scores[tr], labels[tr].astype(float)),
                               bounds=(1e-3, 1e3), method='bounded')
        probs[va] = sigmoid(scores[va] / res.x)
    return np.clip(probs, 0.0, 1.0)


def evaluate_method(col: str, df: pd.DataFrame, success_rate: float) -> dict:
    group_col = "task_index" if "task_index" in df.columns else "task_id"
    sub = df[[group_col, 'success', col]].dropna()
    nan = float('nan')
    if len(sub) < 2 * N_FOLDS or len(np.unique(sub['success'])) < 2:
        return dict(method=col, n=len(sub),
                    auroc=nan, aurc=nan, aurc_ref=nan, aurc_gain=nan,
                    aurc_oracle=nan, e_aurc=nan,
                    ece_raw=nan, ece_ref=nan,
                    ece_platt=nan, ece_gain_platt=nan,
                    ece_isotonic=nan, ece_gain_isotonic=nan,
                    ece_temp=nan, ece_gain_temp=nan,
                    brier_raw=nan, brier_ref=nan,
                    brier_platt=nan, bss_platt=nan,
                    brier_isotonic=nan, bss_isotonic=nan,
                    brier_temp=nan, bss_temp=nan)

    raw    = sub[col].values.astype(float)
    labels = sub['success'].values.astype(int)
    groups = sub[group_col].astype(str).values
    prob_native = _is_prob_native_method(col)
    conf   = _clip_probs(raw) if prob_native else (raw if HIGHER_IS_CONFIDENT[col] else -raw)

    auc_val  = roc_auc_score(labels, conf)
    aurc_val, aurc_rand = aurc(conf, labels)
    aurc_oracle_val = aurc_oracle(labels)

    raw_probs = conf if prob_native else MinMaxScaler().fit_transform(conf.reshape(-1, 1)).ravel()
    ece_raw  = ece_equal_width(raw_probs, labels)
    ece_ref  = ece_random_ref(success_rate)
    b_ref    = brier_ref_val(success_rate)
    b_raw    = brier_score_loss(labels, _clip_probs(raw_probs))

    if prob_native:
        platt = np.full_like(conf, np.nan, dtype=float)
        iso   = np.full_like(conf, np.nan, dtype=float)
        temp  = np.full_like(conf, np.nan, dtype=float)
    else:
        platt = calibrate_platt(conf, labels, groups)
        iso   = calibrate_isotonic(conf, labels, groups)
        temp  = calibrate_temperature(conf, labels, groups)

    vp = ~np.isnan(platt); vi = ~np.isnan(iso); vt = ~np.isnan(temp)

    def _ece_or_nan(probs, mask):
        return round(ece_equal_width(probs[mask], labels[mask]), 4) if mask.sum() > 0 else nan

    def _ece_gain_or_nan(probs, mask):
        return round(ece_ref - ece_equal_width(probs[mask], labels[mask]), 4) if mask.sum() > 0 else nan

    def _brier_or_nan(probs, mask):
        return round(brier_score_loss(labels[mask], probs[mask]), 4) if mask.sum() > 0 else nan

    def _bss_or_nan(probs, mask):
        return round(bss(brier_score_loss(labels[mask], probs[mask]), b_ref), 4) if mask.sum() > 0 else nan

    return dict(
        method=col, n=len(sub),
        auroc=round(auc_val, 4),
        aurc=round(aurc_val, 4),
        aurc_ref=round(aurc_rand, 4),
        aurc_gain=round(aurc_rand - aurc_val, 4),
        aurc_oracle=round(aurc_oracle_val, 4),
        e_aurc=round(aurc_val - aurc_oracle_val, 4),
        ece_raw=round(ece_raw, 4),
        ece_ref=round(ece_ref, 4),
        ece_platt=_ece_or_nan(platt, vp),
        ece_gain_platt=_ece_gain_or_nan(platt, vp),
        ece_isotonic=_ece_or_nan(iso, vi),
        ece_gain_isotonic=_ece_gain_or_nan(iso, vi),
        ece_temp=_ece_or_nan(temp, vt),
        ece_gain_temp=_ece_gain_or_nan(temp, vt),
        brier_raw=round(b_raw, 4),
        brier_ref=round(b_ref, 4),
        brier_platt=_brier_or_nan(platt, vp),
        bss_platt=_bss_or_nan(platt, vp),
        brier_isotonic=_brier_or_nan(iso, vi),
        bss_isotonic=_bss_or_nan(iso, vi),
        brier_temp=_brier_or_nan(temp, vt),
        bss_temp=_bss_or_nan(temp, vt),
    )


# ── Feature loading & merge ────────────────────────────────────────────────────

def load_feature_matrix(feat_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(feat_csv)
    df = df.rename(columns={"step_k": "step"})
    df["task_id"] = df["task_id"].astype(str)
    return df[["task_id", "step", "conf_at_k", "conf_cumul_mean"]
              + [c for c in df.columns if c not in
                 {"task_id", "step", "conf_at_k", "conf_cumul_mean",
                  "source", "outcome"}]]


_CONF_RE = re.compile(r'Confidence:\s*([\d.]+)')

def load_m1_from_verbalized_samples(per_step: pd.DataFrame,
                                     conf_dir: Path) -> pd.DataFrame:
    """
    Load M1 (verbalized confidence at step k) from a verbalized samples directory.

    Supports two formats, auto-detected from conf_dir contents:

    Format A — uq_samples layout (gpt_5_2, gptnano, kimi, qwen30b):
      {conf_dir}/{task_id}/step_{k}_samples.json
      → data["samples"][0]["raw_prediction"]

    Format B — chat layout (claude):
      {conf_dir}/trajectory_1/chat/{task_id}/chat.json
      → data["steps"][k]["raw_prediction"]

    Only overwrites M1 where the loaded value is non-null.
    """
    import json as _json

    df = per_step.copy().reset_index(drop=True)
    if "m1_verbalized_conf" not in df.columns:
        df["m1_verbalized_conf"] = float("nan")

    # Detect format by inspecting conf_dir contents
    chat_dir = conf_dir / "trajectory_1" / "chat"
    # Probe first digit-named subdir to distinguish Format A vs C
    _probe = next((p for p in conf_dir.iterdir()
                   if p.is_dir() and p.name.isdigit()), None)
    _has_result_json   = _probe is not None and (_probe / "result.json").exists()
    _has_samples_files = _probe is not None and any(_probe.glob("step_*_samples.json"))

    if chat_dir.exists():
        fmt = "B"  # webarena claude: trajectory_1/chat/{task_id}/chat.json
    elif _has_result_json and not _has_samples_files:
        fmt = "C"  # mind2web: {task_index}/result.json → step_samples[k][0]
    else:
        fmt = "A"  # uq_samples: {task_id}/step_{k}_samples.json

    # Choose grouping column: mind2web uses task_index, webarena uses task_id
    id_col = "task_index" if (fmt == "C" and "task_index" in df.columns) else "task_id"

    if fmt == "B":
        # Format B: webarena claude chat layout
        for task_id, grp_idx in df.groupby("task_id").groups.items():
            f = chat_dir / str(task_id) / "chat.json"
            if not f.exists():
                continue
            try:
                data  = _json.loads(f.read_text())
                steps = {s["step"]: s.get("raw_prediction", "")
                         for s in data.get("steps", [])}
            except Exception:
                continue
            for i in grp_idx:
                step = int(df.at[i, "step"])
                text = steps.get(step, "")
                m = _CONF_RE.search(text)
                if m:
                    df.at[i, "m1_verbalized_conf"] = float(m.group(1))

    elif fmt == "C":
        # Format C: mind2web result.json — step_samples[k][0] is a raw text string
        for task_idx, grp_idx in df.groupby(id_col).groups.items():
            f = conf_dir / str(task_idx) / "result.json"
            if not f.exists():
                continue
            try:
                data     = _json.loads(f.read_text())
                ss       = data.get("step_samples", [])
                step_map = {k: (ss[k][0] if ss[k] else "") for k in range(len(ss))}
            except Exception:
                continue
            for i in grp_idx:
                step = int(df.at[i, "step"])
                text = step_map.get(step, "")
                m = _CONF_RE.search(str(text))
                if m:
                    df.at[i, "m1_verbalized_conf"] = float(m.group(1))

    else:
        # Format A: uq_samples/{task_id}/step_{k}_samples.json
        for task_id, grp_idx in df.groupby("task_id").groups.items():
            task_dir = conf_dir / str(task_id)
            if not task_dir.exists():
                continue
            for i in grp_idx:
                step = int(df.at[i, "step"])
                f = task_dir / f"step_{step}_samples.json"
                if not f.exists():
                    continue
                try:
                    data  = _json.loads(f.read_text())
                    samps = data.get("samples", [])
                    if not samps:
                        continue
                    text = samps[0].get("raw_prediction", "") if isinstance(samps[0], dict) else samps[0]
                    m = _CONF_RE.search(text)
                    if m:
                        df.at[i, "m1_verbalized_conf"] = float(m.group(1))
                except Exception:
                    pass

    return df


def add_m7_from_m1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute M7 = cumulative mean of M1 (verbalized confidence) from step 0 → k.
    Computed per task, in step order.  If M1 is entirely NaN for a task, M7 stays NaN.
    """
    df = df.copy().reset_index(drop=True)

    if "m1_verbalized_conf" not in df.columns:
        df["m7_conf_cumul_mean"] = float("nan")
        return df

    id_col = "task_index" if "task_index" in df.columns else "task_id"
    m7 = np.full(len(df), float("nan"))

    for _, idx in df.groupby(id_col).groups.items():
        grp_idx = df.loc[idx].sort_values("step").index
        vals = []
        for i in grp_idx:
            v = df.at[i, "m1_verbalized_conf"]
            if pd.notna(v):
                vals.append(float(v))
            m7[i] = float(np.mean(vals)) if vals else float("nan")

    df["m7_conf_cumul_mean"] = m7
    return df


def merge_features(per_step: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """
    Add M7 (derived directly from M1, already in per_step_scores).
    M8/M9/M10 (mean-feature step_scores.csv merge) removed 2026-07-30 -- see
    module docstring.
    """
    df = per_step.copy()
    df = add_m7_from_m1(df)
    return df


# ── Task-level aggregation ─────────────────────────────────────────────────────

def build_task_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "task_id" not in df.columns:
        if "task_id_x" in df.columns:
            df = df.rename(columns={"task_id_x": "task_id"})
        elif "task_id_y" in df.columns:
            df = df.rename(columns={"task_id_y": "task_id"})
    drop_helper_ids = [c for c in ("task_id_x", "task_id_y") if c in df.columns and c != "task_id"]
    if drop_helper_ids:
        df = df.drop(columns=drop_helper_ids)

    all_score_cols = [
        "m1_verbalized_conf", "m2_action_logprob_mean",
        "m3_output_logprob_mean", "m4_cumulative_output_logprob_mean",
        "m5_action_entropy", "m6_cumulative_action_entropy",
        "m7_conf_cumul_mean",
        "m11_htc_full_step", "m12_htc_reduced_step",
    ]
    group_col = "task_index" if "task_index" in df.columns else "task_id"

    rows = []
    for tid, grp in df.groupby(group_col):
        grp = grp.sort_values("step").reset_index(drop=True)
        row = {
            "task_id":   grp["task_id"].iloc[0],
            "num_steps": len(grp),
            "success":   grp["success"].iloc[-1],
        }
        if "model"      in grp.columns: row["model"]      = grp["model"].iloc[0]
        if "benchmark"  in grp.columns: row["benchmark"]  = grp["benchmark"].iloc[0]
        if "task_index" in grp.columns: row["task_index"] = grp["task_index"].iloc[0]

        for col in all_score_cols:
            if col not in grp.columns:
                continue
            try:
                vals = [float(x) for x in grp[col] if pd.notna(x)]
            except (TypeError, ValueError):
                vals = []
            row[f"{col}_last"] = vals[-1]              if vals else float('nan')
            row[f"{col}_mean"] = float(np.mean(vals))  if vals else float('nan')
        rows.append(row)
    return pd.DataFrame(rows)


# ── Per-config runner ──────────────────────────────────────────────────────────

def run_one(baseline_dir: Path, source: str, uq_dir: str, conf_dir_rel=None, htc_source: str | None = None,
            htc_verbalized_dir: str | None = None):
    stale_htc_task = baseline_dir / "htc_task_scores.csv"
    if stale_htc_task.exists():
        stale_htc_task.unlink()

    pss_csv   = baseline_dir / "per_step_scores.csv"

    if not pss_csv.exists():
        print(f"  [SKIP] missing: {pss_csv}")
        return

    per_step  = pd.read_csv(pss_csv)
    if "task_id" not in per_step.columns:
        if "task_id_x" in per_step.columns:
            per_step = per_step.rename(columns={"task_id_x": "task_id"})
        elif "task_id_y" in per_step.columns:
            per_step = per_step.rename(columns={"task_id_y": "task_id"})
    drop_helper_ids = [c for c in ("task_id_x", "task_id_y") if c in per_step.columns and c != "task_id"]
    if drop_helper_ids:
        per_step = per_step.drop(columns=drop_helper_ids)
    # Drop any previously added M7–M10 columns (idempotent re-runs)
    drop_cols = [c for c in per_step.columns
                 if any(c.startswith(p) for p in
                        ("m7_", "m8_", "m9_", "m10_", "m11_", "m12_", "score_"))]
    per_step  = per_step.drop(columns=drop_cols)

    # If success column is missing, load it from task_scores.csv via task_index/task_id
    if "success" not in per_step.columns:
        task_csv = baseline_dir / "task_scores.csv"
        if task_csv.exists():
            task_gt = pd.read_csv(task_csv)
            id_col  = "task_index" if "task_index" in task_gt.columns else "task_id"
            if "success" in task_gt.columns and id_col in per_step.columns:
                success_map = task_gt.set_index(id_col)["success"]
                per_step["success"] = per_step[id_col].map(success_map)

    if "success" not in per_step.columns:
        raise KeyError("'success' column missing from per_step_scores and could not be loaded from task_scores.csv")

    if per_step["success"].dtype == object:
        per_step["success"] = per_step["success"].map({"success": 1, "failure": 0})
    per_step["success"] = pd.to_numeric(per_step["success"], errors="coerce")

    # success_entropy (2026-07-30): M5/M6's own label, same dtype handling.
    if "success_entropy" in per_step.columns:
        if per_step["success_entropy"].dtype == object:
            per_step["success_entropy"] = per_step["success_entropy"].map({"success": 1, "failure": 0})
        per_step["success_entropy"] = pd.to_numeric(per_step["success_entropy"], errors="coerce")

    # Load M1 from external verbalized samples dir if provided
    if conf_dir_rel is not None:
        conf_dir = RESULTS_ROOT / conf_dir_rel
        if conf_dir.exists():
            per_step = load_m1_from_verbalized_samples(per_step, conf_dir)
            m1_loaded = per_step["m1_verbalized_conf"].notna().sum()
            print(f"  M1 loaded from conf_dir: {m1_loaded} non-null values")
        else:
            print(f"  [WARN] conf_dir not found: {conf_dir}")

    benchmark = per_step["benchmark"].iloc[0] if "benchmark" in per_step.columns else "unknown"
    model     = per_step["model"].iloc[0]     if "model"     in per_step.columns else "unknown"

    merged = merge_features(per_step, benchmark)

    # M11's final L2 fit uses task-length sample weighting (1/n_steps per
    # task, same convention as M14/Ours) -- 2026-07-31.
    # compute_htc_step_scores() itself still defaults to
    # task_length_weight=False for standalone/unweighted use.
    htc_step_df = compute_htc_step_scores(htc_source or source, merged, verbalized_dir_override=htc_verbalized_dir,
                                           task_length_weight=True)
    if not htc_step_df.empty:
        join_cols = ["task_id", "step"]
        if benchmark == "mind2web" and "task_index" in merged.columns and "task_index" in htc_step_df.columns:
            join_cols = ["task_index", "step"]
        merged = merged.merge(
            htc_step_df[join_cols + HTC_SCORE_COLS],
            on=join_cols,
            how="left",
        )
        htc_step_df.to_csv(baseline_dir / "htc_step_scores.csv", index=False)
        print(f"  htc_step_scores.csv saved    ({len(htc_step_df)} rows)")
    else:
        print("  [SKIP] HTC unavailable for this source (likely missing token logprobs)")

    if "task_id" not in merged.columns:
        if "task_id_x" in merged.columns:
            merged = merged.rename(columns={"task_id_x": "task_id"})
        elif "task_id_y" in merged.columns:
            merged = merged.rename(columns={"task_id_y": "task_id"})
    drop_helper_ids = [c for c in ("task_id_x", "task_id_y") if c in merged.columns and c != "task_id"]
    if drop_helper_ids:
        merged = merged.drop(columns=drop_helper_ids)

    merged.to_csv(baseline_dir / "per_step_scores.csv", index=False)
    print(f"  per_step_scores.csv updated  ({len(merged)} rows)")

    task_df = build_task_scores(merged)

    task_df.to_csv(baseline_dir / "task_scores.csv", index=False)
    print(f"  task_scores.csv updated      ({len(task_df)} tasks)")

    # Compute metrics for all methods (M1–M10) at per-step level
    success_rate = float(merged["success"].mean())
    all_cols = [
        "m1_verbalized_conf", "m2_action_logprob_mean",
        "m3_output_logprob_mean", "m4_cumulative_output_logprob_mean",
        "m5_action_entropy", "m6_cumulative_action_entropy",
    ] + NEW_SCORE_COLS + HTC_SCORE_COLS

    # Ensure flip direction is set for M1-M6
    from_m16 = {
        "m1_verbalized_conf": True,
        "m2_action_logprob_mean": False,
        "m3_output_logprob_mean": False,
        "m4_cumulative_output_logprob_mean": False,
        "m5_action_entropy": False,
        "m6_cumulative_action_entropy": False,
    }
    for col in all_cols:
        if col not in HIGHER_IS_CONFIDENT:
            HIGHER_IS_CONFIDENT[col] = from_m16.get(col, True)

    # M5/M6 (2026-07-30, bug): their score comes from
    # entropy_dir (the plain rollout), not verbalized_dir like everything
    # else here -- they need the "success_entropy" label (keyStep annotated
    # against entropy_dir specifically), not the shared "success" column.
    ENTROPY_SOURCED_METHODS = {"m5_action_entropy", "m6_cumulative_action_entropy"}

    records = []
    group_col = "task_index" if "task_index" in merged.columns else "task_id"
    for col in all_cols:
        if col not in merged.columns:
            continue
        label_col = "success_entropy" if col in ENTROPY_SOURCED_METHODS else "success"
        if label_col not in merged.columns:
            continue
        sub = merged[[group_col, label_col, col]].dropna()
        if len(sub) == 0:
            continue
        sub = sub.rename(columns={label_col: "success"})
        res = evaluate_method(col, sub, success_rate)
        res["model"]     = model
        res["benchmark"] = benchmark
        records.append(res)

    if records:
        out_cols = [
            "model", "benchmark", "method", "n",
            "auroc", "aurc", "aurc_ref", "aurc_gain", "aurc_oracle", "e_aurc",
            "ece_raw", "ece_ref",
            "ece_platt", "ece_gain_platt",
            "ece_isotonic", "ece_gain_isotonic",
            "ece_temp", "ece_gain_temp",
            "brier_raw", "brier_ref",
            "brier_platt", "bss_platt",
            "brier_isotonic", "bss_isotonic",
            "brier_temp", "bss_temp",
        ]
        metrics_df = pd.DataFrame(records)[[c for c in out_cols if c in pd.DataFrame(records).columns]]
        metrics_df.to_csv(baseline_dir / "metrics.csv", index=False)
        print(f"  metrics.csv updated          ({len(metrics_df)} methods)")
        print()
        hdr = f"  {'Method':<32}  {'AUROC':>7}  {'AURC gain':>10}"
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for _, row in metrics_df.iterrows():
            auroc_val = row.get("auroc", float('nan'))
            gain_val  = row.get("aurc_gain", float('nan'))
            print(f"  {row['method']:<32}  {auroc_val:>7.4f}  {gain_val:>+10.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="Run all model configurations defined in MODEL_CONFIGS")
    ap.add_argument("--baseline_dir", default=None,
                    help="Relative path to baseline results dir under results_total/")
    ap.add_argument("--source",       default=None,
                    help="Relative path to model source dir under results_total/")
    ap.add_argument("--uq_dir",       default=None,
                    help="UQ analysis subdirectory name")
    ap.add_argument("--htc_source",   default=None,
                    help="Optional source override used only for HTC logprob loading")
    ap.add_argument("--htc_verbalized_dir", default=None,
                    help="Optional explicit verbalized-rollout dir for HTC logprob loading "
                         "(e.g. data/<bench>/<model>/raw/verbalized) -- bypasses "
                         "htc_step_baseline.py's RESULTS_ROOT-relative default resolution, which "
                         "only matches the working tree's <bench>/results_<model>_verbalized layout.")
    args = ap.parse_args()

    if args.all:
        configs = MODEL_CONFIGS
    elif args.baseline_dir and args.source and args.uq_dir:
        configs = [(args.baseline_dir, args.source, args.uq_dir, None, args.htc_source)]
    else:
        ap.error("Provide --all or all three of --baseline_dir, --source, --uq_dir")

    for entry in configs:
        bdir_rel, source, uq_dir = entry[0], entry[1], entry[2]
        conf_dir_rel = entry[3] if len(entry) > 3 else None
        htc_source = entry[4] if len(entry) > 4 else None
        baseline_dir = RESULTS_ROOT / bdir_rel
        print(f"\n{'='*60}")
        print(f"  {source}  [{uq_dir}]")
        print(f"  -> {baseline_dir}")
        print(f"{'='*60}")
        run_one(baseline_dir, source, uq_dir, conf_dir_rel, htc_source,
                htc_verbalized_dir=args.htc_verbalized_dir)


if __name__ == "__main__":
    main()
