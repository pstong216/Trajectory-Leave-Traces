from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from run_htc_all import extract_htc_features, search_alpha

RESULTS_ROOT = Path(__file__).resolve().parent.parent


def _mind2web_paths(source: str) -> tuple[Path, Path] | None:
    model = source.split("/", 1)[1]
    gt_map = {
        "kimi": "kimi_result.json",
        "gpt_5_2": "gpt_5_2_result.json",
        "gptnano": "gptnano_result.json",
        "qwen30b": "qwen_result.json",
        "claude": "claude_result.json",
    }
    verbalized_map = {
        "kimi": "results_kimi_verbalized",
        "gpt_5_2": "results_gpt_5_2_verbalized",
        "gptnano": "results_gptnano_verbalized",
        "qwen30b": "results_qwen30b_verbalized",
        "claude": "results_claude_verbalized",
    }
    if model not in gt_map or model not in verbalized_map:
        return None
    base = RESULTS_ROOT / "mind2web"
    return base / verbalized_map[model], base / gt_map[model]


def load_mind2web_task_level(source: str) -> list[dict]:
    resolved = _mind2web_paths(source)
    if resolved is None:
        return []
    verbalized_dir, gt_path = resolved
    if not verbalized_dir.is_dir() or not gt_path.is_file():
        return []

    gt_raw = json.loads(gt_path.read_text())
    gt = {str(t["task_id"]): 1 if t["status"] == "success" else 0
          for t in gt_raw.get("tasks", [])}

    rows = []
    for folder in sorted(verbalized_dir.iterdir()):
        if not folder.is_dir():
            continue
        rpath = folder / "result.json"
        if not rpath.is_file():
            continue
        try:
            result = json.loads(rpath.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        task_index = result.get("task_index")
        task_id = str(result.get("task_id", folder.name))
        task_key = str(task_index) if task_index is not None else folder.name
        if task_key not in gt:
            continue

        dca = []
        for step_lp in result.get("step_logprobs", []):
            if not step_lp:
                continue
            lps = step_lp.get("token_logprobs", [])
            if not lps:
                continue
            dca.append({
                "cm": {
                    "token_confidence_list": [float(np.exp(lp)) for lp in lps],
                    "average_trace_confidence_top1": float(np.mean([np.exp(lp) for lp in lps])),
                    "average_trace_confidence": float(np.mean([
                        np.mean([np.exp(v) for v in top.values() if v > -100]) if top else np.exp(lp)
                        for lp, top in zip(lps, step_lp.get("top_logprobs", []), strict=False)
                    ])) if step_lp.get("top_logprobs") else float(np.mean([np.exp(lp) for lp in lps])),
                }
            })

        if dca:
            rows.append({
                "task_id": task_id,
                "task_index": task_index,
                "success": int(gt[task_key]),
                "label": int(gt[task_key]),
                "dca": dca,
            })
    return rows


def load_webarena_task_level(source: str) -> list[dict]:
    model = source.split("/", 1)[1]
    vroot = RESULTS_ROOT / "webarena" / f"results_{model}_verbalized"
    traj_root = vroot / "trajectory_1"
    uq_root = traj_root / "uq_samples" if traj_root.is_dir() else vroot / "uq_samples"
    act_root = traj_root / "actions" if traj_root.is_dir() else vroot / "actions"
    if not uq_root.is_dir() or not act_root.is_dir():
        return []

    gt = {}
    for f in act_root.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            tid = str(d.get("task_id", f.stem))
            score = d.get("score")
            if score is not None:
                gt[tid] = 1 if float(score) >= 1.0 else 0
        except (OSError, json.JSONDecodeError, ValueError):
            continue

    rows = []
    for task_dir in sorted(uq_root.iterdir()):
        if not task_dir.is_dir():
            continue
        tid = task_dir.name
        if tid not in gt:
            continue
        dca = []
        for f in sorted(task_dir.glob("step_*_logprobs.json")):
            try:
                d = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            samples = d.get("samples", [])
            if not samples:
                continue
            s = samples[0]
            lps = s.get("token_logprobs", [])
            if not lps:
                continue
            dca.append({
                "cm": {
                    "token_confidence_list": [float(np.exp(lp)) for lp in lps],
                    "average_trace_confidence_top1": float(np.mean([np.exp(lp) for lp in lps])),
                    "average_trace_confidence": float(np.mean([
                        np.mean([np.exp(v) for v in top.values() if v > -100]) if top else np.exp(lp)
                        for lp, top in zip(lps, s.get("top_logprobs", []), strict=False)
                    ])) if s.get("top_logprobs") else float(np.mean([np.exp(lp) for lp in lps])),
                }
            })
        if dca:
            rows.append({
                "task_id": tid,
                "success": int(gt[tid]),
                "label": int(gt[tid]),
                "dca": dca,
            })
    return rows


def load_task_level_data(source: str) -> list[dict]:
    if source.startswith("mind2web/"):
        return load_mind2web_task_level(source)
    if source.startswith("webarena/"):
        return load_webarena_task_level(source)
    return []


def compute_htc_task_scores(source: str) -> pd.DataFrame:
    data = load_task_level_data(source)
    if not data:
        return pd.DataFrame()

    X, y = extract_htc_features(data)
    if len(X) == 0 or len(np.unique(y)) < 2:
        return pd.DataFrame()

    n = len(data)
    full_probs = np.full(n, np.nan, dtype=float)
    reduced_probs = np.full(n, np.nan, dtype=float)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in cv.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])
        y_tr = y[train_idx]

        full = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42
        )
        full.fit(X_tr, y_tr)
        full_probs[val_idx] = full.predict_proba(X_va)[:, 1]

        best = search_alpha(X_tr, y_tr)
        if best:
            reduced = LogisticRegression(
                penalty="l1",
                C=1.0 / best["alpha"],
                solver="liblinear",
                max_iter=1000,
                random_state=42,
            )
            sel = np.array(best["selected"], dtype=int)
            reduced.fit(X_tr[:, sel], y_tr)
            reduced_probs[val_idx] = reduced.predict_proba(X_va[:, sel])[:, 1]

    rows = []
    for item, p_full, p_red in zip(data, full_probs, reduced_probs, strict=False):
        row = {
            "task_id": item["task_id"],
            "success": item["success"],
            "m11_htc_full_task": p_full,
            "m12_htc_reduced_task": p_red,
        }
        if "task_index" in item and item["task_index"] is not None:
            row["task_index"] = item["task_index"]
        rows.append(row)
    return pd.DataFrame(rows)
