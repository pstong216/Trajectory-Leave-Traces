#!/usr/bin/env python3
"""
run_htc_all.py — Run HTC (Holistic Trajectory Calibration) on all
model × benchmark combinations.

Benchmarks:  mind2web  |  webarena
Models:      kimi, claude, gpt_5_2, gptnano, qwen30b

Paper: "Agentic Confidence Calibration" (arXiv 2601.15778v1)

48-feature taxonomy (Appendix A.5):
  Dynamics (19)  — cross-step confidence gradients, token gradients,
                   step progression CoV, confidence change
  Position (14)  — first/last step entropy, concentration, spread,
                   volatility, skewness, top1/topk avg
  Stability (10) — mean/std of per-step entropy/concentration/spread,
                   token volatility/skewness
  Structure (5)  — normalized step count, first/last token counts,
                   avg/std tokens per step

Model: LogisticRegression + StandardScaler, 5-fold stratified CV
  HTC-Full    : L2 (ridge), all 48 features
  HTC-Reduced : L1 (lasso), alpha grid search → sparse feature selection

Metrics: AUROC, Brier Score, ECE
"""

import json
import math
import os
import re
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))   # results_total/
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "htc_results")

# mind2web: model → (verbalized_dir_name, gt_filename)
MIND2WEB_MODELS = {
    "kimi":    ("results_kimi_verbalized",    "kimi_result.json"),
    "gpt_5_2": ("results_gpt_5_2_verbalized", "gpt_5_2_result.json"),
    "gptnano": ("results_gptnano_verbalized", "gptnano_result.json"),
    "qwen30b": ("results_qwen30b_verbalized", "qwen_result.json"),
    # claude has no token logprobs in this dataset — skipped
}

# webarena: model → has trajectory_1 sub-folder?
WEBARENA_MODELS = {
    "kimi":    True,   # trajectory_1/uq_samples/
    "claude":  True,
    "gpt_5_2": False,  # uq_samples/ directly
    "gptnano": False,
    "qwen30b": False,
}

ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0,
              6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 50.0]

# ---------------------------------------------------------------------------
# Feature extraction  (48 features, faithful to paper code + Appendix A.5)
# ---------------------------------------------------------------------------

def _step_to_cm(token_logprobs: list, top_logprobs: list) -> dict:
    """
    Convert raw logprob lists to confidence_metrics dict expected by
    extract_htc_features().
    """
    conf_list = [math.exp(lp) for lp in token_logprobs]
    top1_avg  = float(np.mean(conf_list)) if conf_list else 0.0

    # top-k avg: for each token position, mean of exp(valid top-k logprobs)
    topk_avgs = []
    for i, lp in enumerate(token_logprobs):
        if top_logprobs and i < len(top_logprobs) and top_logprobs[i]:
            valid = [v for v in top_logprobs[i].values() if v > -100]
            topk_avgs.append(float(np.mean([math.exp(v) for v in valid]))
                             if valid else math.exp(lp))
        else:
            topk_avgs.append(math.exp(lp))
    topk_avg = float(np.mean(topk_avgs)) if topk_avgs else 0.0

    return {
        "token_confidence_list":           conf_list,
        "average_trace_confidence_top1":   top1_avg,
        "average_trace_confidence":        topk_avg,
    }


def extract_htc_features(data: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    data: list of {'dca': [{'cm': {...}}], 'label': 0/1}
    Returns X (N×48) and y (N).
    """
    features_list, labels = [], []

    for item in data:
        dca = item.get("dca", [])
        if not dca:
            continue

        step_count      = len(dca)
        all_tokens      = [s["cm"]["token_confidence_list"] for s in dca]
        all_top1        = [s["cm"]["average_trace_confidence_top1"] for s in dca]
        all_topk        = [s["cm"]["average_trace_confidence"] for s in dca]
        all_counts      = [len(t) for t in all_tokens]

        # ── A: Dynamics (19) ─────────────────────────────────────────────────

        if len(all_top1) > 1:
            g1 = np.diff(all_top1)
            t1_g_mean  = float(np.mean(g1))
            t1_g_std   = float(np.std(g1))
            t1_g_max   = float(np.max(g1))
            t1_g_min   = float(np.min(g1))
            t1_g_trend = float(g1[-1] - g1[0]) if len(g1) > 1 else 0.0
        else:
            t1_g_mean = t1_g_std = t1_g_max = t1_g_min = t1_g_trend = 0.0

        if len(all_topk) > 1:
            gk = np.diff(all_topk)
            tk_g_mean  = float(np.mean(gk))
            tk_g_std   = float(np.std(gk))
            tk_g_max   = float(np.max(gk))
            tk_g_min   = float(np.min(gk))
            tk_g_trend = float(gk[-1] - gk[0]) if len(gk) > 1 else 0.0
        else:
            tk_g_mean = tk_g_std = tk_g_max = tk_g_min = tk_g_trend = 0.0

        tok_grads = []
        for tokens in all_tokens:
            if len(tokens) > 1:
                tok_grads.extend(np.diff(tokens).tolist())
        if tok_grads:
            tg_mean = float(np.mean(tok_grads))
            tg_std  = float(np.std(tok_grads))
            tg_max  = float(np.max(tok_grads))
            tg_min  = float(np.min(tok_grads))
        else:
            tg_mean = tg_std = tg_max = tg_min = 0.0

        # Per-step distribution stats (H, κ, ρ) — normalized entropy for stability
        H_list, kap_list, rho_list = [], [], []
        for tokens in all_tokens:
            if tokens:
                r  = np.array(tokens)
                mu = np.mean(r) + 1e-8
                pi = r / (np.sum(r) + 1e-8)
                H_list.append(float(-np.sum(pi * np.log(pi + 1e-8))))
                kap_list.append(float(np.max(r) / mu))
                rho_list.append(float(np.std(r) / mu))
            else:
                H_list.append(0.0); kap_list.append(0.0); rho_list.append(0.0)

        if step_count > 1:
            step_prog_ent  = float(np.std(H_list))   / (float(np.mean(H_list))   + 1e-8)
            step_prog_conc = float(np.std(kap_list)) / (float(np.mean(kap_list)) + 1e-8)
            step_prog_spr  = float(np.std(rho_list)) / (float(np.mean(rho_list)) + 1e-8)
        else:
            step_prog_ent = step_prog_conc = step_prog_spr = 0.0

        t1_change = float(all_top1[-1] - all_top1[0])
        tk_change = float(all_topk[-1] - all_topk[0])

        # ── B: Position (14) — raw entropy for first/last (matches paper code) ──

        def _pos(tokens):
            if tokens:
                r     = np.array(tokens)
                mu    = np.mean(r)
                sigma = np.std(r)
                H     = float(-np.sum(r * np.log(r + 1e-8)))          # raw entropy
                kappa = float(np.max(r) / (mu + 1e-8))
                rho   = float(sigma / (mu + 1e-8))
                skew  = float(np.mean(((r - mu) / (sigma + 1e-8)) ** 3))
                return H, kappa, rho, rho, skew
            return 0.0, 0.0, 0.0, 0.0, 0.0

        fH, fK, fS, fV, fSk = _pos(all_tokens[0])
        lH, lK, lS, lV, lSk = _pos(all_tokens[-1])
        ft1, ftk = float(all_top1[0]), float(all_topk[0])
        lt1, ltk = float(all_top1[-1]), float(all_topk[-1])

        # ── C: Stability (10) ────────────────────────────────────────────────

        aH_m   = float(np.mean(H_list))
        aH_s   = float(np.std(H_list))
        aK_m   = float(np.mean(kap_list))
        aK_s   = float(np.std(kap_list))
        aR_m   = float(np.mean(rho_list))
        aR_s   = float(np.std(rho_list))

        vol_list, skew_list = [], []
        for tokens in all_tokens:
            if len(tokens) > 1:
                r  = np.array(tokens)
                mu = np.mean(r); sigma = np.std(r)
                vol_list.append(float(sigma / (mu + 1e-8)))
                skew_list.append(float(np.mean(((r - mu) / (sigma + 1e-8)) ** 3)))

        tv_m  = float(np.mean(vol_list))  if vol_list  else 0.0
        tv_s  = float(np.std(vol_list))   if vol_list  else 0.0
        tsk_m = float(np.mean(skew_list)) if skew_list else 0.0
        tsk_s = float(np.std(skew_list))  if skew_list else 0.0

        # ── D: Structure (5) ─────────────────────────────────────────────────

        norm_sc   = float(step_count) / 10.0
        first_tc  = float(len(all_tokens[0]))
        last_tc   = float(len(all_tokens[-1]))
        avg_tps   = float(np.mean(all_counts)) if all_counts else 0.0
        std_tps   = float(np.std(all_counts))  if len(all_counts) > 1 else 0.0

        # ── Assemble (same order as feature_map_final.py) ────────────────────
        feat = [
            # Dynamics (19)
            t1_g_mean, t1_g_std, t1_g_max, t1_g_min, t1_g_trend,
            tk_g_mean, tk_g_std, tk_g_max, tk_g_min, tk_g_trend,
            tg_mean,   tg_std,   tg_max,   tg_min,
            step_prog_ent, step_prog_conc, step_prog_spr,
            t1_change, tk_change,
            # Position (14)
            fH, fK, fS, fV, fSk, ft1, ftk,
            lH, lK, lS, lV, lSk, lt1, ltk,
            # Stability (10)
            aH_m, aH_s, aK_m, aK_s, aR_m, aR_s,
            tv_m, tv_s, tsk_m, tsk_s,
            # Structure (5)
            std_tps, first_tc, last_tc, avg_tps, norm_sc,
        ]
        features_list.append(feat)
        labels.append(item["label"])

    return np.array(features_list, dtype=np.float64), np.array(labels, dtype=int)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_mind2web(model: str) -> list[dict]:
    """Load mind2web data for one model. Returns list of {dca, label}."""
    base     = os.path.join(REPO_ROOT, "mind2web")
    vdir_name, gt_file = MIND2WEB_MODELS[model]
    vpath    = os.path.join(base, vdir_name)
    gt_path  = os.path.join(base, gt_file)

    # Ground truth: task_id → 0/1
    gt_raw = json.load(open(gt_path))
    gt = {str(t["task_id"]): (1 if t["status"] == "success" else 0)
          for t in gt_raw.get("tasks", [])}

    data = []
    for folder in sorted(os.listdir(vpath)):
        if folder.startswith("."):
            continue
        rpath = os.path.join(vpath, folder, "result.json")
        if not os.path.isfile(rpath):
            continue
        try:
            result = json.load(open(rpath))
        except (json.JSONDecodeError, OSError):
            continue

        # Use task_index (int) to match GT task_id (string)
        task_index = result.get("task_index")
        tid = str(task_index) if task_index is not None else folder
        if tid not in gt:
            continue

        dca = []
        for step_lp in result.get("step_logprobs", []):
            if not step_lp:
                continue
            lps = step_lp.get("token_logprobs", [])
            if not lps:
                continue
            cm = _step_to_cm(lps, step_lp.get("top_logprobs", []))
            dca.append({"cm": cm})

        if dca:
            data.append({"dca": dca, "label": gt[tid]})

    return data


def load_webarena(model: str) -> list[dict]:
    """Load webarena data for one model. Returns list of {dca, label}."""
    base       = os.path.join(REPO_ROOT, "webarena")
    has_traj1  = WEBARENA_MODELS[model]
    vpath      = os.path.join(base, f"results_{model}_verbalized")

    if has_traj1:
        uq_path  = os.path.join(vpath, "trajectory_1", "uq_samples")
        act_path = os.path.join(vpath, "trajectory_1", "actions")
    else:
        uq_path  = os.path.join(vpath, "uq_samples")
        act_path = os.path.join(vpath, "actions")

    # Ground truth from actions/{task_id}.json
    gt = {}
    if os.path.isdir(act_path):
        for fname in os.listdir(act_path):
            if not fname.endswith(".json"):
                continue
            try:
                d = json.load(open(os.path.join(act_path, fname)))
                tid   = str(d.get("task_id", fname.replace(".json", "")))
                score = d.get("score", None)
                if score is not None:
                    gt[tid] = 1 if float(score) >= 1.0 else 0
            except (json.JSONDecodeError, OSError, ValueError):
                pass

    data = []
    if not os.path.isdir(uq_path):
        return data

    for task_folder in sorted(os.listdir(uq_path)):
        if task_folder.startswith("."):
            continue
        task_dir = os.path.join(uq_path, task_folder)
        if not os.path.isdir(task_dir):
            continue

        tid = str(task_folder)
        if tid not in gt:
            continue

        # Collect step files and sort by step index
        step_files = {}
        for fname in os.listdir(task_dir):
            if not fname.endswith("_logprobs.json"):
                continue
            m = re.search(r"step_(\d+)_logprobs", fname)
            if m:
                step_files[int(m.group(1))] = os.path.join(task_dir, fname)

        dca = []
        for k in sorted(step_files):
            try:
                d = json.load(open(step_files[k]))
            except (json.JSONDecodeError, OSError):
                continue
            samples = d.get("samples", [])
            if not samples:
                continue
            s   = samples[0]
            lps = s.get("token_logprobs", [])
            if not lps:
                continue
            cm = _step_to_cm(lps, s.get("top_logprobs", []))
            dca.append({"cm": cm})

        if dca:
            data.append({"dca": dca, "label": gt[tid]})

    return data


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if mask.sum() > 0:
            ece += abs(y_prob[mask].mean() - y_true[mask].mean()) * mask.mean()
    return float(ece)


def cv_evaluate(model, X: np.ndarray, y: np.ndarray,
                n_splits: int = 5, seed: int = 42) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    au, br, ec = [], [], []
    for tr, va in cv.split(X, y):
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[va])[:, 1]
        au.append(roc_auc_score(y[va], p))
        br.append(brier_score_loss(y[va], p))
        ec.append(compute_ece(y[va], p))
    return {
        "auroc": (float(np.mean(au)), float(np.std(au))),
        "brier": (float(np.mean(br)), float(np.std(br))),
        "ece":   (float(np.mean(ec)), float(np.std(ec))),
    }


def search_alpha(X: np.ndarray, y: np.ndarray) -> dict | None:
    best_score, best = -np.inf, None
    for alpha in ALPHA_GRID:
        C = 1.0 / alpha
        m = LogisticRegression(penalty="l1", C=C, solver="liblinear",
                               max_iter=1000, random_state=42)
        m.fit(X, y)
        sel = np.where(m.coef_[0] != 0)[0]
        if len(sel) == 0:
            continue
        m2 = LogisticRegression(penalty="l1", C=C, solver="liblinear",
                                max_iter=1000, random_state=42)
        met = cv_evaluate(m2, X[:, sel], y)
        score = met["auroc"][0] - met["brier"][0] - met["ece"][0]
        if score > best_score:
            best_score = score
            best = {"alpha": alpha, "n_features": int(len(sel)),
                    "selected": sel.tolist(), "metrics": met}
    return best


# ---------------------------------------------------------------------------
# Run one combination
# ---------------------------------------------------------------------------

def run_one(benchmark: str, model: str) -> dict | None:
    print(f"\n{'─'*60}")
    print(f"  {benchmark.upper()} / {model}")
    print(f"{'─'*60}")

    # Load data
    try:
        if benchmark == "mind2web":
            data = load_mind2web(model)
        else:
            data = load_webarena(model)
    except Exception as e:
        print(f"  [SKIP] data load error: {e}")
        return None

    if not data:
        print("  [SKIP] no data loaded.")
        return None

    X, y = extract_htc_features(data)
    if X.shape[0] == 0:
        print("  [SKIP] no valid features extracted.")
        return None
    if len(np.unique(y)) < 2:
        print(f"  [SKIP] only one class (y={np.unique(y)}) — cannot evaluate.")
        return None

    n_pos = int(y.sum())
    print(f"  tasks: {len(X)}  |  success: {n_pos} ({n_pos/len(y):.1%})")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # HTC-Full
    m_full = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                max_iter=1000, random_state=42)
    mf = cv_evaluate(m_full, X_scaled, y)
    print(f"  HTC-Full    AUROC={mf['auroc'][0]:.4f}±{mf['auroc'][1]:.4f}"
          f"  Brier={mf['brier'][0]:.4f}±{mf['brier'][1]:.4f}"
          f"  ECE={mf['ece'][0]:.4f}±{mf['ece'][1]:.4f}")

    # HTC-Reduced
    best = search_alpha(X_scaled, y)
    if best:
        mr = best["metrics"]
        print(f"  HTC-Reduced AUROC={mr['auroc'][0]:.4f}±{mr['auroc'][1]:.4f}"
              f"  Brier={mr['brier'][0]:.4f}±{mr['brier'][1]:.4f}"
              f"  ECE={mr['ece'][0]:.4f}±{mr['ece'][1]:.4f}"
              f"  α={best['alpha']}  feats={best['n_features']}/48")
    else:
        print("  HTC-Reduced [no valid alpha found]")

    return {
        "benchmark": benchmark,
        "model": model,
        "n_tasks": int(len(X)),
        "n_pos": n_pos,
        "pos_rate": float(y.mean()),
        "htc_full": {
            "auroc_mean": mf["auroc"][0], "auroc_std": mf["auroc"][1],
            "brier_mean": mf["brier"][0], "brier_std": mf["brier"][1],
            "ece_mean":   mf["ece"][0],   "ece_std":   mf["ece"][1],
        },
        "htc_reduced": {
            "alpha":         best["alpha"]               if best else None,
            "n_features":    best["n_features"]          if best else None,
            "auroc_mean":    best["metrics"]["auroc"][0] if best else None,
            "auroc_std":     best["metrics"]["auroc"][1] if best else None,
            "brier_mean":    best["metrics"]["brier"][0] if best else None,
            "brier_std":     best["metrics"]["brier"][1] if best else None,
            "ece_mean":      best["metrics"]["ece"][0]   if best else None,
            "ece_std":       best["metrics"]["ece"][1]   if best else None,
        } if best else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("HTC — All Models × Benchmarks")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    # mind2web (skip claude — no logprobs)
    for model in MIND2WEB_MODELS:
        res = run_one("mind2web", model)
        if res:
            all_results.append(res)

    # webarena
    for model in WEBARENA_MODELS:
        res = run_one("webarena", model)
        if res:
            all_results.append(res)

    # ── Save JSON ────────────────────────────────────────────────────────────
    out_json = os.path.join(OUTPUT_DIR, "htc_all_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results → {out_json}")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)

    rows = []
    for r in all_results:
        row = {
            "benchmark": r["benchmark"],
            "model":     r["model"],
            "N":         r["n_tasks"],
            "pos%":      f"{r['pos_rate']:.1%}",
        }
        full = r.get("htc_full", {})
        row["Full_AUROC"] = f"{full.get('auroc_mean', 0):.4f}±{full.get('auroc_std', 0):.4f}"
        row["Full_Brier"] = f"{full.get('brier_mean', 0):.4f}"
        row["Full_ECE"]   = f"{full.get('ece_mean', 0):.4f}"

        red = r.get("htc_reduced") or {}
        row["Red_AUROC"] = f"{red.get('auroc_mean', 0):.4f}±{red.get('auroc_std', 0):.4f}" if red else "—"
        row["Red_Brier"] = f"{red.get('brier_mean', 0):.4f}" if red else "—"
        row["Red_ECE"]   = f"{red.get('ece_mean', 0):.4f}"   if red else "—"
        row["Red_feats"] = str(red.get("n_features", "—"))    if red else "—"
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Save CSV
    out_csv = os.path.join(OUTPUT_DIR, "htc_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSummary CSV → {out_csv}")


if __name__ == "__main__":
    main()
