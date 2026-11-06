#!/usr/bin/env python3
"""
run_htc_prek.py — Pre@k evaluation of HTC features.

For each task trajectory, extract 48 HTC features using only the first k
completed steps (prefix of length k), for k = 1 … 29.  This tests whether
HTC can predict task failure *before* the trajectory is complete, analogous
to the existing RF Pre@k baseline.

Sources evaluated (matching the existing rf_prek_pooled baseline):
  webarena/claude   webarena/gpt_5_2   mind2web/qwen30b

CV protocol: task-level 5-fold stratified CV, using the SAME fold
assignments as rf_prek_pooled/cv_predictions_prek_all.csv for direct
apples-to-apples comparison.

Outcome convention (same as rf_prek_pooled): outcome = 1 → FAILURE.

Outputs
-------
  baseline/htc_results/
    htc_prek_predictions.csv   — per-step predictions (matches rf format)
    htc_prek_auroc_by_k.csv    — AUROC@k for each source and model variant
    htc_prek_auroc_curve.png   — AUROC@k curve vs RF baseline
    htc_prek_summary.json      — overall / mean Pre@k AUROC summary
"""

import json
import math
import os
import re
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from keystep_labels import label_prefix_failure_from_keystep, load_keystep_map

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "htc_results")

EXISTING_PREK = os.path.join(
    REPO_ROOT, "rf_prek_pooled", "cv_predictions_prek_all.csv"
)

# source → (loader_type, verbalized_dir, has_traj1_subdir)
SOURCES = {
    "webarena/claude":  ("webarena", "results_claude_verbalized",  True),
    "webarena/gpt_5_2": ("webarena", "results_gpt_5_2_verbalized", False),
    "mind2web/qwen30b": ("mind2web", "results_qwen30b_verbalized", None),
}

MAX_K  = 29   # match existing pre@k range
ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0,
              6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 50.0]

# ---------------------------------------------------------------------------
# Raw data loaders → {task_id_str: [{token_logprobs, top_logprobs}, ...]}
# ---------------------------------------------------------------------------

def load_webarena_logprobs(model: str, has_traj1: bool) -> dict:
    """Returns {tid: [step_dict, ...]} sorted by step index."""
    base  = os.path.join(REPO_ROOT, "webarena", f"results_{model}_verbalized")
    uq    = os.path.join(base, "trajectory_1", "uq_samples") if has_traj1 \
            else os.path.join(base, "uq_samples")

    result = {}
    if not os.path.isdir(uq):
        return result

    for tid in sorted(os.listdir(uq)):
        if tid.startswith("."):
            continue
        task_dir = os.path.join(uq, tid)
        step_files = {}
        for fname in os.listdir(task_dir):
            m = re.search(r"step_(\d+)_logprobs", fname)
            if m:
                step_files[int(m.group(1))] = os.path.join(task_dir, fname)

        steps = []
        for k in sorted(step_files):
            try:
                d = json.load(open(step_files[k]))
                s = d.get("samples", [{}])[0]
                lps = s.get("token_logprobs", [])
                if lps:
                    steps.append({
                        "token_logprobs": lps,
                        "top_logprobs":   s.get("top_logprobs", []),
                    })
            except (json.JSONDecodeError, OSError, IndexError):
                pass
        if steps:
            result[tid] = steps

    return result


def load_mind2web_logprobs(model: str) -> dict:
    """Returns {tid_str: [step_dict, ...]} using task_index as tid."""
    base   = os.path.join(REPO_ROOT, "mind2web", f"results_{model}_verbalized")
    result = {}

    for folder in sorted(os.listdir(base)):
        if folder.startswith("."):
            continue
        rpath = os.path.join(base, folder, "result.json")
        if not os.path.isfile(rpath):
            continue
        try:
            d = json.load(open(rpath))
        except (json.JSONDecodeError, OSError):
            continue

        ti = d.get("task_index")
        if ti is None:
            continue
        tid = str(ti)

        steps = []
        for sl in d.get("step_logprobs", []):
            if not sl:
                continue
            lps = sl.get("token_logprobs", [])
            if lps:
                steps.append({
                    "token_logprobs": lps,
                    "top_logprobs":   sl.get("top_logprobs", []),
                })
        if steps:
            result[tid] = steps

    return result


# ---------------------------------------------------------------------------
# HTC feature extraction on a prefix of length k
# ---------------------------------------------------------------------------

def _step_to_conf(token_logprobs: list, top_logprobs: list) -> dict:
    """Convert logprobs → confidence_metrics dict."""
    conf   = [math.exp(lp) for lp in token_logprobs]
    t1_avg = float(np.mean(conf)) if conf else 0.0

    tk_avgs = []
    for i, lp in enumerate(token_logprobs):
        if top_logprobs and i < len(top_logprobs) and top_logprobs[i]:
            valid = [v for v in top_logprobs[i].values() if v > -100]
            tk_avgs.append(float(np.mean([math.exp(v) for v in valid]))
                           if valid else math.exp(lp))
        else:
            tk_avgs.append(math.exp(lp))

    return {
        "tokens": conf,
        "top1":   t1_avg,
        "topk":   float(np.mean(tk_avgs)) if tk_avgs else 0.0,
    }


def extract_features_prefix(steps: list, k: int) -> list:
    """
    Extract 48 HTC features using only the first k steps (0-indexed: steps[0..k-1]).
    Returns a flat list of 48 floats, or None if insufficient data.
    """
    prefix = steps[:k]
    if not prefix:
        return None

    cms       = [_step_to_conf(s["token_logprobs"], s["top_logprobs"]) for s in prefix]
    all_tok   = [c["tokens"] for c in cms]
    all_top1  = [c["top1"]   for c in cms]
    all_topk  = [c["topk"]   for c in cms]
    all_cnts  = [len(t)      for t in all_tok]
    step_count = len(cms)

    # ── A: Dynamics (19) ─────────────────────────────────────────────────────

    if step_count > 1:
        g1 = np.diff(all_top1)
        t1gm = float(np.mean(g1));  t1gs = float(np.std(g1))
        t1gx = float(np.max(g1));   t1gn = float(np.min(g1))
        t1gt = float(g1[-1]-g1[0]) if len(g1) > 1 else 0.0
        gk = np.diff(all_topk)
        tkgm = float(np.mean(gk));  tkgs = float(np.std(gk))
        tkgx = float(np.max(gk));   tkgn = float(np.min(gk))
        tkgt = float(gk[-1]-gk[0]) if len(gk) > 1 else 0.0
    else:
        t1gm=t1gs=t1gx=t1gn=t1gt = 0.0
        tkgm=tkgs=tkgx=tkgn=tkgt = 0.0

    tok_grads = []
    for tok in all_tok:
        if len(tok) > 1:
            tok_grads.extend(np.diff(tok).tolist())
    if tok_grads:
        tgm=float(np.mean(tok_grads)); tgs=float(np.std(tok_grads))
        tgx=float(np.max(tok_grads));  tgn=float(np.min(tok_grads))
    else:
        tgm=tgs=tgx=tgn = 0.0

    # per-step H_t, κ_t, ρ_t (normalized entropy for stability)
    H_l, K_l, R_l = [], [], []
    for tok in all_tok:
        if tok:
            r  = np.array(tok); mu = np.mean(r)+1e-8
            pi = r/(np.sum(r)+1e-8)
            H_l.append(float(-np.sum(pi*np.log(pi+1e-8))))
            K_l.append(float(np.max(r)/mu))
            R_l.append(float(np.std(r)/mu))
        else:
            H_l.append(0.0); K_l.append(0.0); R_l.append(0.0)

    if step_count > 1:
        spe  = float(np.std(H_l))/(float(np.mean(H_l))+1e-8)
        spc  = float(np.std(K_l))/(float(np.mean(K_l))+1e-8)
        sps  = float(np.std(R_l))/(float(np.mean(R_l))+1e-8)
    else:
        spe=spc=sps = 0.0

    t1_ch = float(all_top1[-1]-all_top1[0])
    tk_ch = float(all_topk[-1]-all_topk[0])

    # ── B: Position (14) — raw entropy for first/last ────────────────────────

    def _pos(tok):
        if tok:
            r=np.array(tok); mu=np.mean(r); sig=np.std(r)
            H   = float(-np.sum(r*np.log(r+1e-8)))
            kap = float(np.max(r)/(mu+1e-8))
            rho = float(sig/(mu+1e-8))
            skw = float(np.mean(((r-mu)/(sig+1e-8))**3))
            return H, kap, rho, rho, skw
        return 0.0,0.0,0.0,0.0,0.0

    fH,fK,fS,fV,fSk = _pos(all_tok[0])
    lH,lK,lS,lV,lSk = _pos(all_tok[-1])   # "last" = step k
    ft1,ftk = float(all_top1[0]), float(all_topk[0])
    lt1,ltk = float(all_top1[-1]),float(all_topk[-1])

    # ── C: Stability (10) ────────────────────────────────────────────────────

    aHm=float(np.mean(H_l)); aHs=float(np.std(H_l))
    aKm=float(np.mean(K_l)); aKs=float(np.std(K_l))
    aRm=float(np.mean(R_l)); aRs=float(np.std(R_l))

    vol_l, skw_l = [], []
    for tok in all_tok:
        if len(tok) > 1:
            r=np.array(tok); mu=np.mean(r); sig=np.std(r)
            vol_l.append(float(sig/(mu+1e-8)))
            skw_l.append(float(np.mean(((r-mu)/(sig+1e-8))**3)))

    tvm=float(np.mean(vol_l)) if vol_l else 0.0; tvs=float(np.std(vol_l)) if vol_l else 0.0
    tsm=float(np.mean(skw_l)) if skw_l else 0.0; tss=float(np.std(skw_l)) if skw_l else 0.0

    # ── D: Structure (5) ─────────────────────────────────────────────────────

    nsc = float(step_count)/10.0
    ftc = float(len(all_tok[0]))
    ltc = float(len(all_tok[-1]))
    atp = float(np.mean(all_cnts)) if all_cnts else 0.0
    stp = float(np.std(all_cnts))  if len(all_cnts)>1 else 0.0

    return [
        t1gm,t1gs,t1gx,t1gn,t1gt, tkgm,tkgs,tkgx,tkgn,tkgt,
        tgm,tgs,tgx,tgn,
        spe,spc,sps, t1_ch,tk_ch,
        fH,fK,fS,fV,fSk,ft1,ftk,
        lH,lK,lS,lV,lSk,lt1,ltk,
        aHm,aHs,aKm,aKs,aRm,aRs, tvm,tvs,tsm,tss,
        stp,ftc,ltc,atp,nsc,
    ]


# ---------------------------------------------------------------------------
# CV utilities
# ---------------------------------------------------------------------------

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    ece  = 0.0
    for lo,hi in zip(bins[:-1],bins[1:]):
        m = (y_prob>lo)&(y_prob<=hi)
        if m.sum()>0:
            ece += abs(y_prob[m].mean()-y_true[m].mean())*m.mean()
    return float(ece)


def best_alpha_for_fold(X_tr, y_tr, groups):
    """Find best L1 alpha using 3-fold CV within the training fold.
    Grouped by task (2026-07-30, same fix as train_risk_model_temp_sca.py's
    best_alpha_htc(): X_tr/y_tr are pooled (task, k) rows, plain StratifiedKFold
    let rows from the same task span the inner train/val split)."""
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import brier_score_loss
    n_splits = min(3, len(np.unique(y_tr)))
    if n_splits < 2:
        return 1.0
    inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)
    best_score, best_alpha = -np.inf, 1.0
    for alpha in ALPHA_GRID:
        C = 1.0 / alpha
        m = LogisticRegression(penalty="l1", C=C, solver="liblinear",
                               max_iter=1000, random_state=42)
        au_l, br_l, ec_l = [], [], []
        for itr, iva in inner_cv.split(X_tr, y_tr, groups=groups):
            m.fit(X_tr[itr], y_tr[itr])
            if len(np.unique(y_tr[itr])) < 2:
                continue
            p = m.predict_proba(X_tr[iva])[:, 1]
            if len(np.unique(y_tr[iva])) < 2:
                continue
            au_l.append(roc_auc_score(y_tr[iva], p))
            br_l.append(brier_score_loss(y_tr[iva], p))
            ec_l.append(compute_ece(y_tr[iva], p))
        if not au_l:
            continue
        score = float(np.mean(au_l)) - float(np.mean(br_l)) - float(np.mean(ec_l))
        if score > best_score:
            best_score, best_alpha = score, alpha
    return best_alpha


# ---------------------------------------------------------------------------
# Build feature matrix for one source at all k values
# ---------------------------------------------------------------------------

def build_feature_matrix(source: str, logprobs: dict,
                          fold_map: dict) -> pd.DataFrame:
    """
    logprobs : {tid: [step_dicts]}
    fold_map : {tid: fold_int}  — only tasks present here are used
    Returns DataFrame with columns: task_uid, source, step_k, outcome, fold,
                                    feat_0..feat_47
    """
    bench, model = source.split("/")
    rows = []

    for tid, steps in logprobs.items():
        if tid not in fold_map:
            continue
        fold    = fold_map[tid]["fold"]
        final_outcome = fold_map[tid]["outcome"]  # 1=failure, 0=success
        key_step = fold_map[tid].get("key_step")
        n_steps = len(steps)

        for k in range(1, min(n_steps, MAX_K) + 1):
            feat = extract_features_prefix(steps, k)
            if feat is None:
                continue
            outcome = label_prefix_failure_from_keystep(
                final_success=(final_outcome == 0),
                prefix_len=k,
                key_step=key_step,
            )
            if outcome is None:
                continue
            rows.append({
                "task_uid": f"{source}::{tid}",
                "source":   source,
                "step_k":   k,
                "outcome":  outcome,
                "fold":     fold,
                **{f"f{i}": v for i, v in enumerate(feat)},
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load existing fold/outcome assignments ──────────────────────────────
    print("Loading existing pre@k fold assignments...")
    ref = pd.read_csv(EXISTING_PREK)
    # Build {source → {tid → {fold, outcome}}}
    fold_info = {}
    for src in ref.source.unique():
        sub = ref[ref.source==src].drop_duplicates("task_uid")[["task_uid","fold","outcome"]]
        fold_info[src] = {
            row.task_uid.split("::")[-1]: {"fold": int(row.fold), "outcome": int(row.outcome)}
            for row in sub.itertuples()
        }
    print(f"  Sources: {list(fold_info.keys())}")

    # Inject key-step cutoffs for failed WebArena tasks when annotations exist.
    for src in list(fold_info.keys()):
        if not src.startswith("webarena/"):
            continue
        keystep_map = load_keystep_map(src)
        for tid, meta in fold_info[src].items():
            meta["key_step"] = keystep_map.get(tid)

    # ── Load logprob data ───────────────────────────────────────────────────
    print("\nLoading logprob data...")
    logprobs_all = {}
    for source, (bench, vdir, traj1) in SOURCES.items():
        model = source.split("/")[1]
        if bench == "webarena":
            lp = load_webarena_logprobs(model, traj1)
        else:
            lp = load_mind2web_logprobs(model)
        overlap = {k: v for k, v in lp.items() if k in fold_info.get(source, {})}
        logprobs_all[source] = overlap
        print(f"  {source}: {len(overlap)} tasks with logprobs + fold info")

    # ── Build feature matrices ──────────────────────────────────────────────
    print("\nExtracting pre@k HTC features...")
    all_dfs = []
    for source in SOURCES:
        df = build_feature_matrix(source, logprobs_all[source], fold_info.get(source, {}))
        if df.empty:
            print(f"  {source}: 0 tasks — skipping")
            continue
        all_dfs.append(df)
        n_tasks = df["task_uid"].nunique()
        print(f"  {source}: {n_tasks} tasks × {df['step_k'].nunique()} k-values "
              f"= {len(df)} rows")

    feat_df = pd.concat(all_dfs, ignore_index=True)
    feat_cols = [c for c in feat_df.columns if c.startswith("f") and c[1:].isdigit()]

    # ── Cross-validation: train on folds, predict on held-out ──────────────
    print("\nRunning 5-fold CV...")
    pred_rows = []

    for source in SOURCES:
        src_df = feat_df[feat_df.source == source].copy()
        tasks  = src_df.drop_duplicates("task_uid")[["task_uid","fold","outcome"]]
        print(f"\n  {source} — {tasks['task_uid'].nunique()} tasks")

        for fold in sorted(tasks.fold.unique()):
            train_tids = tasks[tasks.fold != fold]["task_uid"].tolist()
            val_tids   = tasks[tasks.fold == fold]["task_uid"].tolist()

            train_df = src_df[src_df.task_uid.isin(train_tids)]
            val_df   = src_df[src_df.task_uid.isin(val_tids)]

            if train_df.empty or val_df.empty:
                continue

            # Need at least 2 classes in training
            if train_df["outcome"].nunique() < 2:
                continue

            # ── Pooled approach: train on ALL (task, k) rows in training fold ──
            # This gives N_tasks × avg_steps rows → much more training data
            # (same strategy as the RF pre@k baseline)
            scaler = StandardScaler()
            X_tr_all = scaler.fit_transform(train_df[feat_cols].values)
            y_tr_all = train_df["outcome"].values

            # Find best α via inner CV on pooled data
            best_alpha = best_alpha_for_fold(X_tr_all, y_tr_all, train_df["task_uid"].values)
            C_red      = 1.0 / best_alpha

            # HTC-Full (L2) — single model for all k
            mf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                    max_iter=1000, random_state=42)
            mf.fit(X_tr_all, y_tr_all)

            # HTC-Reduced (L1, best α) — single model for all k
            mr = LogisticRegression(penalty="l1", C=C_red, solver="liblinear",
                                    max_iter=1000, random_state=42)
            mr.fit(X_tr_all, y_tr_all)

            # Evaluate per k on validation fold
            for k in range(1, MAX_K + 1):
                va_k = val_df[val_df.step_k == k]
                if va_k.empty:
                    continue

                X_va = scaler.transform(va_k[feat_cols].values)
                pf   = mf.predict_proba(X_va)[:, 1]
                pr   = mr.predict_proba(X_va)[:, 1]

                for i, (_, row) in enumerate(va_k.iterrows()):
                    pred_rows.append({
                        "task_uid": row["task_uid"],
                        "source":   source,
                        "step_k":   k,
                        "outcome":  int(row["outcome"]),
                        "fold":     fold,
                        "HTC_Full_prob":    float(pf[i]),
                        "HTC_Reduced_prob": float(pr[i]),
                    })

            print(f"    fold {fold}: α_reduced={best_alpha}", end="  ", flush=True)
        print()

    pred_df = pd.DataFrame(pred_rows)
    out_pred = os.path.join(OUTPUT_DIR, "htc_prek_predictions.csv")
    pred_df.to_csv(out_pred, index=False)
    print(f"\nPredictions saved → {out_pred}")

    # ── Compute AUROC@k ─────────────────────────────────────────────────────
    print("\nComputing AUROC@k...")
    auroc_rows = []

    for source in SOURCES:
        src_pred = pred_df[pred_df.source == source]
        for k in range(1, MAX_K + 1):
            kdf = src_pred[src_pred.step_k == k]
            if kdf.empty or kdf["outcome"].nunique() < 2:
                continue
            for col, label in [("HTC_Full_prob","HTC-Full"),
                                ("HTC_Reduced_prob","HTC-Reduced")]:
                auroc = roc_auc_score(kdf["outcome"], kdf[col])
                auroc_rows.append({"source": source, "step_k": k,
                                   "method": label, "auroc": auroc})

    auroc_df = pd.DataFrame(auroc_rows)
    out_auroc = os.path.join(OUTPUT_DIR, "htc_prek_auroc_by_k.csv")
    auroc_df.to_csv(out_auroc, index=False)

    # ── Compute RF baseline on the same task subset as HTC (fair comparison) ──
    ref_rf_full  = pd.read_csv(EXISTING_PREK)
    htc_task_set = set(pred_df["task_uid"].unique())
    ref_rf_sub   = ref_rf_full[ref_rf_full["task_uid"].isin(htc_task_set)].copy()

    rf_auroc_rows = []
    for source in SOURCES:
        src_rf = ref_rf_sub[ref_rf_sub.source == source]
        for k in range(1, MAX_K + 1):
            kdf = src_rf[src_rf.step_k == k]
            if kdf.empty or kdf["outcome"].nunique() < 2:
                continue
            rf_auroc_rows.append({"source": source, "step_k": k,
                                   "method": "RF-all",
                                   "auroc": roc_auc_score(kdf["outcome"], kdf["RF_prob"])})

    rf_auroc_df = pd.DataFrame(rf_auroc_rows)

    # ── Compute overall summary ──────────────────────────────────────────────
    summary = []

    # HTC methods
    for method in ["HTC-Full", "HTC-Reduced"]:
        mdf       = auroc_df[auroc_df.method == method]
        prob_col  = "HTC_Full_prob" if method == "HTC-Full" else "HTC_Reduced_prob"
        overall   = roc_auc_score(pred_df["outcome"], pred_df[prob_col])
        mean_perk = mdf.groupby("step_k")["auroc"].mean().mean()
        per_src   = mdf.groupby("source")["auroc"].mean().to_dict()
        summary.append({
            "method":           method,
            "n_predictions":    int(len(pred_df)),
            "overall_auroc":    round(float(overall), 4),
            "mean_perk_auroc":  round(float(mean_perk), 4),
            "per_source_auroc": {s: round(v, 4) for s, v in per_src.items()},
        })

    # RF-all on same subset
    if not ref_rf_sub.empty and ref_rf_sub["outcome"].nunique() > 1:
        rf_overall   = roc_auc_score(ref_rf_sub["outcome"], ref_rf_sub["RF_prob"])
        rf_mean_perk = rf_auroc_df.groupby("step_k")["auroc"].mean().mean()
        rf_per_src   = rf_auroc_df.groupby("source")["auroc"].mean().to_dict()
        summary.insert(0, {
            "method":           "RF-all (HTC subset)",
            "n_predictions":    int(len(ref_rf_sub)),
            "overall_auroc":    round(float(rf_overall), 4),
            "mean_perk_auroc":  round(float(rf_mean_perk), 4),
            "per_source_auroc": {s: round(v, 4) for s, v in rf_per_src.items()},
        })

    out_json = os.path.join(OUTPUT_DIR, "htc_prek_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n" + "="*62)
    print("PRE@K AUROC SUMMARY  (RF on same task subset as HTC)")
    print("="*62)

    # RF full-set baselines for context
    rf_full_rows = []
    rf_sum_path = os.path.join(REPO_ROOT, "rf_prek_pooled", "summary.json")
    if os.path.isfile(rf_sum_path):
        for entry in json.load(open(rf_sum_path)):
            rf_full_rows.append({
                "method":           f"RF-{entry['feature_set']} (full)",
                "overall_auroc":    entry["overall_auroc"],
                "mean_perk_auroc":  entry["mean_perk_auroc"],
                "per_source_auroc": entry.get("per_source_auroc", {}),
            })

    show_sources = [s for s in SOURCES if s != "webarena/claude"]
    hdr = f"\n{'Method':<28} {'Overall':>8} {'MeanPre@k':>10}  " \
          + "  ".join(f"{s.split('/')[-1]:>10}" for s in show_sources)
    print(hdr)
    print("-" * (len(hdr) - 1))

    all_rows = rf_full_rows + summary
    for row in all_rows:
        src_vals = "  ".join(
            f"{row['per_source_auroc'].get(s, float('nan')):>10.4f}"
            for s in show_sources
        )
        print(f"{row['method']:<28} {row['overall_auroc']:>8.4f} "
              f"{row['mean_perk_auroc']:>10.4f}  {src_vals}")

    # ── Plot AUROC@k curve ───────────────────────────────────────────────────
    print("\nPlotting...")

    # Only plot sources with actual HTC data
    plot_sources = [s for s in SOURCES
                    if not pred_df[pred_df.source==s].empty]

    n_plot = len(plot_sources)
    fig, axes = plt.subplots(1, n_plot, figsize=(5*n_plot, 4.5), sharey=True)
    if n_plot == 1:
        axes = [axes]

    colors = {"HTC-Full": "#2196F3", "HTC-Reduced": "#F44336",
              "RF-all (subset)": "#4CAF50"}

    for ax, source in zip(axes, plot_sources):
        ax.set_title(source.replace("/", "\n"), fontsize=10)
        ax.set_xlabel("Step k")
        if ax == axes[0]:
            ax.set_ylabel("AUROC")
        ax.set_xlim(1, MAX_K)
        ax.set_ylim(0.3, 1.0)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.grid(True, alpha=0.3)

        # RF on same task subset
        rf_src = ref_rf_sub[ref_rf_sub.source == source]
        rf_k_auroc = []
        for k in range(1, MAX_K + 1):
            kdf = rf_src[rf_src.step_k == k]
            if kdf.empty or kdf["outcome"].nunique() < 2:
                continue
            rf_k_auroc.append((k, roc_auc_score(kdf["outcome"], kdf["RF_prob"])))
        if rf_k_auroc:
            ks, vs = zip(*rf_k_auroc)
            ax.plot(ks, vs, color=colors["RF-all (subset)"],
                    linestyle="--", linewidth=1.5, label="RF-all (subset)", alpha=0.8)

        # HTC variants
        src_auroc = auroc_df[auroc_df.source == source]
        for method in ["HTC-Full", "HTC-Reduced"]:
            mdf = src_auroc[src_auroc.method == method].sort_values("step_k")
            ax.plot(mdf.step_k, mdf.auroc,
                    color=colors[method], linewidth=2.0, label=method,
                    linestyle="-" if method == "HTC-Reduced" else "-.")

        ax.legend(fontsize=8)

    plt.suptitle("Pre@k AUROC: HTC features vs RF baseline (same task subset)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "htc_prek_auroc_curve.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plot saved → {out_png}")
    print(f"Summary  → {out_json}")
    print("\nDone.")


if __name__ == "__main__":
    main()
