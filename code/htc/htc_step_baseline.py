from __future__ import annotations

import json
import math
import os
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

RESULTS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0,
              6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 50.0]
C_GRID = [0.01, 0.1, 1.0, 10.0]

WEBARENA_HAS_TRAJ1 = {
    "kimi": True,
    "claude": True,
    "gpt_5_2": False,
    "gptnano": False,
    "qwen30b": False,
}


def normalize_model_name(model: str) -> str:
    if model.startswith("results_") and model.endswith("_verbalized"):
        model = model[len("results_"):-len("_verbalized")]
    if "/results_" in model and model.endswith("_verbalized"):
        model = model.split("/results_", 1)[1][:-len("_verbalized")]
    for suffix in ("_v", "_log"):
        if model.endswith(suffix):
            return model[: -len(suffix)]
    return model


def load_webarena_logprobs(model: str, has_traj1: bool, base_override: str | None = None) -> dict[str, list[dict]]:
    """
    base_override lets callers point at a differently-laid-out verbalized
    dir (e.g. data/webarena/<model>/raw/verbalized, which is
    always flat -- no trajectory_1/ wrapper -- unlike the working tree,
    where RESULTS_ROOT-relative resolution + WEBARENA_HAS_TRAJ1 apply).
    Tries trajectory_1/uq_samples first when has_traj1 (or when using the
    default RESULTS_ROOT base), falls back to flat uq_samples/ either way,
    so a flat override dir resolves correctly regardless of has_traj1.
    """
    base = base_override if base_override else os.path.join(RESULTS_ROOT, "webarena", f"results_{model}_verbalized")
    candidates = []
    if has_traj1 or base_override is None:
        candidates.append(os.path.join(base, "trajectory_1", "uq_samples"))
    candidates.append(os.path.join(base, "uq_samples"))
    uq = next((c for c in candidates if os.path.isdir(c)), candidates[-1])
    result = {}
    if not os.path.isdir(uq):
        return result

    for tid in sorted(os.listdir(uq)):
        if tid.startswith("."):
            continue
        task_dir = os.path.join(uq, tid)
        if not os.path.isdir(task_dir):
            continue
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
                        "top_logprobs": s.get("top_logprobs", []),
                    })
            except (json.JSONDecodeError, OSError, IndexError):
                pass
        if steps:
            result[tid] = steps
    return result


def load_mind2web_logprobs(model: str, base_override: str | None = None) -> dict[str, list[dict]]:
    base = base_override if base_override else os.path.join(RESULTS_ROOT, "mind2web", f"results_{model}_verbalized")
    result = {}
    if not os.path.isdir(base):
        return result

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
                    "top_logprobs": sl.get("top_logprobs", []),
                })
        if steps:
            result[tid] = steps
    return result


def load_source_logprobs(source: str, base_override: str | None = None) -> dict[str, list[dict]]:
    benchmark, model = source.split("/", 1)
    base_model = normalize_model_name(model)
    if benchmark == "mind2web":
        return load_mind2web_logprobs(base_model, base_override=base_override)
    if benchmark == "webarena":
        return load_webarena_logprobs(base_model, WEBARENA_HAS_TRAJ1.get(base_model, False), base_override=base_override)
    return {}


def _step_to_conf(token_logprobs: list, top_logprobs: list) -> dict:
    conf = [math.exp(lp) for lp in token_logprobs]
    t1_avg = float(np.mean(conf)) if conf else 0.0
    tk_avgs = []
    for i, lp in enumerate(token_logprobs):
        if top_logprobs and i < len(top_logprobs) and top_logprobs[i]:
            valid = [v for v in top_logprobs[i].values() if v > -100]
            tk_avgs.append(float(np.mean([math.exp(v) for v in valid])) if valid else math.exp(lp))
        else:
            tk_avgs.append(math.exp(lp))
    return {
        "tokens": conf,
        "top1": t1_avg,
        "topk": float(np.mean(tk_avgs)) if tk_avgs else 0.0,
    }


def extract_features_prefix(steps: list, k: int) -> list[float] | None:
    prefix = steps[:k]
    if not prefix:
        return None

    cms = [_step_to_conf(s["token_logprobs"], s["top_logprobs"]) for s in prefix]
    all_tok = [c["tokens"] for c in cms]
    all_top1 = [c["top1"] for c in cms]
    all_topk = [c["topk"] for c in cms]
    all_cnts = [len(t) for t in all_tok]
    step_count = len(cms)

    if step_count > 1:
        g1 = np.diff(all_top1)
        t1gm = float(np.mean(g1)); t1gs = float(np.std(g1))
        t1gx = float(np.max(g1)); t1gn = float(np.min(g1))
        t1gt = float(g1[-1] - g1[0]) if len(g1) > 1 else 0.0
        gk = np.diff(all_topk)
        tkgm = float(np.mean(gk)); tkgs = float(np.std(gk))
        tkgx = float(np.max(gk)); tkgn = float(np.min(gk))
        tkgt = float(gk[-1] - gk[0]) if len(gk) > 1 else 0.0
    else:
        t1gm = t1gs = t1gx = t1gn = t1gt = 0.0
        tkgm = tkgs = tkgx = tkgn = tkgt = 0.0

    tok_grads = []
    for tok in all_tok:
        if len(tok) > 1:
            tok_grads.extend(np.diff(tok).tolist())
    if tok_grads:
        tgm = float(np.mean(tok_grads)); tgs = float(np.std(tok_grads))
        tgx = float(np.max(tok_grads)); tgn = float(np.min(tok_grads))
    else:
        tgm = tgs = tgx = tgn = 0.0

    H_l, K_l, R_l = [], [], []
    for tok in all_tok:
        if tok:
            r = np.array(tok); mu = np.mean(r) + 1e-8
            pi = r / (np.sum(r) + 1e-8)
            H_l.append(float(-np.sum(pi * np.log(pi + 1e-8))))
            K_l.append(float(np.max(r) / mu))
            R_l.append(float(np.std(r) / mu))
        else:
            H_l.append(0.0); K_l.append(0.0); R_l.append(0.0)

    if step_count > 1:
        spe = float(np.std(H_l)) / (float(np.mean(H_l)) + 1e-8)
        spc = float(np.std(K_l)) / (float(np.mean(K_l)) + 1e-8)
        sps = float(np.std(R_l)) / (float(np.mean(R_l)) + 1e-8)
    else:
        spe = spc = sps = 0.0

    t1_ch = float(all_top1[-1] - all_top1[0])
    tk_ch = float(all_topk[-1] - all_topk[0])

    def _pos(tok):
        if tok:
            r = np.array(tok); mu = np.mean(r); sig = np.std(r)
            H = float(-np.sum(r * np.log(r + 1e-8)))
            kap = float(np.max(r) / (mu + 1e-8))
            rho = float(sig / (mu + 1e-8))
            skw = float(np.mean(((r - mu) / (sig + 1e-8)) ** 3))
            return H, kap, rho, rho, skw
        return 0.0, 0.0, 0.0, 0.0, 0.0

    fH, fK, fS, fV, fSk = _pos(all_tok[0])
    lH, lK, lS, lV, lSk = _pos(all_tok[-1])
    ft1, ftk = float(all_top1[0]), float(all_topk[0])
    lt1, ltk = float(all_top1[-1]), float(all_topk[-1])

    aHm = float(np.mean(H_l)); aHs = float(np.std(H_l))
    aKm = float(np.mean(K_l)); aKs = float(np.std(K_l))
    aRm = float(np.mean(R_l)); aRs = float(np.std(R_l))

    vol_l, skw_l = [], []
    for tok in all_tok:
        if len(tok) > 1:
            r = np.array(tok); mu = np.mean(r); sig = np.std(r)
            vol_l.append(float(sig / (mu + 1e-8)))
            skw_l.append(float(np.mean(((r - mu) / (sig + 1e-8)) ** 3)))
    tvm = float(np.mean(vol_l)) if vol_l else 0.0
    tvs = float(np.std(vol_l)) if vol_l else 0.0
    tsm = float(np.mean(skw_l)) if skw_l else 0.0
    tss = float(np.std(skw_l)) if skw_l else 0.0

    nsc = float(step_count) / 10.0
    ftc = float(len(all_tok[0]))
    ltc = float(len(all_tok[-1]))
    atp = float(np.mean(all_cnts)) if all_cnts else 0.0
    stp = float(np.std(all_cnts)) if len(all_cnts) > 1 else 0.0

    return [
        t1gm, t1gs, t1gx, t1gn, t1gt, tkgm, tkgs, tkgx, tkgn, tkgt,
        tgm, tgs, tgx, tgn, spe, spc, sps, t1_ch, tk_ch,
        fH, fK, fS, fV, fSk, ft1, ftk,
        lH, lK, lS, lV, lSk, lt1, ltk,
        aHm, aHs, aKm, aKs, aRm, aRs, tvm, tvs, tsm, tss,
        stp, ftc, ltc, atp, nsc,
    ]


def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob > lo) & (y_prob <= hi)
        if m.sum() > 0:
            ece += abs(y_prob[m].mean() - y_true[m].mean()) * m.mean()
    return float(ece)


def _single_holdout_split(X, y, groups, seed=0):
    """One group-respecting 80%/20% train/validation split (2026-07-30: matches
    the original HTC paper's stated protocol -- arxiv 2601.15778, Appendix
    A.2.6: 'Within each fold, we use an 80%/20% split for training and
    validation', a single holdout, NOT a further k-fold inner CV. Grouped by
    task so no task's rows span both sides (same reasoning as everywhere else
    in this codebase). Implemented as the first fold of a 5-way
    StratifiedGroupKFold, which is exactly an 80/20 partition."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return next(iter(splitter.split(X, y, groups=groups)))


def best_alpha_for_fold(X_tr, y_tr, groups):
    """Grouped by task (2026-07-30, same fix as train_risk_model_temp_sca.py's
    best_alpha_htc(): X_tr/y_tr are step-level, plain StratifiedKFold let steps
    from the same task span the inner train/val split). Single 80/20 holdout,
    not 3-fold -- see _single_holdout_split()."""
    itr, iva = _single_holdout_split(X_tr, y_tr, groups)
    best_score, best_alpha = -np.inf, 1.0
    if len(np.unique(y_tr[itr])) < 2 or len(np.unique(y_tr[iva])) < 2:
        return best_alpha
    for alpha in ALPHA_GRID:
        C = 1.0 / alpha
        m = LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=1000, random_state=42)
        m.fit(X_tr[itr], y_tr[itr])
        p = m.predict_proba(X_tr[iva])[:, 1]
        score = (roc_auc_score(y_tr[iva], p) - brier_score_loss(y_tr[iva], p)
                  - compute_ece(y_tr[iva], p))
        if score > best_score:
            best_score, best_alpha = score, alpha
    return best_alpha


def best_C_for_fold(X_tr, y_tr, groups):
    """L2 counterpart of best_alpha_for_fold() (2026-07-30): M11
    (full, L2) was fixed at C=1.0 while M12 (reduced, L1) searched alpha via
    inner CV -- an unfair comparison since only one side got tuned. Same
    single 80/20 group-respecting holdout, same composite score."""
    itr, iva = _single_holdout_split(X_tr, y_tr, groups)
    best_score, best_C = -np.inf, 1.0
    if len(np.unique(y_tr[itr])) < 2 or len(np.unique(y_tr[iva])) < 2:
        return best_C
    for C in C_GRID:
        m = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=1000, random_state=42)
        m.fit(X_tr[itr], y_tr[itr])
        p = m.predict_proba(X_tr[iva])[:, 1]
        score = (roc_auc_score(y_tr[iva], p) - brier_score_loss(y_tr[iva], p)
                  - compute_ece(y_tr[iva], p))
        if score > best_score:
            best_score, best_C = score, C
    return best_C


def build_htc_step_dataset(source: str, per_step_df: pd.DataFrame, verbalized_dir_override: str | None = None) -> pd.DataFrame:
    if "success" not in per_step_df.columns or "step" not in per_step_df.columns:
        return pd.DataFrame()

    benchmark = source.split("/", 1)[0]
    logprobs = load_source_logprobs(source, base_override=verbalized_dir_override)
    if not logprobs:
        return pd.DataFrame()

    rows = []
    group_col = "task_index" if benchmark == "mind2web" and "task_index" in per_step_df.columns else "task_id"
    task_id_col = "task_id"
    if task_id_col not in per_step_df.columns:
        for alt in ("task_id_x", "task_id_y"):
            if alt in per_step_df.columns:
                task_id_col = alt
                break

    for task_key, grp in per_step_df.groupby(group_col):
        lp_steps = logprobs.get(str(task_key))
        if not lp_steps:
            continue

        grp = grp.sort_values("step").copy()
        grp["success"] = pd.to_numeric(grp["success"], errors="coerce")
        grp = grp.dropna(subset=["success"])
        if grp.empty:
            continue

        max_prefix = min(len(lp_steps), int(grp["step"].max()) + 1)
        step_map = {int(r.step): float(r.success) for r in grp.itertuples()}

        for prefix_len in range(1, max_prefix + 1):
            step = prefix_len - 1
            if step not in step_map:
                continue
            feat = extract_features_prefix(lp_steps, prefix_len)
            if feat is None:
                continue
            row = {
                "task_id": grp[task_id_col].iloc[0],
                "step": step,
                "success": step_map[step],
                "task_group": str(task_key),
                "task_final_success": float(grp["success"].iloc[-1]),
            }
            if "task_index" in grp.columns:
                row["task_index"] = grp["task_index"].iloc[0]
            row.update({f"f{i}": v for i, v in enumerate(feat)})
            rows.append(row)

    return pd.DataFrame(rows)


def compute_htc_step_scores(source: str, per_step_df: pd.DataFrame, verbalized_dir_override: str | None = None,
                             task_length_weight: bool = False) -> pd.DataFrame:
    """task_length_weight: when True, M11's final fit is weighted 1/n_steps
    per step (same "task_length" convention as M14/Ours), so every task
    counts equally regardless of length. The 80/20 C search beforehand is
    unaffected either way. Default False here; the actual pipeline
    (update_baselines_with_features.py) passes True. Only touches the
    full/L2 (M11) fit; reduced/L1 (M12) is untouched either way."""
    feat_df = build_htc_step_dataset(source, per_step_df, verbalized_dir_override=verbalized_dir_override)
    if feat_df.empty:
        return pd.DataFrame()

    feat_cols = [c for c in feat_df.columns if c.startswith("f") and c[1:].isdigit()]
    task_meta = feat_df[["task_group", "task_final_success"]].drop_duplicates().reset_index(drop=True)
    if task_meta["task_final_success"].nunique() < 2:
        return pd.DataFrame()

    full_probs = np.full(len(feat_df), np.nan, dtype=float)
    reduced_probs = np.full(len(feat_df), np.nan, dtype=float)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_task_idx, val_task_idx in cv.split(
        task_meta["task_group"].values,
        task_meta["task_final_success"].values.astype(int),
    ):
        train_groups = set(task_meta.iloc[train_task_idx]["task_group"])
        val_groups = set(task_meta.iloc[val_task_idx]["task_group"])

        train_mask = feat_df["task_group"].isin(train_groups).values
        val_mask = feat_df["task_group"].isin(val_groups).values
        train_df = feat_df.loc[train_mask]
        val_df = feat_df.loc[val_mask]
        if train_df.empty or val_df.empty:
            continue

        y_tr = train_df["success"].values.astype(int)
        if len(np.unique(y_tr)) < 2:
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_df[feat_cols].values)
        X_va = scaler.transform(val_df[feat_cols].values)

        best_C = best_C_for_fold(X_tr, y_tr, train_df["task_group"].values)
        full = LogisticRegression(penalty="l2", C=best_C, solver="lbfgs", max_iter=1000, random_state=42)
        full_sample_weight = None
        if task_length_weight:
            step_counts = train_df.groupby("task_group")["step"].transform("count")
            full_sample_weight = (1.0 / step_counts).values
        full.fit(X_tr, y_tr, sample_weight=full_sample_weight)
        full_probs[np.where(val_mask)[0]] = full.predict_proba(X_va)[:, 1]

        best_alpha = best_alpha_for_fold(X_tr, y_tr, train_df["task_group"].values)
        reduced = LogisticRegression(
            penalty="l1", C=1.0 / best_alpha, solver="liblinear", max_iter=1000, random_state=42
        )
        reduced.fit(X_tr, y_tr)
        reduced_probs[np.where(val_mask)[0]] = reduced.predict_proba(X_va)[:, 1]

    out_cols = ["task_id", "step"]
    if "task_index" in feat_df.columns:
        out_cols.append("task_index")
    out = feat_df[out_cols].copy()
    out["m11_htc_full_step"] = full_probs
    out["m12_htc_reduced_step"] = reduced_probs
    return out
