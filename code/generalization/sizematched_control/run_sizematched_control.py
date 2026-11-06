#!/usr/bin/env python3
"""
run_sizematched_control.py — Training-volume-matched control for the
website-category generalization experiment, split into its own folder
so that this volume-controlled 'cross' is not confused with the FULL-pool
cross-category numbers in run_website_generalization.py.

Question this answers: run_website_generalization.py's main result is that
cross-category (trained on the pooled OTHER 4 categories, ~1,500-1,900
tasks) beats in-domain (trained on 80% of the held-out category's own
~200-550 tasks). That comparison confounds two things: (a) does training
data from a DIFFERENT distribution actually transfer, and (b) is
cross-category just winning because it has 3-9x more training data. This
script isolates (a) by holding training-set SIZE constant: cross-category's
training pool is randomly subsampled down to the SAME size as the matched
in-domain control, 5 repeats, mean +/- std reported.

Every method that has fitted parameters gets all three numbers on the SAME
held-out-category test set:
  - full        cross-category, trained on the full pooled-other-4 pool
                (imported unmodified from run_website_generalization.py --
                NOT recomputed here, so this script can never silently
                drift from the "official" full-pool numbers)
  - sizematched cross-category, pool subsampled to match in-domain's size
  - indomain    standard 5-fold CV within the held-out category alone

M1-M7/M15 (added 2026-08-31, design decision -- same day their Brier
calibration was fixed to held-out Platt in the parent script): their
RANKING has no fitted parameters, so AUROC/e-AURC are identical across
all three protocols by construction (a built-in sanity check the output
rows preserve); their Platt calibrator, however, IS fitted on training
rows, so the Brier column gets the same three-protocol treatment --
indomain refits Platt within the held-out category (5-fold CV over its
own tasks), sizematched refits it on a pool subsample matched to the
indomain fold size (5 repeats). Table/CSV only; the figure keeps its
full-pool-only baseline cloud (already 10+ lines).

Requires reproduce.sh to have been run first (same
prerequisite as run_website_generalization.py) -- this script imports
directly from the sibling ../run_website_generalization.py module for
every data loader and every "full cross-category" number, so it can never
report a full-pool number that isn't byte-identical to what that script
itself reports.

Outputs (this folder's own results/, NOT shared with
../results/):
  sizematched_control_results.csv           -- per (method, protocol, category)
  sizematched_control_summary.csv           -- mean +/- std across the 5 categories
  website_generalization_sizematched_control.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

GEN_DIR = Path(__file__).resolve().parent.parent   # .../code/generalization/
sys.path.insert(0, str(GEN_DIR))

from run_website_generalization import (  # noqa: E402
    WEBSITE_CATEGORIES, CAT_ORDER, CAT_LABELS, BASE_FEATURE_COLS_M13, FEATURE_SETS,
    BASELINE_STYLE,
    load_all_ours, load_all_per_step_scores, load_all_htc_feat,
    run_holdout_ours, run_indomain_percat,
    run_holdout_m11, run_holdout_m13, eval_m1_m7_holdout,
    finalize_metrics, fit_model, safe_medians, best_C_for_fold,
    M1_M7_LABEL_COL, HIGHER_IS_CONFIDENT,
)


# ── Ours: size-matched cross-category ───────────────────────────────────────

def run_holdout_ours_sizematched(df: pd.DataFrame, category: str, feat_cols: list[str],
                                  n_train_target: int, n_repeats: int = 5, seed: int = 42) -> dict:
    """SAME held-out category test set and SAME pooled-other-4-categories
    training pool as run_holdout_ours(), but the training pool is randomly
    subsampled down to n_train_target tasks (matching run_indomain_percat()'s
    own average training size for this category), repeated n_repeats times."""
    avail = [c for c in feat_cols if c in df.columns]
    pool = df[df["category"] != category]
    test_df = df[df["category"] == category].copy()
    if pool.empty or test_df.empty or test_df["label"].nunique() < 2:
        return {}
    pool_task_ids = pool["global_task_id"].unique()
    if len(pool_task_ids) < n_train_target:
        return {}

    X_test_raw = test_df[avail]
    y_test = test_df["label"].values.astype(int)

    rng = np.random.default_rng(seed)
    per_repeat_auroc, per_repeat_eaurc, per_repeat_brier = [], [], []
    for rep in range(n_repeats):
        sample_tasks = set(rng.choice(pool_task_ids, size=n_train_target, replace=False))
        train_df = pool[pool["global_task_id"].isin(sample_tasks)]
        if train_df["label"].nunique() < 2:
            continue
        medians = safe_medians(train_df, avail)
        X_train = train_df[avail].fillna(medians).fillna(0.0).values
        y_train = train_df["label"].values.astype(int)
        X_test = X_test_raw.fillna(medians).fillna(0.0).values
        step_counts = train_df.groupby("global_task_id")["step_k"].transform("count")
        sample_weight = (1.0 / step_counts).values
        clf = fit_model("Logistic_L2", X_train, y_train, sample_weight,
                         task_ids=train_df["global_task_id"].values)
        y_prob = clf.predict_proba(X_test)[:, 1]
        m = finalize_metrics(y_test, y_prob)
        if not m:
            continue
        per_repeat_auroc.append(m["auroc"])
        per_repeat_eaurc.append(m["e_aurc"])
        per_repeat_brier.append(m["brier"])

    if not per_repeat_auroc:
        return {}
    return {
        "n_samples": len(y_test),
        "auroc": round(float(np.mean(per_repeat_auroc)), 4),
        "auroc_std": round(float(np.std(per_repeat_auroc)), 4),
        "e_aurc": round(float(np.mean(per_repeat_eaurc)), 4),
        "e_aurc_std": round(float(np.std(per_repeat_eaurc)), 4),
        "brier": round(float(np.mean(per_repeat_brier)), 4),
        "brier_std": round(float(np.std(per_repeat_brier)), 4),
        "n_train_tasks": n_train_target,
        "n_test_tasks": test_df["global_task_id"].nunique(),
        "n_repeats": len(per_repeat_auroc),
    }


# ── M11: in-domain + size-matched cross-category ────────────────────────────

def run_indomain_percat_m11(feat_df: pd.DataFrame, category: str) -> dict:
    """M11 counterpart of run_indomain_percat(): standard 5-fold CV WITHIN
    the held-out category's own pooled HTC rows (across whichever combos
    have logprobs), instead of holding the category out."""
    feat_cols = [c for c in feat_df.columns if c.startswith("f") and c[1:].isdigit()]
    sub = feat_df[feat_df["category"] == category].copy()
    if sub.empty:
        return {}
    task_ids = sub["global_task_group"].unique()
    task_outcome = sub.groupby("global_task_group")["task_final_success"].first().reindex(task_ids).values
    if len(np.unique(task_outcome)) < 2 or len(task_ids) < 10:
        return {}

    skf = StratifiedKFold(n_splits=5, shuffle=False)
    all_y, all_p, n_train_tasks = [], [], []
    for tr_idx, te_idx in skf.split(task_ids, task_outcome):
        tr_tasks, te_tasks = set(task_ids[tr_idx]), set(task_ids[te_idx])
        train_df = sub[sub["global_task_group"].isin(tr_tasks)]
        test_df = sub[sub["global_task_group"].isin(te_tasks)]
        y_tr = train_df["success"].values.astype(int)
        y_te = test_df["success"].values.astype(int)
        if train_df.empty or test_df.empty or len(np.unique(y_tr)) < 2:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_df[feat_cols].values)
        X_te = scaler.transform(test_df[feat_cols].values)
        groups = train_df["global_task_group"].values
        best_C = best_C_for_fold(X_tr, y_tr, groups)
        clf = LogisticRegression(penalty="l2", C=best_C, solver="lbfgs", max_iter=1000, random_state=42)
        step_counts = train_df.groupby("global_task_group")["step"].transform("count")
        clf.fit(X_tr, y_tr, sample_weight=(1.0 / step_counts).values)
        p_success = clf.predict_proba(X_te)[:, 1]
        all_y.append(1 - y_te)
        all_p.append(1.0 - p_success)
        n_train_tasks.append(train_df["global_task_group"].nunique())
    if not all_y:
        return {}
    m = finalize_metrics(np.concatenate(all_y), np.concatenate(all_p))
    if m:
        m["n_train_tasks"] = round(float(np.mean(n_train_tasks)))
        m["n_test_tasks"] = len(task_ids)
    return m


def run_holdout_m11_sizematched(feat_df: pd.DataFrame, category: str, n_train_target: int,
                                 n_repeats: int = 5, seed: int = 42) -> dict:
    feat_cols = [c for c in feat_df.columns if c.startswith("f") and c[1:].isdigit()]
    pool = feat_df[feat_df["category"] != category]
    test_df = feat_df[feat_df["category"] == category].copy()
    if pool.empty or test_df.empty:
        return {}
    y_te = test_df["success"].values.astype(int)
    if len(np.unique(y_te)) < 2:
        return {}
    pool_task_ids = pool["global_task_group"].unique()
    if len(pool_task_ids) < n_train_target:
        return {}

    rng = np.random.default_rng(seed)
    per_repeat_auroc, per_repeat_eaurc, per_repeat_brier = [], [], []
    for rep in range(n_repeats):
        sample_tasks = set(rng.choice(pool_task_ids, size=n_train_target, replace=False))
        train_df = pool[pool["global_task_group"].isin(sample_tasks)]
        y_tr = train_df["success"].values.astype(int)
        if len(np.unique(y_tr)) < 2:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_df[feat_cols].values)
        X_te = scaler.transform(test_df[feat_cols].values)
        groups = train_df["global_task_group"].values
        best_C = best_C_for_fold(X_tr, y_tr, groups)
        clf = LogisticRegression(penalty="l2", C=best_C, solver="lbfgs", max_iter=1000, random_state=42)
        step_counts = train_df.groupby("global_task_group")["step"].transform("count")
        clf.fit(X_tr, y_tr, sample_weight=(1.0 / step_counts).values)
        p_success = clf.predict_proba(X_te)[:, 1]
        m = finalize_metrics(1 - y_te, 1.0 - p_success)
        if not m:
            continue
        per_repeat_auroc.append(m["auroc"])
        per_repeat_eaurc.append(m["e_aurc"])
        per_repeat_brier.append(m["brier"])

    if not per_repeat_auroc:
        return {}
    return {
        "n_samples": len(y_te),
        "auroc": round(float(np.mean(per_repeat_auroc)), 4),
        "auroc_std": round(float(np.std(per_repeat_auroc)), 4),
        "e_aurc": round(float(np.mean(per_repeat_eaurc)), 4),
        "e_aurc_std": round(float(np.std(per_repeat_eaurc)), 4),
        "brier": round(float(np.mean(per_repeat_brier)), 4),
        "brier_std": round(float(np.std(per_repeat_brier)), 4),
        "n_train_tasks": n_train_target,
        "n_test_tasks": test_df["global_task_group"].nunique(),
        "n_repeats": len(per_repeat_auroc),
    }


# ── M13: in-domain + size-matched cross-category ────────────────────────────

def run_indomain_percat_m13(pss: pd.DataFrame, category: str) -> dict:
    """M13 counterpart of run_indomain_percat(): standard 5-fold CV WITHIN
    the held-out category's own pooled M1-M4/M7 rows, instead of holding
    the category out."""
    feat_cols = [c for c in BASE_FEATURE_COLS_M13 if c in pss.columns and pss[c].notna().sum() > 0]
    if len(feat_cols) < 2:
        return {}
    # A handful of tasks (e.g. webarena_claude::112 in devtools) have
    # success=NaN for every step -- missing ground truth, not a keyStep
    # relabeling artifact. build_htc_step_dataset() already drops these for
    # the M11 path; the M13 path reads per_step_scores.csv directly and
    # needs the same drop here (StratifiedKFold rejects NaN in y outright).
    sub = pss[pss[feat_cols].notna().any(axis=1) & (pss["category"] == category)
              & pss["success"].notna()].copy()
    if sub.empty:
        return {}
    task_ids = sub["global_task_group"].unique()
    task_outcome = (sub.sort_values("step").groupby("global_task_group")["success"]
                     .last().reindex(task_ids).values)
    if len(np.unique(task_outcome)) < 2 or len(task_ids) < 10:
        return {}

    skf = StratifiedKFold(n_splits=5, shuffle=False)
    all_y, all_p, n_train_tasks = [], [], []
    for tr_idx, te_idx in skf.split(task_ids, task_outcome):
        tr_tasks, te_tasks = set(task_ids[tr_idx]), set(task_ids[te_idx])
        train_df = sub[sub["global_task_group"].isin(tr_tasks)]
        test_df = sub[sub["global_task_group"].isin(te_tasks)]
        y_tr = train_df["success"].values.astype(int)
        y_te = test_df["success"].values.astype(int)
        if train_df.empty or test_df.empty or len(np.unique(y_tr)) < 2:
            continue
        medians = train_df[feat_cols].median()
        X_tr = train_df[feat_cols].fillna(medians).fillna(0.0).values
        X_te = test_df[feat_cols].fillna(medians).fillna(0.0).values
        scaler = StandardScaler()
        X_trs = scaler.fit_transform(X_tr)
        X_tes = scaler.transform(X_te)
        groups = train_df["global_task_group"].values
        best_C = best_C_for_fold(X_trs, y_tr, groups)
        clf = LogisticRegression(penalty="l2", C=best_C, solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X_trs, y_tr)
        p_success = clf.predict_proba(X_tes)[:, 1]
        all_y.append(1 - y_te)
        all_p.append(1.0 - p_success)
        n_train_tasks.append(train_df["global_task_group"].nunique())
    if not all_y:
        return {}
    m = finalize_metrics(np.concatenate(all_y), np.concatenate(all_p))
    if m:
        m["n_train_tasks"] = round(float(np.mean(n_train_tasks)))
        m["n_test_tasks"] = len(task_ids)
    return m


def run_holdout_m13_sizematched(pss: pd.DataFrame, category: str, n_train_target: int,
                                 n_repeats: int = 5, seed: int = 42) -> dict:
    feat_cols = [c for c in BASE_FEATURE_COLS_M13 if c in pss.columns and pss[c].notna().sum() > 0]
    if len(feat_cols) < 2:
        return {}
    valid = pss[pss[feat_cols].notna().any(axis=1) & pss["success"].notna()]
    pool = valid[valid["category"] != category]
    test_df = valid[valid["category"] == category].copy()
    if pool.empty or test_df.empty:
        return {}
    y_te = test_df["success"].values.astype(int)
    if len(np.unique(y_te)) < 2:
        return {}
    pool_task_ids = pool["global_task_group"].unique()
    if len(pool_task_ids) < n_train_target:
        return {}

    rng = np.random.default_rng(seed)
    per_repeat_auroc, per_repeat_eaurc, per_repeat_brier = [], [], []
    for rep in range(n_repeats):
        sample_tasks = set(rng.choice(pool_task_ids, size=n_train_target, replace=False))
        train_df = pool[pool["global_task_group"].isin(sample_tasks)]
        y_tr = train_df["success"].values.astype(int)
        if len(np.unique(y_tr)) < 2:
            continue
        medians = train_df[feat_cols].median()
        X_tr = train_df[feat_cols].fillna(medians).fillna(0.0).values
        X_te = test_df[feat_cols].fillna(medians).fillna(0.0).values
        scaler = StandardScaler()
        X_trs = scaler.fit_transform(X_tr)
        X_tes = scaler.transform(X_te)
        groups = train_df["global_task_group"].values
        best_C = best_C_for_fold(X_trs, y_tr, groups)
        clf = LogisticRegression(penalty="l2", C=best_C, solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X_trs, y_tr)
        p_success = clf.predict_proba(X_tes)[:, 1]
        m = finalize_metrics(1 - y_te, 1.0 - p_success)
        if not m:
            continue
        per_repeat_auroc.append(m["auroc"])
        per_repeat_eaurc.append(m["e_aurc"])
        per_repeat_brier.append(m["brier"])

    if not per_repeat_auroc:
        return {}
    return {
        "n_samples": len(y_te),
        "auroc": round(float(np.mean(per_repeat_auroc)), 4),
        "auroc_std": round(float(np.std(per_repeat_auroc)), 4),
        "e_aurc": round(float(np.mean(per_repeat_eaurc)), 4),
        "e_aurc_std": round(float(np.std(per_repeat_eaurc)), 4),
        "brier": round(float(np.mean(per_repeat_brier)), 4),
        "brier_std": round(float(np.std(per_repeat_brier)), 4),
        "n_train_tasks": n_train_target,
        "n_test_tasks": test_df["global_task_group"].nunique(),
        "n_repeats": len(per_repeat_auroc),
    }


# ── M1-M7/M15: Brier-only Platt-calibration protocol variants ─────────────

def run_m1m7_platt_variants(pss: pd.DataFrame, category: str,
                            n_repeats: int = 5, seed: int = 42) -> list[dict]:
    """Brier-only in-domain / size-matched variants for M1-M7/M15
    (2026-08-31). The raw score is a fixed function of the trajectory, so
    AUROC/e-AURC cannot move across protocols (ranking always uses the raw
    oriented score, mirroring eval_m1_m7_holdout()); the only fitted
    component is the 1-D Platt calibrator behind Brier, and THAT is what
    each protocol refits:
      indomain     StratifiedKFold(5) over the held-out category's own
                   tasks: Platt fit on 4/5 of tasks, predicted on 1/5,
                   predictions pooled (mirrors run_indomain_percat_m13's
                   split machinery).
      sizematched  Platt fit on a random n_train_target-task subsample of
                   the pooled other-4-categories rows (n_train_target =
                   this method's own indomain mean train-fold size),
                   predicted on the whole held-out category; n_repeats
                   subsamples, mean +/- std.
    The full-pool row is NOT recomputed here -- it stays imported from
    eval_m1_m7_holdout(), same no-drift rule as every other method."""
    records = []
    for col, label_col in M1_M7_LABEL_COL.items():
        if col not in pss.columns:
            continue
        sub = pss[[label_col, col, "category", "global_task_group"]].dropna()
        test = sub[sub["category"] == category]
        pool = sub[sub["category"] != category]
        if len(test) < 4 or test[label_col].nunique() < 2 or pool.empty:
            continue

        def _conf(frame):
            raw = frame[col].values.astype(float)
            return raw if HIGHER_IS_CONFIDENT[col] else -raw

        test_conf = _conf(test)
        y_te_success = test[label_col].values.astype(int)

        def _platt(train_conf, y_train_success):
            if len(np.unique(train_conf)) < 2 or len(np.unique(y_train_success)) < 2:
                return None
            m = LogisticRegression(solver="lbfgs", max_iter=1000)
            m.fit(train_conf.reshape(-1, 1), y_train_success)
            return m

        # ---- indomain: 5-fold CV over the category's own tasks ----
        task_ids = test["global_task_group"].unique()
        task_outcome = (test.groupby("global_task_group")[label_col]
                            .last().reindex(task_ids).values)
        indom = {}
        if len(task_ids) >= 10 and len(np.unique(task_outcome)) >= 2:
            skf = StratifiedKFold(n_splits=5, shuffle=False)
            all_y, all_rank, all_pb, n_tr_tasks = [], [], [], []
            for tr_idx, te_idx in skf.split(task_ids, task_outcome):
                tr_tasks, te_tasks = set(task_ids[tr_idx]), set(task_ids[te_idx])
                tr_df = test[test["global_task_group"].isin(tr_tasks)]
                te_df = test[test["global_task_group"].isin(te_tasks)]
                if tr_df.empty or te_df.empty:
                    continue
                platt = _platt(_conf(tr_df), tr_df[label_col].values.astype(int))
                if platt is None:
                    continue
                conf_te = _conf(te_df)
                p_succ = np.clip(platt.predict_proba(conf_te.reshape(-1, 1))[:, 1],
                                 1e-7, 1 - 1e-7)
                all_y.append(1 - te_df[label_col].values.astype(int))
                all_rank.append(-conf_te)
                all_pb.append(1.0 - p_succ)
                n_tr_tasks.append(tr_df["global_task_group"].nunique())
            if all_y:
                indom = finalize_metrics(np.concatenate(all_y), np.concatenate(all_rank),
                                         p_fail_brier=np.concatenate(all_pb))
            if indom:
                indom["method"] = f"{col}_indomain_percat"
                indom["category"], indom["protocol"] = category, "indomain"
                indom["n_train_tasks"] = round(float(np.mean(n_tr_tasks)))
                indom["n_test_tasks"] = len(task_ids)
                records.append(indom)
        if not indom:
            continue  # no matched size target -> skip sizematched too

        # ---- sizematched: Platt on a size-matched pool subsample ----
        n_train_target = indom["n_train_tasks"]
        pool_task_ids = pool["global_task_group"].unique()
        if len(pool_task_ids) < n_train_target:
            continue
        rng = np.random.default_rng(seed)
        briers, aurocs, eaurcs = [], [], []
        for rep_i in range(n_repeats):
            sample = set(rng.choice(pool_task_ids, size=n_train_target, replace=False))
            tr_df = pool[pool["global_task_group"].isin(sample)]
            platt = _platt(_conf(tr_df), tr_df[label_col].values.astype(int))
            if platt is None:
                continue
            p_succ = np.clip(platt.predict_proba(test_conf.reshape(-1, 1))[:, 1],
                             1e-7, 1 - 1e-7)
            m = finalize_metrics(1 - y_te_success, -test_conf,
                                 p_fail_brier=1.0 - p_succ)
            if not m:
                continue
            briers.append(m["brier"]); aurocs.append(m["auroc"]); eaurcs.append(m["e_aurc"])
        if briers:
            records.append({
                "n_samples": len(y_te_success),
                "auroc": round(float(np.mean(aurocs)), 4),   # constant across repeats
                "auroc_std": round(float(np.std(aurocs)), 4),
                "e_aurc": round(float(np.mean(eaurcs)), 4),  # constant across repeats
                "e_aurc_std": round(float(np.std(eaurcs)), 4),
                "brier": round(float(np.mean(briers)), 4),
                "brier_std": round(float(np.std(briers)), 4),
                "method": f"{col}_cross_sizematched",
                "category": category, "protocol": "sizematched",
                "n_train_tasks": n_train_target,
                "n_test_tasks": test["global_task_group"].nunique(),
                "n_repeats": len(briers),
            })
    return records


# ── Figure ───────────────────────────────────────────────────────────────

# One method = one color (consistent with the main website_generalization_
# figure.png's Ours-family palette); one protocol = one linestyle/marker,
# shared across every method. All 5 methods overlaid in a single 1x3-panel
# figure (same single-figure treatment as ../results/
# website_generalization_figure.png). After review:
# this figure is specifically the size-matched control, so it shows ONLY
# the size-matched cross-category line per method (not the full-pool
# line -- that's what ../results/website_generalization_figure.png is
# for) against its matched in-domain counterpart; in-domain is shown only
# for Ours macro+micro (macro-only/micro-only in-domain lines dropped --
# M11/M13 have no feature-set variants, so their in-domain line stays).
METHOD_STYLE = {
    "ours_l2_all":        dict(color="#d1372b", label="Ours — macro+micro"),
    "ours_l2_macro_only": dict(color="#f0876e", label="Ours — macro only"),
    "ours_l2_uq_only":    dict(color="#8f2a3a", label="Ours — micro/UQ only"),
    "m11_htc_full_step":  dict(color="#e0821e", label="HTC full"),
    "m13_stacked_htc_l2": dict(color="#6b6560", label="Stacked L2"),
}
PROTOCOL_STYLE = {
    "sizematched": dict(ls="--", marker="s", suffix=""),
    "indomain":    dict(ls=":",  marker="o", suffix="(in-domain)"),
}
# in-domain shown only for this one method; sizematched shown for all 5
INDOMAIN_METHODS = {"ours_l2_all"}


def make_sizematched_control_figure(results_df: pd.DataFrame, out_path: Path) -> None:
    metrics = [("auroc", "AUROC ↑"), ("e_aurc", "e-AURC ↓"), ("brier", "Brier ↓")]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))

    def series_for(method: str, metric: str) -> np.ndarray | None:
        sub = results_df[results_df.method == method].set_index("category")
        if sub.empty:
            return None
        return np.array([sub.loc[c, metric] if c in sub.index else np.nan for c in CAT_ORDER], dtype=float)

    for ax, (mkey, mtitle) in zip(axes, metrics):
        # M1-M7/M15 muted baseline cloud. 2026-08-31 (caught in review): these
        # now plot their SIZE-MATCHED variant, like every other line in this
        # figure -- previously they plotted the full-pool number, which made
        # this "size-matched control" figure silently mix two protocols. The
        # size-matched Platt rows come from run_m1m7_platt_variants(); the
        # bare-name (full-pool) fallback stays for any method lacking one.
        # Note AUROC/e-AURC are bit-identical across protocols for these
        # methods (fixed ranking), so only the Brier panel moves, by
        # <=0.002 -- correctness of labelling, not a visible change.
        for method, bstyle in BASELINE_STYLE.items():
            if method in ("m13_stacked_htc_l2", "m2_action_logprob_mean",
                          "m3_output_logprob_mean", "m4_cumulative_output_logprob_mean"):
                continue  # M13 has its own lines below; M2/M3/M4 dropped by design
            vals = series_for(f"{method}_cross_sizematched", mkey)
            if vals is None or np.all(np.isnan(vals)):
                vals = series_for(method, mkey)
            if vals is None or np.all(np.isnan(vals)):
                continue
            ax.plot(range(5), vals, color=bstyle["color"], marker=bstyle["marker"],
                     linestyle=bstyle["ls"], linewidth=1.4, markersize=6, alpha=0.8,
                     label=bstyle["label"], zorder=1)

        for base_method, mstyle in METHOD_STYLE.items():
            variants = {"sizematched": f"{base_method}_cross_sizematched"}
            if base_method in INDOMAIN_METHODS:
                variants["indomain"] = f"{base_method}_indomain_percat"
            for proto, method_name in variants.items():
                vals = series_for(method_name, mkey)
                if vals is None or np.all(np.isnan(vals)):
                    continue
                pstyle = PROTOCOL_STYLE[proto]
                is_hero = base_method == "ours_l2_all" and proto == "sizematched"
                ax.plot(range(5), vals, color=mstyle["color"], marker=pstyle["marker"],
                         linestyle=pstyle["ls"], linewidth=3.0 if is_hero else 1.8,
                         markersize=8 if is_hero else 6, alpha=1.0 if is_hero else 0.85,
                         label=f"{mstyle['label']} {pstyle['suffix']}",
                         zorder=6 if is_hero else 3)
                if is_hero:
                    for x, v in enumerate(vals):
                        if np.isnan(v):
                            continue
                        va = "bottom" if mkey == "auroc" else "top"
                        ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points",
                                    xytext=(0, 14 if mkey == "auroc" else -14), ha="center", va=va,
                                    fontsize=10, fontweight="bold", color=mstyle["color"],
                                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                              edgecolor="none", alpha=0.8),
                                    zorder=10)

        if mkey != "auroc":
            # e-AURC/Brier: the hero (best) line sits near the axis bottom, so
            # its below-point annotations need headroom or they collide with
            # the rotated x-tick labels underneath.
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin - 0.14 * (ymax - ymin), ymax)

        ax.set_title(mtitle, fontsize=15, fontweight="bold")
        ax.set_xticks(range(5))
        ax.set_xticklabels(CAT_LABELS, rotation=18, ha="right", fontsize=10.5)
        ax.set_xlabel("Website Category", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    # group by method (Ours macro+micro first), protocol order full/sizematched/indomain within each
    method_order = list(METHOD_STYLE)
    proto_order = list(PROTOCOL_STYLE)

    def sort_key(i):
        lbl = labels[i]
        m_idx = next((mi for mi, m in enumerate(method_order)
                      if METHOD_STYLE[m]["label"] in lbl), len(method_order))
        p_idx = next((pi for pi, p in enumerate(proto_order)
                      if PROTOCOL_STYLE[p]["suffix"] in lbl), len(proto_order))
        return (m_idx, p_idx)

    order = sorted(range(len(labels)), key=sort_key)
    leg = fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=5, fontsize=13, frameon=True,
               bbox_to_anchor=(0.5, -0.32), markerscale=1.8, handlelength=3.2,
               handletextpad=0.6, columnspacing=1.4, labelspacing=0.8,
               title="Method (dashed=Ours/HTC/Stacked, thin muted=top-ranked baselines -- ALL cross-category size-matched; dotted=in-domain)")
    leg.get_title().set_fontsize(13.5)

    # fig.suptitle("Training-Volume-Matched Control: Cross-Category vs. In-Domain",
    #              fontsize=17, fontweight="bold", y=1.02)
    # fig.text(0.5, 0.965, "(website_generalization · cross-category trained on the SAME sample size as "
    #           "in-domain · Ours x3 feature sets, M11, M13, M1-M7 baselines)",
    #           ha="center", fontsize=11.5, color="#555555")
    # No bottom rect reservation: the legend lives outside the axes (negative
    # bbox_to_anchor y) via fig.legend, so keep the 3 panels at their natural
    # tight_layout size and let bbox_inches="tight" on savefig grow the canvas
    # downward to fit the legend instead of shrinking the panels to fit it.
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("Loading Ours + category maps (via run_website_generalization)...")
    ours_df, cat_maps = load_all_ours()
    print("Loading per_step_scores.csv (M13 features)...")
    pss = load_all_per_step_scores(cat_maps)
    print("Loading HTC feature dataset (M11 features)...")
    htc_df = load_all_htc_feat(cat_maps)

    records = []

    # ---- M1-M7/M15: full row imported from eval_m1_m7_holdout() (ranking
    # fixed, Brier via held-out Platt on the full pool), plus Brier-only
    # in-domain/size-matched Platt variants (2026-08-31, see
    # run_m1m7_platt_variants()'s docstring) ----
    print(f"\n{'='*70}\nM1-M7/M15 (fixed ranking; Platt-calibration protocol variants)\n{'='*70}")
    for category in WEBSITE_CATEGORIES:
        for r in eval_m1_m7_holdout(pss, category):
            r["protocol"] = "full"
            records.append(r)
        for r in run_m1m7_platt_variants(pss, category):
            records.append(r)
            if r["protocol"] == "sizematched":
                print(f"  {r['method'][:-18]:36s} {category:18s} n_train={r['n_train_tasks']:4d}  "
                      f"brier sm={r['brier']:.4f}±{r['brier_std']:.4f}")

    # ---- Ours ----
    for fs_name, feat_cols in FEATURE_SETS.items():
        print(f"\n{'='*70}\nOurs / {fs_name}\n{'='*70}")
        indomain_train_size = {}
        for category in WEBSITE_CATEGORIES:
            full = run_holdout_ours(ours_df, category, feat_cols)
            if full:
                full["method"], full["category"], full["protocol"] = f"ours_l2_{fs_name}", category, "full"
                records.append(full)

            indom = run_indomain_percat(ours_df, category, feat_cols)
            if not indom:
                print(f"  [SKIP] {category}: in-domain insufficient data")
                continue
            indom["method"] = f"ours_l2_{fs_name}_indomain_percat"
            indom["category"], indom["protocol"] = category, "indomain"
            records.append(indom)
            indomain_train_size[category] = indom["n_train_tasks"]

            sm = run_holdout_ours_sizematched(ours_df, category, feat_cols, indom["n_train_tasks"])
            if not sm:
                print(f"  [SKIP] {category}: size-matched insufficient data")
                continue
            sm["method"], sm["category"], sm["protocol"] = f"ours_l2_{fs_name}_cross_sizematched", category, "sizematched"
            records.append(sm)
            print(f"  {category:18s} n_train={indom['n_train_tasks']:4d}  "
                  f"full_auroc={full.get('auroc', float('nan')):.4f}  "
                  f"sizematched_auroc={sm['auroc']:.4f}±{sm['auroc_std']:.4f}  "
                  f"indomain_auroc={indom['auroc']:.4f}")

    # ---- M11 ----
    if not htc_df.empty:
        print(f"\n{'='*70}\nM11 (HTC full/L2)\n{'='*70}")
        for category in WEBSITE_CATEGORIES:
            full = run_holdout_m11(htc_df, category)
            if full:
                full["method"], full["category"], full["protocol"] = "m11_htc_full_step", category, "full"
                records.append(full)

            indom = run_indomain_percat_m11(htc_df, category)
            if not indom:
                print(f"  [SKIP] {category}: in-domain insufficient data")
                continue
            indom["method"] = "m11_htc_full_step_indomain_percat"
            indom["category"], indom["protocol"] = category, "indomain"
            records.append(indom)

            sm = run_holdout_m11_sizematched(htc_df, category, indom["n_train_tasks"])
            if not sm:
                print(f"  [SKIP] {category}: size-matched insufficient data")
                continue
            sm["method"], sm["category"], sm["protocol"] = "m11_htc_full_step_cross_sizematched", category, "sizematched"
            records.append(sm)
            print(f"  {category:18s} n_train={indom['n_train_tasks']:4d}  "
                  f"full_auroc={full.get('auroc', float('nan')):.4f}  "
                  f"sizematched_auroc={sm['auroc']:.4f}±{sm['auroc_std']:.4f}  "
                  f"indomain_auroc={indom['auroc']:.4f}")
    else:
        print("\n[SKIP] M11: HTC feature dataset empty (no logprobs)")

    # ---- M13 ----
    print(f"\n{'='*70}\nM13 (stacked HTC_L2)\n{'='*70}")
    for category in WEBSITE_CATEGORIES:
        full = run_holdout_m13(pss, category)
        if full:
            full["method"], full["category"], full["protocol"] = "m13_stacked_htc_l2", category, "full"
            records.append(full)

        indom = run_indomain_percat_m13(pss, category)
        if not indom:
            print(f"  [SKIP] {category}: in-domain insufficient data")
            continue
        indom["method"] = "m13_stacked_htc_l2_indomain_percat"
        indom["category"], indom["protocol"] = category, "indomain"
        records.append(indom)

        sm = run_holdout_m13_sizematched(pss, category, indom["n_train_tasks"])
        if not sm:
            print(f"  [SKIP] {category}: size-matched insufficient data")
            continue
        sm["method"], sm["category"], sm["protocol"] = "m13_stacked_htc_l2_cross_sizematched", category, "sizematched"
        records.append(sm)
        print(f"  {category:18s} n_train={indom['n_train_tasks']:4d}  "
              f"full_auroc={full.get('auroc', float('nan')):.4f}  "
              f"sizematched_auroc={sm['auroc']:.4f}±{sm['auroc_std']:.4f}  "
              f"indomain_auroc={indom['auroc']:.4f}")

    results_df = pd.DataFrame(records)
    results_df.to_csv(out_dir / "sizematched_control_results.csv", index=False)

    summary = results_df.groupby("method")[["auroc", "e_aurc", "brier"]].agg(["mean", "std", "count"])
    summary.to_csv(out_dir / "sizematched_control_summary.csv")
    print(f"\n{'='*70}\nSUMMARY (mean +/- std across 5 held-out categories)\n{'='*70}")
    print(summary.to_string())

    print("\nRendering figure...")
    fig_path = out_dir / "website_generalization_sizematched_control.png"
    make_sizematched_control_figure(results_df, fig_path)
    print(f"  saved -> {fig_path}")

    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
