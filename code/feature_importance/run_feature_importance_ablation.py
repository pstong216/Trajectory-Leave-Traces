#!/usr/bin/env python3
"""
run_feature_importance_ablation.py — Grouped leave-one-cluster-out ablation
for Ours (Logistic_L2, macro+micro), fully self-contained in its own
 folder (2026-08-01).

Why grouped, not per-feature: Ours uses L2 (ridge) regularization, which
famously SPLITS credit between correlated features rather than
concentrating it (unlike L1, which zeroes out redundant ones) -- L2 prefers
(0.5, 0.5) over (1, 0) for two collinear features because the L2 penalty on
the split is smaller (0.5^2+0.5^2=0.5 vs 1^2+0^2=1), even though both give
almost the same prediction. So a single feature's raw |coefficient| can look
small purely because a correlated "teammate" is sharing the credit, not
because the underlying signal is weak. Retraining with an entire correlated
CLUSTER removed (instead of reading a fixed coefficient) sidesteps this: the
model can't silently shift weight onto a held-out cluster's own members, so
the measured AUROC/e-AURC/Brier drop reflects that cluster's true marginal
contribution.

Groups come from feature_correlation_clustering.py's |r| > 0.70 hierarchical
clustering (pooled across all 10 combos, each combo z-scored on its own
steps first) -- 13 multi-feature clusters (30 of the 49 features) + 19
standalone features evaluated as their own singleton "group". See FEATURE_GROUPS
below for the exact membership and correlation_clusters.csv for the full
matrix this was derived from.

Protocol: for each of the 32 groups, for each of the 10 combos, retrain
Logistic_L2 on (49 features - that group) with the SAME standard 5-fold CV
protocol as the main table (StratifiedKFold, shuffle=True random_state=42
since 2026-08-01, task_id-grouped, task_length sample weighting, inner
C-search via fit_model()) -- i.e. the exact training code used to produce
the paper's reported Ours numbers, just imported, never reimplemented.
Importance(group) = full-model metric
MINUS leave-group-out metric, averaged across the 10 combos: a positive
AUROC delta / negative e-AURC or Brier delta means removing the group hurt
performance, i.e. the group carries real signal.

Requires reproduce.sh to have been run first (for the
in-domain reference numbers used as a full-model sanity check -- ablation
itself is trained fresh from data, no reproduction_output
dependency).

Outputs (this folder's own results/):
  feature_ablation_results.csv    -- per (group, combo): full vs leave-out metrics
  feature_ablation_summary.csv    -- per group: mean +/- std delta across 10 combos
  feature_ablation_ranking.png    -- bar chart, groups ranked by mean AUROC delta
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

PR = Path(__file__).resolve().parent.parent.parent   # 
sys.path.insert(0, str(PR / "code" / "l2_training"))
sys.path.insert(0, str(PR / "code" / "baseline_calculation"))

from train_risk_model_temp_sca import fit_model, safe_medians, aurc, aurc_oracle  # noqa: E402
from keystep_labels import load_keystep_map_entropy_source  # noqa: E402

COMBOS = [
    ("webarena", "claude"), ("webarena", "gpt_5_2"), ("webarena", "gptnano"),
    ("webarena", "kimi"), ("webarena", "qwen30b"),
    ("mind2web", "claude"), ("mind2web", "gpt_5_2"), ("mind2web", "gptnano"),
    ("mind2web", "kimi"), ("mind2web", "qwen30b"),
]

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
UQ_FEATS = [
    f"{ind}_{suf}"
    for ind in ["U_sem", "U_change", "U_ground", "U_cval", "C_S", "R_T"]
    for suf in ["at_k", "cumul_mean", "cumul_max"]
]
ALL_FEATS = MACRO_FEATS + UQ_FEATS

# ── Feature groups (|r| > 0.70 hierarchical clusters + standalone singletons) ─
# From feature_correlation_clustering.py, run 2026-08-01, pooled across all
# 10 combos. Names are descriptive shorthand for what the cluster measures,
# not the underlying feature names (see membership lists).
MULTI_CLUSTERS = {
    "trajectory_length_proxies": ["step_k", "n_distinct_actions", "n_click", "n_distinct_elements"],
    "uq_sem_confidence_cumul":   ["U_sem_cumul_mean", "U_sem_cumul_max", "C_S_cumul_mean", "C_S_cumul_max"],
    "action_stall_signals":      ["action_count_current", "steps_since_new_action"],
    "action_repetition":         ["max_action_count", "action_repeat_rate"],
    "grounding_error_loop":      ["n_grounding_errors", "grounding_loop_count"],
    "grounding_error_current":   ["consecutive_errors_now", "grounding_error_at_k"],
    "typing_volume":             ["n_type", "type_text_length"],
    "reasoning_length_cumul":    ["reasoning_len_cumul_mean", "reasoning_len_cumul_max"],
    "uq_sem_confidence_atk":     ["U_sem_at_k", "C_S_at_k"],
    "uq_change_cumul":           ["U_change_cumul_mean", "U_change_cumul_max"],
    "uq_ground_reasoning_atk":   ["U_ground_at_k", "R_T_at_k"],
    "uq_ground_reasoning_mean":  ["U_ground_cumul_mean", "R_T_cumul_mean"],
    "uq_ground_reasoning_max":   ["U_ground_cumul_max", "R_T_cumul_max"],
}
_clustered = {f for feats in MULTI_CLUSTERS.values() for f in feats}
SINGLETON_FEATS = [f for f in ALL_FEATS if f not in _clustered]
FEATURE_GROUPS = {**MULTI_CLUSTERS, **{f: [f] for f in SINGLETON_FEATS}}

assert sorted(f for feats in FEATURE_GROUPS.values() for f in feats) == sorted(ALL_FEATS), \
    "FEATURE_GROUPS must partition ALL_FEATS exactly (no feature missing or duplicated)"


def embedded_thr_base(bench: str, model: str) -> Path:
    return PR / "data" / bench / model / "embedded" / "uq_analysis_change_qwen" / "thr_0p3"


def load_combo(bench: str, model: str) -> pd.DataFrame:
    fm_path = embedded_thr_base(bench, model) / "feature_matrix.csv"
    df = pd.read_csv(fm_path)
    annot = load_keystep_map_entropy_source(f"{bench}/{model}", str(PR / "keystep_labels" / bench))
    df["label"] = df["outcome"].copy()
    for tid, ks in annot.items():
        if ks is None:
            continue
        mask = df["task_id"].astype(str) == str(tid)
        if not mask.any():
            continue
        if int(df.loc[mask, "outcome"].iloc[0]) == 0:
            continue
        df.loc[mask & (df["step_k"] < ks), "label"] = 0
        df.loc[mask & (df["step_k"] >= ks), "label"] = 1
    return df


def finalize_metrics(y_fail, p_fail) -> dict:
    yt = np.asarray(y_fail).astype(int)
    yp = np.asarray(p_fail).astype(float)
    valid = ~np.isnan(yp)
    yt, yp = yt[valid], yp[valid]
    if len(yt) < 4 or len(np.unique(yt)) < 2:
        return {}
    auroc = roc_auc_score(yt, yp)
    aurc_val = aurc(yt, yp)[0]
    aurc_oracle_val = aurc_oracle(yt)
    brier = brier_score_loss(yt, np.clip(yp, 1e-7, 1 - 1e-7))
    return {
        "n_samples": int(len(yt)),
        "auroc": round(float(auroc), 4),
        "e_aurc": round(float(aurc_val - aurc_oracle_val), 4),
        "brier": round(float(brier), 4),
    }


def run_5fold_cv(df: pd.DataFrame, feat_cols: list[str]) -> dict:
    """Standard main-table protocol: StratifiedKFold(5, shuffle=True,
    random_state=42) on task_id/outcome, task_length sample weighting,
    Logistic_L2 with inner C-search -- identical to
    train_risk_model_temp_sca.py's run_cv(), just without its
    calibration/ECE machinery (not needed for a ranking-only ablation).
    shuffle=True since 2026-08-01, matching run_cv()'s own default flip --
    see that function's docstring, and README.md's "Ours(L2)
    outer-CV protocol change" section, for the full rationale."""
    avail = [c for c in feat_cols if c in df.columns]
    task_ids = df["task_id"].unique()
    task_outcomes = df.groupby("task_id")["outcome"].first().reindex(task_ids).values
    if len(np.unique(task_outcomes)) < 2:
        return {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_y, all_p = [], []
    for tr_idx, te_idx in skf.split(task_ids, task_outcomes):
        tr_tasks, te_tasks = set(task_ids[tr_idx]), set(task_ids[te_idx])
        train_df = df[df["task_id"].isin(tr_tasks)]
        test_df = df[df["task_id"].isin(te_tasks)]
        y_tr = train_df["label"].values.astype(int)
        if len(np.unique(y_tr)) < 2:
            continue
        medians = safe_medians(train_df, avail)
        X_tr = train_df[avail].fillna(medians).fillna(0.0).values
        X_te = test_df[avail].fillna(medians).fillna(0.0).values
        step_counts = train_df.groupby("task_id")["step_k"].transform("count")
        sample_weight = (1.0 / step_counts).values
        clf = fit_model("Logistic_L2", X_tr, y_tr, sample_weight, task_ids=train_df["task_id"].values)
        y_prob = clf.predict_proba(X_te)[:, 1]
        all_y.append(test_df["label"].values.astype(int))
        all_p.append(y_prob)

    if not all_y:
        return {}
    return finalize_metrics(np.concatenate(all_y), np.concatenate(all_p))


def main():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"{len(FEATURE_GROUPS)} feature groups ({len(MULTI_CLUSTERS)} multi-feature "
          f"clusters, {len(SINGLETON_FEATS)} standalone)")
    for name, feats in FEATURE_GROUPS.items():
        print(f"  {name:28s} ({len(feats)}): {feats if len(feats) > 1 else ''}")

    print("\n" + "=" * 70)
    print("Loading all 10 combos...")
    combo_dfs = {}
    for bench, model in COMBOS:
        df = load_combo(bench, model)
        combo_dfs[f"{bench}_{model}"] = df
        print(f"  {bench}/{model:10s}  rows={len(df):5d}  tasks={df['task_id'].nunique():4d}")

    records = []

    print("\n" + "=" * 70)
    print("Full model (all 49 features) -- reference baseline per combo...")
    full_metrics = {}
    for combo, df in combo_dfs.items():
        m = run_5fold_cv(df, ALL_FEATS)
        if not m:
            print(f"  [SKIP] {combo}: insufficient data")
            continue
        full_metrics[combo] = m
        records.append({"group": "__full_model__", "n_features_removed": 0, "combo": combo, **m})
        print(f"  {combo:20s}  auroc={m['auroc']:.4f}  e_aurc={m['e_aurc']:.4f}  brier={m['brier']:.4f}")

    print("\n" + "=" * 70)
    print("Leave-one-group-out ablation...")
    for gi, (group_name, group_feats) in enumerate(FEATURE_GROUPS.items(), 1):
        remaining = [f for f in ALL_FEATS if f not in group_feats]
        print(f"\n[{gi}/{len(FEATURE_GROUPS)}] removing '{group_name}' ({len(group_feats)} features) "
              f"-> {len(remaining)} remain")
        for combo, df in combo_dfs.items():
            if combo not in full_metrics:
                continue
            m = run_5fold_cv(df, remaining)
            if not m:
                print(f"  [SKIP] {combo}: insufficient data")
                continue
            full = full_metrics[combo]
            row = {
                "group": group_name, "n_features_removed": len(group_feats), "combo": combo,
                **m,
                "delta_auroc": round(full["auroc"] - m["auroc"], 4),      # positive = removing hurt
                "delta_e_aurc": round(m["e_aurc"] - full["e_aurc"], 4),   # positive = removing hurt (e_aurc got worse/bigger)
                "delta_brier": round(m["brier"] - full["brier"], 4),      # positive = removing hurt
            }
            records.append(row)
            print(f"  {combo:20s}  auroc={m['auroc']:.4f} (Δ{row['delta_auroc']:+.4f})  "
                  f"e_aurc={m['e_aurc']:.4f} (Δ{row['delta_e_aurc']:+.4f})  "
                  f"brier={m['brier']:.4f} (Δ{row['delta_brier']:+.4f})")

    results_df = pd.DataFrame(records)
    results_df.to_csv(out_dir / "feature_ablation_results.csv", index=False)

    ablation_only = results_df[results_df.group != "__full_model__"]
    summary = ablation_only.groupby(["group", "n_features_removed"])[
        ["delta_auroc", "delta_e_aurc", "delta_brier"]
    ].agg(["mean", "std", "count"])
    summary = summary.sort_values(("delta_auroc", "mean"), ascending=False)
    summary.to_csv(out_dir / "feature_ablation_summary.csv")

    print("\n" + "=" * 70)
    print("SUMMARY (mean delta AUROC across 10 combos, removing this group hurts if positive)")
    print("=" * 70)
    print(summary.to_string())

    print("\nRendering ranking figure...")
    make_ranking_figure(ablation_only, out_dir / "feature_ablation_ranking.png")
    print(f"  saved -> {out_dir / 'feature_ablation_ranking.png'}")

    print(f"\nSaved -> {out_dir}")


def make_ranking_figure(ablation_only: pd.DataFrame, out_path: Path) -> None:
    agg = ablation_only.groupby("group")["delta_auroc"].agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(agg) + 2))
    colors = ["#2a78d6" if v >= 0 else "#c0392b" for v in agg["mean"]]
    ax.barh(agg.index, agg["mean"], xerr=agg["std"], color=colors, alpha=0.85,
            error_kw=dict(ecolor="#888888", elinewidth=1, capsize=2))
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Δ AUROC (full model − leave-group-out), mean ± std across 10 combos")
    ax.set_title("Feature-Group Importance: Leave-One-Cluster-Out Ablation\n"
                  "(Ours, Logistic_L2, macro+micro)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
