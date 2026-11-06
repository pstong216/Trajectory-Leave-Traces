#!/usr/bin/env python3
"""
run_macro_internal.py -- Macro (31 feat.) + HTC "internal-state" (48 feat.)
features, BOTH extracted from the SAME trajectories (the baseline/
verbalized rollout -- M1-M7/M11's own source), trained with the same
Logistic_L2 protocol as Ours.

Why this exists: an earlier variant (not released) combined Macro/Micro (Ours' own
rollout) with HTC (verbalized rollout) -- two INDEPENDENT samples of the
same task, requiring an inner join that lost 30-65% of the rows. This is
avoidable for Macro specifically: macro features are a
pure function of the action sequence (+ the model's own reasoning text),
so nothing stops us from computing macro features FROM the verbalized
rollout directly, on the SAME trajectories HTC's logprobs come from. Micro
(the Qwen-embedding UQ features) can't be recomputed this way -- it needs
the Qwen re-embedding pipeline run against this rollout, which hasn't been
done -- so this experiment is Macro + Internal only, not Macro+Micro+Internal.

Feature extraction reuses build_feature_matrix.py's own functions
UNCHANGED (extract_macro_webarena/mind2web, thought_feats_at_k,
combined_error_feats_at_k, reasoning_len_feats_at_k) -- only the
ORCHESTRATION differs from that file's own build_feature_matrix(), for two
reasons:
  1. data/<bench>/<model>/raw/{plain,verbalized}/ is a
     FLATTENED copy (actions/, uq_samples/ directly under it) -- the
     original build_feature_matrix()'s webarena branch hardcodes
     base_dir/"trajectory_1"/"actions" etc, which doesn't exist here.
     Mind2Web's branch has no trajectory_1 reference at all (task dirs
     sit directly under base_dir already), so it would have worked
     unchanged -- webarena is the one that needs the path fix.
  2. Micro (extract_uq_at_k/load_step_uq) is skipped entirely -- no UQ
     directory exists for the verbalized rollout.

Labels/population: taken from reproduction_output/baseline/<bench>/<model>/
results/per_step_scores.csv's own "success" column, NOT re-derived from
raw score files here. per_step_scores.csv's "success" is already
keystep-relabeled correctly for the BASELINE role
(compute_baselines.py's build_per_step_df -> label_step_from_keystep,
using <model>_baseline.json, not <model>_ours.json -- using the wrong
keystep file was exactly the historical bug this repo's README warns
about at length). Reusing that already-correct column, rather than
re-deriving keystep-relabeling here, avoids that entire class of mistake.
Macro (this script) and HTC (build_htc_step_dataset) both inner-join onto
this same (task, step) skeleton -- since all three are now sourced from
the SAME verbalized rollout, retention should be much higher than the
Macro/Micro-vs-HTC join in that earlier variant (verified below, not
assumed).

Outputs (this folder's own results/):
  macro_internal_per_combo.csv    per (combo, feature_set): auroc/e_aurc/brier/n_rows/n_tasks
  macro_internal_summary.csv      mean +/- std across usable combos
  macro_internal_significance.csv paired t-test/Wilcoxon
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold

PR = Path(__file__).resolve().parent.parent.parent  # 
sys.path.insert(0, str(PR / "code" / "l2_training"))
sys.path.insert(0, str(PR / "code" / "htc"))
sys.path.insert(0, str(PR / "code" / "feature_importance"))
from train_risk_model_temp_sca import fit_model, safe_medians  # noqa: E402
from htc_step_baseline import build_htc_step_dataset  # noqa: E402
from run_feature_importance_ablation import MACRO_FEATS, finalize_metrics  # noqa: E402
from build_feature_matrix import (  # noqa: E402
    extract_macro_webarena, extract_macro_mind2web,
    thought_feats_at_k, combined_error_feats_at_k, reasoning_len_feats_at_k,
    load_thought_texts_webarena, load_reasoning_lengths_webarena,
    load_thought_texts_mind2web, load_reasoning_lengths_mind2web,
    _wa_is_grounding, _wa_is_parse_err, _m2w_is_parse_err,
    STOP_RE, _M2W_STOP_RE,
)

REPRO = PR / "reproduction_output"

COMBOS = [
    ("webarena", "claude"), ("webarena", "gpt_5_2"), ("webarena", "gptnano"),
    ("webarena", "kimi"), ("webarena", "qwen30b"),
    ("mind2web", "claude"), ("mind2web", "gpt_5_2"), ("mind2web", "gptnano"),
    ("mind2web", "kimi"), ("mind2web", "qwen30b"),
]
HTC_FEATS = [f"f{i}" for i in range(48)]


# ── Macro extraction from the FLATTENED verbalized rollout ─────────────────

def build_macro_webarena(base_dir: Path) -> pd.DataFrame:
    actions_dir = base_dir / "actions"
    uq_samples_dir = base_dir / "uq_samples"
    rows = []
    if not actions_dir.exists():
        return pd.DataFrame()
    for act_file in sorted(actions_dir.glob("*.json"), key=lambda p: int(p.stem)):
        task_id = int(act_file.stem)
        data = json.loads(act_file.read_text())
        actions = [a for a in data.get("actions", []) if not STOP_RE.match(a)]
        if not actions:
            continue
        rlens = load_reasoning_lengths_webarena(uq_samples_dir, task_id, len(actions))
        ttexts = load_thought_texts_webarena(uq_samples_dir, task_id, len(actions))
        explicit_grounding = [_wa_is_grounding(a) for a in actions]
        parse_flags = [_wa_is_parse_err(a) for a in actions]
        for k in range(len(actions)):
            row = {"task_id": task_id}
            row.update(extract_macro_webarena(actions, k))
            row.update(thought_feats_at_k(ttexts, k))
            row.update(combined_error_feats_at_k(parse_flags, explicit_grounding, ttexts, k))
            row.update(reasoning_len_feats_at_k(rlens, k))
            rows.append(row)
    return pd.DataFrame(rows)


def build_macro_mind2web(base_dir: Path) -> pd.DataFrame:
    rows = []
    task_dirs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    for task_dir in sorted(task_dirs, key=lambda p: int(p.name)):
        task_index = int(task_dir.name)
        result_file = task_dir / "result.json"
        if not result_file.exists():
            continue
        d = json.loads(result_file.read_text())
        actions = [a for a in d.get("action_history", []) if not _M2W_STOP_RE.match(a)]
        if not actions:
            continue
        rlens = load_reasoning_lengths_mind2web(base_dir, task_index, len(actions))
        ttexts = load_thought_texts_mind2web(base_dir, task_index, len(actions))
        explicit_grounding = [False] * len(actions)
        parse_flags = [_m2w_is_parse_err(a) for a in actions]
        for k in range(len(actions)):
            row = {"task_id": task_index}
            row.update(extract_macro_mind2web(actions, k))
            row.update(thought_feats_at_k(ttexts, k))
            row.update(combined_error_feats_at_k(parse_flags, explicit_grounding, ttexts, k))
            row.update(reasoning_len_feats_at_k(rlens, k))
            rows.append(row)
    return pd.DataFrame(rows)


def load_macro_from_verbalized(bench: str, model: str) -> pd.DataFrame:
    base_dir = PR / "data" / bench / model / "raw" / "verbalized"
    if not base_dir.exists():
        return pd.DataFrame()
    if bench == "webarena":
        return build_macro_webarena(base_dir)
    return build_macro_mind2web(base_dir)


def load_htc(bench: str, model: str) -> pd.DataFrame | None:
    pss_path = REPRO / "baseline" / bench / model / "results" / "per_step_scores.csv"
    if not pss_path.is_file():
        return None
    per_step = pd.read_csv(pss_path)
    verbalized_dir = str(PR / "data" / bench / model / "raw" / "verbalized")
    htc = build_htc_step_dataset(f"{bench}/{model}", per_step, verbalized_dir_override=verbalized_dir)
    return htc if not htc.empty else None


def load_labels(bench: str, model: str) -> pd.DataFrame:
    """(task, step) -> label (canonical y_fail=1) from per_step_scores.csv's
    own already-keystep-relabeled "success" column (BASELINE role -- this
    file is built from <model>_baseline.json, the correct annotation for
    this rollout, never re-derived here)."""
    pss_path = REPRO / "baseline" / bench / model / "results" / "per_step_scores.csv"
    df = pd.read_csv(pss_path)
    id_col = "task_index" if (bench == "mind2web" and "task_index" in df.columns) else "task_id"
    out = df[[id_col, "step", "success"]].dropna().copy()
    out["join_id"] = out[id_col].astype(str)
    out["label"] = 1 - out["success"].astype(int)
    out["outcome"] = out["label"]  # task-level outcome == last step's label; used for fold stratification
    return out[["join_id", "step", "label", "outcome"]]


def build_joined(bench: str, model: str) -> tuple[pd.DataFrame, dict]:
    macro = load_macro_from_verbalized(bench, model)
    htc = load_htc(bench, model)
    labels = load_labels(bench, model)
    stats_d = {"macro_rows": len(macro), "macro_tasks": macro["task_id"].nunique() if not macro.empty else 0}
    if macro.empty or htc is None:
        stats_d.update(htc_rows=0, htc_tasks=0, joined_rows=0, joined_tasks=0)
        return pd.DataFrame(), stats_d

    htc_id_col = "task_index" if (bench == "mind2web" and "task_index" in htc.columns) else "task_id"
    stats_d["htc_rows"] = len(htc)
    stats_d["htc_tasks"] = htc[htc_id_col].nunique()

    macro["join_id"] = macro["task_id"].astype(str)
    htc_rename = {c: f"htc_{c}" for c in HTC_FEATS}
    htc_rename["step"] = "step_k"
    htc = htc.rename(columns=htc_rename)
    htc_feats_renamed = [f"htc_{c}" for c in HTC_FEATS]
    htc["join_id"] = htc[htc_id_col].astype(str)
    labels = labels.rename(columns={"step": "step_k"})

    joined = macro.merge(htc[["join_id", "step_k"] + htc_feats_renamed], on=["join_id", "step_k"], how="inner")
    joined = joined.merge(labels, on=["join_id", "step_k"], how="inner")
    stats_d["joined_rows"] = len(joined)
    stats_d["joined_tasks"] = joined["join_id"].nunique()
    return joined, stats_d


def run_5fold_cv_prebuilt(df: pd.DataFrame, feat_cols: list[str], folds: list[tuple]) -> dict:
    all_y, all_p = [], []
    for tr_tasks, te_tasks in folds:
        train_df = df[df["join_id"].isin(tr_tasks)]
        test_df = df[df["join_id"].isin(te_tasks)]
        y_tr = train_df["label"].values.astype(int)
        if len(np.unique(y_tr)) < 2:
            continue
        medians = safe_medians(train_df, feat_cols)
        X_tr = train_df[feat_cols].fillna(medians).fillna(0.0).values
        X_te = test_df[feat_cols].fillna(medians).fillna(0.0).values
        step_counts = train_df.groupby("join_id")["step_k"].transform("count")
        sample_weight = (1.0 / step_counts).values
        clf = fit_model("Logistic_L2", X_tr, y_tr, sample_weight, task_ids=train_df["join_id"].values)
        y_prob = clf.predict_proba(X_te)[:, 1]
        all_y.append(test_df["label"].values.astype(int))
        all_p.append(y_prob)
    if not all_y:
        return {}
    return finalize_metrics(np.concatenate(all_y), np.concatenate(all_p))


def make_folds(df: pd.DataFrame, seed: int = 42) -> list[tuple]:
    task_ids = df["join_id"].unique()
    task_outcomes = df.groupby("join_id")["outcome"].first().reindex(task_ids).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = []
    for tr_idx, te_idx in skf.split(task_ids, task_outcomes):
        folds.append((set(task_ids[tr_idx]), set(task_ids[te_idx])))
    return folds


def main():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    htc_feats_renamed = [f"htc_{c}" for c in HTC_FEATS]

    rows = []
    print("=" * 70)
    for bench, model in COMBOS:
        print(f"\n{bench}/{model}")
        joined, stats_d = build_joined(bench, model)
        print(f"  Macro pop (verbalized rollout): {stats_d['macro_rows']} rows / {stats_d['macro_tasks']} tasks")
        print(f"  HTC pop:                        {stats_d.get('htc_rows', 0)} rows / {stats_d.get('htc_tasks', 0)} tasks")
        if joined.empty:
            print("  [SKIP] no HTC data (claude has no logprobs) or empty join")
            continue
        retain_macro = stats_d["joined_rows"] / stats_d["macro_rows"] * 100
        retain_htc = stats_d["joined_rows"] / stats_d["htc_rows"] * 100
        print(f"  Joined (inner):  {stats_d['joined_rows']} rows / {stats_d['joined_tasks']} tasks  "
              f"({retain_macro:.0f}% of macro rows, {retain_htc:.0f}% of HTC rows retained)")

        if joined["label"].nunique() < 2:
            print("  [SKIP] insufficient label diversity after join")
            continue

        folds = make_folds(joined)
        for fs_name, feat_cols in [("macro", MACRO_FEATS),
                                    ("htc_only", htc_feats_renamed),
                                    ("macro_internal", MACRO_FEATS + htc_feats_renamed)]:
            avail = [c for c in feat_cols if c in joined.columns]
            missing = set(feat_cols) - set(avail)
            if missing:
                print(f"  [WARN] {fs_name}: missing {len(missing)} feat cols: {sorted(missing)[:5]}...")
            m = run_5fold_cv_prebuilt(joined, avail, folds)
            if not m:
                print(f"  [SKIP] {fs_name}: fold training failed")
                continue
            rows.append({"combo": f"{bench}_{model}", "benchmark": bench, "feature_set": fs_name,
                         "n_features": len(avail), "n_rows": stats_d["joined_rows"],
                         "n_tasks": stats_d["joined_tasks"], **m})
            print(f"  {fs_name:16s} ({len(avail):3d} feat)  auroc={m['auroc']:.4f}  "
                  f"e_aurc={m['e_aurc']:.4f}  brier={m['brier']:.4f}")

    per_combo = pd.DataFrame(rows)
    per_combo.to_csv(out_dir / "macro_internal_per_combo.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY (mean +/- std across combos with usable HTC data)")
    print("=" * 70)
    summary = per_combo.groupby("feature_set")[["auroc", "e_aurc", "brier"]].agg(["mean", "std", "count"])
    print(summary.to_string())
    summary.to_csv(out_dir / "macro_internal_summary.csv")

    piv = per_combo.pivot_table(index="combo", columns="feature_set", values=["auroc", "e_aurc", "brier"])
    sig_rows = []
    comparisons = [
        ("macro_internal", "macro"),
        ("macro_internal", "htc_only"),
        ("macro", "htc_only"),
    ]
    for a, b in comparisons:
        print(f"\nPaired delta ({a} - {b}):")
        for metric in ["auroc", "e_aurc", "brier"]:
            if (metric, a) not in piv.columns or (metric, b) not in piv.columns:
                continue
            d = (piv[(metric, a)] - piv[(metric, b)]).dropna()
            if len(d) < 2:
                continue
            t, p = stats.ttest_1samp(d, 0.0)
            w, pw = stats.wilcoxon(d)
            n_worse = int((d < 0).sum()) if metric == "auroc" else int((d > 0).sum())
            print(f"  {metric:8s} mean={d.mean():+.4f}  n={len(d)}  n_worse={n_worse}  "
                  f"t_p={p:.4f}  wilcoxon_p={pw:.4f}")
            sig_rows.append({"comparison": f"{a}_minus_{b}", "metric": metric,
                             "mean_diff": round(float(d.mean()), 4), "n": len(d),
                             "n_worse": n_worse, "t_test_p": round(float(p), 4),
                             "wilcoxon_p": round(float(pw), 4)})
    pd.DataFrame(sig_rows).to_csv(out_dir / "macro_internal_significance.csv", index=False)

    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
