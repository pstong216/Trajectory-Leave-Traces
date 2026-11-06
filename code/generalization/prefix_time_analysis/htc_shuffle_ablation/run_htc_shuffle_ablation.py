#!/usr/bin/env python3
"""
run_htc_shuffle_ablation.py — Does M11 (HTC Full)'s outer 5-fold CV need to
be shuffled? (Is HTC's CV shuffled, and would results be worse
without shuffling?)

Confirmed by reading the code (not run here, just cited): the production
call site, update_baselines_with_features.py:582, calls
compute_htc_step_scores(..., task_length_weight=True), and inside that
function (htc_step_baseline.py:394) the outer 5-fold split is

    StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

-- i.e. yes, shuffled, with a fixed seed for reproducibility. This script
answers the second half empirically rather than by guessing: refit the
SAME feature set (from build_htc_step_dataset(), unchanged -- only the
outer fold assignment differs) with shuffle=False, and compare against
shuffle=True at 5 different seeds (42 -- the production seed -- plus 0,1,2,3)
to see whether shuffle=False falls outside the shuffled seeds' spread.

Why unshuffled CV is a real risk here, not just a formality: task_group
order comes from build_htc_step_dataset()'s per_step_df.groupby(group_col),
which pandas sorts by key by default. For WebArena, task_id is the
benchmark's own numeric task index, and WebArena's suite is laid out in
site-sized contiguous blocks (shopping tasks, then CMS, then Reddit, ...).
An unshuffled StratifiedKFold hands out consecutive tasks to consecutive
folds, so an unshuffled split risks each fold being dominated by 1-2
website categories -- exactly the kind of distribution shift the
"Task-Regime Generalization" / run_website_generalization.py experiment
already shows is hard. Mind2Web's task_index ordering was not separately
audited here.

Reads only from reproduction_output (already-computed
per_step_scores.csv has the "success"/"step"/task id columns
build_htc_step_dataset() needs; the m7/m11/m12 columns already present in
that CSV are ignored -- build_htc_step_dataset() never reads them) and
data/<bench>/<model>/raw/verbalized (the logprobs source),
mirroring reproduce.sh's own htc_verbalized_dir argument exactly.

Outputs (this folder's results/):
  htc_shuffle_ablation_region.csv     region x scope x shuffle-mode AUROC
  htc_shuffle_ablation_summary.txt    printed comparison, saved
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
PTA_DIR = THIS_DIR.parent               # prefix_time_analysis/
GEN_DIR = PTA_DIR.parent                # code/generalization/
HTC_DIR = GEN_DIR.parent / "htc"        # code/htc/
sys.path.insert(0, str(GEN_DIR))
sys.path.insert(0, str(HTC_DIR))
sys.path.insert(0, str(PTA_DIR))

from htc_step_baseline import build_htc_step_dataset, best_C_for_fold  # noqa: E402
from run_prefix_time_regions import _with_benchmark, BENCHMARKS, REGIONS  # noqa: E402
from run_prefix_visibility import PR, REPRO, MATCHED8_COMBOS, _finalize  # noqa: E402

SHUFFLE_SEEDS = [42, 0, 1, 2, 3]  # 42 = production's own seed


def fit_m11_full(feat_df: pd.DataFrame, shuffle: bool, seed: int | None) -> np.ndarray:
    """Stripped-down copy of htc_step_baseline.compute_htc_step_scores(),
    full/L2 (M11) branch only -- M12/reduced dropped (not needed for this
    ablation, and roughly halves runtime). task_length_weight=True always,
    matching the production call site
    (update_baselines_with_features.py:582)."""
    feat_cols = [c for c in feat_df.columns if c.startswith("f") and c[1:].isdigit()]
    task_meta = feat_df[["task_group", "task_final_success"]].drop_duplicates().reset_index(drop=True)
    full_probs = np.full(len(feat_df), np.nan, dtype=float)
    if task_meta["task_final_success"].nunique() < 2:
        return full_probs

    cv_kwargs = dict(n_splits=5, shuffle=shuffle)
    if shuffle:
        cv_kwargs["random_state"] = seed
    cv = StratifiedKFold(**cv_kwargs)

    for train_task_idx, val_task_idx in cv.split(
        task_meta["task_group"].values, task_meta["task_final_success"].values.astype(int),
    ):
        train_groups = set(task_meta.iloc[train_task_idx]["task_group"])
        val_groups = set(task_meta.iloc[val_task_idx]["task_group"])
        train_mask = feat_df["task_group"].isin(train_groups).values
        val_mask = feat_df["task_group"].isin(val_groups).values
        train_df, val_df = feat_df.loc[train_mask], feat_df.loc[val_mask]
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
        step_counts = train_df.groupby("task_group")["step"].transform("count")
        full.fit(X_tr, y_tr, sample_weight=(1.0 / step_counts).values)
        full_probs[np.where(val_mask)[0]] = full.predict_proba(X_va)[:, 1]

    return full_probs


def load_feat_df(bench: str, model: str) -> pd.DataFrame | None:
    pss = REPRO / "baseline" / bench / model / "results" / "per_step_scores.csv"
    if not pss.is_file():
        return None
    per_step_df = pd.read_csv(pss)
    verbalized_dir = str(PR / "data" / bench / model / "raw" / "verbalized")
    feat_df = build_htc_step_dataset(f"{bench}/{model}", per_step_df, verbalized_dir_override=verbalized_dir)
    if feat_df.empty:
        return None
    group_col = "task_index" if (bench == "mind2web" and "task_index" in feat_df.columns) else "task_id"
    feat_df["source"] = f"{bench}_{model}"
    feat_df["group_col_name"] = group_col
    return feat_df


def scores_to_bucketed(feat_df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    df = feat_df.copy()
    df["score"] = 1.0 - probs          # higher = more predicted failure, matches load_htc()'s convention
    df["label"] = 1 - df["success"].astype(int)
    group_col = df["group_col_name"].iloc[0]
    out = _finalize(df, label_col="label", group_col=group_col, score_col="score")
    out["source"] = df["source"].iloc[0]
    return out


def region_auroc(bucket_rows: list[pd.DataFrame]) -> pd.DataFrame:
    all_df = pd.concat(bucket_rows, ignore_index=True)
    all_df = _with_benchmark(all_df)
    rows = []
    for scope in ["pooled"] + BENCHMARKS:
        scoped = all_df if scope == "pooled" else all_df[all_df["benchmark"] == scope]
        bucket_aucs = {}
        for b in range(5):
            sub = scoped[scoped["bucket"] == b]
            if sub["y_vis"].nunique() < 2:
                bucket_aucs[b] = (np.nan, len(sub))
                continue
            bucket_aucs[b] = (roc_auc_score(sub["y_vis"].values, sub["score"].values), len(sub))
        for region, (lo, hi) in REGIONS:
            vals = [bucket_aucs[b] for b in range(lo, hi + 1) if not np.isnan(bucket_aucs[b][0])]
            if not vals:
                auc = np.nan
            else:
                aucs, ns = zip(*vals)
                auc = float(np.average(aucs, weights=ns))
            rows.append({"scope": scope, "region": region, "auroc": round(auc, 4) if not np.isnan(auc) else np.nan})
    return pd.DataFrame(rows)


def main():
    out_dir = THIS_DIR / "results"
    out_dir.mkdir(exist_ok=True)

    t0 = time.time()
    feat_dfs = {}
    for bench, model in MATCHED8_COMBOS:
        print(f"Building HTC feature set for {bench}/{model}...")
        fd = load_feat_df(bench, model)
        if fd is not None:
            feat_dfs[(bench, model)] = fd
    print(f"Loaded {len(feat_dfs)}/{len(MATCHED8_COMBOS)} combos in {time.time()-t0:.1f}s")

    all_rows = []

    print("\nFitting shuffle=False (unshuffled outer 5-fold CV)...")
    bucket_rows = []
    for (bench, model), fd in feat_dfs.items():
        probs = fit_m11_full(fd, shuffle=False, seed=None)
        bucket_rows.append(scores_to_bucketed(fd, probs))
    reg = region_auroc(bucket_rows)
    reg["mode"] = "unshuffled"
    all_rows.append(reg)

    for seed in SHUFFLE_SEEDS:
        print(f"Fitting shuffle=True, seed={seed}...")
        bucket_rows = []
        for (bench, model), fd in feat_dfs.items():
            probs = fit_m11_full(fd, shuffle=True, seed=seed)
            bucket_rows.append(scores_to_bucketed(fd, probs))
        reg = region_auroc(bucket_rows)
        reg["mode"] = f"shuffled_seed{seed}"
        all_rows.append(reg)

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(out_dir / "htc_shuffle_ablation_region.csv", index=False)

    lines = []
    lines.append("=== HTC Full (M11) outer-CV shuffle ablation, matched-8 population (no claude) ===\n")
    for scope in ["pooled"] + BENCHMARKS:
        lines.append(f"\n-- scope={scope} --")
        piv = result[result.scope == scope].pivot_table(index="mode", columns="region", values="auroc")
        piv = piv.reindex(columns=["early", "mid", "late"])
        piv = piv.reindex(["unshuffled"] + [f"shuffled_seed{s}" for s in SHUFFLE_SEEDS])
        lines.append(piv.to_string())
        shuf_only = piv.drop(index="unshuffled")
        lines.append("\nshuffled mean +/- std:")
        lines.append((shuf_only.mean().round(4).astype(str) + " +/- " + shuf_only.std().round(4).astype(str)).to_string())
        lines.append(f"unshuffled vs shuffled mean delta:\n{(piv.loc['unshuffled'] - shuf_only.mean()).round(4).to_string()}")

    summary = "\n".join(lines)
    print("\n" + summary)
    (out_dir / "htc_shuffle_ablation_summary.txt").write_text(summary + "\n")
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
