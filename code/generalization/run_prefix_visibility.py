#!/usr/bin/env python3
"""
run_prefix_visibility.py — Key-step-aware prefix visibility curves,
self-contained inside this repository (2026-07-31 rebuild, expanded to the
full M1-M7/M11/M13 baseline set, not just M11.
M12 dropped 2026-07-31, design decision -- same as the main table.

"We ask when failure becomes discriminable from a trajectory prefix, before
considering any intervention policy. For a failed trajectory with first
positive key-step label k*_i, prefixes before k*_i are treated as negative
and prefixes at or after k*_i as positive; successful trajectories are
always negative:

    y_vis_{i,k} = 1[trajectory i fails and k >= k*_i]                  (8)

We divide each trajectory into five relative-time buckets and report AUROC
for this visible-failure target."

Implementation:
  - relative position of step k within a T-step trajectory: rank/max(T-1,1)
    in [0,1] (rank = 0-indexed position in the trajectory's OWN step
    sequence, not the raw "step" column value, in case of gaps)
  - bucket = floor(rel_pos * 5), clipped to {0..4} ("bucket 1".."bucket 5")
  - y_vis computed per (task, step) exactly per eq. (8), reusing each
    method's own already-relabeled per-step label to recover key_step (same
    approach as run_detection_false_cut.py -- avoids a second annotation
    lookup)
  - AUROC(y_vis, score) computed within each bucket, pooled across all
    available combos, for every method

Every source file here comes from reproduction_output/ (a
reproduce.sh run), never results_total/baseline directly.

Outputs (code/generalization/results/):
  prefix_visibility_auroc_by_bucket.csv
  prefix_visibility_keystep_bucket_hist.csv   (Ours, all 10 combos -- most
                                                complete keyStep distribution)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PR = Path(__file__).resolve().parent.parent.parent
REPRO = PR / "reproduction_output"

sys.path.insert(0, str(PR / "code" / "baseline_extension"))
from compute_m2_cumulative import add_m2_cumulative  # noqa: E402 -- M15 feature
N_BUCKETS = 5

COMBOS = [
    ("webarena", "claude"), ("webarena", "gpt_5_2"), ("webarena", "gptnano"),
    ("webarena", "kimi"), ("webarena", "qwen30b"),
    ("mind2web", "claude"), ("mind2web", "gpt_5_2"), ("mind2web", "gptnano"),
    ("mind2web", "kimi"), ("mind2web", "qwen30b"),
]
MATCHED8_COMBOS = [(b, m) for b, m in COMBOS if m != "claude"]

HIGHER_IS_CONFIDENT = {
    "m1_verbalized_conf": True, "m2_action_logprob_mean": False,
    "m3_output_logprob_mean": False, "m4_cumulative_output_logprob_mean": False,
    "m5_action_entropy": False, "m6_cumulative_action_entropy": False,
    "m7_conf_cumul_mean": True,
    "m15_cumulative_action_logprob_mean": False,  # same orientation as m2
}
M1_M7_LABEL_COL = {
    "m1_verbalized_conf": "success", "m2_action_logprob_mean": "success",
    "m3_output_logprob_mean": "success", "m4_cumulative_output_logprob_mean": "success",
    "m5_action_entropy": "success_entropy", "m6_cumulative_action_entropy": "success_entropy",
    "m7_conf_cumul_mean": "success",
    "m15_cumulative_action_logprob_mean": "success",
}
HTC_COLS = ["m11_htc_full_step"]


def _finalize(df: pd.DataFrame, label_col: str, group_col: str, score_col: str) -> pd.DataFrame:
    """Adds final_failed, key_step, T, rel_pos, bucket, y_vis, key_step_bucket
    per row -- all derived from the method's own already-relabeled per-step
    label (label_col: 1=failure)."""
    df = df.sort_values([group_col, "step"]).reset_index(drop=True)
    out_rows = []
    for _, g in df.groupby(group_col):
        g = g.sort_values("step").copy()
        final_failed = bool(g[label_col].iloc[-1] == 1)
        T = len(g)
        key_step = None
        if final_failed:
            failed_steps = g.loc[g[label_col] == 1, "step"]
            key_step = int(failed_steps.min()) if len(failed_steps) else 0

        denom = max(T - 1, 1)
        rank = np.arange(T)
        rel_pos = rank / denom
        bucket = np.clip((rel_pos * N_BUCKETS).astype(int), 0, N_BUCKETS - 1)

        y_vis = (g["step"].values >= key_step).astype(int) if final_failed else np.zeros(T, dtype=int)

        g["final_failed"] = final_failed
        g["key_step"] = key_step
        g["rel_pos"] = rel_pos
        g["bucket"] = bucket
        g["y_vis"] = y_vis

        if final_failed:
            ks_idx = min(int(np.searchsorted(g["step"].values, key_step)), T - 1)
            g["key_step_bucket"] = int(bucket[ks_idx])
        else:
            g["key_step_bucket"] = np.nan

        out_rows.append(g)
    out = pd.concat(out_rows, ignore_index=True)
    return out[[group_col, "step", "bucket", "y_vis", "final_failed", "key_step_bucket", score_col]]


# ── Per-method loaders (mirrors run_detection_false_cut.py) ────────────────

def load_ours(combos=COMBOS) -> pd.DataFrame | None:
    frames = []
    for bench, model in combos:
        pattern = REPRO / "ours" / f"{bench}_{model}" / "thr_0p3" / "train_risk_model_temp_sca"
        runs = sorted(pattern.glob("*/cv_predictions_all.csv"))
        if not runs:
            continue
        df = pd.read_csv(runs[-1])
        df = df.rename(columns={"step_k": "step", "Logistic_L2_prob": "score"})
        df["task_id"] = df["task_id"].astype(str)
        out = _finalize(df, label_col="label", group_col="task_id", score_col="score")
        out["source"] = f"{bench}_{model}"
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else None


def load_m1_m7(method: str) -> pd.DataFrame | None:
    label_col = M1_M7_LABEL_COL[method]
    higher_conf = HIGHER_IS_CONFIDENT[method]
    frames = []
    for bench, model in COMBOS:
        p = REPRO / "baseline" / bench / model / "results" / "per_step_scores.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        if method == "m15_cumulative_action_logprob_mean":
            df = add_m2_cumulative(df).rename(
                columns={"m2c_cumulative_action_logprob_mean": method})
        if method not in df.columns or label_col not in df.columns:
            continue
        group_col = "task_index" if (bench == "mind2web" and "task_index" in df.columns) else "task_id"
        sub = df[[group_col, "step", label_col, method]].dropna().copy()
        if sub.empty:
            continue
        sub[group_col] = sub[group_col].astype(str)
        raw = sub[method].values.astype(float)
        sub["score"] = raw if higher_conf else -raw  # higher = more confident of SUCCESS
        sub["score"] = -sub["score"]                  # -> higher = more predicted FAILURE
        sub["label"] = 1 - sub[label_col].astype(int)
        out = _finalize(sub, label_col="label", group_col=group_col, score_col="score")
        out["source"] = f"{bench}_{model}"
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else None


def load_htc(method: str) -> pd.DataFrame | None:
    frames = []
    for bench, model in COMBOS:
        p = REPRO / "baseline" / bench / model / "results" / "per_step_scores.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        if method not in df.columns:
            continue
        group_col = "task_index" if (bench == "mind2web" and "task_index" in df.columns) else "task_id"
        sub = df[[group_col, "step", "success", method]].dropna().copy()
        if sub.empty:
            continue
        sub[group_col] = sub[group_col].astype(str)
        sub["score"] = 1.0 - sub[method].astype(float)
        sub["label"] = 1 - sub["success"].astype(int)
        out = _finalize(sub, label_col="label", group_col=group_col, score_col="score")
        out["source"] = f"{bench}_{model}"
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else None


def load_m13() -> pd.DataFrame | None:
    frames = []
    for bench, model in COMBOS:
        p = REPRO / "baseline" / bench / model / "results" / "stacked_step_scores.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        if "m13_stacked_step" not in df.columns:
            continue
        group_col = "task_index" if (bench == "mind2web" and "task_index" in df.columns) else "task_id"
        sub = df[[group_col, "step", "success", "m13_stacked_step"]].dropna().copy()
        if sub.empty:
            continue
        sub[group_col] = sub[group_col].astype(str)
        sub["score"] = 1.0 - sub["m13_stacked_step"].astype(float)
        sub["label"] = 1 - sub["success"].astype(int)
        out = _finalize(sub, label_col="label", group_col=group_col, score_col="score")
        out["source"] = f"{bench}_{model}"
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else None


# ── Metrics ──────────────────────────────────────────────────────────────

def auroc_by_bucket(df: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    for b in range(N_BUCKETS):
        sub = df[df["bucket"] == b]
        if sub["y_vis"].nunique() < 2:
            rows.append({"bucket": b + 1, "auroc": np.nan, "n": len(sub), "method": method})
            continue
        auc = roc_auc_score(sub["y_vis"].values, sub["score"].values)
        rows.append({"bucket": b + 1, "auroc": round(float(auc), 4), "n": len(sub), "method": method})
    return pd.DataFrame(rows)


def keystep_bucket_hist(df: pd.DataFrame) -> pd.DataFrame:
    tagged = df[df["final_failed"]][["task_id" if "task_id" in df.columns else "task_index",
                                      "source", "key_step_bucket"]].copy()
    id_col = tagged.columns[0]
    tagged["gid"] = tagged["source"] + "::" + tagged[id_col].astype(str)
    failed_tasks = tagged.drop_duplicates("gid")
    counts = failed_tasks["key_step_bucket"].value_counts().sort_index()
    total = len(failed_tasks)
    rows = [{"bucket": int(b) + 1, "n_failed_tasks": int(counts.get(b, 0)),
             "frac_of_failed_tasks": round(counts.get(b, 0) / total, 4) if total else np.nan}
            for b in range(N_BUCKETS)]
    return pd.DataFrame(rows)


def main():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    methods: dict[str, pd.DataFrame] = {}

    print("Loading Ours (all 10 combos)...")
    methods["ours_l2_all"] = load_ours()
    print("Loading Ours (8 combos matching M2-M4/M11's logprob coverage)...")
    methods["ours_l2_matched8"] = load_ours(MATCHED8_COMBOS)

    for m in M1_M7_LABEL_COL:
        print(f"Loading {m}...")
        methods[m] = load_m1_m7(m)

    for m in HTC_COLS:
        print(f"Loading {m}...")
        methods[m] = load_htc(m)

    print("Loading m13_stacked_htc_l2...")
    methods["m13_stacked_htc_l2"] = load_m13()

    auroc_rows = []
    for name, df in methods.items():
        if df is None or df.empty:
            print(f"  [SKIP] {name}: no data")
            continue
        # webarena frames key on "task_id", mind2web frames key on
        # "task_index" -- pd.concat aligns by column name, so a pooled
        # M1-M13 frame carries BOTH columns (NaN on whichever a given row's
        # bench doesn't use); coalesce before counting distinct tasks.
        id_cols = [c for c in ("task_id", "task_index") if c in df.columns]
        tid = df[id_cols[0]] if len(id_cols) == 1 else df["task_id"].fillna(df["task_index"])
        n_tasks = df.assign(_tid=tid).drop_duplicates(["source", "_tid"]).shape[0]
        print(f"  {name:22s}  {len(df):6d} rows, {n_tasks:5d} tasks")
        auroc_rows.append(auroc_by_bucket(df, name))
    auroc_all = pd.concat(auroc_rows, ignore_index=True)
    auroc_all.to_csv(out_dir / "prefix_visibility_auroc_by_bucket.csv", index=False)

    print(f"\n{'method':<22} " + " ".join(f"bucket{b}" for b in range(1, N_BUCKETS + 1)))
    for name in methods:
        if methods[name] is None:
            continue
        sub = auroc_all[auroc_all.method == name].sort_values("bucket")
        vals = " ".join(f"{v:7.4f}" for v in sub["auroc"].values)
        print(f"{name:<22} {vals}")

    hist = keystep_bucket_hist(methods["ours_l2_all"])
    hist.to_csv(out_dir / "prefix_visibility_keystep_bucket_hist.csv", index=False)
    print("\nkeyStep relative-time bucket distribution (Ours, failed tasks, all 10 combos):")
    print(hist.to_string(index=False))

    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
