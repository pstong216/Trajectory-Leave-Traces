#!/usr/bin/env python3
"""
run_detection_false_cut.py — Detection vs. false-cut tradeoff under a
keystep-aware early-cut policy, self-contained inside this repository
(2026-07-31 rebuild; 2026-08-08 CORRECTNESS FIX -- see below).

Policy: at each step, if the method's predicted P(failure) exceeds
threshold tau, the trajectory is CUT (stopped) at that step. Sweeping tau
traces a tradeoff curve, analogous to an ROC curve but framed around the
cut DECISION rather than raw ranking:

  detection_rate(tau) = (# failed tasks whose cut_step >= keyStep) / (# failed tasks)
      -- a "detection": the policy stops the trajectory at or after the point
      where failure became locked in (keyStep), so no further steps are
      wasted on an already-doomed trajectory.
  false_cut_rate(tau) = (# successful tasks that got cut at any step) / (# successful tasks)
      -- a "false cut": the trajectory was cut even though, left alone, it
      would have succeeded.

A failed task cut BEFORE its own keyStep, or never cut at all, counts as
"not detected" (neither a detection nor a false cut -- it's simply a miss).

2026-08-08 CORRECTNESS FIX (previously wrong, caught in review):
final_failed and key_step used to be INFERRED from the last value of each
method's own already-relabeled per-step "label" column. This is wrong for
a real, non-trivial slice of tasks: whenever a failed task's annotated key
step falls AT OR BEYOND the last step actually recorded for that
trajectory (verified: 23/277 mind2web/claude tasks alone -- e.g. a task
with 30 recorded steps [0..29] and keyStep=30), the relabeled "label"
column never flips to 1 anywhere in the observed data even though the
task truly failed -- silently reclassifying it as "successful" and
corrupting every downstream rate (this exact bug, independently caught in
../../code/label_strategy_ablation/, moved mind2web's failed-task count
from 1424 to the correct 1599 once fixed). Fixed here the same way:
final_failed now comes from the RAW ground-truth outcome file (never a
relabeled column), and key_step comes directly from the external keystep
annotation file (never inferred from any predictor's own label column).
Two data sources, matched to which rollout each method's predictions
come from (role split documented in README.md's "baseline
vs ours key-step split" section -- getting this backwards is exactly the
historical bug that README warns about):
  - Ours (cv_predictions_all/macro_only/uq_only.csv): OURS role --
    keystep_labels/<bench>/<model>_ours.json; true outcome from the
    SAME rollout's ground truth (webarena: raw/verbalized/actions/*.json
    "score" field -- Ours and baseline share the verbalized rollout as
    their score source even though their OWN feature rollout differs;
    mind2web: raw/verbalized/eval.json, matching what build_per_step_df
    itself used to build the "success" column these predictions were
    trained against).
  - M1-M7/M15/M11/M13 (per_step_scores.csv / stacked_step_scores.csv):
    BASELINE role -- keystep_labels/<bench>/<model>_baseline.json; same
    ground-truth outcome files (both roles score against the verbalized
    rollout's own outcome, per compute_baselines.py's
    build_per_step_df()).

ALSO FIXED (2026-08-08, same audit): every method is now evaluated on the
SAME population for a fair comparison -- claude is uniformly excluded
(MATCHED8_COMBOS) for every method shown together (M2-M4/M11 already had
zero claude data structurally; M1/M6/M7/Ours used to silently include it,
handing them 2 extra combos of easy population none of the other
compared methods got). See run_matched_task_intersection() for the
further, stricter cut: after matching combos, some individual TASKS still
only exist in one method's population (e.g. HTC's extract_features_prefix
returning None for a handful of tasks) -- intersect down to the task set
every compared method actually has a score for, so no method benefits
from a systematically-different task mix even within the matched-8 combos.

Every source file here comes from reproduction_output/ (a
reproduce.sh run) and data/ (for raw
ground-truth + keystep annotation), never results_total/baseline directly.

Methods (each evaluated over its own available population before the
final intersection step -- same convention as the main baseline table,
e.g. M2-M4/M11 are unavailable for claude, which has no exposed token
logprobs):
  - Ours       cv_predictions_all.csv, Logistic_L2_prob = P(failure) natively
  - M1/M7      verbalized confidence (raw + cumulative mean)
  - M2/M3/M4   action/output logprob means
  - M5/M6      action entropy (raw + cumulative), entropy_dir label
  - M11        HTC full/L2, already out-of-fold CV probabilities (M12
                dropped 2026-07-31, design decision -- same as the main table)
  - M13        Stacked HTC_L2 over M1-M4/M7, already out-of-fold

Outputs (code/generalization/results/):
  detection_false_cut_curves.csv   -- tau, detection_rate, false_cut_rate per method
  detection_false_cut_summary.csv  -- AUC of each tradeoff curve + population size
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PR = Path(__file__).resolve().parent.parent.parent
REPRO = PR / "reproduction_output"

sys.path.insert(0, str(PR / "code" / "baseline_extension"))
from compute_m2_cumulative import add_m2_cumulative  # noqa: E402 -- M15 feature

sys.path.insert(0, str(PR / "code" / "baseline_calculation"))
from compute_baselines import load_ground_truth as load_mind2web_ground_truth  # noqa: E402

sys.path.insert(0, str(PR / "code" / "l2_training"))
from train_risk_model_temp_sca import load_annotation  # noqa: E402

COMBOS = [
    ("webarena", "claude"), ("webarena", "gpt_5_2"), ("webarena", "gptnano"),
    ("webarena", "kimi"), ("webarena", "qwen30b"),
    ("mind2web", "claude"), ("mind2web", "gpt_5_2"), ("mind2web", "gptnano"),
    ("mind2web", "kimi"), ("mind2web", "qwen30b"),
]

# M2-M4/M11 need raw token logprobs, unavailable for claude in either
# benchmark. 2026-08-08: now the DEFAULT population for every method's
# loader (not just an extra "matched8" line) -- see module docstring.
MATCHED8_COMBOS = [(b, m) for b, m in COMBOS if m != "claude"]

# raw-score orientation, exactly matching update_baselines_with_features.py's
# HIGHER_IS_CONFIDENT: True => higher raw value = more confident of SUCCESS.
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
HTC_COLS = {"m11_htc_full_step": "success"}


# ── TRUE ground truth (never inferred from a relabeled per-step column) ────

def load_true_outcomes(bench: str, model: str) -> dict[str, bool]:
    """task_id(str) -> True iff the task TRULY failed, from the raw
    ground-truth file used to build per_step_scores.csv in the first
    place (compute_baselines.py / compute_webarena_step_baselines.py's
    own source) -- never from any relabeled "success"/"label" column.

    The public release ships this map as data/<bench>/<model>/task_outcomes.json
    (derived from the raw rollout at packaging time), so the analysis runs
    without the multi-GB raw trajectories. The raw-reading branches below are
    kept as the authoritative fallback whenever raw/ is present."""
    derived = PR / "data" / bench / model / "task_outcomes.json"
    if derived.is_file():
        return {str(k): bool(v) for k, v in json.loads(derived.read_text()).items()}
    if bench == "webarena":
        actions_dir = PR / "data" / bench / model / "raw" / "verbalized" / "actions"
        out: dict[str, bool] = {}
        if not actions_dir.is_dir():
            return out
        for p in actions_dir.glob("*.json"):
            try:
                d = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            score = d.get("score")
            if score is None:
                continue
            tid = str(d.get("task_id", p.stem))
            out[tid] = (float(score) != 1.0)
        return out
    # mind2web: prefer the verbalized rollout's own eval.json (matches what
    # build_per_step_df used for M1-4/M7/M11-14's success ground truth),
    # fall back to raw/ground_truth.json.
    gt_path = PR / "data" / "mind2web" / model / "raw" / "verbalized" / "eval.json"
    if not gt_path.is_file():
        gt_path = PR / "data" / "mind2web" / model / "raw" / "ground_truth.json"
    gt = load_mind2web_ground_truth(str(gt_path))
    return {tid: (status != "success") for tid, status in gt.items()}


def load_key_steps(bench: str, model: str, role: str) -> dict[str, int | None]:
    """task_id(str) -> keyStep, from the external annotation file for the
    given role ("ours" or "baseline") -- never inferred from a predictor's
    own label column."""
    p = PR / "keystep_labels" / bench / f"{model}_{role}.json"
    if not p.is_file():
        return {}
    return load_annotation(p)


def _finalize_true(df: pd.DataFrame, outcome_map: dict[str, bool],
                    key_step_map: dict[str, int | None], group_col: str) -> pd.DataFrame:
    """Adds final_failed (from outcome_map, ground truth) and key_step
    (from key_step_map, ground truth) per row. Tasks missing from
    outcome_map are DROPPED (not guessed) -- better to lose a few tasks
    than silently mislabel them."""
    df = df.sort_values([group_col, "step"]).reset_index(drop=True)
    out_rows = []
    for tid, g in df.groupby(group_col):
        tid_str = str(tid)
        if tid_str not in outcome_map:
            continue
        final_failed = bool(outcome_map[tid_str])
        if final_failed:
            ks = key_step_map.get(tid_str)
            key_step = int(ks) if ks is not None else 0
        else:
            key_step = None
        out_rows.append(g.assign(final_failed=final_failed, key_step=key_step))
    if not out_rows:
        return df.iloc[0:0].assign(final_failed=pd.Series(dtype=bool), key_step=pd.Series(dtype=float))
    return pd.concat(out_rows, ignore_index=True)


def collect_tasks(df: pd.DataFrame, group_col: str, p_fail_col: str,
                   source: str | None = None) -> list[dict]:
    tasks = []
    for tid, g in df.groupby(group_col):
        g = g.sort_values("step")
        t = {
            "final_failed": bool(g["final_failed"].iloc[0]),
            "key_step": g["key_step"].iloc[0],
            "steps": list(zip(g["step"].values.tolist(), g[p_fail_col].values.tolist())),
        }
        if source is not None:
            t["global_task_id"] = f"{source}::{tid}"
        tasks.append(t)
    return tasks


def load_ours(combos=MATCHED8_COMBOS, variant: str = "all") -> list[dict]:
    all_tasks = []
    for bench, model in combos:
        pattern = REPRO / "ours" / f"{bench}_{model}" / "thr_0p3" / "train_risk_model_temp_sca"
        runs = sorted(pattern.glob(f"*/cv_predictions_{variant}.csv"))
        if not runs:
            print(f"  [SKIP] ours {bench}/{model}: no cv_predictions_{variant}.csv")
            continue
        df = pd.read_csv(runs[-1])
        df = df.rename(columns={"step_k": "step", "Logistic_L2_prob": "p_fail"})
        df["task_id"] = df["task_id"].astype(str)
        outcome_map = load_true_outcomes(bench, model)
        key_step_map = load_key_steps(bench, model, role="ours")
        df = _finalize_true(df, outcome_map, key_step_map, group_col="task_id")
        tasks = collect_tasks(df, "task_id", "p_fail", source=f"{bench}_{model}")
        print(f"  ours  {bench}/{model:10s}  {len(tasks)} tasks")
        all_tasks.extend(tasks)
    return all_tasks


def load_m1_m7(method: str, combos=MATCHED8_COMBOS) -> list[dict]:
    label_col = M1_M7_LABEL_COL[method]
    higher_conf = HIGHER_IS_CONFIDENT[method]
    all_tasks = []
    for bench, model in combos:
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
        sub = df[[group_col, "step", method]].dropna().copy()
        if sub.empty:
            continue
        sub[group_col] = sub[group_col].astype(str)
        raw = sub[method].values.astype(float)
        sub["p_fail"] = -(raw if higher_conf else -raw)  # higher = more predicted failure
        outcome_map = load_true_outcomes(bench, model)
        key_step_map = load_key_steps(bench, model, role="baseline")
        sub = _finalize_true(sub, outcome_map, key_step_map, group_col=group_col)
        tasks = collect_tasks(sub, group_col, "p_fail", source=f"{bench}_{model}")
        all_tasks.extend(tasks)
    return all_tasks


def load_htc(method: str, combos=MATCHED8_COMBOS) -> list[dict]:
    all_tasks = []
    for bench, model in combos:
        p = REPRO / "baseline" / bench / model / "results" / "per_step_scores.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        if method not in df.columns:
            continue
        group_col = "task_index" if (bench == "mind2web" and "task_index" in df.columns) else "task_id"
        sub = df[[group_col, "step", method]].dropna().copy()
        if sub.empty:
            continue
        sub[group_col] = sub[group_col].astype(str)
        sub["p_fail"] = 1.0 - sub[method].astype(float)
        outcome_map = load_true_outcomes(bench, model)
        key_step_map = load_key_steps(bench, model, role="baseline")
        sub = _finalize_true(sub, outcome_map, key_step_map, group_col=group_col)
        tasks = collect_tasks(sub, group_col, "p_fail", source=f"{bench}_{model}")
        all_tasks.extend(tasks)
    return all_tasks


def load_m13(combos=MATCHED8_COMBOS) -> list[dict]:
    all_tasks = []
    for bench, model in combos:
        p = REPRO / "baseline" / bench / model / "results" / "stacked_step_scores.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        if "m13_stacked_step" not in df.columns:
            continue
        group_col = "task_index" if (bench == "mind2web" and "task_index" in df.columns) else "task_id"
        sub = df[[group_col, "step", "m13_stacked_step"]].dropna().copy()
        if sub.empty:
            continue
        sub[group_col] = sub[group_col].astype(str)
        sub["p_fail"] = 1.0 - sub["m13_stacked_step"].astype(float)
        outcome_map = load_true_outcomes(bench, model)
        key_step_map = load_key_steps(bench, model, role="baseline")
        sub = _finalize_true(sub, outcome_map, key_step_map, group_col=group_col)
        tasks = collect_tasks(sub, group_col, "p_fail", source=f"{bench}_{model}")
        all_tasks.extend(tasks)
    return all_tasks


def intersect_tasks(methods: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Restrict every method's task list to the INTERSECTION of
    global_task_id across all methods -- so even within the matched-8
    combos, no method is scored on a systematically different task mix
    (e.g. a handful of tasks HTC's extract_features_prefix silently drops).
    Reports the population cost of this step; does not hide it."""
    id_sets = [set(t["global_task_id"] for t in tasks) for tasks in methods.values() if tasks]
    if not id_sets:
        return methods
    common = set.intersection(*id_sets)
    out = {}
    for name, tasks in methods.items():
        before = len(tasks)
        kept = [t for t in tasks if t["global_task_id"] in common]
        out[name] = kept
        if before:
            print(f"  {name:28s}  {before:5d} -> {len(kept):5d} tasks "
                  f"({len(kept)/before:.0%} retained after intersection)")
    return out


def tradeoff_curve(tasks: list[dict], n_thresholds: int = 200) -> pd.DataFrame:
    all_p = np.concatenate([[p for _, p in t["steps"]] for t in tasks])
    thresholds = np.unique(np.concatenate([
        np.linspace(all_p.min(), all_p.max(), n_thresholds),
        [all_p.max() + 1e-6],
    ]))
    thresholds = np.sort(thresholds)[::-1]

    failed = [t for t in tasks if t["final_failed"]]
    success = [t for t in tasks if not t["final_failed"]]
    n_failed, n_success = len(failed), len(success)

    rows = []
    for tau in thresholds:
        n_detected = 0
        for t in failed:
            cut_step = next((s for s, p in t["steps"] if p > tau), None)
            if cut_step is not None and cut_step >= t["key_step"]:
                n_detected += 1
        n_false_cut = 0
        for t in success:
            cut_step = next((s for s, p in t["steps"] if p > tau), None)
            if cut_step is not None:
                n_false_cut += 1
        rows.append({
            "tau": round(float(tau), 5),
            "detection_rate": n_detected / n_failed if n_failed else float("nan"),
            "false_cut_rate": n_false_cut / n_success if n_success else float("nan"),
        })
    return pd.DataFrame(rows)


def curve_auc(df: pd.DataFrame) -> float:
    d = df.sort_values("false_cut_rate")
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(d["detection_rate"].values, d["false_cut_rate"].values))


def main():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    methods = {}

    print("Loading Ours (matched-8 combos, claude excluded)...")
    methods["ours_l2_all"] = load_ours()

    for m in M1_M7_LABEL_COL:
        print(f"Loading {m}...")
        methods[m] = load_m1_m7(m)

    for m in HTC_COLS:
        print(f"Loading {m}...")
        methods[m] = load_htc(m)

    print("Loading m13_stacked_htc_l2...")
    methods["m13_stacked_htc_l2"] = load_m13()

    print("\nIntersecting to the common task set across all methods...")
    methods = intersect_tasks(methods)

    all_curves = []
    summary = {}
    for name, tasks in methods.items():
        if not tasks:
            print(f"  [SKIP] {name}: no tasks")
            continue
        n_failed = sum(t["final_failed"] for t in tasks)
        n_success = len(tasks) - n_failed
        curve = tradeoff_curve(tasks)
        curve["method"] = name
        all_curves.append(curve)
        auc = curve_auc(curve)
        summary[name] = {"auc": round(auc, 4), "n_tasks": len(tasks),
                          "n_failed": n_failed, "n_success": n_success}
        print(f"  {name:28s}  AUC={auc:.4f}  n_tasks={len(tasks):5d}  "
              f"n_failed={n_failed:5d}  n_success={n_success:5d}")

    pd.concat(all_curves, ignore_index=True).to_csv(out_dir / "detection_false_cut_curves.csv", index=False)
    (out_dir / "detection_false_cut_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(summary).T.to_csv(out_dir / "detection_false_cut_summary.csv")

    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
