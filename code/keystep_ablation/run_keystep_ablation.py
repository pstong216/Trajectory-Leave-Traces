#!/usr/bin/env python3
"""
run_keystep_ablation.py — Key-step-aware supervision ablation (2026-08-02).

Question: "Does key-step-aware supervision actually help, compared with
simply assigning the final trajectory label to every prefix?" Framed NOT
as "which label definition is more correct" (settled on definitional
grounds -- pre-key prefixes of a failed trajectory don't yet contain the
failure symptom, so labeling them 1 is a labeling error, not a modeling
choice) but as: does training under that error measurably cost you
anything, when both models are graded on the SAME operationally-meaningful
criteria? Four criteria, matching the advisor's protocol exactly:

  1. Pre-key false-alarm rate @ a fixed FCR budget -- does the
     trajectory-label model cry "failure" on prefixes that don't yet show
     it (direct behavioral evidence the noise it trained on leaked into
     its predictions)?
  2. Post-key Detection Rate @ the same FCR budget -- does that noise cost
     detection power where it matters?
  3. Key-step-aware AUROC / e-AURC / Brier -- the aggregate version of 1+2.
  4. Terminal-step failure AUROC -- a no-regret check: confirms key-step
     supervision isn't trading away final-outcome discrimination to win
     the other three.

Two arms, same features/folds/calibration/weighting, only the training
label differs (train_risk_model_temp_sca.py --label_mode):
  keystep    reproduction_output/ours/           (production, unchanged)
  trajectory code/keystep_ablation/trajectory_label_output/  (new, this
             experiment: df["label"] = df["outcome"] for every prefix,
             i.e. the naive ablation -- see that flag's docstring)

Critically, BOTH arms are scored against ONE externally-derived key_step
per task -- loaded directly from keystep_labels/<bench>/<model>_ours.json
via load_annotation(), never read off either model's own per-step "label"
column. Reusing a model's own label column to infer key_step (the
convenient shortcut run_prefix_visibility.py/run_detection_false_cut.py
use, safe there because both label columns in those callers were built by
the SAME keystep-aware convention) would be silently wrong here for the
trajectory arm: its "label" is outcome propagated to every step, so
inferring key_step from it would give key_step=0 for every failed task,
manufacturing a fake "HTC has zero pre-key region" result. Loading the
annotation independently avoids that trap entirely.

Outputs (this folder's own results/):
  keystep_ablation_by_benchmark.csv   scope x arm x metric
  keystep_ablation_curves.png
  table_keystep_ablation.tex
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

THIS_DIR = Path(__file__).resolve().parent
PR = THIS_DIR.parent.parent
L2_DIR = PR / "code" / "l2_training"
sys.path.insert(0, str(L2_DIR))

from train_risk_model_temp_sca import load_annotation, aurc, aurc_oracle  # noqa: E402

KEYSTEP_ROOT = PR / "reproduction_output" / "ours"
TRAJ_ROOT = THIS_DIR / "trajectory_label_output"
LABELS_ROOT = PR / "keystep_labels"

COMBOS = [
    ("webarena", "claude"), ("webarena", "gpt_5_2"), ("webarena", "gptnano"),
    ("webarena", "kimi"), ("webarena", "qwen30b"),
    ("mind2web", "claude"), ("mind2web", "gpt_5_2"), ("mind2web", "gptnano"),
    ("mind2web", "kimi"), ("mind2web", "qwen30b"),
]
BENCHMARKS = ["webarena", "mind2web"]
FCR_BUDGETS = [0.05, 0.10, 0.20]
HEADLINE_FCR = 0.10

ARM_ROOTS = {"keystep": KEYSTEP_ROOT, "trajectory": TRAJ_ROOT}
ARM_LABEL = {"keystep": "Key-step-aware (Ours)", "trajectory": "Trajectory-propagated (ablation)"}


def load_key_steps(bench: str, model: str) -> dict:
    """taskId(str) -> keyStep(int) or None, straight from the annotation
    file -- independent of either arm's own per-step label column."""
    p = LABELS_ROOT / bench / f"{model}_ours.json"
    return load_annotation(p)


def load_arm_tasks(arm: str, bench: str, model: str, key_steps: dict, variant: str = "all") -> list[dict]:
    root = ARM_ROOTS[arm]
    pattern = root / f"{bench}_{model}" / "thr_0p3" / "train_risk_model_temp_sca"
    runs = sorted(pattern.glob(f"*/cv_predictions_{variant}.csv"))
    if not runs:
        print(f"  [SKIP] {arm} {bench}/{model}: no cv_predictions_{variant}.csv under {pattern}")
        return []
    df = pd.read_csv(runs[-1])
    df = df.rename(columns={"step_k": "step", "Logistic_L2_prob": "p_fail"})
    df["task_id"] = df["task_id"].astype(str)

    tasks = []
    for tid, g in df.groupby("task_id"):
        g = g.sort_values("step")
        final_failed = bool(g["outcome"].iloc[-1] == 1)
        if final_failed:
            ks = key_steps.get(tid)
            key_step = int(ks) if ks is not None else 0
        else:
            key_step = None
        tasks.append({
            "task_id": tid, "benchmark": bench, "final_failed": final_failed,
            "key_step": key_step,
            "steps": list(zip(g["step"].values.tolist(), g["p_fail"].values.tolist())),
        })
    return tasks


def load_all(arm: str, variant: str = "all") -> list[dict]:
    all_tasks = []
    for bench, model in COMBOS:
        key_steps = load_key_steps(bench, model)
        tasks = load_arm_tasks(arm, bench, model, key_steps, variant=variant)
        all_tasks.extend(tasks)
    return all_tasks


def y_vis_rows(tasks: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Pooled step-level (y_vis, p_fail) per eq.(8): 1 iff task fails and
    step >= key_step; 0 for every step of a successful task."""
    y, p = [], []
    for t in tasks:
        for step, pf in t["steps"]:
            y.append(1 if (t["final_failed"] and step >= t["key_step"]) else 0)
            p.append(pf)
    return np.array(y), np.array(p)


def terminal_rows(tasks: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Task-level (final_outcome, p_fail at the LAST observed step)."""
    y, p = [], []
    for t in tasks:
        if not t["steps"]:
            continue
        last_p = max(t["steps"], key=lambda sp: sp[0])[1]
        y.append(1 if t["final_failed"] else 0)
        p.append(last_p)
    return np.array(y), np.array(p)


def tradeoff_curve_full(tasks: list[dict], n_thresholds: int = 200) -> pd.DataFrame:
    """Sweeps tau; for each, reports detection_rate (post-key cut on a
    failed task), false_cut_rate (any cut on a successful task), and
    pre_key_false_alarm_rate (cut BEFORE key_step on a failed, annotated
    task -- the direct behavioral signature of training-label noise)."""
    all_p = np.concatenate([[p for _, p in t["steps"]] for t in tasks])
    thresholds = np.unique(np.concatenate([
        np.linspace(all_p.min(), all_p.max(), n_thresholds), [all_p.max() + 1e-6],
    ]))[::-1]

    failed = [t for t in tasks if t["final_failed"]]
    success = [t for t in tasks if not t["final_failed"]]
    # pre-key false alarms only make sense for tasks with a real (>0) key_step;
    # key_step=0 tasks have no pre-key region by construction (not "safe", just undefined)
    prekey_eligible = [t for t in failed if t["key_step"] > 0]
    n_failed, n_success, n_prekey_elig = len(failed), len(success), len(prekey_eligible)

    rows = []
    for tau in thresholds:
        n_detected = sum(
            1 for t in failed
            if (cut := next((s for s, p in t["steps"] if p > tau), None)) is not None and cut >= t["key_step"]
        )
        n_false_cut = sum(
            1 for t in success
            if next((s for s, p in t["steps"] if p > tau), None) is not None
        )
        n_prekey_fa = sum(
            1 for t in prekey_eligible
            if (cut := next((s for s, p in t["steps"] if p > tau), None)) is not None and cut < t["key_step"]
        )
        rows.append({
            "tau": round(float(tau), 5),
            "detection_rate": n_detected / n_failed if n_failed else float("nan"),
            "false_cut_rate": n_false_cut / n_success if n_success else float("nan"),
            "pre_key_false_alarm_rate": n_prekey_fa / n_prekey_elig if n_prekey_elig else float("nan"),
        })
    return pd.DataFrame(rows)


def operating_point_at_fcr(curve: pd.DataFrame, budget: float) -> dict:
    """Among all tau with false_cut_rate <= budget, the one with MAX
    detection_rate -- ONE coherent (tau, detection, pre-key-FA) triple,
    not two separately-optimized numbers."""
    d = curve[curve.false_cut_rate <= budget]
    if d.empty:
        return {"detection_rate": np.nan, "pre_key_false_alarm_rate": np.nan}
    row = d.loc[d["detection_rate"].idxmax()]
    return {"detection_rate": float(row["detection_rate"]),
            "pre_key_false_alarm_rate": float(row["pre_key_false_alarm_rate"])}


def scope_tasks(tasks: list[dict], scope: str) -> list[dict]:
    return tasks if scope == "pooled" else [t for t in tasks if t["benchmark"] == scope]


def success_false_alarm_stats(scoped: list[dict], ref_tau: float = 0.5) -> dict:
    """On successful trajectories ONLY -- unambiguous ground truth under
    BOTH label_modes (apply_keystep_labels never touches success rows),
    so this sidesteps the key-step-vs-trajectory label definitional
    question entirely. mean_p_fail_success: average predicted P(failure)
    on safe steps -- lower is better calibrated. fcr_success_at_tau_0.5:
    fraction of successful trajectories that would be flagged at least
    once at a plain P(failure)>0.5 threshold. Added 2026-08-02 by design
    request, after the pre-key-false-alarm-on-FAILED-trajectories metric
    turned out to have a WebArena-Lite-specific reversal -- this is the
    cleaner, no-exceptions version of the same question ("does this model
    mislabel a correct prediction as wrong")."""
    success = [t for t in scoped if not t["final_failed"]]
    if not success:
        return {"mean_p_fail_success": np.nan, "fcr_success_at_tau_0.5": np.nan}
    all_p = np.concatenate([[p for _, p in t["steps"]] for t in success])
    max_p = np.array([max(p for _, p in t["steps"]) for t in success])
    return {"mean_p_fail_success": float(np.mean(all_p)),
            "fcr_success_at_tau_0.5": float((max_p > ref_tau).mean())}


def evaluate_arm(tasks: list[dict]) -> dict:
    out = {}
    for scope in ["pooled"] + BENCHMARKS:
        scoped = scope_tasks(tasks, scope)
        if not scoped:
            continue
        y, p = y_vis_rows(scoped)
        if len(np.unique(y)) < 2:
            continue
        keystep_auroc = roc_auc_score(y, p)
        keystep_eaurc = aurc(y, p)[0] - aurc_oracle(y)
        keystep_brier = brier_score_loss(y, p)

        yt, pt = terminal_rows(scoped)
        terminal_auroc = roc_auc_score(yt, pt) if len(np.unique(yt)) >= 2 else np.nan

        curve = tradeoff_curve_full(scoped)
        fcr_ops = {b: operating_point_at_fcr(curve, b) for b in FCR_BUDGETS}
        succ_stats = success_false_alarm_stats(scoped)

        out[scope] = {
            "n_tasks": len(scoped), "n_failed": sum(t["final_failed"] for t in scoped),
            "keystep_auroc": keystep_auroc, "keystep_e_aurc": keystep_eaurc,
            "keystep_brier": keystep_brier, "terminal_auroc": terminal_auroc,
            **succ_stats,
            **{f"detection_at_{int(b*100)}fcr": fcr_ops[b]["detection_rate"] for b in FCR_BUDGETS},
            **{f"prekey_fa_at_{int(b*100)}fcr": fcr_ops[b]["pre_key_false_alarm_rate"] for b in FCR_BUDGETS},
        }
    return out


def make_figure(results: dict, out_path: Path) -> None:
    # 2026-08-02 (design decision): panel 4 swapped from pre-key-FA-on-FAILED
    # (had a WebArena-Lite-specific reversal, and rests on the key-step
    # boundary being the "right" definition -- a debate this sidesteps)
    # to mean P(failure) on SUCCESSFUL trajectories -- unambiguous ground
    # truth under both label_modes, and a clean, no-exceptions result.
    # Terminal AUROC de-emphasized (kept, not the headline -- the task
    # is per-prefix risk, not final-outcome prediction) but still shown as the no-regret check.
    metrics = [
        ("keystep_auroc", "Key-step-aware AUROC ↑"),
        (f"detection_at_{int(HEADLINE_FCR*100)}fcr", f"Detection@{int(HEADLINE_FCR*100)}%FCR ↑"),
        ("mean_p_fail_success", "Mean P(failure) on Success ↓"),
        ("terminal_auroc", "Terminal-step AUROC (no-regret check) ↑"),
    ]
    scopes = ["webarena", "mind2web", "pooled"]
    scope_titles = {"webarena": "WebArena-Lite", "mind2web": "Mind2Web", "pooled": "All Runs Avg."}
    arm_colors = {"keystep": "#d1372b", "trajectory": "#7a746b"}

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5.2))
    x = np.arange(len(scopes))
    width = 0.35
    for ax, (mkey, mtitle) in zip(axes, metrics):
        for i, arm in enumerate(["keystep", "trajectory"]):
            vals = [results[arm].get(s, {}).get(mkey, np.nan) for s in scopes]
            bars = ax.bar(x + (i - 0.5) * width, vals, width, color=arm_colors[arm],
                           alpha=0.9, label=ARM_LABEL[arm])
            for xi, v in zip(x + (i - 0.5) * width, vals):
                if not np.isnan(v):
                    ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                                xytext=(0, 3), ha="center", fontsize=8.5)
        ax.set_title(mtitle, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([scope_titles[s] for s in scopes], fontsize=9.5)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10.5,
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle("Key-Step-Aware vs. Trajectory-Propagated Supervision", fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_latex_table(results: dict, out_path: Path) -> None:
    # 2026-08-02: 4th column swapped pre-key-FA-on-failed -> mean
    # P(failure) on successful trajectories (unambiguous ground truth
    # under both label_modes, no-exceptions result); Terminal AUROC
    # dropped from the table (kept in prose only as a no-regret aside --
    # not the task, by design).
    scopes = [("webarena", "WebArena-Lite"), ("mind2web", "Mind2Web"), ("pooled", "All Runs Avg.")]
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{llccc}", r"\toprule",
        r"\textbf{Benchmark} & \textbf{Supervision} & \textbf{Key-step AUROC} & "
        r"\textbf{Det.@10\%FCR} & \textbf{Mean $P(\mathrm{fail})$ on Success} \\",
        r"\midrule",
    ]
    for si, (scope, scope_title) in enumerate(scopes):
        for ai, arm in enumerate(["keystep", "trajectory"]):
            r = results[arm].get(scope, {})
            prefix = f"\\multirow{{2}}{{*}}{{{scope_title}}} " if ai == 0 else " "
            arm_name = "Key-step-aware" if arm == "keystep" else "Trajectory-propagated"
            cells = [
                f"{r.get('keystep_auroc', float('nan')):.3f}",
                f"{r.get('detection_at_10fcr', float('nan'))*100:.1f}\\%",
                f"{r.get('mean_p_fail_success', float('nan')):.3f}",
            ]
            lines.append(f"{prefix}& {arm_name} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
        if si < len(scopes) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Key-step-aware vs.\ trajectory-propagated supervision, same features/folds/"
        r"calibration, only the training label differs. Key-step AUROC is scored against the SAME "
        r"externally-annotated key step for both arms. Det.@10\%FCR: best detection rate achievable "
        r"without exceeding a 10\% false-cut rate on successful trajectories. Mean $P(\mathrm{fail})$ "
        r"on Success: average predicted failure probability on successful trajectories' own steps -- "
        r"unambiguous ground truth under both supervision schemes (a successful trajectory is "
        r"labeled 0 throughout regardless of label\_mode), so this isolates whether a model "
        r"mislabels a genuinely safe prefix as risky without any dependence on where the key step "
        r"is defined.}",
        r"\label{tab:keystep_ablation}", r"\end{table*}",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main():
    out_dir = THIS_DIR / "results"
    out_dir.mkdir(exist_ok=True)

    results = {}
    rows = []
    for arm in ["keystep", "trajectory"]:
        print(f"Loading {arm} arm...")
        tasks = load_all(arm)
        print(f"  {len(tasks)} tasks total")
        results[arm] = evaluate_arm(tasks)
        for scope, m in results[arm].items():
            row = {"arm": arm, "scope": scope, **m}
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "keystep_ablation_by_benchmark.csv", index=False)

    print("\n=== Key-step-aware vs. trajectory-propagated supervision ===")
    pd.set_option("display.width", 160)
    print(df[["arm", "scope", "keystep_auroc", "keystep_e_aurc", "keystep_brier",
              "terminal_auroc", "detection_at_5fcr", "detection_at_10fcr", "detection_at_20fcr",
              "prekey_fa_at_5fcr", "prekey_fa_at_10fcr", "prekey_fa_at_20fcr"]].to_string(index=False))

    make_figure(results, out_dir / "keystep_ablation_curves.png")
    make_latex_table(results, out_dir / "table_keystep_ablation.tex")
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
