#!/usr/bin/env python3
"""
run_feature_set_false_alarm.py — Same success-only false-alarm check as
run_success_false_alarm.py, but comparing FEATURE SETS (Macro-only vs.
Micro-only vs. Macro&Micro), CROSSED with training label (key-step-aware
vs. trajectory-propagated) -- 6 lines per panel (the trajectory-propagated arm
is shown alongside the key-step arm for direct comparison).

Restricted to successful trajectories throughout, same reasoning as
run_success_false_alarm.py: ground truth is unambiguous there (always 0)
under BOTH label_modes, so this isolates which feature set / label
combination is most prone to flagging a genuinely safe prefix as risky,
independent of any key-step definitional question.

Outputs (code/keystep_ablation/results/):
  feature_set_false_alarm_curve.csv     feature_set x arm x scope x tau -> false_cut_rate
  feature_set_false_alarm_summary.csv   feature_set x arm x scope -> mean_p_fail, frac_p_gt_0.5, fcr@{0.3,0.5,0.7}
  feature_set_false_alarm_curves.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from run_keystep_ablation import load_all, scope_tasks, BENCHMARKS, ARM_LABEL  # noqa: E402
from run_success_false_alarm import false_cut_curve, summary_stats, REF_TAUS  # noqa: E402

FEATURE_SETS = ["macro_only", "uq_only", "all"]
ARMS = ["keystep", "trajectory"]
FS_LABEL = {"macro_only": "Macro-only", "uq_only": "Micro-only", "all": "Macro&Micro"}
FS_COLOR = {"macro_only": "#f0876e", "uq_only": "#8f2a3a", "all": "#d1372b"}
ARM_LS = {"keystep": "-", "trajectory": "--"}
ARM_SHORT = {"keystep": "key-step", "trajectory": "trajectory"}


def main():
    out_dir = THIS_DIR / "results"
    out_dir.mkdir(exist_ok=True)

    tasks_by = {}
    for arm in ARMS:
        for fs in FEATURE_SETS:
            print(f"Loading {arm} arm, feature_set={fs}...")
            tasks_by[(arm, fs)] = load_all(arm, variant=fs)

    curve_rows, summary_rows = [], []
    for (arm, fs), tasks in tasks_by.items():
        for scope in ["pooled"] + BENCHMARKS:
            scoped = scope_tasks(tasks, scope)
            curve = false_cut_curve(scoped)
            if curve.empty:
                continue
            curve["feature_set"] = fs
            curve["arm"] = arm
            curve["scope"] = scope
            curve_rows.append(curve)
            stats = summary_stats(scoped)
            summary_rows.append({"feature_set": fs, "arm": arm, "scope": scope, **stats})

    curve_df = pd.concat(curve_rows, ignore_index=True)
    curve_df.to_csv(out_dir / "feature_set_false_alarm_curve.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "feature_set_false_alarm_summary.csv", index=False)

    print("\n=== On successful trajectories only, by feature set x arm: mislabel rate ===")
    pd.set_option("display.width", 160)
    print(summary_df.round(4).to_string(index=False))

    # Font scale matched to generalization/detection_false_cut_analysis/
    # run_detection_false_cut_regions.py's detection_false_cut_curves.png
    # (title=38, xlabel/ylabel=31, tick=24, legend=26, figsize=(21,9.5),
    # wspace=0.25) -- same figsize+fontsize values, not just proportionally
    # scaled, so the two figures read at an identical visual scale.
    fig, axes = plt.subplots(1, len(BENCHMARKS), figsize=(21, 9.5), sharey=True,
                              gridspec_kw={"wspace": 0.25})
    titles = {"webarena": "WebArena-Lite", "mind2web": "Mind2Web"}
    for ax, scope in zip(axes, BENCHMARKS):
        for arm in ARMS:
            for fs in FEATURE_SETS:
                d = curve_df[(curve_df.feature_set == fs) & (curve_df.arm == arm)
                              & (curve_df.scope == scope)].sort_values("tau")
                ax.plot(d["tau"], d["false_cut_rate"], color=FS_COLOR[fs], linestyle=ARM_LS[arm],
                         linewidth=3.0, alpha=0.95 if arm == "keystep" else 0.85,
                         label=f"{FS_LABEL[fs]} ({ARM_SHORT[arm]})")
        ax.set_title(titles[scope], fontsize=38, fontweight="bold")
        ax.set_xlabel(r"Threshold $\gamma$", fontsize=31)
        ax.tick_params(axis="both", labelsize=24)
        ax.invert_xaxis()
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("False Cut Rate on successful trajectories", fontsize=31)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=26,
               bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_set_false_alarm_curves.png", dpi=200,
                bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
