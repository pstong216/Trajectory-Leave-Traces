#!/usr/bin/env python3
"""
run_success_false_alarm.py — On successful trajectories ONLY, which model
is more likely to mislabel a correct (safe) prefix as wrong (predicted
failure)? (Follow-up to the key-step ablation.)

Deliberately restricted to successful trajectories specifically because
the ground truth there is unambiguous under BOTH training schemes: a
successful trajectory is labeled 0 at every step whether the model was
trained with --label_mode keystep or --label_mode trajectory (only
FAILED trajectories differ between the two labeling schemes --
apply_keystep_labels() never touches success rows). So this sidesteps the
whole "which label definition is correct" framing question entirely --
there's no disagreement about what the right answer is on this
population, only about which model gets it right more often.

For every threshold tau, False Cut Rate = fraction of successful
trajectories that would be flagged (any step's P(failure) > tau) --
reusing the exact same quantity already computed in
run_keystep_ablation.py's tradeoff_curve_full(), just reported across the
FULL threshold sweep here instead of at 3 fixed FCR budgets. Also reports
two threshold-free summaries: mean predicted P(failure) on
successful-trajectory steps, and the fraction of those steps with
P(failure) > 0.5.

Outputs (code/keystep_ablation/results/):
  success_false_alarm_curve.csv     arm x scope x tau -> false_cut_rate
  success_false_alarm_summary.csv   arm x scope -> mean_p_fail, frac_p_gt_0.5, fcr@{0.3,0.5,0.7}
  success_false_alarm_curves.png
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

REF_TAUS = [0.3, 0.5, 0.7]


def false_cut_curve(tasks: list[dict], n_thresholds: int = 200) -> pd.DataFrame:
    """Successful trajectories only. For each tau: fraction whose max
    step P(failure) exceeds tau (i.e. would be flagged at least once)."""
    success = [t for t in tasks if not t["final_failed"]]
    if not success:
        return pd.DataFrame()
    max_p = np.array([max(p for _, p in t["steps"]) for t in success])
    all_p = np.concatenate([[p for _, p in t["steps"]] for t in success])
    thresholds = np.unique(np.concatenate([
        np.linspace(all_p.min(), all_p.max(), n_thresholds), [all_p.max() + 1e-6],
    ]))[::-1]
    rows = [{"tau": round(float(tau), 5), "false_cut_rate": float((max_p > tau).mean())}
            for tau in thresholds]
    return pd.DataFrame(rows)


def summary_stats(tasks: list[dict]) -> dict:
    success = [t for t in tasks if not t["final_failed"]]
    all_p = np.concatenate([[p for _, p in t["steps"]] for t in success])
    out = {"n_success_tasks": len(success), "n_success_steps": len(all_p),
           "mean_p_fail": float(np.mean(all_p)), "frac_steps_p_gt_0.5": float((all_p > 0.5).mean())}
    for tau in REF_TAUS:
        max_p = np.array([max(p for _, p in t["steps"]) for t in success])
        out[f"fcr_at_tau_{tau}"] = float((max_p > tau).mean())
    return out


def main():
    out_dir = THIS_DIR / "results"
    out_dir.mkdir(exist_ok=True)

    arms = {}
    for arm in ["keystep", "trajectory"]:
        print(f"Loading {arm} arm...")
        arms[arm] = load_all(arm)

    curve_rows, summary_rows = [], []
    for arm, tasks in arms.items():
        for scope in ["pooled"] + BENCHMARKS:
            scoped = scope_tasks(tasks, scope)
            curve = false_cut_curve(scoped)
            if curve.empty:
                continue
            curve["arm"] = arm
            curve["scope"] = scope
            curve_rows.append(curve)
            stats = summary_stats(scoped)
            summary_rows.append({"arm": arm, "scope": scope, **stats})

    curve_df = pd.concat(curve_rows, ignore_index=True)
    curve_df.to_csv(out_dir / "success_false_alarm_curve.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "success_false_alarm_summary.csv", index=False)

    print("\n=== On successful trajectories only: does each model call safe prefixes 'failure'? ===")
    pd.set_option("display.width", 160)
    print(summary_df.round(4).to_string(index=False))

    # figure: tau (x) vs false_cut_rate (y), one line per arm, faceted by
    # benchmark only (no pooled panel, larger fonts, no suptitle)
    fig, axes = plt.subplots(1, len(BENCHMARKS), figsize=(14, 6.4), sharey=True)
    titles = {"webarena": "WebArena-Lite", "mind2web": "Mind2Web"}
    colors = {"keystep": "#d1372b", "trajectory": "#7a746b"}
    for ax, scope in zip(axes, BENCHMARKS):
        for arm in ["keystep", "trajectory"]:
            d = curve_df[(curve_df.arm == arm) & (curve_df.scope == scope)].sort_values("tau")
            ax.plot(d["tau"], d["false_cut_rate"], color=colors[arm], linewidth=3.0,
                     label=ARM_LABEL[arm])
        ax.set_title(titles[scope], fontsize=24, fontweight="bold")
        ax.set_xlabel(r"Threshold $\gamma$", fontsize=20)
        ax.tick_params(axis="both", labelsize=17)
        ax.invert_xaxis()
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("False Cut Rate on successful trajectories", fontsize=20)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=19,
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "success_false_alarm_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
