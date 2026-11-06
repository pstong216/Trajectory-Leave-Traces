#!/usr/bin/env python3
"""
Regenerates the paper's `success_false_alarm_curves.pdf/.png` figure
(the 6-line feature-set x arm breakdown, restricted to successful
trajectories -- same data as run_feature_set_false_alarm.py's
feature_set_false_alarm_curve.csv) with a layout tuned specifically for
this figure: no title/xlabel/legend overlap, and a LARGER legend font
by design (2026-08-18).

Reads the already-computed curve CSV (does not recompute anything).
Writes directly into the paper's Overleaf-synced fig/ directory.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
CURVE_CSV = THIS_DIR / "results" / "feature_set_false_alarm_curve.csv"
OUT_DIR = THIS_DIR.parent.parent / "figures"   # repository root /figures
BENCHMARKS = ["webarena", "mind2web"]
LEGEND_FONTSIZE = 44
LEGEND_LINEWIDTH = 8.0

FEATURE_SETS = ["macro_only", "uq_only", "all"]
ARMS = ["keystep", "trajectory"]
FS_LABEL = {"macro_only": "Macro-only", "uq_only": "Micro-only", "all": "Macro&Micro"}
FS_COLOR = {"macro_only": "#ffb041", "uq_only": "#641724", "all": "#f11818"}
ARM_LS = {"keystep": "-", "trajectory": "--"}
ARM_SHORT = {"keystep": "key-step", "trajectory": "trajectory"}


def main():
    curve_df = pd.read_csv(CURVE_CSV)

    fig, axes = plt.subplots(1, len(BENCHMARKS), figsize=(24, 12), sharey=True)
    titles = {"webarena": "WebArena-Lite", "mind2web": "Mind2Web"}
    for ax, scope in zip(axes, BENCHMARKS):
        for arm in ARMS:
            for fs in FEATURE_SETS:
                d = curve_df[(curve_df.feature_set == fs) & (curve_df.arm == arm)
                              & (curve_df.scope == scope)].sort_values("tau")
                ax.plot(d["tau"], d["false_cut_rate"], color=FS_COLOR[fs], linestyle=ARM_LS[arm],
                         linewidth=4.0, alpha=0.95 if arm == "keystep" else 0.85,
                         label=f"{FS_LABEL[fs]} ({ARM_SHORT[arm]})")
        ax.set_title(titles[scope], fontsize=40, fontweight="bold")
        ax.set_xlabel(r"Threshold $\gamma$", fontsize=30)
        ax.tick_params(axis="both", labelsize=38)
        ax.invert_xaxis()
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("False Cut Rate", fontsize=38)
    handles, labels = axes[0].get_legend_handles_labels()
    # legend font intentionally larger than axis labels/ticks by design
    # Keep the plot area fixed and place the legend below the canvas.  This
    # prevents a larger legend from shrinking either benchmark panel.
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.14, wspace=0.16)
    legend = fig.legend(handles, labels, loc="upper center", ncol=2,
                        fontsize=LEGEND_FONTSIZE,
                        bbox_to_anchor=(0.08, 0.02, 0.90, 0.0), mode="expand",
                        frameon=False, columnspacing=1.1, handletextpad=0.6,
                        borderaxespad=0.0)
    for legend_line in legend.get_lines():
        legend_line.set_linewidth(LEGEND_LINEWIDTH)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        out_path = OUT_DIR / f"success_false_alarm_curves.{ext}"
        fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
        print(f"Saved -> {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
