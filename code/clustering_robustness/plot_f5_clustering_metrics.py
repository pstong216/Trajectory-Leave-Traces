#!/usr/bin/env python3
"""Appendix clustering-robustness figure, recomputed against the CURRENT labels.

Why this replaces the earlier asset
-----------------------------------
The previous f5_clustering_metrics.pdf was built from May-2026 training runs
(`uq_results/.../2026052*`), which predate the key-step annotation revision:
for webarena/gpt_5_2 that snapshot disagrees with the current
keystep_labels/webarena/gpt_5_2_ours.json on 46 of 125 tasks (20.0% of shared
prefixes). It also reported the *best over ~13 model families* and used the old
`aurc` column (negative), whereas the paper reports Logistic_L2 and E-AURC.

Clustering itself is label-independent, so nothing needed re-embedding: the nine
feature matrices were reused unchanged and only the downstream predictor was
retrained. Reproduced with:

    for enc in qwen bge_m3 minilm; do for thr in 0p2 0p3 0p4; do
      PYTHONHASHSEED=42 python3 code/l2_training/train_risk_model_temp_sca.py \
        --source webarena/gpt_5_2 --uq_dir uq_analysis_change_$enc \
        --csv_path data/webarena/gpt_5_2/embedded/uq_analysis_change_$enc/thr_$thr/feature_matrix.csv \
        --annot_path keystep_labels/webarena/gpt_5_2_ours.json --output_root <out>
    done; done

Sanity check: qwen/thr_0p3 reproduces reproduction_output/ours/webarena_gpt_5_2/
.../20260801_144231/summary_table.csv byte for byte.

Note f5b_aggregate_micro_stability.pdf is NOT affected -- it is a Spearman
correlation between Micro scores across settings and never touches labels.
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent            # repository root
DEFAULT_OUT = REPO / "figures" / "f5_clustering_metrics.pdf"
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--data", type=Path, default=HERE / "clustering_relabeled_summary.csv")
_ap.add_argument("--stability", type=Path, default=HERE / "f5b_aggregate_micro_stability_data.csv")
_ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
A = _ap.parse_args()

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # slots 1-3, all-pairs validated
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
ENC = [("qwen", "Qwen", BLUE, "o", "-"),
       ("minilm", "Sentence-BERT", ORANGE, "s", (0, (4.5, 1.8))),
       ("bge_m3", "BGE-M3", AQUA, "^", (0, (1.2, 1.2)))]

df = pd.read_csv(A.data)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 6.8, "axes.labelsize": 6.6, "axes.titlesize": 6.9,
    "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "legend.fontsize": 6.4,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
})

# Row-major 3x2: each row is one metric, columns are the two feature sets, so a
# reader compares Micro-only against Macro+Micro by looking left-to-right.
PANELS = [("Micro-only", "auroc", "AUROC"), ("Micro-only", "e_aurc", "E-AURC"),
          ("Micro-only", "brier", "Brier"),
          ("Macro+Micro", "auroc", "AUROC"), ("Macro+Micro", "e_aurc", "E-AURC"),
          ("Macro+Micro", "brier", "Brier")]

# Shared limits per metric across both rows, so Micro-only and Macro+Micro stay
# comparable. Padding is deliberately modest: autoscaling each panel would blow a
# <0.01 spread up to full height, but too much padding flattens every line into a
# featureless horizontal rule. Each panel also prints its own span, so the reader
# gets the magnitude from a number rather than from pixel height.
YLIM = {}
for m in ["auroc", "e_aurc", "brier"]:
    lo, hi = df[m].min(), df[m].max()
    pad = max((hi - lo) * 0.35, 0.006)
    YLIM[m] = (lo - pad, hi + pad)

# One figure, not two: as separate floats the metric grid and the stability plot
# kept getting queued onto different appendix pages, leaving the small stability
# panel stranded on a page of its own.
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(6.3, 2.75))
gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 0.92],
              left=0.062, right=0.988, top=0.845, bottom=0.135, hspace=0.72, wspace=0.40)
axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

flat = [a for row in axes for a in row]
for ax, (fs, metric, mlabel), tag in zip(flat, PANELS, "abcdef"):
    for key, disp, color, mk, ls in ENC:
        s = df[(df.feature_set == fs) & (df.encoder == key)].sort_values("threshold")
        ax.plot(s.threshold, s[metric], color=color, marker=mk, ms=2.6, mew=0,
                lw=1.4, ls=ls, label=disp, clip_on=False)
    arrow = r"$\uparrow$" if metric == "auroc" else r"$\downarrow$"
    ax.set_title(f"({tag}) {fs} {mlabel} {arrow}", pad=4)
    ax.set_ylim(*YLIM[metric])
    ax.set_xlabel(r"clustering threshold $\delta$", labelpad=1.5)
    ax.set_xticks([0.2, 0.3, 0.4])
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    sub = df[df.feature_set == fs][metric]
    ax.text(0.97, 0.05, f"span {sub.max() - sub.min():.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.6, color=MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

handles, labels = axes[0][0].get_legend_handles_labels()
leg = fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
                 bbox_to_anchor=(0.5, 1.002), handlelength=1.7, columnspacing=0.9,
                 handletextpad=0.5)
for t in leg.get_texts():
    t.set_color(INK)

# ---- bottom panel: aggregate-Micro stability across the 36 setting pairs ----
st = pd.read_csv(A.stability)
axb = fig.add_subplot(gs[:, 3])
rng = __import__("numpy").random.default_rng(0)
x = rng.normal(0, 0.055, len(st))
axb.scatter(x, st.spearman, s=7, color=BLUE, alpha=0.75, linewidths=0, zorder=3)
med = st.spearman.median()
axb.axhline(med, color=ORANGE, lw=1.4, zorder=4)
axb.annotate(f"median {med:.3f}", xy=(0.30, med), xytext=(0, 3), textcoords="offset points",
             color=ORANGE, fontsize=5.8, ha="right", va="bottom")
axb.annotate(f"min {st.spearman.min():.3f}", xy=(0.30, st.spearman.min()), xytext=(0, -1),
             textcoords="offset points", color=MUTED, fontsize=5.8, ha="right", va="top")
axb.set_xlim(-0.32, 0.32); axb.set_xticks([])
axb.set_ylabel("Spearman corr., 36 setting pairs", labelpad=2)
axb.set_title("(g) Aggregate Micro\nstability", pad=4, fontsize=6.6)
axb.grid(axis="y", color=GRID, lw=0.5, alpha=0.7); axb.set_axisbelow(True)
for sp in ("top", "right", "bottom"):
    axb.spines[sp].set_visible(False)

fig.savefig(A.out, bbox_inches="tight", pad_inches=0.014)
fig.savefig(HERE / "f5_clustering_metrics.png", dpi=230, bbox_inches="tight", pad_inches=0.014)
print("wrote", A.out)
