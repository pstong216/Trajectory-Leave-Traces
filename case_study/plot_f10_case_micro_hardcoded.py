#!/usr/bin/env python3
"""MAIN-TEXT figure -- Macro / Micro / Macro&Micro on one trajectory (single column).

Trajectory: WebArena-Lite / Qwen3-VL-30B / task_id 163 (30 prefixes)
  k*  ..... 8    cause Action:ParameterError
  notes ... "The agent repeatedly targeted the wrong element IDs (such as clicking
            'Previous Month' or day buttons) instead of selecting the year 2030 from
            the dropdown, eventually leading to a step limit exhaustion after
            canceling and restarting the form."

Why this trajectory is in the appendix and not the main text: it comes from the
**plain** rollout, which is where Micro's N=10 sampling was run. HTC's logprobs come
from the **verbalized** rollout -- a different run of the same task (only 13 tasks
across all settings have identical action sequences in both, and Micro is flat in
every one of them). So this figure makes NO internal-signal claim; it only compares
our own three feature sets, which are all computed on this one rollout and therefore
genuinely paired. The main-text Figure 6 carries the observable-vs-internal claim on
a properly paired population.

Provenance (under results_total/):
  curves .......... reproduction_output/ours/webarena_qwen30b/thr_0p3/
                    train_risk_model_temp_sca/*20260801*/cv_predictions_{all,macro_only,uq_only}.csv
                    (out-of-fold Logistic_L2 predictions; task 163 held out of training)
  k* / cause ...... keystep_labels/webarena/qwen30b_ours.json
  actions ......... data/webarena/qwen30b/raw/plain/actions/163.json
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "figures" / "f10_case_micro.pdf"
_ap = argparse.ArgumentParser(description="Render the appendix Macro/Micro case figure.")
_ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
OUT = _ap.parse_args().out

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # slots 1-3, all-pairs validated
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

R = json.loads((HERE / "case163_risk.json").read_text())
k = R["step_k"]
SERIES = [
    ("Macro",        R["p_macro"], BLUE,   "-",              "o"),
    ("Micro",        R["p_micro"], ORANGE, (0, (4.5, 1.8)),  "s"),
    ("Macro&Micro",  R["p_all"],   AQUA,   (0, (1.2, 1.2)),  "^"),
]
KSTAR, TAU = 8, 0.5

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7.2, "axes.labelsize": 7.2,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
})

fig, ax = plt.subplots(figsize=(3.17, 1.95))
fig.subplots_adjust(left=0.155, right=0.985, top=0.845, bottom=0.185)

ax.axvspan(KSTAR, k[-1], color="#f4a58c", alpha=0.12, lw=0)
ax.axhline(TAU, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
ax.axvline(KSTAR, color=MUTED, lw=0.9, ls=(0, (1, 1.6)), zorder=1)

for name, y, c, ls, mk in SERIES:
    ax.plot(k, y, color=c, lw=1.6, ls=ls, marker=mk, ms=2.6, mew=0,
            label=name, zorder=4, clip_on=False)

ax.text(KSTAR + 0.4, 1.03, r"$k^\star$", color=MUTED, fontsize=7, va="bottom")
ax.text(k[-1] - 0.3, TAU + 0.035, r"$\gamma$", color=MUTED, fontsize=6.8,
        ha="right", va="bottom")

# two short callouts; the full narrative lives in the text
ax.annotate("repeats the\nsame click", xy=(9, 0.82), xytext=(0.6, 0.62),
            color=INK, fontsize=6.2, ha="left", va="center", linespacing=1.1,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.55, shrinkA=1, shrinkB=3))
ax.annotate("decision\ndestabilizes", xy=(11, 0.955), xytext=(13.5, 0.55),
            color=INK, fontsize=6.2, ha="left", va="center", linespacing=1.1,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.55, shrinkA=1, shrinkB=3))

ax.set_xlim(0, k[-1]); ax.set_ylim(0, 1.0)
ax.set_xlabel("prefix step $k$", labelpad=2)
ax.set_ylabel("predicted prefix risk", labelpad=3)
ax.set_yticks([0, 0.5, 1.0])
ax.set_xticks([0, 8, 15, 22, 29])
ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.7); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
leg = ax.legend(loc="upper left", bbox_to_anchor=(-0.03, 1.30), ncol=3, frameon=False,
                handlelength=1.5, handletextpad=0.4, columnspacing=0.8, borderpad=0)
for t in leg.get_texts(): t.set_color(INK)

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.014)
fig.savefig(HERE / "f10_case_micro.png", dpi=230, bbox_inches="tight", pad_inches=0.014)
print("wrote", OUT)
