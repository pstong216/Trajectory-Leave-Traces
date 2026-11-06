#!/usr/bin/env python3
"""Figure 6 (Sec 4.5 case study) -- self-contained, no pipeline dependency.

Renders ../f10_case_study.pdf from the values inlined below.

Trajectory: Online Mind2Web / GPT-5.2 / task_id 173
  task ......... "Find young cats in Seattle and show off the newest additions."
  outcome ...... environment early-stop at step 16, "Same typing action for 3 times"
  k*  .......... 3      (first prefix labelled tending-to-fail)
  cause ........ Reflection:OutcomeMisinterpretation
  alarm ........ observable risk crosses gamma=0.5 at step 6; HTC never crosses (max 0.453)

Provenance of the numbers (all under results_total/):
  p_macro / p_htc   code/case_study/dump_preds.py -> preds/preds_mind2web_gpt_5_2.csv
                    (Logistic_L2, 5-fold, folds shared across feature sets; reproduces
                     code/macro_internal/results/macro_internal_per_combo.csv exactly)
  macro features    run_macro_internal.build_joined("mind2web","gpt_5_2"), join_id 173
  k* / cause        keystep_labels/mind2web/gpt_5_2_baseline.json
                    (BASELINE role -- this population is the verbalized rollout)
  action history    data/mind2web/gpt_5_2/raw/verbalized/173/result.json

To regenerate the inputs rather than trust these constants, run
    python3 dump_preds.py --combos mind2web/gpt_5_2
in this same directory.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "figures" / "f11_case_internal.pdf"
_ap = argparse.ArgumentParser(description="Render the Sec 4.5 case-study figure.")
_ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                 help="destination PDF (default: the paper's latex/fig/)")
OUT = _ap.parse_args().out

BLUE, ORANGE = "#2a78d6", "#eb6834"   # categorical slots 1 & 2 (CVD-validated pair)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

R = json.loads((HERE / "case173_risk.json").read_text())
k, pm, ph = R["step_k"], R["p_macro"], R["p_htc"]

FEATS = [
    ("action\nrepeat rate", [1.00, 1.00, 1.00, 1.00, 1.25, 1.50, 1.75, 2.00,
                             1.80, 2.00, 1.83, 2.00, 2.17, 2.33, 2.50, 2.67]),
]
# other Macro features quoted in the text, kept for reference / larger variants:
#   steps_since_new_action [0,0,0,0,1,2,3,4,0,1,0,1,2,3,4,5]
#   n_double_loops         [0,0,0,0,0,1,1,2,2,2,2,2,2,2,2,2]
#   n_distinct_actions     [1,2,3,4,4,4,4,4,5,5,6,6,6,6,6,6]   (stalls at 4 over k=3..7)

KSTAR, ALARM, TAU = 3, 6, 0.5

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7.2, "axes.labelsize": 7.2,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
})

fig = plt.figure(figsize=(3.17, 1.80))
gs = GridSpec(2, 1, height_ratios=[2.9, 1], hspace=0.26,
              left=0.155, right=0.985, top=0.835, bottom=0.175)

ax = fig.add_subplot(gs[0])
ax.axvspan(KSTAR, k[-1], color="#f4a58c", alpha=0.13, lw=0)
ax.axhline(TAU, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
ax.axvline(KSTAR, color=MUTED, lw=0.7, ls=(0, (1, 1.6)), zorder=1)
ax.plot(k, pm, color=BLUE, lw=1.7, marker="o", ms=2.9, mew=0,
        label="Observable (Macro)", zorder=4, clip_on=False)
ax.plot(k, ph, color=ORANGE, lw=1.7, ls=(0, (4.5, 1.8)), marker="s", ms=2.6, mew=0,
        label="Internal (HTC)", zorder=3, clip_on=False)
ax.plot([ALARM], [pm[ALARM]], marker="o", ms=6.4, mfc="none",
        mec=BLUE, mew=1.25, zorder=5, clip_on=False)
ax.annotate("alarm $k{=}6$", xy=(ALARM, pm[ALARM]), xytext=(2.75, 0.80),
            color=BLUE, ha="left", va="center", fontsize=6.6,
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7, shrinkA=2, shrinkB=5))
ax.text(0.28, TAU + 0.035, r"threshold $\gamma$", color=MUTED, fontsize=6.4, va="bottom")
ax.text(KSTAR + 0.22, 1.055, r"$k^\star$", color=MUTED, fontsize=7.0, va="bottom")
ax.text(k[-1], ph[-1] - 0.10, "never fires", color=ORANGE, fontsize=6.6,
        ha="right", va="top")
ax.set_ylim(0, 1.0); ax.set_xlim(0, k[-1])
ax.set_ylabel("predicted prefix risk", labelpad=2)
ax.set_yticks([0, 0.5, 1.0]); ax.set_xticklabels([])
ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.7); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
leg = ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.30), ncol=2, frameon=False,
                handlelength=1.9, columnspacing=1.1, handletextpad=0.5, borderpad=0)
for t in leg.get_texts(): t.set_color(INK)

for row, (name, vals) in enumerate(FEATS, start=1):
    a = fig.add_subplot(gs[row])
    a.axvspan(KSTAR, k[-1], color="#f4a58c", alpha=0.13, lw=0)
    a.axvline(KSTAR, color=MUTED, lw=0.7, ls=(0, (1, 1.6)))
    a.axvline(ALARM, color=BLUE, lw=0.7, alpha=0.55)
    a.plot(k, vals, color=BLUE, lw=1.3, marker="o", ms=2.1, mew=0, clip_on=False)
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.30 or 0.4
    a.set_ylim(lo - pad, hi + pad); a.set_yticks([lo, hi])
    a.set_ylabel(name, fontsize=5.9, labelpad=2, color=MUTED, linespacing=1.05)
    a.set_xlim(0, k[-1])
    for s in ("top", "right"): a.spines[s].set_visible(False)
    if row < len(FEATS):
        a.set_xticklabels([])
    else:
        a.set_xlabel("prefix step $k$", labelpad=1.5)
        a.set_xticks([0, 3, 6, 9, 12, 15])

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.012)
print("wrote", OUT)
