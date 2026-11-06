#!/usr/bin/env python3
"""
run_detection_false_cut_regions.py — Detection-rate-at-fixed-FCR-budget,
split by benchmark (2026-08-01; 2026-08-08 correctness fix).

run_detection_false_cut.py implements the key-step-aware early-cut policy
and the Detection/False-Cut-Rate tradeoff curve (a trajectory is cut the
first step its P(failure) exceeds tau; a cut on a failed trajectory
counts as "detected" only if it happens at/after that trajectory's own
keyStep; a cut on a successful trajectory is always a "false cut"). This
script does not re-derive that logic -- it now IMPORTS every loader
(load_ours/load_m1_m7/load_htc/load_m13/intersect_tasks) UNCHANGED rather
than keeping its own duplicate copies, which is itself the fix for a real
bug found 2026-08-08: this file used to carry its own second copy of
these loaders (collect_tasks_bench + a from-label-column _finalize), so
the base file's earlier corrections didn't propagate here automatically.
Duplicated logic is exactly how a fix in one place silently fails to
reach the other -- eliminated by importing instead of copying.

2026-08-08 CORRECTNESS FIX (previously wrong, caught in review):
final_failed/key_step used to be inferred from the last value of each
method's own already-relabeled per-step "label" column -- wrong whenever
a failed task's annotated key step falls at/beyond its last recorded
step (see run_detection_false_cut.py's module docstring for the full
explanation and the exact empirical count this affected: mind2web's
failed-task count was silently short by 175/1599). Now: final_failed
comes from the raw ground-truth outcome file, key_step from the external
keystep annotation file -- both loaded once per combo, correctly
role-scoped (ours vs baseline), never re-derived from a predictor's own
output. Also fixed: every method is evaluated on the matched-8-combo
population (claude excluded uniformly, not just for Ours as an extra
"matched8" line) AND further intersected down to the common task set
every compared method actually has a score for (see
run_detection_false_cut.intersect_tasks()) -- so no method benefits from
a systematically easier/different task mix.

This script still adds the three things beyond the base file's own
pooled-only output:
  1. Per-benchmark curves (webarena vs mind2web), not just pooled --
     needed because the "which method wins" story flips by benchmark.
  2. Detection Rate read off at FIXED False-Cut-Rate budgets (5%/10%/20%)
     rather than reporting a single curve-AUC number -- this is what an
     operator actually cares about ("if I'm willing to falsely cut at most
     10% of successful runs, what fraction of failures do I catch"), and
     is what the paper text quotes.
  3. Macro-only / Micro-only Ours variants (cv_predictions_macro_only.csv /
     cv_predictions_uq_only.csv, same run dir as cv_predictions_all.csv),
     same ablation role as in prefix_time_analysis/.

Detection@FCR-budget definition: among all tau on the swept curve whose
false_cut_rate <= budget, report the MAX detection_rate achieved (i.e. the
best operating point that still respects the budget). Matches standard
"detection rate under a false-cut budget" reporting convention.

Requires reproduce.sh to have been run first -- reads
exclusively from reproduction_output/ + data/
+ keystep_labels/.

Outputs (this folder's own results/):
  detection_at_fcr_by_benchmark.csv   scope x budget x method -> detection_rate
  detection_false_cut_curves.png
  table_early_cut.tex
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
GEN_DIR = THIS_DIR.parent
sys.path.insert(0, str(GEN_DIR))

from run_detection_false_cut import (  # noqa: E402
    MATCHED8_COMBOS, M1_M7_LABEL_COL, HTC_COLS,
    load_ours, load_m1_m7, load_htc, load_m13, intersect_tasks,
    tradeoff_curve,
)

BENCHMARKS = ["webarena", "mind2web"]
FCR_BUDGETS = [0.05, 0.10, 0.15, 0.20]
LEGEND_FONTSIZE = 44
LEGEND_LINEWIDTH = 8.0

TABLE_METHODS = ["m1_verbalized_conf", "m6_cumulative_action_entropy",
                  "m11_htc_full_step", "ours_l2_uq_only", "ours_l2_macro_only", "ours_l2_all"]

METHOD_STYLE = {
    "ours_l2_all":                       dict(color="#d1372b", marker="D", ls="-",  lw=4.2, ms=7,   label="Macro&Micro (Ours)", z=5),
    "ours_l2_macro_only":                dict(color="#f0876e", marker="^", ls="-",  lw=3.2, ms=6,   label="Macro Only", z=3),
    "ours_l2_uq_only":                   dict(color="#8f2a3a", marker="v", ls="-",  lw=3.2, ms=6,   label="Micro Only", z=3),
    "m11_htc_full_step":                 dict(color="#e07a1e", marker="o", ls="--", lw=3.4, ms=6.5, label="HTC Full", z=4),
    "m13_stacked_htc_l2":                dict(color="#7a746b", marker="*", ls="--", lw=2.6, ms=6,   label="Stacked_L2", z=2),
    "m1_verbalized_conf":                dict(color="#8fb996", marker="o", ls=":",  lw=2.4, ms=5,   label="Verbalized Conf.", z=2),
    "m6_cumulative_action_entropy":      dict(color="#7a9d4f", marker="D", ls=":",  lw=2.4, ms=5,   label="Action Entropy", z=2),
}


def _with_benchmark(tasks: list[dict]) -> list[dict]:
    """global_task_id is "{bench}_{model}::{tid}" -- bench is a single
    token with no underscore of its own (verified project-wide), so
    split("_",1)[0] recovers it unambiguously."""
    for t in tasks:
        t["benchmark"] = t["global_task_id"].split("_", 1)[0]
    return tasks


def load_all_methods() -> dict[str, list[dict]]:
    methods: dict[str, list[dict]] = {}
    print("Loading Ours (matched-8 combos, claude excluded)...")
    methods["ours_l2_all"] = load_ours(MATCHED8_COMBOS, variant="all")
    print("Loading Ours macro-only...")
    methods["ours_l2_macro_only"] = load_ours(MATCHED8_COMBOS, variant="macro_only")
    print("Loading Ours micro/UQ-only...")
    methods["ours_l2_uq_only"] = load_ours(MATCHED8_COMBOS, variant="uq_only")
    for m in M1_M7_LABEL_COL:
        print(f"Loading {m}...")
        methods[m] = load_m1_m7(m, MATCHED8_COMBOS)
    for m in HTC_COLS:
        print(f"Loading {m}...")
        methods[m] = load_htc(m, MATCHED8_COMBOS)
    print("Loading m13_stacked_htc_l2...")
    methods["m13_stacked_htc_l2"] = load_m13(MATCHED8_COMBOS)

    methods = {k: v for k, v in methods.items() if v}
    print("\nIntersecting to the common task set across all methods...")
    methods = intersect_tasks(methods)
    for name in methods:
        methods[name] = _with_benchmark(methods[name])
    return methods


def detection_at_fcr(curve: pd.DataFrame, budgets=FCR_BUDGETS) -> dict:
    """Best (max) detection_rate among all tau whose false_cut_rate <= budget."""
    d = curve.sort_values("false_cut_rate")
    out = {}
    for b in budgets:
        sub = d[d["false_cut_rate"] <= b]
        out[b] = float(sub["detection_rate"].max()) if not sub.empty else float("nan")
    return out


def curve_for_scope(tasks: list[dict], scope: str) -> pd.DataFrame | None:
    scoped = tasks if scope == "pooled" else [t for t in tasks if t["benchmark"] == scope]
    failed = sum(t["final_failed"] for t in scoped)
    success = len(scoped) - failed
    if failed == 0 or success == 0:
        return None
    return tradeoff_curve(scoped)


def build_tables(methods: dict[str, list[dict]]):
    fcr_rows = []
    curves = {}  # (method, scope) -> curve df, kept for plotting
    for name, tasks in methods.items():
        for scope in ["pooled"] + BENCHMARKS:
            curve = curve_for_scope(tasks, scope)
            if curve is None:
                continue
            curves[(name, scope)] = curve
            det = detection_at_fcr(curve)
            n_scoped = tasks if scope == "pooled" else [t for t in tasks if t["benchmark"] == scope]
            n_failed = sum(t["final_failed"] for t in n_scoped)
            n_success = len(n_scoped) - n_failed
            for budget, rate in det.items():
                fcr_rows.append({"method": name, "scope": scope, "fcr_budget": budget,
                                  "detection_rate": round(rate, 4) if not np.isnan(rate) else np.nan,
                                  "n_failed": n_failed, "n_success": n_success})
    return pd.DataFrame(fcr_rows), curves


def make_figure(curves: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(BENCHMARKS), figsize=(24, 12), sharey=True)
    bench_titles = {"webarena": "WebArena-Lite", "mind2web": "Mind2Web"}

    for ax, bench in zip(axes, BENCHMARKS):
        for method, style in METHOD_STYLE.items():
            curve = curves.get((method, bench))
            if curve is None:
                continue
            d = curve.sort_values("false_cut_rate")
            d = d[d["false_cut_rate"] <= 0.20]
            ax.plot(d["false_cut_rate"], d["detection_rate"], color=style["color"],
                     linestyle=style["ls"], linewidth=style["lw"], alpha=0.95,
                     label=style["label"], zorder=style["z"])
        for b in FCR_BUDGETS:
            ax.axvline(b, color="#999999", linestyle=":", linewidth=0.7, alpha=0.5, zorder=1)
        ax.set_title(bench_titles[bench], fontsize=40, fontweight="bold")
        ax.set_xlabel("False Cut Rate", fontsize=30)
        ax.set_xlim(0, 0.20)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="both", labelsize=38)

    axes[0].set_ylabel("Detection Rate", fontsize=38)
    handles, labels = axes[0].get_legend_handles_labels()
    # Match the success/false-alarm figure's fixed plot area.  Constrain the
    # larger legend to the canvas width so it grows downward, not sideways.
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.14, wspace=0.16)
    legend = fig.legend(handles, labels, loc="upper center", ncol=2,
                        fontsize=LEGEND_FONTSIZE,
                        bbox_to_anchor=(0.08, 0.02, 0.90, 0.0), mode="expand",
                        frameon=False, borderaxespad=0.0)
    for legend_line in legend.get_lines():
        legend_line.set_linewidth(LEGEND_LINEWIDTH)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


METHOD_LABEL_TEX = {
    "m1_verbalized_conf": "Verb. Conf.", "m6_cumulative_action_entropy": "Action Ent.",
    "m11_htc_full_step": "HTC Full", "ours_l2_uq_only": "Micro-only",
    "ours_l2_macro_only": "Macro-only", "ours_l2_all": "Macro\\&Micro",
}


def make_latex_table(fcr_df: pd.DataFrame, out_path: Path) -> None:
    scope_labels = [("webarena", "WebArena-Lite"), ("mind2web", "Mind2Web"), ("pooled", "Pooled")]
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{ll" + "c" * len(FCR_BUDGETS) + "}", r"\toprule",
        r"\textbf{Benchmark} & \textbf{Method} & "
        + " & ".join(rf"\textbf{{Det.@{int(round(b*100))}\%FCR}}" for b in FCR_BUDGETS)
        + r" \\",
        r"\midrule",
    ]
    for si, (scope, scope_title) in enumerate(scope_labels):
        scoped = fcr_df[(fcr_df.scope == scope) & (fcr_df.method.isin(TABLE_METHODS))]
        col_best = {b: scoped[scoped.fcr_budget == b]["detection_rate"].max() for b in FCR_BUDGETS}
        for mi, method in enumerate(TABLE_METHODS):
            r = fcr_df[(fcr_df.scope == scope) & (fcr_df.method == method)].set_index("fcr_budget")["detection_rate"]
            cells = []
            for b in FCR_BUDGETS:
                v = r.get(b, float("nan"))
                txt = "--" if pd.isna(v) else f"{v*100:.1f}\\%"
                is_best = (not pd.isna(v)) and np.isclose(v, col_best[b])
                cells.append(f"\\textbf{{{txt}}}" if is_best else txt)
            prefix = f"\\multirow{{{len(TABLE_METHODS)}}}{{*}}{{{scope_title}}} " if mi == 0 else " "
            lines.append(f"{prefix}& {METHOD_LABEL_TEX[method]} & " + " & ".join(cells) + r" \\")
        if si < len(scope_labels) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        # Kept in sync with the paper's sections/table_early_cut.tex (2026-09-01):
        # reader-facing wording, and the risk threshold is \gamma, not \tau
        # (\tau is the trajectory prefix; f_\theta is the predictor).
        r"\caption{Detection rate under a key-step-aware early-cut policy at fixed False-Cut-Rate "
        r"(FCR) budgets. A failed trajectory is detected only when it is cut at or after its "
        r"externally annotated key step; a cut on a successful trajectory is always false. "
        r"All methods use the same 1,576-trajectory intersection over eight non-Claude settings. "
        r"Det.@$b$\%FCR is the best detection rate among thresholds whose FCR does not exceed "
        r"$b\%$.}",
        r"\label{tab:early_cut}", r"\end{table*}",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main():
    out_dir = THIS_DIR / "results"
    out_dir.mkdir(exist_ok=True)

    methods = load_all_methods()
    for name, tasks in methods.items():
        print(f"  {name:22s}  {len(tasks):6d} tasks")

    fcr_df, curves = build_tables(methods)
    fcr_df.to_csv(out_dir / "detection_at_fcr_by_benchmark.csv", index=False)

    print("\n=== Detection Rate @ fixed FCR budgets ===")
    print(fcr_df[fcr_df.method.isin(TABLE_METHODS)].pivot_table(
        index=["method", "scope"], columns="fcr_budget", values="detection_rate").to_string())

    make_figure(curves, out_dir / "detection_false_cut_curves.png")
    make_latex_table(fcr_df, out_dir / "table_early_cut.tex")
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
