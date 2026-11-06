#!/usr/bin/env python3
"""
run_prefix_time_regions.py — Prefix-time analysis, split by benchmark and
aggregated into early/mid/late regions (2026-08-01).

Lives in its own folder, `code/generalization/prefix_time_analysis/`,
separate from the other generalization experiments' shared `results/`
(2026-08-01, design decision -- this experiment's CSVs/figure/table were
originally written into `code/generalization/results/`, mixed in with
run_website_generalization.py / run_detection_false_cut.py /
run_prefix_visibility.py's own outputs, same "keep it in its own folder"
convention already used by `sizematched_control/`).

run_prefix_visibility.py already implements eq. (8) (visible-failure
labeling: for a failed trajectory with key step k*, step k is
visible-failure iff k >= k*; successes are always negative) and reports
AUROC in 5 relative-time buckets, POOLED across all 10 (benchmark, model)
combos. This script does not re-derive that labeling logic (imports the
loaders and _finalize() verbatim from run_prefix_visibility.py, so the
per-row y_vis/bucket assignment can never drift from the already-verified
pooled numbers) -- it adds two things run_prefix_visibility.py does not
report:

  1. Bucket-level AUROC computed separately PER BENCHMARK (webarena vs
     mind2web), not just pooled -- needed to see whether a method's
     early-vs-late behavior is uniform across the two environments or
     driven by one of them.
  2. Bucket 1-5 collapsed into three coarser regions -- early (buckets
     1-2, 0-0.4T), mid (buckets 3-4, 0.4-0.8T), late (bucket 5,
     0.8-1.0T) -- as the n-weighted mean of the region's constituent
     per-bucket AUROC values. Boundaries match the paper's own
     table_prefix_time.tex caption ("Early averages the first two bins
     (0-0.4T), Mid averages 0.4-0.8T, Late uses 0.8-1.0T"), NOT a naive
     even 1-1-1-way split of 5 buckets into thirds. We deliberately do NOT
     pool a region's raw rows into one AUROC computation: doing so scores
     cross-bucket pairs (e.g. a bucket-2 positive vs. a bucket-1 negative)
     that eq. (8)'s per-bucket AUROC never scores, which can push the
     pooled number above (or below) both constituent buckets and produce
     a non-monotonic, hard-to-explain region trend that has nothing to do
     with within-bucket separability.

  3. Macro-only / Micro-only feature-set variants of Ours, read from the
     same train_risk_model_temp_sca run's cv_predictions_macro_only.csv /
     cv_predictions_uq_only.csv (present for all 10 combos, same run dir
     as cv_predictions_all.csv used by load_ours()) -- lets the writeup
     say which feature family (trajectory-level macro vs step-level
     micro/UQ) drives the prefix-time result, same ablation
     run_website_generalization.py already does for the cross-category
     experiment.

  4. A claude-matched-8 population variant (2026-08-01), run
     alongside the all-10 numbers rather than replacing them. HTC Full
     (M11) and M12 structurally exclude claude on both benchmarks (the
     Claude API exposes no logprobs -- see README "claude has no M2-M4 or
     M11/M12 ... by design"), so the original all-10 comparison put every
     other method (including Ours) on an easier/harder population than
     M11 without anyone choosing that. M1/M6/M13 turned out to still
     include claude (verified empirically -- M13 apparently imputes
     around the missing m2-m4 features rather than dropping claude rows,
     not investigated further here), so restricting to the matched-8
     population is not a no-op for them either. `_restrict_to_matched8()`
     drops claude rows post-hoc from every method's already-loaded frame
     (source string contains "claude"); Ours additionally calls
     load_ours()/load_ours_variant() with MATCHED8_COMBOS directly rather
     than loading-then-dropping, since that's the pre-existing,
     already-verified mechanism run_prefix_visibility.py itself uses for
     ours_l2_matched8.

Benchmark is recovered from each row's own "source" column (already set
by run_prefix_visibility.py's loaders as f"{bench}_{model}"): every bench
name here is a single token ("webarena"/"mind2web") with no underscore of
its own, so source.split("_", 1)[0] is unambiguous even though model
names themselves can contain underscores (e.g. "gpt_5_2").

Requires reproduce.sh to have been run first (same
prerequisite as run_prefix_visibility.py) -- reads exclusively from
reproduction_output/.

Outputs (this folder's own results/, NOT the shared
code/generalization/results/):
  prefix_time_auroc_by_bucket_per_benchmark.csv   bucket x {pooled,webarena,mind2web} x method
  prefix_time_auroc_by_region.csv                 region  x {pooled,webarena,mind2web} x method
  prefix_time_curves.png
  table_prefix_time.tex
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

THIS_DIR = Path(__file__).resolve().parent
GEN_DIR = THIS_DIR.parent  # code/generalization/ -- run_prefix_visibility.py lives here
sys.path.insert(0, str(GEN_DIR))

from run_prefix_visibility import (  # noqa: E402
    COMBOS, HTC_COLS, M1_M7_LABEL_COL, MATCHED8_COMBOS, N_BUCKETS, PR, REPRO,
    _finalize, load_htc, load_m1_m7, load_m13, load_ours,
)

MATCHED8_SOURCES = {f"{b}_{m}" for b, m in MATCHED8_COMBOS}  # every combo except claude, both benchmarks

REGIONS = [("early", (0, 1)), ("mid", (2, 3)), ("late", (4, 4))]  # 0-indexed bucket ranges, inclusive
BENCHMARKS = ["webarena", "mind2web"]

# Headline comparison for the write-up: Ours (macro+micro) vs the
# strongest single baseline family (HTC full/L2, M11 -- matches the
# "best single baseline" role M11 already plays in run_prefix_visibility.py
# / README's pooled summary).
HEADLINE_METHODS = ["ours_l2_all", "m11_htc_full_step"]
TABLE_METHODS = ["m1_verbalized_conf", "m6_cumulative_action_entropy",
                  "m11_htc_full_step", "ours_l2_uq_only", "ours_l2_macro_only", "ours_l2_all"]

METHOD_STYLE = {
    "ours_l2_all":                       dict(color="#d1372b", marker="D", ls="-",  lw=2.6, ms=7,   label="Macro&Micro (Ours)", z=5),
    "ours_l2_macro_only":                dict(color="#f0876e", marker="^", ls="-.", lw=1.8, ms=6,   label="Macro-only (Ours)", z=3),
    "ours_l2_uq_only":                   dict(color="#8f2a3a", marker="v", ls=":",  lw=1.8, ms=6,   label="Micro/UQ-only (Ours)", z=3),
    "m11_htc_full_step":                 dict(color="#e07a1e", marker="o", ls="--", lw=2.0, ms=6.5, label="HTC Full (M11)", z=4),
    "m13_stacked_htc_l2":                dict(color="#7a746b", marker="*", ls="--", lw=1.4, ms=6,   label="Stacked HTC_L2 (M13)", z=2),
    "m1_verbalized_conf":                dict(color="#8fb996", marker="o", ls=":",  lw=1.2, ms=5,   label="Verbalized Conf. (M1)", z=2),
    "m6_cumulative_action_entropy":      dict(color="#7a9d4f", marker="D", ls=":",  lw=1.2, ms=5,   label="Cumul. Action Entropy (M6)", z=2),
}


def load_ours_variant(variant: str, combos=COMBOS) -> pd.DataFrame | None:
    """Same as run_prefix_visibility.load_ours(), but reads
    cv_predictions_{variant}.csv instead of cv_predictions_all.csv --
    variant in {"macro_only", "uq_only"}, produced by the same
    train_risk_model_temp_sca run (same run dir, sibling file)."""
    frames = []
    for bench, model in combos:
        pattern = REPRO / "ours" / f"{bench}_{model}" / "thr_0p3" / "train_risk_model_temp_sca"
        runs = sorted(pattern.glob(f"*/cv_predictions_{variant}.csv"))
        if not runs:
            continue
        df = pd.read_csv(runs[-1])
        df = df.rename(columns={"step_k": "step", "Logistic_L2_prob": "score"})
        df["task_id"] = df["task_id"].astype(str)
        out = _finalize(df, label_col="label", group_col="task_id", score_col="score")
        out["source"] = f"{bench}_{model}"
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else None


def _restrict_to_matched8(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Drops claude rows post-hoc (source in {'webarena_claude','mind2web_claude'}).
    Used for methods (M1/M6/M7/M13/...) whose own loader has no combos= param
    and that -- unlike M11/M12 -- turned out to still include claude."""
    if df is None or df.empty:
        return df
    out = df[df["source"].isin(MATCHED8_SOURCES)]
    return out if not out.empty else None


def load_all_methods(matched8: bool = False) -> dict[str, pd.DataFrame]:
    combos = MATCHED8_COMBOS if matched8 else COMBOS
    tag = "matched-8 (claude excluded)" if matched8 else "all 10 combos"
    methods: dict[str, pd.DataFrame] = {}
    print(f"Loading Ours ({tag})...")
    methods["ours_l2_all"] = load_ours(combos)
    print("Loading Ours macro-only...")
    methods["ours_l2_macro_only"] = load_ours_variant("macro_only", combos)
    print("Loading Ours micro/UQ-only...")
    methods["ours_l2_uq_only"] = load_ours_variant("uq_only", combos)
    for m in M1_M7_LABEL_COL:
        print(f"Loading {m}...")
        df = load_m1_m7(m)
        methods[m] = _restrict_to_matched8(df) if matched8 else df
    for m in HTC_COLS:
        print(f"Loading {m}...")
        df = load_htc(m)
        methods[m] = _restrict_to_matched8(df) if matched8 else df  # already claude-free; filter is a no-op
    print("Loading m13_stacked_htc_l2...")
    df = load_m13()
    methods["m13_stacked_htc_l2"] = _restrict_to_matched8(df) if matched8 else df
    return {k: v for k, v in methods.items() if v is not None and not v.empty}


def _with_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["benchmark"] = df["source"].str.split("_", n=1).str[0]
    assert df["benchmark"].isin(BENCHMARKS).all(), \
        f"unexpected benchmark tokens: {sorted(set(df['benchmark']) - set(BENCHMARKS))}"
    return df


def _auroc(sub: pd.DataFrame) -> tuple[float, int]:
    if sub["y_vis"].nunique() < 2:
        return float("nan"), len(sub)
    return float(roc_auc_score(sub["y_vis"].values, sub["score"].values)), len(sub)


def bucket_table(methods: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in methods.items():
        df = _with_benchmark(df)
        for scope in ["pooled"] + BENCHMARKS:
            scoped = df if scope == "pooled" else df[df["benchmark"] == scope]
            for b in range(N_BUCKETS):
                auc, n = _auroc(scoped[scoped["bucket"] == b])
                rows.append({"method": name, "scope": scope, "bucket": b + 1,
                             "auroc": round(auc, 4) if not np.isnan(auc) else np.nan, "n": n})
    return pd.DataFrame(rows)


def region_table(bucket_df: pd.DataFrame) -> pd.DataFrame:
    """n-weighted mean of each region's constituent per-bucket AUROC values
    (see module docstring for why we don't pool raw rows across buckets)."""
    rows = []
    for (name, scope), g in bucket_df.groupby(["method", "scope"]):
        g = g.set_index("bucket")
        for region, (lo, hi) in REGIONS:
            sub = g.loc[lo + 1:hi + 1]  # bucket column is 1-indexed; lo/hi are 0-indexed
            sub = sub.dropna(subset=["auroc"])
            if sub.empty:
                auc, n = np.nan, 0
            else:
                n = int(sub["n"].sum())
                auc = float(np.average(sub["auroc"], weights=sub["n"]))
            rows.append({"method": name, "scope": scope, "region": region,
                         "auroc": round(auc, 4) if not np.isnan(auc) else np.nan, "n": n})
    return pd.DataFrame(rows)


def first_bucket_above(bucket_df: pd.DataFrame, method: str, scope: str, threshold: float) -> str:
    sub = bucket_df[(bucket_df.method == method) & (bucket_df.scope == scope)].sort_values("bucket")
    hit = sub[sub["auroc"] > threshold]
    if hit.empty:
        return "never"
    b = int(hit.iloc[0]["bucket"])
    return f"bucket {b} ({(b - 1) / N_BUCKETS:.1f}-{b / N_BUCKETS:.1f}T)"


def make_figure(bucket_df: pd.DataFrame, out_path: Path, title: str = "Prefix-Time Visibility of Failure, by Benchmark") -> None:
    fig, axes = plt.subplots(1, len(BENCHMARKS), figsize=(13, 5.6), sharey=True)
    bench_titles = {"webarena": "WebArena-Lite", "mind2web": "Mind2Web"}

    for ax, bench in zip(axes, BENCHMARKS):
        for method, style in METHOD_STYLE.items():
            sub = bucket_df[(bucket_df.method == method) & (bucket_df.scope == bench)].sort_values("bucket")
            if sub.empty or sub["auroc"].isna().all():
                continue
            ax.plot(sub["bucket"], sub["auroc"], color=style["color"], marker=style["marker"],
                     linestyle=style["ls"], linewidth=style["lw"], markersize=style["ms"],
                     alpha=0.95 if method in HEADLINE_METHODS else 0.75,
                     label=style["label"], zorder=style["z"])
        ax.axhline(0.5, color="#999999", linestyle="-", linewidth=0.8, alpha=0.6, zorder=1)
        ax.set_title(bench_titles[bench], fontsize=14, fontweight="bold")
        ax.set_xticks(range(1, N_BUCKETS + 1))
        ax.set_xticklabels([f"{(b - 1) / N_BUCKETS:.1f}–{b / N_BUCKETS:.1f}T" for b in range(1, N_BUCKETS + 1)],
                            fontsize=9.5)
        ax.set_xlabel("Relative trajectory position", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    axes[0].set_ylabel("AUROC (visible-failure target)", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.09), frameon=False)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


METHOD_LABEL_TEX = {
    "m1_verbalized_conf": "Verb. Conf.", "m6_cumulative_action_entropy": "Cumul. Act. Ent.",
    "m11_htc_full_step": "HTC Full", "ours_l2_uq_only": "Micro-only",
    "ours_l2_macro_only": "Macro-only", "ours_l2_all": "Macro\\&Micro",
}


def make_latex_table(bucket_df: pd.DataFrame, region_df: pd.DataFrame, out_path: Path,
                      label: str = "tab:prefix_time", note: str = "") -> None:
    scope_labels = [("webarena", "WebArena-Lite"), ("mind2web", "Mind2Web"), ("pooled", "All Runs Avg.")]
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{llcccc}", r"\toprule",
        r"\textbf{Benchmark} & \textbf{Method} & \textbf{Early} & \textbf{Mid} & \textbf{Late} & \textbf{Earliest $>$0.65} \\",
        r"\midrule",
    ]
    for si, (scope, scope_title) in enumerate(scope_labels):
        scoped = region_df[(region_df.scope == scope) & (region_df.method.isin(TABLE_METHODS))]
        col_best = {reg: scoped[scoped.region == reg]["auroc"].max() for reg in ["early", "mid", "late"]}
        for mi, method in enumerate(TABLE_METHODS):
            r = region_df[(region_df.scope == scope) & (region_df.method == method)].set_index("region")["auroc"]
            vals = {reg: r.get(reg, float("nan")) for reg in ["early", "mid", "late"]}
            cells = []
            for reg in ["early", "mid", "late"]:
                v = vals[reg]
                txt = "--" if pd.isna(v) else f"{v:.3f}"
                # bold the true per-column (scope, region) best, not a fixed "Ours" row --
                # HTC Full beats Ours on WebArena-Lite Early in this rerun, so a hardcoded
                # "always bold Macro&Micro" rule would misreport that cell.
                is_col_best = (not pd.isna(v)) and np.isclose(v, col_best[reg])
                cells.append(f"\\textbf{{{txt}}}" if is_col_best else txt)
            hit = first_bucket_above(bucket_df, method, scope, 0.65)
            hit_txt = "--" if hit == "never" else hit.split("(")[1].rstrip(")")
            prefix = f"\\multirow{{{len(TABLE_METHODS)}}}{{*}}{{{scope_title}}} " if mi == 0 else " "
            lines.append(f"{prefix}& {METHOD_LABEL_TEX[method]} & {cells[0]} & {cells[1]} & {cells[2]} & {hit_txt} \\\\")
        if si < len(scope_labels) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Key-step-aware prefix visibility analysis (2026-08-01 rerun). AUROC is computed for "
        r"detecting whether a prefix is in the visible failure state: successful prefixes and pre-key-step "
        r"failed prefixes are negative, while post-key-step failed prefixes are positive. Early averages the "
        r"first two bins ($0$--$0.4T$), Mid averages $0.4$--$0.8T$, and Late uses $0.8$--$1.0T$." + note + "}",
        rf"\label{{{label}}}", r"\end{table*}",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def run_variant(matched8: bool, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    suffix = "_matched8" if matched8 else ""
    label = "tab:prefix_time_matched8" if matched8 else "tab:prefix_time"
    note = (r" Claude is excluded from every method here (not just HTC/M11, which structurally "
             r"has no logprobs for it) so every row is scored on the identical 8-combo population.") \
        if matched8 else ""
    fig_title = ("Prefix-Time Visibility of Failure, by Benchmark (Claude excluded, matched-8 population)"
                 if matched8 else "Prefix-Time Visibility of Failure, by Benchmark")

    methods = load_all_methods(matched8=matched8)
    for name, df in methods.items():
        print(f"  {name:22s}  {len(df):6d} rows")

    bucket_df = bucket_table(methods)
    bucket_df.to_csv(out_dir / f"prefix_time_auroc_by_bucket_per_benchmark{suffix}.csv", index=False)

    region_df = region_table(bucket_df)
    region_df.to_csv(out_dir / f"prefix_time_auroc_by_region{suffix}.csv", index=False)

    print(f"\n=== [{'matched-8, no claude' if matched8 else 'all 10 combos'}] Region-level AUROC "
          "(early=buckets1-2 [0-0.4T], mid=buckets3-4 [0.4-0.8T], late=bucket5 [0.8-1.0T]) ===")
    print(region_df[region_df.method.isin(TABLE_METHODS)].pivot_table(
        index=["method", "scope"], columns="region", values="auroc").reindex(
        columns=["early", "mid", "late"]).to_string())

    print("\n=== First bucket with AUROC > 0.65 (per method x scope) ===")
    for name in TABLE_METHODS:
        for scope in ["pooled"] + BENCHMARKS:
            print(f"  {name:22s} {scope:10s} {first_bucket_above(bucket_df, name, scope, 0.65)}")

    make_figure(bucket_df, out_dir / f"prefix_time_curves{suffix}.png", title=fig_title)
    make_latex_table(bucket_df, region_df, out_dir / f"table_prefix_time{suffix}.tex", label=label, note=note)
    return bucket_df, region_df


def main():
    out_dir = THIS_DIR / "results"
    out_dir.mkdir(exist_ok=True)

    print("########## ALL 10 COMBOS ##########")
    _, region_all = run_variant(matched8=False, out_dir=out_dir)

    print("\n\n########## MATCHED-8 (CLAUDE EXCLUDED EVERYWHERE) ##########")
    _, region_m8 = run_variant(matched8=True, out_dir=out_dir)

    print("\n\n=== Headline comparison: does dropping Claude from every method change the "
          "Ours-vs-HTC Early-region story? (pooled + per-benchmark) ===")
    for scope in ["pooled"] + BENCHMARKS:
        row = {"scope": scope}
        for tag, region_df in [("all10", region_all), ("matched8", region_m8)]:
            for method in ["ours_l2_all", "m11_htc_full_step"]:
                v = region_df[(region_df.scope == scope) & (region_df.method == method) &
                               (region_df.region == "early")]["auroc"]
                row[f"{method}_{tag}"] = round(float(v.iloc[0]), 4) if not v.empty else float("nan")
        print(f"  {scope:10s} " + "  ".join(f"{k}={v}" for k, v in row.items() if k != "scope"))

    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
