#!/usr/bin/env python3
"""
run_family_significance_analysis.py — Is the Micro (UQ) increment over Macro
statistically significant, consistent, and complementary? (A point-estimate AUROC-only comparison isn't enough, so this
uses paired significance tests across AUROC/e-AURC/Brier, along three axes:
significant + consistent + complementary.)

Reuses NOTHING recomputed -- reads directly from each combo's own
cv_results.json (already produced by reproduce.sh's Ours run),
which stores PER-FOLD test metrics for all three feature sets
(macro_only/uq_only/all). This is the same underlying CV run that produced
every previously-reported Ours(L2) number in the main table, so these
significance numbers can never silently drift from what's already reported.

Two levels of comparison, both requested:

  (A) Fold-level paired test (10 combos x 5 folds = 50 paired points):
      for each metric, pairs = (all_fold_metric, macro_only_fold_metric)
      at the SAME (combo, fold) -- Wilcoxon signed-rank (nonparametric,
      standard for CV-fold paired comparisons) + paired t-test for reference.

  (B) Combo-level ("cell") consistency: aggregate each combo's 5 folds to
      one mean per feature set, count how many of the 10 combos show
      all > macro_only (a "win"), binomial sign test against p=0.5.

  (C) Complementarity: uq_only vs macro_only per combo (does micro alone
      ever beat macro alone?) and all vs macro_only per combo across all
      three metrics (does stacking help even where micro alone is weaker?).

Requires reproduce.sh to have been run first.

Outputs (this folder's own results/):
  family_significance_foldlevel.csv   -- the 50 paired points per metric
  family_significance_summary.csv     -- test statistics for (A)/(B)/(C)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from itertools import product

from scipy import stats

PR = Path(__file__).resolve().parent.parent.parent  # 
REPRO = PR / "reproduction_output"

COMBOS = [
    ("webarena", "claude"), ("webarena", "gpt_5_2"), ("webarena", "gptnano"),
    ("webarena", "kimi"), ("webarena", "qwen30b"),
    ("mind2web", "claude"), ("mind2web", "gpt_5_2"), ("mind2web", "gptnano"),
    ("mind2web", "kimi"), ("mind2web", "qwen30b"),
]

METRICS = ["auroc", "e_aurc", "brier"]
LOWER_BETTER = {"auroc": False, "e_aurc": True, "brier": True}


def exact_sign_flip_permutation_test(diffs: np.ndarray) -> float:
    """Exact two-sided paired permutation test: under H0 (no systematic
    effect), each of the N paired differences is equally likely to have
    come out positive or negative. Enumerate all 2^N sign-flip
    combinations exactly (N=10 -> 1024, trivial), compare |mean(observed)|
    to the null distribution of |mean(sign-flipped)|. No normality
    assumption, no reliance on Wilcoxon's asymptotic approximation, and
    correctly sized for the true N=10 independent units (one per combo) --
    the right test given the fold-level test's pseudo-replication problem."""
    n = len(diffs)
    obs = abs(diffs.mean())
    null_means = []
    for signs in product([1, -1], repeat=n):
        null_means.append(np.mean(np.array(signs) * diffs))
    null_means = np.array(null_means)
    return float((np.abs(null_means) >= obs - 1e-12).mean())


def combo_level_magnitude_test(diff: np.ndarray, label: str) -> dict:
    """diff: per-combo (already averaged across each combo's own 5 folds
    -> N=10 truly independent points, one per backbone x benchmark combo)
    "improvement" values, positive = the treatment condition improved over
    the baseline. Runs a one-sample paired t-test / Wilcoxon signed-rank /
    exact sign-flip permutation test against 0, plus a t-based 95% CI on
    the mean."""
    n = len(diff)
    t_stat, t_p = stats.ttest_1samp(diff, 0.0)
    try:
        w_stat, w_p = stats.wilcoxon(diff)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    perm_p = exact_sign_flip_permutation_test(diff)
    se = diff.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    ci_lo, ci_hi = diff.mean() - tcrit * se, diff.mean() + tcrit * se
    print(f"  {label:10s}  N={n}  mean diff={diff.mean():+.4f}  95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"             paired t: t={t_stat:+.3f} p={t_p:.4g}   "
          f"wilcoxon: p={w_p:.4g}   exact permutation: p={perm_p:.4g}")
    return {
        "n": n, "mean_diff": round(float(diff.mean()), 4),
        "ci95_lo": round(float(ci_lo), 4), "ci95_hi": round(float(ci_hi), 4),
        "ttest_stat": round(float(t_stat), 4), "ttest_p": float(t_p),
        "wilcoxon_p": float(w_p) if not np.isnan(w_p) else None,
        "exact_permutation_p": float(perm_p),
    }


def load_fold_metrics(bench: str, model: str) -> dict[str, list[dict]] | None:
    pattern = REPRO / "ours" / f"{bench}_{model}" / "thr_0p3" / "train_risk_model_temp_sca"
    runs = sorted(pattern.glob("*/cv_results.json"))
    if not runs:
        return None
    d = json.loads(runs[-1].read_text())
    return d["test"]


def main():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    fold_rows = []
    for bench, model in COMBOS:
        test = load_fold_metrics(bench, model)
        if test is None:
            print(f"  [SKIP] {bench}/{model}: no cv_results.json")
            continue
        combo = f"{bench}_{model}"
        for fs in ["all", "macro_only", "uq_only"]:
            folds = test.get(fs, {}).get("Logistic_L2", [])
            for f in folds:
                row = {"combo": combo, "feature_set": fs, "fold": f.get("fold")}
                for m in METRICS:
                    row[m] = f.get(m)
                fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "family_significance_foldlevel.csv", index=False)
    print(f"Loaded {fold_df['combo'].nunique()} combos, "
          f"{fold_df[fold_df.feature_set=='all'].shape[0]} (combo,fold) rows for 'all'")

    # ---- (A) fold-level paired test: all vs macro_only, 10 combos x 5 folds ----
    summary_rows = []
    print("\n" + "=" * 70)
    print("(A) Fold-level paired test: all vs. macro_only (N=50 paired points)")
    print("=" * 70)
    piv_all = fold_df[fold_df.feature_set == "all"].set_index(["combo", "fold"])
    piv_macro = fold_df[fold_df.feature_set == "macro_only"].set_index(["combo", "fold"])
    common_idx = piv_all.index.intersection(piv_macro.index)
    for m in METRICS:
        a = piv_all.loc[common_idx, m].values.astype(float)
        b = piv_macro.loc[common_idx, m].values.astype(float)
        diff = a - b  # positive = "all" has higher raw value
        if LOWER_BETTER[m]:
            improvement = -diff  # positive = "all" improved (lower is better)
        else:
            improvement = diff
        n = len(diff)
        t_stat, t_p = stats.ttest_rel(a, b)
        try:
            w_stat, w_p = stats.wilcoxon(diff)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        print(f"  {m:8s}  N={n}  mean improvement={improvement.mean():+.4f}  "
              f"paired t: t={t_stat:+.3f} p={t_p:.4g}   wilcoxon: W={w_stat} p={w_p:.4g}")
        summary_rows.append({
            "analysis": "fold_level_paired_all_vs_macro", "metric": m, "n": n,
            "mean_improvement": round(float(improvement.mean()), 4),
            "std_improvement": round(float(improvement.std()), 4),
            "ttest_stat": round(float(t_stat), 4), "ttest_p": float(t_p),
            "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else None,
            "wilcoxon_p": float(w_p) if not np.isnan(w_p) else None,
        })

    # ---- (B) combo-level ("cell") consistency + sign test ----
    print("\n" + "=" * 70)
    print("(B) Combo-level consistency: all vs. macro_only (N=10 cells)")
    print("=" * 70)
    combo_means = fold_df.groupby(["combo", "feature_set"])[METRICS].mean().reset_index()
    cm_all = combo_means[combo_means.feature_set == "all"].set_index("combo")
    cm_macro = combo_means[combo_means.feature_set == "macro_only"].set_index("combo")
    cm_uq = combo_means[combo_means.feature_set == "uq_only"].set_index("combo")
    common_combos = cm_all.index.intersection(cm_macro.index)
    for m in METRICS:
        a = cm_all.loc[common_combos, m]
        b = cm_macro.loc[common_combos, m]
        wins = (a < b) if LOWER_BETTER[m] else (a > b)
        n_wins = int(wins.sum())
        n = len(wins)
        sign_p = stats.binomtest(n_wins, n, 0.5, alternative="greater").pvalue
        max_gain = (b - a).abs().max() if LOWER_BETTER[m] else (a - b).max()
        print(f"  {m:8s}  {n_wins}/{n} cells improved by 'all'   sign test p={sign_p:.4g}   "
              f"max per-cell gain={max_gain:.4f}")
        summary_rows.append({
            "analysis": "combo_level_wins_all_vs_macro", "metric": m, "n": n,
            "n_wins": n_wins, "sign_test_p_greater": float(sign_p),
            "max_gain": round(float(max_gain), 4),
        })

    # ---- (B2) combo-level MAGNITUDE test: paired t-test / Wilcoxon / exact ----
    # permutation test on the 10 combo-mean differences -- the statistically
    # correct unit (no pseudo-replication) that also uses "how much", not
    # just win/loss direction (unlike the sign test above).
    print("\n" + "=" * 70)
    print("(B2) Combo-level MAGNITUDE test: all vs. macro_only (N=10, correct unit)")
    print("=" * 70)
    for m in METRICS:
        a = cm_all.loc[common_combos, m].values.astype(float)
        b = cm_macro.loc[common_combos, m].values.astype(float)
        diff = (b - a) if LOWER_BETTER[m] else (a - b)  # positive = "all" improved
        res = combo_level_magnitude_test(diff, m)
        res["metric"] = m
        res["analysis"] = "combo_level_magnitude_all_vs_macro"
        summary_rows.append(res)

    # ---- (C) complementarity ----
    print("\n" + "=" * 70)
    print("(C) Complementarity: uq_only vs. macro_only (AUROC), and does "
          "stacking ('all') still help where micro alone trails macro alone?")
    print("=" * 70)
    common_uq = cm_uq.index.intersection(cm_macro.index)
    uq_wins = (cm_uq.loc[common_uq, "auroc"] > cm_macro.loc[common_uq, "auroc"])
    print(f"  uq_only beats macro_only on AUROC in {int(uq_wins.sum())}/{len(uq_wins)} cells "
          f"(the rest: macro_only alone is stronger)")
    trailing_combos = uq_wins[~uq_wins].index.tolist()
    print(f"  combos where uq_only trails macro_only: {trailing_combos}")
    for m in METRICS:
        a = cm_all.loc[trailing_combos, m].values.astype(float)
        b = cm_macro.loc[trailing_combos, m].values.astype(float)
        wins_in_trailing = (a < b) if LOWER_BETTER[m] else (a > b)
        diff = (b - a) if LOWER_BETTER[m] else (a - b)
        print(f"    within those, stacking ('all') still beats macro_only on {m}: "
              f"{int(wins_in_trailing.sum())}/{len(trailing_combos)}")
        res = combo_level_magnitude_test(diff, f"{m} (n={len(trailing_combos)})")
        res["metric"] = m
        res["analysis"] = "complementarity_stacking_helps_where_uq_alone_trails"
        res["n_wins"] = int(wins_in_trailing.sum())
        summary_rows.append(res)
    summary_rows.append({
        "analysis": "complementarity_uq_only_vs_macro_only", "metric": "auroc",
        "n": len(uq_wins), "n_wins": int(uq_wins.sum()),
    })

    # ---- (D) benchmark-split: does the pooled null hide opposite-signed  ----
    # per-benchmark effects? (2026-08-01: mind2web looked
    # consistently positive on inspection). N=5 per benchmark (5 backbones),
    # so the exact/Wilcoxon tests are floored at p=2^(1-5)=0.0625 no matter
    # how consistent the direction is -- report the paired t-test too, which
    # isn't subject to that combinatorial floor, but treat N=5 results as
    # suggestive, not conclusive, regardless of which test is used.
    print("\n" + "=" * 70)
    print("(D) Benchmark-split: all vs. macro_only, mind2web-only vs. webarena-only (N=5 each)")
    print("=" * 70)
    for bench in ["mind2web", "webarena"]:
        bench_combos = [c for c in common_combos if c.startswith(bench)]
        for m in METRICS:
            a = cm_all.loc[bench_combos, m].values.astype(float)
            b = cm_macro.loc[bench_combos, m].values.astype(float)
            diff = (b - a) if LOWER_BETTER[m] else (a - b)
            res = combo_level_magnitude_test(diff, f"{bench}/{m}")
            res["metric"] = m
            res["analysis"] = f"benchmark_split_{bench}_all_vs_macro"
            res["n_wins"] = int((diff > 0).sum())
            summary_rows.append(res)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "family_significance_summary.csv", index=False)
    print(f"\nSaved -> {out_dir / 'family_significance_foldlevel.csv'}")
    print(f"Saved -> {out_dir / 'family_significance_summary.csv'}")


if __name__ == "__main__":
    main()
