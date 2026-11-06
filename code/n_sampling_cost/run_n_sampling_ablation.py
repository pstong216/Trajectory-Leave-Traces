#!/usr/bin/env python3
"""
run_n_sampling_ablation.py -- Mind2Web-only sample-budget (N) ablation for
the micro/UQ features, rerun against this release's CURRENT keystep labels
and CURRENT main-table CV protocol (2026-08-02).

Motivation: mind2web is the one benchmark where micro/UQ significantly
improves over macro-only (see ../feature_importance/NARRATIVE.md, Level 1:
paired t-test across 5 backbones, AUROC p=0.023). But micro features need
N LLM re-samples per step, unlike macro (free, N=1: computed from the
agent's own single generated trajectory). This script asks: how much of
that mind2web gain survives at N=3/5/8 instead of the production N=10, and
is it worth the extra sampling cost.

This REDOES an ablation the authors already ran once (see
`../../reviewers/Screenshot 2026-08-02 at 12.26.12 PM.png`, author response
to Reviewer jRcR, 11 Jul 2026) -- that version used the OLD keystep labels
and the OLD (pre-2026-08-01) CV protocol. Nothing here changes the
methodology; it only reruns the same question against the current,
corrected labels/protocol so the numbers in a revised response are not
stale relative to the rest of the paper (main table, family significance,
feature ablation all already moved to the 2026-08-01 refresh).

Everything reads from  only (2026-08-02: the N=3/5/8
feature_matrix.csv files were originally produced by
codes/subsample_n_ablation/subsample_reembed.py against results_total/'s
raw data and copied in here verbatim -- same Qwen3-Embedding-0.6B model,
same 0.3 cosine-distance clustering threshold, same ">=10 stored samples
to be eligible" policy as the N=10 baseline; row counts verified identical
to N=10 for all 5 backbones and 0 outcome-column mismatches before
copying, i.e. subsampling changes NOTHING about which steps are covered or
their labels, only how many resamples went into each step's UQ metrics).
See data/mind2web/<model>/embedded/uq_analysis_change_qwen_n{3,5,8}/
for the copied files and their manifest.json (N, strategy, seed, source).

Training/eval protocol: byte-for-byte the current main-table protocol,
imported (not reimplemented) from ../feature_importance/
run_feature_importance_ablation.py -- run_5fold_cv() (StratifiedKFold(5,
shuffle=True, random_state=42), task_length sample weighting, Logistic_L2
with inner C-search via fit_model()) and the same keystep relabeling
(load_keystep_map_entropy_source + the outcome->label masking loop used by
that script's own load_combo()).

Outputs (this folder's own results/):
  n_ablation_per_backbone.csv    per (N, backbone): auroc/e_aurc/brier, all vs macro_only
  n_ablation_summary.csv         mean +/- std across 5 mind2web backbones, delta vs macro_only
  n_ablation_significance.csv    paired t-test (N=5 backbones) per sample-budget vs macro_only
  cost_tradeoff.csv              sampling cost (N/10) x retained AUROC gain RG(N)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FI_DIR = Path(__file__).resolve().parent.parent / "feature_importance"
sys.path.insert(0, str(FI_DIR))
from run_feature_importance_ablation import (  # noqa: E402
    MACRO_FEATS, UQ_FEATS, ALL_FEATS, run_5fold_cv, PR,
)
from keystep_labels import load_keystep_map_entropy_source  # noqa: E402

MIND2WEB_MODELS = ["claude", "gpt_5_2", "gptnano", "kimi", "qwen30b"]
N_VALUES = [3, 5, 8]  # the sample budgets under test; N=10 = production reference


def canonical_n10_path(model: str) -> Path:
    return PR / "data" / "mind2web" / model / "embedded" / "uq_analysis_change_qwen" / "thr_0p3" / "feature_matrix.csv"


def subsampled_path(model: str, n: int) -> Path:
    return (PR / "data" / "mind2web" / model / "embedded"
            / f"uq_analysis_change_qwen_n{n}" / "feature_matrix.csv")


def load_labeled(fm_path: Path, model: str) -> pd.DataFrame:
    """Same keystep-relabeling recipe as run_feature_importance_ablation.py's
    load_combo() / run_website_generalization.py's load_ours_source() --
    applied identically regardless of which feature_matrix.csv is loaded, so
    N never changes what a step's label is, only its UQ feature values."""
    df = pd.read_csv(fm_path)
    annot = load_keystep_map_entropy_source(f"mind2web/{model}", str(PR / "keystep_labels" / "mind2web"))
    df["label"] = df["outcome"].copy()
    for tid, ks in annot.items():
        if ks is None:
            continue
        mask = df["task_id"].astype(str) == str(tid)
        if not mask.any():
            continue
        if int(df.loc[mask, "outcome"].iloc[0]) == 0:
            continue
        df.loc[mask & (df["step_k"] < ks), "label"] = 0
        df.loc[mask & (df["step_k"] >= ks), "label"] = 1
    return df


def main():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    rows = []

    print("=" * 70)
    print("Loading N=10 (production) + verifying population matches N<10 exactly...")
    for model in MIND2WEB_MODELS:
        n10_path = canonical_n10_path(model)
        df10 = load_labeled(n10_path, model)
        for n in N_VALUES:
            sp = subsampled_path(model, n)
            if not sp.is_file():
                raise FileNotFoundError(f"missing {sp}")
            dfn = pd.read_csv(sp)
            assert len(dfn) == len(df10), (
                f"{model} N={n}: row count {len(dfn)} != N=10's {len(df10)} -- "
                "population mismatch, subsampling policy may have changed")
        print(f"  {model:10s}  rows={len(df10):5d}  tasks={df10['task_id'].nunique():4d}  "
              f"(matches N=3/5/8 exactly)")

    print("\n" + "=" * 70)
    print("Macro-only reference (N-independent, no LLM sampling needed)...")
    macro_ref = {}
    for model in MIND2WEB_MODELS:
        df10 = load_labeled(canonical_n10_path(model), model)
        m = run_5fold_cv(df10, MACRO_FEATS)
        macro_ref[model] = m
        rows.append({"model": model, "n": 1, "config": "macro_only", **m})
        print(f"  {model:10s}  auroc={m['auroc']:.4f}  e_aurc={m['e_aurc']:.4f}  brier={m['brier']:.4f}")

    print("\n" + "=" * 70)
    print("Macro+Micro at N=10 (production reference, 'full' budget)...")
    full_all = {}
    for model in MIND2WEB_MODELS:
        df10 = load_labeled(canonical_n10_path(model), model)
        m = run_5fold_cv(df10, ALL_FEATS)
        full_all[model] = m
        rows.append({"model": model, "n": 10, "config": "all", **m})
        print(f"  {model:10s}  auroc={m['auroc']:.4f}  e_aurc={m['e_aurc']:.4f}  brier={m['brier']:.4f}")

    print("\n" + "=" * 70)
    for n in N_VALUES:
        print(f"Macro+Micro at N={n}...")
        for model in MIND2WEB_MODELS:
            dfn = load_labeled(subsampled_path(model, n), model)
            m = run_5fold_cv(dfn, ALL_FEATS)
            rows.append({"model": model, "n": n, "config": "all", **m})
            print(f"  {model:10s}  auroc={m['auroc']:.4f}  e_aurc={m['e_aurc']:.4f}  brier={m['brier']:.4f}")

    per_backbone = pd.DataFrame(rows)
    per_backbone.to_csv(out_dir / "n_ablation_per_backbone.csv", index=False)

    # ── Paired deltas (all@N minus macro_only), per backbone first, then mean/std ──
    print("\n" + "=" * 70)
    print("Delta vs macro-only (paired per backbone, then mean +/- std across 5 backbones)")
    print("=" * 70)
    summary_rows = []
    sig_rows = []
    for n in N_VALUES + [10]:
        d_auroc, d_eaurc, d_brier = [], [], []
        for model in MIND2WEB_MODELS:
            all_m = per_backbone[(per_backbone.model == model) & (per_backbone.n == n) & (per_backbone.config == "all")].iloc[0]
            ref = macro_ref[model]
            d_auroc.append(all_m["auroc"] - ref["auroc"])
            d_eaurc.append(all_m["e_aurc"] - ref["e_aurc"])
            d_brier.append(all_m["brier"] - ref["brier"])
        d_auroc, d_eaurc, d_brier = np.array(d_auroc), np.array(d_eaurc), np.array(d_brier)
        row = {
            "n": n,
            "d_auroc_mean": round(float(d_auroc.mean()), 4), "d_auroc_std": round(float(d_auroc.std()), 4),
            "d_eaurc_mean": round(float(d_eaurc.mean()), 4), "d_eaurc_std": round(float(d_eaurc.std()), 4),
            "d_brier_mean": round(float(d_brier.mean()), 4), "d_brier_std": round(float(d_brier.std()), 4),
        }
        summary_rows.append(row)
        print(f"  N={n:2d}  dAUROC={row['d_auroc_mean']:+.4f}+/-{row['d_auroc_std']:.4f}  "
              f"dE-AURC={row['d_eaurc_mean']:+.4f}+/-{row['d_eaurc_std']:.4f}  "
              f"dBrier={row['d_brier_mean']:+.4f}+/-{row['d_brier_std']:.4f}")

        for metric_name, diffs in [("auroc", d_auroc), ("e_aurc", -d_eaurc), ("brier", -d_brier)]:
            # e_aurc/brier flipped so positive = improvement, matching auroc's sign convention
            t_stat, p_t = stats.ttest_1samp(diffs, 0.0)
            try:
                w_stat, p_w = stats.wilcoxon(diffs)
            except ValueError:
                p_w = float("nan")
            sig_rows.append({"n": n, "metric": metric_name, "mean_diff": round(float(diffs.mean()), 4),
                              "t_p": round(float(p_t), 4), "wilcoxon_p": round(float(p_w), 4) if not np.isnan(p_w) else np.nan})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "n_ablation_summary.csv", index=False)
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(out_dir / "n_ablation_significance.csv", index=False)

    print("\n" + "=" * 70)
    print("Significance (N=5 backbones, paired t-test + Wilcoxon vs macro_only; "
          "positive mean_diff = improvement on all three metrics)")
    print(sig_df.to_string(index=False))

    # ── Cost/tradeoff: $-cost = N/10 (decode volume); time-cost from the
    # reviewer-response formulas T(N;c,kappa), evaluated at illustrative
    # normalized units (kappa=0.1, t_p=t_d=1 -- shape-only, not real seconds,
    # per that response's own "avoid universal wall-clock numbers" caution).
    print("\n" + "=" * 70)
    print("Cost-performance tradeoff, RG(N) = [AUROC(N)-AUROC(macro_only)] / [AUROC(10)-AUROC(macro_only)]")
    print("=" * 70)
    KAPPA, T_P, T_D = 0.1, 1.0, 1.0

    def t_seq(n):
        return (T_P + T_D) + (n - 1) * (KAPPA * T_P + T_D)

    def t_par(n):
        return T_P + T_D + (KAPPA * T_P + T_D)  # constant, requires concurrency c >= N-1

    t_seq_n10 = t_seq(10)
    full_gain = summary_df.loc[summary_df.n == 10, "d_auroc_mean"].iloc[0]
    cost_rows = []
    for n in N_VALUES + [10]:
        gain_n = summary_df.loc[summary_df.n == n, "d_auroc_mean"].iloc[0]
        rg = round(float(gain_n / full_gain), 3) if full_gain != 0 else float("nan")
        cost_rows.append({
            "n": n,
            "dollar_cost_frac_of_N10": round(n / 10, 2),
            "time_cost_sequential_frac_of_N10": round(t_seq(n) / t_seq_n10, 3),
            "time_cost_parallel_frac_of_N10": round(t_par(n) / t_seq_n10, 3),
            "d_auroc_vs_macro_only": gain_n,
            "RG_auroc": rg,
        })
        print(f"  N={n:2d}  $cost={n/10:.2f}  seq_time={t_seq(n)/t_seq_n10:.3f}  "
              f"par_time={t_par(n)/t_seq_n10:.3f}  dAUROC={gain_n:+.4f}  RG(N)={rg:.3f}")

    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(out_dir / "cost_tradeoff.csv", index=False)

    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
