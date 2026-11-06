#!/usr/bin/env python3
"""
update_main_table_ours_rows.py — Refresh only the three "Ours - *" rows of
main_results_table.csv from the latest run_all_final.sh
output (2026-08-01, after flipping Ours' outer-CV default to
shuffle=True -- see train_risk_model_temp_sca.py::run_cv()'s docstring).

main_results_table.csv has no committed generator elsewhere in this repo
(checked); this script only touches the 3 rows that depend on
train_risk_model_temp_sca.py ("Ours - macro+micro" / "Ours - macro only" /
"Ours - micro/UQ only") -- M1-M13 baseline rows are untouched, since
they don't depend on Ours' outer-CV setting at all.

Verified before use: for each combo, reads the LATEST (by mtime)
summary_table.csv under code/l2_training/output/<bench>_<model>/.../,
picks feature_set in {all, macro_only, uq_only} x metric in
{auroc, e_aurc, brier, ece_adaptive}, and re-derives the __mean/__std
columns as sample mean / sample std (ddof=1) across the 10 combos --
confirmed this ddof=1 formula reproduces the file's own pre-existing
__mean/__std columns for an untouched row (M1) to 4-decimal precision
before trusting it to write new values.

Usage: python3 update_main_table_ours_rows.py [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

L2_DIR = Path(__file__).resolve().parent
PR = L2_DIR.parent.parent
MAIN_TABLE = PR / "main_results_table.csv"
OUTPUT_ROOT = PR / "reproduction_output" / "ours"

COMBOS = [
    ("webarena", "claude"), ("webarena", "gpt_5_2"), ("webarena", "gptnano"),
    ("webarena", "kimi"), ("webarena", "qwen30b"),
    ("mind2web", "claude"), ("mind2web", "gpt_5_2"), ("mind2web", "gptnano"),
    ("mind2web", "kimi"), ("mind2web", "qwen30b"),
]

# (main_results_table.csv row label, summary_table.csv feature_set value)
ROW_TO_FEATURE_SET = {
    "Ours - macro+micro":   "all",
    "Ours - macro only":    "macro_only",
    "Ours - micro/UQ only": "uq_only",
}
METRIC_MAP = {"auroc": "auroc", "e_aurc": "e_aurc", "brier": "brier", "ece": "ece_adaptive"}


def latest_summary_table(bench: str, model: str) -> pd.DataFrame:
    pattern = OUTPUT_ROOT / f"{bench}_{model}" / "thr_0p3" / "train_risk_model_temp_sca"
    runs = sorted(pattern.glob("*/summary_table.csv"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"no summary_table.csv found for {bench}/{model} under {pattern}")
    return pd.read_csv(runs[-1]), runs[-1]


def main():
    global OUTPUT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the new values, don't write the file")
    ap.add_argument("--output_root", type=Path, default=OUTPUT_ROOT,
                    help="root holding <bench>_<model>/thr_0p3/train_risk_model_temp_sca/*/summary_table.csv")
    args = ap.parse_args()
    OUTPUT_ROOT = args.output_root

    df = pd.read_csv(MAIN_TABLE)

    per_combo = {}  # (row_label, metric) -> {combo_col_suffix: value}
    used_files = {}
    for bench, model in COMBOS:
        st, path = latest_summary_table(bench, model)
        used_files[f"{bench}_{model}"] = path
        for row_label, fs in ROW_TO_FEATURE_SET.items():
            sub = st[(st.feature_set == fs) & (st.model == "Logistic_L2")].set_index("metric")
            for out_metric, st_metric in METRIC_MAP.items():
                if st_metric not in sub.index:
                    raise KeyError(f"{st_metric} missing for {bench}/{model} feature_set={fs} in {path}")
                val = float(sub.loc[st_metric, "mean"])
                per_combo.setdefault((row_label, out_metric), {})[f"{bench}_{model}"] = val

    print("Source files used (latest summary_table.csv per combo):")
    for k, v in used_files.items():
        print(f"  {k:20s} {v}")

    combo_suffixes = [f"{b}_{m}" for b, m in COMBOS]
    for row_label in ROW_TO_FEATURE_SET:
        row_idx = df.index[df["method"] == row_label]
        if len(row_idx) != 1:
            raise KeyError(f"expected exactly 1 row for {row_label!r}, found {len(row_idx)}")
        ridx = row_idx[0]
        for out_metric in METRIC_MAP:
            vals = np.array([per_combo[(row_label, out_metric)][c] for c in combo_suffixes])
            for c, v in zip(combo_suffixes, vals):
                col = f"{out_metric}__{c}"
                old = df.at[ridx, col]
                df.at[ridx, col] = round(float(v), 4)
                print(f"  [{row_label}] {col}: {old} -> {round(float(v), 4)}")
            mean_col, std_col = f"{out_metric}__mean", f"{out_metric}__std"
            new_mean, new_std = round(float(vals.mean()), 4), round(float(vals.std(ddof=1)), 4)
            print(f"  [{row_label}] {mean_col}: {df.at[ridx, mean_col]} -> {new_mean}")
            print(f"  [{row_label}] {std_col}: {df.at[ridx, std_col]} -> {new_std}")
            df.at[ridx, mean_col] = new_mean
            df.at[ridx, std_col] = new_std

    if args.dry_run:
        print("\n--dry-run: not writing.")
        return

    df.to_csv(MAIN_TABLE, index=False)
    print(f"\nWrote -> {MAIN_TABLE}")


if __name__ == "__main__":
    main()
