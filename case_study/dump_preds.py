#!/usr/bin/env python3
"""Re-run the macro_internal CV but KEEP per-row predictions.

Reuses run_macro_internal.py's own build_joined/make_folds and the same
fit_model/safe_medians/MACRO_FEATS, with identical folds (seed=42), so the
aggregate metrics reproduce their macro_internal_per_combo.csv exactly.
Only difference: predictions are retained instead of discarded.

Output: preds_<bench>_<model>.csv with columns
  join_id, task_id, step_k, label, outcome, p_macro, p_htc, p_mi
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd

MI = Path(__file__).resolve().parent.parent / "code" / "macro_internal"
sys.path.insert(0, str(MI))
import run_macro_internal as R  # noqa: E402
from train_risk_model_temp_sca import fit_model, safe_medians  # noqa: E402

OUT = Path(__file__).resolve().parent / "preds"
OUT.mkdir(exist_ok=True)

HTC_RENAMED = [f"htc_{c}" for c in R.HTC_FEATS]
FEATURE_SETS = [
    ("p_macro", R.MACRO_FEATS),
    ("p_htc", HTC_RENAMED),
    ("p_mi", R.MACRO_FEATS + HTC_RENAMED),
]


def cv_predictions(df, feat_cols, folds):
    """Same loop as run_5fold_cv_prebuilt, but returns a per-row prob Series."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for tr_tasks, te_tasks in folds:
        train_df = df[df["join_id"].isin(tr_tasks)]
        test_df = df[df["join_id"].isin(te_tasks)]
        y_tr = train_df["label"].values.astype(int)
        if len(np.unique(y_tr)) < 2:
            continue
        medians = safe_medians(train_df, feat_cols)
        X_tr = train_df[feat_cols].fillna(medians).fillna(0.0).values
        X_te = test_df[feat_cols].fillna(medians).fillna(0.0).values
        step_counts = train_df.groupby("join_id")["step_k"].transform("count")
        sample_weight = (1.0 / step_counts).values
        clf = fit_model("Logistic_L2", X_tr, y_tr, sample_weight,
                        task_ids=train_df["join_id"].values)
        out.loc[test_df.index] = clf.predict_proba(X_te)[:, 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", default="all",
                    help="'all' or comma list like webarena/gptnano,mind2web/kimi")
    args = ap.parse_args()

    if args.combos == "all":
        combos = [(b, m) for b, m in R.COMBOS if m != "claude"]
    else:
        combos = [tuple(c.split("/")) for c in args.combos.split(",")]

    for bench, model in combos:
        print(f"\n=== {bench}/{model} ===", flush=True)
        joined, st = R.build_joined(bench, model)
        if joined.empty or joined["label"].nunique() < 2:
            print("  [SKIP]")
            continue
        joined = joined.reset_index(drop=True)
        print(f"  joined {st['joined_rows']} rows / {st['joined_tasks']} tasks", flush=True)

        folds = R.make_folds(joined)
        keep = ["join_id", "step_k", "label", "outcome"]
        if "task_id" in joined.columns and "task_id" not in keep:
            keep.insert(1, "task_id")
        res = joined[keep].copy()

        for col, feats in FEATURE_SETS:
            avail = [c for c in feats if c in joined.columns]
            res[col] = cv_predictions(joined, avail, folds)
            from sklearn.metrics import roc_auc_score
            ok = res[col].notna()
            auc = roc_auc_score(res.loc[ok, "label"], res.loc[ok, col])
            print(f"  {col:8s} ({len(avail):3d} feat)  auroc={auc:.4f}", flush=True)

        p = OUT / f"preds_{bench}_{model}.csv"
        res.to_csv(p, index=False)
        print(f"  -> {p}", flush=True)


if __name__ == "__main__":
    main()
