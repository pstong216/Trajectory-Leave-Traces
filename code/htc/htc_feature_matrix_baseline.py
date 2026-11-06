from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

RESULTS_ROOT = Path(__file__).resolve().parent.parent

ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0,
              6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 50.0]

EXCLUDE_DEFAULT = {
    "task_id", "source", "outcome", "label",
    "conf_at_k", "conf_cumul_mean",
}


def load_annotation(annot_path: Path) -> dict[str, int | None]:
    with open(annot_path) as f:
        data = json.load(f)
    return {str(v["taskId"]): v.get("keyStep") for v in data["items"].values()}


def resolve_annotation_path(results_dir: Path, source: str) -> Path:
    stem = source.replace("/", "_")
    annot_dir = results_dir / "annotate_vis" / "result"
    for cand in (annot_dir / f"{stem}_gemini.json", annot_dir / f"{stem}.json"):
        if cand.exists():
            return cand
    return annot_dir / f"{stem}_gemini.json"


def apply_keystep_labels(df: pd.DataFrame, annotation: dict[str, int | None]) -> pd.DataFrame:
    df = df.copy()
    task_id_str = df["task_id"].astype(str)
    annotated_with_ks = {tid for tid, ks in annotation.items() if ks is not None}

    failed_mask = df["outcome"] == 1
    keep_mask = (~failed_mask) | task_id_str.isin(annotated_with_ks)
    dropped_tasks = df.loc[failed_mask & ~task_id_str.isin(annotated_with_ks), "task_id"].nunique()
    df = df[keep_mask].copy()

    df["label"] = df["outcome"].copy()
    relabeled = 0
    for tid, ks in annotation.items():
        if ks is None:
            continue
        mask = df["task_id"].astype(str) == str(tid)
        if not mask.any():
            continue
        if int(df.loc[mask, "outcome"].iloc[0]) == 0:
            continue
        df.loc[mask & (df["step_k"] < ks), "label"] = 0
        df.loc[mask & (df["step_k"] >= ks), "label"] = 1
        relabeled += 1

    print(f"  [keystep] relabeled={relabeled} failed tasks  "
          f"dropped={dropped_tasks} unannotated failed tasks  "
          f"remaining tasks={df['task_id'].nunique()}", flush=True)
    return df


def compute_ece_equal_width(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob > lo) & (y_prob <= hi)
        if m.sum() > 0:
            val += abs(y_prob[m].mean() - y_true[m].mean()) * m.mean()
    return float(val)


def safe_medians(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].median().fillna(0.0)


def choose_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in EXCLUDE_DEFAULT:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def best_alpha_for_fold(X_tr: np.ndarray, y_tr: np.ndarray, groups: np.ndarray) -> float:
    """Grouped by task_id (2026-07-30, same fix as train_risk_model_temp_sca.py's
    best_alpha_htc(): X_tr/y_tr are step-level, plain StratifiedKFold let steps
    from the same task span the inner train/val split)."""
    best_score, best_alpha = -np.inf, 1.0
    inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)
    for alpha in ALPHA_GRID:
        C = 1.0 / alpha
        model = LogisticRegression(
            penalty="l1", C=C, solver="liblinear",
            max_iter=1000, random_state=42,
        )
        au_l, br_l, ec_l = [], [], []
        for itr, iva in inner_cv.split(X_tr, y_tr, groups=groups):
            if len(np.unique(y_tr[itr])) < 2 or len(np.unique(y_tr[iva])) < 2:
                continue
            model.fit(X_tr[itr], y_tr[itr])
            p = model.predict_proba(X_tr[iva])[:, 1]
            au_l.append(roc_auc_score(y_tr[iva], p))
            br_l.append(brier_score_loss(y_tr[iva], p))
            ec_l.append(compute_ece_equal_width(y_tr[iva], p))
        if not au_l:
            continue
        score = float(np.mean(au_l)) - float(np.mean(br_l)) - float(np.mean(ec_l))
        if score > best_score:
            best_score, best_alpha = score, alpha
    return best_alpha


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    prob = np.clip(y_prob.astype(float), 1e-7, 1 - 1e-7)
    return {
        "auroc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "auprc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier_raw": round(float(brier_score_loss(y_true, prob)), 4),
        "ece_raw": round(float(compute_ece_equal_width(y_true, prob)), 4),
    }


def summarise(rows: list[dict]) -> pd.DataFrame:
    out = []
    for method in sorted({r["method"] for r in rows}):
        sub = [r for r in rows if r["method"] == method]
        for metric in ("auroc", "auprc", "brier_raw", "ece_raw"):
            vals = [r[metric] for r in sub if metric in r]
            out.append({
                "method": method,
                "metric": metric,
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "n_folds": len(vals),
            })
    return pd.DataFrame(out)


def run_cv(df: pd.DataFrame, feat_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_meta = (df.sort_values("step_k")
                   .groupby("task_id", as_index=False)
                   .agg(task_final_failure=("label", "max")))

    preds_full = np.full(len(df), np.nan, dtype=float)
    preds_reduced = np.full(len(df), np.nan, dtype=float)
    metric_rows = []

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(
        cv.split(task_meta["task_id"].values, task_meta["task_final_failure"].values.astype(int)), 1
    ):
        train_tasks = set(task_meta.iloc[tr_idx]["task_id"])
        val_tasks = set(task_meta.iloc[va_idx]["task_id"])

        train_df = df[df["task_id"].isin(train_tasks)].copy()
        val_df = df[df["task_id"].isin(val_tasks)].copy()

        medians = safe_medians(train_df, feat_cols)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_df[feat_cols].fillna(medians).fillna(0.0).values)
        X_va = scaler.transform(val_df[feat_cols].fillna(medians).fillna(0.0).values)
        y_tr = train_df["label"].values.astype(int)
        y_va = val_df["label"].values.astype(int)

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            continue

        full = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs",
            max_iter=2000, random_state=42,
        )
        full.fit(X_tr, y_tr)
        p_full = full.predict_proba(X_va)[:, 1]
        preds_full[val_df.index] = p_full
        m_full = evaluate(y_va, p_full)
        m_full.update({"fold": fold, "method": "htc_style_l2"})
        metric_rows.append(m_full)

        best_alpha = best_alpha_for_fold(X_tr, y_tr, train_df["task_id"].values)
        reduced = LogisticRegression(
            penalty="l1", C=1.0 / best_alpha, solver="liblinear",
            max_iter=2000, random_state=42,
        )
        reduced.fit(X_tr, y_tr)
        p_red = reduced.predict_proba(X_va)[:, 1]
        preds_reduced[val_df.index] = p_red
        m_red = evaluate(y_va, p_red)
        m_red.update({"fold": fold, "method": "htc_style_l1"})
        metric_rows.append(m_red)

        print(f"  fold {fold}: L2 AUROC={m_full['auroc']}  Brier_raw={m_full['brier_raw']}  "
              f"L1 AUROC={m_red['auroc']}  Brier_raw={m_red['brier_raw']}", flush=True)

    pred_df = df[["task_id", "step_k", "outcome", "label"]].copy()
    pred_df["htc_style_l2_prob"] = preds_full
    pred_df["htc_style_l1_prob"] = preds_reduced
    return pd.DataFrame(metric_rows), pred_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--annot_path", default=None)
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = RESULTS_ROOT / csv_path

    out_dir = Path(args.output_dir) if args.output_dir else (
        RESULTS_ROOT / "uq_results" / "htc_code_reuse" / args.source.replace("/", "_")
    )
    if not out_dir.is_absolute():
        out_dir = RESULTS_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        annot_path = Path(args.annot_path) if args.annot_path else resolve_annotation_path(RESULTS_ROOT, args.source)
        if not annot_path.is_absolute():
            annot_path = RESULTS_ROOT / annot_path
        annotation = load_annotation(annot_path)
        print(f"Annotation : {annot_path}  ({len(annotation)} entries)")
        df = apply_keystep_labels(df, annotation)
    else:
        print("Annotation : skipped (label column already present)")

    df = df.reset_index(drop=True)

    feat_cols = choose_feature_cols(df)
    print(f"Source     : {args.source}")
    print(f"Matrix     : {csv_path}")
    print(f"Output     : {out_dir}")
    print(f"Tasks      : {df['task_id'].nunique()}  Steps: {len(df)}")
    print(f"Features   : {len(feat_cols)}", flush=True)

    metrics_df, pred_df = run_cv(df, feat_cols)
    summary_df = summarise(metrics_df.to_dict("records"))

    metrics_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "summary_table.csv", index=False)
    pred_df.to_csv(out_dir / "cv_predictions.csv", index=False)

    print(f"\nDone. Results in {out_dir}")
    print("  fold_metrics.csv")
    print("  summary_table.csv")
    print("  cv_predictions.csv")


if __name__ == "__main__":
    main()
