"""
compute_m2_cumulative.py -- fills the gap identified: M2 (action_logprob_mean)
has no cumulative (0->k) counterpart, unlike M3->M4 and M1->M7.

Computes m2c_cumulative_action_logprob_mean = running mean of
m2_action_logprob_mean over steps 0..k for each task, exactly the same
recipe used for M4 (cumulative of M3, in compute_baselines.py's
score_verbalized_task) and M7 (cumulative of M1, add_m7_from_m1() in
update_baselines_with_features.py).

Reads directly from the already-computed
reproduction_output/baseline/<benchmark>/<model>/results/per_step_scores.csv
for all 10 combos -- no retraining, no re-scraping of raw logprobs, so this
can't disagree with the pipeline's own M2/M4/M7 numbers.

Evaluates AUROC / e-AURC / Brier for M2 (at-k) vs the new M2-cumulative,
using the identical scoring convention as update_baselines_with_features.py's
evaluate_method(): m2 is "lower is more confident" (it's a mean negative-
log-prob, HIGHER_IS_CONFIDENT["m2_action_logprob_mean"] = False), so
confidence = -raw_value in both cases.

Output: results/m2_cumulative_comparison.csv (one row per combo + a mean row).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import MinMaxScaler

REPRO_ROOT = Path(__file__).resolve().parents[2] / "reproduction_output" / "baseline"
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(exist_ok=True)

COMBOS = [
    ("mind2web", "claude"), ("mind2web", "gpt_5_2"), ("mind2web", "gptnano"),
    ("mind2web", "kimi"), ("mind2web", "qwen30b"),
    ("webarena", "claude"), ("webarena", "gpt_5_2"), ("webarena", "gptnano"),
    ("webarena", "kimi"), ("webarena", "qwen30b"),
]


def add_m2_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    """Running mean of m2_action_logprob_mean, 0->k, per task -- same recipe
    as add_m7_from_m1() / the m3_running accumulator in score_verbalized_task()."""
    df = df.copy().reset_index(drop=True)
    id_col = "task_index" if "task_index" in df.columns else "task_id"
    m2c = np.full(len(df), np.nan)
    for _, idx in df.groupby(id_col).groups.items():
        grp_idx = df.loc[idx].sort_values("step").index
        vals = []
        for i in grp_idx:
            v = df.at[i, "m2_action_logprob_mean"]
            if pd.notna(v):
                vals.append(float(v))
            m2c[i] = float(np.mean(vals)) if vals else np.nan
    df["m2c_cumulative_action_logprob_mean"] = m2c
    return df


def aurc(scores, labels):
    n = len(scores)
    order = np.argsort(-scores)
    errors = (1 - labels[order]).astype(float)
    risks = np.cumsum(errors) / np.arange(1, n + 1)
    return float(np.mean(risks))


def aurc_oracle(labels):
    n = len(labels)
    n_f = int((1 - labels).sum())
    n_s = n - n_f
    if n_f == 0:
        return 0.0
    if n_s == 0:
        return 1.0
    i_vals = np.arange(n_s + 1, n + 1, dtype=float)
    return float(np.sum((i_vals - n_s) / i_vals) / n)


def evaluate(col: str, df: pd.DataFrame) -> dict:
    group_col = "task_index" if "task_index" in df.columns else "task_id"
    sub = df[[group_col, "success", col]].dropna()
    if len(sub) < 10 or sub["success"].nunique() < 2:
        return dict(n=len(sub), auroc=np.nan, e_aurc=np.nan, brier=np.nan)
    labels = sub["success"].values.astype(int)
    raw = sub[col].values.astype(float)
    conf = -raw  # HIGHER_IS_CONFIDENT["m2_action_logprob_mean"] = False
    auroc = roc_auc_score(labels, conf)
    aurc_val = aurc(conf, labels)
    e_aurc = aurc_val - aurc_oracle(labels)
    # NOTE (2026-08-31): this brier column is UNCALIBRATED -- MinMax on the
    # full data is a range rescale, not a proper calibrator. It exists only
    # as a rough intermediate diagnostic in this script's own CSV; nothing
    # paper-facing consumes it. The main table's M15 brier comes from
    # baseline_calculation/update_baselines_with_features.py's held-out
    # brier_platt (see add_m15_to_main_table.py), and the generalization
    # suite's M15 brier from run_website_generalization.py's held-out Platt.
    probs = MinMaxScaler().fit_transform(conf.reshape(-1, 1)).ravel()
    brier = brier_score_loss(labels, np.clip(probs, 1e-7, 1 - 1e-7))
    return dict(n=len(sub), auroc=round(auroc, 4), e_aurc=round(e_aurc, 4),
                brier=round(brier, 4))


def main():
    rows = []
    for benchmark, model in COMBOS:
        csv_path = REPRO_ROOT / benchmark / model / "results" / "per_step_scores.csv"
        df = pd.read_csv(csv_path)
        df = add_m2_cumulative(df)

        m2_stats = evaluate("m2_action_logprob_mean", df)
        m2c_stats = evaluate("m2c_cumulative_action_logprob_mean", df)

        rows.append(dict(
            combo=f"{benchmark}_{model}",
            n=m2_stats["n"],
            m2_auroc=m2_stats["auroc"], m2c_auroc=m2c_stats["auroc"],
            d_auroc=round(m2c_stats["auroc"] - m2_stats["auroc"], 4),
            m2_e_aurc=m2_stats["e_aurc"], m2c_e_aurc=m2c_stats["e_aurc"],
            d_e_aurc=round(m2c_stats["e_aurc"] - m2_stats["e_aurc"], 4),
            m2_brier=m2_stats["brier"], m2c_brier=m2c_stats["brier"],
            d_brier=round(m2c_stats["brier"] - m2_stats["brier"], 4),
        ))

    res = pd.DataFrame(rows)
    # claude has no token-logprobs in this pipeline at all (m2/m3/m4 are
    # NaN for it already in per_step_scores.csv -- not a bug introduced
    # here), so average only over the 8 combos where m2 exists.
    valid = res.dropna(subset=["m2_auroc"])
    mean_row = valid.drop(columns=["combo", "n"]).mean(numeric_only=True).to_dict()
    mean_row["combo"] = "MEAN (8 combos w/ logprobs, excl. claude)"
    mean_row["n"] = valid["n"].sum()
    res = pd.concat([res, pd.DataFrame([mean_row])], ignore_index=True)

    out_csv = OUT_DIR / "m2_cumulative_comparison.csv"
    res.to_csv(out_csv, index=False)
    print(res.to_string(index=False))
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
