# Baseline README

This folder contains the baseline UQ pipeline used for `mind2web` and `webarena`.

The important point is that the current baseline pipeline is **step-level**, uses
**key-step relabeling**, and writes the final `metrics.csv` through
[`update_baselines_with_features.py`](./update_baselines_with_features.py).

## Current Pipeline

The active pipeline has 3 stages:

1. Compute `M1-M6`
2. Compute feature baselines `M8-M10`
3. Merge everything, add `M7`, add step-level HTC `M11-M12`, and write final metrics

The active scripts are:

- [`compute_baselines.py`](./compute_baselines.py)
  For `mind2web`
- [`compute_webarena_step_baselines.py`](./compute_webarena_step_baselines.py)
  For `webarena`
- [`compute_mean_feature_baseline.py`](./compute_mean_feature_baseline.py)
  Builds `M8-M10`
- [`update_baselines_with_features.py`](./update_baselines_with_features.py)
  Final merge script; writes final `per_step_scores.csv`, `task_scores.csv`, `metrics.csv`
- [`htc_step_baseline.py`](./htc_step_baseline.py)
  Step-level HTC implementation used by the merge script
- [`keystep_labels.py`](./keystep_labels.py)
  Key-step label loader and relabeling helpers

## Methods

The current baseline methods are:

- `M1`: verbalized confidence
- `M2`: action token mean negative logprob
- `M3`: full output mean negative logprob
- `M4`: cumulative mean of `M3`
- `M5`: action entropy over multi-sample predictions
- `M6`: cumulative mean of `M5`
- `M7`: cumulative mean of `M1`
- `M8`: mean-feature score using micro features
- `M9`: mean-feature score using macro features
- `M10`: mean-feature score using all features
- `M11`: step-level HTC full
- `M12`: step-level HTC reduced
- `M13`: stacked logistic over the M1-M7 score vector, HTC_L2 protocol
- `M14`: stacked logistic over the M1-M7 score vector, Logistic_L2 protocol
  (same classifier/protocol as the paper main-table Macro/Micro/M&M rows)

## Labeling

The pipeline uses **key-step aligned step labels**.

For a failed trajectory with `keyStep = k`:

- steps `< k` are labeled `success`
- steps `>= k` are labeled `failure`

For successful trajectories:

- all steps are labeled `success`

Label logic lives in:

- [`keystep_labels.py`](./keystep_labels.py)

### Annotation directories

Current annotation sources:

- `mind2web`: `annotate_vis/result`
- `webarena`: `annotation/result`

Important for `webarena`:

- files ending with `_m` are **not** used for the baseline path
- the baseline path uses files like `webarena_<model>_gemini.json`

## Output Files

Each model result directory contains:

- `per_step_scores.csv`
  Step-level baseline scores and labels
- `task_scores.csv`
  Task-level aggregated scores (`_last` and `_mean`)
- `metrics.csv`
  Final evaluation metrics
- `htc_step_scores.csv`
  Present only when HTC is available

## Metric Semantics

The final metrics are written only by:

- [`update_baselines_with_features.py`](./update_baselines_with_features.py)

### AUROC

- ranking / discrimination
- higher is better

### AURC

- selective prediction risk-coverage area
- lower is better internally
- in `metrics.csv` we report `aurc_gain = aurc_ref - aurc`
- so `aurc_gain` higher is better

### ECE

- calibration quality
- lower is better

### Brier

- probability prediction error
- lower is better

## Calibration

### M1-M10

`M1-M10` use post-hoc calibration inside
[`update_baselines_with_features.py`](./update_baselines_with_features.py):

- `Platt`
- `Isotonic`
- `Temperature`

Important details:

- calibration is done on **step rows**
- folds are split by **task**
- implementation uses `StratifiedGroupKFold`
- each fold is about `80% tasks train / 20% tasks val`

For `M1-M10`:

- `ece_raw` / `brier_raw` are computed from a min-max scaled raw score
- calibrated variants are reported as:
  - `ece_platt`, `ece_isotonic`, `ece_temp`
  - `brier_platt`, `brier_isotonic`, `brier_temp`

### M11-M12 (HTC)

HTC is treated differently.

- HTC outputs are already `LogisticRegression.predict_proba()`
- HTC does **not** use min-max
- HTC does **not** use extra calibration

So for `M11-M12`:

- `ece_raw` / `brier_raw` are computed directly from HTC probabilities
- calibrated columns are left empty (`NaN`)

## HTC

The active HTC implementation is:

- [`htc_step_baseline.py`](./htc_step_baseline.py)

This is **step-level HTC**, not the old task-level HTC.

It works as:

1. Load token logprobs and top logprobs
2. Extract 48 HTC handcrafted features on each prefix
3. Train logistic regression
4. Output step-level HTC probabilities

Current HTC predictors:

- `M11`: L2 logistic regression
- `M12`: L1 logistic regression

`StandardScaler` is used inside each training fold before logistic regression.

## Stacked Logistic (M13-M14)

Implemented in:

- [`stacked_logistic_baseline.py`](./stacked_logistic_baseline.py)

Supervised baselines that compose a per-step feature vector from the
external baseline scalar scores `M1-M7` (read from each model's
`per_step_scores.csv`) and train logistic regression under two protocols:

`M13` — HTC_L2 protocol (same as `htc_step_baseline.py` / M11):

- task-level `StratifiedKFold(5, shuffle, seed=42)` on task final success
- per fold: median imputation (train medians) + `StandardScaler`
- `LogisticRegression(penalty="l2", C=1.0, solver="lbfgs")`, no sample weights

`M14` — Logistic_L2 protocol (same as
`codes/train_risk_model/train_risk_model_temp_sca.py`, the classifier behind
the paper main table's Macro/Micro/M&M rows):

- task-level `StratifiedKFold(5)` with NO shuffle, tasks in first-appearance
  order, stratified on task final outcome
- `sample_weight = task_length` (each step weighted 1/n_steps of its task)
- `StandardScaler` + `GridSearchCV` over `C in {0.01, 0.1, 1.0, 10.0}`
  (inner cv=3, scoring=`roc_auc`), `LogisticRegression(max_iter=2000, lbfgs)`

`M8-M10` are deliberately excluded from the vector — they are derived from
our own HLVM feature matrix, not external baseline signals.

Columns that are entirely missing for a model (e.g. `M2-M4` when no logprobs)
are dropped; rows need at least one observed feature. Outputs are out-of-fold
`P(success)` probabilities — probability-native, no post-hoc calibration.

Run with:

```bash
python3 baseline/stacked_logistic_baseline.py
```

Outputs:

- `baseline/<benchmark>/<model>/results/stacked_step_scores.csv`
- `baseline/stacked_logistic_metrics.csv` (AUROC, raw AURC, proper E-AURC = AURC - AURC_oracle, aurc_gain, Brier, ECE)

Step-level results (2026-07-09), M13 / M14:

| Source | dims | AUROC | E-AURC | Brier |
|--------|------|-------|--------|-------|
| webarena/claude | 4 | 0.580 / 0.569 | 0.200 / 0.252 | 0.244 / 0.255 |
| webarena/gpt_5_2 | 7 | 0.525 / 0.534 | 0.308 / 0.287 | 0.253 / 0.252 |
| webarena/gptnano | 7 | 0.581 / 0.572 | 0.286 / 0.316 | 0.205 / 0.208 |
| webarena/kimi | 7 | 0.690 / 0.664 | 0.204 / 0.225 | 0.220 / 0.233 |
| webarena/qwen30b | 7 | 0.696 / 0.723 | 0.189 / 0.176 | 0.158 / 0.157 |
| mind2web/claude | 4 | 0.656 / 0.657 | 0.247 / 0.251 | 0.195 / 0.202 |
| mind2web/gpt_5_2 | 7 | 0.589 / 0.607 | 0.264 / 0.256 | 0.162 / 0.167 |
| mind2web/gptnano | 7 | 0.618 / 0.635 | 0.199 / 0.198 | 0.100 / 0.101 |
| mind2web/kimi | 7 | 0.583 / 0.616 | 0.288 / 0.271 | 0.173 / 0.173 |
| mind2web/qwen30b | 7 | 0.738 / 0.718 | 0.123 / 0.128 | 0.060 / 0.063 |

(claude has no logprobs, so its vector is M1, M5, M6, M7 only.)

Caveat: single-method AUROCs in `metrics.csv` are computed on each method's
non-null rows, while the stacked model uses all rows with >=1 observed
feature (missing values median-imputed), so `n` differs slightly.

## Dataset-Specific Usage

## Mind2Web

Use:

- [`compute_baselines.py`](./compute_baselines.py)
- [`compute_mean_feature_baseline.py`](./compute_mean_feature_baseline.py)
- [`update_baselines_with_features.py`](./update_baselines_with_features.py)

Typical pattern:

```bash
python3 baseline/compute_baselines.py \
  --verbalized_dir mind2web/results_gpt_5_2_verbalized \
  --entropy_dir mind2web/gpt_5_2 \
  --output_dir baseline/mind2web/gpt_5_2/results \
  --ground_truth mind2web/gpt_5_2_result.json \
  --model gpt_5_2 \
  --benchmark mind2web \
  --annotation_dir annotate_vis/result

python3 baseline/update_baselines_with_features.py \
  --baseline_dir baseline/mind2web/gpt_5_2/results \
  --source mind2web/gpt_5_2 \
  --uq_dir uq_analysis_change_qwen
```

### Mind2Web source conventions

For many `mind2web` models:

- verbalized dir is used for `M1-M4`
- multi-sample dir is used for `M5-M6`

Examples:

- `claude`
  - `M1`: `mind2web/results_claude_verbalized`
  - `M5-M6`: `mind2web/claude`
  - no HTC, because no usable token logprobs
- `gpt_5_2`
  - `M1-M4`: `mind2web/results_gpt_5_2_verbalized`
  - `M5-M6`: `mind2web/gpt_5_2`
  - HTC available
- `gptnano`
  - `M1-M4`: `mind2web/results_gptnano_verbalized`
  - `M5-M6`: `mind2web/gptnano/results_gpt_5_nano_total` or the model-specific multi-sample source in use
  - HTC available if verbalized dir has logprobs
- `kimi`
  - `M1-M4`: `mind2web/results_kimi_verbalized`
  - `M5-M6`: `mind2web/kimi`
  - HTC available
- `qwen30b`
  - `M1-M4`: `mind2web/results_qwen30b_verbalized`
  - `M5-M6`: `mind2web/qwen30b`
  - HTC source may need override

For `qwen30b`, HTC may need:

```bash
python3 baseline/update_baselines_with_features.py \
  --baseline_dir baseline/mind2web/qwen30b/results \
  --source mind2web/qwen30b \
  --uq_dir uq_analysis_change \
  --htc_source mind2web/results_qwen30b_verbalized
```

## WebArena

Use:

- [`compute_webarena_step_baselines.py`](./compute_webarena_step_baselines.py)
- [`compute_mean_feature_baseline.py`](./compute_mean_feature_baseline.py)
- [`update_baselines_with_features.py`](./update_baselines_with_features.py)

Typical pattern:

```bash
python3 baseline/compute_webarena_step_baselines.py \
  --verbalized_dir webarena/results_gpt_5_2_verbalized \
  --entropy_dir webarena/gpt_5_2 \
  --actions_dir webarena/gpt_5_2/trajectory_1/actions \
  --output_dir baseline/webarena/gpt_5_2/results \
  --model gpt_5_2 \
  --benchmark webarena \
  --annotation_dir annotation/result

python3 baseline/update_baselines_with_features.py \
  --baseline_dir baseline/webarena/gpt_5_2/results \
  --source webarena/gpt_5_2 \
  --uq_dir uq_analysis_change_qwen
```

### WebArena source conventions

Current correct source split is:

- `M1-M4`: `webarena/results_<model>_verbalized`
- `M5-M6`: `webarena/<model>`
- `HTC`: normally loaded from `source`, so the source should have usable logprobs

In the current refreshed runs:

- `claude`
  - no logprobs
  - no HTC
- `gpt_5_2`
  - HTC available
- `gptnano`
  - HTC available
- `kimi`
  - HTC available
- `qwen30b`
  - HTC available

## Old / Experimental Scripts

These scripts are **not** the main production baseline path now:

- [`run_htc_all.py`](./run_htc_all.py)
  Old task-level HTC experiment
- [`run_htc_prek.py`](./run_htc_prek.py)
  Old prefix HTC experiment
- [`htc_feature_matrix_baseline.py`](./htc_feature_matrix_baseline.py)
  HTC-style logistic directly on feature matrix; useful for side experiments only
- [`htc_baseline.py`](./htc_baseline.py)
  Older HTC-related helper / experiment file

They are still useful for reference, but not for the current main baseline tables.

## Important Notes

### 1. `compute_metrics.py` is not part of the active path

It was removed from the active workflow because metrics are now written only by:

- [`update_baselines_with_features.py`](./update_baselines_with_features.py)

### 2. Some methods may be unavailable by data source

Examples:

- no logprobs → no `M2-M4`
- no multi-sample → no valid `M5-M6`
- no HTC logprob source → no `M11-M12`

### 3. A model can require split sources

Do not assume one folder contains everything.

Some models store:

- verbalized confidence / logprobs in one folder
- multi-sample entropy data in another folder

Always verify:

- where `raw_prediction` lives
- where `token_logprobs/top_logprobs` live
- where 10-sample `step_samples` live

## Recommended Reading Order

If you want to understand the codebase quickly:

1. [`README.md`](./README.md)
2. [`keystep_labels.py`](./keystep_labels.py)
3. [`compute_baselines.py`](./compute_baselines.py)
4. [`compute_webarena_step_baselines.py`](./compute_webarena_step_baselines.py)
5. [`compute_mean_feature_baseline.py`](./compute_mean_feature_baseline.py)
6. [`htc_step_baseline.py`](./htc_step_baseline.py)
7. [`update_baselines_with_features.py`](./update_baselines_with_features.py)

## Current Status

The refreshed baseline results now exist for:

- `mind2web`: `claude`, `gpt_5_2`, `gptnano`, `kimi`, `qwen30b`
- `webarena`: `claude`, `gpt_5_2`, `gptnano`, `kimi`, `qwen30b`

These result directories should be treated as the current baseline outputs.
