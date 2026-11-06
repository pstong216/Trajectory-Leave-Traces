#!/usr/bin/env bash
#
# Regenerate every table and figure in the paper from the data shipped in this
# repository. No raw trajectories and no model API calls are required.
#
#   bash reproduce.sh              # regenerate all figures/tables (~15 min)
#   bash reproduce.sh --retrain    # also retrain the Ours/L2 predictors first (~20 min extra)
#
# Outputs:
#   figures/                                  the paper's figures
#   code/<analysis>/results/                  per-analysis tables and CSVs
#
# Determinism: every random_state in the code is fixed, but multi-threaded BLAS
# can still perturb the last digits, so the thread caps below are part of the
# protocol, not an optimisation. See docs/REPRODUCIBILITY.md.
set -euo pipefail

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=42

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY=${PYTHON:-python3}
MODELS="claude gpt_5_2 gptnano kimi qwen30b"

step() { printf '\n\033[1m=== %s\033[0m\n' "$1"; }

if [[ "${1:-}" == "--retrain" ]]; then
  step "Retraining Ours/L2 predictors (10 backbone x benchmark settings)"
  for bench in webarena mind2web; do
    for model in $MODELS; do
      echo "  -> $bench/$model"
      $PY code/l2_training/train_risk_model_temp_sca.py \
        --source "$bench/$model" \
        --csv_path "data/$bench/$model/embedded/uq_analysis_change_qwen/thr_0p3/feature_matrix.csv" \
        --annot_path "keystep_labels/$bench/${model}_ours.json" \
        --output_root reproduction_output/ours >/dev/null
    done
  done
fi

step "Table 1 -- main results (Ours rows re-derived from the training runs)"
$PY code/l2_training/update_main_table_ours_rows.py --dry-run | tail -5

step "Table 2 + Figure 4 -- detection / false-cut trade-off"
$PY code/generalization/detection_false_cut_analysis/run_detection_false_cut_regions.py | tail -3

step "Table 3 -- LLM-judge run-to-run consistency"
( cd code/llm_judge_consistency && $PY prefix_label_consistency.py \
    --passes results/gpt52_pass1,results/gpt52_pass2,results/gpt52_pass3 \
    --manifest results/sample_150.json \
    --out results/gpt52_prefix_consistency_report.json | tail -3 )

step "Table 4 -- failure-cause distribution"
$PY code/failure_cause_distribution/run_failure_cause_distribution.py | tail -3

step "Table 5 + Figure 10 -- prefix-time discrimination"
$PY code/generalization/prefix_time_analysis/run_prefix_time_regions.py | tail -3

step "Tables 7-8 -- Micro sampling budget and cost"
$PY code/n_sampling_cost/run_n_sampling_ablation.py | tail -3

step "Figure 3 -- false cuts under key-step vs final-outcome supervision"
$PY code/keystep_ablation/run_feature_set_false_alarm.py | tail -3
$PY code/keystep_ablation/regenerate_paper_success_false_alarm_fig.py

step "Figure 5 -- cross-category generalization (size-matched)"
$PY code/generalization/sizematched_control/run_sizematched_control.py | tail -3

step "Figure 6 -- Macro/Micro case study"
$PY case_study/plot_f10_case_micro_hardcoded.py

step "Figure 7 -- clustering robustness"
$PY code/clustering_robustness/plot_f5_clustering_metrics.py

step "Figure 8 -- observable vs internal signals case study"
$PY case_study/plot_f10_case_study_hardcoded.py

step "Figure 9 -- feature-group ablation"
$PY code/feature_importance/run_feature_importance_ablation.py | tail -3

step "Collecting figures"
$PY collect_figures.py

printf '\n\033[1mDone.\033[0m Figures in figures/ (named by paper figure number);\n'
printf 'tables and intermediate CSVs in each analysis'"'"'s own results/ directory.\n'
printf 'Figures 1-2 are hand-drawn schematics with no generating script, and Table 6\n'
printf 'is a re-layout of main_results_table.csv.\n'
