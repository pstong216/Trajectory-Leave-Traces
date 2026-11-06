#!/bin/bash
# Run all 10 (5 models x 2 datasets) Logistic-L2 "Ours" training, self-contained:
# reads feature matrices and keystep labels from paper_release/data and
# paper_release/keystep_labels (NOT the original results_total/ tree), so this
# script works even if the original tree changes later.
#
# All --csv_path / --annot_path / --output_root are passed as ABSOLUTE paths
# deliberately: train_risk_model_temp_sca.py auto-derives its results_dir as
# Path(__file__).parents[2], which is only correct at its ORIGINAL location
# (codes/train_risk_model/); from this copy's location that would resolve to
# paper_release/ instead. Absolute paths bypass that entirely -- do not
# switch these to relative paths without re-checking resolve_annotation_path()
# and the results_dir fallback in main().

set -e

# Reproducibility (2026-07-30): pin BLAS/OpenMP thread count to 1. Without this,
# multi-threaded BLAS (Apple Accelerate on this machine) can produce tiny
# floating-point differences run-to-run (addition isn't associative across
# threads), which occasionally nudges an iterative solver (lbfgs) to a
# different result even with every random_state fixed -- this is what caused
# one anomalous AUROC during testing that would not reproduce afterward. Must
# be set as env vars before python starts; setting them inside the .py file is
# too late since BLAS reads thread count at process/library load time.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PAPER_RELEASE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_ROOT="$PAPER_RELEASE_ROOT/data"
LABELS_ROOT="$PAPER_RELEASE_ROOT/keystep_labels"
LOG_DIR="$SCRIPT_DIR/run_all_final_logs"
OUTPUT_ROOT="$SCRIPT_DIR/output"
mkdir -p "$LOG_DIR"

run_one() {
    local bench_model="$1"    # e.g. webarena/claude
    local bench="${bench_model%%/*}"
    local model="${bench_model##*/}"
    local csv_path="$DATA_ROOT/$bench_model/embedded/uq_analysis_change_qwen/thr_0p3/feature_matrix.csv"
    local annot_path="$LABELS_ROOT/$bench/${model}_ours.json"
    local log="$LOG_DIR/${bench_model//\//_}.log"
    echo "[$(date '+%H:%M:%S')] START  $bench_model"
    PYTHONHASHSEED=42 python "$SCRIPT_DIR/train_risk_model_temp_sca.py" \
        --source "$bench_model" \
        --csv_path "$csv_path" \
        --annot_path "$annot_path" \
        --output_root "$OUTPUT_ROOT" \
        > "$log" 2>&1
    echo "[$(date '+%H:%M:%S')] DONE   $bench_model  (log: $log)"
}

for model in claude gpt_5_2 gptnano kimi qwen30b; do
    run_one "webarena/$model"
done
for model in claude gpt_5_2 gptnano kimi qwen30b; do
    run_one "mind2web/$model"
done

echo ""
echo "All 10 runs complete. Results in: $OUTPUT_ROOT"
