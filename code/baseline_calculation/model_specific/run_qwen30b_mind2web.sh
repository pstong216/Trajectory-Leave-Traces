#!/usr/bin/env bash
# Run baseline UQ scoring for qwen30b on mind2web.
# Execute from any directory; paths are relative to this script's location.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"   # results_total/ (paper_release copy: model_specific/ is 4 levels down)

python3 "$REPO_ROOT/baseline/compute_baselines.py" \
  --verbalized_dir "$REPO_ROOT/mind2web/results_qwen30b_verbalized" \
  --entropy_dir    "$REPO_ROOT/mind2web/qwen30b" \
  --output_dir     "$SCRIPT_DIR/results" \
  --ground_truth   "$REPO_ROOT/mind2web/qwen_result.json" \
  --model          qwen30b \
  --benchmark      mind2web
