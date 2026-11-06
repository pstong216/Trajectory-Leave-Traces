#!/usr/bin/env bash
# Run baseline UQ scoring for kimi on webarena.
# Execute from any directory; paths are relative to this script's location.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"   # results_total/ (paper_release copy: model_specific/ is 4 levels down)

python3 "$SCRIPT_DIR/compute_kimi_baselines.py" \
  --verbalized_dir "$REPO_ROOT/webarena/results_kimi_verbalized/trajectory_1" \
  --entropy_dir    "$REPO_ROOT/webarena/kimi" \
  --output_dir     "$SCRIPT_DIR/results" \
  --model          kimi \
  --benchmark      webarena
