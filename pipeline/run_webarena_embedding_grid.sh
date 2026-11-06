#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/reembed_hlvm_variants.py"
ROOT="${ROLLOUT_ROOT:?set ROLLOUT_ROOT to the directory containing webarena/<model>/ rollouts}"
OUT_ROOT="${ROOT}/webarena/embeded_hlvm"

sources=(claude gpt_5_2 gptnano kimi qwen30b)
models=(minilm qwen bge_m3)
thresholds=(0.4 0.3 0.2)

for source_name in "${sources[@]}"; do
  for model in "${models[@]}"; do
    for thr in "${thresholds[@]}"; do
      # skip already-completed qwen/thr_0p3 (exists for all 5 models)
      if [[ "${model}" == "qwen" && "${thr}" == "0.3" ]]; then
        echo "--- Skipping webarena/${source_name} | ${model} | threshold=${thr} (already done) ---"
        continue
      fi
      echo "=== webarena/${source_name} | ${model} | threshold=${thr} ==="
      python3 "${PY_SCRIPT}" \
        --source "webarena/${source_name}" \
        --model "${model}" \
        --threshold "${thr}" \
        --out_root "${OUT_ROOT}" \
        "$@"
    done
  done
done

echo ""
echo "All webarena embedding runs finished."
echo "Results in: ${OUT_ROOT}"
