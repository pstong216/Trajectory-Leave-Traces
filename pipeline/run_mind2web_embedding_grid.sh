#!/bin/zsh
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <source-name> [--force]"
  echo "Example: $0 claude"
  exit 1
fi

SOURCE_NAME="$1"
shift || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/reembed_hlvm_variants.py"
SOURCE="mind2web/${SOURCE_NAME}"

models=(minilm qwen bge_m3)
thresholds=(0.4 0.3 0.2)

for model in "${models[@]}"; do
  for thr in "${thresholds[@]}"; do
    echo "=== ${SOURCE} | ${model} | threshold=${thr} ==="
    python3 "${PY_SCRIPT}" --source "${SOURCE}" --model "${model}" --threshold "${thr}" "$@"
  done
done
