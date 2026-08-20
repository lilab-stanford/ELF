#!/bin/bash
# Extract ELF slide embeddings. One job per tile-level foundation model.
#
# Required:
#   FEATURE_ROOT  parent of <fm>/<dataset>/h5_files
#   DATASET       e.g. ebrains
#
# Optional:
#   CKPT          default: ../checkpoints/elf_slide_encoder.pth
#   OUTPUT_ROOT   default: ../evaluation/<dataset>
#   GPUS          space-separated GPU ids, default: 0 1 2 3 4
#
# Example:
#   FEATURE_ROOT=/path/to/features DATASET=ebrains bash extract_multiple_model.sh

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

if [ "${CONDA_DEFAULT_ENV:-}" != "py310" ]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate py310
fi
PYTHON="${PYTHON:-python}"

CKPT="${CKPT:-$ROOT/checkpoints/elf_slide_encoder.pth}"
FEATURE_ROOT="${FEATURE_ROOT:?set FEATURE_ROOT to the parent of <fm>/<dataset>/h5_files}"
DATASET="${DATASET:?set DATASET (e.g. ebrains)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/evaluation/${DATASET}}"
FMS=(${FMS:-uni gigapath h0 virchow2 conch_v1_5})
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4}"

mkdir -p "$OUTPUT_ROOT"
pids=()
for i in "${!FMS[@]}"; do
  fm="${FMS[$i]}"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  echo "launch $fm on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$HERE/extract_multiple_model_slide_embedding.py" \
    --checkpoint "$CKPT" \
    --input-dir "$FEATURE_ROOT/$fm/$DATASET/h5_files" \
    --output-path "$OUTPUT_ROOT/${fm}_elf" \
    --feature-models "$fm" \
    --gpu 0 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done
exit "$fail"
