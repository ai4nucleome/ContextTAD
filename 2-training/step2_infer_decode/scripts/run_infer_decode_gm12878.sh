#!/usr/bin/env bash
set -euo pipefail

# Generic inference + decode (no metric evaluation).
# Usage:
#   bash run_infer_decode_gm12878.sh <ckpt> [gpu_id] [run_id] [method] [prompt_id]

CKPT="${1:?Usage: $0 <ckpt> [gpu_id] [run_id] [method] [prompt_id]}"
GPU_ID="${2:-0}"
RUN_ID="${3:-infer_gm12878_$(date +%Y%m%d_%H%M%S)}"
METHOD="${4:-auto}"
PROMPT_ID="${5:-default}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLISH_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRAIN_ROOT="$PUBLISH_ROOT/2-training"
PRED_PY="$TRAIN_ROOT/core/predict_structure.py"
POST_PY="$TRAIN_ROOT/core/post_process.py"

DATA_DIR="${TAD_DATA_DIR:-$PUBLISH_ROOT/0-data/1_dp_train_infer_data}"
SAM3_DIR="${SAM3_PATH:-$TRAIN_ROOT/sam3}"
CONDA_SH="/home/weicai/anaconda3/etc/profile.d/conda.sh"

RUN_DIR="$PUBLISH_ROOT/2-training/step2_infer_decode/outputs/$RUN_ID"
RAW_DIR="$RUN_DIR/raw_preds"
BEDS_DIR="$RUN_DIR/beds"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$RAW_DIR" "$BEDS_DIR" "$LOG_DIR"

COVERAGES=(4000M 2000M 1000M 500M 250M 125M 62_5M)
CHROMS=(chr15 chr16 chr17)

source "$CONDA_SH"
conda activate 3dgenome

for cov in "${COVERAGES[@]}"; do
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  TAD_DATA_DIR="$DATA_DIR" \
  SAM3_PATH="$SAM3_DIR" \
  python "$PRED_PY" \
    --ckpt "$CKPT" \
    --output_dir "$RAW_DIR" \
    --coverages "$cov" \
    --chroms "${CHROMS[@]}" \
    --method "$METHOD" \
    --prompt_id "$PROMPT_ID" \
    --output_prefix ContextTAD_structure \
    > "$LOG_DIR/predict_${cov}.log" 2>&1

done

for cov in 4000M 2000M 1000M 500M; do
  python "$POST_PY" \
    --raw_bed "$RAW_DIR/ContextTAD_structure_${cov}.bed" \
    --output_bed "$BEDS_DIR/GM12878_${cov}_contexttad.bed" \
    --keep_ratio 0.66 \
    --dedup_gap_bins 4 \
    --max_children 12 \
    --max_l1plus_size 2000000 \
    --max_children_coverage_ratio 2.5 \
    > "$LOG_DIR/postprocess_${cov}.log" 2>&1

done

for cov in 250M 125M 62_5M; do
  python "$POST_PY" \
    --raw_bed "$RAW_DIR/ContextTAD_structure_${cov}.bed" \
    --output_bed "$BEDS_DIR/GM12878_${cov}_contexttad.bed" \
    --keep_ratio 0.60 \
    --dedup_gap_bins 4 \
    --max_children 12 \
    --max_l1plus_size 2000000 \
    --max_children_coverage_ratio 2.5 \
    > "$LOG_DIR/postprocess_${cov}.log" 2>&1

done

echo "[done] $RUN_DIR"
