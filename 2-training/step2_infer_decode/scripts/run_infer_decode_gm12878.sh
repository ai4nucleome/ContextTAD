#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_infer_decode_gm12878.sh --checkpoint PATH [options]

Options:
  --checkpoint PATH    Trained ContextTAD checkpoint (required)
  --gpu-id ID          GPU used for inference (default: 0)
  --run-id ID          Output run ID (default: infer_gm12878_<timestamp>)
  --data-dir PATH      Processed GM12878 array directory
  --output-root PATH   Parent directory for inference runs
  --fdr-alpha FLOAT    Final-TAD target-decoy FDR level (default: 0.05)
  --peak-radius N      Peak exclusion radius in bins (default: 4)
  -h, --help           Show this help message
EOF
}

CKPT=""
GPU_ID=0
RUN_ID=""
DATA_DIR_OVERRIDE=""
OUTPUT_ROOT_OVERRIDE=""
FDR_ALPHA="${FDR_ALPHA:-0.05}"
PEAK_RADIUS="${PEAK_RADIUS:-4}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CKPT="${2:?missing value for --checkpoint}"; shift 2 ;;
    --gpu-id) GPU_ID="${2:?missing value for --gpu-id}"; shift 2 ;;
    --run-id) RUN_ID="${2:?missing value for --run-id}"; shift 2 ;;
    --data-dir) DATA_DIR_OVERRIDE="${2:?missing value for --data-dir}"; shift 2 ;;
    --output-root) OUTPUT_ROOT_OVERRIDE="${2:?missing value for --output-root}"; shift 2 ;;
    --fdr-alpha) FDR_ALPHA="${2:?missing value for --fdr-alpha}"; shift 2 ;;
    --peak-radius) PEAK_RADIUS="${2:?missing value for --peak-radius}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$CKPT" ]] || { echo "[error] --checkpoint is required" >&2; usage >&2; exit 2; }
RUN_ID="${RUN_ID:-infer_gm12878_$(date +%Y%m%d_%H%M%S)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CORE_DIR="$PROJECT_ROOT/2-training/core"
DATA_DIR="${DATA_DIR_OVERRIDE:-${TAD_DATA_DIR:-$PROJECT_ROOT/0-data/1_dp_train_infer_data}}"
OUTPUT_ROOT="${OUTPUT_ROOT_OVERRIDE:-$PROJECT_ROOT/2-training/step2_infer_decode/outputs}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
BEDS_DIR="$RUN_DIR/beds"
LOG_DIR="$RUN_DIR/logs"
COVERAGES=(4000M 2000M 1000M 500M 250M 125M 62_5M)
CHROMS=(chr15 chr16 chr17)

if [[ ! -f "$CKPT" ]]; then
  echo "[error] checkpoint not found: $CKPT" >&2
  exit 2
fi
if [[ ! -f "$DATA_DIR/window_list.json" ]]; then
  echo "[error] inference data not found: $DATA_DIR/window_list.json" >&2
  exit 2
fi
mkdir -p "$BEDS_DIR" "$LOG_DIR"

for cov in "${COVERAGES[@]}"; do
  CUDA_VISIBLE_DEVICES="$GPU_ID" python "$CORE_DIR/infer_decode_v32.py" \
    --checkpoint "$CKPT" \
    --data-root "$DATA_DIR" \
    --output-dir "$RUN_DIR" \
    --dataset "$cov" \
    --chroms "${CHROMS[@]}" \
    --fdr-alpha "$FDR_ALPHA" \
    --peak-radius "$PEAK_RADIUS" \
    > "$LOG_DIR/infer_decode_${cov}.log" 2>&1
done

echo "[done] $RUN_DIR"
