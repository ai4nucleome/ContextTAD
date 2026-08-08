#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_infer_decode_othercell.sh --checkpoint PATH [options]

Options:
  --checkpoint PATH    Trained ContextTAD checkpoint (required)
  --gpu-id ID          GPU used for inference (default: 0)
  --run-id ID          Output run ID (default: infer_othercell_<timestamp>)
  --data-root PATH     Directory containing K562/ and IMR90/ arrays
  --output-root PATH   Parent directory for inference runs
  --fdr-alpha FLOAT    Target-decoy FDR level (default: 0.005)
  --peak-radius N      Peak exclusion radius in bins (default: 5)
  -h, --help           Show this help message
EOF
}

CKPT=""
GPU_ID=0
RUN_ID=""
DATA_ROOT_OVERRIDE=""
OUTPUT_ROOT_OVERRIDE=""
FDR_ALPHA="${FDR_ALPHA:-0.005}"
PEAK_RADIUS="${PEAK_RADIUS:-5}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CKPT="${2:?missing value for --checkpoint}"; shift 2 ;;
    --gpu-id) GPU_ID="${2:?missing value for --gpu-id}"; shift 2 ;;
    --run-id) RUN_ID="${2:?missing value for --run-id}"; shift 2 ;;
    --data-root) DATA_ROOT_OVERRIDE="${2:?missing value for --data-root}"; shift 2 ;;
    --output-root) OUTPUT_ROOT_OVERRIDE="${2:?missing value for --output-root}"; shift 2 ;;
    --fdr-alpha) FDR_ALPHA="${2:?missing value for --fdr-alpha}"; shift 2 ;;
    --peak-radius) PEAK_RADIUS="${2:?missing value for --peak-radius}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$CKPT" ]] || { echo "[error] --checkpoint is required" >&2; usage >&2; exit 2; }
RUN_ID="${RUN_ID:-infer_othercell_$(date +%Y%m%d_%H%M%S)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CORE_DIR="$PROJECT_ROOT/2-training/core"
DATA_ROOT="${DATA_ROOT_OVERRIDE:-${OTHERCELL_DATA_ROOT:-$PROJECT_ROOT/0-data/1_dp_train_infer_data/other_celltypes}}"
OUTPUT_ROOT="${OUTPUT_ROOT_OVERRIDE:-$PROJECT_ROOT/2-training/step2_infer_decode/outputs}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
RAW_DIR="$RUN_DIR/raw_preds"
BEDS_DIR="$RUN_DIR/beds"
LOG_DIR="$RUN_DIR/logs"
STATS_DIR="$RUN_DIR/postprocess_stats"
CHROMS=(chr15 chr16 chr17)

if [[ ! -f "$CKPT" ]]; then
  echo "[error] checkpoint not found: $CKPT" >&2
  exit 2
fi
mkdir -p "$RAW_DIR" "$BEDS_DIR" "$LOG_DIR" "$STATS_DIR"

for cell in K562 IMR90; do
  data_dir="$DATA_ROOT/$cell"
  if [[ ! -f "$data_dir/window_list.json" ]]; then
    echo "[error] inference data not found: $data_dir/window_list.json" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="$GPU_ID" python "$CORE_DIR/predict_structure.py" \
    --ckpt "$CKPT" \
    --data_dir "$data_dir" \
    --output_dir "$RAW_DIR" \
    --coverages "$cell" \
    --chroms "${CHROMS[@]}" \
    --output_prefix ContextTAD_structure \
    --peak_radius "$PEAK_RADIUS" \
    --fdr_alpha "$FDR_ALPHA" \
    --fdr_metric prominence \
    > "$LOG_DIR/predict_${cell}.log" 2>&1

  python "$CORE_DIR/post_process.py" \
    --raw_bed "$RAW_DIR/ContextTAD_structure_${cell}.bed" \
    --output_bed "$BEDS_DIR/${cell}_contexttad.bed" \
    --stats_json "$STATS_DIR/${cell}.json" \
    --min_len_bins 5 \
    --dedup_gap_bins 5 \
    --max_children 5 \
    --max_l1plus_size 2000000 \
    --max_children_coverage_ratio 3.0 \
    --snap_bin 2 \
    > "$LOG_DIR/postprocess_${cell}.log" 2>&1
done

echo "[done] $RUN_DIR"
