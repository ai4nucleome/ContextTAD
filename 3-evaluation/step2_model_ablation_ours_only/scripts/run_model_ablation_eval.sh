#!/usr/bin/env bash
set -euo pipefail

# Evaluate one model's GM12878 BEDs (ours only) via exp1/3/4/6.
# Usage:
#   bash run_model_ablation_eval.sh <gm12878_beds_dir> [run_id]

GM_BEDS_DIR="${1:?Usage: $0 <gm12878_beds_dir> [run_id]}"
RUN_ID="${2:-model_ablation_eval_$(date +%Y%m%d_%H%M%S)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$STEP_DIR/../.." && pwd)"
EXPS_ROOT="${EXPS_ROOT:-$PROJECT_ROOT/3-evaluation/common}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-$PROJECT_ROOT/0-data/2_eval_tads_data}"
TOOLS_ROOT="${TOOLS_ROOT:-$EVAL_DATA_ROOT/tools_output}"

RUN_DIR="$STEP_DIR/outputs/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
CSV_DIR="$RUN_DIR/csv"
BAK_DIR="$RUN_DIR/backup_beds"
mkdir -p "$LOG_DIR" "$CSV_DIR" "$BAK_DIR"

backup_file_if_exists() {
  local src="$1"
  if [[ -f "$src" ]]; then
    local rel="${src#$TOOLS_ROOT/}"
    mkdir -p "$BAK_DIR/$(dirname "$rel")"
    cp "$src" "$BAK_DIR/$rel"
  fi
}

restore_backups() {
  if [[ ! -d "$BAK_DIR" ]]; then
    return
  fi
  while IFS= read -r f; do
    local rel="${f#$BAK_DIR/}"
    local dst="$TOOLS_ROOT/$rel"
    mkdir -p "$(dirname "$dst")"
    cp "$f" "$dst"
  done < <(find "$BAK_DIR" -type f | sort)
}
trap restore_backups EXIT

copy_replace_bed() {
  local src="$1"
  local dst="$2"
  [[ -f "$src" ]] || { echo "[error] missing source bed: $src"; exit 2; }
  backup_file_if_exists "$dst"
  backup_file_if_exists "${dst}.L0"
  backup_file_if_exists "${dst}.L1+"
  backup_file_if_exists "${dst}.meanif"
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
  rm -f "${dst}.L0" "${dst}.L1+" "${dst}.meanif"
}

for pair in \
  "4000M:4000M" \
  "2000M:2000M" \
  "1000M:1000M" \
  "500M:500M" \
  "250M:250M" \
  "125M:125M" \
  "62_5M:62.5M"
do
  cov_dir="${pair%%:*}"
  cov_file="${pair##*:}"
  src="$GM_BEDS_DIR/$cov_dir/4DNFIXP4QG5B_Rao2014_GM12878_${cov_file}_5K_contexttad.bed"
  if [[ ! -f "$src" ]]; then
    src="$GM_BEDS_DIR/GM12878_${cov_dir}_contexttad.bed"
  fi
  dst="$TOOLS_ROOT/$cov_dir/ContextTAD/4DNFIXP4QG5B_Rao2014_GM12878_${cov_file}_5K_contexttad.bed"
  copy_replace_bed "$src" "$dst"
done

EVAL_DATA_ROOT="$EVAL_DATA_ROOT" EVAL_TOOLS_ROOT="$TOOLS_ROOT" \
  bash "$EXPS_ROOT/exp1_tadnum_ctcf_chiapet/run_exp1.sh" > "$LOG_DIR/exp1.log" 2>&1
EVAL_DATA_ROOT="$EVAL_DATA_ROOT" EVAL_TOOLS_ROOT="$TOOLS_ROOT" \
  bash "$EXPS_ROOT/exp3_tadb_left_ctcf_chipseq/run_exp3.sh" > "$LOG_DIR/exp3.log" 2>&1
EVAL_DATA_ROOT="$EVAL_DATA_ROOT" EVAL_TOOLS_ROOT="$TOOLS_ROOT" \
  bash "$EXPS_ROOT/exp4_tadb_right_ctcf_chipseq/run_exp4.sh" > "$LOG_DIR/exp4.log" 2>&1
EVAL_DATA_ROOT="$EVAL_DATA_ROOT" EVAL_TOOLS_ROOT="$TOOLS_ROOT" \
  bash "$EXPS_ROOT/exp6_tadnum_ctcf_chiapet_downsample/run_exp6.sh" > "$LOG_DIR/exp6.log" 2>&1

cp "$EXPS_ROOT/exp1_tadnum_ctcf_chiapet/exp1_csv/exp1_results.csv" "$CSV_DIR/"
cp "$EXPS_ROOT/exp3_tadb_left_ctcf_chipseq/exp3_csv/exp3_results.csv" "$CSV_DIR/"
cp "$EXPS_ROOT/exp4_tadb_right_ctcf_chipseq/exp4_csv/exp4_results.csv" "$CSV_DIR/"
cp "$EXPS_ROOT/exp6_tadnum_ctcf_chiapet_downsample/exp6_csv/exp6_results_all.csv" "$CSV_DIR/"

echo "[done] run_dir=$RUN_DIR"
