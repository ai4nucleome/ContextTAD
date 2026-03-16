#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$MAIN_DIR/../.." && pwd)"
OUT_ROOT="$MAIN_DIR/outputs/runs"
RUN_ID="${3:-main_results_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$OUT_ROOT/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
CSV_DIR="$RUN_DIR/csv"
BAK_DIR="$RUN_DIR/backup_beds"
mkdir -p "$LOG_DIR" "$CSV_DIR" "$BAK_DIR"

GM_BEDS_DIR="${1:?Usage: $0 <gm12878_beds_dir> <othercell_beds_dir> [run_id]}"
OTH_BEDS_DIR="${2:?Usage: $0 <gm12878_beds_dir> <othercell_beds_dir> [run_id]}"

EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-$PROJECT_ROOT/0-data/2_eval_tads_data}"
TOOLS_ROOT="${TOOLS_ROOT:-$EVAL_DATA_ROOT/tools_output}"
EXPS_ROOT="${EXPS_ROOT:-$PROJECT_ROOT/3-evaluation/common}"

if [[ ! -d "$TOOLS_ROOT" ]]; then
  echo "[error] tools_output directory not found: $TOOLS_ROOT" >&2
  echo "        set EVAL_DATA_ROOT or TOOLS_ROOT before running." >&2
  exit 2
fi
if [[ ! -d "$EXPS_ROOT" ]]; then
  echo "[error] exp backend directory not found: $EXPS_ROOT" >&2
  exit 2
fi

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
  if [[ ! -f "$src" ]]; then
    echo "[error] missing source bed: $src" >&2
    exit 2
  fi
  backup_file_if_exists "$dst"
  backup_file_if_exists "${dst}.L0"
  backup_file_if_exists "${dst}.L1+"
  backup_file_if_exists "${dst}.meanif"
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
  rm -f "${dst}.L0" "${dst}.L1+" "${dst}.meanif"
}

echo "[1/3] Replace ContextTAD beds"
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
    # fallback: flat layout used by eval runs
    src="$GM_BEDS_DIR/GM12878_${cov_dir}_contexttad.bed"
  fi
  dst="$TOOLS_ROOT/$cov_dir/ContextTAD/4DNFIXP4QG5B_Rao2014_GM12878_${cov_file}_5K_contexttad.bed"
  copy_replace_bed "$src" "$dst"
done

k562_src="$OTH_BEDS_DIR/K562/K562_contexttad.bed"
imr90_src="$OTH_BEDS_DIR/IMR90/IMR90_contexttad.bed"
if [[ ! -f "$k562_src" ]]; then
  k562_src="$OTH_BEDS_DIR/K562_contexttad.bed"
fi
if [[ ! -f "$imr90_src" ]]; then
  imr90_src="$OTH_BEDS_DIR/IMR90_contexttad.bed"
fi
copy_replace_bed "$k562_src" "$TOOLS_ROOT/other_celltypes/K562/ContextTAD/K562_contexttad.bed"
copy_replace_bed "$imr90_src" "$TOOLS_ROOT/other_celltypes/IMR90/ContextTAD/IMR90_contexttad.bed"

echo "[2/3] Run exp1/2/3/4/6/7/8"
EVAL_DATA_ROOT="$EVAL_DATA_ROOT" EVAL_TOOLS_ROOT="$TOOLS_ROOT" \
  bash "$EXPS_ROOT/exp1_tadnum_ctcf_chiapet/run_exp1.sh" > "$LOG_DIR/exp1.log" 2>&1
EVAL_DATA_ROOT="$EVAL_DATA_ROOT" EVAL_TOOLS_ROOT="$TOOLS_ROOT" \
  CLEAN_OUTPUT=1 OUTPUT_DIR="$RUN_DIR/exp2_struct_protein" FIG_DIR="$RUN_DIR/exp2_csv" \
  bash "$EXPS_ROOT/exp2_struct_protein/run_exp2.sh" > "$LOG_DIR/exp2.log" 2>&1
EVAL_DATA_ROOT="$EVAL_DATA_ROOT" EVAL_TOOLS_ROOT="$TOOLS_ROOT" \
  bash "$EXPS_ROOT/exp3_tadb_left_ctcf_chipseq/run_exp3.sh" > "$LOG_DIR/exp3.log" 2>&1
EVAL_DATA_ROOT="$EVAL_DATA_ROOT" EVAL_TOOLS_ROOT="$TOOLS_ROOT" \
  bash "$EXPS_ROOT/exp4_tadb_right_ctcf_chipseq/run_exp4.sh" > "$LOG_DIR/exp4.log" 2>&1
EVAL_DATA_ROOT="$EVAL_DATA_ROOT" EVAL_TOOLS_ROOT="$TOOLS_ROOT" \
  bash "$EXPS_ROOT/exp6_tadnum_ctcf_chiapet_downsample/run_exp6.sh" > "$LOG_DIR/exp6.log" 2>&1
EVAL_DATA_ROOT="$EVAL_DATA_ROOT" EVAL_TOOLS_ROOT="$TOOLS_ROOT" \
  bash "$EXPS_ROOT/exp7_othercell_tadnum_ctcf_chiapet/run_exp7.sh" > "$LOG_DIR/exp7.log" 2>&1
EVAL_DATA_ROOT="$EVAL_DATA_ROOT" EVAL_TOOLS_ROOT="$TOOLS_ROOT" \
  bash "$EXPS_ROOT/exp8_othercell_both_ctcf_chipseq/run_exp8.sh" > "$LOG_DIR/exp8.log" 2>&1
if [[ "${RUN_EXP10:-0}" == "1" ]]; then
  EVAL_DATA_ROOT="$EVAL_DATA_ROOT" bash "$EXPS_ROOT/exp5_coolpup/run_exp5.sh" > "$LOG_DIR/exp5.log" 2>&1
fi

echo "[3/3] Collect CSV outputs"
cp "$EXPS_ROOT/exp1_tadnum_ctcf_chiapet/exp1_csv/exp1_results.csv" "$CSV_DIR/"
cp "$RUN_DIR/exp2_csv/exp2_results.csv" "$CSV_DIR/" 2>/dev/null || true
cp "$EXPS_ROOT/exp3_tadb_left_ctcf_chipseq/exp3_csv/exp3_results.csv" "$CSV_DIR/"
cp "$EXPS_ROOT/exp4_tadb_right_ctcf_chipseq/exp4_csv/exp4_results.csv" "$CSV_DIR/"
cp "$EXPS_ROOT/exp6_tadnum_ctcf_chiapet_downsample/exp6_csv/exp6_results_all.csv" "$CSV_DIR/"
cp "$EXPS_ROOT/exp7_othercell_tadnum_ctcf_chiapet/exp7_csv/exp7_results.csv" "$CSV_DIR/"
cp "$EXPS_ROOT/exp8_othercell_both_ctcf_chipseq/exp8_csv/exp8_boundary_counts.csv" "$CSV_DIR/"

echo "[done] run_dir=$RUN_DIR"
